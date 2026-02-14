import logging
import threading
import time
import uuid
from queue import Queue

import torch
from kubernetes import client, config, watch
from langchain_huggingface import HuggingFaceEmbeddings
from pymilvus import Collection, utility
from pymilvus.model.hybrid import BGEM3EmbeddingFunction

# 引入项目中的工具函数 (假设 utils.py 在同一目录)
from utils.milvus_adapter import connect_milvus_by_env, csr_to_milvus_format


class RuntimeBridge:
    def __init__(
            self,
            collection_name="knowledge_base_v3",
            embedding_model_path="../models/Qwen/Qwen3-Embedding-0.6B",
            sparse_model_path="BAAI/bge-m3",
            dynamic_partition_name="dynamic_events"
    ):
        self.collection_name = collection_name
        self.partition_name = dynamic_partition_name
        self.event_queue = Queue()
        # 配置日志
        self.logger = logging.getLogger(type(self).__name__)
        self.logger.setLevel(logging.INFO)
        handler = logging.FileHandler("bridge.log", encoding='utf-8')
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

        # 1. 初始化 K8s 客户端
        self._init_k8s()

        # 2. 初始化 Milvus 连接与分区
        self._init_milvus()

        # 3. 初始化模型 (显存警告：Agent和Bridge若在同机，需注意显存分配)
        self._init_models(embedding_model_path, sparse_model_path)

    def _init_k8s(self):
        try:
            config.load_incluster_config()
            self.logger.info("Loaded in-cluster config.")
        except config.ConfigException:
            config.load_kube_config()
            self.logger.info("Loaded kube-config.")
        self.v1 = client.CoreV1Api()

    def _init_milvus(self):
        connect_milvus_by_env()
        if not utility.has_collection(self.collection_name):
            raise RuntimeError(
                f"Collection {self.collection_name} does not exist! Please run build_knowledge_base.py first.")
        self.collection = Collection(self.collection_name)

        # 检查并创建动态分区
        if not self.collection.has_partition(self.partition_name):
            self.collection.create_partition("dynamic_events")
            self.logger.info("Created partition: dynamic_events")

        # 加载集合 (注意：写入时不需要 Load，但我们需要做反向查询，所以必须 Load)
        # 为了节省内存，可以只 Load 静态分区用于查询，但 Milvus API 通常 Load 整个 Collection
        self.collection.load()

    def _init_models(self, dense_path, sparse_path):
        self.logger.info("Loading Embedding Models...")
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Dense Model (Qwen)
        self.dense_ef = HuggingFaceEmbeddings(
            model_name=dense_path,
            model_kwargs={"device": device, "trust_remote_code": True},
            encode_kwargs={"normalize_embeddings": True}
        )

        # Sparse Model (BGE-M3)
        self.sparse_ef = BGEM3EmbeddingFunction(
            model_name=sparse_path,
            use_fp16=True,
            device=device
        )
        self.logger.info("Models loaded.")

    def start(self):
        """启动监听线程和处理线程"""
        # 线程 1: K8s Event Watcher
        watcher_thread = threading.Thread(target=self._k8s_event_watcher, daemon=True)
        watcher_thread.start()

        # 线程 2: Event Processor (Mapping & Storage)
        processor_thread = threading.Thread(target=self._event_processor, daemon=True)
        processor_thread.start()

        self.logger.info("Runtime Bridge Started. Press Ctrl+C to exit.")
        try:
            while True: time.sleep(1)
        except KeyboardInterrupt:
            self.logger.info("Stopping...")

    def _k8s_event_watcher(self):
        """监听 K8s Warning 事件 (Sensory Layer)"""
        w = watch.Watch()
        while True:
            try:
                self.logger.info("Watching K8s Events...")
                for event in w.stream(self.v1.list_event_for_all_namespaces):
                    obj = event['object']
                    # 过滤逻辑：只关注 Warning 类型
                    if obj.type == "Warning":
                        # 构造中间态数据
                        event_payload = {
                            "kind": "Event",
                            "reason": obj.reason,  # e.g., OOMKilled
                            "message": obj.message,  # 详细日志
                            "namespace": obj.metadata.namespace,
                            "name": obj.involved_object.name,
                            "obj_kind": obj.involved_object.kind,
                            "timestamp": str(obj.last_timestamp)
                        }
                        self.event_queue.put(event_payload)
            except Exception as e:
                self.logger.error(f"Watcher Error: {e}")
                time.sleep(5)

    def _event_processor(self):
        """处理队列中的事件 (Mapping & Storage Layer)"""
        while True:
            event_data = self.event_queue.get()
            try:
                self._process_single_event(event_data)
            except Exception as e:
                self.logger.error(f"Processing Error: {e}")
            finally:
                self.event_queue.task_done()

    def _process_single_event(self, event):
        """核心逻辑：逆向实例化 + 存储"""
        self.logger.info(f"Processing Event: {event['reason']} on {event['name']}")

        # 1. 文本化 (Textification)
        # 构造类似 Query 的文本，用于去静态库里搜索
        query_text = f"Kubernetes fault: {event['reason']}. Error log: {event['message']}"
        full_display_text = f"【Runtime Alert】\nResource: {event['namespace']}/{event['name']}\nReason: {event['reason']}\nMessage: {event['message']}"

        # 2. 向量化 (Vectorization)
        # Dense Vector
        dense_vec = self.dense_ef.embed_query(query_text)
        # Sparse Vector (用于精确匹配错误码)
        sparse_result = self.sparse_ef.encode_documents([query_text])
        sparse_vec = csr_to_milvus_format(sparse_result["sparse"])[0]

        # 3. 实体映射 (Entity Mapping / Reverse Instantiation)
        # 论文 3.5.1: 利用稀疏向量相似度自动建立指向静态知识库的边
        anchor = self._find_static_anchor(sparse_vec)

        # 4. 构造数据并写入动态分区 (Dynamic Storage)
        self._insert_to_milvus(event, full_display_text, dense_vec, sparse_vec, anchor)

    def _find_static_anchor(self, sparse_vec):
        """在 static_knowledge 分区搜索最相关的排查文档"""
        search_params = {
            "metric_type": "IP",
            "params": {"drop_ratio_search": 0.2}
        }

        # 只在静态分区搜索
        results = self.collection.search(
            data=[sparse_vec],
            anns_field="sparse_vector",
            param=search_params,
            limit=1,
            partition_names=["static_knowledge"],  # <--- 关键：限制在静态分区
            output_fields=["title", "pk"]
        )

        if results and results[0]:
            hit = results[0][0]
            self.logger.info(f"Mapped to Static Anchor: {hit.entity.get('title')} (Score: {hit.score})")
            return {"pk": hit.entity.get("pk"), "title": hit.entity.get("title")}
        return None

    def _insert_to_milvus(self, event, text, dense_vec, sparse_vec, anchor):
        """将动态节点写入 dynamic_events 分区"""

        # 构造 related_links JSON，指向静态锚点
        related_links = []
        if anchor:
            related_links.append({
                "type": "static_anchor",  # 标记这是动静对齐的边
                "text": f"Troubleshooting Guide: {anchor['title']}",
                "pk": anchor['pk']
            })

        # 填充 Schema 字段 (必须与 build_knowledge_base.py 一致)
        # 不需要字段给默认值
        entry = {
            "pk": str(uuid.uuid4()),
            "text": text,
            "source": "k8s_informer",
            "title": f"Runtime Error: {event['reason']} - {event['name']}",
            # 向量
            "vector": dense_vec,
            "summary_vector": dense_vec,  # 复用
            "sparse_vector": sparse_vec,
            "title_sparse": sparse_vec,  # 复用
            # 动静关联
            "related_links": related_links,
            "parent_id": anchor['pk'] if anchor else "",
            # 动态标记
            "node_type": "dynamic_event",
            # 填充字段 (默认值)
            "child_ids": [],
            "level": 0,
            "token_count": len(text),
            "left_sibling": "",
            "right_sibling": "",
            "from_split": False,
            "merged": False,
            "nav_next_step": "",
            "nav_see_also": "",
            "entry_urls": [],
            "summary": ""
        }

        # 写入动态分区
        self.collection.insert([entry], partition_name="dynamic_events")  # <--- 关键：指定分区
        self.logger.info("Inserted into dynamic_events partition.")


if __name__ == "__main__":
    bridge = RuntimeBridge()
    bridge.start()