import logging
import threading
import time
import uuid
from queue import Queue
from typing import Any

from kubernetes import client, config, watch
from kubernetes.client.models import CoreV1Event
from pymilvus import Collection, utility

from retriever import MilvusHybridRetriever
from utils import get_sparse_embed_model, get_dense_embed_model
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
        # 事件风暴去重
        self.dedup_cache = {}  # Key: (namespace, name, reason), Value: last_seen_timestamp
        self.dedup_ttl = 10
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
            self.collection.create_partition(self.partition_name)
            self.logger.info("Created partition: dynamic_events")

        # 加载集合 (注意：写入时不需要 Load，但我们需要做反向查询，所以必须 Load)
        # 为了节省内存，可以只 Load 静态分区用于查询，但 Milvus API 通常 Load 整个 Collection
        self.collection.load()

    def _init_models(self, dense_path, sparse_path):
        """加载嵌入模型(Qwen和BGE-M3)"""
        self.logger.info("Loading Embedding Models...")
        self.dense_ef = get_dense_embed_model(dense_path)
        self.sparse_ef = get_sparse_embed_model(sparse_path)
        self.logger.info("Models loaded.")

    def start(self):
        """启动监听线程和处理线程"""
        # 1. 生产者线程: K8s Event Watcher
        watcher_thread = threading.Thread(target=self._k8s_event_watcher, daemon=True)
        watcher_thread.start()

        # 2. 消费者线程: Event Processor (Mapping & Storage)
        processor_thread = threading.Thread(target=self._event_processor, daemon=True)
        processor_thread.start()

        self.logger.info("Runtime Bridge Started. Press Ctrl+C to exit.")
        try:
            while True: time.sleep(1)
        except KeyboardInterrupt:
            self.logger.info("Stopping...")

    def _is_duplicate(self, event):
        key = (event["namespace"], event["name"], event["reason"])
        now = time.time()

        if key in self.dedup_cache:
            last_seen = self.dedup_cache[key]
            if now - last_seen < self.dedup_ttl:
                return True  # 是重复事件，跳过

        # 更新缓存
        self.dedup_cache[key] = now
        # 简单清理逻辑：防止内存无限增长（生产环境可用 LRUCache）
        if len(self.dedup_cache) > 1000:
            self.dedup_cache.clear()
        return False

    def _k8s_event_watcher(self):
        """监听 K8s Warning 事件 (Sensory Layer)"""
        w = watch.Watch()
        while True:
            try:
                self.logger.info("Watching K8s Events...")
                for event in w.stream(self.v1.list_event_for_all_namespaces):
                    obj: CoreV1Event = event['object']
                    # 过滤逻辑：只关注 Warning 类型
                    if obj.type == "Warning":
                        # 构造中间态数据
                        event_payload = {
                            "kind": "Event",
                            "reason": obj.reason,   # e.g., OOMKilled
                            "message": obj.message,     # 详细日志
                            "namespace": obj.metadata.namespace,
                            "name": obj.involved_object.name,
                            "obj_kind": obj.involved_object.kind,
                            "timestamp": str(obj.last_timestamp)
                        }
                        if self._is_duplicate(event_payload):
                            self.logger.debug(f"Skipping duplicate event: {obj.reason}")
                            continue
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

        # 1. 文本化 (Retrieval-oriented Textification)
        # A. 抽象文本 (用于动静对齐 & Summary)
        # 目的：去掉实体干扰，纯粹描述故障现象，以便匹配静态手册
        abstract_text = f"故障: {event['reason']}. 日志: {event['message']}"

        # B. 具体文本 (用于存储 & Agent检索)
        # 目的：包含实体名称(payment-service)、Namespace等，以便用户能搜到
        specific_text = (
            f"【运行警报】\n"
            f"Resource: {event['namespace']}/{event['name']}\n"  # <--- 关键：包含实体名
            f"Reason: {event['reason']}\n"
            f"Message: {event['message']}\n"
            f"Time: {event['timestamp']}"
        )

        # 2. 实体映射 (Entity Mapping / Reverse Instantiation)
        align_sparse_result = self.sparse_ef.encode_queries([abstract_text])["sparse"]
        align_sparse_vec = csr_to_milvus_format(align_sparse_result)[0]
        anchor = self._find_static_anchor(align_sparse_vec)

        # 3. 面向存储的向量化 (Storage-oriented Vectorization)
        # A. 稠密向量 (Dense Vector)
        storage_dense_vec = self.dense_ef.embed_documents([specific_text])[0]
        summary_dense_vec = self.dense_ef.embed_query(abstract_text)

        # B. 稀疏向量 (Sparse Vector)
        storage_sparse_result = self.sparse_ef.encode_documents([specific_text])["sparse"]
        storage_sparse_vec = csr_to_milvus_format(storage_sparse_result)[0]

        # 4. 构造数据并写入动态分区 (Dynamic Storage)
        self._insert_to_milvus(
            event=event,
            text=specific_text,             # 存具体内容
            summary=abstract_text,          # 存抽象内容
            dense_vec=storage_dense_vec,    # 存具体向量 (供 Agent 检索)
            sparse_vec=storage_sparse_vec,  # 存具体向量 (供 Agent 检索)
            summary_vec=summary_dense_vec,  # 存抽象向量 (备用)
            anchor=anchor
        )

    def _find_static_anchor(self, sparse_vec: list[float]):
        """在 static_knowledge 分区搜索最相关的排查文档"""
        search_params = MilvusHybridRetriever.sparse_search_params

        # 只在静态分区搜索
        results = self.collection.search(
            data=[sparse_vec],
            anns_field="sparse_vector",
            param=search_params,
            limit=1,
            partition_names=["static_knowledge"],   # <--- 关键：限制在静态分区
            output_fields=["title", "pk"]
        )

        if results and results[0]:
            hit = results[0][0]
            entity = hit['entity']
            self.logger.info(f"Mapped to Static Anchor: {entity.get('title')} (Score: {hit.score})")
            return {"pk": entity.get("pk"), "title": entity.get("title")}
        return None

    def _insert_to_milvus(
            self,
            event,
            text: str,
            summary: str,
            dense_vec: list[float],
            sparse_vec: dict[int, float],
            summary_vec: list[float],
            anchor: dict[str, Any]
    ):
        """
        将动态节点写入 Milvus 的 dynamic_events 分区

        Args:
            event (dict): 原始事件数据
            text (str): 具体文本 (包含实体名，用于 text 字段)
            summary (str): 抽象文本 (去实体化，用于 summary 字段)
            dense_vec (list): 基于 specific_text 的稠密向量 (用于 vector 字段，供 Agent 检索)
            sparse_vec (dict): 基于 specific_text 的稀疏向量 (用于 sparse_vector 字段，供 Agent 检索)
            summary_vec (list): 基于 abstract_text 的稠密向量 (用于 summary_vector 字段，保留语义特征)
            anchor (dict): 静态锚点信息 {pk, title}
        """
        # 构造 related_links JSON，指向静态锚点
        related_links = []
        if anchor:
            related_links.append({
                "type": "static_anchor",  # 边类型：指向静态知识
                "text": f"Troubleshooting Guide: {anchor['title']}",  # 边的描述
                "pk": anchor['pk'],  # 目标节点 ID
                "url": ""  # 动态节点通常没有 URL
            })

        # 构造符合 Schema 的数据条目 (必须与 build_knowledge_base.py 一致)
        # 不需要字段给默认值
        entry = {
            # 核心字段
            "pk": str(uuid.uuid4()),        # 动态生成 UUID
            "text": text,                   # 具体现场日志 (包含 payment-service 等实体)
            "summary": summary,             # 存储去实例化后的故障摘要
            # 向量
            "vector": dense_vec,            # [检索用] 包含实体语义
            "sparse_vector": sparse_vec,    # [检索用] 包含实体关键词权重
            "title_sparse": sparse_vec,     # [检索用] 标题稀疏向量复用正文的
            "summary_vector": summary_vec,  # [分析用] 纯粹的故障模式语义 (无实体干扰)
            # 元数据与动静关联
            "source": "k8s_informer",           # 数据源标记
            "title": f"Runtime Alert: {event['reason']} - {event['name']}",
            "node_type": "dynamic_event",       # 节点类型标记 (Agent根据此字段判断是否高亮)
            "related_links": related_links,     # JSON 格式的关联链路
            "parent_id": anchor['pk'] if anchor else "",    # 逻辑上的父节点指向静态手册
            # 填充字段 (默认值)
            "child_ids": [],                # 动态节点无子节点
            "level": 0,
            "token_count": len(text),
            "left_sibling": "",
            "right_sibling": "",
            "from_split": False,
            "merged": False,
            "nav_next_step": "",            # 动态节点无预定义的下一步导航
            "nav_see_also": "",
            "entry_urls": [],               # 无外部 URL
        }

        # 写入动态分区
        self.collection.insert([entry], partition_name=self.partition_name)     # <--- 关键：指定分区
        self.logger.info(f"Inserted into {self.partition_name} partition.")


if __name__ == "__main__":
    bridge = RuntimeBridge()
    bridge.start()