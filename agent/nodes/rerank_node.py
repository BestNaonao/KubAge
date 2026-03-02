from typing import List, Dict, Any

import torch
from torch.nn.functional import log_softmax
from langchain_core.documents import Document
from langchain_core.runnables import RunnableConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

from agent.schemas import OperationType, ProblemAnalysis
from utils import NodeType, SourceType

RERANK_SYSTEM_PROMPT = """You are performing binary relevance judgment for retrieval reranking.
Determine whether the given Kubernetes documentation fragment contains authoritative and actionable information that can directly help answer or resolve the technical question described in the Query.

The Document may include a "[Retrieval Context]" label indicating how it was retrieved:
- Anchor/Direct-Hit: Judge direct topical and semantic relevance to the query.
- Parent/Context: Judge whether it provides necessary definitions, scope, or prerequisites for the query topic.
- Link/Sibling: Judge whether it offers essential troubleshooting steps or related configuration details missed by the Direct Hit.

Answer "yes" only if:
- The document provides concrete concepts, command references, configuration details, API semantics, or troubleshooting guidance or necessary context.
- The document meaningfully reduces uncertainty in solving the Query.

Answer "no" if:
- The Document is irrelevant or too generic even considering its retrieval context.
- The document does not independently contribute practical value to resolving the Query.

You must answer with exactly one word: yes or no."""


class RerankNode:
    def __init__(self, model_path: str, top_n: int = 5, max_length: int = 8192):
        """
        初始化 Qwen3-Reranker (CausalLM 模式)
        """
        print(f"⏳ Loading Gen-Reranker model from {model_path}...")
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.top_n = top_n
        self.max_length = max_length

        # 1. 加载 Tokenizer (注意 padding_side='left')
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side='left')

        # 2. 加载模型 (AutoModelForCausalLM)
        # 如果显存允许，推荐开启 flash_attention_2
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                dtype=torch.float16 if self.device == "cuda" else torch.float32,
                attn_implementation="flash_attention_2" # 显存充足且支持时可解开注释
            ).to(self.device).eval()
        except Exception as e:
            print(f"⚠️ Failed to load with float16/flash_attn, falling back to default: {e}")
            self.model = AutoModelForCausalLM.from_pretrained(model_path).to(self.device).eval()

        # 3. 预计算 Prompt 组件
        self.token_false_id = self.tokenizer.convert_tokens_to_ids("no")
        self.token_true_id = self.tokenizer.convert_tokens_to_ids("yes")

        prefix = f"<|im_start|>system\n{RERANK_SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n"
        suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

        self.prefix_tokens = self.tokenizer.encode(prefix, add_special_tokens=False)
        self.suffix_tokens = self.tokenizer.encode(suffix, add_special_tokens=False)

        # 针对不同操作类型定制关注点
        self.base_instruct = "Given a technical query about Kubernetes, retrieve relevant documentation passages that provide answers or context."
        self.op_prompt_map = {
            OperationType.DIAGNOSIS: (
                "Given a troubleshooting scenario, retrieve documentation that explains error causes, "
                "debugging steps, log interpretation, or known issues related to the query. "
                "Prioritize actionable debugging guides over theoretical concepts. "
                "General debugging methodologies are acceptable when they clearly apply to the "
                "same failure category, even if specific resource or entity names are not mentioned."
            ),  # 诊断场景：关注错误原因、排查步骤、命令输出解释、日志分析
            OperationType.RESOURCE_DELETION: (
                "Given a request to delete or remove resources, retrieve documentation that describes "
                "the deletion command syntax, potential side effects, cascading deletion policies (e.g., ownerReferences), "
                "and how to safely execute the removal."
            ),  # 删除/危险操作：关注副作用、级联影响、安全操作命令、恢复方法
            OperationType.CONFIGURE: (
                "Given a configuration task, retrieve documentation that details the YAML resource definition, "
                "specific field semantics (under .spec), environment variables, or annotation options required "
                "to implement the requested configuration."
            ),  # 配置变更：关注 YAML 字段定义、spec 结构、配置项含义、取值范围
            OperationType.SCALING: (
                "Given a scaling or resource adjustment request, retrieve documentation concerning "
                "replica settings, HorizontalPodAutoscaler (HPA) configurations, 'kubectl scale' commands, "
                "or resource requests and limits strategies."
            ),  # 扩缩容：关注 HPA、replicas 字段、资源限制(Limit/Request)、扩展命令
            OperationType.KNOWLEDGE_QA: (
                "Given a conceptual question, retrieve documentation that provides clear definitions, "
                "architectural overviews, component comparisons, or design principles. "
                "Prioritize comprehensive explanations over specific command syntax."
            ),  # 知识问答：关注概念定义、架构原理、组件对比 (e.g. Deployment vs StatefulSet)
            OperationType.RESOURCE_INQUIRY: (
                "Given a request to query or view resource status, retrieve documentation about "
                "'kubectl get', 'kubectl describe', output formatting, or the meaning of specific "
                "status fields and conditions."
            ),  # 资源查询：关注 kubectl get/describe 用法、JSONPath、字段含义
            OperationType.RESOURCE_CREATION: (
                "Given a resource creation task, retrieve documentation providing 'kubectl create/apply' examples, "
                "boilerplate YAML templates, or prerequisites for deploying the specified resource type."
            ),  # 资源创建：关注 create/apply 命令、最小可用 YAML 示例
        }

        print("✅ Gen-Reranker model loaded.")

    def _format_input_pair(self, operation_type: OperationType, query: str, doc: Document) -> str:
        """
        构造模型输入：<Instruct> + <Query> + <Document (Title + Content)>
        """
        # 根据操作类型生成动态指令，提高重排针对性
        dynamic_instruction = self.op_prompt_map.get(operation_type, self.base_instruct)

        # 利用文档元数据中的 Title 增强上下文
        title = doc.metadata.get("title", "Untitled Section")
        content = doc.page_content

        # 显式注入 retrieval_source 信息，如果没有 metadata，给默认值
        source_type = doc.metadata.get("source_type", SourceType.UNKNOWN)
        source_desc = doc.metadata.get("source_desc", "Retrieved via vector search")

        # 构造上下文描述字符串。例如: "[Retrieval Context]: Type=parent. Info=Parent of: 'Debug Service'"
        context_str = f"[Retrieval Context]: Type={source_type.value}. Info={source_desc}"

        # 拼接增强后的 Document 内容。将 Context 放在 Content 之前，确保模型先看到文档的定位
        enriched_doc = f"[Title]: {title}\n{context_str}\n[Content]: {content}"

        return f"<Instruct>: {dynamic_instruction}\n<Query>: {query}\n<Document>: {enriched_doc}"

    def _compute_scores(self, pairs: List[str], token_budget: int = 8192) -> List[float]:
        """
        基于 Token Budget 的动态分批推理
        :param pairs: 输入的文本对列表
        :param token_budget: 显存允许的最大 token 总数 (Batch Size * Max Seq Len)
        :return: 按照输入顺序排列的分数列表
        """
        # 1. 预处理：Tokenize 并手动拼接 Prefix/Suffix
        # 为了效率，这里使用 batch_encode 这里的 truncation 只是为了防止单条超长
        inputs = self.tokenizer(
            pairs,
            padding=False,
            truncation='longest_first',
            return_attention_mask=False,
            max_length=self.max_length - len(self.prefix_tokens) - len(self.suffix_tokens)
        )
        raw_input_ids = inputs['input_ids']

        # 构造带索引的数据：(original_index, full_input_ids)
        indexed_data = []
        for idx, token_ids in enumerate(raw_input_ids):
            full_ids = self.prefix_tokens + token_ids + self.suffix_tokens
            indexed_data.append((idx, full_ids))

        # 2. 排序：按照 token 数量降序排列
        # 降序的好处：优先处理最长的，如果最长的单条都爆显存，能尽早发现；且通常长短文本分组更紧凑
        indexed_data.sort(key=lambda x: len(x[1]), reverse=True)

        # 3. 动态分批 (Greedy Batching based on Token Budget)
        batches = []
        current_batch = []
        current_max_len = 0

        for original_idx, ids in indexed_data:
            seq_len = len(ids)

            # 试探性计算：如果把当前样本加入 batch，新的 batch 占用多少显存？
            # 显存占用 ∝ (当前Batch数量 + 1) * max(当前最大长度, 新样本长度)
            # 因为是降序排列，新样本长度一定 <= current_max_len (除非是 Batch 的第一个)
            next_max_len = max(current_max_len, seq_len)
            next_batch_size = len(current_batch) + 1

            # 计算 Token 消耗 (矩形面积)
            estimated_token_count = next_batch_size * next_max_len

            if estimated_token_count <= token_budget:
                # 放入当前 Batch
                current_batch.append((original_idx, ids))
                current_max_len = next_max_len
            else:
                # 超出预算，封包当前 Batch，开启下一个新 Batch 初始化
                if current_batch:
                    batches.append(current_batch)

                current_batch = [(original_idx, ids)]
                current_max_len = seq_len

        # 处理最后一个 Batch
        if current_batch:
            batches.append(current_batch)

        # 4. 批量推理
        results = []  # 存储 (original_index, score)

        for batch in batches:
            # batch 结构: [(idx, ids), (idx, ids), ...]
            batch_indices = [item[0] for item in batch]
            batch_ids = [item[1] for item in batch]

            # Pad 当前 Batch (此时 batch 内长度差异很小，padding 很少)
            padded_inputs = self.tokenizer.pad(
                {'input_ids': batch_ids},
                padding=True,
                return_tensors="pt"
            )

            # Move to device
            for key in padded_inputs:
                padded_inputs[key] = padded_inputs[key].to(self.device)

            # Inference
            with torch.no_grad():
                outputs = self.model(**padded_inputs)
                batch_logits = outputs.logits[:, -1, :]

                true_vec = batch_logits[:, self.token_true_id]
                false_vec = batch_logits[:, self.token_false_id]

                combined = torch.stack([false_vec, true_vec], dim=1)
                probs = log_softmax(combined, dim=1)
                batch_scores = probs[:, 1].exp().tolist()

            # 收集结果
            for idx, score in zip(batch_indices, batch_scores):
                results.append((idx, score))

        # 5. 顺序还原 (Reorder)
        # 按照 original_index 从小到大排序，恢复原始顺序，只提取分数
        results.sort(key=lambda x: x[0])
        final_scores = [item[1] for item in results]

        return final_scores

    def __call__(self, state: dict, config: RunnableConfig) -> Dict[str, Any]:
        """
        执行重排序，包含对动态事件的保护逻辑
        """
        print("\n--- [Gen-Rerank Node] Running ---")

        retrieved_docs = state.get("retrieved_docs", [])
        if len(retrieved_docs) <= 0:
            return {"retrieved_docs": []}

        # 1. 分离动态事件和普通文档
        dynamic_docs = []
        static_docs = []
        for doc in retrieved_docs:
            if doc.metadata.get("node_type") == NodeType.EVENT:
                # 给动态事件赋予逻辑高分，方便后续统一排序或调试
                doc.metadata["rerank_score"] = 0.999
                dynamic_docs.append(doc)
            else:
                static_docs.append(doc)

        print(f"   ⚖️ Reranking: {len(dynamic_docs)} dynamic events + {len(static_docs)} static docs.")

        # 2. 对静态文档进行正常的 Rerank
        reranked_static = []
        if static_docs:
            analysis: ProblemAnalysis = state.get("analysis")
            # 获取技术摘要和操作类型
            query_text = analysis.technical_summary if analysis else state["messages"][-1].content
            op_type = analysis.target_operation if analysis else None
            print(f"🎯 Context: {op_type.value if op_type else "Raw Input"} | Query: {query_text[:100]}...")

            # 准备数据对
            input_texts = [self._format_input_pair(op_type, query_text, doc) for doc in static_docs]
            try:
                # 计算分数
                scores = self._compute_scores(input_texts)

                # 绑定分数并排序
                doc_score_pairs = list(zip(static_docs, scores))
                doc_score_pairs.sort(key=lambda x: x[1], reverse=True)

                print(f"📊 Reranking Results (Top {self.top_n}):")

                for doc, score in doc_score_pairs[:self.top_n]:
                    doc.metadata["rerank_score"] = float(score)
                    reranked_static.append(doc)
                    print(f"   [{doc.metadata.get('source_type', SourceType.UNKNOWN)}] Score: {score:.4f} "
                          f"| Title: {doc.metadata.get('title')}")
            except Exception as e:
                print(f"❌ Rerank Failed: {e}")
                # 如果重排失败，降级返回原始结果的前 N 个
                reranked_static = static_docs[:self.top_n]

        # 3. 结果合并：强制保留动态事件，并放在最前
        # 动态事件是“事实”，不需要 Rerank 过滤（因为检索时只取了 Top-2，已经很精简了）
        final_docs = dynamic_docs + reranked_static
        print(f"✅ Final Output: {len(dynamic_docs)} Events + {len(reranked_static)} Static Docs")
        return {"retrieved_docs": final_docs}