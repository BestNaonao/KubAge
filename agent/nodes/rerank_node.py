from typing import List, Dict, Any

import torch
import torch.nn.functional as F
from langchain_core.documents import Document
from langchain_core.runnables import RunnableConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

from agent.prompts import RERANK_SYSTEM_PROMPT
from agent.schemas import OperationType
from agent.state import AgentState


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
            # 诊断场景：关注错误原因、排查步骤、命令输出解释、日志分析
            OperationType.DIAGNOSIS: (
                "Given a troubleshooting scenario, retrieve documentation that explains error causes, "
                "debugging steps, log interpretation, or known issues related to the query. "
                "Prioritize actionable debugging guides over theoretical concepts."
            ),

            # 删除/危险操作：关注副作用、级联影响、安全操作命令、恢复方法
            OperationType.RESOURCE_DELETION: (
                "Given a request to delete or remove resources, retrieve documentation that describes "
                "the deletion command syntax, potential side effects, cascading deletion policies (e.g., ownerReferences), "
                "and how to safely execute the removal."
            ),

            # 配置变更：关注 YAML 字段定义、spec 结构、配置项含义、取值范围
            OperationType.CONFIGURE: (
                "Given a configuration task, retrieve documentation that details the YAML resource definition, "
                "specific field semantics (under .spec), environment variables, or annotation options required "
                "to implement the requested configuration."
            ),

            # 扩缩容：关注 HPA、replicas 字段、资源限制(Limit/Request)、扩展命令
            OperationType.SCALING: (
                "Given a scaling or resource adjustment request, retrieve documentation concerning "
                "replica settings, HorizontalPodAutoscaler (HPA) configurations, 'kubectl scale' commands, "
                "or resource requests and limits strategies."
            ),

            # 知识问答：关注概念定义、架构原理、组件对比 (e.g. Deployment vs StatefulSet)
            OperationType.KNOWLEDGE_QA: (
                "Given a conceptual question, retrieve documentation that provides clear definitions, "
                "architectural overviews, component comparisons, or design principles. "
                "Prioritize comprehensive explanations over specific command syntax."
            ),

            # 资源查询：关注 kubectl get/describe 用法、JSONPath、字段含义
            OperationType.RESOURCE_INQUIRY: (
                "Given a request to query or view resource status, retrieve documentation about "
                "'kubectl get', 'kubectl describe', output formatting, or the meaning of specific "
                "status fields and conditions."
            ),

            # 资源创建：关注 create/apply 命令、最小可用 YAML 示例
            OperationType.RESOURCE_CREATION: (
                "Given a resource creation task, retrieve documentation providing 'kubectl create/apply' examples, "
                "boilerplate YAML templates, or prerequisites for deploying the specified resource type."
            )
        }

        print("✅ Gen-Reranker model loaded.")

    def _format_input_pair(self, instruction: str, query: str, doc: Document) -> str:
        """
        构造模型输入：<Instruct> + <Query> + <Document (Title + Content)>
        """
        # 利用文档元数据中的 Title 增强上下文
        title = doc.metadata.get("title", "Untitled Section")
        content = doc.page_content

        # 显式拼接标题，这对 Reranker 极其重要
        enriched_doc = f"Title: {title}\nContent: {content}"

        return f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {enriched_doc}"

    def _compute_scores(self, pairs: List[str]) -> List[float]:
        """
        核心打分逻辑：手动拼接 tokens 并计算 yes/no 概率
        """
        # 1. Tokenize query+doc pairs (不带 padding)
        inputs = self.tokenizer(
            pairs,
            padding=False,
            truncation='longest_first',
            return_attention_mask=False,
            max_length=self.max_length - len(self.prefix_tokens) - len(self.suffix_tokens)
        )

        # 2. 手动拼接 prefix + content + suffix
        input_ids_list = inputs['input_ids']
        for i, token_ids in enumerate(input_ids_list):
            input_ids_list[i] = self.prefix_tokens + token_ids + self.suffix_tokens

        # 3. Batch Pad
        # tokenizer.pad 会自动生成 attention_mask 并处理 left padding
        batch_inputs = self.tokenizer.pad(
            {'input_ids': input_ids_list},
            padding=True,
            return_tensors="pt"
        )

        # 4. Move to device
        for key in batch_inputs:
            batch_inputs[key] = batch_inputs[key].to(self.device)

        # 5. Inference
        with torch.no_grad():
            outputs = self.model(**batch_inputs)
            # 取最后一个 token 的 logits
            batch_scores = outputs.logits[:, -1, :]

            # 提取 yes 和 no 的 logits
            true_vector = batch_scores[:, self.token_true_id]
            false_vector = batch_scores[:, self.token_false_id]

            # 堆叠并计算 softmax
            combined_logits = torch.stack([false_vector, true_vector], dim=1)
            probs = F.log_softmax(combined_logits, dim=1)

            # 取 index 1 ("yes") 的概率作为最终得分
            scores = probs[:, 1].exp().tolist()

        return scores

    def __call__(self, state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
        print("\n--- [Gen-Rerank Node] Running ---")

        retrieved_docs = state.get("retrieved_chunks", [])
        if len(retrieved_docs) <= 1:
            return {"retrieved_chunks": []}

        # 确定 Query (优先使用 Analysis 阶段的技术摘要)
        analysis = state.get("analysis")

        if analysis:
            # 优先使用技术摘要
            query_text = analysis.technical_summary
            # 获取操作类型
            op_type = analysis.target_operation
            print(f"🎯 Context: {op_type} | Query: {query_text[:50]}...")
        else:
            query_text = state["messages"][-1].content
            op_type = None
            print(f"🎯 Context: Raw Input | Query: {query_text[:50]}...")

        # 根据操作类型生成动态指令，提高重排针对性
        dynamic_instruction = self.op_prompt_map.get(op_type, self.base_instruct)
        print(f"📋 Instruction: {dynamic_instruction}")

        # 准备数据对
        input_texts = [
            self._format_input_pair(dynamic_instruction, query_text, doc)
            for doc in retrieved_docs
        ]

        try:
            # 计算分数
            scores = self._compute_scores(input_texts)

            # 绑定分数并排序
            doc_score_pairs = list(zip(retrieved_docs, scores))
            doc_score_pairs.sort(key=lambda x: x[1], reverse=True)

            print(f"📊 Reranking Results (Top {self.top_n}):")
            reranked_docs = []
            for doc, score in doc_score_pairs[:self.top_n]:
                doc.metadata["rerank_score"] = float(score)
                reranked_docs.append(doc)
                print(f"   Score: {score:.4f} | Source: {doc.metadata.get('source', 'unknown')}")

            return {"retrieved_chunks": reranked_docs}

        except Exception as e:
            print(f"❌ Rerank Failed: {e}")
            # 如果重排失败，降级返回原始结果的前 N 个
            return {"retrieved_chunks": retrieved_docs[:self.top_n]}