import torch
import torch.nn.functional as F
from typing import List, Dict, Any
from langchain_core.runnables import RunnableConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

from agent.state import AgentState
from agent.prompts import RERANK_SYSTEM_PROMPT


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

        self.default_instruction = "Evaluate the document based on the system criteria."
        print("✅ Gen-Reranker model loaded.")

    def _format_instruction(self, query: str, doc_content: str):
        return "<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}".format(
            instruction=self.default_instruction, query=query, doc=doc_content
        )

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
        for i, ele in enumerate(input_ids_list):
            input_ids_list[i] = self.prefix_tokens + ele + self.suffix_tokens

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
        if analysis and analysis.technical_summary:
            query = analysis.technical_summary
            print(f"🎯 Query: {query[:50]}...")
        else:
            query = state["messages"][-1].content
            print(f"🎯 Query (Raw): {query[:50]}...")

        # 准备数据对
        pairs = [self._format_instruction(query, doc.page_content) for doc in retrieved_docs]

        try:
            # 计算分数
            scores = self._compute_scores(pairs)

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