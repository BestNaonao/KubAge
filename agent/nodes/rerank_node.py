from typing import Dict, Any

import torch
from langchain_core.runnables import RunnableConfig
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from agent.state import AgentState


class RerankNode:
    def __init__(self, model_path: str, top_n: int = 5):
        """
        初始化重排节点
        :param model_path: 本地模型路径，如 ".model/Qwen/Qwen3-Reranker-0.6B"
        :param device: 运行设备 "cuda" or "cpu"
        :param top_n: 重排后保留的文档数量
        """
        print(f"⏳ Loading Reranker model from {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.to("cuda" if torch.cuda.is_available() else "cpu")
        self.model.eval()

        self.top_n = top_n
        print("✅ Reranker model loaded.")

    def __call__(self, state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
        print("\n--- [Rerank Node] Running ---")

        retrieved_docs = state.get("retrieved_chunks", [])
        if not retrieved_docs:
            print("⚠️ No documents to rerank.")
            return {"retrieved_chunks": []}

        # 1. 确定重排使用的 Query
        # 策略：使用 Analysis 阶段生成的 Technical Summary (技术摘要) 作为最准确的查询意图
        # 如果没有摘要，回退到用户原始输入
        analysis = state.get("analysis")
        if analysis and analysis.technical_summary:
            query = analysis.technical_summary
            print(f"🎯 Using Technical Summary for reranking: {query[:50]}...")
        else:
            query = state["messages"][-1].content
            print(f"🎯 Using User Input for reranking: {query[:50]}...")

        # 2. 构造模型输入 pairs: [[query, doc1], [query, doc2], ...]
        pairs = [[query, doc.page_content] for doc in retrieved_docs]

        # 3. 执行推理打分
        with torch.no_grad():
            inputs = self.tokenizer(
                pairs,
                padding=True,
                truncation=True,
                return_tensors='pt',
                max_length=512
            ).to("cuda" if torch.cuda.is_available() else "cpu")

            scores = self.model(**inputs, return_dict=True).logits.view(-1, ).float()

        # 4. 排序与截断
        # 将分数与文档绑定
        doc_score_pairs = list(zip(retrieved_docs, scores.cpu().numpy()))

        # 按分数降序排列
        doc_score_pairs.sort(key=lambda x: x[1], reverse=True)

        # 筛选 Top N
        reranked_docs = []
        print(f"📊 Reranking Results (Top {self.top_n}):")
        for doc, score in doc_score_pairs[:self.top_n]:
            # 将重排分数写入 metadata，方便后续 debug
            doc.metadata["rerank_score"] = float(score)
            reranked_docs.append(doc)
            print(f"   Score: {score:.4f} | Source: {doc.metadata.get('source', 'unknown')}")

        return {"retrieved_chunks": reranked_docs}