import logging
from typing import List, Dict

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_huggingface import HuggingFaceEmbeddings
from pymilvus import (
    Collection,
    AnnSearchRequest,
    RRFRanker
)
from pymilvus.model.hybrid import BGEM3EmbeddingFunction

from utils.milvus_adapter import csr_to_milvus_format, decode_hit_to_document, SCALAR_FIELDS


class MilvusHybridRetriever(BaseRetriever):
    """
    支持 Milvus 混合检索 (Dense + Sparse) 的自定义 Retriever
    """
    collection_name: str
    dense_embedding_func: HuggingFaceEmbeddings  # LangChain Embeddings interface (Qwen)
    sparse_embedding_func: BGEM3EmbeddingFunction  # Pymilvus BGE-M3 Interface

    # 字段配置 (必须与 Schema 一致)
    dense_field: str = "vector"
    sparse_text_field: str = "sparse_vector"
    sparse_title_field: str = "title_sparse"
    text_output_field: str = "text"

    # 搜索配置
    top_k: int = 5
    dense_search_params: Dict = {"metric_type": "COSINE", "params": {"nprobe": 10}}
    sparse_search_params: Dict = {"metric_type": "IP", "params": {"drop_ratio_search": 0.2}}
    logger: logging.Logger = logging.getLogger(__name__)

    def _get_relevant_documents(
            self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        # 1. 生成查询向量
        # 1.1 Dense Vector (Qwen)
        dense_vec = self.dense_embedding_func.embed_query(query)

        # 1.2 Sparse Vector (BGE-M3)
        # 返回的是 CSR 矩阵，需要转换
        sparse_result = self.sparse_embedding_func.encode_queries([query])["sparse"]
        sparse_vec_list = csr_to_milvus_format(sparse_result)
        sparse_vec = sparse_vec_list[0]  # 取出当前 query 的向量

        return self.search_with_vectors(dense_vec, sparse_vec)


    def search_with_vectors(self, dense_vec: List[float], sparse_vec: Dict[int, float]) -> List[Document]:
        """
        计算与检索的核心方法，供 RetrievalNode 调用，使用缓存好的向量进行搜索。
        """
        # 2. 构建搜索请求 (全路召回：Text Dense + Text Sparse + Title Sparse)
        search_requests = [
            # (A) Text 稠密检索
            AnnSearchRequest(
                data=[dense_vec],
                anns_field=self.dense_field,
                param=self.dense_search_params,
                limit=self.top_k
            ),
            # (B) Text 稀疏检索
            AnnSearchRequest(
                data=[sparse_vec],
                anns_field=self.sparse_text_field,
                param=self.sparse_search_params,
                limit=self.top_k
            ),
            # (C) Title 稀疏检索
            AnnSearchRequest(
                data=[sparse_vec],
                anns_field=self.sparse_title_field,
                param=self.sparse_search_params,
                limit=self.top_k
            )
        ]

        # 3. 执行混合搜索 (Hybrid Search)
        # 使用 RRF (Reciprocal Rank Fusion) 进行重排序融合，无需手动调整权重，它基于排名的倒数进行融合，鲁棒性很好
        try:
            collection = Collection(self.collection_name)
            results = collection.hybrid_search(
                reqs=search_requests,
                rerank=RRFRanker(),  # 或者使用 WeightedRanker(0.6, 0.2, 0.2)
                limit=self.top_k,
                output_fields=SCALAR_FIELDS # 只拉取所有标量，不拉取向量
            )
        except Exception as e:
            self.logger.error(f"Hybrid search failed: {e}")  # 建议使用日志而非print
            return []

        # 4. 结果转换为 LangChain Document
        documents = []
        if not results:
            return []

        for hits in results:  # hybrid_search 返回的是 list of hits
            for hit in hits:
                documents.append(decode_hit_to_document(hit, content_field=self.text_output_field))

        return documents