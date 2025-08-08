"""
Bridge module for integrating KNN search with LangChain/LangGraph.

Provides retriever interface and vector store compatibility for KNN search,
enabling seamless integration with LangChain-based RAG applications.
"""

from typing import List, Dict, Any, Optional, Tuple, Callable
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.callbacks import CallbackManagerForRetrieverRun
import numpy as np
import asyncio
from functools import partial

from src.services.knn_search_service import (
    KNNSearchService,
    KNNSearchParams,
    SearchStrategy,
    SearchResult
)
from src.config.embedding_config import get_embedding_config, DEFAULT_EMBEDDING_MODEL


class KNNRetriever(BaseRetriever):
    """
    KNN 向量搜尋 Retriever
    實現 LangChain BaseRetriever 介面，可直接用於 LangGraph RAG
    """
    
    search_service: Any
    search_strategy: Any = None # Assuming SearchStrategy is defined elsewhere or will be added
    search_params: Any = None # Assuming KNNSearchParams is defined elsewhere or will be added
    
    def __init__(self, 
                 index_name: str = None,
                 embedding_model: str = DEFAULT_EMBEDDING_MODEL,
                 search_strategy: Any = None, # Assuming SearchStrategy is defined elsewhere or will be added
                 k: int = 10,
                 **kwargs):
        """
        初始化 KNN Retriever
        
        Args:
            index_name: OpenSearch 索引名稱
            embedding_model: Embedding 模型名稱
            search_strategy: 搜尋策略
            k: 返回文件數量
        """
        super().__init__(**kwargs)
        
        # 取得 embedding 配置
        config = get_embedding_config(embedding_model)
        
        # 初始化搜尋服務
        self.search_service = get_knn_search_service(
            index_name=index_name,
            embedding_model=config.model_name
        )
        
        self.search_strategy = search_strategy
        self.search_params = None # Assuming KNNSearchParams is defined elsewhere or will be added
    
    def _get_relevant_documents(
        self, 
        query: str, 
        *, 
        run_manager: Optional[CallbackManagerForRetrieverRun] = None
    ) -> List[Document]:
        """
        同步取得相關文件
        
        Args:
            query: 查詢文字
            run_manager: 回調管理器
            
        Returns:
            相關文件列表
        """
        # 執行異步搜尋
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            results = loop.run_until_complete(
                search_knn(
                    query_text=query,
                    search_service=self.search_service,
                    params=self.search_params,
                    strategy=self.search_strategy
                )
            )
        finally:
            loop.close()
        
        # 轉換為 LangChain Documents
        return create_document_from_knn_result(results)
    
    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Optional[CallbackManagerForRetrieverRun] = None
    ) -> List[Document]:
        """
        異步取得相關文件
        
        Args:
            query: 查詢文字
            run_manager: 回調管理器
            
        Returns:
            相關文件列表
        """
        results = await search_knn(
            query_text=query,
            search_service=self.search_service,
            params=self.search_params,
            strategy=self.search_strategy
        )
        
        return create_document_from_knn_result(results)


def create_hybrid_bm25_retriever(
    index_name: str = None,
    top_k: int = 8
) -> Callable[[str], List[Document]]:
    """
    建立混合 BM25 檢索函式 (用於 LangGraph)
    
    Args:
        index_name: 索引名稱
        top_k: 返回結果數
        
    Returns:
        BM25 搜尋函式
    """
    from opensearchpy import OpenSearch
    from src.config import settings
    
    client = OpenSearch(
        hosts=[{'host': settings.opensearch_host, 'port': settings.opensearch_port}],
        use_ssl=False,
        verify_certs=False
    )
    
    index_name = index_name or settings.opensearch_index
    
    async def bm25_search(query: str) -> List[Document]:
        """執行 BM25 文字搜尋"""
        body = {
            "query": {
                "multi_match": {
                    "query": query,
                    "fields": ["title^2", "content", "tags"],
                    "type": "best_fields",
                    "operator": "or"
                }
            },
            "size": top_k,
            "_source": ["doc_id", "title", "content", "tags", "category", "metadata"],
            "highlight": {
                "fields": {
                    "content": {
                        "fragment_size": 150,
                        "number_of_fragments": 2
                    }
                }
            }
        }
        
        response = client.search(
            index=index_name,
            body=body
        )
        
        # 轉換為 Document
        docs = []
        for hit in response["hits"]["hits"]:
            source = hit["_source"]
            
            # 提取高亮內容
            highlights = []
            if "highlight" in hit:
                highlights = hit["highlight"].get("content", [])
            
            doc = Document(
                page_content=source.get("content", ""),
                metadata={
                    "doc_id": source.get("doc_id", hit["_id"]),
                    "title": source.get("title", ""),
                    "score": hit["_score"],
                    "tags": source.get("tags", []),
                    "category": source.get("category", ""),
                    "highlights": highlights,
                    **source.get("metadata", {})
                }
            )
            docs.append(doc)
        
        return docs
    
    return bm25_search


def create_knn_langraph_components(
    index_name: str = None,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    vector_search_k: int = 10,
    bm25_search_k: int = 8
) -> Dict[str, Any]:
    """
    建立 LangGraph RAG 所需的所有 KNN 元件
    
    Args:
        index_name: 索引名稱
        embedding_model: Embedding 模型
        vector_search_k: 向量搜尋返回數
        bm25_search_k: BM25 搜尋返回數
        
    Returns:
        包含所有必要元件的字典
    """
    # 建立 KNN Retriever (向量搜尋)
    vector_retriever = KNNRetriever(
        index_name=index_name,
        embedding_model=embedding_model,
        search_strategy=None, # Assuming SearchStrategy is defined elsewhere or will be added
        k=vector_search_k
    )
    
    # 建立 Hybrid Retriever (混合搜尋)
    hybrid_retriever = KNNRetriever(
        index_name=index_name,
        embedding_model=embedding_model,
        search_strategy=None, # Assuming SearchStrategy is defined elsewhere or will be added
        k=vector_search_k
    )
    
    # 建立 BM25 搜尋函式
    bm25_search_fn = create_hybrid_bm25_retriever(
        index_name=index_name,
        top_k=bm25_search_k
    )
    
    # 建立上下文建構函式
    def build_context_with_scores(docs: List[Document], max_chars: int = 6000) -> str:
        """建構包含分數的上下文"""
        context_parts = []
        total_chars = 0
        
        for i, doc in enumerate(docs, 1):
            content = doc.page_content.strip()
            title = doc.metadata.get("title", f"文件{i}")
            score = doc.metadata.get("score", 0)
            tags = doc.metadata.get("tags", [])
            
            # 格式化文件
            formatted = f"【{title}】(相關度: {score:.2f})\n"
            if tags:
                formatted += f"標籤: {', '.join(tags)}\n"
            formatted += f"{content}\n"
            
            # 檢查長度限制
            if total_chars + len(formatted) > max_chars:
                break
            
            context_parts.append(formatted)
            total_chars += len(formatted)
        
        return "\n".join(context_parts)
    
    return {
        "vector_retriever": vector_retriever,
        "hybrid_retriever": hybrid_retriever,
        "bm25_search_fn": bm25_search_fn,
        "build_context_fn": build_context_with_scores,
        "search_service": get_knn_search_service(index_name=index_name)
    }


# 範例：如何在 LangGraph 中使用
"""
from app.api.knn_langchain_bridge import create_knn_langraph_components

# 建立所有 KNN 元件
knn_components = create_knn_langraph_components(
    index_name="your_index",
    embedding_model="gemini-embedding-001",
    vector_search_k=10,
    bm25_search_k=8
)

# 在 build_graph 中使用
graph_app = build_graph(
    llm=llm,
    retriever=knn_components["hybrid_retriever"],  # 使用混合搜尋
    bm25_search_fn=knn_components["bm25_search_fn"],
    build_context_fn=knn_components["build_context_fn"],
    policy=policy
)

# 或者直接使用搜尋服務
search_service = knn_components["search_service"]
results = await search_service.knn_search(
    query_text="你的查詢",
    strategy=SearchStrategy.RERANK
)
"""