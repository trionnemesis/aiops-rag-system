"""
KNN 向量搜尋服務
提供進階的 k-NN 查詢功能與多種搜尋策略
"""

from typing import List, Dict, Any, Optional, Tuple
from opensearchpy import OpenSearch
import numpy as np
from datetime import datetime
import logging
from dataclasses import dataclass
from enum import Enum
import time

from src.config import settings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
from src.services.prometheus_service import (
    vector_search_counter,
    vector_search_latency,
    vector_search_results,
    ef_search_value,
    opensearch_cluster_health,
    opensearch_index_docs,
    opensearch_index_size
)

logger = logging.getLogger(__name__)


class SearchStrategy(Enum):
    """搜尋策略列舉"""
    KNN_ONLY = "knn_only"  # 純向量搜尋
    HYBRID = "hybrid"  # 混合搜尋 (向量 + 文字)
    MULTI_VECTOR = "multi_vector"  # 多向量搜尋
    RERANK = "rerank"  # 重新排序


@dataclass
class KNNSearchParams:
    """KNN 搜尋參數"""
    k: int = 10
    num_candidates: int = 100
    boost: float = 1.0
    filter: Optional[Dict[str, Any]] = None
    min_score: Optional[float] = None


@dataclass
class SearchResult:
    """搜尋結果"""
    doc_id: str
    title: str
    content: str
    score: float
    metadata: Dict[str, Any]
    highlights: Optional[List[str]] = None


class KNNSearchService:
    """KNN 向量搜尋服務"""
    
    def __init__(self, 
                 index_name: str = None,
                 embedding_model: str = "models/embedding-001"):
        """
        初始化搜尋服務
        
        Args:
            index_name: 索引名稱
            embedding_model: Embedding 模型名稱
        """
        self.index_name = index_name or settings.opensearch_index
        
        # 初始化 OpenSearch 客戶端
        self.client = OpenSearch(
            hosts=[{'host': settings.opensearch_host, 'port': settings.opensearch_port}],
            use_ssl=False,
            verify_certs=False,
            timeout=30
        )
        
        # 初始化 Embedding 模型
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model=embedding_model,
            google_api_key=settings.google_api_key
        )
        
        logger.info(f"初始化 KNN 搜尋服務 - 索引: {self.index_name}")
    
    async def knn_search(self, 
                        query_text: str,
                        params: KNNSearchParams = None,
                        strategy: SearchStrategy = SearchStrategy.KNN_ONLY) -> List[SearchResult]:
        """
        執行 KNN 搜尋
        
        Args:
            query_text: 查詢文字
            params: 搜尋參數
            strategy: 搜尋策略
            
        Returns:
            搜尋結果列表
        """
        params = params or KNNSearchParams()
        
        # 記錄開始時間
        start_time = time.time()
        
        # 增加搜尋計數
        vector_search_counter.labels(
            strategy=strategy.value,
            index=self.index_name
        ).inc()
        
        try:
            # 生成查詢向量
            query_embedding = await self.embeddings.aembed_query(query_text)
            
            # 根據策略執行搜尋
            if strategy == SearchStrategy.KNN_ONLY:
                results = await self._knn_only_search(query_embedding, params)
            elif strategy == SearchStrategy.HYBRID:
                results = await self._hybrid_search(query_text, query_embedding, params)
            elif strategy == SearchStrategy.MULTI_VECTOR:
                results = await self._multi_vector_search(query_text, params)
            elif strategy == SearchStrategy.RERANK:
                results = await self._rerank_search(query_text, query_embedding, params)
            else:
                raise ValueError(f"不支援的搜尋策略: {strategy}")
            
            # 記錄結果數量
            vector_search_results.labels(
                strategy=strategy.value,
                index=self.index_name
            ).observe(len(results))
            
            return results
            
        finally:
            # 記錄執行時間
            duration = time.time() - start_time
            vector_search_latency.labels(
                strategy=strategy.value,
                index=self.index_name
            ).observe(duration)
    
    async def _knn_only_search(self, 
                              query_embedding: List[float],
                              params: KNNSearchParams) -> List[SearchResult]:
        """
        純 KNN 向量搜尋
        
        Args:
            query_embedding: 查詢向量
            params: 搜尋參數
            
        Returns:
            搜尋結果
        """
        # 建構 KNN 查詢
        knn_query = {
            "knn": {
                "embedding": {
                    "vector": query_embedding,
                    "k": params.k,
                    "num_candidates": params.num_candidates,
                    "boost": params.boost
                }
            }
        }
        
        # 加入過濾條件
        if params.filter:
            knn_query["knn"]["embedding"]["filter"] = params.filter
        
        # 執行搜尋
        body = {
            "size": params.k,
            "query": knn_query,
            "_source": ["doc_id", "title", "content", "tags", "category", "metadata"],
            "min_score": params.min_score
        }
        
        response = self.client.search(
            index=self.index_name,
            body=body
        )
        
        return self._parse_search_results(response)
    
    async def _hybrid_search(self, 
                           query_text: str,
                           query_embedding: List[float],
                           params: KNNSearchParams) -> List[SearchResult]:
        """
        混合搜尋 (向量 + BM25)
        
        Args:
            query_text: 查詢文字
            query_embedding: 查詢向量
            params: 搜尋參數
            
        Returns:
            搜尋結果
        """
        # 建構混合查詢
        body = {
            "size": params.k,
            "query": {
                "bool": {
                    "should": [
                        # KNN 查詢
                        {
                            "knn": {
                                "embedding": {
                                    "vector": query_embedding,
                                    "k": params.k,
                                    "num_candidates": params.num_candidates,
                                    "boost": params.boost
                                }
                            }
                        },
                        # BM25 文字搜尋
                        {
                            "multi_match": {
                                "query": query_text,
                                "fields": ["title^2", "content", "tags"],
                                "type": "best_fields",
                                "boost": 0.5
                            }
                        }
                    ]
                }
            },
            "_source": ["doc_id", "title", "content", "tags", "category", "metadata"],
            "highlight": {
                "fields": {
                    "content": {
                        "fragment_size": 150,
                        "number_of_fragments": 3
                    }
                }
            }
        }
        
        # 加入過濾條件
        if params.filter:
            body["query"]["bool"]["filter"] = params.filter
        
        response = self.client.search(
            index=self.index_name,
            body=body
        )
        
        return self._parse_search_results(response, include_highlights=True)
    
    async def _multi_vector_search(self, 
                                 query_text: str,
                                 params: KNNSearchParams) -> List[SearchResult]:
        """
        多向量搜尋 (使用查詢擴展)
        
        Args:
            query_text: 查詢文字
            params: 搜尋參數
            
        Returns:
            搜尋結果
        """
        # 生成多個查詢變體
        query_variants = await self._generate_query_variants(query_text)
        
        # 生成多個向量
        embeddings = []
        for variant in query_variants:
            embedding = await self.embeddings.aembed_query(variant)
            embeddings.append(embedding)
        
        # 執行多個 KNN 查詢
        all_results = []
        for embedding in embeddings:
            results = await self._knn_only_search(embedding, params)
            all_results.extend(results)
        
        # 去重並重新排序
        unique_results = self._deduplicate_and_rerank(all_results)
        
        return unique_results[:params.k]
    
    async def _rerank_search(self, 
                           query_text: str,
                           query_embedding: List[float],
                           params: KNNSearchParams) -> List[SearchResult]:
        """
        重新排序搜尋結果
        
        Args:
            query_text: 查詢文字
            query_embedding: 查詢向量
            params: 搜尋參數
            
        Returns:
            重新排序的結果
        """
        # 先執行寬鬆的搜尋
        loose_params = KNNSearchParams(
            k=params.k * 3,  # 取更多候選結果
            num_candidates=params.num_candidates * 2,
            filter=params.filter
        )
        
        initial_results = await self._knn_only_search(query_embedding, loose_params)
        
        # 重新計算分數
        reranked_results = []
        for result in initial_results:
            # 計算語義相似度
            content_embedding = await self.embeddings.aembed_query(result.content[:500])
            semantic_score = self._cosine_similarity(query_embedding, content_embedding)
            
            # 計算關鍵詞匹配度
            keyword_score = self._calculate_keyword_score(query_text, result.content)
            
            # 綜合分數
            combined_score = (semantic_score * 0.7) + (keyword_score * 0.3)
            
            result.score = combined_score
            reranked_results.append(result)
        
        # 重新排序
        reranked_results.sort(key=lambda x: x.score, reverse=True)
        
        return reranked_results[:params.k]
    
    async def _generate_query_variants(self, query_text: str) -> List[str]:
        """
        生成查詢變體
        
        Args:
            query_text: 原始查詢
            
        Returns:
            查詢變體列表
        """
        # 這裡可以使用 LLM 生成查詢變體
        # 簡單示例：返回原查詢和一些變體
        return [
            query_text,
            f"如何{query_text}",
            f"{query_text}的最佳實踐",
            f"解決{query_text}的方法"
        ]
    
    def _parse_search_results(self, 
                            response: Dict[str, Any],
                            include_highlights: bool = False) -> List[SearchResult]:
        """
        解析搜尋結果
        
        Args:
            response: OpenSearch 回應
            include_highlights: 是否包含高亮
            
        Returns:
            搜尋結果列表
        """
        results = []
        
        for hit in response["hits"]["hits"]:
            source = hit["_source"]
            
            # 提取高亮
            highlights = None
            if include_highlights and "highlight" in hit:
                highlights = hit["highlight"].get("content", [])
            
            result = SearchResult(
                doc_id=source.get("doc_id", hit["_id"]),
                title=source.get("title", ""),
                content=source.get("content", ""),
                score=hit["_score"],
                metadata={
                    "tags": source.get("tags", []),
                    "category": source.get("category", ""),
                    **source.get("metadata", {})
                },
                highlights=highlights
            )
            results.append(result)
        
        return results
    
    def _deduplicate_and_rerank(self, results: List[SearchResult]) -> List[SearchResult]:
        """
        去重並重新排序結果
        
        Args:
            results: 原始結果列表
            
        Returns:
            去重後的結果
        """
        seen = {}
        unique_results = []
        
        for result in results:
            if result.doc_id not in seen:
                seen[result.doc_id] = result
                unique_results.append(result)
            else:
                # 保留分數較高的
                if result.score > seen[result.doc_id].score:
                    seen[result.doc_id] = result
        
        # 按分數排序
        unique_results = list(seen.values())
        unique_results.sort(key=lambda x: x.score, reverse=True)
        
        return unique_results
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """
        計算餘弦相似度
        
        Args:
            vec1: 向量1
            vec2: 向量2
            
        Returns:
            相似度分數
        """
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def _calculate_keyword_score(self, query: str, content: str) -> float:
        """
        計算關鍵詞匹配分數
        
        Args:
            query: 查詢文字
            content: 內容文字
            
        Returns:
            匹配分數
        """
        query_words = set(query.lower().split())
        content_words = set(content.lower().split())
        
        if not query_words:
            return 0.0
        
        matches = len(query_words & content_words)
        return matches / len(query_words)
    
    async def explain_search(self, 
                           query_text: str,
                           doc_id: str) -> Dict[str, Any]:
        """
        解釋搜尋結果的評分
        
        Args:
            query_text: 查詢文字
            doc_id: 文件 ID
            
        Returns:
            評分解釋
        """
        # 生成查詢向量
        query_embedding = await self.embeddings.aembed_query(query_text)
        
        # 建構解釋查詢
        body = {
            "query": {
                "knn": {
                    "embedding": {
                        "vector": query_embedding,
                        "k": 1,
                        "filter": {
                            "term": {"doc_id": doc_id}
                        }
                    }
                }
            },
            "explain": True,
            "_source": ["doc_id", "title"],
            "size": 1
        }
        
        response = self.client.search(
            index=self.index_name,
            body=body
        )
        
        if response["hits"]["hits"]:
            hit = response["hits"]["hits"][0]
            return {
                "doc_id": doc_id,
                "title": hit["_source"].get("title", ""),
                "score": hit["_score"],
                "explanation": hit.get("_explanation", {})
            }
        
        return {"error": "Document not found"}
    
    def to_langchain_documents(self, results: List[SearchResult]) -> List[Document]:
        """
        轉換搜尋結果為 LangChain Document 格式
        
        Args:
            results: 搜尋結果
            
        Returns:
            LangChain Document 列表
        """
        documents = []
        
        for result in results:
            doc = Document(
                page_content=result.content,
                metadata={
                    "doc_id": result.doc_id,
                    "title": result.title,
                    "score": result.score,
                    **result.metadata
                }
            )
            documents.append(doc)
        
        return documents