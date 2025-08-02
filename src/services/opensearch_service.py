from opensearchpy import OpenSearch, AsyncOpenSearch
from typing import List, Dict, Any, Optional
import numpy as np
from src.config import settings
from src.services.knn_search_service import KNNSearchService, KNNSearchParams, SearchStrategy, SearchResult
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import json
import logging
import asyncio
from src.services.prometheus_service import (
    opensearch_cluster_health,
    opensearch_index_docs,
    opensearch_index_size,
    ef_search_value
)

logger = logging.getLogger(__name__)

class OpenSearchService:
    def __init__(self):
        self.client = OpenSearch(
            hosts=[{'host': settings.opensearch_host, 'port': settings.opensearch_port}],
            use_ssl=False,
            verify_certs=False
        )
        self.index_name = settings.opensearch_index
        
        # 初始化 KNN 搜尋服務
        self.knn_service = KNNSearchService(index_name=self.index_name)
        
        # 初始化 Embedding 模型 (確保與 KNN 服務使用相同模型)
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=settings.google_api_key
        )
        
    async def create_index(self):
        """創建 OpenSearch 索引與映射"""
        index_body = {
            "settings": {
                "index": {
                    "knn": True,
                    "knn.algo_param.ef_search": 100
                }
            },
            "mappings": {
                "properties": {
                    "event_id": {"type": "keyword"},
                    "title": {"type": "text"},
                    "content": {"type": "text"},
                    "tags": {"type": "keyword"},
                    "embedding": {
                        "type": "knn_vector",
                        "dimension": settings.opensearch_embedding_dim,
                        "method": {
                            "name": "hnsw",
                            "space_type": "l2",
                            "engine": "nmslib",
                            "parameters": {
                                "ef_construction": 128,
                                "m": 24
                            }
                        }
                    },
                    "created_at": {"type": "date"}
                }
            }
        }
        
        if not self.client.indices.exists(index=self.index_name):
            self.client.indices.create(index=self.index_name, body=index_body)
    
    async def index_document(self, doc_id: str, content: str, embedding: List[float], 
                           title: str = "", tags: List[str] = None):
        """索引文檔到 OpenSearch"""
        document = {
            "event_id": doc_id,
            "title": title,
            "content": content,
            "tags": tags or [],
            "embedding": embedding,
            "created_at": "now"
        }
        
        response = self.client.index(
            index=self.index_name,
            id=doc_id,
            body=document,
            refresh=True
        )
        return response
    
    async def search_similar_documents(self, query_text: str = None, 
                                     query_embedding: List[float] = None,
                                     k: int = None,
                                     strategy: SearchStrategy = SearchStrategy.HYBRID,
                                     filter_dict: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """使用 k-NN 搜尋相似文檔
        
        Args:
            query_text: 查詢文字 (如果提供將生成 embedding)
            query_embedding: 查詢向量 (如果已有向量可直接使用)
            k: 返回結果數
            strategy: 搜尋策略
            filter_dict: 過濾條件
            
        Returns:
            搜尋結果列表
        """
        k = k or settings.top_k_results
        
        # 如果提供文字而非向量，使用 KNN 服務進行搜尋
        if query_text:
            params = KNNSearchParams(
                k=k,
                filter=filter_dict,
                num_candidates=k * 10  # HNSW 候選數量
            )
            
            results = await self.knn_service.knn_search(
                query_text=query_text,
                params=params,
                strategy=strategy
            )
            
            # 轉換為原格式
            return [{
                "event_id": r.doc_id,
                "title": r.title,
                "content": r.content,
                "tags": r.metadata.get("tags", []),
                "score": r.score
            } for r in results]
        
        # 向後兼容：如果直接提供向量
        elif query_embedding:
            # 確保向量維度正確
            if len(query_embedding) != settings.opensearch_embedding_dim:
                logger.warning(f"向量維度不符: 預期 {settings.opensearch_embedding_dim}, 實際 {len(query_embedding)}")
            
            query = {
                "size": k,
                "query": {
                    "knn": {
                        "embedding": {
                            "vector": query_embedding,
                            "k": k,
                            "num_candidates": k * 10
                        }
                    }
                },
                "_source": ["event_id", "title", "content", "tags"]
            }
            
            if filter_dict:
                query["query"]["knn"]["embedding"]["filter"] = filter_dict
            
            response = self.client.search(
                index=self.index_name,
                body=query
            )
            
            return [{
                **hit["_source"],
                "score": hit["_score"]
            } for hit in response["hits"]["hits"]]
        
        else:
            raise ValueError("必須提供 query_text 或 query_embedding")
    
    async def delete_index(self):
        """刪除索引（用於測試）"""
        if self.client.indices.exists(index=self.index_name):
            self.client.indices.delete(index=self.index_name)
    
    async def update_metrics(self):
        """更新 Prometheus 指標"""
        try:
            # 獲取叢集健康狀態
            health = self.client.cluster.health()
            health_value = {"green": 2, "yellow": 1, "red": 0}.get(health["status"], 0)
            opensearch_cluster_health.labels(cluster="opensearch").set(health_value)
            
            # 獲取索引統計
            if self.client.indices.exists(index=self.index_name):
                stats = self.client.indices.stats(index=self.index_name)
                
                # 文檔數量
                doc_count = stats["indices"][self.index_name]["primaries"]["docs"]["count"]
                opensearch_index_docs.labels(index=self.index_name).set(doc_count)
                
                # 索引大小
                index_size = stats["indices"][self.index_name]["primaries"]["store"]["size_in_bytes"]
                opensearch_index_size.labels(index=self.index_name).set(index_size)
                
                # 獲取當前 ef_search 值
                settings = self.client.indices.get_settings(index=self.index_name)
                ef_search = settings[self.index_name]["settings"]["index"].get("knn.algo_param.ef_search", 100)
                ef_search_value.labels(index=self.index_name).set(int(ef_search))
                
        except Exception as e:
            logger.error(f"更新 Prometheus 指標失敗: {e}")
    
    async def start_metrics_collection(self, interval: int = 30):
        """開始定期收集指標
        
        Args:
            interval: 收集間隔（秒）
        """
        while True:
            await self.update_metrics()
            await asyncio.sleep(interval)