from opensearchpy import OpenSearch, AsyncOpenSearch
from typing import List, Dict, Any
import numpy as np
from src.config import settings
import json

class OpenSearchService:
    def __init__(self):
        self.client = OpenSearch(
            hosts=[{'host': settings.opensearch_host, 'port': settings.opensearch_port}],
            use_ssl=False,
            verify_certs=False
        )
        self.index_name = settings.opensearch_index
        
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
    
    async def search_similar_documents(self, query_embedding: List[float], 
                                     k: int = None) -> List[Dict[str, Any]]:
        """使用 k-NN 搜尋相似文檔"""
        k = k or settings.top_k_results
        
        query = {
            "size": k,
            "query": {
                "knn": {
                    "embedding": {
                        "vector": query_embedding,
                        "k": k
                    }
                }
            },
            "_source": ["event_id", "title", "content", "tags"]
        }
        
        response = self.client.search(
            index=self.index_name,
            body=query
        )
        
        return [hit["_source"] for hit in response["hits"]["hits"]]
    
    async def delete_index(self):
        """刪除索引（用於測試）"""
        if self.client.indices.exists(index=self.index_name):
            self.client.indices.delete(index=self.index_name)