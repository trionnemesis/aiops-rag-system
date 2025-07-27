"""
LangChain 向量資料庫管理器
使用 LangChain 的 VectorStore 介面抽象化向量資料庫操作
"""
from typing import List, Dict, Any, Optional
from langchain_community.vectorstores import OpenSearchVectorSearch
from langchain_core.vectorstores import VectorStore
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from opensearchpy import OpenSearch
from src.config import settings
from src.services.langchain.model_manager import model_manager


class VectorStoreManager:
    """統一的向量資料庫管理器"""
    
    def __init__(self):
        self._vector_store: Optional[VectorStore] = None
        self._opensearch_client: Optional[OpenSearch] = None
        
    @property
    def opensearch_client(self) -> OpenSearch:
        """獲取 OpenSearch 客戶端"""
        if self._opensearch_client is None:
            self._opensearch_client = OpenSearch(
                hosts=[{
                    'host': settings.opensearch_host, 
                    'port': settings.opensearch_port
                }],
                use_ssl=False,
                verify_certs=False
            )
        return self._opensearch_client
    
    @property
    def vector_store(self) -> VectorStore:
        """獲取向量資料庫實例"""
        if self._vector_store is None:
            self._vector_store = OpenSearchVectorSearch(
                opensearch_url=f"http://{settings.opensearch_host}:{settings.opensearch_port}",
                index_name=settings.opensearch_index,
                embedding_function=model_manager.embedding_model,
                engine="nmslib",
                space_type="l2",
                ef_construction=128,
                m=24,
                http_auth=None,
                use_ssl=False,
                verify_certs=False,
                ssl_assert_hostname=False,
                ssl_show_warn=False
            )
        return self._vector_store
    
    async def create_index(self):
        """創建向量索引（如果不存在）"""
        # OpenSearchVectorSearch 會在第一次添加文檔時自動創建索引
        # 但我們可以手動創建以確保索引設置正確
        index_body = {
            "settings": {
                "index": {
                    "knn": True,
                    "knn.algo_param.ef_search": 100
                }
            },
            "mappings": {
                "properties": {
                    "vector_field": {
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
                    "text": {"type": "text"},
                    "metadata": {"type": "object"}
                }
            }
        }
        
        if not self.opensearch_client.indices.exists(index=settings.opensearch_index):
            self.opensearch_client.indices.create(
                index=settings.opensearch_index, 
                body=index_body
            )
    
    async def add_documents(self, documents: List[Document]) -> List[str]:
        """添加文檔到向量資料庫
        
        Args:
            documents: LangChain Document 物件列表
            
        Returns:
            文檔 ID 列表
        """
        return await self.vector_store.aadd_documents(documents)
    
    async def add_texts(
        self, 
        texts: List[str], 
        metadatas: Optional[List[dict]] = None
    ) -> List[str]:
        """添加文本到向量資料庫
        
        Args:
            texts: 文本列表
            metadatas: 元數據列表
            
        Returns:
            文檔 ID 列表
        """
        return await self.vector_store.aadd_texts(texts, metadatas)
    
    def as_retriever(self, **kwargs):
        """獲取檢索器
        
        Args:
            **kwargs: 檢索器參數，如 search_kwargs={"k": 5}
            
        Returns:
            VectorStoreRetriever 實例
        """
        search_kwargs = kwargs.get("search_kwargs", {})
        if "k" not in search_kwargs:
            search_kwargs["k"] = settings.top_k_results
        
        return self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs=search_kwargs
        )
    
    async def similarity_search(
        self, 
        query: str, 
        k: Optional[int] = None,
        filter: Optional[dict] = None
    ) -> List[Document]:
        """相似度搜尋
        
        Args:
            query: 查詢文本
            k: 返回結果數量
            filter: 過濾條件
            
        Returns:
            相似文檔列表
        """
        k = k or settings.top_k_results
        return await self.vector_store.asimilarity_search(
            query, 
            k=k,
            filter=filter
        )
    
    async def similarity_search_with_score(
        self, 
        query: str, 
        k: Optional[int] = None,
        filter: Optional[dict] = None
    ) -> List[tuple[Document, float]]:
        """帶分數的相似度搜尋
        
        Args:
            query: 查詢文本
            k: 返回結果數量
            filter: 過濾條件
            
        Returns:
            (文檔, 分數) 元組列表
        """
        k = k or settings.top_k_results
        return await self.vector_store.asimilarity_search_with_score(
            query, 
            k=k,
            filter=filter
        )
    
    async def delete_index(self):
        """刪除索引（用於測試或重建）"""
        if self.opensearch_client.indices.exists(index=settings.opensearch_index):
            self.opensearch_client.indices.delete(index=settings.opensearch_index)
    
    def get_retriever_with_hyde(self, hyde_prompt_template):
        """獲取帶 HyDE 的檢索器
        
        Args:
            hyde_prompt_template: HyDE 提示詞模板
            
        Returns:
            帶 HyDE 的檢索器
        """
        from langchain.retrievers import HyDERetriever
        
        return HyDERetriever(
            base_retriever=self.as_retriever(),
            llm=model_manager.flash_model,
            prompt=hyde_prompt_template
        )


# 全局單例實例
vector_store_manager = VectorStoreManager()