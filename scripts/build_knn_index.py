#!/usr/bin/env python3
"""
建立 OpenSearch KNN Vector 索引
支援 HNSW 算法參數配置與批次文件索引
"""

import asyncio
import json
import sys
import os
from typing import List, Dict, Any, Optional
from datetime import datetime
import numpy as np
from tqdm import tqdm
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from opensearchpy import OpenSearch, helpers
from src.config import settings
from src.services.gemini_service import GeminiService
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import logging

# 設定日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class KNNIndexBuilder:
    """KNN 向量索引建置器"""
    
    def __init__(self, 
                 index_name: str = None,
                 embedding_dim: int = None,
                 model_name: str = "models/embedding-001"):
        """
        初始化索引建置器
        
        Args:
            index_name: 索引名稱
            embedding_dim: 向量維度
            model_name: Embedding 模型名稱
        """
        self.index_name = index_name or settings.opensearch_index
        self.embedding_dim = embedding_dim or settings.opensearch_embedding_dim
        self.model_name = model_name
        
        # 初始化 OpenSearch 客戶端
        self.client = OpenSearch(
            hosts=[{'host': settings.opensearch_host, 'port': settings.opensearch_port}],
            use_ssl=False,
            verify_certs=False,
            timeout=60
        )
        
        # 初始化 Embedding 模型
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model=self.model_name,
            google_api_key=settings.google_api_key
        )
        
        logger.info(f"初始化完成 - 索引: {self.index_name}, 維度: {self.embedding_dim}")
    
    def create_knn_index(self, 
                        hnsw_m: int = 16,
                        hnsw_ef_construction: int = 128,
                        hnsw_ef_search: int = 100,
                        space_type: str = "l2",
                        engine: str = "nmslib") -> Dict[str, Any]:
        """
        建立 KNN 索引與映射
        
        Args:
            hnsw_m: HNSW M 參數 (圖中每個節點的最大連接數)
            hnsw_ef_construction: HNSW 建構時的動態列表大小
            hnsw_ef_search: 搜尋時的動態列表大小
            space_type: 距離計算方式 (l2, l1, linf, cosinesimil)
            engine: KNN 引擎 (nmslib, faiss, lucene)
            
        Returns:
            建立結果
        """
        # 檢查索引是否已存在
        if self.client.indices.exists(index=self.index_name):
            logger.warning(f"索引 {self.index_name} 已存在")
            response = input("是否要刪除現有索引並重建? (y/N): ")
            if response.lower() == 'y':
                self.client.indices.delete(index=self.index_name)
                logger.info(f"已刪除索引 {self.index_name}")
            else:
                return {"status": "cancelled", "message": "索引已存在，取消建立"}
        
        # 索引設定
        index_body = {
            "settings": {
                "index": {
                    "knn": True,
                    "knn.algo_param.ef_search": hnsw_ef_search,
                    "number_of_shards": 3,
                    "number_of_replicas": 1,
                    "refresh_interval": "30s",
                    "max_result_window": 10000
                }
            },
            "mappings": {
                "properties": {
                    # 文件識別碼
                    "doc_id": {
                        "type": "keyword"
                    },
                    # 文件標題
                    "title": {
                        "type": "text",
                        "analyzer": "standard",
                        "fields": {
                            "keyword": {
                                "type": "keyword",
                                "ignore_above": 256
                            }
                        }
                    },
                    # 文件內容
                    "content": {
                        "type": "text",
                        "analyzer": "standard"
                    },
                    # 標籤
                    "tags": {
                        "type": "keyword"
                    },
                    # 分類
                    "category": {
                        "type": "keyword"
                    },
                    # KNN 向量欄位
                    "embedding": {
                        "type": "knn_vector",
                        "dimension": self.embedding_dim,
                        "method": {
                            "name": "hnsw",
                            "space_type": space_type,
                            "engine": engine,
                            "parameters": {
                                "ef_construction": hnsw_ef_construction,
                                "m": hnsw_m
                            }
                        }
                    },
                    # 元資料
                    "metadata": {
                        "type": "object",
                        "enabled": True
                    },
                    # 建立時間
                    "created_at": {
                        "type": "date",
                        "format": "yyyy-MM-dd HH:mm:ss||yyyy-MM-dd||epoch_millis"
                    },
                    # 更新時間
                    "updated_at": {
                        "type": "date",
                        "format": "yyyy-MM-dd HH:mm:ss||yyyy-MM-dd||epoch_millis"
                    }
                }
            }
        }
        
        # 建立索引
        try:
            response = self.client.indices.create(
                index=self.index_name,
                body=index_body
            )
            logger.info(f"成功建立索引 {self.index_name}")
            logger.info(f"HNSW 參數 - M: {hnsw_m}, EF Construction: {hnsw_ef_construction}, EF Search: {hnsw_ef_search}")
            return response
        except Exception as e:
            logger.error(f"建立索引失敗: {str(e)}")
            raise
    
    async def generate_embedding(self, text: str) -> List[float]:
        """
        生成文字的向量表示
        
        Args:
            text: 輸入文字
            
        Returns:
            向量表示
        """
        try:
            embedding = await self.embeddings.aembed_query(text)
            # 確保維度正確
            if len(embedding) != self.embedding_dim:
                logger.warning(f"向量維度不符: 預期 {self.embedding_dim}, 實際 {len(embedding)}")
            return embedding
        except Exception as e:
            logger.error(f"生成向量失敗: {str(e)}")
            raise
    
    async def index_document(self, 
                           doc_id: str,
                           title: str,
                           content: str,
                           tags: List[str] = None,
                           category: str = None,
                           metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        索引單一文件
        
        Args:
            doc_id: 文件 ID
            title: 標題
            content: 內容
            tags: 標籤列表
            category: 分類
            metadata: 元資料
            
        Returns:
            索引結果
        """
        # 生成向量
        embedding_text = f"{title}\n{content}"
        embedding = await self.generate_embedding(embedding_text)
        
        # 準備文件
        document = {
            "doc_id": doc_id,
            "title": title,
            "content": content,
            "tags": tags or [],
            "category": category or "general",
            "embedding": embedding,
            "metadata": metadata or {},
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # 索引文件
        try:
            response = self.client.index(
                index=self.index_name,
                id=doc_id,
                body=document,
                refresh=False  # 批次索引時設為 False
            )
            return response
        except Exception as e:
            logger.error(f"索引文件 {doc_id} 失敗: {str(e)}")
            raise
    
    async def bulk_index_documents(self, 
                                 documents: List[Dict[str, Any]],
                                 batch_size: int = 100) -> Dict[str, Any]:
        """
        批次索引文件
        
        Args:
            documents: 文件列表
            batch_size: 批次大小
            
        Returns:
            索引結果統計
        """
        total = len(documents)
        success = 0
        failed = 0
        
        logger.info(f"開始批次索引 {total} 個文件")
        
        # 分批處理
        for i in range(0, total, batch_size):
            batch = documents[i:i + batch_size]
            actions = []
            
            # 生成向量並準備批次動作
            for doc in tqdm(batch, desc=f"處理批次 {i//batch_size + 1}"):
                try:
                    # 生成向量
                    embedding_text = f"{doc.get('title', '')}\n{doc.get('content', '')}"
                    embedding = await self.generate_embedding(embedding_text)
                    
                    # 準備索引動作
                    action = {
                        "_index": self.index_name,
                        "_id": doc["doc_id"],
                        "_source": {
                            "doc_id": doc["doc_id"],
                            "title": doc.get("title", ""),
                            "content": doc.get("content", ""),
                            "tags": doc.get("tags", []),
                            "category": doc.get("category", "general"),
                            "embedding": embedding,
                            "metadata": doc.get("metadata", {}),
                            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        }
                    }
                    actions.append(action)
                    
                except Exception as e:
                    logger.error(f"處理文件 {doc.get('doc_id', 'unknown')} 失敗: {str(e)}")
                    failed += 1
            
            # 執行批次索引
            if actions:
                try:
                    response = helpers.bulk(self.client, actions)
                    success += response[0]
                    failed += response[1]
                except Exception as e:
                    logger.error(f"批次索引失敗: {str(e)}")
                    failed += len(actions)
        
        # 刷新索引
        self.client.indices.refresh(index=self.index_name)
        
        result = {
            "total": total,
            "success": success,
            "failed": failed,
            "success_rate": f"{(success/total)*100:.2f}%"
        }
        
        logger.info(f"批次索引完成: {result}")
        return result
    
    def get_index_stats(self) -> Dict[str, Any]:
        """
        取得索引統計資訊
        
        Returns:
            索引統計
        """
        try:
            # 取得索引統計
            stats = self.client.indices.stats(index=self.index_name)
            
            # 取得文件數量
            count = self.client.count(index=self.index_name)
            
            # 取得索引設定
            settings = self.client.indices.get_settings(index=self.index_name)
            
            # 取得映射
            mappings = self.client.indices.get_mapping(index=self.index_name)
            
            return {
                "index_name": self.index_name,
                "document_count": count["count"],
                "size_in_bytes": stats["indices"][self.index_name]["total"]["store"]["size_in_bytes"],
                "shards": settings[self.index_name]["settings"]["index"]["number_of_shards"],
                "replicas": settings[self.index_name]["settings"]["index"]["number_of_replicas"],
                "knn_enabled": settings[self.index_name]["settings"]["index"].get("knn", False),
                "ef_search": settings[self.index_name]["settings"]["index"].get("knn.algo_param.ef_search", "N/A"),
                "embedding_dimension": mappings[self.index_name]["mappings"]["properties"]["embedding"]["dimension"]
            }
        except Exception as e:
            logger.error(f"取得索引統計失敗: {str(e)}")
            return {"error": str(e)}


async def main():
    """主程式"""
    parser = argparse.ArgumentParser(description="建立 OpenSearch KNN Vector 索引")
    parser.add_argument("--index-name", type=str, help="索引名稱")
    parser.add_argument("--dimension", type=int, default=768, help="向量維度")
    parser.add_argument("--hnsw-m", type=int, default=16, help="HNSW M 參數")
    parser.add_argument("--hnsw-ef-construction", type=int, default=128, help="HNSW EF Construction")
    parser.add_argument("--hnsw-ef-search", type=int, default=100, help="HNSW EF Search")
    parser.add_argument("--load-sample", action="store_true", help="載入範例資料")
    
    args = parser.parse_args()
    
    # 建立索引建置器
    builder = KNNIndexBuilder(
        index_name=args.index_name,
        embedding_dim=args.dimension
    )
    
    # 建立索引
    result = builder.create_knn_index(
        hnsw_m=args.hnsw_m,
        hnsw_ef_construction=args.hnsw_ef_construction,
        hnsw_ef_search=args.hnsw_ef_search
    )
    
    print(json.dumps(result, indent=2, ensure_ascii=False))
    
    # 載入範例資料
    if args.load_sample:
        from scripts.init_opensearch import SAMPLE_DOCUMENTS
        
        # 準備文件
        documents = []
        for doc in SAMPLE_DOCUMENTS:
            documents.append({
                "doc_id": doc["event_id"],
                "title": doc["title"],
                "content": doc["content"],
                "tags": doc["tags"],
                "category": "incident_report"
            })
        
        # 批次索引
        result = await builder.bulk_index_documents(documents)
        print("\n批次索引結果:")
        print(json.dumps(result, indent=2, ensure_ascii=False))
    
    # 顯示索引統計
    stats = builder.get_index_stats()
    print("\n索引統計:")
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main())