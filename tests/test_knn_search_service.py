"""
KNN 搜尋服務單元測試
測試各種搜尋策略的內部邏輯、參數處理和錯誤處理
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from typing import List, Dict, Any
import numpy as np

from src.services.knn_search_service import (
    KNNSearchService,
    SearchStrategy,
    KNNSearchParams,
    SearchResult
)


@pytest.fixture
def mock_opensearch_client():
    """模擬 OpenSearch 客戶端"""
    client = Mock()
    client.search = Mock(return_value={
        "hits": {
            "total": {"value": 2},
            "hits": [
                {
                    "_id": "1",
                    "_score": 0.95,
                    "_source": {
                        "doc_id": "doc1",
                        "title": "Test Document 1",
                        "content": "This is test content 1",
                        "tags": ["test", "document"],
                        "category": "test",
                        "metadata": {"author": "test_author"}
                    }
                },
                {
                    "_id": "2",
                    "_score": 0.85,
                    "_source": {
                        "doc_id": "doc2", 
                        "title": "Test Document 2",
                        "content": "This is test content 2",
                        "tags": ["test"],
                        "category": "test",
                        "metadata": {"author": "test_author2"}
                    }
                }
            ]
        }
    })
    return client


@pytest.fixture
def mock_embeddings():
    """模擬 Embedding 模型"""
    embeddings = Mock()
    # 返回固定的向量
    embeddings.aembed_query = AsyncMock(return_value=[0.1] * 768)
    return embeddings


@pytest.fixture
def knn_service(mock_opensearch_client, mock_embeddings):
    """建立測試用的 KNN 搜尋服務"""
    with patch('src.services.knn_search_service.OpenSearch', return_value=mock_opensearch_client):
        with patch('src.services.knn_search_service.GoogleGenerativeAIEmbeddings', return_value=mock_embeddings):
            service = KNNSearchService(index_name="test_index")
            return service


class TestKNNSearchService:
    """KNN 搜尋服務測試類別"""
    
    @pytest.mark.asyncio
    async def test_knn_only_search(self, knn_service: KNNSearchService):
        """測試純 KNN 向量搜尋"""
        # 設定參數
        params = KNNSearchParams(k=5, num_candidates=50, boost=1.5)
        query_text = "test query"
        
        # 執行搜尋
        results = await knn_service.knn_search(
            query_text=query_text,
            params=params,
            strategy=SearchStrategy.KNN_ONLY
        )
        
        # 驗證結果
        assert len(results) == 2
        assert results[0].doc_id == "doc1"
        assert results[0].score == 0.95
        assert results[0].title == "Test Document 1"
        
        # 驗證 OpenSearch 查詢
        knn_service.client.search.assert_called_once()
        call_args = knn_service.client.search.call_args
        query_body = call_args[1]['body']
        
        # 檢查 KNN 查詢結構
        assert 'knn' in query_body['query']
        assert query_body['query']['knn']['embedding']['k'] == 5
        assert query_body['query']['knn']['embedding']['num_candidates'] == 50
        assert query_body['query']['knn']['embedding']['boost'] == 1.5
    
    @pytest.mark.asyncio
    async def test_knn_search_with_filter(self, knn_service: KNNSearchService):
        """測試帶過濾條件的 KNN 搜尋"""
        # 設定參數，包含過濾條件
        params = KNNSearchParams(
            k=10,
            filter={"term": {"category": "technical"}},
            min_score=0.7
        )
        
        # 執行搜尋
        results = await knn_service.knn_search(
            query_text="technical query",
            params=params,
            strategy=SearchStrategy.KNN_ONLY
        )
        
        # 驗證查詢包含過濾條件
        call_args = knn_service.client.search.call_args
        query_body = call_args[1]['body']
        
        assert 'filter' in query_body['query']['knn']['embedding']
        assert query_body['query']['knn']['embedding']['filter'] == {"term": {"category": "technical"}}
        assert query_body['min_score'] == 0.7
    
    @pytest.mark.asyncio
    async def test_hybrid_search(self, knn_service: KNNSearchService):
        """測試混合搜尋策略 (向量 + BM25)"""
        # 模擬帶高亮的回應
        knn_service.client.search.return_value = {
            "hits": {
                "total": {"value": 1},
                "hits": [
                    {
                        "_id": "1",
                        "_score": 0.9,
                        "_source": {
                            "doc_id": "doc1",
                            "title": "Hybrid Search Result",
                            "content": "Content with keywords",
                            "tags": ["hybrid"],
                            "category": "test",
                            "metadata": {}
                        },
                        "highlight": {
                            "content": ["Content with <em>keywords</em>"]
                        }
                    }
                ]
            }
        }
        
        # 執行混合搜尋
        params = KNNSearchParams(k=5)
        results = await knn_service.knn_search(
            query_text="keywords test",
            params=params,
            strategy=SearchStrategy.HYBRID
        )
        
        # 驗證結果
        assert len(results) == 1
        assert results[0].highlights == ["Content with <em>keywords</em>"]
        
        # 驗證查詢結構
        call_args = knn_service.client.search.call_args
        query_body = call_args[1]['body']
        
        # 檢查是否包含 bool 查詢
        assert 'bool' in query_body['query']
        assert 'should' in query_body['query']['bool']
        
        # 檢查是否同時包含 KNN 和文字搜尋
        should_clauses = query_body['query']['bool']['should']
        assert len(should_clauses) == 2
        assert 'knn' in should_clauses[0]
        assert 'multi_match' in should_clauses[1]
        
        # 檢查高亮設定
        assert 'highlight' in query_body
        assert 'content' in query_body['highlight']['fields']
    
    @pytest.mark.asyncio
    async def test_multi_vector_search(self, knn_service: KNNSearchService):
        """測試多向量搜尋策略"""
        # 模擬多次搜尋回應
        knn_service.client.search.side_effect = [
            # 第一個查詢變體的結果
            {
                "hits": {
                    "total": {"value": 2},
                    "hits": [
                        {
                            "_id": "1",
                            "_score": 0.9,
                            "_source": {
                                "doc_id": "doc1",
                                "title": "Multi Vector Result 1",
                                "content": "Content 1",
                                "tags": [],
                                "category": "test",
                                "metadata": {}
                            }
                        },
                        {
                            "_id": "2",
                            "_score": 0.8,
                            "_source": {
                                "doc_id": "doc2",
                                "title": "Multi Vector Result 2", 
                                "content": "Content 2",
                                "tags": [],
                                "category": "test",
                                "metadata": {}
                            }
                        }
                    ]
                }
            },
            # 第二個查詢變體的結果（包含重複的 doc1）
            {
                "hits": {
                    "total": {"value": 2},
                    "hits": [
                        {
                            "_id": "1",
                            "_score": 0.95,  # 更高的分數
                            "_source": {
                                "doc_id": "doc1",
                                "title": "Multi Vector Result 1",
                                "content": "Content 1",
                                "tags": [],
                                "category": "test",
                                "metadata": {}
                            }
                        },
                        {
                            "_id": "3",
                            "_score": 0.7,
                            "_source": {
                                "doc_id": "doc3",
                                "title": "Multi Vector Result 3",
                                "content": "Content 3",
                                "tags": [],
                                "category": "test",
                                "metadata": {}
                            }
                        }
                    ]
                }
            },
            # 其他查詢變體的結果
            {"hits": {"total": {"value": 0}, "hits": []}},
            {"hits": {"total": {"value": 0}, "hits": []}}
        ]
        
        # 執行多向量搜尋
        params = KNNSearchParams(k=3)
        results = await knn_service.knn_search(
            query_text="test query",
            params=params,
            strategy=SearchStrategy.MULTI_VECTOR
        )
        
        # 驗證結果
        assert len(results) == 3  # 應該返回去重後的結果
        
        # 驗證去重邏輯（doc1 應該保留分數較高的）
        doc1_result = next((r for r in results if r.doc_id == "doc1"), None)
        assert doc1_result is not None
        assert doc1_result.score == 0.95  # 應該保留較高的分數
        
        # 驗證多次調用 embeddings
        assert knn_service.embeddings.aembed_query.call_count >= 4  # 原查詢 + 查詢變體
    
    @pytest.mark.asyncio
    async def test_rerank_search(self, knn_service: KNNSearchService):
        """測試重新排序搜尋策略"""
        # 模擬初始搜尋結果（更多候選）
        knn_service.client.search.return_value = {
            "hits": {
                "total": {"value": 3},
                "hits": [
                    {
                        "_id": "1",
                        "_score": 0.8,
                        "_source": {
                            "doc_id": "doc1",
                            "title": "Document 1",
                            "content": "Test content with relevant keywords",
                            "tags": [],
                            "category": "test",
                            "metadata": {}
                        }
                    },
                    {
                        "_id": "2",
                        "_score": 0.85,
                        "_source": {
                            "doc_id": "doc2",
                            "title": "Document 2",
                            "content": "Another test document",
                            "tags": [],
                            "category": "test",
                            "metadata": {}
                        }
                    },
                    {
                        "_id": "3",
                        "_score": 0.75,
                        "_source": {
                            "doc_id": "doc3",
                            "title": "Document 3",
                            "content": "Test query keywords match",
                            "tags": [],
                            "category": "test",
                            "metadata": {}
                        }
                    }
                ]
            }
        }
        
        # 模擬餘弦相似度計算
        with patch.object(knn_service, '_cosine_similarity', side_effect=[0.9, 0.7, 0.95]):
            # 執行重新排序搜尋
            params = KNNSearchParams(k=2)
            results = await knn_service.knn_search(
                query_text="test query keywords",
                params=params,
                strategy=SearchStrategy.RERANK
            )
            
            # 驗證結果
            assert len(results) == 2  # 應該只返回 k 個結果
            
            # 驗證初始搜尋使用了更寬鬆的參數
            call_args = knn_service.client.search.call_args
            query_body = call_args[1]['body']
            assert query_body['size'] == 6  # k * 3
            
            # 驗證重新排序邏輯
            # 應該為每個結果生成新的 embedding
            assert knn_service.embeddings.aembed_query.call_count >= 4  # 1個查詢 + 3個文檔
    
    @pytest.mark.asyncio
    async def test_error_handling_invalid_strategy(self, knn_service: KNNSearchService):
        """測試無效策略的錯誤處理"""
        with pytest.raises(ValueError, match="不支援的搜尋策略"):
            await knn_service.knn_search(
                query_text="test",
                params=KNNSearchParams(),
                strategy="INVALID_STRATEGY"  # 無效的策略
            )
    
    @pytest.mark.asyncio
    async def test_error_handling_opensearch_error(self, knn_service: KNNSearchService):
        """測試 OpenSearch 錯誤處理"""
        # 模擬 OpenSearch 錯誤
        knn_service.client.search.side_effect = Exception("OpenSearch connection error")
        
        # 執行搜尋應該傳播錯誤
        with pytest.raises(Exception, match="OpenSearch connection error"):
            await knn_service.knn_search(
                query_text="test",
                params=KNNSearchParams(),
                strategy=SearchStrategy.KNN_ONLY
            )
    
    def test_cosine_similarity(self, knn_service: KNNSearchService):
        """測試餘弦相似度計算"""
        # 測試相同向量
        vec1 = [1.0, 0.0, 0.0]
        assert knn_service._cosine_similarity(vec1, vec1) == pytest.approx(1.0)
        
        # 測試正交向量
        vec2 = [0.0, 1.0, 0.0]
        assert knn_service._cosine_similarity(vec1, vec2) == pytest.approx(0.0)
        
        # 測試零向量
        vec3 = [0.0, 0.0, 0.0]
        assert knn_service._cosine_similarity(vec1, vec3) == 0.0
    
    def test_calculate_keyword_score(self, knn_service: KNNSearchService):
        """測試關鍵詞匹配分數計算"""
        # 完全匹配
        query = "test query keywords"
        content = "This is a test query with all keywords"
        score = knn_service._calculate_keyword_score(query, content)
        assert score == 1.0
        
        # 部分匹配
        content2 = "This is a test document"
        score2 = knn_service._calculate_keyword_score(query, content2)
        assert score2 == pytest.approx(1/3)
        
        # 無匹配
        content3 = "No matching words here"
        score3 = knn_service._calculate_keyword_score(query, content3)
        assert score3 == 0.0
        
        # 空查詢
        assert knn_service._calculate_keyword_score("", content) == 0.0
    
    def test_deduplicate_and_rerank(self, knn_service: KNNSearchService):
        """測試去重和重新排序邏輯"""
        # 建立測試結果（包含重複）
        results = [
            SearchResult("doc1", "Title 1", "Content 1", 0.9, {}),
            SearchResult("doc2", "Title 2", "Content 2", 0.8, {}),
            SearchResult("doc1", "Title 1", "Content 1", 0.95, {}),  # 重複，但分數更高
            SearchResult("doc3", "Title 3", "Content 3", 0.85, {})
        ]
        
        # 執行去重和重新排序
        unique_results = knn_service._deduplicate_and_rerank(results)
        
        # 驗證結果
        assert len(unique_results) == 3  # 去重後應該只有3個
        assert unique_results[0].doc_id == "doc1"
        assert unique_results[0].score == 0.95  # 保留較高分數
        assert unique_results[1].doc_id == "doc3"
        assert unique_results[2].doc_id == "doc2"
    
    def test_to_langchain_documents(self, knn_service: KNNSearchService):
        """測試轉換為 LangChain Document 格式"""
        # 建立測試結果
        results = [
            SearchResult(
                doc_id="doc1",
                title="Test Title",
                content="Test content",
                score=0.9,
                metadata={"author": "test_author", "date": "2024-01-01"}
            )
        ]
        
        # 轉換為 LangChain 格式
        documents = knn_service.to_langchain_documents(results)
        
        # 驗證結果
        assert len(documents) == 1
        doc = documents[0]
        assert doc.page_content == "Test content"
        assert doc.metadata["doc_id"] == "doc1"
        assert doc.metadata["title"] == "Test Title"
        assert doc.metadata["score"] == 0.9
        assert doc.metadata["author"] == "test_author"
        assert doc.metadata["date"] == "2024-01-01"
    
    @pytest.mark.asyncio
    async def test_explain_search(self, knn_service: KNNSearchService):
        """測試搜尋結果解釋功能"""
        # 模擬解釋查詢回應
        knn_service.client.search.return_value = {
            "hits": {
                "total": {"value": 1},
                "hits": [
                    {
                        "_id": "1",
                        "_score": 0.9,
                        "_source": {
                            "doc_id": "doc1",
                            "title": "Test Document"
                        },
                        "_explanation": {
                            "value": 0.9,
                            "description": "knn similarity",
                            "details": []
                        }
                    }
                ]
            }
        }
        
        # 執行解釋查詢
        explanation = await knn_service.explain_search("test query", "doc1")
        
        # 驗證結果
        assert explanation["doc_id"] == "doc1"
        assert explanation["title"] == "Test Document"
        assert explanation["score"] == 0.9
        assert "explanation" in explanation
        
        # 驗證查詢包含 explain 參數
        call_args = knn_service.client.search.call_args
        query_body = call_args[1]['body']
        assert query_body['explain'] is True
        assert query_body['query']['knn']['embedding']['filter']['term']['doc_id'] == "doc1"
    
    @pytest.mark.asyncio
    async def test_explain_search_not_found(self, knn_service: KNNSearchService):
        """測試解釋不存在文檔的搜尋結果"""
        # 模擬空結果
        knn_service.client.search.return_value = {
            "hits": {
                "total": {"value": 0},
                "hits": []
            }
        }
        
        # 執行解釋查詢
        explanation = await knn_service.explain_search("test query", "non_existent")
        
        # 驗證錯誤回應
        assert explanation == {"error": "Document not found"}


class TestSearchParams:
    """測試搜尋參數類別"""
    
    def test_default_params(self):
        """測試預設參數"""
        params = KNNSearchParams()
        assert params.k == 10
        assert params.num_candidates == 100
        assert params.boost == 1.0
        assert params.filter is None
        assert params.min_score is None
    
    def test_custom_params(self):
        """測試自定義參數"""
        params = KNNSearchParams(
            k=20,
            num_candidates=200,
            boost=2.0,
            filter={"term": {"category": "test"}},
            min_score=0.5
        )
        assert params.k == 20
        assert params.num_candidates == 200
        assert params.boost == 2.0
        assert params.filter == {"term": {"category": "test"}}
        assert params.min_score == 0.5