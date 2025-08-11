"""
KNN 搜尋服務單元測試
測試 knn_search_service.py 中各種搜尋策略的內部邏輯
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import List, Dict, Any
import numpy as np

from src.services.knn_search_service import (
    KNNSearchService, 
    SearchStrategy, 
    KNNSearchParams,
    SearchResult
)
from langchain_core.documents import Document


class TestKNNSearchService:
    """KNN 搜尋服務測試類"""
    
    @pytest.fixture
    def mock_opensearch_client(self):
        """模擬 OpenSearch 客戶端"""
        client = Mock()
        client.search = Mock(return_value={
            "hits": {
                "hits": [
                    {
                        "_id": "test-id-1",
                        "_score": 0.95,
                        "_source": {
                            "doc_id": "doc-1",
                            "title": "測試文件 1",
                            "content": "這是測試內容 1",
                            "tags": ["test", "doc"],
                            "category": "測試",
                            "metadata": {"author": "測試作者"}
                        }
                    },
                    {
                        "_id": "test-id-2",
                        "_score": 0.85,
                        "_source": {
                            "doc_id": "doc-2",
                            "title": "測試文件 2",
                            "content": "這是測試內容 2",
                            "tags": ["test"],
                            "category": "測試",
                            "metadata": {}
                        }
                    }
                ]
            }
        })
        return client
    
    @pytest.fixture
    def mock_embeddings(self):
        """模擬 Embeddings 模型"""
        embeddings = Mock()
        embeddings.aembed_query = AsyncMock(return_value=[0.1, 0.2, 0.3, 0.4, 0.5])
        return embeddings
    
    @pytest.fixture
    def search_service(self, mock_opensearch_client, mock_embeddings):
        """建立測試用的搜尋服務"""
        with patch('src.services.knn_search_service.OpenSearch', return_value=mock_opensearch_client):
            with patch('src.services.knn_search_service.GoogleGenerativeAIEmbeddings', return_value=mock_embeddings):
                service = KNNSearchService()
                return service
    
    @pytest.mark.asyncio
    async def test_knn_only_search(self, search_service, mock_opensearch_client):
        """測試純 KNN 向量搜尋"""
        # 設定參數
        params = KNNSearchParams(k=5, num_candidates=50, boost=1.5, min_score=0.7)
        query_text = "測試查詢"
        
        # 執行搜尋
        results = await search_service.knn_search(
            query_text=query_text,
            params=params,
            strategy=SearchStrategy.KNN_ONLY
        )
        
        # 驗證結果
        assert len(results) == 2
        assert results[0].doc_id == "doc-1"
        assert results[0].score == 0.95
        assert results[1].doc_id == "doc-2"
        
        # 驗證 OpenSearch 呼叫
        mock_opensearch_client.search.assert_called_once()
        call_args = mock_opensearch_client.search.call_args[1]
        
        assert call_args['index'] == search_service.index_name
        assert call_args['body']['size'] == params.k
        assert call_args['body']['query']['knn']['embedding']['k'] == params.k
        assert call_args['body']['query']['knn']['embedding']['boost'] == params.boost
        assert call_args['body']['min_score'] == params.min_score
    
    @pytest.mark.asyncio
    async def test_knn_search_with_filter(self, search_service, mock_opensearch_client):
        """測試帶過濾條件的 KNN 搜尋"""
        # 設定參數與過濾條件
        filter_condition = {"term": {"category": "技術文件"}}
        params = KNNSearchParams(k=10, filter=filter_condition)
        
        # 執行搜尋
        results = await search_service.knn_search(
            query_text="技術查詢",
            params=params,
            strategy=SearchStrategy.KNN_ONLY
        )
        
        # 驗證過濾條件被正確傳遞
        call_args = mock_opensearch_client.search.call_args[1]
        assert call_args['body']['query']['knn']['embedding']['filter'] == filter_condition
    
    @pytest.mark.asyncio
    async def test_hybrid_search(self, search_service, mock_opensearch_client):
        """測試混合搜尋策略 (向量 + BM25)"""
        # 模擬包含高亮的回應
        mock_opensearch_client.search.return_value = {
            "hits": {
                "hits": [
                    {
                        "_id": "test-id-1",
                        "_score": 0.95,
                        "_source": {
                            "doc_id": "doc-1",
                            "title": "測試文件",
                            "content": "完整的測試內容",
                            "tags": ["test"],
                            "category": "測試",
                            "metadata": {}
                        },
                        "highlight": {
                            "content": ["<em>測試</em>內容的高亮部分"]
                        }
                    }
                ]
            }
        }
        
        # 執行混合搜尋
        params = KNNSearchParams(k=5)
        results = await search_service.knn_search(
            query_text="測試查詢",
            params=params,
            strategy=SearchStrategy.HYBRID
        )
        
        # 驗證結果
        assert len(results) == 1
        assert results[0].highlights == ["<em>測試</em>內容的高亮部分"]
        
        # 驗證查詢結構
        call_args = mock_opensearch_client.search.call_args[1]
        query = call_args['body']['query']['bool']['should']
        
        # 應該包含 KNN 和 multi_match 查詢
        assert len(query) == 2
        assert 'knn' in query[0]
        assert 'multi_match' in query[1]
        assert query[1]['multi_match']['fields'] == ["title^2", "content", "tags"]
    
    @pytest.mark.asyncio
    async def test_multi_vector_search(self, search_service, mock_embeddings):
        """測試多向量搜尋策略"""
        # 模擬生成多個向量
        mock_embeddings.aembed_query = AsyncMock(side_effect=[
            [0.1, 0.2, 0.3],  # 原始查詢向量
            [0.2, 0.3, 0.4],  # 變體1向量
            [0.3, 0.4, 0.5],  # 變體2向量
            [0.4, 0.5, 0.6],  # 變體3向量
        ])
        
        # 執行多向量搜尋
        params = KNNSearchParams(k=3)
        results = await search_service.knn_search(
            query_text="系統錯誤分析",
            params=params,
            strategy=SearchStrategy.MULTI_VECTOR
        )
        
        # 驗證生成了多個查詢變體
        assert mock_embeddings.aembed_query.call_count >= 4  # 原查詢 + 至少3個變體
        
        # 驗證結果去重
        doc_ids = [r.doc_id for r in results]
        assert len(doc_ids) == len(set(doc_ids))  # 沒有重複
    
    @pytest.mark.asyncio
    async def test_rerank_search(self, search_service, mock_embeddings):
        """測試重新排序搜尋策略"""
        # 為重新排序準備多個 embedding 呼叫
        mock_embeddings.aembed_query = AsyncMock(side_effect=[
            [0.1, 0.2, 0.3, 0.4, 0.5],  # 查詢向量
            [0.15, 0.25, 0.35, 0.45, 0.55],  # 文件1內容向量
            [0.05, 0.15, 0.25, 0.35, 0.45],  # 文件2內容向量
        ])
        
        # 執行重新排序搜尋
        params = KNNSearchParams(k=2)
        results = await search_service.knn_search(
            query_text="測試查詢",
            params=params,
            strategy=SearchStrategy.RERANK
        )
        
        # 驗證結果
        assert len(results) <= params.k
        # 驗證分數已被重新計算（應該在0-1之間）
        for result in results:
            assert 0 <= result.score <= 1
    
    def test_parse_search_results(self, search_service):
        """測試搜尋結果解析"""
        # 模擬 OpenSearch 回應
        response = {
            "hits": {
                "hits": [
                    {
                        "_id": "test-id",
                        "_score": 0.9,
                        "_source": {
                            "doc_id": "doc-123",
                            "title": "測試標題",
                            "content": "測試內容",
                            "tags": ["tag1", "tag2"],
                            "category": "測試類別",
                            "metadata": {"key": "value"}
                        },
                        "highlight": {
                            "content": ["高亮內容"]
                        }
                    }
                ]
            }
        }
        
        # 解析結果
        results = search_service._parse_search_results(response, include_highlights=True)
        
        assert len(results) == 1
        result = results[0]
        assert result.doc_id == "doc-123"
        assert result.title == "測試標題"
        assert result.content == "測試內容"
        assert result.score == 0.9
        assert result.metadata["tags"] == ["tag1", "tag2"]
        assert result.metadata["category"] == "測試類別"
        assert result.metadata["key"] == "value"
        assert result.highlights == ["高亮內容"]
    
    def test_deduplicate_and_rerank(self, search_service):
        """測試去重和重新排序"""
        # 建立包含重複的結果
        results = [
            SearchResult("doc-1", "標題1", "內容1", 0.9, {}),
            SearchResult("doc-2", "標題2", "內容2", 0.8, {}),
            SearchResult("doc-1", "標題1", "內容1", 0.95, {}),  # 重複，但分數更高
            SearchResult("doc-3", "標題3", "內容3", 0.85, {}),
        ]
        
        # 執行去重和排序
        unique_results = search_service._deduplicate_and_rerank(results)
        
        # 驗證結果
        assert len(unique_results) == 3  # 去重後應該只有3個
        assert unique_results[0].doc_id == "doc-1"
        assert unique_results[0].score == 0.95  # 保留了較高的分數
        assert unique_results[1].doc_id == "doc-3"
        assert unique_results[2].doc_id == "doc-2"
    
    def test_cosine_similarity(self, search_service):
        """測試餘弦相似度計算"""
        vec1 = [1, 0, 0]
        vec2 = [0, 1, 0]
        vec3 = [1, 0, 0]
        
        # 正交向量
        similarity1 = search_service._cosine_similarity(vec1, vec2)
        assert similarity1 == 0.0
        
        # 相同向量
        similarity2 = search_service._cosine_similarity(vec1, vec3)
        assert similarity2 == 1.0
        
        # 一般情況
        vec4 = [0.5, 0.5, 0]
        similarity3 = search_service._cosine_similarity(vec1, vec4)
        assert 0 < similarity3 < 1
    
    def test_calculate_keyword_score(self, search_service):
        """測試關鍵詞匹配分數計算"""
        query = "系統 錯誤 分析"
        
        # 完全匹配
        content1 = "這是一個系統錯誤分析的案例"
        score1 = search_service._calculate_keyword_score(query, content1)
        assert score1 == 1.0
        
        # 部分匹配
        content2 = "系統運行正常，沒有錯誤"
        score2 = search_service._calculate_keyword_score(query, content2)
        assert 0 < score2 < 1
        
        # 無匹配
        content3 = "完全不相關的內容"
        score3 = search_service._calculate_keyword_score(query, content3)
        assert score3 == 0.0
    
    def test_to_langchain_documents(self, search_service):
        """測試轉換為 LangChain Document 格式"""
        # 建立搜尋結果
        results = [
            SearchResult(
                doc_id="doc-1",
                title="標題1",
                content="內容1",
                score=0.9,
                metadata={"author": "作者1", "date": "2024-01-01"}
            ),
            SearchResult(
                doc_id="doc-2",
                title="標題2",
                content="內容2",
                score=0.8,
                metadata={"author": "作者2"}
            )
        ]
        
        # 轉換為 LangChain Documents
        documents = search_service.to_langchain_documents(results)
        
        assert len(documents) == 2
        assert isinstance(documents[0], Document)
        assert documents[0].page_content == "內容1"
        assert documents[0].metadata["doc_id"] == "doc-1"
        assert documents[0].metadata["title"] == "標題1"
        assert documents[0].metadata["score"] == 0.9
        assert documents[0].metadata["author"] == "作者1"
    
    @pytest.mark.asyncio
    async def test_error_handling_invalid_strategy(self, search_service):
        """測試無效策略的錯誤處理"""
        with pytest.raises(ValueError, match="不支援的搜尋策略"):
            await search_service.knn_search(
                query_text="測試",
                params=KNNSearchParams(),
                strategy="invalid_strategy"  # 無效的策略
            )
    
    @pytest.mark.asyncio
    async def test_explain_search(self, search_service, mock_opensearch_client):
        """測試搜尋結果解釋功能"""
        # 模擬解釋查詢的回應
        mock_opensearch_client.search.return_value = {
            "hits": {
                "hits": [
                    {
                        "_id": "test-id",
                        "_score": 0.9,
                        "_source": {
                            "doc_id": "doc-123",
                            "title": "測試文件"
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
        explanation = await search_service.explain_search("測試查詢", "doc-123")
        
        assert explanation["doc_id"] == "doc-123"
        assert explanation["score"] == 0.9
        assert "explanation" in explanation
        assert explanation["explanation"]["description"] == "knn similarity"


class TestKNNSearchParams:
    """測試搜尋參數類"""
    
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
        filter_condition = {"term": {"category": "test"}}
        params = KNNSearchParams(
            k=20,
            num_candidates=200,
            boost=2.0,
            filter=filter_condition,
            min_score=0.5
        )
        
        assert params.k == 20
        assert params.num_candidates == 200
        assert params.boost == 2.0
        assert params.filter == filter_condition
        assert params.min_score == 0.5


class TestSearchResult:
    """測試搜尋結果類"""
    
    def test_search_result_creation(self):
        """測試搜尋結果建立"""
        result = SearchResult(
            doc_id="test-123",
            title="測試標題",
            content="測試內容",
            score=0.95,
            metadata={"key": "value"},
            highlights=["高亮1", "高亮2"]
        )
        
        assert result.doc_id == "test-123"
        assert result.title == "測試標題"
        assert result.content == "測試內容"
        assert result.score == 0.95
        assert result.metadata == {"key": "value"}
        assert result.highlights == ["高亮1", "高亮2"]