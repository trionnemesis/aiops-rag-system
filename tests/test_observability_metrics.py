"""
Observability Metrics 模組單元測試
測試 Prometheus 指標的註冊、更新和收集功能
"""

import pytest
from unittest.mock import patch, MagicMock, Mock
import time
from prometheus_client import CollectorRegistry, REGISTRY

from app.observability.metrics import (
    # 指標物件
    api_request_counter,
    api_request_duration,
    node_execution_time,
    node_error_counter,
    llm_token_counter,
    llm_request_duration,
    llm_error_counter,
    retriever_docs_counter,
    retriever_relevance_histogram,
    retriever_duration,
    system_info,
    active_requests,
    vector_store_documents,
    cache_hit_rate,
    
    # 功能函數
    track_api_request,
    track_node_metrics,
    track_llm_usage,
    track_retrieval_metrics,
    update_cache_metrics,
    start_metrics_server,
    get_metrics_snapshot,
    registry
)


class TestMetricsDefinitions:
    """測試指標定義"""
    
    def test_api_metrics_exist(self):
        """測試 API 層級指標存在且正確定義"""
        assert api_request_counter._name == "rag_api_requests_total"
        assert api_request_counter._labelnames == ('endpoint', 'method', 'status')
        
        assert api_request_duration._name == "rag_api_request_duration_seconds"
        assert api_request_duration._labelnames == ('endpoint', 'method')
    
    def test_node_metrics_exist(self):
        """測試節點層級指標存在且正確定義"""
        assert node_execution_time._name == "langgraph_node_execution_seconds"
        assert node_execution_time._labelnames == ('node_name',)
        
        assert node_error_counter._name == "langgraph_node_errors_total"
        assert node_error_counter._labelnames == ('node_name', 'error_type')
    
    def test_llm_metrics_exist(self):
        """測試 LLM 相關指標存在且正確定義"""
        assert llm_token_counter._name == "llm_tokens_total"
        assert llm_token_counter._labelnames == ('model', 'token_type')
        
        assert llm_request_duration._name == "llm_request_duration_seconds"
        assert llm_request_duration._labelnames == ('model', 'operation')
        
        assert llm_error_counter._name == "llm_errors_total"
        assert llm_error_counter._labelnames == ('model', 'error_type')
    
    def test_retriever_metrics_exist(self):
        """測試檢索器相關指標存在且正確定義"""
        assert retriever_docs_counter._name == "retriever_documents_retrieved"
        assert retriever_docs_counter._labelnames == ('retriever_type',)
        
        assert retriever_relevance_histogram._name == "retriever_relevance_score"
        assert retriever_relevance_histogram._labelnames == ('retriever_type',)
        
        assert retriever_duration._name == "retriever_duration_seconds"
        assert retriever_duration._labelnames == ('retriever_type',)
    
    def test_system_metrics_exist(self):
        """測試系統健康指標存在且正確定義"""
        assert system_info._name == "rag_system_info"
        assert active_requests._name == "rag_active_requests"
        assert vector_store_documents._name == "vector_store_documents_total"
        assert cache_hit_rate._name == "rag_cache_hit_rate"


class TestTrackApiRequest:
    """測試 API 請求追蹤裝飾器"""
    
    @patch('app.observability.metrics.api_request_counter')
    @patch('app.observability.metrics.api_request_duration')
    @patch('app.observability.metrics.active_requests')
    def test_track_api_request_success(self, mock_active, mock_duration, mock_counter):
        """測試成功的 API 請求追蹤"""
        # 建立測試函數
        @track_api_request("/test", "POST")
        async def test_endpoint():
            return {"status": "success"}
        
        # 執行函數
        import asyncio
        result = asyncio.run(test_endpoint())
        
        # 驗證結果
        assert result == {"status": "success"}
        
        # 驗證指標更新
        mock_counter.labels.assert_called_with(endpoint="/test", method="POST", status="success")
        mock_counter.labels().inc.assert_called_once()
        
        mock_duration.labels.assert_called_with(endpoint="/test", method="POST")
        mock_duration.labels().observe.assert_called_once()
        
        # 驗證 active_requests 增減
        assert mock_active.inc.call_count == 1
        assert mock_active.dec.call_count == 1
    
    @patch('app.observability.metrics.api_request_counter')
    @patch('app.observability.metrics.api_request_duration')
    @patch('app.observability.metrics.active_requests')
    def test_track_api_request_error(self, mock_active, mock_duration, mock_counter):
        """測試失敗的 API 請求追蹤"""
        # 建立會失敗的測試函數
        @track_api_request("/test", "GET")
        async def failing_endpoint():
            raise ValueError("測試錯誤")
        
        # 執行函數應該拋出異常
        import asyncio
        with pytest.raises(ValueError, match="測試錯誤"):
            asyncio.run(failing_endpoint())
        
        # 驗證指標更新
        mock_counter.labels.assert_called_with(endpoint="/test", method="GET", status="error")
        mock_counter.labels().inc.assert_called_once()
        
        # 即使失敗也應該記錄持續時間
        mock_duration.labels.assert_called_with(endpoint="/test", method="GET")
        mock_duration.labels().observe.assert_called_once()
        
        # 驗證 active_requests 增減
        assert mock_active.inc.call_count == 1
        assert mock_active.dec.call_count == 1


class TestTrackNodeMetrics:
    """測試節點指標追蹤裝飾器"""
    
    @patch('app.observability.metrics.node_execution_time')
    @patch('app.observability.metrics.node_error_counter')
    def test_track_node_metrics_success(self, mock_error_counter, mock_execution_time):
        """測試成功的節點執行追蹤"""
        # 建立測試函數
        @track_node_metrics("test_node")
        def test_node(state):
            state["processed"] = True
            return state
        
        # 執行函數
        initial_state = {"data": "test"}
        result = test_node(initial_state)
        
        # 驗證結果
        assert result["processed"] is True
        assert result["data"] == "test"
        
        # 驗證執行時間被記錄
        mock_execution_time.labels.assert_called_with(node_name="test_node")
        mock_execution_time.labels().observe.assert_called_once()
        
        # 驗證沒有錯誤計數
        mock_error_counter.labels.assert_not_called()
    
    @patch('app.observability.metrics.node_execution_time')
    @patch('app.observability.metrics.node_error_counter')
    def test_track_node_metrics_error(self, mock_error_counter, mock_execution_time):
        """測試失敗的節點執行追蹤"""
        # 建立會失敗的測試函數
        @track_node_metrics("error_node")
        def failing_node(state):
            raise RuntimeError("節點執行失敗")
        
        # 執行函數應該拋出異常
        with pytest.raises(RuntimeError, match="節點執行失敗"):
            failing_node({"data": "test"})
        
        # 驗證執行時間被記錄（即使失敗）
        mock_execution_time.labels.assert_called_with(node_name="error_node")
        mock_execution_time.labels().observe.assert_called_once()
        
        # 驗證錯誤被計數
        mock_error_counter.labels.assert_called_with(
            node_name="error_node",
            error_type="RuntimeError"
        )
        mock_error_counter.labels().inc.assert_called_once()


class TestTrackLLMUsage:
    """測試 LLM 使用追蹤"""
    
    @patch('app.observability.metrics.llm_token_counter')
    @patch('app.observability.metrics.llm_request_duration')
    def test_track_llm_usage_basic(self, mock_duration, mock_token_counter):
        """測試基本的 LLM 使用追蹤"""
        track_llm_usage(
            model="gpt-4",
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            duration=2.5
        )
        
        # 驗證 token 計數
        expected_calls = [
            (('gpt-4', 'prompt'), 100),
            (('gpt-4', 'completion'), 50),
            (('gpt-4', 'total'), 150)
        ]
        
        for (model, token_type), count in expected_calls:
            mock_token_counter.labels.assert_any_call(model=model, token_type=token_type)
        
        # 驗證持續時間記錄
        mock_duration.labels.assert_called_with(model="gpt-4", operation="inference")
        mock_duration.labels().observe.assert_called_with(2.5)
    
    @patch('app.observability.metrics.llm_error_counter')
    def test_track_llm_usage_error(self, mock_error_counter):
        """測試 LLM 錯誤追蹤"""
        track_llm_usage(
            model="gpt-3.5-turbo",
            error_type="RateLimitError"
        )
        
        # 驗證錯誤計數
        mock_error_counter.labels.assert_called_with(
            model="gpt-3.5-turbo",
            error_type="RateLimitError"
        )
        mock_error_counter.labels().inc.assert_called_once()


class TestTrackRetrievalMetrics:
    """測試檢索指標追蹤"""
    
    @patch('app.observability.metrics.retriever_docs_counter')
    @patch('app.observability.metrics.retriever_relevance_histogram')
    @patch('app.observability.metrics.retriever_duration')
    def test_track_retrieval_metrics(self, mock_duration, mock_relevance, mock_docs):
        """測試檢索指標追蹤"""
        track_retrieval_metrics(
            retriever_type="vector_store",
            num_docs=5,
            relevance_scores=[0.9, 0.85, 0.8, 0.75, 0.7],
            duration=0.5
        )
        
        # 驗證文件數量
        mock_docs.labels.assert_called_with(retriever_type="vector_store")
        mock_docs.labels().observe.assert_called_with(5)
        
        # 驗證相關性分數
        mock_relevance.labels.assert_called_with(retriever_type="vector_store")
        assert mock_relevance.labels().observe.call_count == 5
        
        # 驗證持續時間
        mock_duration.labels.assert_called_with(retriever_type="vector_store")
        mock_duration.labels().observe.assert_called_with(0.5)


class TestUpdateCacheMetrics:
    """測試快取指標更新"""
    
    @patch('app.observability.metrics.cache_hit_rate')
    def test_update_cache_metrics(self, mock_cache_hit_rate):
        """測試快取命中率更新"""
        update_cache_metrics(
            cache_name="embedding_cache",
            hits=80,
            misses=20
        )
        
        # 驗證命中率計算正確
        mock_cache_hit_rate.labels.assert_called_with(cache_name="embedding_cache")
        mock_cache_hit_rate.labels().set.assert_called_with(0.8)  # 80/(80+20)
    
    @patch('app.observability.metrics.cache_hit_rate')
    def test_update_cache_metrics_zero_total(self, mock_cache_hit_rate):
        """測試總數為零時的快取指標"""
        update_cache_metrics(
            cache_name="empty_cache",
            hits=0,
            misses=0
        )
        
        # 驗證命中率為 0
        mock_cache_hit_rate.labels.assert_called_with(cache_name="empty_cache")
        mock_cache_hit_rate.labels().set.assert_called_with(0.0)


class TestMetricsServer:
    """測試指標伺服器"""
    
    @patch('app.observability.metrics.start_http_server')
    def test_start_metrics_server(self, mock_start_server):
        """測試啟動指標伺服器"""
        start_metrics_server(port=9090)
        
        # 驗證伺服器啟動
        mock_start_server.assert_called_once_with(9090, registry=registry)


class TestGetMetricsSnapshot:
    """測試獲取指標快照"""
    
    @patch('app.observability.metrics.generate_latest')
    def test_get_metrics_snapshot(self, mock_generate_latest):
        """測試獲取指標快照"""
        # 模擬 Prometheus 格式的輸出
        mock_generate_latest.return_value = b"""
# HELP rag_api_requests_total Total number of RAG API requests
# TYPE rag_api_requests_total counter
rag_api_requests_total{endpoint="/query",method="POST",status="success"} 100.0
"""
        
        snapshot = get_metrics_snapshot()
        
        # 驗證返回的是字串格式
        assert isinstance(snapshot, str)
        assert "rag_api_requests_total" in snapshot
        
        # 驗證使用了正確的 registry
        mock_generate_latest.assert_called_once_with(registry)


class TestMetricsIntegration:
    """測試指標的整合使用"""
    
    def test_custom_registry_isolation(self):
        """測試自定義註冊表的隔離性"""
        # 驗證使用的是自定義 registry 而非默認的
        assert registry != REGISTRY
        assert isinstance(registry, CollectorRegistry)
    
    @patch('app.observability.metrics.time.time')
    def test_duration_measurement(self, mock_time):
        """測試持續時間測量的準確性"""
        # 模擬時間流逝
        mock_time.side_effect = [100.0, 102.5]  # 2.5 秒
        
        @track_api_request("/test", "GET")
        async def timed_function():
            # 模擬一些處理
            return "result"
        
        # 使用 patch 來捕獲 observe 呼叫
        with patch('app.observability.metrics.api_request_duration') as mock_duration:
            import asyncio
            asyncio.run(timed_function())
            
            # 驗證記錄的時間是 2.5 秒
            mock_duration.labels().observe.assert_called_with(2.5)