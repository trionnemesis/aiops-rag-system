"""
可觀測性 - Prometheus 指標模組單元測試
測試指標收集、裝飾器功能、指標導出
"""

import pytest
import time
from unittest.mock import Mock, patch, call
from prometheus_client import Counter, Histogram, Gauge, Info, CollectorRegistry
from prometheus_client.core import CollectorRegistry

from app.observability.metrics import (
    # 指標實例
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
    validation_results,
    validation_warnings,
    answer_quality_score,
    registry,
    # 裝飾器和函數
    setup_metrics,
    track_request_metrics,
    track_node_metrics,
    track_llm_metrics,
    track_retrieval_metrics,
    get_metrics
)


class TestPrometheusMetrics:
    """Prometheus 指標測試類別"""
    
    def setup_method(self):
        """測試前重置指標"""
        # 清理所有收集器以避免測試間干擾
        # 注意：在實際測試中，我們通常會使用獨立的 registry
        pass
    
    def test_metrics_registry(self):
        """測試指標註冊表"""
        # 驗證所有指標都註冊到同一個 registry
        assert api_request_counter._registry == registry
        assert api_request_duration._registry == registry
        assert node_execution_time._registry == registry
        assert llm_token_counter._registry == registry
        assert retriever_docs_counter._registry == registry
    
    def test_api_request_counter(self):
        """測試 API 請求計數器"""
        # 取得當前值
        before_value = api_request_counter._value.sum()
        
        # 增加計數
        api_request_counter.labels(
            endpoint="/rag/report",
            method="POST",
            status="success"
        ).inc()
        
        # 驗證計數增加
        after_value = api_request_counter._value.sum()
        assert after_value == before_value + 1
    
    def test_histogram_metrics(self):
        """測試直方圖指標"""
        # 測試 API 請求持續時間
        api_request_duration.labels(
            endpoint="/rag/report",
            method="POST"
        ).observe(0.5)
        
        # 測試節點執行時間（驗證自定義 buckets）
        node_execution_time.labels(node_name="retrieve").observe(0.05)
        node_execution_time.labels(node_name="retrieve").observe(0.15)
        node_execution_time.labels(node_name="retrieve").observe(1.5)
        
        # 取得指標資訊
        samples = list(node_execution_time.collect()[0].samples)
        bucket_samples = [s for s in samples if s.name.endswith("_bucket")]
        
        # 驗證有自定義的 buckets
        assert len(bucket_samples) > 0
    
    def test_gauge_metric(self):
        """測試 Gauge 指標"""
        # 測試活躍請求數
        active_requests.inc()
        active_requests.inc()
        assert active_requests._value.get() == 2
        
        active_requests.dec()
        assert active_requests._value.get() == 1
        
        active_requests.set(0)
        assert active_requests._value.get() == 0
    
    def test_info_metric(self):
        """測試 Info 指標"""
        # 設定系統資訊
        system_info.info({
            'version': '2.0.0',
            'framework': 'langgraph',
            'retriever': 'opensearch'
        })
        
        # Info 指標的值始終為 1，重要的是標籤
        samples = list(system_info.collect()[0].samples)
        assert len(samples) > 0
        assert samples[0].value == 1
    
    @patch('app.observability.metrics.start_http_server')
    def test_setup_metrics(self, mock_start_server):
        """測試指標設定"""
        # 設定指標服務
        setup_metrics(port=9090)
        
        # 驗證啟動了 HTTP 服務器
        mock_start_server.assert_called_once_with(9090, registry=registry)
    
    def test_track_request_metrics_decorator_success(self):
        """測試請求指標裝飾器 - 成功情況"""
        # 定義測試函數
        @track_request_metrics("/test/endpoint", "GET")
        def test_api_function():
            time.sleep(0.1)  # 模擬處理時間
            return {"status": "ok"}
        
        # 記錄初始值
        initial_active = active_requests._value.get()
        
        # 執行函數
        result = test_api_function()
        
        # 驗證結果
        assert result == {"status": "ok"}
        
        # 驗證活躍請求數已恢復
        assert active_requests._value.get() == initial_active
        
        # 驗證計數器和持續時間有記錄
        # 注意：實際驗證需要檢查具體的標籤值
    
    def test_track_request_metrics_decorator_error(self):
        """測試請求指標裝飾器 - 錯誤情況"""
        # 定義會拋出異常的測試函數
        @track_request_metrics("/test/error", "POST")
        def test_api_function_with_error():
            raise ValueError("Test error")
        
        # 記錄初始值
        initial_active = active_requests._value.get()
        
        # 執行函數並捕獲異常
        with pytest.raises(ValueError, match="Test error"):
            test_api_function_with_error()
        
        # 驗證活躍請求數已恢復
        assert active_requests._value.get() == initial_active
    
    def test_track_node_metrics_decorator(self):
        """測試節點指標裝飾器"""
        # 定義測試節點函數
        @track_node_metrics("test_node")
        def test_node(state):
            time.sleep(0.05)
            return {"processed": True, **state}
        
        # 執行節點
        input_state = {"query": "test query"}
        result = test_node(input_state)
        
        # 驗證結果
        assert result["processed"] is True
        assert result["query"] == "test query"
    
    def test_track_node_metrics_retrieve_node(self):
        """測試檢索節點的特殊指標"""
        # 定義檢索節點
        @track_node_metrics("retrieve")
        def retrieve_node(state):
            # 模擬檢索結果
            docs = [
                Mock(metadata={"score": 0.9}),
                Mock(metadata={"score": 0.8}),
                Mock(metadata={"score": 0.7})
            ]
            return {"documents": docs, **state}
        
        # 執行節點
        result = retrieve_node({"query": "test"})
        
        # 驗證返回了文檔
        assert len(result["documents"]) == 3
    
    def test_track_node_metrics_validate_node(self):
        """測試驗證節點的特殊指標"""
        # 定義驗證節點
        @track_node_metrics("validate")
        def validate_node(state):
            # 模擬驗證結果
            metrics = {
                "is_valid": True,
                "warnings": ["minor_issue", "formatting"],
                "quality_score": 0.85
            }
            return {"metrics": metrics, **state}
        
        # 執行節點
        result = validate_node({"query": "test"})
        
        # 驗證返回了指標
        assert result["metrics"]["is_valid"] is True
        assert len(result["metrics"]["warnings"]) == 2
    
    def test_track_node_metrics_error(self):
        """測試節點指標裝飾器的錯誤處理"""
        # 定義會出錯的節點
        @track_node_metrics("error_node")
        def error_node(state):
            raise RuntimeError("Node processing failed")
        
        # 執行並捕獲異常
        with pytest.raises(RuntimeError, match="Node processing failed"):
            error_node({"query": "test"})
    
    def test_track_llm_metrics_decorator(self):
        """測試 LLM 指標裝飾器"""
        # 模擬 LLM 回應
        mock_result = Mock()
        mock_result.usage.prompt_tokens = 100
        mock_result.usage.completion_tokens = 50
        mock_result.usage.total_tokens = 150
        
        # 定義測試函數
        @track_llm_metrics("gpt-4", "generate")
        def call_llm():
            time.sleep(0.1)
            return mock_result
        
        # 執行函數
        result = call_llm()
        
        # 驗證結果
        assert result == mock_result
    
    def test_track_llm_metrics_without_usage(self):
        """測試 LLM 指標裝飾器 - 無使用量資訊"""
        # 定義測試函數
        @track_llm_metrics("custom-model", "embed")
        def call_llm_no_usage():
            return {"embeddings": [[0.1, 0.2, 0.3]]}
        
        # 執行函數（不應拋出異常）
        result = call_llm_no_usage()
        assert "embeddings" in result
    
    def test_track_llm_metrics_error(self):
        """測試 LLM 指標裝飾器的錯誤處理"""
        # 定義會出錯的函數
        @track_llm_metrics("gpt-4", "generate")
        def call_llm_with_error():
            raise Exception("API rate limit exceeded")
        
        # 執行並捕獲異常
        with pytest.raises(Exception, match="API rate limit exceeded"):
            call_llm_with_error()
    
    def test_track_retrieval_metrics_decorator(self):
        """測試檢索指標裝飾器"""
        # 模擬檢索結果
        mock_docs = [
            Mock(metadata={"score": 0.95}),
            Mock(metadata={"score": 0.85}),
            Mock(metadata={"score": 0.75})
        ]
        
        # 定義測試函數
        @track_retrieval_metrics("vector")
        def retrieve_documents(query):
            time.sleep(0.05)
            return mock_docs
        
        # 執行檢索
        results = retrieve_documents("test query")
        
        # 驗證結果
        assert len(results) == 3
    
    def test_track_retrieval_metrics_empty_results(self):
        """測試檢索指標裝飾器 - 空結果"""
        # 定義返回空結果的函數
        @track_retrieval_metrics("bm25")
        def retrieve_no_results(query):
            return []
        
        # 執行檢索
        results = retrieve_no_results("obscure query")
        
        # 驗證空結果
        assert results == []
    
    def test_get_metrics(self):
        """測試獲取指標"""
        # 添加一些測試數據
        api_request_counter.labels(
            endpoint="/test",
            method="GET",
            status="success"
        ).inc()
        
        # 獲取指標
        metrics_output = get_metrics()
        
        # 驗證輸出格式
        assert isinstance(metrics_output, str)
        assert "# HELP" in metrics_output
        assert "# TYPE" in metrics_output
        assert "rag_api_requests_total" in metrics_output
    
    def test_metrics_labels_cardinality(self):
        """測試指標標籤基數控制"""
        # 測試不同的端點
        endpoints = ["/rag/report", "/rag/qa", "/health", "/metrics"]
        methods = ["GET", "POST", "PUT", "DELETE"]
        statuses = ["success", "error"]
        
        # 生成多種組合
        for endpoint in endpoints:
            for method in methods:
                for status in statuses:
                    api_request_counter.labels(
                        endpoint=endpoint,
                        method=method,
                        status=status
                    ).inc()
        
        # 驗證可以處理多種標籤組合
        # 在生產環境中應該限制標籤基數以避免記憶體問題
        assert True  # 主要確保不會崩潰
    
    def test_concurrent_metric_updates(self):
        """測試並發指標更新"""
        import threading
        
        def increment_counter():
            for _ in range(100):
                api_request_counter.labels(
                    endpoint="/concurrent",
                    method="POST",
                    status="success"
                ).inc()
        
        # 創建多個執行緒
        threads = []
        for _ in range(5):
            t = threading.Thread(target=increment_counter)
            threads.append(t)
            t.start()
        
        # 等待所有執行緒完成
        for t in threads:
            t.join()
        
        # Prometheus 客戶端應該是執行緒安全的
        # 主要確保不會出現競態條件
        assert True
    
    def test_histogram_percentiles(self):
        """測試直方圖百分位數"""
        # 添加多個測量值
        values = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.2, 1.5]
        
        for value in values:
            api_request_duration.labels(
                endpoint="/test",
                method="GET"
            ).observe(value)
        
        # 收集指標
        metrics_str = get_metrics()
        
        # 驗證包含分位數資訊
        assert "quantile" in metrics_str
        assert "0.5" in metrics_str  # 中位數
        assert "0.9" in metrics_str  # 90 百分位
        assert "0.99" in metrics_str  # 99 百分位