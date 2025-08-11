"""
可觀測性 - 分散式追蹤模組單元測試
測試 OpenTelemetry 追蹤配置、Span 生成、裝飾器功能
"""

import pytest
import time
from unittest.mock import Mock, patch, call, MagicMock
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.resources import Resource

from app.observability.tracing import (
    setup_tracing,
    get_tracer,
    trace_node,
    trace_llm_call,
    trace_retrieval,
    tracer
)


class TestDistributedTracing:
    """分散式追蹤測試類別"""
    
    def setup_method(self):
        """測試前重置 tracer"""
        # 重置全局 tracer
        import app.observability.tracing
        app.observability.tracing.tracer = None
    
    @patch('app.observability.tracing.trace')
    @patch('app.observability.tracing.TracerProvider')
    @patch('app.observability.tracing.FastAPIInstrumentor')
    @patch('app.observability.tracing.HTTPXClientInstrumentor')
    def test_setup_tracing_basic(self, mock_httpx_inst, mock_fastapi_inst, 
                                mock_tracer_provider, mock_trace):
        """測試基本追蹤設定"""
        # 設定模擬
        mock_provider_instance = Mock()
        mock_tracer_provider.return_value = mock_provider_instance
        mock_tracer_instance = Mock()
        mock_trace.get_tracer.return_value = mock_tracer_instance
        
        # 執行設定
        setup_tracing(
            service_name="test-service",
            service_version="1.0.0",
            console_export=True
        )
        
        # 驗證 TracerProvider 設定
        mock_tracer_provider.assert_called_once()
        call_args = mock_tracer_provider.call_args[1]
        assert isinstance(call_args['resource'], Resource)
        
        # 驗證設定了 tracer provider
        mock_trace.set_tracer_provider.assert_called_once_with(mock_provider_instance)
        
        # 驗證獲取了 tracer
        mock_trace.get_tracer.assert_called_once_with("test-service", "1.0.0")
        
        # 驗證自動儀表化
        mock_fastapi_inst.instrument.assert_called_once()
        mock_httpx_inst.instrument.assert_called_once()
    
    @patch('app.observability.tracing.trace')
    @patch('app.observability.tracing.TracerProvider')
    @patch('app.observability.tracing.JaegerExporter')
    @patch('app.observability.tracing.BatchSpanProcessor')
    def test_setup_tracing_with_jaeger(self, mock_batch_processor, mock_jaeger_exporter,
                                      mock_tracer_provider, mock_trace):
        """測試使用 Jaeger 的追蹤設定"""
        # 設定模擬
        mock_provider_instance = Mock()
        mock_tracer_provider.return_value = mock_provider_instance
        mock_exporter_instance = Mock()
        mock_jaeger_exporter.return_value = mock_exporter_instance
        mock_processor_instance = Mock()
        mock_batch_processor.return_value = mock_processor_instance
        
        # 執行設定
        setup_tracing(
            service_name="test-service",
            jaeger_endpoint="localhost:6831"
        )
        
        # 驗證 Jaeger exporter 設定
        mock_jaeger_exporter.assert_called_once_with(
            agent_host_name="localhost",
            agent_port=6831
        )
        
        # 驗證添加了 span processor
        mock_batch_processor.assert_called_once_with(mock_exporter_instance)
        mock_provider_instance.add_span_processor.assert_called()
    
    @patch('app.observability.tracing.trace')
    @patch('app.observability.tracing.TracerProvider')
    @patch('app.observability.tracing.OTLPSpanExporter')
    @patch('app.observability.tracing.BatchSpanProcessor')
    def test_setup_tracing_with_otlp(self, mock_batch_processor, mock_otlp_exporter,
                                    mock_tracer_provider, mock_trace):
        """測試使用 OTLP 的追蹤設定"""
        # 設定模擬
        mock_provider_instance = Mock()
        mock_tracer_provider.return_value = mock_provider_instance
        mock_exporter_instance = Mock()
        mock_otlp_exporter.return_value = mock_exporter_instance
        mock_processor_instance = Mock()
        mock_batch_processor.return_value = mock_processor_instance
        
        # 執行設定
        setup_tracing(
            service_name="test-service",
            otlp_endpoint="localhost:4317"
        )
        
        # 驗證 OTLP exporter 設定
        mock_otlp_exporter.assert_called_once_with(
            endpoint="localhost:4317",
            insecure=True
        )
        
        # 驗證添加了 span processor
        mock_batch_processor.assert_called_once_with(mock_exporter_instance)
        mock_provider_instance.add_span_processor.assert_called()
    
    def test_get_tracer_not_initialized(self):
        """測試未初始化時獲取 tracer"""
        with pytest.raises(RuntimeError, match="Tracer not initialized"):
            get_tracer()
    
    @patch('app.observability.tracing.trace')
    def test_get_tracer_initialized(self, mock_trace):
        """測試已初始化時獲取 tracer"""
        # 設定全局 tracer
        import app.observability.tracing
        mock_tracer = Mock()
        app.observability.tracing.tracer = mock_tracer
        
        # 獲取 tracer
        result = get_tracer()
        
        # 驗證返回了正確的 tracer
        assert result == mock_tracer
    
    def test_trace_node_decorator_success(self):
        """測試節點追蹤裝飾器 - 成功情況"""
        # 設定模擬 tracer 和 span
        mock_tracer = Mock()
        mock_span = MagicMock()
        mock_tracer.start_as_current_span.return_value.__enter__.return_value = mock_span
        
        # 設定全局 tracer
        import app.observability.tracing
        app.observability.tracing.tracer = mock_tracer
        
        # 定義測試函數
        @trace_node("test_node")
        def test_function(state):
            time.sleep(0.01)
            return {"result": "success", **state}
        
        # 執行函數
        input_state = {"request_id": "req-123", "query": "test query"}
        result = test_function(input_state)
        
        # 驗證結果
        assert result["result"] == "success"
        assert result["query"] == "test query"
        
        # 驗證 span 建立
        mock_tracer.start_as_current_span.assert_called_once_with(
            "langgraph.node.test_node",
            attributes={
                "node.name": "test_node",
                "request.id": "req-123",
                "query.text": "test query",
                "query.length": 10
            }
        )
        
        # 驗證設定了執行時間屬性
        mock_span.set_attribute.assert_any_call("node.execution_time_ms", pytest.approx(10, abs=20))
        
        # 驗證設定了成功狀態
        mock_span.set_status.assert_called_once()
        status_call = mock_span.set_status.call_args[0][0]
        assert status_call.status_code == StatusCode.OK
    
    def test_trace_node_decorator_error(self):
        """測試節點追蹤裝飾器 - 錯誤情況"""
        # 設定模擬 tracer 和 span
        mock_tracer = Mock()
        mock_span = MagicMock()
        mock_tracer.start_as_current_span.return_value.__enter__.return_value = mock_span
        
        # 設定全局 tracer
        import app.observability.tracing
        app.observability.tracing.tracer = mock_tracer
        
        # 定義會出錯的測試函數
        @trace_node("error_node")
        def test_function_with_error(state):
            raise ValueError("Test error")
        
        # 執行函數並捕獲異常
        with pytest.raises(ValueError, match="Test error"):
            test_function_with_error({"request_id": "req-123"})
        
        # 驗證記錄了異常
        mock_span.record_exception.assert_called_once()
        
        # 驗證設定了錯誤狀態和屬性
        mock_span.set_status.assert_called_once()
        status_call = mock_span.set_status.call_args[0][0]
        assert status_call.status_code == StatusCode.ERROR
        
        mock_span.set_attribute.assert_any_call("error", True)
        mock_span.set_attribute.assert_any_call("error.type", "ValueError")
        mock_span.set_attribute.assert_any_call("error.message", "Test error")
    
    def test_trace_node_retrieve_special_attributes(self):
        """測試檢索節點的特殊屬性"""
        # 設定模擬 tracer 和 span
        mock_tracer = Mock()
        mock_span = MagicMock()
        mock_tracer.start_as_current_span.return_value.__enter__.return_value = mock_span
        
        # 設定全局 tracer
        import app.observability.tracing
        app.observability.tracing.tracer = mock_tracer
        
        # 定義檢索節點
        @trace_node("retrieve")
        def retrieve_node(state):
            return {
                "documents": ["doc1", "doc2", "doc3"],
                **state
            }
        
        # 執行函數
        result = retrieve_node({"query": "test"})
        
        # 驗證設定了文檔數量屬性
        mock_span.set_attribute.assert_any_call("retrieve.doc_count", 3)
    
    def test_trace_node_synthesize_special_attributes(self):
        """測試合成節點的特殊屬性"""
        # 設定模擬 tracer 和 span
        mock_tracer = Mock()
        mock_span = MagicMock()
        mock_tracer.start_as_current_span.return_value.__enter__.return_value = mock_span
        
        # 設定全局 tracer
        import app.observability.tracing
        app.observability.tracing.tracer = mock_tracer
        
        # 定義合成節點
        @trace_node("synthesize")
        def synthesize_node(state):
            return {
                "answer": "This is a test answer with some length",
                **state
            }
        
        # 執行函數
        result = synthesize_node({"query": "test"})
        
        # 驗證設定了答案長度屬性
        mock_span.set_attribute.assert_any_call("synthesize.answer_length", 38)
    
    def test_trace_node_validate_special_attributes(self):
        """測試驗證節點的特殊屬性"""
        # 設定模擬 tracer 和 span
        mock_tracer = Mock()
        mock_span = MagicMock()
        mock_tracer.start_as_current_span.return_value.__enter__.return_value = mock_span
        
        # 設定全局 tracer
        import app.observability.tracing
        app.observability.tracing.tracer = mock_tracer
        
        # 定義驗證節點
        @trace_node("validate")
        def validate_node(state):
            return {
                "metrics": {
                    "is_valid": True,
                    "warnings": ["warning1", "warning2"],
                    "quality_score": 0.85
                },
                **state
            }
        
        # 執行函數
        result = validate_node({"query": "test"})
        
        # 驗證設定了驗證相關屬性
        mock_span.set_attribute.assert_any_call("validate.is_valid", True)
        mock_span.set_attribute.assert_any_call("validate.warning_count", 2)
    
    def test_trace_llm_call_decorator_success(self):
        """測試 LLM 調用追蹤裝飾器 - 成功情況"""
        # 設定模擬 tracer 和 span
        mock_tracer = Mock()
        mock_span = MagicMock()
        mock_tracer.start_as_current_span.return_value.__enter__.return_value = mock_span
        
        # 設定全局 tracer
        import app.observability.tracing
        app.observability.tracing.tracer = mock_tracer
        
        # 模擬 LLM 結果
        mock_result = Mock()
        mock_result.usage.prompt_tokens = 100
        mock_result.usage.completion_tokens = 50
        mock_result.usage.total_tokens = 150
        
        # 定義測試函數
        @trace_llm_call("gpt-4", "generate")
        def call_llm():
            time.sleep(0.01)
            return mock_result
        
        # 執行函數
        result = call_llm()
        
        # 驗證結果
        assert result == mock_result
        
        # 驗證 span 建立
        mock_tracer.start_as_current_span.assert_called_once_with(
            "llm.generate",
            attributes={
                "llm.model": "gpt-4",
                "llm.operation": "generate"
            }
        )
        
        # 驗證設定了 token 使用量屬性
        mock_span.set_attribute.assert_any_call("llm.prompt_tokens", 100)
        mock_span.set_attribute.assert_any_call("llm.completion_tokens", 50)
        mock_span.set_attribute.assert_any_call("llm.total_tokens", 150)
    
    def test_trace_llm_call_decorator_without_usage(self):
        """測試 LLM 調用追蹤裝飾器 - 無使用量資訊"""
        # 設定模擬 tracer 和 span
        mock_tracer = Mock()
        mock_span = MagicMock()
        mock_tracer.start_as_current_span.return_value.__enter__.return_value = mock_span
        
        # 設定全局 tracer
        import app.observability.tracing
        app.observability.tracing.tracer = mock_tracer
        
        # 定義測試函數
        @trace_llm_call("custom-model", "embed")
        def call_llm():
            return {"embeddings": [[0.1, 0.2, 0.3]]}
        
        # 執行函數
        result = call_llm()
        
        # 驗證結果
        assert "embeddings" in result
        
        # 驗證沒有設定 token 屬性
        token_attrs = [call for call in mock_span.set_attribute.call_args_list 
                      if "tokens" in str(call)]
        assert len(token_attrs) == 0
    
    def test_trace_llm_call_decorator_error(self):
        """測試 LLM 調用追蹤裝飾器 - 錯誤情況"""
        # 設定模擬 tracer 和 span
        mock_tracer = Mock()
        mock_span = MagicMock()
        mock_tracer.start_as_current_span.return_value.__enter__.return_value = mock_span
        
        # 設定全局 tracer
        import app.observability.tracing
        app.observability.tracing.tracer = mock_tracer
        
        # 定義會出錯的測試函數
        @trace_llm_call("gpt-4", "generate")
        def call_llm_with_error():
            raise Exception("API rate limit")
        
        # 執行函數並捕獲異常
        with pytest.raises(Exception, match="API rate limit"):
            call_llm_with_error()
        
        # 驗證記錄了異常和錯誤狀態
        mock_span.record_exception.assert_called_once()
        mock_span.set_status.assert_called_once()
        status_call = mock_span.set_status.call_args[0][0]
        assert status_call.status_code == StatusCode.ERROR
    
    def test_trace_retrieval_decorator_success(self):
        """測試檢索追蹤裝飾器 - 成功情況"""
        # 設定模擬 tracer 和 span
        mock_tracer = Mock()
        mock_span = MagicMock()
        mock_tracer.start_as_current_span.return_value.__enter__.return_value = mock_span
        
        # 設定全局 tracer
        import app.observability.tracing
        app.observability.tracing.tracer = mock_tracer
        
        # 模擬檢索結果
        mock_docs = [
            Mock(metadata={"score": 0.9}),
            Mock(metadata={"score": 0.8}),
            Mock(metadata={"score": 0.7})
        ]
        
        # 定義測試函數
        @trace_retrieval("vector")
        def retrieve_documents(query):
            time.sleep(0.01)
            return mock_docs
        
        # 執行函數
        results = retrieve_documents("test query")
        
        # 驗證結果
        assert len(results) == 3
        
        # 驗證 span 建立
        mock_tracer.start_as_current_span.assert_called_once_with(
            "retrieval.vector",
            attributes={
                "retrieval.type": "vector",
                "retrieval.query": "test query",
                "retrieval.query_length": 10
            }
        )
        
        # 驗證設定了檢索結果屬性
        mock_span.set_attribute.assert_any_call("retrieval.result_count", 3)
        mock_span.set_attribute.assert_any_call("retrieval.max_score", 0.9)
        mock_span.set_attribute.assert_any_call("retrieval.min_score", 0.7)
        mock_span.set_attribute.assert_any_call("retrieval.avg_score", 0.8)
    
    def test_trace_retrieval_decorator_empty_results(self):
        """測試檢索追蹤裝飾器 - 空結果"""
        # 設定模擬 tracer 和 span
        mock_tracer = Mock()
        mock_span = MagicMock()
        mock_tracer.start_as_current_span.return_value.__enter__.return_value = mock_span
        
        # 設定全局 tracer
        import app.observability.tracing
        app.observability.tracing.tracer = mock_tracer
        
        # 定義測試函數
        @trace_retrieval("bm25")
        def retrieve_documents(query):
            return []
        
        # 執行函數
        results = retrieve_documents("test query")
        
        # 驗證結果
        assert results == []
        
        # 驗證設定了結果數量為 0
        mock_span.set_attribute.assert_any_call("retrieval.result_count", 0)
    
    def test_trace_retrieval_decorator_error(self):
        """測試檢索追蹤裝飾器 - 錯誤情況"""
        # 設定模擬 tracer 和 span
        mock_tracer = Mock()
        mock_span = MagicMock()
        mock_tracer.start_as_current_span.return_value.__enter__.return_value = mock_span
        
        # 設定全局 tracer
        import app.observability.tracing
        app.observability.tracing.tracer = mock_tracer
        
        # 定義會出錯的測試函數
        @trace_retrieval("vector")
        def retrieve_with_error(query):
            raise ConnectionError("Database connection failed")
        
        # 執行函數並捕獲異常
        with pytest.raises(ConnectionError, match="Database connection failed"):
            retrieve_with_error("test query")
        
        # 驗證記錄了異常和錯誤狀態
        mock_span.record_exception.assert_called_once()
        mock_span.set_status.assert_called_once()
        status_call = mock_span.set_status.call_args[0][0]
        assert status_call.status_code == StatusCode.ERROR
    
    def test_long_query_truncation(self):
        """測試長查詢的截斷"""
        # 設定模擬 tracer 和 span
        mock_tracer = Mock()
        mock_span = MagicMock()
        mock_tracer.start_as_current_span.return_value.__enter__.return_value = mock_span
        
        # 設定全局 tracer
        import app.observability.tracing
        app.observability.tracing.tracer = mock_tracer
        
        # 定義測試函數
        @trace_node("test_node")
        def test_function(state):
            return state
        
        # 執行函數，使用超長查詢
        long_query = "x" * 200
        test_function({"query": long_query, "request_id": "req-123"})
        
        # 驗證查詢被截斷到 100 字符
        call_args = mock_tracer.start_as_current_span.call_args[1]
        assert len(call_args["attributes"]["query.text"]) == 100
        assert call_args["attributes"]["query.length"] == 200