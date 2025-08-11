"""
Observability Tracing 模組單元測試
測試分散式追蹤配置和裝飾器功能
"""

import pytest
from unittest.mock import patch, MagicMock, Mock, call
import asyncio
import os

from app.observability.tracing import (
    setup_tracing,
    get_tracer,
    trace_node,
    trace_llm_call,
    trace_retrieval,
    tracer,
    get_current_span_context
)


class TestSetupTracing:
    """測試追蹤設定"""
    
    @patch('app.observability.tracing.TracerProvider')
    @patch('app.observability.tracing.trace')
    @patch('app.observability.tracing.JaegerExporter')
    @patch('app.observability.tracing.BatchSpanProcessor')
    @patch('app.observability.tracing.FastAPIInstrumentor')
    @patch('app.observability.tracing.HTTPXClientInstrumentor')
    def test_setup_tracing_with_jaeger(
        self,
        mock_httpx_inst,
        mock_fastapi_inst,
        mock_span_processor,
        mock_jaeger_exporter,
        mock_trace,
        mock_tracer_provider
    ):
        """測試使用 Jaeger 的追蹤設定"""
        # 建立模擬物件
        mock_provider_instance = Mock()
        mock_tracer_provider.return_value = mock_provider_instance
        mock_tracer_instance = Mock()
        mock_trace.get_tracer.return_value = mock_tracer_instance
        
        # 執行設定
        setup_tracing(
            service_name="test-service",
            service_version="1.0.0",
            jaeger_endpoint="localhost:6831",
            otlp_endpoint=None,
            console_export=False
        )
        
        # 驗證 TracerProvider 建立
        mock_tracer_provider.assert_called_once()
        call_args = mock_tracer_provider.call_args[1]
        assert call_args["resource"] is not None
        
        # 驗證 Jaeger exporter 建立
        mock_jaeger_exporter.assert_called_once_with(
            agent_host_name="localhost",
            agent_port=6831
        )
        
        # 驗證 span processor 添加
        mock_span_processor.assert_called_once()
        mock_provider_instance.add_span_processor.assert_called_once()
        
        # 驗證自動儀表化
        mock_fastapi_inst.instrument.assert_called_once()
        mock_httpx_inst.instrument.assert_called_once()
        
        # 驗證全局 tracer 被設定
        from app.observability import tracing
        assert tracing.tracer is not None
    
    @patch('app.observability.tracing.TracerProvider')
    @patch('app.observability.tracing.trace')
    @patch('app.observability.tracing.OTLPSpanExporter')
    @patch('app.observability.tracing.BatchSpanProcessor')
    def test_setup_tracing_with_otlp(
        self,
        mock_span_processor,
        mock_otlp_exporter,
        mock_trace,
        mock_tracer_provider
    ):
        """測試使用 OTLP 的追蹤設定"""
        mock_provider_instance = Mock()
        mock_tracer_provider.return_value = mock_provider_instance
        
        setup_tracing(
            service_name="test-service",
            service_version="2.0.0",
            jaeger_endpoint=None,
            otlp_endpoint="localhost:4317",
            console_export=False
        )
        
        # 驗證 OTLP exporter 建立
        mock_otlp_exporter.assert_called_once_with(
            endpoint="localhost:4317",
            insecure=True
        )
        
        # 驗證 span processor 添加
        mock_span_processor.assert_called_once()
        mock_provider_instance.add_span_processor.assert_called_once()
    
    @patch('app.observability.tracing.TracerProvider')
    @patch('app.observability.tracing.trace')
    @patch('app.observability.tracing.ConsoleSpanExporter')
    @patch('app.observability.tracing.BatchSpanProcessor')
    def test_setup_tracing_with_console(
        self,
        mock_span_processor,
        mock_console_exporter,
        mock_trace,
        mock_tracer_provider
    ):
        """測試使用控制台輸出的追蹤設定"""
        mock_provider_instance = Mock()
        mock_tracer_provider.return_value = mock_provider_instance
        
        setup_tracing(
            service_name="test-service",
            service_version="1.0.0",
            jaeger_endpoint=None,
            otlp_endpoint=None,
            console_export=True
        )
        
        # 驗證 Console exporter 建立
        mock_console_exporter.assert_called_once()
        
        # 驗證 span processor 添加
        mock_span_processor.assert_called_once()
        mock_provider_instance.add_span_processor.assert_called_once()
    
    @patch.dict(os.environ, {"ENVIRONMENT": "production"})
    @patch('app.observability.tracing.Resource')
    def test_setup_tracing_resource_creation(self, mock_resource):
        """測試資源資訊建立"""
        with patch('app.observability.tracing.TracerProvider'):
            with patch('app.observability.tracing.trace'):
                setup_tracing(
                    service_name="prod-service",
                    service_version="3.0.0"
                )
        
        # 驗證資源建立包含正確資訊
        mock_resource.create.assert_called_once()
        resource_attrs = mock_resource.create.call_args[0][0]
        assert resource_attrs["service.name"] == "prod-service"
        assert resource_attrs["service.version"] == "3.0.0"
        assert resource_attrs["deployment.environment"] == "production"


class TestGetTracer:
    """測試獲取 tracer"""
    
    def test_get_tracer_not_initialized(self):
        """測試未初始化時獲取 tracer"""
        # 暫時設定 tracer 為 None
        from app.observability import tracing
        original_tracer = tracing.tracer
        tracing.tracer = None
        
        with pytest.raises(RuntimeError, match="Tracer not initialized"):
            get_tracer()
        
        # 恢復原始值
        tracing.tracer = original_tracer
    
    def test_get_tracer_initialized(self):
        """測試已初始化時獲取 tracer"""
        from app.observability import tracing
        if tracing.tracer is None:
            # 如果未初始化，先設定一個模擬的
            tracing.tracer = Mock()
        
        tracer = get_tracer()
        assert tracer is not None


class TestTraceNode:
    """測試節點追蹤裝飾器"""
    
    @patch('app.observability.tracing.tracer')
    def test_trace_node_sync_success(self, mock_tracer):
        """測試同步節點的成功追蹤"""
        # 建立模擬 span
        mock_span = MagicMock()
        mock_tracer.start_as_current_span.return_value.__enter__.return_value = mock_span
        
        # 建立測試函數
        @trace_node("test_node")
        def test_function(state):
            state["processed"] = True
            return state
        
        # 執行函數
        initial_state = {"data": "test", "query": "test query"}
        result = test_function(initial_state)
        
        # 驗證結果
        assert result["processed"] is True
        
        # 驗證 span 建立
        mock_tracer.start_as_current_span.assert_called_once_with("test_node")
        
        # 驗證屬性設定
        mock_span.set_attribute.assert_any_call("node.name", "test_node")
        mock_span.set_attribute.assert_any_call("query", "test query")
        
        # 驗證狀態設定
        from opentelemetry.trace import StatusCode
        mock_span.set_status.assert_called_once()
        status_call = mock_span.set_status.call_args[0][0]
        assert status_call.status_code == StatusCode.OK
    
    @patch('app.observability.tracing.tracer')
    def test_trace_node_sync_error(self, mock_tracer):
        """測試同步節點的錯誤追蹤"""
        # 建立模擬 span
        mock_span = MagicMock()
        mock_tracer.start_as_current_span.return_value.__enter__.return_value = mock_span
        
        # 建立會失敗的測試函數
        @trace_node("error_node")
        def failing_function(state):
            raise ValueError("測試錯誤")
        
        # 執行函數應該拋出異常
        with pytest.raises(ValueError, match="測試錯誤"):
            failing_function({"data": "test"})
        
        # 驗證錯誤狀態和屬性
        mock_span.set_attribute.assert_any_call("node.name", "error_node")
        mock_span.set_attribute.assert_any_call("error.type", "ValueError")
        mock_span.set_attribute.assert_any_call("error.message", "測試錯誤")
        
        # 驗證錯誤狀態
        from opentelemetry.trace import StatusCode
        mock_span.set_status.assert_called_once()
        status_call = mock_span.set_status.call_args[0][0]
        assert status_call.status_code == StatusCode.ERROR
    
    @patch('app.observability.tracing.tracer')
    @pytest.mark.asyncio
    async def test_trace_node_async_success(self, mock_tracer):
        """測試非同步節點的成功追蹤"""
        # 建立模擬 span
        mock_span = MagicMock()
        mock_tracer.start_as_current_span.return_value.__enter__.return_value = mock_span
        
        # 建立非同步測試函數
        @trace_node("async_node")
        async def async_function(state):
            await asyncio.sleep(0.01)  # 模擬非同步操作
            state["async_processed"] = True
            return state
        
        # 執行函數
        result = await async_function({"data": "async test"})
        
        # 驗證結果
        assert result["async_processed"] is True
        
        # 驗證 span 建立和屬性
        mock_tracer.start_as_current_span.assert_called_once_with("async_node")
        mock_span.set_attribute.assert_any_call("node.name", "async_node")


class TestTraceLLMCall:
    """測試 LLM 呼叫追蹤裝飾器"""
    
    @patch('app.observability.tracing.tracer')
    def test_trace_llm_call_success(self, mock_tracer):
        """測試 LLM 呼叫的成功追蹤"""
        # 建立模擬 span
        mock_span = MagicMock()
        mock_tracer.start_as_current_span.return_value.__enter__.return_value = mock_span
        
        # 建立測試函數
        @trace_llm_call("gpt-4", "chat")
        def llm_function(prompt):
            return {
                "content": "回應內容",
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 20,
                    "total_tokens": 30
                }
            }
        
        # 執行函數
        result = llm_function("測試提示")
        
        # 驗證結果
        assert result["content"] == "回應內容"
        
        # 驗證 span 建立
        mock_tracer.start_as_current_span.assert_called_once_with("llm_call")
        
        # 驗證屬性設定
        mock_span.set_attribute.assert_any_call("llm.model", "gpt-4")
        mock_span.set_attribute.assert_any_call("llm.operation", "chat")
        mock_span.set_attribute.assert_any_call("llm.prompt.length", 4)  # "測試提示" 的長度
        mock_span.set_attribute.assert_any_call("llm.response.length", 4)  # "回應內容" 的長度
        mock_span.set_attribute.assert_any_call("llm.usage.prompt_tokens", 10)
        mock_span.set_attribute.assert_any_call("llm.usage.completion_tokens", 20)
        mock_span.set_attribute.assert_any_call("llm.usage.total_tokens", 30)
    
    @patch('app.observability.tracing.tracer')
    def test_trace_llm_call_without_usage(self, mock_tracer):
        """測試沒有使用資訊的 LLM 呼叫追蹤"""
        # 建立模擬 span
        mock_span = MagicMock()
        mock_tracer.start_as_current_span.return_value.__enter__.return_value = mock_span
        
        # 建立測試函數
        @trace_llm_call("custom-model", "generate")
        def llm_function(prompt):
            return "簡單回應"
        
        # 執行函數
        result = llm_function("測試")
        
        # 驗證結果
        assert result == "簡單回應"
        
        # 驗證基本屬性設定
        mock_span.set_attribute.assert_any_call("llm.model", "custom-model")
        mock_span.set_attribute.assert_any_call("llm.operation", "generate")


class TestTraceRetrieval:
    """測試檢索追蹤裝飾器"""
    
    @patch('app.observability.tracing.tracer')
    def test_trace_retrieval_success(self, mock_tracer):
        """測試檢索的成功追蹤"""
        # 建立模擬 span
        mock_span = MagicMock()
        mock_tracer.start_as_current_span.return_value.__enter__.return_value = mock_span
        
        # 建立測試函數
        @trace_retrieval("vector_store")
        def retrieval_function(query, k=5):
            return [
                {"content": "文件1", "score": 0.9},
                {"content": "文件2", "score": 0.8},
                {"content": "文件3", "score": 0.7}
            ]
        
        # 執行函數
        result = retrieval_function("測試查詢", k=3)
        
        # 驗證結果
        assert len(result) == 3
        
        # 驗證 span 建立
        mock_tracer.start_as_current_span.assert_called_once_with("retrieval")
        
        # 驗證屬性設定
        mock_span.set_attribute.assert_any_call("retrieval.type", "vector_store")
        mock_span.set_attribute.assert_any_call("retrieval.query", "測試查詢")
        mock_span.set_attribute.assert_any_call("retrieval.k", 3)
        mock_span.set_attribute.assert_any_call("retrieval.num_results", 3)
    
    @patch('app.observability.tracing.tracer')
    def test_trace_retrieval_with_scores(self, mock_tracer):
        """測試包含分數的檢索追蹤"""
        # 建立模擬 span
        mock_span = MagicMock()
        mock_tracer.start_as_current_span.return_value.__enter__.return_value = mock_span
        
        # 建立測試函數
        @trace_retrieval("hybrid_search")
        def retrieval_function(query):
            from langchain_core.documents import Document
            return [
                Document(page_content="內容1", metadata={"score": 0.95}),
                Document(page_content="內容2", metadata={"score": 0.85})
            ]
        
        # 執行函數
        result = retrieval_function("測試")
        
        # 驗證屬性設定包含分數
        mock_span.set_attribute.assert_any_call("retrieval.num_results", 2)
        # 驗證設定了分數相關屬性
        attribute_calls = [call[0] for call in mock_span.set_attribute.call_args_list]
        assert any("retrieval.scores" in str(call) for call in attribute_calls)


class TestGetCurrentSpanContext:
    """測試獲取當前 span 上下文"""
    
    @patch('app.observability.tracing.trace')
    def test_get_current_span_context(self, mock_trace):
        """測試獲取當前 span 上下文"""
        # 建立模擬 span 和上下文
        mock_span = Mock()
        mock_context = Mock()
        mock_span.get_span_context.return_value = mock_context
        mock_trace.get_current_span.return_value = mock_span
        
        # 獲取上下文
        context = get_current_span_context()
        
        # 驗證
        assert context == mock_context
        mock_trace.get_current_span.assert_called_once()
        mock_span.get_span_context.assert_called_once()
    
    @patch('app.observability.tracing.trace')
    def test_get_current_span_context_no_active_span(self, mock_trace):
        """測試沒有活動 span 時獲取上下文"""
        # 模擬沒有活動 span
        mock_trace.get_current_span.return_value = None
        
        # 獲取上下文
        context = get_current_span_context()
        
        # 驗證返回 None
        assert context is None