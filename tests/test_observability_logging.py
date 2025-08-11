"""
可觀測性 - 結構化日誌模組單元測試
測試 loguru 日誌配置、結構化格式、請求上下文管理
"""

import pytest
import json
import sys
from unittest.mock import Mock, patch, call
from datetime import datetime
from loguru import logger

from app.observability.logging import (
    serialize_record,
    setup_logging,
    get_logger,
    set_request_context,
    clear_request_context,
    with_request_context,
    request_context
)


class TestStructuredLogging:
    """結構化日誌測試類別"""
    
    def setup_method(self):
        """測試前清理 logger 配置"""
        # 移除所有 handler 避免測試間干擾
        logger.remove()
        # 清理請求上下文
        clear_request_context()
    
    def test_serialize_record_basic(self):
        """測試基本日誌記錄序列化"""
        # 建立模擬的日誌記錄
        mock_time = Mock()
        mock_time.isoformat.return_value = "2024-01-01T12:00:00"
        
        record = {
            "time": mock_time,
            "level": Mock(name="INFO"),
            "message": "Test message",
            "name": "test_module",
            "function": "test_function",
            "line": 42,
            "extra": {"custom_field": "custom_value"}
        }
        
        # 序列化記錄
        result = serialize_record(record)
        parsed = json.loads(result)
        
        # 驗證基本欄位
        assert parsed["timestamp"] == "2024-01-01T12:00:00"
        assert parsed["level"] == "INFO"
        assert parsed["message"] == "Test message"
        assert parsed["module"] == "test_module"
        assert parsed["function"] == "test_function"
        assert parsed["line"] == 42
        assert parsed["extra"]["custom_field"] == "custom_value"
    
    def test_serialize_record_with_context(self):
        """測試帶請求上下文的日誌序列化"""
        # 設定請求上下文
        set_request_context(
            request_id="req-123",
            node_name="retrieve",
            user_id="user-456",
            session_id="session-789"
        )
        
        # 建立模擬的日誌記錄
        mock_time = Mock()
        mock_time.isoformat.return_value = "2024-01-01T12:00:00"
        
        record = {
            "time": mock_time,
            "level": Mock(name="INFO"),
            "message": "Test with context",
            "name": "test_module",
            "function": "test_function",
            "line": 42
        }
        
        # 序列化記錄
        result = serialize_record(record)
        parsed = json.loads(result)
        
        # 驗證上下文欄位
        assert parsed["request_id"] == "req-123"
        assert parsed["node_name"] == "retrieve"
        assert parsed["user_id"] == "user-456"
        assert parsed["session_id"] == "session-789"
    
    def test_serialize_record_with_exception(self):
        """測試帶異常資訊的日誌序列化"""
        # 建立模擬的異常
        mock_exception = Mock()
        mock_exception.type.__name__ = "ValueError"
        mock_exception.value = "Test error"
        mock_exception.traceback.raw = "Traceback details..."
        
        # 建立模擬的日誌記錄
        mock_time = Mock()
        mock_time.isoformat.return_value = "2024-01-01T12:00:00"
        
        record = {
            "time": mock_time,
            "level": Mock(name="ERROR"),
            "message": "Error occurred",
            "name": "test_module",
            "function": "test_function",
            "line": 42,
            "exception": mock_exception
        }
        
        # 序列化記錄
        result = serialize_record(record)
        parsed = json.loads(result)
        
        # 驗證異常欄位
        assert "exception" in parsed
        assert parsed["exception"]["type"] == "ValueError"
        assert parsed["exception"]["value"] == "Test error"
        assert parsed["exception"]["traceback"] == "Traceback details..."
    
    @patch('app.observability.logging.logger')
    def test_setup_logging_json_format(self, mock_logger):
        """測試 JSON 格式日誌設定"""
        # 設定 JSON 格式日誌
        setup_logging(level="DEBUG", json_logs=True, log_file=None)
        
        # 驗證 logger 配置
        mock_logger.remove.assert_called_once()
        
        # 驗證添加 handler 的調用
        add_calls = mock_logger.add.call_args_list
        assert len(add_calls) >= 1
        
        # 檢查 stdout handler
        stdout_call = add_calls[0]
        assert stdout_call[0][0] == sys.stdout
        assert stdout_call[1]["level"] == "DEBUG"
        assert stdout_call[1]["serialize"] is True
        # format 應該是 serialize_record 函數
        assert callable(stdout_call[1]["format"])
    
    @patch('app.observability.logging.logger')
    def test_setup_logging_human_format(self, mock_logger):
        """測試人類可讀格式日誌設定"""
        # 設定人類可讀格式日誌
        setup_logging(level="INFO", json_logs=False, log_file=None)
        
        # 驗證 logger 配置
        mock_logger.remove.assert_called_once()
        
        # 驗證添加 handler 的調用
        add_calls = mock_logger.add.call_args_list
        assert len(add_calls) >= 1
        
        # 檢查 stdout handler
        stdout_call = add_calls[0]
        assert stdout_call[0][0] == sys.stdout
        assert stdout_call[1]["level"] == "INFO"
        assert stdout_call[1]["serialize"] is False
        # format 應該是格式字串
        assert isinstance(stdout_call[1]["format"], str)
        assert "{time:" in stdout_call[1]["format"]
        assert "{level:" in stdout_call[1]["format"]
        assert "{extra[request_id]}" in stdout_call[1]["format"]
    
    @patch('app.observability.logging.logger')
    def test_setup_logging_with_file(self, mock_logger):
        """測試帶文件輸出的日誌設定"""
        # 設定帶文件輸出的日誌
        setup_logging(level="WARNING", json_logs=True, log_file="/tmp/test.log")
        
        # 驗證添加了兩個 handler
        add_calls = mock_logger.add.call_args_list
        assert len(add_calls) >= 2  # stdout + file
        
        # 檢查文件 handler
        file_call = None
        for call in add_calls:
            if call[0][0] == "/tmp/test.log":
                file_call = call
                break
        
        assert file_call is not None
        assert file_call[1]["level"] == "WARNING"
        assert file_call[1]["rotation"] == "100 MB"
        assert file_call[1]["retention"] == "7 days"
        assert file_call[1]["compression"] == "zip"
        assert file_call[1]["serialize"] is True
    
    def test_get_logger_with_name(self):
        """測試獲取命名 logger"""
        # 獲取命名 logger
        test_logger = get_logger("test_module")
        
        # 驗證返回的是 logger 實例
        # 注意：loguru 的 bind 返回的是 logger 本身，但帶有綁定的上下文
        assert hasattr(test_logger, 'info')
        assert hasattr(test_logger, 'error')
        assert hasattr(test_logger, 'debug')
    
    def test_get_logger_without_name(self):
        """測試獲取預設 logger"""
        # 獲取預設 logger
        default_logger = get_logger()
        
        # 驗證返回的是 logger 實例
        assert hasattr(default_logger, 'info')
        assert hasattr(default_logger, 'error')
        assert hasattr(default_logger, 'debug')
    
    def test_set_and_clear_request_context(self):
        """測試設定和清除請求上下文"""
        # 初始狀態應該是空的
        ctx = request_context.get()
        assert ctx == {}
        
        # 設定上下文
        set_request_context(request_id="req-123", node_name="test")
        ctx = request_context.get()
        assert ctx["request_id"] == "req-123"
        assert ctx["node_name"] == "test"
        
        # 添加更多上下文
        set_request_context(user_id="user-456")
        ctx = request_context.get()
        assert ctx["request_id"] == "req-123"  # 保留原有值
        assert ctx["node_name"] == "test"      # 保留原有值
        assert ctx["user_id"] == "user-456"    # 新增值
        
        # 清除上下文
        clear_request_context()
        ctx = request_context.get()
        assert ctx == {}
    
    def test_with_request_context_decorator(self):
        """測試請求上下文裝飾器"""
        # 定義測試函數
        @with_request_context(node_name="test_node", operation="test_op")
        def test_function():
            ctx = request_context.get()
            return ctx.copy()  # 返回當前上下文的副本
        
        # 設定初始上下文
        set_request_context(request_id="req-123")
        
        # 調用函數
        result = test_function()
        
        # 驗證函數內的上下文
        assert result["request_id"] == "req-123"  # 保留原有值
        assert result["node_name"] == "test_node"  # 裝飾器設定的值
        assert result["operation"] == "test_op"    # 裝飾器設定的值
        
        # 驗證函數執行後上下文恢復
        ctx = request_context.get()
        assert ctx["request_id"] == "req-123"
        assert "node_name" not in ctx  # 裝飾器的值已清除
        assert "operation" not in ctx  # 裝飾器的值已清除
    
    def test_with_request_context_decorator_exception(self):
        """測試請求上下文裝飾器的異常處理"""
        # 定義會拋出異常的測試函數
        @with_request_context(node_name="error_node")
        def test_function_with_error():
            raise ValueError("Test error")
        
        # 設定初始上下文
        set_request_context(request_id="req-123")
        
        # 調用函數並捕獲異常
        with pytest.raises(ValueError, match="Test error"):
            test_function_with_error()
        
        # 驗證異常後上下文已恢復
        ctx = request_context.get()
        assert ctx["request_id"] == "req-123"
        assert "node_name" not in ctx  # 裝飾器的值已清除
    
    def test_with_request_context_nested(self):
        """測試嵌套的請求上下文裝飾器"""
        # 定義嵌套的測試函數
        @with_request_context(level="outer")
        def outer_function():
            @with_request_context(level="inner", extra="data")
            def inner_function():
                ctx = request_context.get()
                return ctx.copy()
            
            return inner_function()
        
        # 設定初始上下文
        set_request_context(request_id="req-123")
        
        # 調用外層函數
        result = outer_function()
        
        # 驗證最內層的上下文
        assert result["request_id"] == "req-123"
        assert result["level"] == "inner"  # 內層覆蓋外層
        assert result["extra"] == "data"
        
        # 驗證所有裝飾器執行後上下文恢復
        ctx = request_context.get()
        assert ctx["request_id"] == "req-123"
        assert "level" not in ctx
        assert "extra" not in ctx
    
    @patch('sys.stdout.write')
    def test_logging_integration(self, mock_stdout):
        """測試日誌系統的整合運作"""
        # 設定日誌（JSON 格式）
        setup_logging(level="INFO", json_logs=True)
        
        # 設定請求上下文
        set_request_context(request_id="test-req-123", node_name="integration_test")
        
        # 獲取 logger 並記錄訊息
        test_logger = get_logger("test_module")
        test_logger.info("Integration test message", extra_field="extra_value")
        
        # 由於 loguru 的內部機制複雜，這裡主要驗證基本功能可以運作
        # 實際的輸出格式測試可能需要更複雜的設置
        
        # 清理上下文
        clear_request_context()
    
    def test_contextvars_isolation(self):
        """測試 ContextVar 的執行緒/協程隔離性"""
        import asyncio
        
        async def set_context_async(value: str):
            set_request_context(async_id=value)
            await asyncio.sleep(0.01)  # 模擬異步操作
            ctx = request_context.get()
            return ctx.get("async_id")
        
        async def test_isolation():
            # 設定主上下文
            set_request_context(main_id="main")
            
            # 並行執行多個異步任務
            tasks = [
                set_context_async("task1"),
                set_context_async("task2"),
                set_context_async("task3")
            ]
            
            results = await asyncio.gather(*tasks)
            
            # 驗證每個任務都有自己的上下文
            assert results == ["task1", "task2", "task3"]
            
            # 驗證主上下文未受影響
            ctx = request_context.get()
            assert ctx.get("main_id") == "main"
            assert "async_id" not in ctx
        
        # 執行異步測試
        asyncio.run(test_isolation())