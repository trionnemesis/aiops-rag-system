"""
Observability Logging 模組單元測試
測試結構化日誌、請求上下文注入等功能
"""

import pytest
from unittest.mock import patch, MagicMock, call
import json
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


class TestLoggingModule:
    """測試日誌模組功能"""
    
    @pytest.fixture
    def mock_record(self):
        """建立模擬的日誌記錄"""
        mock_time = MagicMock()
        mock_time.isoformat.return_value = "2024-01-01T10:00:00"
        
        return {
            "time": mock_time,
            "level": MagicMock(name="INFO"),
            "message": "測試訊息",
            "name": "test_module",
            "function": "test_function",
            "line": 42,
            "extra": {"custom_field": "custom_value"}
        }
    
    def test_serialize_record_basic(self, mock_record):
        """測試基本日誌序列化"""
        result = serialize_record(mock_record)
        log_entry = json.loads(result)
        
        assert log_entry["timestamp"] == "2024-01-01T10:00:00"
        assert log_entry["level"] == "INFO"
        assert log_entry["message"] == "測試訊息"
        assert log_entry["module"] == "test_module"
        assert log_entry["function"] == "test_function"
        assert log_entry["line"] == 42
        assert log_entry["extra"]["custom_field"] == "custom_value"
    
    def test_serialize_record_with_context(self, mock_record):
        """測試帶請求上下文的日誌序列化"""
        # 設定請求上下文
        request_context.set({
            "request_id": "req-123",
            "node_name": "test_node",
            "user_id": "user-456",
            "session_id": "session-789"
        })
        
        result = serialize_record(mock_record)
        log_entry = json.loads(result)
        
        assert log_entry["request_id"] == "req-123"
        assert log_entry["node_name"] == "test_node"
        assert log_entry["user_id"] == "user-456"
        assert log_entry["session_id"] == "session-789"
        
        # 清理上下文
        clear_request_context()
    
    def test_serialize_record_with_exception(self, mock_record):
        """測試帶異常資訊的日誌序列化"""
        # 添加異常資訊
        mock_exception = MagicMock()
        mock_exception.type.__name__ = "ValueError"
        mock_exception.value = "測試錯誤"
        mock_exception.traceback.raw = "Traceback (most recent call last):\n  File test.py"
        
        mock_record["exception"] = mock_exception
        
        result = serialize_record(mock_record)
        log_entry = json.loads(result)
        
        assert "exception" in log_entry
        assert log_entry["exception"]["type"] == "ValueError"
        assert log_entry["exception"]["value"] == "測試錯誤"
        assert "Traceback" in log_entry["exception"]["traceback"]
    
    @patch('app.observability.logging.logger')
    def test_setup_logging_json_format(self, mock_logger):
        """測試 JSON 格式日誌設定"""
        setup_logging(level="DEBUG", json_logs=True, log_file=None)
        
        # 驗證 logger 設定
        mock_logger.remove.assert_called_once()
        mock_logger.add.assert_called()
        
        # 檢查添加的參數
        add_call = mock_logger.add.call_args_list[0]
        assert add_call[1]["level"] == "DEBUG"
        assert add_call[1]["serialize"] is True
    
    @patch('app.observability.logging.logger')
    def test_setup_logging_human_readable_format(self, mock_logger):
        """測試人類可讀格式日誌設定"""
        setup_logging(level="INFO", json_logs=False, log_file=None)
        
        mock_logger.remove.assert_called_once()
        mock_logger.add.assert_called()
        
        add_call = mock_logger.add.call_args_list[0]
        assert add_call[1]["level"] == "INFO"
        assert add_call[1]["serialize"] is False
        # 檢查格式字串包含必要元素
        format_string = add_call[1]["format"]
        assert "{time:" in format_string
        assert "{level}" in format_string
        assert "{message}" in format_string
    
    @patch('app.observability.logging.logger')
    def test_setup_logging_with_file(self, mock_logger):
        """測試設定日誌檔案輸出"""
        setup_logging(level="WARNING", json_logs=True, log_file="/tmp/test.log")
        
        # 應該有兩次 add 呼叫（控制台和檔案）
        assert mock_logger.add.call_count == 2
        
        # 檢查檔案輸出設定
        file_call = mock_logger.add.call_args_list[1]
        assert file_call[0][0] == "/tmp/test.log"
        assert file_call[1]["rotation"] == "100 MB"
        assert file_call[1]["retention"] == "7 days"
        assert file_call[1]["compression"] == "zip"
    
    def test_get_logger_with_name(self):
        """測試取得命名的 logger"""
        test_logger = get_logger("test_module")
        
        # 應該是 loguru logger 的綁定版本
        assert hasattr(test_logger, "bind")
        assert hasattr(test_logger, "info")
        assert hasattr(test_logger, "error")
    
    def test_get_logger_without_name(self):
        """測試取得預設 logger"""
        test_logger = get_logger()
        
        # 應該是原始的 loguru logger
        assert test_logger == logger
    
    def test_set_and_clear_request_context(self):
        """測試設定和清除請求上下文"""
        # 初始狀態
        ctx = request_context.get()
        assert ctx == {}
        
        # 設定上下文
        set_request_context(request_id="req-123", node_name="test_node")
        ctx = request_context.get()
        assert ctx["request_id"] == "req-123"
        assert ctx["node_name"] == "test_node"
        
        # 添加更多上下文
        set_request_context(user_id="user-456")
        ctx = request_context.get()
        assert ctx["request_id"] == "req-123"
        assert ctx["node_name"] == "test_node"
        assert ctx["user_id"] == "user-456"
        
        # 清除上下文
        clear_request_context()
        ctx = request_context.get()
        assert ctx == {}
    
    def test_with_request_context_decorator(self):
        """測試請求上下文裝飾器"""
        # 定義測試函數
        @with_request_context(node_name="decorated_node", operation="test_op")
        def test_function():
            ctx = request_context.get()
            return ctx
        
        # 設定初始上下文
        set_request_context(request_id="original-req")
        
        # 執行函數
        result = test_function()
        
        # 驗證函數內的上下文
        assert result["node_name"] == "decorated_node"
        assert result["operation"] == "test_op"
        assert result["request_id"] == "original-req"  # 保留原有的
        
        # 驗證上下文已恢復
        ctx = request_context.get()
        assert ctx["request_id"] == "original-req"
        assert "node_name" not in ctx
        assert "operation" not in ctx
        
        # 清理
        clear_request_context()
    
    def test_with_request_context_decorator_exception(self):
        """測試裝飾器在異常情況下的行為"""
        @with_request_context(node_name="error_node")
        def failing_function():
            raise ValueError("測試錯誤")
        
        # 設定初始上下文
        set_request_context(request_id="original-req")
        
        # 執行應該拋出異常
        with pytest.raises(ValueError, match="測試錯誤"):
            failing_function()
        
        # 驗證上下文已恢復
        ctx = request_context.get()
        assert ctx["request_id"] == "original-req"
        assert "node_name" not in ctx
        
        # 清理
        clear_request_context()
    
    def test_context_isolation(self):
        """測試上下文隔離性"""
        # 在不同的上下文中設定不同的值
        set_request_context(request_id="req-1")
        ctx1 = request_context.get()
        
        # 模擬另一個請求（在實際應用中會是不同的線程/協程）
        clear_request_context()
        set_request_context(request_id="req-2")
        ctx2 = request_context.get()
        
        # 驗證兩個上下文是獨立的
        assert ctx1 != ctx2
        assert ctx2["request_id"] == "req-2"
        
        # 清理
        clear_request_context()
    
    @patch('app.observability.logging.logger')
    def test_logging_with_context_integration(self, mock_logger):
        """測試日誌與上下文的整合"""
        # 設定日誌（使用 JSON 格式）
        setup_logging(json_logs=True)
        
        # 設定上下文
        set_request_context(request_id="integration-test", node_name="test_node")
        
        # 取得 logger 並記錄
        test_logger = get_logger("integration_test")
        
        # 由於 get_logger 返回的是綁定的 logger，我們需要測試實際的日誌輸出
        # 這裡我們主要確認上下文設定和清除機制正常工作
        ctx = request_context.get()
        assert ctx["request_id"] == "integration-test"
        assert ctx["node_name"] == "test_node"
        
        # 清理
        clear_request_context()


class TestSerializeRecord:
    """測試日誌序列化功能的邊界情況"""
    
    def test_serialize_with_non_serializable_object(self):
        """測試處理無法序列化的物件"""
        mock_time = MagicMock()
        mock_time.isoformat.return_value = "2024-01-01T10:00:00"
        
        # 建立包含無法序列化物件的記錄
        record = {
            "time": mock_time,
            "level": MagicMock(name="INFO"),
            "message": "測試",
            "name": "test",
            "function": "test",
            "line": 1,
            "extra": {
                "custom_object": object(),  # 無法直接序列化
                "normal_field": "normal_value"
            }
        }
        
        # 應該使用 default=str 處理
        result = serialize_record(record)
        log_entry = json.loads(result)
        
        assert "extra" in log_entry
        assert "custom_object" in log_entry["extra"]
        assert log_entry["extra"]["normal_field"] == "normal_value"
    
    def test_serialize_empty_context(self):
        """測試空上下文的處理"""
        mock_time = MagicMock()
        mock_time.isoformat.return_value = "2024-01-01T10:00:00"
        
        record = {
            "time": mock_time,
            "level": MagicMock(name="DEBUG"),
            "message": "空上下文測試",
            "name": "test",
            "function": "test",
            "line": 1
        }
        
        # 確保上下文為空
        clear_request_context()
        
        result = serialize_record(record)
        log_entry = json.loads(result)
        
        # 上下文欄位應該是 None 或不存在
        assert log_entry.get("request_id") is None
        assert log_entry.get("node_name") is None
        assert log_entry.get("user_id") is None
        assert log_entry.get("session_id") is None