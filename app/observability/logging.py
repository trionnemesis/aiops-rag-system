"""結構化日誌配置模塊

使用 loguru 實現結構化日誌，包含請求 ID、節點名稱、執行時間等上下文資訊
"""

import sys
import json
from loguru import logger
from typing import Dict, Any, Optional
import uuid
from contextvars import ContextVar

# 使用 ContextVar 存儲請求級別的上下文
request_context: ContextVar[Dict[str, Any]] = ContextVar('request_context', default={})


def serialize_record(record: Dict[str, Any]) -> str:
    """將日誌記錄序列化為 JSON 格式，便於 Loki/Elasticsearch 收集"""
    # 提取基礎資訊
    log_entry = {
        "timestamp": record["time"].isoformat(),
        "level": record["level"].name,
        "message": record["message"],
        "module": record["name"],
        "function": record["function"],
        "line": record["line"],
    }
    
    # 添加請求上下文
    ctx = request_context.get()
    if ctx:
        log_entry.update({
            "request_id": ctx.get("request_id"),
            "node_name": ctx.get("node_name"),
            "user_id": ctx.get("user_id"),
            "session_id": ctx.get("session_id"),
        })
    
    # 添加額外的欄位
    if record.get("extra"):
        log_entry["extra"] = record["extra"]
    
    # 添加異常資訊
    if record.get("exception"):
        log_entry["exception"] = {
            "type": record["exception"].type.__name__,
            "value": str(record["exception"].value),
            "traceback": record["exception"].traceback.raw
        }
    
    return json.dumps(log_entry, ensure_ascii=False, default=str)


def setup_logging(
    level: str = "INFO",
    json_logs: bool = True,
    log_file: Optional[str] = None
) -> None:
    """配置結構化日誌
    
    Args:
        level: 日誌級別 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        json_logs: 是否使用 JSON 格式輸出（生產環境建議開啟）
        log_file: 日誌文件路徑（可選）
    """
    # 移除預設的 handler
    logger.remove()
    
    # 配置格式
    if json_logs:
        # JSON 格式，適用於生產環境
        format_string = serialize_record
    else:
        # 人類可讀格式，適用於開發環境
        format_string = (
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{extra[request_id]}</cyan> | "
            "<cyan>{extra[node_name]}</cyan> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        )
    
    # 添加控制台輸出
    logger.add(
        sys.stdout,
        format=format_string,
        level=level,
        serialize=json_logs
    )
    
    # 添加文件輸出（如果指定）
    if log_file:
        logger.add(
            log_file,
            format=format_string,
            level=level,
            rotation="100 MB",  # 日誌輪替
            retention="7 days",  # 保留 7 天
            compression="zip",   # 壓縮舊日誌
            serialize=json_logs
        )
    
    logger.info(
        "Logging configured",
        level=level,
        json_logs=json_logs,
        log_file=log_file
    )


def get_logger(name: str = None) -> logger:
    """獲取配置好的 logger 實例
    
    Args:
        name: logger 名稱（通常使用 __name__）
    
    Returns:
        配置好的 logger 實例
    """
    if name:
        return logger.bind(module=name)
    return logger


def set_request_context(**kwargs) -> None:
    """設定請求級別的上下文資訊
    
    Args:
        **kwargs: 上下文資訊，如 request_id, node_name, user_id 等
    """
    ctx = request_context.get()
    ctx.update(kwargs)
    request_context.set(ctx)


def clear_request_context() -> None:
    """清除請求上下文"""
    request_context.set({})


def with_request_context(**kwargs):
    """裝飾器：為函數添加請求上下文
    
    使用範例：
        @with_request_context(node_name="retrieve")
        def retrieve_documents():
            logger.info("Retrieving documents")
    """
    def decorator(func):
        def wrapper(*args, **func_kwargs):
            # 保存原有上下文
            old_ctx = request_context.get().copy()
            
            try:
                # 設定新的上下文
                set_request_context(**kwargs)
                return func(*args, **func_kwargs)
            finally:
                # 恢復原有上下文
                request_context.set(old_ctx)
        
        return wrapper
    return decorator