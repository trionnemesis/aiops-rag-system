"""分散式追蹤配置模塊

使用 OpenTelemetry 實現分散式追蹤，支援 Jaeger 和 OTLP 導出
"""

import os
from typing import Optional, Dict, Any
from functools import wraps
import time

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    ConsoleSpanExporter,
)
from opentelemetry.sdk.resources import Resource, SERVICE_NAME, SERVICE_VERSION
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from opentelemetry.trace import Status, StatusCode

# 全局 tracer 實例
tracer: Optional[trace.Tracer] = None


def setup_tracing(
    service_name: str = "langgraph-rag",
    service_version: str = "1.0.0",
    jaeger_endpoint: Optional[str] = None,
    otlp_endpoint: Optional[str] = None,
    console_export: bool = False
) -> None:
    """配置分散式追蹤
    
    Args:
        service_name: 服務名稱
        service_version: 服務版本
        jaeger_endpoint: Jaeger 收集器端點 (如: "localhost:6831")
        otlp_endpoint: OTLP 收集器端點 (如: "localhost:4317")
        console_export: 是否同時輸出到控制台（用於開發調試）
    """
    global tracer
    
    # 創建資源資訊
    resource = Resource.create({
        SERVICE_NAME: service_name,
        SERVICE_VERSION: service_version,
        "deployment.environment": os.getenv("ENVIRONMENT", "development"),
    })
    
    # 設定 TracerProvider
    provider = TracerProvider(resource=resource)
    trace.set_tracer_provider(provider)
    
    # 添加導出器
    if jaeger_endpoint:
        jaeger_exporter = JaegerExporter(
            agent_host_name=jaeger_endpoint.split(":")[0],
            agent_port=int(jaeger_endpoint.split(":")[1]) if ":" in jaeger_endpoint else 6831,
        )
        provider.add_span_processor(BatchSpanProcessor(jaeger_exporter))
    
    if otlp_endpoint:
        otlp_exporter = OTLPSpanExporter(
            endpoint=otlp_endpoint,
            insecure=True,  # 開發環境使用不安全連接
        )
        provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
    
    if console_export:
        console_exporter = ConsoleSpanExporter()
        provider.add_span_processor(BatchSpanProcessor(console_exporter))
    
    # 獲取 tracer
    tracer = trace.get_tracer(service_name, service_version)
    
    # 自動儀表化
    FastAPIInstrumentor.instrument()
    HTTPXClientInstrumentor.instrument()


def get_tracer() -> trace.Tracer:
    """獲取 tracer 實例"""
    global tracer
    if tracer is None:
        raise RuntimeError("Tracer not initialized. Call setup_tracing() first.")
    return tracer


def trace_node(node_name: str):
    """裝飾器：為 LangGraph 節點添加追蹤
    
    使用範例：
        @trace_node("retrieve")
        def retrieve_node(state):
            # 節點邏輯
            return state
    """
    def decorator(func):
        @wraps(func)
        def wrapper(state: Dict[str, Any], *args, **kwargs):
            tracer = get_tracer()
            
            # 從狀態中提取上下文資訊
            request_id = state.get("request_id", "unknown")
            query = state.get("query", "")
            
            # 創建 span
            with tracer.start_as_current_span(
                f"langgraph.node.{node_name}",
                attributes={
                    "node.name": node_name,
                    "request.id": request_id,
                    "query.text": query[:100],  # 僅記錄前 100 字符
                    "query.length": len(query),
                }
            ) as span:
                start_time = time.time()
                
                try:
                    # 執行節點函數
                    result = func(state, *args, **kwargs)
                    
                    # 記錄執行時間
                    execution_time = time.time() - start_time
                    span.set_attribute("node.execution_time_ms", execution_time * 1000)
                    
                    # 記錄節點特定的屬性
                    if node_name == "retrieve" and "documents" in result:
                        span.set_attribute("retrieve.doc_count", len(result.get("documents", [])))
                    elif node_name == "synthesize" and "answer" in result:
                        span.set_attribute("synthesize.answer_length", len(result.get("answer", "")))
                    elif node_name == "validate" and "metrics" in result:
                        metrics = result.get("metrics", {})
                        span.set_attribute("validate.is_valid", metrics.get("is_valid", False))
                        if "warnings" in metrics:
                            span.set_attribute("validate.warning_count", len(metrics["warnings"]))
                    
                    # 設定成功狀態
                    span.set_status(Status(StatusCode.OK))
                    
                    return result
                    
                except Exception as e:
                    # 記錄錯誤
                    span.record_exception(e)
                    span.set_status(
                        Status(StatusCode.ERROR, f"Node {node_name} failed: {str(e)}")
                    )
                    span.set_attribute("error", True)
                    span.set_attribute("error.type", type(e).__name__)
                    span.set_attribute("error.message", str(e))
                    raise
        
        return wrapper
    return decorator


def trace_llm_call(model_name: str, operation: str = "generate"):
    """裝飾器：追蹤 LLM 調用
    
    Args:
        model_name: 模型名稱 (如 "gpt-4", "gemini-pro")
        operation: 操作類型 (如 "generate", "embed")
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            tracer = get_tracer()
            
            with tracer.start_as_current_span(
                f"llm.{operation}",
                attributes={
                    "llm.model": model_name,
                    "llm.operation": operation,
                }
            ) as span:
                start_time = time.time()
                
                try:
                    result = func(*args, **kwargs)
                    
                    # 記錄執行時間
                    execution_time = time.time() - start_time
                    span.set_attribute("llm.execution_time_ms", execution_time * 1000)
                    
                    # 如果結果包含 token 使用資訊
                    if hasattr(result, "usage"):
                        span.set_attribute("llm.prompt_tokens", result.usage.prompt_tokens)
                        span.set_attribute("llm.completion_tokens", result.usage.completion_tokens)
                        span.set_attribute("llm.total_tokens", result.usage.total_tokens)
                    
                    span.set_status(Status(StatusCode.OK))
                    return result
                    
                except Exception as e:
                    span.record_exception(e)
                    span.set_status(
                        Status(StatusCode.ERROR, f"LLM call failed: {str(e)}")
                    )
                    raise
        
        return wrapper
    return decorator


def trace_retrieval(retriever_type: str = "vector"):
    """裝飾器：追蹤檢索操作
    
    Args:
        retriever_type: 檢索器類型 (如 "vector", "bm25", "hybrid")
    """
    def decorator(func):
        @wraps(func)
        def wrapper(query: str, *args, **kwargs):
            tracer = get_tracer()
            
            with tracer.start_as_current_span(
                f"retrieval.{retriever_type}",
                attributes={
                    "retrieval.type": retriever_type,
                    "retrieval.query": query[:100],
                    "retrieval.query_length": len(query),
                }
            ) as span:
                start_time = time.time()
                
                try:
                    result = func(query, *args, **kwargs)
                    
                    # 記錄執行時間
                    execution_time = time.time() - start_time
                    span.set_attribute("retrieval.execution_time_ms", execution_time * 1000)
                    
                    # 記錄檢索結果
                    if isinstance(result, list):
                        span.set_attribute("retrieval.result_count", len(result))
                        
                        # 記錄相關性分數分佈
                        if result and hasattr(result[0], "metadata") and "score" in result[0].metadata:
                            scores = [doc.metadata["score"] for doc in result if "score" in doc.metadata]
                            if scores:
                                span.set_attribute("retrieval.max_score", max(scores))
                                span.set_attribute("retrieval.min_score", min(scores))
                                span.set_attribute("retrieval.avg_score", sum(scores) / len(scores))
                    
                    span.set_status(Status(StatusCode.OK))
                    return result
                    
                except Exception as e:
                    span.record_exception(e)
                    span.set_status(
                        Status(StatusCode.ERROR, f"Retrieval failed: {str(e)}")
                    )
                    raise
        
        return wrapper
    return decorator