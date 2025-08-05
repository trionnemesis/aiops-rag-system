"""度量指標收集模塊

使用 prometheus-client 收集系統健康狀況和效能指標
"""

from prometheus_client import Counter, Histogram, Gauge, Info, CollectorRegistry
from prometheus_client import start_http_server, generate_latest
from typing import Optional, Dict, Any
import time
from functools import wraps

# 創建自定義註冊表（可選，用於隔離指標）
registry = CollectorRegistry()

# === API 層級指標 ===
api_request_counter = Counter(
    'rag_api_requests_total',
    'Total number of RAG API requests',
    ['endpoint', 'method', 'status'],
    registry=registry
)

api_request_duration = Histogram(
    'rag_api_request_duration_seconds',
    'RAG API request duration in seconds',
    ['endpoint', 'method'],
    registry=registry
)

# === LangGraph 節點層級指標 ===
node_execution_time = Histogram(
    'langgraph_node_execution_seconds',
    'Execution time of LangGraph nodes in seconds',
    ['node_name'],
    buckets=(0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0),
    registry=registry
)

node_error_counter = Counter(
    'langgraph_node_errors_total',
    'Total number of errors in LangGraph nodes',
    ['node_name', 'error_type'],
    registry=registry
)

# === LLM 相關指標 ===
llm_token_counter = Counter(
    'llm_tokens_total',
    'Total number of tokens used by LLM',
    ['model', 'token_type'],  # token_type: prompt, completion, total
    registry=registry
)

llm_request_duration = Histogram(
    'llm_request_duration_seconds',
    'LLM request duration in seconds',
    ['model', 'operation'],
    registry=registry
)

llm_error_counter = Counter(
    'llm_errors_total',
    'Total number of LLM errors',
    ['model', 'error_type'],
    registry=registry
)

# === 檢索器相關指標 ===
retriever_docs_counter = Histogram(
    'retriever_documents_retrieved',
    'Number of documents retrieved',
    ['retriever_type'],
    buckets=(0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 15, 20, 25, 30),
    registry=registry
)

retriever_relevance_histogram = Histogram(
    'retriever_relevance_score',
    'Distribution of document relevance scores',
    ['retriever_type'],
    buckets=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
    registry=registry
)

retriever_duration = Histogram(
    'retriever_duration_seconds',
    'Retrieval operation duration in seconds',
    ['retriever_type'],
    registry=registry
)

# === 系統健康指標 ===
system_info = Info(
    'rag_system_info',
    'RAG system information',
    registry=registry
)

active_requests = Gauge(
    'rag_active_requests',
    'Number of active RAG requests',
    registry=registry
)

# === 驗證相關指標 ===
validation_results = Counter(
    'rag_validation_results_total',
    'RAG validation results',
    ['result'],  # result: valid, invalid
    registry=registry
)

validation_warnings = Counter(
    'rag_validation_warnings_total',
    'RAG validation warnings by type',
    ['warning_type'],
    registry=registry
)

answer_quality_score = Histogram(
    'rag_answer_quality_score',
    'Distribution of answer quality scores',
    buckets=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
    registry=registry
)


def setup_metrics(port: int = 8000) -> None:
    """設定 Prometheus 指標收集服務器
    
    Args:
        port: Prometheus 指標暴露端口
    """
    # 設定系統資訊
    system_info.info({
        'version': '1.0.0',
        'framework': 'langgraph',
        'retriever': 'opensearch'
    })
    
    # 啟動 HTTP 服務器暴露指標
    start_http_server(port, registry=registry)


def track_request_metrics(endpoint: str, method: str = "POST"):
    """裝飾器：追蹤 API 請求指標
    
    使用範例：
        @track_request_metrics("/rag/report")
        def rag_report(request):
            # API 邏輯
            return response
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # 增加活躍請求計數
            active_requests.inc()
            
            # 記錄開始時間
            start_time = time.time()
            status = "success"
            
            try:
                result = func(*args, **kwargs)
                return result
                
            except Exception as e:
                status = "error"
                raise
                
            finally:
                # 記錄請求計數
                api_request_counter.labels(
                    endpoint=endpoint,
                    method=method,
                    status=status
                ).inc()
                
                # 記錄請求耗時
                duration = time.time() - start_time
                api_request_duration.labels(
                    endpoint=endpoint,
                    method=method
                ).observe(duration)
                
                # 減少活躍請求計數
                active_requests.dec()
        
        return wrapper
    return decorator


def track_node_metrics(node_name: str):
    """裝飾器：追蹤 LangGraph 節點指標
    
    使用範例：
        @track_node_metrics("retrieve")
        def retrieve_node(state):
            # 節點邏輯
            return state
    """
    def decorator(func):
        @wraps(func)
        def wrapper(state: Dict[str, Any], *args, **kwargs):
            start_time = time.time()
            
            try:
                result = func(state, *args, **kwargs)
                
                # 記錄節點執行時間
                duration = time.time() - start_time
                node_execution_time.labels(node_name=node_name).observe(duration)
                
                # 根據節點類型記錄特定指標
                if node_name == "retrieve" and "documents" in result:
                    docs = result.get("documents", [])
                    retriever_docs_counter.labels(retriever_type="vector").observe(len(docs))
                    
                    # 記錄相關性分數
                    for doc in docs:
                        if hasattr(doc, "metadata") and "score" in doc.metadata:
                            retriever_relevance_histogram.labels(
                                retriever_type="vector"
                            ).observe(doc.metadata["score"])
                
                elif node_name == "validate" and "metrics" in result:
                    metrics = result.get("metrics", {})
                    
                    # 記錄驗證結果
                    is_valid = metrics.get("is_valid", False)
                    validation_results.labels(
                        result="valid" if is_valid else "invalid"
                    ).inc()
                    
                    # 記錄警告
                    warnings = metrics.get("warnings", [])
                    for warning in warnings:
                        validation_warnings.labels(warning_type=warning).inc()
                    
                    # 記錄品質分數
                    if "quality_score" in metrics:
                        answer_quality_score.observe(metrics["quality_score"])
                
                return result
                
            except Exception as e:
                # 記錄錯誤
                node_error_counter.labels(
                    node_name=node_name,
                    error_type=type(e).__name__
                ).inc()
                raise
        
        return wrapper
    return decorator


def track_llm_metrics(model: str, operation: str = "generate"):
    """裝飾器：追蹤 LLM 調用指標
    
    Args:
        model: 模型名稱
        operation: 操作類型
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                
                # 記錄執行時間
                duration = time.time() - start_time
                llm_request_duration.labels(
                    model=model,
                    operation=operation
                ).observe(duration)
                
                # 記錄 token 使用量
                if hasattr(result, "usage"):
                    llm_token_counter.labels(
                        model=model,
                        token_type="prompt"
                    ).inc(result.usage.prompt_tokens)
                    
                    llm_token_counter.labels(
                        model=model,
                        token_type="completion"
                    ).inc(result.usage.completion_tokens)
                    
                    llm_token_counter.labels(
                        model=model,
                        token_type="total"
                    ).inc(result.usage.total_tokens)
                
                return result
                
            except Exception as e:
                llm_error_counter.labels(
                    model=model,
                    error_type=type(e).__name__
                ).inc()
                raise
        
        return wrapper
    return decorator


def track_retrieval_metrics(retriever_type: str = "vector"):
    """裝飾器：追蹤檢索操作指標
    
    Args:
        retriever_type: 檢索器類型
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                
                # 記錄執行時間
                duration = time.time() - start_time
                retriever_duration.labels(
                    retriever_type=retriever_type
                ).observe(duration)
                
                # 記錄檢索結果
                if isinstance(result, list):
                    retriever_docs_counter.labels(
                        retriever_type=retriever_type
                    ).observe(len(result))
                    
                    # 記錄相關性分數
                    for doc in result:
                        if hasattr(doc, "metadata") and "score" in doc.metadata:
                            retriever_relevance_histogram.labels(
                                retriever_type=retriever_type
                            ).observe(doc.metadata["score"])
                
                return result
                
            except Exception as e:
                raise
        
        return wrapper
    return decorator


def get_metrics() -> str:
    """獲取當前的 Prometheus 指標（用於手動導出）
    
    Returns:
        Prometheus 格式的指標字串
    """
    return generate_latest(registry).decode('utf-8')