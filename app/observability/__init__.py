"""可觀測性模塊：結構化日誌、分散式追蹤、度量指標收集

註：原本這裡只匯出了一小部分名稱，但 app/graph/nodes.py 與 app/api/routes.py
實際上會 import trace_node / track_node_metrics / set_request_context 等，
造成 ImportError。以下改為完整匯出三個子模組的公開介面。
"""

from .logging import (
    setup_logging,
    get_logger,
    set_request_context,
    clear_request_context,
    with_request_context,
)
from .tracing import (
    setup_tracing,
    tracer,
    get_tracer,
    trace_node,
    trace_llm_call,
    trace_retrieval,
)
from .metrics import (
    setup_metrics,
    get_metrics,
    track_request_metrics,
    track_node_metrics,
    track_llm_metrics,
    track_retrieval_metrics,
    node_execution_time,
    node_error_counter,
    llm_token_counter,
    llm_request_duration,
    llm_error_counter,
    retriever_docs_counter,
    retriever_relevance_histogram,
    retriever_duration,
    api_request_counter,
    api_request_duration,
    active_requests,
    validation_results,
    validation_warnings,
    answer_quality_score,
)

__all__ = [
    # 日誌
    "setup_logging",
    "get_logger",
    "set_request_context",
    "clear_request_context",
    "with_request_context",
    # 追蹤
    "setup_tracing",
    "tracer",
    "get_tracer",
    "trace_node",
    "trace_llm_call",
    "trace_retrieval",
    # 度量
    "setup_metrics",
    "get_metrics",
    "track_request_metrics",
    "track_node_metrics",
    "track_llm_metrics",
    "track_retrieval_metrics",
    "node_execution_time",
    "node_error_counter",
    "llm_token_counter",
    "llm_request_duration",
    "llm_error_counter",
    "retriever_docs_counter",
    "retriever_relevance_histogram",
    "retriever_duration",
    "api_request_counter",
    "api_request_duration",
    "active_requests",
    "validation_results",
    "validation_warnings",
    "answer_quality_score",
]
