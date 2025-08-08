from __future__ import annotations
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, field_validator
from typing import Any, Dict, Optional, List
import time, uuid

from langchain_community.vectorstores import OpenSearchVectorSearch
from langchain_core.documents import Document

from app.graph.build import build_graph
from app.observability import (
    get_logger, 
    set_request_context, 
    clear_request_context,
    track_request_metrics,
    tracer
)

# 使用結構化日誌
logger = get_logger(__name__)

router = APIRouter()

# ======== 這裡沿用你現有的 llm / retriever 物件 ========
# llm: 你已配置好的 LLM（例如 Gemini/OpenAI 的 LangChain 介面）
llm = ...  # TODO: 注入你現有的 llm
# retriever: 你現有的 OpenSearch 向量檢索（務必對應 knn_vector）
retriever = ...  # TODO: 注入你現有的 retriever

# （選配）BM25 與 RRF：若未準備，保留 None，整體仍可運作
def bm25_search_fn(query: str, top_k: int = 8):
    # TODO: 以 opensearch-py 呼叫 _search + match 查詢回傳 List[Document]
    return []

# 建立一次 graph（服務啟動時）
policy = {
    "use_hyde": True,              # 可按情況調整
    "use_multi_query": True,
    "multi_query_alts": 2,
    "use_rrf": False,             # 若你已配置 BM25 與 RRF，可改 True
    "top_k": 8,
    "max_ctx_chars": 6000,
    "strict_citation": True,
    "fallback_text": "（臨時降階回覆，稍後補充細節與來源）",
    "min_docs": 2,
    "min_answer_len": 40,
}
graph_app = build_graph(
    llm=llm,
    retriever=retriever,
    bm25_search_fn=bm25_search_fn,    # 沒有就換成 None
    rrf_fuse_fn=None,                 # 沒有就 None（或使用預設 simple_rrf_fuse）
)

class RAGRequest(BaseModel):
    """RAG request model with validation"""
    query: str = Field(
        ..., 
        min_length=1, 
        max_length=1000,
        description="User query for RAG processing"
    )
    
    # Additional optional fields
    raw_texts: Optional[List[str]] = Field(
        default=None,
        max_items=100,
        description="Optional raw log/alert texts for processing"
    )
    
    # Configuration overrides
    top_k: Optional[int] = Field(
        default=None,
        ge=1,
        le=50,
        description="Number of documents to retrieve"
    )
    
    use_hyde: Optional[bool] = Field(
        default=None,
        description="Whether to use HyDE for query expansion"
    )
    
    use_multi_query: Optional[bool] = Field(
        default=None,
        description="Whether to use multi-query expansion"
    )
    
    @field_validator('query')
    @classmethod
    def validate_query(cls, v: str) -> str:
        """Validate query is not empty or just whitespace"""
        if not v or not v.strip():
            raise ValueError("Query cannot be empty or whitespace only")
        # Remove excessive whitespace
        v = ' '.join(v.split())
        # Check for suspicious patterns that might overload the system
        if len(v) > 1000:
            raise ValueError("Query is too long (max 1000 characters)")
        return v
    
    @field_validator('raw_texts')
    @classmethod
    def validate_raw_texts(cls, v: Optional[List[str]]) -> Optional[List[str]]:
        """Validate and clean raw texts"""
        if v is None:
            return v
        
        cleaned = []
        for text in v:
            if text and text.strip():
                # Limit individual text length
                text = text.strip()[:5000]
                cleaned.append(text)
        
        if not cleaned:
            return None
        
        return cleaned

@router.post("/rag/report")
@track_request_metrics("/rag/report", method="POST")
def rag_report(req: RAGRequest) -> Dict[str, Any]:
    """維持原路由與回傳結構；內部切到 LangGraph 執行"""
    request_id = str(uuid.uuid4())
    
    # 設定請求上下文（用於結構化日誌）
    set_request_context(request_id=request_id, endpoint="/rag/report")
    
    # 使用 OpenTelemetry 創建追蹤 span
    with tracer.start_as_current_span(
        "rag_report",
        attributes={
            "request.id": request_id,
            "query.text": req.query[:100],
            "query.length": len(req.query),
        }
    ) as span:
        start = time.time()
        
        try:
            logger.info("Processing RAG request", query=req.query[:100], request_id=request_id)
            
            # run_id 便於追蹤，thread_id 用於狀態持久化
            cfg = {
                "configurable": {
                    "run_id": request_id,
                    "thread_id": f"thread-{request_id}"  # 使用 request_id 作為 thread_id
                }
            }
            
            # 將 request_id 加入狀態，讓節點可以使用
            initial_state = {
                "query": req.query,
                "request_id": request_id
            }
            
            # Add raw_texts if provided
            if req.raw_texts:
                initial_state["raw_texts"] = req.raw_texts
            
            result = graph_app.invoke(initial_state, config=cfg)
            latency = int((time.time() - start) * 1000)
            
            # 記錄成功
            logger.info("RAG request completed successfully", 
                       request_id=request_id, 
                       latency_ms=latency,
                       doc_count=len(result.get("docs", [])))
            
            # 添加追蹤屬性
            span.set_attribute("response.latency_ms", latency)
            span.set_attribute("response.doc_count", len(result.get("docs", [])))
            span.set_attribute("response.answer_length", len(result.get("answer", "")))
            
            # 你原本的回傳格式若不同，請在這裡轉換；以下是常見結構
            return {
                "ok": True,
                "data": {
                    "answer": result.get("answer", ""),
                    "metrics": {
                        **result.get("metrics", {}),
                        "latency_ms": latency,
                        "request_id": request_id,
                    },
                    "warnings": result.get("metrics", {}).get("warnings", []),
                },
            }
        except Exception as e:
            # 記錄錯誤
            logger.error("RAG request failed", 
                        request_id=request_id,
                        error=str(e),
                        error_type=type(e).__name__)
            
            # 在追蹤中記錄錯誤
            span.record_exception(e)
            span.set_attribute("error", True)
            
            raise HTTPException(status_code=500, detail=f"graph_error: {e}")
        finally:
            # 清理請求上下文
            clear_request_context()

# 添加健康檢查端點
@router.get("/health")
def health_check():
    """健康檢查端點"""
    return {"status": "healthy", "service": "langgraph-rag"}

# 添加 Prometheus 指標端點
@router.get("/metrics")
def get_metrics():
    """暴露 Prometheus 指標"""
    from app.observability.metrics import get_metrics as prometheus_metrics
    from fastapi.responses import PlainTextResponse
    
    return PlainTextResponse(prometheus_metrics(), media_type="text/plain")