"""
示例：如何在 API 端點中處理錯誤
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, Optional
import logging
import time

from app.graph.build import build_graph

router = APIRouter()
logger = logging.getLogger(__name__)

class RAGRequest(BaseModel):
    query: str
    raw_texts: Optional[list[str]] = None  # 可選的原始文本輸入

class RAGResponse(BaseModel):
    answer: str
    error: Optional[str] = None
    processing_time: float
    metrics: Dict[str, Any] = {}

@router.post("/rag/query", response_model=RAGResponse)
async def rag_query_with_error_handling(request: RAGRequest):
    """
    具有完整錯誤處理的 RAG 查詢端點
    """
    start_time = time.time()
    
    try:
        # 構建輸入狀態
        input_state = {
            "query": request.query,
        }
        
        # 如果提供了原始文本，加入狀態
        if request.raw_texts:
            input_state["raw_texts"] = request.raw_texts
        
        # 調用圖形處理
        result = graph_app.invoke(input_state)
        
        # 檢查是否有錯誤
        if result.get("error"):
            # 錯誤已經由 error_handler_node 處理
            # 返回友好的錯誤訊息
            logger.warning(f"Graph processing error: {result['error']}")
            return RAGResponse(
                answer=result.get("answer", "系統處理時發生錯誤"),
                error=result.get("error"),
                processing_time=time.time() - start_time,
                metrics=result.get("metrics", {})
            )
        
        # 正常返回結果
        return RAGResponse(
            answer=result.get("answer", ""),
            processing_time=time.time() - start_time,
            metrics=result.get("metrics", {})
        )
        
    except Exception as e:
        # 捕獲未預期的錯誤
        logger.error(f"Unexpected error in RAG query: {str(e)}", exc_info=True)
        
        # 返回 HTTP 500 錯誤
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Internal server error",
                "message": "系統發生未預期的錯誤，請聯繫技術支援",
                "request_id": str(time.time())  # 簡易請求 ID
            }
        )

@router.get("/health")
async def health_check():
    """
    健康檢查端點，可以擴展來檢查各個服務的狀態
    """
    health_status = {
        "status": "healthy",
        "services": {}
    }
    
    # 檢查 LLM 服務
    try:
        # 簡單的 ping 測試
        test_result = llm.invoke("ping")
        health_status["services"]["llm"] = "operational"
    except Exception as e:
        health_status["services"]["llm"] = f"degraded: {str(e)}"
        health_status["status"] = "degraded"
    
    # 檢查向量資料庫
    try:
        # 簡單的檢索測試
        test_docs = retriever.get_relevant_documents("test", k=1)
        health_status["services"]["vector_db"] = "operational"
    except Exception as e:
        health_status["services"]["vector_db"] = f"degraded: {str(e)}"
        health_status["status"] = "degraded"
    
    return health_status

# 錯誤處理中間件示例
@router.middleware("http")
async def add_error_handling_headers(request, call_next):
    """
    添加錯誤處理相關的 HTTP 標頭
    """
    try:
        response = await call_next(request)
        
        # 添加重試相關的標頭
        if response.status_code >= 500:
            response.headers["Retry-After"] = "5"  # 建議 5 秒後重試
            response.headers["X-RateLimit-Remaining"] = "10"  # 剩餘配額
        
        return response
        
    except Exception as e:
        logger.error(f"Middleware error: {str(e)}", exc_info=True)
        raise