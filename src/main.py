from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import logging
from src.models.schemas import ReportRequest, ReportResponse, InsightReport
from src.services.rag_service import RAGService
from src.services.opensearch_service import OpenSearchService
from src.services.exceptions import (
    RAGServiceError, VectorDBError, GeminiAPIError, 
    PrometheusError, HyDEGenerationError, DocumentRetrievalError,
    ReportGenerationError, CacheError
)
from src.config import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Lifespan context manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting up AIOps RAG System...")
    try:
        opensearch = OpenSearchService()
        await opensearch.create_index()
        logger.info("OpenSearch index initialized")
    except Exception as e:
        logger.error(f"Failed to initialize OpenSearch: {str(e)}")
        raise
    yield
    # Shutdown
    logger.info("Shutting down...")

# Create FastAPI app
app = FastAPI(
    title=settings.api_title,
    version=settings.api_version,
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
rag_service = RAGService()

# Global exception handler for RequestValidationError
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """處理請求驗證錯誤"""
    logger.warning(f"Validation error: {exc.errors()}")
    return JSONResponse(
        status_code=422,
        content={
            "status": "error",
            "message": "Invalid request data",
            "details": exc.errors()
        }
    )

# Global exception handler for RAGServiceError
@app.exception_handler(RAGServiceError)
async def rag_service_exception_handler(request: Request, exc: RAGServiceError):
    """處理 RAG 服務相關錯誤"""
    error_mapping = {
        VectorDBError: (503, "Vector database service unavailable"),
        GeminiAPIError: (503, "AI model service unavailable"),
        PrometheusError: (503, "Monitoring service unavailable"),
        HyDEGenerationError: (500, "Failed to generate search query"),
        DocumentRetrievalError: (500, "Failed to retrieve relevant documents"),
        ReportGenerationError: (500, "Failed to generate report"),
        CacheError: (500, "Cache operation failed")
    }
    
    status_code = 500
    message = str(exc)
    
    for error_type, (code, default_message) in error_mapping.items():
        if isinstance(exc, error_type):
            status_code = code
            message = f"{default_message}: {str(exc)}" if str(exc) else default_message
            break
    
    logger.error(f"{exc.__class__.__name__}: {str(exc)}", exc_info=True)
    
    return JSONResponse(
        status_code=status_code,
        content={
            "status": "error",
            "message": message,
            "error_type": exc.__class__.__name__
        }
    )

@app.get("/")
async def root():
    return {
        "message": "AIOps 智慧維運報告 RAG 系統",
        "version": settings.api_version,
        "docs": "/docs"
    }

@app.get("/health")
async def health_check():
    """健康檢查端點"""
    try:
        # 可以加入更多健康檢查邏輯
        return {"status": "healthy", "version": settings.api_version}
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "error": str(e)}
        )

@app.post("/api/v1/generate_report", response_model=ReportResponse)
async def generate_report(request: ReportRequest):
    """
    生成智慧維運報告
    
    接收監控數據，透過 RAG 架構生成深度分析報告
    """
    try:
        logger.info(f"Received monitoring data: {request.monitoring_data}")
        
        # 驗證必要欄位
        if not request.monitoring_data:
            raise HTTPException(
                status_code=400, 
                detail="Monitoring data cannot be empty"
            )
        
        enriched_data = request.monitoring_data
        
        # 如果有主機名稱，嘗試從 Prometheus 獲取額外數據
        if "主機" in enriched_data:
            hostname = enriched_data["主機"]
            try:
                enriched_data = await rag_service.enrich_with_prometheus(
                    hostname, 
                    enriched_data
                )
                logger.info(f"Successfully enriched data for host: {hostname}")
            except PrometheusError as e:
                # Prometheus 錯誤不應阻止報告生成，記錄警告並繼續
                logger.warning(f"Failed to enrich with Prometheus data: {str(e)}")
            except Exception as e:
                # 其他錯誤也不應阻止報告生成
                logger.warning(f"Unexpected error during Prometheus enrichment: {str(e)}")
        
        # 生成報告
        report = await rag_service.generate_report(enriched_data)
        
        response = ReportResponse(
            status="success",
            report=report,
            monitoring_data=enriched_data
        )
        
        logger.info("Report generated successfully")
        return response
    
    except RAGServiceError:
        # RAGServiceError 會被全域錯誤處理器捕捉
        raise
    except HTTPException:
        # 重新拋出 HTTPException
        raise
    except Exception as e:
        # 捕捉所有其他未預期的錯誤
        logger.error(f"Unexpected error during report generation: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500, 
            detail="An internal error occurred. Please try again later."
        )

@app.get("/api/v1/metrics/{hostname}")
async def get_host_metrics(hostname: str):
    """
    獲取指定主機的實時監控指標
    """
    try:
        if not hostname:
            raise HTTPException(status_code=400, detail="Hostname cannot be empty")
            
        prometheus_service = rag_service.prometheus
        metrics = await prometheus_service.get_host_metrics(hostname)
        
        if not metrics:
            raise HTTPException(
                status_code=404, 
                detail=f"No metrics found for host: {hostname}"
            )
        
        return {
            "status": "success",
            "hostname": hostname,
            "metrics": metrics
        }
    except PrometheusError as e:
        logger.error(f"Prometheus error fetching metrics: {str(e)}")
        raise HTTPException(
            status_code=503, 
            detail=f"Monitoring service unavailable: {str(e)}"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error fetching metrics: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to fetch metrics")

@app.get("/api/v1/cache/info")
async def get_cache_info():
    """
    獲取快取狀態資訊
    
    返回 HyDE 和 Embedding 快取的命中率和使用情況
    """
    try:
        cache_info = rag_service.get_cache_info()
        
        # 計算命中率
        hyde_total = cache_info["hyde_cache"]["hits"] + cache_info["hyde_cache"]["misses"]
        hyde_hit_rate = (cache_info["hyde_cache"]["hits"] / hyde_total * 100) if hyde_total > 0 else 0
        
        embedding_total = cache_info["embedding_cache"]["hits"] + cache_info["embedding_cache"]["misses"]
        embedding_hit_rate = (cache_info["embedding_cache"]["hits"] / embedding_total * 100) if embedding_total > 0 else 0
        
        return {
            "status": "success",
            "cache_info": {
                "hyde_cache": {
                    **cache_info["hyde_cache"],
                    "hit_rate": f"{hyde_hit_rate:.2f}%"
                },
                "embedding_cache": {
                    **cache_info["embedding_cache"],
                    "hit_rate": f"{embedding_hit_rate:.2f}%"
                }
            }
        }
    except CacheError as e:
        logger.error(f"Cache error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Cache operation failed: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error getting cache info: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to get cache information")

@app.post("/api/v1/cache/clear")
async def clear_cache():
    """
    清除所有快取
    
    用於測試或需要強制更新時
    """
    try:
        rag_service.clear_cache()
        logger.info("Cache cleared successfully")
        return {
            "status": "success",
            "message": "All caches have been cleared"
        }
    except CacheError as e:
        logger.error(f"Cache error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to clear cache: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error clearing cache: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to clear cache")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)