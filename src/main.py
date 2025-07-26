from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging
from src.models.schemas import ReportRequest, ReportResponse, InsightReport
from src.services.rag_service import RAGService
from src.services.opensearch_service import OpenSearchService
from src.config import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Lifespan context manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting up AIOps RAG System...")
    opensearch = OpenSearchService()
    await opensearch.create_index()
    logger.info("OpenSearch index initialized")
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

@app.get("/")
async def root():
    return {
        "message": "AIOps 智慧維運報告 RAG 系統",
        "version": settings.api_version,
        "docs": "/docs"
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/api/v1/generate_report", response_model=ReportResponse)
async def generate_report(request: ReportRequest):
    """
    生成智慧維運報告
    
    接收監控數據，透過 RAG 架構生成深度分析報告
    """
    try:
        logger.info(f"Received monitoring data: {request.monitoring_data}")
        
        # 如果有主機名稱，可以從 Prometheus 獲取額外數據
        if "主機" in request.monitoring_data:
            hostname = request.monitoring_data["主機"]
            enriched_data = await rag_service.enrich_with_prometheus(
                hostname, 
                request.monitoring_data
            )
        else:
            enriched_data = request.monitoring_data
        
        # 生成報告
        report = await rag_service.generate_report(enriched_data)
        
        response = ReportResponse(
            status="success",
            report=report,
            monitoring_data=enriched_data
        )
        
        logger.info("Report generated successfully")
        return response
        
    except Exception as e:
        logger.error(f"Error generating report: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/metrics/{hostname}")
async def get_host_metrics(hostname: str):
    """
    獲取指定主機的實時監控指標
    """
    try:
        prometheus_service = rag_service.prometheus
        metrics = await prometheus_service.get_host_metrics(hostname)
        return {
            "status": "success",
            "hostname": hostname,
            "metrics": metrics
        }
    except Exception as e:
        logger.error(f"Error fetching metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)