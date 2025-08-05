"""主應用程式入口點

初始化可觀測性功能並啟動 FastAPI 應用
"""

import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.observability import setup_logging, setup_tracing, setup_metrics

# 從環境變數讀取配置
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
JSON_LOGS = os.getenv("JSON_LOGS", "true").lower() == "true"
LOG_FILE = os.getenv("LOG_FILE", None)

JAEGER_ENDPOINT = os.getenv("JAEGER_ENDPOINT", "localhost:6831")
OTLP_ENDPOINT = os.getenv("OTLP_ENDPOINT", None)
TRACE_CONSOLE = os.getenv("TRACE_CONSOLE", "false").lower() == "true"

METRICS_PORT = int(os.getenv("METRICS_PORT", "8000"))

# 初始化可觀測性
def init_observability():
    """初始化所有可觀測性功能"""
    # 設定結構化日誌
    setup_logging(
        level=LOG_LEVEL,
        json_logs=JSON_LOGS,
        log_file=LOG_FILE
    )
    
    # 設定分散式追蹤
    setup_tracing(
        service_name="langgraph-rag",
        service_version="1.0.0",
        jaeger_endpoint=JAEGER_ENDPOINT if JAEGER_ENDPOINT != "none" else None,
        otlp_endpoint=OTLP_ENDPOINT,
        console_export=TRACE_CONSOLE
    )
    
    # 設定 Prometheus 指標收集
    setup_metrics(port=METRICS_PORT)

# 創建 FastAPI 應用
def create_app() -> FastAPI:
    """創建並配置 FastAPI 應用"""
    # 初始化可觀測性
    init_observability()
    
    # 創建應用
    app = FastAPI(
        title="LangGraph RAG API",
        description="具備完整可觀測性的 RAG 系統",
        version="1.0.0"
    )
    
    # 添加 CORS 中間件
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # 註冊路由
    app.include_router(router, prefix="/api/v1")
    
    # 根路徑重定向到文檔
    @app.get("/")
    def root():
        return {"message": "Welcome to LangGraph RAG API", "docs": "/docs"}
    
    return app

# 創建應用實例
app = create_app()

if __name__ == "__main__":
    import uvicorn
    
    # 啟動服務器
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8080,
        reload=os.getenv("ENV", "development") == "development"
    )