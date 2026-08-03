import os
from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    """應用程式設定類別，所有設定值皆從環境變數讀取"""
    
    # API Configuration
    api_title: str = "AIOps 智慧維運報告 RAG 系統"
    api_version: str = "1.0.0"
    api_port: int = 8000
    
    # Gemini Configuration
    gemini_api_key: str
    # 模型 ID 可由環境變數 GEMINI_FLASH_MODEL / GEMINI_PRO_MODEL 覆寫，
    # 避免 Google 退役舊模型時需要改動程式碼。
    gemini_flash_model: str = "gemini-3.5-flash"
    gemini_pro_model: str = "gemini-3.1-pro"
    
    # OpenSearch Configuration
    opensearch_host: str = "localhost"
    opensearch_port: int = 9200
    opensearch_index: str = "aiops-knowledge-base"
    opensearch_embedding_dim: int = 768
    opensearch_user: Optional[str] = None
    opensearch_password: Optional[str] = None
    
    # Prometheus Configuration
    prometheus_host: str = "localhost"
    prometheus_port: int = 9090
    
    # Redis Configuration
    redis_url: str = "redis://localhost:6379"
    
    # RAG Configuration
    top_k_results: int = 5
    max_summary_length: int = 150
    
    # Observability Configuration
    environment: str = "development"
    log_level: str = "INFO"
    json_logs: bool = True
    log_file: Optional[str] = None
    
    # Tracing Configuration
    jaeger_endpoint: str = "localhost:6831"
    otlp_endpoint: Optional[str] = None
    trace_console: bool = False
    
    # Metrics Configuration
    metrics_port: int = 8000
    
    # Development Configuration
    testing: bool = False

    @property
    def google_api_key(self) -> str:
        """Gemini API key 的別名。

        opensearch_service / knn_search_service / embedding_config 都是以
        `settings.google_api_key` 取用（對應 langchain-google-genai 的參數名），
        但 Settings 上只定義了 gemini_api_key，會拋 AttributeError。
        這裡補上別名，維持單一真實來源。
        """
        return self.gemini_api_key


    class Config:
        env_file = ".env"
        case_sensitive = False
        # 使用環境變數前綴，例如 APP_API_TITLE
        # env_prefix = "APP_"

# 建立全域設定實例
settings = Settings()

# 輸出設定載入狀態（僅在開發環境）
if settings.environment == "development" and not settings.testing:
    print(f"[Config] Loaded settings for environment: {settings.environment}")
    print(f"[Config] OpenSearch: {settings.opensearch_host}:{settings.opensearch_port}")
    print(f"[Config] Redis: {settings.redis_url}")
    print(f"[Config] Prometheus: {settings.prometheus_host}:{settings.prometheus_port}")