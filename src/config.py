import os
from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # API Configuration
    api_title: str = "AIOps 智慧維運報告 RAG 系統"
    api_version: str = "1.0.0"
    
    # Gemini Configuration
    gemini_api_key: str = ""
    gemini_flash_model: str = "gemini-1.5-flash"
    gemini_pro_model: str = "gemini-1.5-pro"
    
    # OpenSearch Configuration
    opensearch_host: str = "localhost"
    opensearch_port: int = 9200
    opensearch_index: str = "aiops-knowledge-base"
    opensearch_embedding_dim: int = 768
    
    # Prometheus Configuration
    prometheus_host: str = "localhost"
    prometheus_port: int = 9090
    
    # RAG Configuration
    top_k_results: int = 5
    max_summary_length: int = 150
    
    class Config:
        env_file = ".env"
        case_sensitive = False

settings = Settings()