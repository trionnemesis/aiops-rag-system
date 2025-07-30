"""
Embedding 模型配置
確保整個系統使用一致的 embedding 模型和維度
"""

from typing import Dict, Any
from dataclasses import dataclass


@dataclass
class EmbeddingConfig:
    """Embedding 配置"""
    model_name: str
    dimension: int
    model_type: str  # "google", "openai", "huggingface"
    chunk_size: int  # 文字分塊大小
    chunk_overlap: int  # 分塊重疊大小


# 支援的 Embedding 模型配置
EMBEDDING_MODELS: Dict[str, EmbeddingConfig] = {
    # Google Gemini Embeddings
    "gemini-embedding-001": EmbeddingConfig(
        model_name="models/embedding-001",
        dimension=768,
        model_type="google",
        chunk_size=1000,
        chunk_overlap=200
    ),
    
    # OpenAI Embeddings
    "text-embedding-ada-002": EmbeddingConfig(
        model_name="text-embedding-ada-002",
        dimension=1536,
        model_type="openai",
        chunk_size=1000,
        chunk_overlap=200
    ),
    
    "text-embedding-3-small": EmbeddingConfig(
        model_name="text-embedding-3-small",
        dimension=1536,
        model_type="openai",
        chunk_size=1000,
        chunk_overlap=200
    ),
    
    "text-embedding-3-large": EmbeddingConfig(
        model_name="text-embedding-3-large",
        dimension=3072,
        model_type="openai",
        chunk_size=1000,
        chunk_overlap=200
    ),
    
    # Hugging Face Embeddings
    "sentence-transformers/all-MiniLM-L6-v2": EmbeddingConfig(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        dimension=384,
        model_type="huggingface",
        chunk_size=512,
        chunk_overlap=50
    ),
    
    "sentence-transformers/all-mpnet-base-v2": EmbeddingConfig(
        model_name="sentence-transformers/all-mpnet-base-v2",
        dimension=768,
        model_type="huggingface",
        chunk_size=512,
        chunk_overlap=50
    ),
}

# 預設使用的模型
DEFAULT_EMBEDDING_MODEL = "gemini-embedding-001"


def get_embedding_config(model_key: str = None) -> EmbeddingConfig:
    """
    取得 Embedding 配置
    
    Args:
        model_key: 模型鍵值，如果為 None 則使用預設模型
        
    Returns:
        Embedding 配置
        
    Raises:
        ValueError: 如果模型不存在
    """
    model_key = model_key or DEFAULT_EMBEDDING_MODEL
    
    if model_key not in EMBEDDING_MODELS:
        raise ValueError(
            f"不支援的 Embedding 模型: {model_key}。"
            f"支援的模型: {list(EMBEDDING_MODELS.keys())}"
        )
    
    return EMBEDDING_MODELS[model_key]


def validate_embedding_dimension(embedding: list, expected_model: str = None) -> bool:
    """
    驗證 embedding 向量維度是否正確
    
    Args:
        embedding: embedding 向量
        expected_model: 預期的模型名稱
        
    Returns:
        是否維度正確
    """
    config = get_embedding_config(expected_model)
    actual_dim = len(embedding)
    expected_dim = config.dimension
    
    if actual_dim != expected_dim:
        raise ValueError(
            f"Embedding 維度不符: 預期 {expected_dim} (模型: {config.model_name}), "
            f"實際 {actual_dim}"
        )
    
    return True


def get_embedding_model_instance(model_key: str = None):
    """
    取得 Embedding 模型實例
    
    Args:
        model_key: 模型鍵值
        
    Returns:
        Embedding 模型實例
    """
    from src.config import settings
    
    config = get_embedding_config(model_key)
    
    if config.model_type == "google":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        return GoogleGenerativeAIEmbeddings(
            model=config.model_name,
            google_api_key=settings.google_api_key
        )
    
    elif config.model_type == "openai":
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(
            model=config.model_name,
            openai_api_key=settings.openai_api_key
        )
    
    elif config.model_type == "huggingface":
        from langchain_community.embeddings import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(
            model_name=config.model_name,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
    
    else:
        raise ValueError(f"不支援的模型類型: {config.model_type}")