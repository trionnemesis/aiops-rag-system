"""
LangChain 模型管理器
統一管理 Gemini 模型實例和相關配置
"""
import os
from typing import Optional, Dict, Any
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.embeddings import Embeddings
from src.config import settings


class ModelManager:
    """統一的模型管理器"""
    
    def __init__(self):
        self._flash_model: Optional[BaseChatModel] = None
        self._pro_model: Optional[BaseChatModel] = None
        self._embedding_model: Optional[Embeddings] = None
        self._is_testing = os.environ.get('TESTING', '').lower() == 'true'
        
    def _get_api_key(self) -> str:
        """獲取 API Key，測試環境下使用假的 key"""
        if self._is_testing:
            return "test-api-key"
        return settings.gemini_api_key
        
    @property
    def flash_model(self) -> BaseChatModel:
        """獲取 Gemini Flash 模型實例（用於快速任務）"""
        if self._flash_model is None:
            self._flash_model = ChatGoogleGenerativeAI(
                model=settings.gemini_flash_model,
                google_api_key=self._get_api_key(),
                temperature=0.7,
                max_output_tokens=2048,
                convert_system_message_to_human=True
            )
        return self._flash_model
    
    @property
    def pro_model(self) -> BaseChatModel:
        """獲取 Gemini Pro 模型實例（用於複雜任務）"""
        if self._pro_model is None:
            self._pro_model = ChatGoogleGenerativeAI(
                model=settings.gemini_pro_model,
                google_api_key=self._get_api_key(),
                temperature=0.3,
                max_output_tokens=4096,
                convert_system_message_to_human=True
            )
        return self._pro_model
    
    @property
    def embedding_model(self) -> Embeddings:
        """獲取嵌入模型實例"""
        if self._embedding_model is None:
            self._embedding_model = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001",
                google_api_key=self._get_api_key(),
                task_type="retrieval_document"
            )
        return self._embedding_model
    
    def get_model(self, model_type: str = "flash") -> BaseChatModel:
        """根據類型獲取模型
        
        Args:
            model_type: "flash" 或 "pro"
            
        Returns:
            對應的模型實例
        """
        if model_type == "pro":
            return self.pro_model
        return self.flash_model
    
    def update_model_params(self, model_type: str, **kwargs) -> None:
        """動態更新模型參數
        
        Args:
            model_type: "flash" 或 "pro"
            **kwargs: 要更新的參數
        """
        model = self.get_model(model_type)
        for key, value in kwargs.items():
            if hasattr(model, key):
                setattr(model, key, value)


# 全局單例實例
model_manager = ModelManager()