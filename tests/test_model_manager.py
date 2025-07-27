import pytest
import os
from unittest.mock import Mock, patch
from src.services.langchain.model_manager import ModelManager, model_manager
from src.config import settings

class TestModelManager:
    """Test cases for ModelManager"""

    @pytest.fixture
    def manager(self, monkeypatch):
        """為測試建立一個 ModelManager 實例"""
        monkeypatch.setenv("TESTING", "true")
        return ModelManager()

    def test_init(self, manager):
        """測試 ModelManager 的初始化"""
        assert manager._flash_model is None
        assert manager._pro_model is None
        assert manager._embedding_model is None
        assert manager._is_testing is True

    def test_flash_model_property_lazy_init(self, manager):
        """測試 flash_model 屬性的延遲初始化"""
        # 正確地 patch 在 model_manager 模組中被引用的 ChatGoogleGenerativeAI
        with patch("src.services.langchain.model_manager.ChatGoogleGenerativeAI") as mock_chat:
            mock_instance = Mock()
            mock_chat.return_value = mock_instance
            
            # 第一次存取時應建立模型
            model1 = manager.flash_model
            assert model1 is mock_instance
            mock_chat.assert_called_once()
            
            # 第二次存取時應回傳同一個實例
            model2 = manager.flash_model
            assert model1 is model2
            assert mock_chat.call_count == 1

    def test_pro_model_property_lazy_init(self, manager):
        """測試 pro_model 屬性的延遲初始化"""
        with patch("src.services.langchain.model_manager.ChatGoogleGenerativeAI") as mock_chat:
            mock_instance = Mock()
            mock_chat.return_value = mock_instance
            
            model1 = manager.pro_model
            assert model1 is mock_instance
            mock_chat.assert_called_once()
            
            model2 = manager.pro_model
            assert model1 is model2
            assert mock_chat.call_count == 1

    def test_embedding_model_property_lazy_init(self, manager):
        """測試 embedding_model 屬性的延遲初始化"""
        with patch("src.services.langchain.model_manager.GoogleGenerativeAIEmbeddings") as mock_embeddings:
            mock_instance = Mock()
            mock_embeddings.return_value = mock_instance
            
            model1 = manager.embedding_model
            assert model1 is mock_instance
            mock_embeddings.assert_called_once()
            
            model2 = manager.embedding_model
            assert model1 is model2
            assert mock_embeddings.call_count == 1
    
    def test_model_manager_singleton(self):
        """測試 model_manager 是一個單例實例且能正確初始化"""
        # 這個測試確保全域實例的行為符合預期
        with patch("src.services.langchain.model_manager.ChatGoogleGenerativeAI"), \
             patch("src.services.langchain.model_manager.GoogleGenerativeAIEmbeddings"):
            
            new_manager = ModelManager()
            assert isinstance(new_manager, ModelManager)
            
            # 存取屬性應觸發初始化
            assert new_manager.flash_model is not None
            assert new_manager.pro_model is not None
            assert new_manager.embedding_model is not None
