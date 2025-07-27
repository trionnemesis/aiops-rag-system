import pytest
from unittest.mock import Mock, patch

# 關鍵修正：明確地匯入 'module' 本身，而不是被 __init__.py 導出的單例
import src.services.langchain.model_manager as model_manager_module
from src.services.langchain.model_manager import ModelManager, model_manager


class TestModelManager:
    @pytest.fixture
    def manager(self, monkeypatch):
        monkeypatch.setenv("TESTING", "true")
        # 清除快取，確保每個測試都是獨立的
        manager_instance = ModelManager()
        manager_instance._flash_model = None
        manager_instance._pro_model = None
        manager_instance._embedding_model = None
        return manager_instance

    def test_init(self, manager):
        assert manager._flash_model is None
        assert manager._pro_model is None
        assert manager._embedding_model is None
        assert manager._is_testing is True

    # 關鍵修正：使用 patch 直接作用在模組導入路徑上
    def test_flash_model_property_lazy_init(self, manager):
        with patch('src.services.langchain.model_manager.ChatGoogleGenerativeAI') as mock_chat:
            mock_instance = Mock()
            mock_chat.return_value = mock_instance
            
            model1 = manager.flash_model
            assert model1 is mock_instance
            mock_chat.assert_called_once()
            
            model2 = manager.flash_model
            assert model1 is model2
            assert mock_chat.call_count == 1

    def test_pro_model_property_lazy_init(self, manager):
        with patch('src.services.langchain.model_manager.ChatGoogleGenerativeAI') as mock_chat:
            mock_instance = Mock()
            mock_chat.return_value = mock_instance
            
            model1 = manager.pro_model
            assert model1 is mock_instance
            mock_chat.assert_called_once()
            
            model2 = manager.pro_model
            assert model1 is model2
            assert mock_chat.call_count == 1

    def test_embedding_model_property_lazy_init(self, manager):
        with patch('src.services.langchain.model_manager.GoogleGenerativeAIEmbeddings') as mock_embeddings:
            mock_instance = Mock()
            mock_embeddings.return_value = mock_instance
            
            model1 = manager.embedding_model
            assert model1 is mock_instance
            mock_embeddings.assert_called_once()
            
            model2 = manager.embedding_model
            assert model1 is model2
            assert mock_embeddings.call_count == 1

    def test_model_manager_singleton(self):
        # 確保全域實例的行為符合預期
        with patch('src.services.langchain.model_manager.ChatGoogleGenerativeAI'), \
             patch('src.services.langchain.model_manager.GoogleGenerativeAIEmbeddings'):
            
            assert isinstance(model_manager, ModelManager)
            # 存取屬性應觸發初始化
            assert model_manager.flash_model is not None
