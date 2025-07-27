import pytest
import os
from unittest.mock import Mock, patch, MagicMock
from langchain_core.language_models import BaseChatModel
from langchain_core.embeddings import Embeddings

from src.services.langchain.model_manager import ModelManager, model_manager
from src.config import settings


class TestModelManager:
    """Test cases for ModelManager"""

    @pytest.fixture
    def manager(self):
        """Create a ModelManager instance"""
        return ModelManager()

    @pytest.fixture
    def mock_chat_model(self):
        """Create a mock chat model"""
        mock = Mock(spec=BaseChatModel)
        mock.temperature = 0.7
        mock.max_output_tokens = 2048
        return mock

    @pytest.fixture
    def mock_embedding_model(self):
        """Create a mock embedding model"""
        return Mock(spec=Embeddings)

    def test_init(self, manager):
        """Test ModelManager initialization"""
        assert manager._flash_model is None
        assert manager._pro_model is None
        assert manager._embedding_model is None
        assert isinstance(manager._is_testing, bool)

    def test_get_api_key_testing(self, manager, monkeypatch):
        """Test API key retrieval in testing environment"""
        monkeypatch.setenv("TESTING", "true")
        manager._is_testing = True
        
        api_key = manager._get_api_key()
        assert api_key == "test-api-key"

    def test_get_api_key_production(self, manager, monkeypatch):
        """Test API key retrieval in production environment"""
        monkeypatch.setenv("TESTING", "false")
        manager._is_testing = False
        
        with patch.object(settings, 'gemini_api_key', 'real-api-key'):
            api_key = manager._get_api_key()
            assert api_key == "real-api-key"

    def test_flash_model_property_lazy_init(self, manager):
        """Test flash_model property lazy initialization"""
        with patch("src.services.langchain.model_manager.ChatGoogleGenerativeAI") as mock_chat:
            mock_instance = Mock()
            mock_chat.return_value = mock_instance
            
            # First access should create model
            model1 = manager.flash_model
            assert model1 is mock_instance
            
            # Verify correct initialization parameters
            mock_chat.assert_called_once_with(
                model=settings.gemini_flash_model,
                google_api_key=manager._get_api_key(),
                temperature=0.7,
                max_output_tokens=2048,
                convert_system_message_to_human=True
            )
            
            # Second access should return same instance
            model2 = manager.flash_model
            assert model1 is model2
            assert mock_chat.call_count == 1

    def test_pro_model_property_lazy_init(self, manager):
        """Test pro_model property lazy initialization"""
        with patch("src.services.langchain.model_manager.ChatGoogleGenerativeAI") as mock_chat:
            mock_instance = Mock()
            mock_chat.return_value = mock_instance
            
            # First access should create model
            model1 = manager.pro_model
            assert model1 is mock_instance
            
            # Verify correct initialization parameters
            mock_chat.assert_called_once_with(
                model=settings.gemini_pro_model,
                google_api_key=manager._get_api_key(),
                temperature=0.3,
                max_output_tokens=4096,
                convert_system_message_to_human=True
            )
            
            # Second access should return same instance
            model2 = manager.pro_model
            assert model1 is model2
            assert mock_chat.call_count == 1

    def test_embedding_model_property_lazy_init(self, manager):
        """Test embedding_model property lazy initialization"""
        with patch("src.services.langchain.model_manager.GoogleGenerativeAIEmbeddings") as mock_embeddings:
            mock_instance = Mock()
            mock_embeddings.return_value = mock_instance
            
            # First access should create model
            model1 = manager.embedding_model
            assert model1 is mock_instance
            
            # Verify correct initialization parameters
            mock_embeddings.assert_called_once_with(
                model="models/embedding-001",
                google_api_key=manager._get_api_key(),
                task_type="retrieval_document"
            )
            
            # Second access should return same instance
            model2 = manager.embedding_model
            assert model1 is model2
            assert mock_embeddings.call_count == 1

    def test_get_model_flash(self, manager, mock_chat_model):
        """Test get_model with flash type"""
        manager._flash_model = mock_chat_model
        
        model = manager.get_model("flash")
        assert model is mock_chat_model

    def test_get_model_pro(self, manager, mock_chat_model):
        """Test get_model with pro type"""
        manager._pro_model = mock_chat_model
        
        model = manager.get_model("pro")
        assert model is mock_chat_model

    def test_get_model_default(self, manager, mock_chat_model):
        """Test get_model with default type"""
        manager._flash_model = mock_chat_model
        
        model = manager.get_model()
        assert model is mock_chat_model

    def test_get_model_invalid_type(self, manager, mock_chat_model):
        """Test get_model with invalid type defaults to flash"""
        manager._flash_model = mock_chat_model
        
        model = manager.get_model("invalid")
        assert model is mock_chat_model

    def test_update_model_params_flash(self, manager, mock_chat_model):
        """Test updating flash model parameters"""
        manager._flash_model = mock_chat_model
        
        manager.update_model_params("flash", temperature=0.9, max_output_tokens=1024)
        
        assert mock_chat_model.temperature == 0.9
        assert mock_chat_model.max_output_tokens == 1024

    def test_update_model_params_pro(self, manager):
        """Test updating pro model parameters"""
        mock_model = Mock()
        mock_model.temperature = 0.3
        mock_model.max_output_tokens = 4096
        manager._pro_model = mock_model
        
        manager.update_model_params("pro", temperature=0.5, max_output_tokens=2048)
        
        assert mock_model.temperature == 0.5
        assert mock_model.max_output_tokens == 2048

    def test_update_model_params_non_existent_attribute(self, manager, mock_chat_model):
        """Test updating model with non-existent parameters"""
        manager._flash_model = mock_chat_model
        
        # Should not raise error for non-existent attributes
        manager.update_model_params("flash", non_existent_param=123)
        
        # Original attributes should remain unchanged
        assert mock_chat_model.temperature == 0.7

    def test_update_model_params_mixed_attributes(self, manager):
        """Test updating model with mixed valid and invalid parameters"""
        mock_model = Mock()
        mock_model.temperature = 0.7
        # Don't set invalid_param attribute
        manager._flash_model = mock_model
        
        manager.update_model_params(
            "flash", 
            temperature=0.8,
            invalid_param="value"
        )
        
        # Valid parameter should be updated
        assert mock_model.temperature == 0.8
        # Invalid parameter should be ignored

    def test_model_manager_singleton(self):
        """Test that model_manager is a singleton instance"""
        assert isinstance(model_manager, ModelManager)
        assert model_manager._flash_model is None
        assert model_manager._pro_model is None
        assert model_manager._embedding_model is None

    def test_testing_environment_detection(self, monkeypatch):
        """Test correct detection of testing environment"""
        # Test with TESTING=true
        monkeypatch.setenv("TESTING", "true")
        manager1 = ModelManager()
        assert manager1._is_testing is True
        
        # Test with TESTING=false
        monkeypatch.setenv("TESTING", "false")
        manager2 = ModelManager()
        assert manager2._is_testing is False
        
        # Test with TESTING not set
        monkeypatch.delenv("TESTING", raising=False)
        manager3 = ModelManager()
        assert manager3._is_testing is False