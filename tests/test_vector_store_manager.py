import pytest
import sys
from unittest.mock import Mock, patch, MagicMock

# 關鍵修正：明確地匯入 'module' 本身
import src.services.langchain.vector_store_manager as vector_store_manager_module
from src.services.langchain.vector_store_manager import VectorStoreManager, vector_store_manager


class TestVectorStoreManager:
    @pytest.fixture
    def manager(self):
        # 每次都返回一個新的、乾淨的實例
        manager_instance = VectorStoreManager()
        manager_instance._opensearch_client = None
        manager_instance._vector_store = None
        return manager_instance

    def test_opensearch_client_property(self, manager):
        # Create a mock instance
        mock_client = Mock()
        
        # Directly set the internal state
        manager._opensearch_client = mock_client
        
        # Verify the property returns our mock
        client = manager.opensearch_client
        assert client is mock_client

    def test_vector_store_property(self, manager):
        # Create a mock instance
        mock_store = Mock()
        
        # Directly set the internal state
        manager._vector_store = mock_store
        
        # Verify the property returns our mock
        store = manager.vector_store
        assert store is mock_store

    def test_get_retriever_with_hyde(self, manager):
        # 模擬 manager 的 as_retriever 方法
        manager.as_retriever = Mock(return_value=Mock())

        # 關鍵修正：patch model_manager 的匯入路徑
        with patch('src.services.langchain.vector_store_manager.model_manager') as mock_model_manager:
            
            # 使用 MagicMock 來模擬一個不存在的模組
            mock_hyde_retriever_class = MagicMock()
            mock_hyde_instance = Mock()
            mock_hyde_retriever_class.return_value = mock_hyde_instance
            
            # 使用 patch.dict 來動態模擬 sys.modules 中的 langchain.retrievers
            with patch.dict(sys.modules, {'langchain.retrievers': MagicMock(HyDERetriever=mock_hyde_retriever_class)}):
                retriever = manager.get_retriever_with_hyde(Mock())
                assert retriever is mock_hyde_instance
                mock_hyde_retriever_class.assert_called_once()
                # 驗證 HyDERetriever 是用 model_manager.flash_model 初始化的
                mock_hyde_retriever_class.assert_called_with(
                    base_retriever=manager.as_retriever(),
                    llm=mock_model_manager.flash_model,
                    prompt=mock_hyde_retriever_class.call_args[1]['prompt']
                )

    def test_vector_store_manager_singleton(self):
        assert isinstance(vector_store_manager, VectorStoreManager)

    def test_opensearch_client_lazy_initialization(self, manager):
        """Test that opensearch_client creates a client on first access"""
        # Initially should be None
        assert manager._opensearch_client is None
        
        # Patch at the correct location
        with patch('src.services.langchain.vector_store_manager.OpenSearch') as mock_opensearch:
            mock_instance = Mock()
            mock_opensearch.return_value = mock_instance
            
            # Access the property
            client = manager.opensearch_client
            
            # Should have created the client
            assert client is mock_instance
            mock_opensearch.assert_called_once()

    def test_vector_store_lazy_initialization(self, manager):
        """Test that vector_store creates a store on first access"""
        # Initially should be None
        assert manager._vector_store is None
        
        # Patch at the correct location
        with patch('src.services.langchain.vector_store_manager.OpenSearchVectorSearch') as mock_vectorstore, \
             patch('src.services.langchain.vector_store_manager.model_manager') as mock_model_manager:
            
            mock_instance = Mock()
            mock_vectorstore.return_value = mock_instance
            
            # Access the property
            store = manager.vector_store
            
            # Should have created the store
            assert store is mock_instance
            mock_vectorstore.assert_called_once()
