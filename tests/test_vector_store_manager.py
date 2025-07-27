import pytest
from unittest.mock import Mock, patch, MagicMock
from src.services.langchain.vector_store_manager import VectorStoreManager, vector_store_manager

class TestVectorStoreManager:
    @pytest.fixture
    def manager(self):
        return VectorStoreManager()

    # 關鍵修正：patch 正確的模組路徑
    def test_opensearch_client_property(self, manager):
        with patch('src.services.langchain.vector_store_manager.OpenSearch') as mock_opensearch:
            mock_instance = Mock()
            mock_opensearch.return_value = mock_instance
            client = manager.opensearch_client
            assert client is mock_instance
            mock_opensearch.assert_called_once()

    def test_vector_store_property(self, manager):
        with patch('src.services.langchain.vector_store_manager.OpenSearchVectorSearch') as mock_vectorstore, \
             patch('src.services.langchain.vector_store_manager.model_manager'):
            
            mock_instance = Mock()
            mock_vectorstore.return_value = mock_instance
            store = manager.vector_store
            assert store is mock_instance
            mock_vectorstore.assert_called_once()
            
    def test_get_retriever_with_hyde(self, manager):
        # 修正：使用 MagicMock 來模擬模組導入
        with patch.object(manager, 'as_retriever', return_value=Mock()) as mock_as_retriever, \
             patch('src.services.langchain.vector_store_manager.model_manager'):
            
            # 動態 patch import
            mock_hyde_retriever_class = MagicMock()
            mock_hyde_instance = Mock()
            mock_hyde_retriever_class.return_value = mock_hyde_instance
            
            with patch.dict('sys.modules', {'langchain.retrievers': MagicMock(HyDERetriever=mock_hyde_retriever_class)}):
                retriever = manager.get_retriever_with_hyde(Mock())
                assert retriever is mock_hyde_instance
                mock_hyde_retriever_class.assert_called_once()
    
    def test_vector_store_manager_singleton(self):
        assert isinstance(vector_store_manager, VectorStoreManager)
        # 修正：單例在測試期間可能已經被初始化，所以不檢查 is None
        # 我們只確保它是一個正確的實例
