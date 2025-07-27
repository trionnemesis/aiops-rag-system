import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from src.services.langchain.vector_store_manager import VectorStoreManager, vector_store_manager
from src.config import settings


class TestVectorStoreManager:
    """Test cases for VectorStoreManager"""

    @pytest.fixture
    def mock_opensearch_client(self):
        """Create a mock OpenSearch client"""
        mock_client = Mock()
        mock_client.indices = Mock()
        mock_client.indices.exists = Mock(return_value=False)
        mock_client.indices.create = Mock()
        mock_client.indices.delete = Mock()
        return mock_client

    @pytest.fixture
    def mock_vector_store(self):
        """Create a mock VectorStore"""
        mock_store = Mock()
        mock_store.aadd_documents = AsyncMock(return_value=["doc1", "doc2"])
        mock_store.aadd_texts = AsyncMock(return_value=["text1", "text2"])
        mock_store.asimilarity_search = AsyncMock(return_value=[
            Document(page_content="Test content", metadata={"id": "1"})
        ])
        mock_store.asimilarity_search_with_score = AsyncMock(return_value=[
            (Document(page_content="Test content", metadata={"id": "1"}), 0.9)
        ])
        mock_store.as_retriever = Mock(return_value=Mock())
        return mock_store

    @pytest.fixture
    def manager(self):
        """Create a VectorStoreManager instance"""
        return VectorStoreManager()

    def test_init(self, manager):
        """Test VectorStoreManager initialization"""
        assert manager._vector_store is None
        assert manager._opensearch_client is None

    def test_opensearch_client_property(self, manager):
        """Test opensearch_client property lazy initialization"""
        with patch("src.services.langchain.vector_store_manager.OpenSearch") as mock_opensearch:
            mock_opensearch.return_value = Mock()
            
            # First access should create client
            client1 = manager.opensearch_client
            assert client1 is not None
            mock_opensearch.assert_called_once_with(
                hosts=[{
                    'host': settings.opensearch_host, 
                    'port': settings.opensearch_port
                }],
                use_ssl=False,
                verify_certs=False
            )
            
            # Second access should return same client
            client2 = manager.opensearch_client
            assert client1 is client2
            assert mock_opensearch.call_count == 1

    def test_vector_store_property(self, manager):
        """Test vector_store property lazy initialization"""
        with patch("src.services.langchain.vector_store_manager.OpenSearchVectorSearch") as mock_vectorstore:
            with patch("src.services.langchain.vector_store_manager.model_manager") as mock_model_manager:
                mock_vectorstore.return_value = Mock()
                mock_model_manager.embedding_model = Mock()
                
                # First access should create vector store
                store1 = manager.vector_store
                assert store1 is not None
                mock_vectorstore.assert_called_once()
                
                # Verify correct parameters
                call_kwargs = mock_vectorstore.call_args[1]
                assert call_kwargs["index_name"] == settings.opensearch_index
                assert call_kwargs["engine"] == "nmslib"
                assert call_kwargs["space_type"] == "l2"
                assert call_kwargs["ef_construction"] == 128
                assert call_kwargs["m"] == 24
                
                # Second access should return same store
                store2 = manager.vector_store
                assert store1 is store2
                assert mock_vectorstore.call_count == 1

    @pytest.mark.asyncio
    async def test_create_index_not_exists(self, manager, mock_opensearch_client):
        """Test creating index when it doesn't exist"""
        manager._opensearch_client = mock_opensearch_client
        mock_opensearch_client.indices.exists.return_value = False
        
        await manager.create_index()
        
        # Verify index existence was checked
        mock_opensearch_client.indices.exists.assert_called_once_with(index=settings.opensearch_index)
        
        # Verify index was created
        mock_opensearch_client.indices.create.assert_called_once()
        call_args = mock_opensearch_client.indices.create.call_args
        assert call_args[1]["index"] == settings.opensearch_index
        
        # Verify index body structure
        index_body = call_args[1]["body"]
        assert index_body["settings"]["index"]["knn"] is True
        assert index_body["mappings"]["properties"]["vector_field"]["dimension"] == settings.opensearch_embedding_dim

    @pytest.mark.asyncio
    async def test_create_index_already_exists(self, manager, mock_opensearch_client):
        """Test creating index when it already exists"""
        manager._opensearch_client = mock_opensearch_client
        mock_opensearch_client.indices.exists.return_value = True
        
        await manager.create_index()
        
        # Verify index existence was checked
        mock_opensearch_client.indices.exists.assert_called_once_with(index=settings.opensearch_index)
        
        # Verify index was NOT created
        mock_opensearch_client.indices.create.assert_not_called()

    @pytest.mark.asyncio
    async def test_add_documents(self, manager, mock_vector_store):
        """Test adding documents to vector store"""
        manager._vector_store = mock_vector_store
        
        documents = [
            Document(page_content="Doc 1", metadata={"id": "1"}),
            Document(page_content="Doc 2", metadata={"id": "2"})
        ]
        
        result = await manager.add_documents(documents)
        
        assert result == ["doc1", "doc2"]
        mock_vector_store.aadd_documents.assert_called_once_with(documents)

    @pytest.mark.asyncio
    async def test_add_texts_with_metadata(self, manager, mock_vector_store):
        """Test adding texts with metadata to vector store"""
        manager._vector_store = mock_vector_store
        
        texts = ["Text 1", "Text 2"]
        metadatas = [{"id": "1"}, {"id": "2"}]
        
        result = await manager.add_texts(texts, metadatas)
        
        assert result == ["text1", "text2"]
        mock_vector_store.aadd_texts.assert_called_once_with(texts, metadatas)

    @pytest.mark.asyncio
    async def test_add_texts_without_metadata(self, manager, mock_vector_store):
        """Test adding texts without metadata to vector store"""
        manager._vector_store = mock_vector_store
        
        texts = ["Text 1", "Text 2"]
        
        result = await manager.add_texts(texts)
        
        assert result == ["text1", "text2"]
        mock_vector_store.aadd_texts.assert_called_once_with(texts, None)

    def test_as_retriever_default_k(self, manager, mock_vector_store):
        """Test getting retriever with default k value"""
        manager._vector_store = mock_vector_store
        mock_retriever = Mock()
        mock_vector_store.as_retriever.return_value = mock_retriever
        
        retriever = manager.as_retriever()
        
        mock_vector_store.as_retriever.assert_called_once_with(
            search_type="similarity",
            search_kwargs={"k": settings.top_k_results}
        )
        assert retriever == mock_retriever

    def test_as_retriever_custom_k(self, manager, mock_vector_store):
        """Test getting retriever with custom k value"""
        manager._vector_store = mock_vector_store
        mock_retriever = Mock()
        mock_vector_store.as_retriever.return_value = mock_retriever
        
        retriever = manager.as_retriever(search_kwargs={"k": 10})
        
        mock_vector_store.as_retriever.assert_called_once_with(
            search_type="similarity",
            search_kwargs={"k": 10}
        )
        assert retriever == mock_retriever

    def test_as_retriever_with_additional_kwargs(self, manager, mock_vector_store):
        """Test getting retriever with additional search kwargs"""
        manager._vector_store = mock_vector_store
        mock_retriever = Mock()
        mock_vector_store.as_retriever.return_value = mock_retriever
        
        retriever = manager.as_retriever(search_kwargs={"k": 5, "filter": {"type": "alert"}})
        
        mock_vector_store.as_retriever.assert_called_once_with(
            search_type="similarity",
            search_kwargs={"k": 5, "filter": {"type": "alert"}}
        )

    @pytest.mark.asyncio
    async def test_similarity_search_default_k(self, manager, mock_vector_store):
        """Test similarity search with default k value"""
        manager._vector_store = mock_vector_store
        
        result = await manager.similarity_search("test query")
        
        assert len(result) == 1
        assert result[0].page_content == "Test content"
        mock_vector_store.asimilarity_search.assert_called_once_with(
            "test query", 
            k=settings.top_k_results,
            filter=None
        )

    @pytest.mark.asyncio
    async def test_similarity_search_custom_k(self, manager, mock_vector_store):
        """Test similarity search with custom k value"""
        manager._vector_store = mock_vector_store
        
        result = await manager.similarity_search("test query", k=10)
        
        mock_vector_store.asimilarity_search.assert_called_once_with(
            "test query", 
            k=10,
            filter=None
        )

    @pytest.mark.asyncio
    async def test_similarity_search_with_filter(self, manager, mock_vector_store):
        """Test similarity search with filter"""
        manager._vector_store = mock_vector_store
        filter_dict = {"type": "alert", "severity": "high"}
        
        result = await manager.similarity_search("test query", filter=filter_dict)
        
        mock_vector_store.asimilarity_search.assert_called_once_with(
            "test query", 
            k=settings.top_k_results,
            filter=filter_dict
        )

    @pytest.mark.asyncio
    async def test_similarity_search_with_score_default_k(self, manager, mock_vector_store):
        """Test similarity search with score using default k"""
        manager._vector_store = mock_vector_store
        
        result = await manager.similarity_search_with_score("test query")
        
        assert len(result) == 1
        assert result[0][0].page_content == "Test content"
        assert result[0][1] == 0.9
        mock_vector_store.asimilarity_search_with_score.assert_called_once_with(
            "test query", 
            k=settings.top_k_results,
            filter=None
        )

    @pytest.mark.asyncio
    async def test_similarity_search_with_score_custom_params(self, manager, mock_vector_store):
        """Test similarity search with score using custom parameters"""
        manager._vector_store = mock_vector_store
        filter_dict = {"type": "alert"}
        
        result = await manager.similarity_search_with_score("test query", k=5, filter=filter_dict)
        
        mock_vector_store.asimilarity_search_with_score.assert_called_once_with(
            "test query", 
            k=5,
            filter=filter_dict
        )

    @pytest.mark.asyncio
    async def test_delete_index_exists(self, manager, mock_opensearch_client):
        """Test deleting index when it exists"""
        manager._opensearch_client = mock_opensearch_client
        mock_opensearch_client.indices.exists.return_value = True
        
        await manager.delete_index()
        
        mock_opensearch_client.indices.exists.assert_called_once_with(index=settings.opensearch_index)
        mock_opensearch_client.indices.delete.assert_called_once_with(index=settings.opensearch_index)

    @pytest.mark.asyncio
    async def test_delete_index_not_exists(self, manager, mock_opensearch_client):
        """Test deleting index when it doesn't exist"""
        manager._opensearch_client = mock_opensearch_client
        mock_opensearch_client.indices.exists.return_value = False
        
        await manager.delete_index()
        
        mock_opensearch_client.indices.exists.assert_called_once_with(index=settings.opensearch_index)
        mock_opensearch_client.indices.delete.assert_not_called()

    def test_get_retriever_with_hyde(self, manager, mock_vector_store):
        """Test getting retriever with HyDE"""
        manager._vector_store = mock_vector_store
        mock_retriever = Mock()
        mock_vector_store.as_retriever.return_value = mock_retriever
        
        mock_prompt_template = Mock()
        
        with patch("src.services.langchain.vector_store_manager.model_manager") as mock_model_manager:
            mock_model_manager.flash_model = Mock()
            
            with patch("src.services.langchain.vector_store_manager.HyDERetriever") as mock_hyde:
                mock_hyde_retriever = Mock()
                mock_hyde.return_value = mock_hyde_retriever
                
                result = manager.get_retriever_with_hyde(mock_prompt_template)
                
                mock_hyde.assert_called_once_with(
                    base_retriever=mock_retriever,
                    llm=mock_model_manager.flash_model,
                    prompt=mock_prompt_template
                )
                assert result == mock_hyde_retriever

    def test_vector_store_manager_singleton(self):
        """Test that vector_store_manager is a singleton instance"""
        assert isinstance(vector_store_manager, VectorStoreManager)
        assert vector_store_manager._vector_store is None
        assert vector_store_manager._opensearch_client is None