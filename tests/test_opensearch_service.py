import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from src.services.opensearch_service import OpenSearchService
from src.config import settings


class TestOpenSearchService:
    """Test cases for OpenSearchService"""

    @pytest.fixture
    def mock_opensearch_client(self):
        """Create a mock OpenSearch client"""
        mock_client = Mock()
        mock_client.indices = Mock()
        mock_client.indices.exists = Mock(return_value=False)
        mock_client.indices.create = Mock()
        mock_client.indices.delete = Mock()
        mock_client.index = Mock(return_value={"_id": "test-id", "_result": "created"})
        mock_client.search = Mock(return_value={
            "hits": {
                "hits": [
                    {
                        "_source": {
                            "event_id": "event1",
                            "title": "Test Event",
                            "content": "Test content",
                            "tags": ["test"]
                        }
                    }
                ]
            }
        })
        return mock_client

    @pytest.fixture
    def opensearch_service(self, mock_opensearch_client):
        """Create an OpenSearchService instance with mocked client"""
        with patch("src.services.opensearch_service.OpenSearch", return_value=mock_opensearch_client):
            service = OpenSearchService()
            service.client = mock_opensearch_client
            return service

    def test_init(self):
        """Test OpenSearchService initialization"""
        with patch("src.services.opensearch_service.OpenSearch") as mock_opensearch:
            service = OpenSearchService()
            
            # Verify OpenSearch client was created with correct parameters
            mock_opensearch.assert_called_once_with(
                hosts=[{'host': settings.opensearch_host, 'port': settings.opensearch_port}],
                use_ssl=False,
                verify_certs=False
            )
            assert service.index_name == settings.opensearch_index

    @pytest.mark.asyncio
    async def test_create_index_not_exists(self, opensearch_service):
        """Test creating index when it doesn't exist"""
        opensearch_service.client.indices.exists.return_value = False
        
        await opensearch_service.create_index()
        
        # Verify index existence was checked
        opensearch_service.client.indices.exists.assert_called_once_with(index=opensearch_service.index_name)
        
        # Verify index was created
        opensearch_service.client.indices.create.assert_called_once()
        call_args = opensearch_service.client.indices.create.call_args
        assert call_args[1]["index"] == opensearch_service.index_name
        
        # Verify index body structure
        index_body = call_args[1]["body"]
        assert index_body["settings"]["index"]["knn"] is True
        assert index_body["mappings"]["properties"]["embedding"]["type"] == "knn_vector"
        assert index_body["mappings"]["properties"]["embedding"]["dimension"] == settings.opensearch_embedding_dim

    @pytest.mark.asyncio
    async def test_create_index_already_exists(self, opensearch_service):
        """Test creating index when it already exists"""
        opensearch_service.client.indices.exists.return_value = True
        
        await opensearch_service.create_index()
        
        # Verify index existence was checked
        opensearch_service.client.indices.exists.assert_called_once_with(index=opensearch_service.index_name)
        
        # Verify index was NOT created
        opensearch_service.client.indices.create.assert_not_called()

    @pytest.mark.asyncio
    async def test_index_document_basic(self, opensearch_service):
        """Test indexing a document with basic parameters"""
        doc_id = "test-doc-1"
        content = "Test document content"
        embedding = [0.1, 0.2, 0.3]
        
        response = await opensearch_service.index_document(doc_id, content, embedding)
        
        # Verify document was indexed
        opensearch_service.client.index.assert_called_once()
        call_args = opensearch_service.client.index.call_args
        
        assert call_args[1]["index"] == opensearch_service.index_name
        assert call_args[1]["id"] == doc_id
        assert call_args[1]["refresh"] is True
        
        # Verify document structure
        document = call_args[1]["body"]
        assert document["event_id"] == doc_id
        assert document["content"] == content
        assert document["embedding"] == embedding
        assert document["title"] == ""
        assert document["tags"] == []
        assert document["created_at"] == "now"
        
        assert response["_id"] == "test-id"

    @pytest.mark.asyncio
    async def test_index_document_with_all_fields(self, opensearch_service):
        """Test indexing a document with all optional fields"""
        doc_id = "test-doc-2"
        content = "Test document content"
        embedding = [0.1, 0.2, 0.3]
        title = "Test Title"
        tags = ["tag1", "tag2"]
        
        response = await opensearch_service.index_document(
            doc_id, content, embedding, title=title, tags=tags
        )
        
        # Verify document structure
        call_args = opensearch_service.client.index.call_args
        document = call_args[1]["body"]
        assert document["title"] == title
        assert document["tags"] == tags

    @pytest.mark.asyncio
    async def test_search_similar_documents_default_k(self, opensearch_service):
        """Test searching similar documents with default k value"""
        query_embedding = [0.1, 0.2, 0.3]
        
        results = await opensearch_service.search_similar_documents(query_embedding)
        
        # Verify search was called
        opensearch_service.client.search.assert_called_once()
        call_args = opensearch_service.client.search.call_args
        
        assert call_args[1]["index"] == opensearch_service.index_name
        
        # Verify query structure
        query = call_args[1]["body"]
        assert query["size"] == settings.top_k_results
        assert query["query"]["knn"]["embedding"]["vector"] == query_embedding
        assert query["query"]["knn"]["embedding"]["k"] == settings.top_k_results
        assert "_source" in query
        
        # Verify results
        assert len(results) == 1
        assert results[0]["event_id"] == "event1"
        assert results[0]["title"] == "Test Event"

    @pytest.mark.asyncio
    async def test_search_similar_documents_custom_k(self, opensearch_service):
        """Test searching similar documents with custom k value"""
        query_embedding = [0.1, 0.2, 0.3]
        custom_k = 10
        
        await opensearch_service.search_similar_documents(query_embedding, k=custom_k)
        
        # Verify query structure with custom k
        call_args = opensearch_service.client.search.call_args
        query = call_args[1]["body"]
        assert query["size"] == custom_k
        assert query["query"]["knn"]["embedding"]["k"] == custom_k

    @pytest.mark.asyncio
    async def test_search_similar_documents_empty_results(self, opensearch_service):
        """Test searching similar documents with no results"""
        opensearch_service.client.search.return_value = {"hits": {"hits": []}}
        
        query_embedding = [0.1, 0.2, 0.3]
        results = await opensearch_service.search_similar_documents(query_embedding)
        
        assert results == []

    @pytest.mark.asyncio
    async def test_delete_index_exists(self, opensearch_service):
        """Test deleting index when it exists"""
        opensearch_service.client.indices.exists.return_value = True
        
        await opensearch_service.delete_index()
        
        # Verify index existence was checked
        opensearch_service.client.indices.exists.assert_called_once_with(index=opensearch_service.index_name)
        
        # Verify index was deleted
        opensearch_service.client.indices.delete.assert_called_once_with(index=opensearch_service.index_name)

    @pytest.mark.asyncio
    async def test_delete_index_not_exists(self, opensearch_service):
        """Test deleting index when it doesn't exist"""
        opensearch_service.client.indices.exists.return_value = False
        
        await opensearch_service.delete_index()
        
        # Verify index existence was checked
        opensearch_service.client.indices.exists.assert_called_once_with(index=opensearch_service.index_name)
        
        # Verify index was NOT deleted
        opensearch_service.client.indices.delete.assert_not_called()