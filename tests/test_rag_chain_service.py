import pytest
import json
from datetime import datetime
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from langchain.schema import Document
from langchain_core.output_parsers import StrOutputParser

from src.services.langchain.rag_chain_service import RAGChainService
from src.services.exceptions import (
    HyDEGenerationError, DocumentRetrievalError, 
    ReportGenerationError, PrometheusError, GeminiAPIError
)
from src.models.schemas import InsightReport


class TestRAGChainService:
    """Test cases for RAGChainService"""

    @pytest.fixture
    def mock_dependencies(self):
        """Mock all dependencies for RAGChainService"""
        with patch("src.services.langchain.rag_chain_service.PrometheusService") as mock_prometheus:
            with patch("src.services.langchain.rag_chain_service.model_manager") as mock_model_manager:
                with patch("src.services.langchain.rag_chain_service.prompt_manager") as mock_prompt_manager:
                    with patch("src.services.langchain.rag_chain_service.vector_store_manager") as mock_vector_store:
                        # Setup mocks
                        mock_prometheus.return_value = Mock()
                        
                        # Mock models
                        mock_model_manager.pro_model = Mock()
                        mock_model_manager.flash_model = Mock()
                        mock_model_manager.embedding_model = Mock()
                        mock_model_manager.embedding_model.aembed_documents = AsyncMock(
                            return_value=[[0.1, 0.2, 0.3]]
                        )
                        
                        # Mock prompts
                        mock_prompt_manager.get_prompt = Mock(side_effect=lambda x: Mock())
                        
                        # Mock vector store
                        mock_retriever = Mock()
                        mock_retriever.invoke = Mock(return_value=[
                            Document(page_content="Test document content")
                        ])
                        mock_vector_store.as_retriever = Mock(return_value=mock_retriever)
                        mock_vector_store.similarity_search = AsyncMock(return_value=[
                            Document(page_content="Test document content")
                        ])
                        
                        yield {
                            "prometheus": mock_prometheus,
                            "model_manager": mock_model_manager,
                            "prompt_manager": mock_prompt_manager,
                            "vector_store_manager": mock_vector_store,
                            "retriever": mock_retriever
                        }

    @pytest.fixture
    def rag_service(self, mock_dependencies):
        """Create a RAGChainService instance with mocked dependencies"""
        service = RAGChainService()
        service.retriever = mock_dependencies["retriever"]
        return service

    def test_init(self, mock_dependencies):
        """Test RAGChainService initialization"""
        service = RAGChainService()
        assert service.prometheus is not None
        assert hasattr(service, "hyde_chain")
        assert hasattr(service, "retriever")
        assert hasattr(service, "report_chain")
        assert hasattr(service, "full_rag_chain")

    def test_build_hyde_chain_error(self, mock_dependencies):
        """Test HyDE chain building with error"""
        mock_dependencies["prompt_manager"].get_prompt.side_effect = Exception("Prompt error")
        
        with pytest.raises(HyDEGenerationError, match="Failed to build HyDE chain"):
            service = RAGChainService()

    def test_build_report_chain_error(self):
        """Test report chain building with error"""
        with patch("src.services.langchain.rag_chain_service.prompt_manager") as mock_prompt:
            # Make hyde chain succeed but report chain fail
            call_count = 0
            def side_effect(prompt_name):
                nonlocal call_count
                call_count += 1
                if call_count == 1:  # hyde_generation
                    return Mock()
                else:  # final_report
                    raise Exception("Report prompt error")
            
            mock_prompt.get_prompt.side_effect = side_effect
            
            with patch("src.services.langchain.rag_chain_service.model_manager"):
                with patch("src.services.langchain.rag_chain_service.vector_store_manager"):
                    with pytest.raises(ReportGenerationError, match="Failed to build report chain"):
                        service = RAGChainService()

    def test_safe_retrieval_hyde_success(self, rag_service, mock_dependencies):
        """Test safe retrieval with successful HyDE"""
        # Mock successful HyDE chain
        rag_service.hyde_chain = Mock()
        rag_service.hyde_chain.invoke = Mock(return_value="Hyde query")
        
        # Mock retriever
        test_docs = [Document(page_content="Test content")]
        rag_service.retriever.invoke = Mock(return_value=test_docs)
        
        result = rag_service._safe_retrieval({
            "monitoring_data_str": '{"test": "data"}'
        })
        
        assert result == test_docs
        rag_service.hyde_chain.invoke.assert_called_once()
        rag_service.retriever.invoke.assert_called_with("Hyde query")

    def test_safe_retrieval_hyde_failure_fallback(self, rag_service):
        """Test safe retrieval with HyDE failure, using fallback"""
        # Mock HyDE chain failure
        rag_service.hyde_chain = Mock()
        rag_service.hyde_chain.invoke = Mock(side_effect=Exception("HyDE failed"))
        
        # Mock fallback retriever
        test_docs = [Document(page_content="Fallback content")]
        rag_service.retriever.invoke = Mock(return_value=test_docs)
        
        result = rag_service._safe_retrieval({
            "monitoring_data_str": '{"test": "data"}'
        })
        
        assert result == test_docs
        rag_service.retriever.invoke.assert_called_with('{"test": "data"}')

    def test_safe_retrieval_all_methods_fail(self, rag_service):
        """Test safe retrieval when all methods fail"""
        # Mock both HyDE and fallback failures
        rag_service.hyde_chain = Mock()
        rag_service.hyde_chain.invoke = Mock(side_effect=Exception("HyDE failed"))
        rag_service.retriever.invoke = Mock(side_effect=Exception("Retrieval failed"))
        
        with pytest.raises(DocumentRetrievalError, match="All retrieval methods failed"):
            rag_service._safe_retrieval({
                "monitoring_data_str": '{"test": "data"}'
            })

    def test_generate_summary_context_with_documents(self, rag_service):
        """Test summary context generation with documents"""
        docs = [
            Document(page_content="Doc 1 content"),
            Document(page_content="Doc 2 content"),
            Document(page_content="Doc 3 content")
        ]
        
        result = rag_service._generate_summary_context(
            docs, '{"test": "data"}'
        )
        
        assert "文檔 1:\nDoc 1 content" in result
        assert "文檔 2:\nDoc 2 content" in result
        assert "文檔 3:\nDoc 3 content" in result

    def test_generate_summary_context_no_documents(self, rag_service):
        """Test summary context generation with no documents"""
        result = rag_service._generate_summary_context(
            [], '{"test": "data"}'
        )
        
        assert "未找到相關文檔" in result
        assert '{"test": "data"}' in result

    def test_generate_summary_context_many_documents(self, rag_service):
        """Test summary context generation with more than 5 documents"""
        docs = [Document(page_content=f"Doc {i} content") for i in range(10)]
        
        result = rag_service._generate_summary_context(
            docs, '{"test": "data"}'
        )
        
        # Should only include first 5 documents
        assert "文檔 5:\nDoc 4 content" in result
        assert "文檔 6:" not in result

    def test_parse_report_sections_complete(self, rag_service):
        """Test parsing complete report sections"""
        report_text = """洞見分析
        This is the insight analysis
        
        建議與行動方案
        These are the recommendations"""
        
        result = rag_service._parse_report_sections(report_text)
        
        assert result["insight_analysis"] == "This is the insight analysis"
        assert result["recommendations"] == "These are the recommendations"

    def test_parse_report_sections_no_recommendations(self, rag_service):
        """Test parsing report with no recommendations section"""
        report_text = """洞見分析
        This is only the insight analysis"""
        
        result = rag_service._parse_report_sections(report_text)
        
        assert result["insight_analysis"] == "This is only the insight analysis"
        assert result["recommendations"] == ""

    def test_parse_report_sections_empty(self, rag_service):
        """Test parsing empty report"""
        result = rag_service._parse_report_sections("")
        
        assert result["insight_analysis"] == ""
        assert result["recommendations"] == ""

    @pytest.mark.asyncio
    async def test_get_cached_embedding_success(self, rag_service, mock_dependencies):
        """Test cached embedding generation success"""
        result = await rag_service._get_cached_embedding("test text")
        
        assert result == [0.1, 0.2, 0.3]
        mock_dependencies["model_manager"].embedding_model.aembed_documents.assert_called_once_with(["test text"])

    @pytest.mark.asyncio
    async def test_get_cached_embedding_failure(self, rag_service, mock_dependencies):
        """Test cached embedding generation failure"""
        mock_dependencies["model_manager"].embedding_model.aembed_documents.side_effect = Exception("API error")
        
        with pytest.raises(GeminiAPIError, match="Failed to generate embeddings"):
            await rag_service._get_cached_embedding("test text")

    @pytest.mark.asyncio
    async def test_get_cached_hyde_success(self, rag_service):
        """Test cached HyDE generation success"""
        rag_service.hyde_chain = AsyncMock()
        rag_service.hyde_chain.ainvoke = AsyncMock(return_value="HyDE result")
        
        result = await rag_service._get_cached_hyde('{"test": "data"}')
        
        assert result == "HyDE result"

    @pytest.mark.asyncio
    async def test_get_cached_hyde_failure(self, rag_service):
        """Test cached HyDE generation failure"""
        rag_service.hyde_chain = AsyncMock()
        rag_service.hyde_chain.ainvoke = AsyncMock(side_effect=Exception("HyDE error"))
        
        with pytest.raises(HyDEGenerationError, match="Failed to generate HyDE query"):
            await rag_service._get_cached_hyde('{"test": "data"}')

    @pytest.mark.asyncio
    async def test_generate_report_success(self, rag_service):
        """Test successful report generation"""
        # Mock the full RAG chain
        rag_service.full_rag_chain = AsyncMock()
        rag_service.full_rag_chain.ainvoke = AsyncMock(return_value={
            "insight_analysis": "Test insight",
            "recommendations": "Test recommendations"
        })
        
        monitoring_data = {"host": "test-host", "cpu": 80}
        report = await rag_service.generate_report(monitoring_data)
        
        assert isinstance(report, InsightReport)
        assert report.insight_analysis == "Test insight"
        assert report.recommendations == "Test recommendations"
        assert isinstance(report.generated_at, datetime)

    @pytest.mark.asyncio
    async def test_generate_report_failure(self, rag_service):
        """Test report generation failure"""
        rag_service.full_rag_chain = AsyncMock()
        rag_service.full_rag_chain.ainvoke = AsyncMock(side_effect=Exception("Chain error"))
        
        with pytest.raises(ReportGenerationError, match="Failed to generate report"):
            await rag_service.generate_report({"test": "data"})

    @pytest.mark.asyncio
    async def test_generate_report_with_steps_success(self, rag_service, mock_dependencies):
        """Test report generation with steps (debug mode)"""
        # Mock HyDE
        rag_service._get_cached_hyde = AsyncMock(return_value="HyDE query")
        
        # Mock report chain
        rag_service.report_chain = AsyncMock()
        rag_service.report_chain.ainvoke = AsyncMock(return_value={
            "insight_analysis": "Test insight",
            "recommendations": "Test recommendations"
        })
        
        monitoring_data = {"host": "test-host", "cpu": 80}
        result = await rag_service.generate_report_with_steps(monitoring_data)
        
        assert "report" in result
        assert "steps" in result
        assert result["steps"]["hyde_query"] == "HyDE query"
        assert result["steps"]["documents_found"] == 1
        assert isinstance(result["report"], InsightReport)

    @pytest.mark.asyncio
    async def test_generate_report_with_steps_hyde_failure(self, rag_service, mock_dependencies):
        """Test report generation with steps when HyDE fails"""
        # Mock HyDE failure
        rag_service._get_cached_hyde = AsyncMock(side_effect=Exception("HyDE error"))
        
        # Mock report chain
        rag_service.report_chain = AsyncMock()
        rag_service.report_chain.ainvoke = AsyncMock(return_value={
            "insight_analysis": "Test insight",
            "recommendations": "Test recommendations"
        })
        
        monitoring_data = {"host": "test-host", "cpu": 80}
        result = await rag_service.generate_report_with_steps(monitoring_data)
        
        assert result["steps"]["hyde_query"] == "HyDE generation failed, using fallback"

    @pytest.mark.asyncio
    async def test_enrich_with_prometheus_success(self, rag_service):
        """Test successful Prometheus enrichment"""
        rag_service.prometheus.get_host_metrics = AsyncMock(return_value={
            "CPU使用率": "75%",
            "RAM使用率": "60%"
        })
        
        monitoring_data = {"host": "test-host", "alert": "high_cpu"}
        result = await rag_service.enrich_with_prometheus("test-host", monitoring_data)
        
        assert result["host"] == "test-host"
        assert result["alert"] == "high_cpu"
        assert result["CPU使用率"] == "75%"
        assert result["RAM使用率"] == "60%"

    @pytest.mark.asyncio
    async def test_enrich_with_prometheus_failure(self, rag_service):
        """Test Prometheus enrichment failure"""
        rag_service.prometheus.get_host_metrics = AsyncMock(
            side_effect=Exception("Prometheus error")
        )
        
        with pytest.raises(PrometheusError, match="Failed to enrich with Prometheus data"):
            await rag_service.enrich_with_prometheus("test-host", {})

    def test_create_custom_chain_with_hyde(self, rag_service, mock_dependencies):
        """Test creating custom chain with HyDE enabled"""
        chain = rag_service.create_custom_chain(hyde_enabled=True)
        
        assert chain is not None
        # Verify the chain structure includes HyDE

    def test_create_custom_chain_without_hyde(self, rag_service, mock_dependencies):
        """Test creating custom chain without HyDE"""
        chain = rag_service.create_custom_chain(hyde_enabled=False)
        
        assert chain is not None
        # Verify the chain structure doesn't include HyDE

    def test_create_custom_chain_with_retriever_kwargs(self, rag_service, mock_dependencies):
        """Test creating custom chain with custom retriever kwargs"""
        custom_kwargs = {"search_kwargs": {"k": 5}}
        chain = rag_service.create_custom_chain(retriever_kwargs=custom_kwargs)
        
        assert chain is not None
        mock_dependencies["vector_store_manager"].as_retriever.assert_called_with(**custom_kwargs)

    def test_clear_cache(self, rag_service):
        """Test cache clearing"""
        # Mock cache clear methods
        rag_service._get_cached_embedding.cache_clear = Mock()
        rag_service._get_cached_hyde.cache_clear = Mock()
        
        rag_service.clear_cache()
        
        rag_service._get_cached_embedding.cache_clear.assert_called_once()
        rag_service._get_cached_hyde.cache_clear.assert_called_once()

    def test_get_cache_info(self, rag_service):
        """Test getting cache information"""
        # Mock cache info
        mock_cache_info = Mock()
        mock_cache_info.hits = 10
        mock_cache_info.misses = 5
        mock_cache_info.maxsize = 100
        mock_cache_info.currsize = 15
        
        rag_service._get_cached_embedding.cache_info = Mock(return_value=mock_cache_info)
        rag_service._get_cached_hyde.cache_info = Mock(return_value=mock_cache_info)
        
        result = rag_service.get_cache_info()
        
        assert result["embedding_cache"]["hits"] == 10
        assert result["embedding_cache"]["misses"] == 5
        assert result["hyde_cache"]["hits"] == 10
        assert result["hyde_cache"]["misses"] == 5