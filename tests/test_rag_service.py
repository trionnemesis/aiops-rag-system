import pytest
import json
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime

from src.services.rag_service import RAGService
from src.services.exceptions import (
    RAGServiceError, CacheError, HyDEGenerationError,
    DocumentRetrievalError, ReportGenerationError, PrometheusError
)
from src.models.schemas import InsightReport


class TestRAGService:
    """Test cases for RAGService"""

    @pytest.fixture
    def mock_rag_chain_service(self):
        """Create a mock RAGChainService"""
        mock_service = Mock()
        mock_service.generate_report = AsyncMock()
        mock_service.generate_report_with_steps = AsyncMock()
        mock_service.enrich_with_prometheus = AsyncMock()
        mock_service.clear_cache = Mock()
        mock_service.get_cache_info = Mock()
        mock_service.create_custom_chain = Mock()
        mock_service.prometheus = Mock()
        return mock_service

    @pytest.fixture
    def rag_service(self, mock_rag_chain_service):
        """Create a RAGService instance with mocked dependencies"""
        with patch("src.services.rag_service.RAGChainService", return_value=mock_rag_chain_service):
            service = RAGService()
            service.rag_chain_service = mock_rag_chain_service
            return service

    def test_init(self):
        """Test RAGService initialization"""
        with patch("src.services.rag_service.RAGChainService") as mock_rag_chain:
            mock_instance = Mock()
            mock_rag_chain.return_value = mock_instance
            
            service = RAGService()
            
            mock_rag_chain.assert_called_once()
            assert service.rag_chain_service is mock_instance

    def test_create_cache_key_complete_data(self):
        """Test cache key creation with complete monitoring data"""
        monitoring_data = {
            "主機": "server-01",
            "CPU使用率": "80%",
            "RAM使用率": "60%",
            "磁碟使用率": "45%",
            "服務名稱": "web-service",
            "other_field": "ignored"
        }
        
        key = RAGService._create_cache_key(monitoring_data)
        
        # Parse the key to verify structure
        key_data = json.loads(key)
        assert key_data["host"] == "server-01"
        assert key_data["cpu"] == "80%"
        assert key_data["ram"] == "60%"
        assert key_data["disk"] == "45%"
        assert key_data["service"] == "web-service"
        assert "other_field" not in key_data

    def test_create_cache_key_partial_data(self):
        """Test cache key creation with partial monitoring data"""
        monitoring_data = {
            "主機": "server-02",
            "CPU使用率": "90%"
        }
        
        key = RAGService._create_cache_key(monitoring_data)
        
        # Parse the key to verify defaults
        key_data = json.loads(key)
        assert key_data["host"] == "server-02"
        assert key_data["cpu"] == "90%"
        assert key_data["ram"] == 0
        assert key_data["disk"] == 0
        assert key_data["service"] == ""

    def test_create_cache_key_empty_data(self):
        """Test cache key creation with empty monitoring data"""
        monitoring_data = {}
        
        key = RAGService._create_cache_key(monitoring_data)
        
        # Parse the key to verify all defaults
        key_data = json.loads(key)
        assert key_data["host"] == ""
        assert key_data["cpu"] == 0
        assert key_data["ram"] == 0
        assert key_data["disk"] == 0
        assert key_data["service"] == ""

    def test_create_cache_key_consistent_ordering(self):
        """Test that cache key has consistent ordering regardless of input order"""
        data1 = {"服務名稱": "api", "主機": "host1", "CPU使用率": "50%"}
        data2 = {"主機": "host1", "CPU使用率": "50%", "服務名稱": "api"}
        
        key1 = RAGService._create_cache_key(data1)
        key2 = RAGService._create_cache_key(data2)
        
        assert key1 == key2

    @pytest.mark.asyncio
    async def test_generate_report_success(self, rag_service, mock_rag_chain_service):
        """Test successful report generation"""
        mock_report = InsightReport(
            insight_analysis="Test insight",
            recommendations="Test recommendations",
            generated_at=datetime.now()
        )
        mock_rag_chain_service.generate_report.return_value = mock_report
        
        monitoring_data = {"host": "test-host", "cpu": 80}
        result = await rag_service.generate_report(monitoring_data)
        
        assert result == mock_report
        mock_rag_chain_service.generate_report.assert_called_once_with(monitoring_data)

    @pytest.mark.asyncio
    async def test_generate_report_hyde_error(self, rag_service, mock_rag_chain_service):
        """Test report generation with HyDE error"""
        mock_rag_chain_service.generate_report.side_effect = HyDEGenerationError("HyDE failed")
        
        with pytest.raises(HyDEGenerationError, match="HyDE failed"):
            await rag_service.generate_report({"test": "data"})

    @pytest.mark.asyncio
    async def test_generate_report_with_steps_success(self, rag_service, mock_rag_chain_service):
        """Test report generation with steps"""
        mock_result = {
            "report": InsightReport(
                insight_analysis="Test insight",
                recommendations="Test recommendations",
                generated_at=datetime.now()
            ),
            "steps": {
                "hyde_query": "Test query",
                "documents_found": 3
            }
        }
        mock_rag_chain_service.generate_report_with_steps.return_value = mock_result
        
        result = await rag_service.generate_report_with_steps({"test": "data"})
        
        assert result == mock_result
        mock_rag_chain_service.generate_report_with_steps.assert_called_once()

    @pytest.mark.asyncio
    async def test_enrich_with_prometheus_success(self, rag_service, mock_rag_chain_service):
        """Test successful Prometheus enrichment"""
        enriched_data = {
            "host": "test-host",
            "CPU使用率": "75%",
            "RAM使用率": "60%"
        }
        mock_rag_chain_service.enrich_with_prometheus.return_value = enriched_data
        
        result = await rag_service.enrich_with_prometheus("test-host", {"host": "test-host"})
        
        assert result == enriched_data
        mock_rag_chain_service.enrich_with_prometheus.assert_called_once_with(
            "test-host", {"host": "test-host"}
        )

    @pytest.mark.asyncio
    async def test_enrich_with_prometheus_error(self, rag_service, mock_rag_chain_service):
        """Test Prometheus enrichment error"""
        mock_rag_chain_service.enrich_with_prometheus.side_effect = PrometheusError("Connection failed")
        
        with pytest.raises(PrometheusError, match="Connection failed"):
            await rag_service.enrich_with_prometheus("test-host", {})

    def test_clear_cache_success(self, rag_service, mock_rag_chain_service):
        """Test successful cache clearing"""
        rag_service.clear_cache()
        
        mock_rag_chain_service.clear_cache.assert_called_once()

    def test_clear_cache_error(self, rag_service, mock_rag_chain_service):
        """Test cache clearing error"""
        mock_rag_chain_service.clear_cache.side_effect = Exception("Cache error")
        
        with pytest.raises(CacheError, match="Failed to clear cache: Cache error"):
            rag_service.clear_cache()

    def test_get_cache_info_success(self, rag_service, mock_rag_chain_service):
        """Test successful cache info retrieval"""
        mock_cache_info = {
            "hyde_cache": {"hits": 10, "misses": 5},
            "embedding_cache": {"hits": 20, "misses": 10}
        }
        mock_rag_chain_service.get_cache_info.return_value = mock_cache_info
        
        result = rag_service.get_cache_info()
        
        assert result == mock_cache_info
        mock_rag_chain_service.get_cache_info.assert_called_once()

    def test_get_cache_info_error(self, rag_service, mock_rag_chain_service):
        """Test cache info retrieval error"""
        mock_rag_chain_service.get_cache_info.side_effect = Exception("Info error")
        
        with pytest.raises(CacheError, match="Failed to get cache info: Info error"):
            rag_service.get_cache_info()

    def test_create_custom_chain(self, rag_service, mock_rag_chain_service):
        """Test creating custom chain"""
        mock_chain = Mock()
        mock_rag_chain_service.create_custom_chain.return_value = mock_chain
        
        kwargs = {"hyde_enabled": True, "retriever_kwargs": {"k": 5}}
        result = rag_service.create_custom_chain(**kwargs)
        
        assert result == mock_chain
        mock_rag_chain_service.create_custom_chain.assert_called_once_with(**kwargs)

    def test_prometheus_property(self, rag_service, mock_rag_chain_service):
        """Test prometheus property access"""
        mock_prometheus = Mock()
        mock_rag_chain_service.prometheus = mock_prometheus
        
        result = rag_service.prometheus
        
        assert result == mock_prometheus