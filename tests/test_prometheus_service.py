import pytest
import aiohttp
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from src.services.prometheus_service import PrometheusService
from src.config import settings


class TestPrometheusService:
    """Test cases for PrometheusService"""

    @pytest.fixture
    def prometheus_service(self):
        """Create a PrometheusService instance for testing"""
        return PrometheusService()

    @pytest.fixture
    def mock_response_success(self):
        """Create a mock successful response"""
        mock_resp = AsyncMock()
        mock_resp.json = AsyncMock(return_value={
            "status": "success",
            "data": {
                "result": [
                    {
                        "value": [1234567890, "50.5"]
                    }
                ]
            }
        })
        return mock_resp

    @pytest.fixture
    def mock_response_error(self):
        """Create a mock error response"""
        mock_resp = AsyncMock()
        mock_resp.json = AsyncMock(return_value={
            "status": "error",
            "errorType": "bad_data",
            "error": "Invalid query"
        })
        return mock_resp

    def test_init(self, prometheus_service):
        """Test PrometheusService initialization"""
        expected_url = f"http://{settings.prometheus_host}:{settings.prometheus_port}"
        assert prometheus_service.base_url == expected_url

    @pytest.mark.asyncio
    async def test_query_success(self, prometheus_service, mock_response_success):
        """Test successful Prometheus query"""
        with patch("aiohttp.ClientSession") as mock_session:
            mock_session.return_value.__aenter__.return_value.get.return_value.__aenter__.return_value = mock_response_success
            
            result = await prometheus_service.query("up")
            assert result["result"][0]["value"][1] == "50.5"

    @pytest.mark.asyncio
    async def test_query_error(self, prometheus_service, mock_response_error):
        """Test Prometheus query error handling"""
        with patch("aiohttp.ClientSession") as mock_session:
            mock_session.return_value.__aenter__.return_value.get.return_value.__aenter__.return_value = mock_response_error
            
            with pytest.raises(Exception, match="Prometheus query failed"):
                await prometheus_service.query("invalid_query")

    @pytest.mark.asyncio
    async def test_query_range_success(self, prometheus_service, mock_response_success):
        """Test successful Prometheus range query"""
        start = datetime.now() - timedelta(hours=1)
        end = datetime.now()
        
        with patch("aiohttp.ClientSession") as mock_session:
            mock_session.return_value.__aenter__.return_value.get.return_value.__aenter__.return_value = mock_response_success
            
            result = await prometheus_service.query_range("up", start, end)
            assert result["result"][0]["value"][1] == "50.5"

    @pytest.mark.asyncio
    async def test_query_range_error(self, prometheus_service, mock_response_error):
        """Test Prometheus range query error handling"""
        start = datetime.now() - timedelta(hours=1)
        end = datetime.now()
        
        with patch("aiohttp.ClientSession") as mock_session:
            mock_session.return_value.__aenter__.return_value.get.return_value.__aenter__.return_value = mock_response_error
            
            with pytest.raises(Exception, match="Prometheus range query failed"):
                await prometheus_service.query_range("invalid_query", start, end)

    @pytest.mark.asyncio
    async def test_query_range_with_custom_step(self, prometheus_service, mock_response_success):
        """Test Prometheus range query with custom step"""
        start = datetime.now() - timedelta(hours=1)
        end = datetime.now()
        
        with patch("aiohttp.ClientSession") as mock_session:
            mock_get = AsyncMock(return_value=AsyncMock(__aenter__=AsyncMock(return_value=mock_response_success)))
            mock_session.return_value.__aenter__.return_value.get = mock_get
            
            await prometheus_service.query_range("up", start, end, step="30s")
            
            # Verify the step parameter was passed
            _, kwargs = mock_get.call_args
            assert kwargs["params"]["step"] == "30s"

    @pytest.mark.asyncio
    async def test_get_host_metrics_all_metrics(self, prometheus_service):
        """Test get_host_metrics with all metrics available"""
        hostname = "test-host"
        
        # Mock responses for different metrics
        cpu_response = {
            "status": "success",
            "data": {"result": [{"value": [1234567890, "75.5"]}]}
        }
        mem_response = {
            "status": "success",
            "data": {"result": [{"value": [1234567890, "60.2"]}]}
        }
        io_response = {
            "status": "success",
            "data": {"result": [{"value": [1234567890, "10.3"]}]}
        }
        net_response = {
            "status": "success",
            "data": {"result": [
                {"value": [1234567890, "1250000"]},
                {"value": [1234567890, "750000"]}
            ]}
        }
        
        responses = [cpu_response, mem_response, io_response, net_response]
        response_iter = iter(responses)
        
        async def mock_json():
            return next(response_iter)
        
        with patch("aiohttp.ClientSession") as mock_session:
            mock_resp = AsyncMock()
            mock_resp.json = mock_json
            mock_session.return_value.__aenter__.return_value.get.return_value.__aenter__.return_value = mock_resp
            
            metrics = await prometheus_service.get_host_metrics(hostname)
            
            assert metrics["CPU使用率"] == "75.5%"
            assert metrics["RAM使用率"] == "60.2%"
            assert metrics["磁碟I/O等待"] == "10.3%"
            assert metrics["網路流出量"] == "16 Mbps"

    @pytest.mark.asyncio
    async def test_get_host_metrics_empty_results(self, prometheus_service):
        """Test get_host_metrics with empty results"""
        hostname = "test-host"
        
        # Mock empty responses
        empty_response = {
            "status": "success",
            "data": {"result": []}
        }
        
        with patch("aiohttp.ClientSession") as mock_session:
            mock_resp = AsyncMock()
            mock_resp.json = AsyncMock(return_value=empty_response)
            mock_session.return_value.__aenter__.return_value.get.return_value.__aenter__.return_value = mock_resp
            
            metrics = await prometheus_service.get_host_metrics(hostname)
            
            # Should return empty dict when no results
            assert metrics == {}

    @pytest.mark.asyncio
    async def test_get_host_metrics_partial_results(self, prometheus_service):
        """Test get_host_metrics with some metrics missing"""
        hostname = "test-host"
        
        # Mock responses - CPU has data, others are empty
        cpu_response = {
            "status": "success",
            "data": {"result": [{"value": [1234567890, "75.5"]}]}
        }
        empty_response = {
            "status": "success",
            "data": {"result": []}
        }
        
        responses = [cpu_response, empty_response, empty_response, empty_response]
        response_iter = iter(responses)
        
        async def mock_json():
            return next(response_iter)
        
        with patch("aiohttp.ClientSession") as mock_session:
            mock_resp = AsyncMock()
            mock_resp.json = mock_json
            mock_session.return_value.__aenter__.return_value.get.return_value.__aenter__.return_value = mock_resp
            
            metrics = await prometheus_service.get_host_metrics(hostname)
            
            assert metrics["CPU使用率"] == "75.5%"
            assert "RAM使用率" not in metrics
            assert "磁碟I/O等待" not in metrics
            assert "網路流出量" not in metrics

    @pytest.mark.asyncio
    async def test_get_host_metrics_query_construction(self, prometheus_service):
        """Test that get_host_metrics constructs correct queries"""
        hostname = "test-host"
        queries_captured = []
        
        async def capture_query(url, params=None):
            queries_captured.append(params["query"])
            mock_resp = AsyncMock()
            mock_resp.json = AsyncMock(return_value={
                "status": "success",
                "data": {"result": []}
            })
            return AsyncMock(__aenter__=AsyncMock(return_value=mock_resp))
        
        with patch("aiohttp.ClientSession") as mock_session:
            mock_session.return_value.__aenter__.return_value.get = capture_query
            
            await prometheus_service.get_host_metrics(hostname)
            
            # Verify all 4 queries were made
            assert len(queries_captured) == 4
            
            # Verify each query contains the hostname
            for query in queries_captured:
                assert hostname in query