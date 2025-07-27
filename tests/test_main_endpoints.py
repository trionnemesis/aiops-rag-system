import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, AsyncMock
from src.main import app
from src.services.exceptions import PrometheusError, CacheError


class TestMainEndpoints:
    """Additional tests for main.py endpoints to improve coverage"""

    @pytest.fixture
    def client(self):
        """Create a test client"""
        return TestClient(app)

    def test_health_endpoint_exception(self, client):
        """Test health endpoint when an exception occurs"""
        with patch("src.main.settings", side_effect=Exception("Config error")):
            response = client.get("/health")
            assert response.status_code == 503
            assert response.json()["status"] == "unhealthy"
            assert "Config error" in response.json()["error"]

    @pytest.mark.asyncio
    async def test_generate_report_prometheus_enrichment_error(self, client):
        """Test report generation when Prometheus enrichment fails"""
        request_data = {
            "monitoring_data": {
                "主機": "test-host",
                "CPU使用率": "80%"
            }
        }
        
        with patch("src.main.rag_service.enrich_with_prometheus", 
                  side_effect=PrometheusError("Prometheus connection failed")):
            with patch("src.main.rag_service.generate_report", 
                      new_callable=AsyncMock,
                      return_value={
                          "insight_analysis": "Test insight",
                          "recommendations": "Test recommendations"
                      }):
                response = client.post("/api/v1/generate_report", json=request_data)
                
                # Should still succeed despite Prometheus error
                assert response.status_code == 200
                data = response.json()
                assert data["status"] == "success"
                assert data["report"]["insight_analysis"] == "Test insight"

    @pytest.mark.asyncio
    async def test_generate_report_unexpected_prometheus_error(self, client):
        """Test report generation with unexpected error during Prometheus enrichment"""
        request_data = {
            "monitoring_data": {
                "主機": "test-host",
                "CPU使用率": "80%"
            }
        }
        
        with patch("src.main.rag_service.enrich_with_prometheus", 
                  side_effect=Exception("Unexpected error")):
            with patch("src.main.rag_service.generate_report", 
                      new_callable=AsyncMock,
                      return_value={
                          "insight_analysis": "Test insight",
                          "recommendations": "Test recommendations"
                      }):
                response = client.post("/api/v1/generate_report", json=request_data)
                
                # Should still succeed
                assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_generate_report_unexpected_error(self, client):
        """Test report generation with completely unexpected error"""
        request_data = {
            "monitoring_data": {"CPU": "80%"}
        }
        
        with patch("src.main.rag_service.generate_report", 
                  side_effect=RuntimeError("Unexpected runtime error")):
            response = client.post("/api/v1/generate_report", json=request_data)
            
            assert response.status_code == 500
            assert "An internal error occurred" in response.json()["detail"]

    def test_get_host_metrics_not_found(self, client):
        """Test get_host_metrics when no metrics are found"""
        with patch("src.main.rag_service.prometheus.get_host_metrics", 
                  new_callable=AsyncMock,
                  return_value={}):
            response = client.get("/api/v1/metrics/test-host")
            
            assert response.status_code == 404
            assert "No metrics found for host: test-host" in response.json()["detail"]

    def test_get_host_metrics_prometheus_error(self, client):
        """Test get_host_metrics with Prometheus error"""
        with patch("src.main.rag_service.prometheus.get_host_metrics", 
                  side_effect=PrometheusError("Connection timeout")):
            response = client.get("/api/v1/metrics/test-host")
            
            assert response.status_code == 503
            assert "Monitoring service unavailable" in response.json()["detail"]

    def test_get_host_metrics_unexpected_error(self, client):
        """Test get_host_metrics with unexpected error"""
        with patch("src.main.rag_service.prometheus.get_host_metrics", 
                  side_effect=Exception("Unexpected error")):
            response = client.get("/api/v1/metrics/test-host")
            
            assert response.status_code == 500
            assert "Failed to fetch metrics" in response.json()["detail"]

    def test_cache_info_with_data(self, client):
        """Test cache info endpoint with actual cache data"""
        mock_cache_info = {
            "hyde_cache": {
                "hits": 50,
                "misses": 50,
                "size": 100
            },
            "embedding_cache": {
                "hits": 75,
                "misses": 25,
                "size": 100
            }
        }
        
        with patch("src.main.rag_service.get_cache_info", return_value=mock_cache_info):
            response = client.get("/api/v1/cache/info")
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "success"
            assert data["cache_info"]["hyde_cache"]["hit_rate"] == "50.00%"
            assert data["cache_info"]["embedding_cache"]["hit_rate"] == "75.00%"

    def test_cache_info_zero_requests(self, client):
        """Test cache info endpoint when no requests have been made"""
        mock_cache_info = {
            "hyde_cache": {
                "hits": 0,
                "misses": 0,
                "size": 0
            },
            "embedding_cache": {
                "hits": 0,
                "misses": 0,
                "size": 0
            }
        }
        
        with patch("src.main.rag_service.get_cache_info", return_value=mock_cache_info):
            response = client.get("/api/v1/cache/info")
            
            assert response.status_code == 200
            data = response.json()
            assert data["cache_info"]["hyde_cache"]["hit_rate"] == "0.00%"
            assert data["cache_info"]["embedding_cache"]["hit_rate"] == "0.00%"

    def test_cache_info_cache_error(self, client):
        """Test cache info endpoint with CacheError"""
        with patch("src.main.rag_service.get_cache_info", 
                  side_effect=CacheError("Cache unavailable")):
            response = client.get("/api/v1/cache/info")
            
            assert response.status_code == 500
            assert "Cache operation failed" in response.json()["detail"]

    def test_cache_info_unexpected_error(self, client):
        """Test cache info endpoint with unexpected error"""
        with patch("src.main.rag_service.get_cache_info", 
                  side_effect=Exception("Unexpected error")):
            response = client.get("/api/v1/cache/info")
            
            assert response.status_code == 500
            assert "Failed to get cache information" in response.json()["detail"]

    def test_clear_cache_cache_error(self, client):
        """Test clear cache endpoint with CacheError"""
        with patch("src.main.rag_service.clear_cache", 
                  side_effect=CacheError("Cannot clear cache")):
            response = client.post("/api/v1/cache/clear")
            
            assert response.status_code == 500
            assert "Failed to clear cache: Cannot clear cache" in response.json()["detail"]

    def test_clear_cache_unexpected_error(self, client):
        """Test clear cache endpoint with unexpected error"""
        with patch("src.main.rag_service.clear_cache", 
                  side_effect=Exception("Unexpected error")):
            response = client.post("/api/v1/cache/clear")
            
            assert response.status_code == 500
            assert "Failed to clear cache" in response.json()["detail"]