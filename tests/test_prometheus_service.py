import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from datetime import datetime, timedelta
from src.services.prometheus_service import PrometheusService

class TestPrometheusService:
    @pytest.fixture
    def prometheus_service(self):
        return PrometheusService()

    # 關鍵修正：建立一個更完整的 mock session
    @pytest.mark.asyncio
    async def test_query_success(self, prometheus_service):
        mock_resp = AsyncMock()
        mock_resp.json = AsyncMock(return_value={
            "status": "success",
            "data": {"result": [{"value": [0, "50.5"]}]}
        })
        
        # 建立一個正確的異步上下文管理器
        mock_get_cm = MagicMock()
        mock_get_cm.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_get_cm.__aexit__ = AsyncMock(return_value=None)
        
        # 建立 session mock，get 應該是一個同步方法返回異步上下文管理器
        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=mock_get_cm)
        
        # 建立 ClientSession 的異步上下文管理器
        mock_client_session = MagicMock()
        mock_client_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_client_session.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_client_session):
            data = await prometheus_service.query("up")
            assert data["result"][0]["value"][1] == "50.5"

    @pytest.mark.asyncio
    async def test_query_error(self, prometheus_service):
        mock_resp = AsyncMock()
        mock_resp.json = AsyncMock(return_value={"status": "error", "error": "test error"})

        # 建立一個正確的異步上下文管理器
        mock_get_cm = MagicMock()
        mock_get_cm.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_get_cm.__aexit__ = AsyncMock(return_value=None)
        
        # 建立 session mock，get 應該是一個同步方法返回異步上下文管理器
        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=mock_get_cm)
        
        # 建立 ClientSession 的異步上下文管理器
        mock_client_session = MagicMock()
        mock_client_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_client_session.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_client_session):
            with pytest.raises(Exception, match="Prometheus query failed"):
                await prometheus_service.query("invalid")

    @pytest.mark.asyncio
    async def test_query_range_success(self, prometheus_service):
        """測試 query_range 方法"""
        mock_resp = AsyncMock()
        mock_resp.json = AsyncMock(return_value={
            "status": "success",
            "data": {"result": [{"values": [[1234567890, "100"], [1234567900, "200"]]}]}
        })
        
        # 建立一個正確的異步上下文管理器
        mock_get_cm = MagicMock()
        mock_get_cm.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_get_cm.__aexit__ = AsyncMock(return_value=None)
        
        # 建立 session mock
        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=mock_get_cm)
        
        # 建立 ClientSession 的異步上下文管理器
        mock_client_session = MagicMock()
        mock_client_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_client_session.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_client_session):
            start = datetime.now() - timedelta(hours=1)
            end = datetime.now()
            data = await prometheus_service.query_range("up", start, end)
            assert data["result"][0]["values"][0][1] == "100"

    @pytest.mark.asyncio
    async def test_query_range_error(self, prometheus_service):
        """測試 query_range 方法的錯誤處理"""
        mock_resp = AsyncMock()
        mock_resp.json = AsyncMock(return_value={"status": "error", "error": "range query failed"})

        # 建立一個正確的異步上下文管理器
        mock_get_cm = MagicMock()
        mock_get_cm.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_get_cm.__aexit__ = AsyncMock(return_value=None)
        
        # 建立 session mock
        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=mock_get_cm)
        
        # 建立 ClientSession 的異步上下文管理器
        mock_client_session = MagicMock()
        mock_client_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_client_session.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_client_session):
            start = datetime.now() - timedelta(hours=1)
            end = datetime.now()
            with pytest.raises(Exception, match="Prometheus range query failed"):
                await prometheus_service.query_range("invalid", start, end)

    @pytest.mark.asyncio
    async def test_get_host_metrics(self, prometheus_service):
        """測試 get_host_metrics 方法"""
        # Mock 多個查詢的回應
        cpu_resp = {
            "status": "success",
            "data": {"result": [{"value": [0, "25.5"]}]}
        }
        mem_resp = {
            "status": "success",
            "data": {"result": [{"value": [0, "65.2"]}]}
        }
        io_resp = {
            "status": "success",
            "data": {"result": [{"value": [0, "10.8"]}]}
        }
        net_resp = {
            "status": "success",
            "data": {"result": [
                {"value": [0, "10.5"]},  # 10.5 Mbps
                {"value": [0, "20.3"]}   # 20.3 Mbps
            ]}
        }
        
        # 建立 query mock
        query_mock = AsyncMock(side_effect=[cpu_resp["data"], mem_resp["data"], 
                                           io_resp["data"], net_resp["data"]])
        
        with patch.object(prometheus_service, 'query', query_mock):
            metrics = await prometheus_service.get_host_metrics("test-host")
            
            assert metrics["CPU使用率"] == "25.5%"
            assert metrics["RAM使用率"] == "65.2%"
            assert metrics["磁碟I/O等待"] == "10.8%"
            assert metrics["網路流出量"] == "31 Mbps"  # 10.5 + 20.3 = 30.8 ≈ 31
            
            # 確認調用了4次查詢
            assert query_mock.call_count == 4

    @pytest.mark.asyncio
    async def test_get_host_metrics_empty_results(self, prometheus_service):
        """測試 get_host_metrics 方法處理空結果"""
        empty_resp = {
            "status": "success",
            "data": {"result": []}
        }
        
        # 建立 query mock，返回空結果
        query_mock = AsyncMock(return_value=empty_resp["data"])
        
        with patch.object(prometheus_service, 'query', query_mock):
            metrics = await prometheus_service.get_host_metrics("test-host")
            
            # 空結果不應該有任何指標
            assert "CPU使用率" not in metrics
            assert "RAM使用率" not in metrics
            assert "磁碟I/O等待" not in metrics
            assert "網路流出量" not in metrics
