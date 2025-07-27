import pytest
from unittest.mock import AsyncMock, patch
from src.services.prometheus_service import PrometheusService

class TestPrometheusService:
    @pytest.fixture
    def prometheus_service(self):
        return PrometheusService()

    # 關鍵修正：建立一個更完整的 mock session
    @pytest.mark.asyncio
    async def test_query_success(self, prometheus_service):
        mock_resp = AsyncMock()
        mock_resp.json.return_value = {
            "status": "success",
            "data": {"result": [{"value": [0, "50.5"]}]}
        }
        
        # get() 回傳一個 context manager
        mock_get = AsyncMock()
        mock_get.__aenter__.return_value = mock_resp
        
        # session 回傳一個 context manager，其 get 方法回傳上面的 mock_get
        mock_session = AsyncMock()
        mock_session.__aenter__.return_value.get.return_value = mock_get

        with patch("aiohttp.ClientSession", return_value=mock_session):
            data = await prometheus_service.query("up")
            assert data["result"][0]["value"][1] == "50.5"

    @pytest.mark.asyncio
    async def test_query_error(self, prometheus_service):
        mock_resp = AsyncMock()
        mock_resp.json.return_value = {"status": "error", "error": "test error"}

        mock_get = AsyncMock()
        mock_get.__aenter__.return_value = mock_resp
        
        mock_session = AsyncMock()
        mock_session.__aenter__.return_value.get.return_value = mock_get

        with patch("aiohttp.ClientSession", return_value=mock_session):
            with pytest.raises(Exception, match="Prometheus query failed"):
                await prometheus_service.query("invalid")
