# tests/test_prometheus_service.py
# (使用與上一則回覆中相同的修正後程式碼，確保所有 async with 都被妥善處理)
import pytest
from unittest.mock import AsyncMock, patch
from src.services.prometheus_service import PrometheusService

class TestPrometheusService:
    @pytest.fixture
    def prometheus_service(self):
        return PrometheusService()

    @pytest.mark.asyncio
    async def test_query_success(self, prometheus_service):
        """測試成功的 Prometheus 查詢"""
        mock_response = AsyncMock()
        mock_response.json.return_value = {
            "status": "success",
            "data": {"result": [{"value": [1234567890, "50.5"]}]}
        }
        
        # 建立一個模擬的 session 上下文管理器
        mock_session_context = AsyncMock()
        mock_session_context.__aenter__.return_value.get.return_value.__aenter__.return_value = mock_response

        with patch("aiohttp.ClientSession", return_value=mock_session_context):
            result = await prometheus_service.query("up")
            assert result["data"]["result"][0]["value"][1] == "50.5"

    @pytest.mark.asyncio
    async def test_query_error(self, prometheus_service):
        """測試 Prometheus 查詢錯誤處理"""
        mock_response = AsyncMock()
        mock_response.json.return_value = {"status": "error"}

        mock_session_context = AsyncMock()
        mock_session_context.__aenter__.return_value.get.return_value.__aenter__.return_value = mock_response

        with patch("aiohttp.ClientSession", return_value=mock_session_context):
            with pytest.raises(Exception, match="Prometheus query failed"):
                await prometheus_service.query("invalid")
