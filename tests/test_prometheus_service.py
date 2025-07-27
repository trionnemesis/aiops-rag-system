import pytest
import aiohttp
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
from src.services.prometheus_service import PrometheusService
from src.config import settings

class TestPrometheusService:
    """Test cases for PrometheusService"""

    @pytest.fixture
    def prometheus_service(self):
        """為測試建立一個 PrometheusService 實例"""
        return PrometheusService()

    def _create_mock_session(self, json_data):
        """輔助函式，用於建立一個模擬的 aiohttp session"""
        mock_resp = AsyncMock()
        # 設定 mock 的 json() 方法回傳值
        mock_resp.json.return_value = json_data
        
        mock_session = AsyncMock()
        # 讓 get() 方法的回傳值也是一個非同步上下文管理器
        mock_session.get.return_value.__aenter__.return_value = mock_resp
        return mock_session

    def test_init(self, prometheus_service):
        """測試 PrometheusService 的初始化"""
        expected_url = f"http://{settings.prometheus_host}:{settings.prometheus_port}"
        assert prometheus_service.base_url == expected_url

    @pytest.mark.asyncio
    async def test_query_success(self, prometheus_service):
        """測試成功的 Prometheus 查詢"""
        mock_json_data = {
            "status": "success",
            "data": {"result": [{"value": [1234567890, "50.5"]}]}
        }
        mock_session = self._create_mock_session(mock_json_data)
        
        with patch("aiohttp.ClientSession", return_value=mock_session):
            result = await prometheus_service.query("up")
            assert result["result"][0]["value"][1] == "50.5"

    @pytest.mark.asyncio
    async def test_query_error(self, prometheus_service):
        """測試 Prometheus 查詢錯誤處理"""
        mock_json_data = {
            "status": "error", "errorType": "bad_data", "error": "Invalid query"
        }
        mock_session = self._create_mock_session(mock_json_data)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            with pytest.raises(Exception, match="Prometheus query failed"):
                await prometheus_service.query("invalid_query")
    
    @pytest.mark.asyncio
    async def test_get_host_metrics_all_metrics(self, prometheus_service):
        """測試 get_host_metrics 能取得所有指標"""
        hostname = "test-host"
        
        # 為不同指標設定模擬的回應
        cpu_response = {"status": "success", "data": {"result": [{"value": [0, "75.5"]}]}}
        mem_response = {"status": "success", "data": {"result": [{"value": [0, "60.2"]}]}}
        io_response = {"status": "success", "data": {"result": [{"value": [0, "10.3"]}]}}
        net_response = {"status": "success", "data": {"result": [{"value": [0, "16.0"]}]}} # 模擬加總後的值

        # 使用 AsyncMock 來模擬 query 方法，並設定其依序返回不同的結果
        mock_query = AsyncMock(side_effect=[
            cpu_response["data"],
            mem_response["data"],
            io_response["data"],
            net_response["data"]
        ])

        # 直接 patch service 物件的 query 方法
        with patch.object(prometheus_service, 'query', mock_query):
            metrics = await prometheus_service.get_host_metrics(hostname)
            
            assert metrics["CPU使用率"] == "75.5%"
            assert metrics["RAM使用率"] == "60.2%"
            assert metrics["磁碟I/O等待"] == "10.3%"
            assert metrics["網路流出量"] == "16 Mbps" # 網路流量的計算比較複雜，此處簡化
            assert mock_query.call_count == 4
