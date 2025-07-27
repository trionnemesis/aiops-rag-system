import pytest
from httpx import AsyncClient, ASGITransport
from src.main import app
from src.services.rag_service import RAGService
from src.models.schemas import InsightReport
from src.services.exceptions import (
    VectorDBError, GeminiAPIError, PrometheusError, 
    HyDEGenerationError, DocumentRetrievalError, 
    ReportGenerationError, CacheError
)
from datetime import datetime

# --- 基礎測試 ---

@pytest.mark.asyncio
async def test_root():
    """測試根目錄 (/) 端點是否正常回應"""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        response = await ac.get("/")
    assert response.status_code == 200
    assert "message" in response.json()
    assert "version" in response.json()

@pytest.mark.asyncio
async def test_health():
    """測試健康檢查 (/health) 端點是否回報健康"""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        response = await ac.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert "version" in response.json()

# --- 成功案例測試 ---

@pytest.mark.asyncio
async def test_generate_report_success(monkeypatch):
    """測試報告生成成功的情況"""
    fake_report_model = InsightReport(
        insight_analysis="這是一個模擬的洞見分析。",
        recommendations="這是一條模擬的具體建議。",
        generated_at=datetime(2025, 1, 1, 12, 30, 0)
    )

    async def mock_generate_report(*args, **kwargs):
        return fake_report_model

    monkeypatch.setattr(RAGService, "generate_report", mock_generate_report)

    monitoring_data = {
        "主機": "web-test-01",
        "採集時間": "2025-01-01T10:00:00Z",
        "CPU使用率": "50%",
        "RAM使用率": "70%",
        "磁碟I/O等待": "5%",
        "網路流出量": "100 Mbps"
    }
    
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        response = await ac.post(
            "/api/v1/generate_report",
            json={"monitoring_data": monitoring_data}
        )
    
    assert response.status_code == 200
    response_json = response.json()
    
    assert response_json["status"] == "success"
    assert response_json["monitoring_data"] == monitoring_data
    assert "report" in response_json

# --- 輸入驗證測試 ---

@pytest.mark.asyncio
async def test_generate_report_empty_data():
    """測試空監控數據的驗證"""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        response = await ac.post(
            "/api/v1/generate_report",
            json={"monitoring_data": {}}
        )
    
    assert response.status_code == 400
    assert "Monitoring data cannot be empty" in response.json()["detail"]

@pytest.mark.asyncio
async def test_generate_report_invalid_request():
    """測試無效請求格式"""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        response = await ac.post(
            "/api/v1/generate_report",
            json={"invalid_field": "test"}
        )
    
    assert response.status_code == 422
    assert response.json()["status"] == "error"
    assert "Invalid request data" in response.json()["message"]

# --- 錯誤處理測試 ---

@pytest.mark.asyncio
async def test_generate_report_vector_db_error(monkeypatch):
    """測試向量資料庫錯誤處理"""
    async def mock_generate_report(*args, **kwargs):
        raise VectorDBError("Failed to connect to vector database")

    monkeypatch.setattr(RAGService, "generate_report", mock_generate_report)

    monitoring_data = {"主機": "test-01", "CPU使用率": "50%"}
    
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        response = await ac.post(
            "/api/v1/generate_report",
            json={"monitoring_data": monitoring_data}
        )
    
    assert response.status_code == 503
    response_json = response.json()
    assert response_json["status"] == "error"
    assert "Vector database service unavailable" in response_json["message"]
    assert response_json["error_type"] == "VectorDBError"

@pytest.mark.asyncio
async def test_generate_report_gemini_api_error(monkeypatch):
    """測試 Gemini API 錯誤處理"""
    async def mock_generate_report(*args, **kwargs):
        raise GeminiAPIError("API quota exceeded")

    monkeypatch.setattr(RAGService, "generate_report", mock_generate_report)

    monitoring_data = {"主機": "test-01", "CPU使用率": "50%"}
    
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        response = await ac.post(
            "/api/v1/generate_report",
            json={"monitoring_data": monitoring_data}
        )
    
    assert response.status_code == 503
    response_json = response.json()
    assert response_json["status"] == "error"
    assert "AI model service unavailable" in response_json["message"]

@pytest.mark.asyncio
async def test_generate_report_hyde_generation_error(monkeypatch):
    """測試 HyDE 生成錯誤處理"""
    async def mock_generate_report(*args, **kwargs):
        raise HyDEGenerationError("Failed to generate HyDE query")

    monkeypatch.setattr(RAGService, "generate_report", mock_generate_report)

    monitoring_data = {"主機": "test-01", "CPU使用率": "50%"}
    
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        response = await ac.post(
            "/api/v1/generate_report",
            json={"monitoring_data": monitoring_data}
        )
    
    assert response.status_code == 500
    response_json = response.json()
    assert "Failed to generate search query" in response_json["message"]

@pytest.mark.asyncio
async def test_prometheus_enrichment_error(monkeypatch):
    """測試 Prometheus 數據豐富錯誤處理（不應阻止報告生成）"""
    fake_report_model = InsightReport(
        insight_analysis="即使 Prometheus 失敗也能生成報告",
        recommendations="建議檢查 Prometheus 服務",
        generated_at=datetime(2025, 1, 1, 12, 30, 0)
    )
    
    async def mock_enrich_with_prometheus(*args, **kwargs):
        raise PrometheusError("Prometheus is down")
    
    async def mock_generate_report(*args, **kwargs):
        return fake_report_model

    monkeypatch.setattr(RAGService, "enrich_with_prometheus", mock_enrich_with_prometheus)
    monkeypatch.setattr(RAGService, "generate_report", mock_generate_report)

    monitoring_data = {"主機": "test-01", "CPU使用率": "50%"}
    
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        response = await ac.post(
            "/api/v1/generate_report",
            json={"monitoring_data": monitoring_data}
        )
    
    # 應該成功生成報告，即使 Prometheus 失敗
    assert response.status_code == 200
    assert response.json()["status"] == "success"

# --- Metrics 端點測試 ---

@pytest.mark.asyncio
async def test_get_host_metrics_empty_hostname():
    """測試空主機名稱"""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        response = await ac.get("/api/v1/metrics/")
    
    assert response.status_code == 404  # FastAPI 會返回 404 for empty path param

@pytest.mark.asyncio
async def test_get_host_metrics_prometheus_error(monkeypatch):
    """測試 Prometheus 錯誤"""
    from src.services.prometheus_service import PrometheusService
    
    async def mock_get_host_metrics(*args, **kwargs):
        raise PrometheusError("Connection timeout")
    
    monkeypatch.setattr(PrometheusService, "get_host_metrics", mock_get_host_metrics)
    
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        response = await ac.get("/api/v1/metrics/test-host")
    
    assert response.status_code == 503
    assert "Monitoring service unavailable" in response.json()["detail"]

# --- Cache 端點測試 ---

@pytest.mark.asyncio
async def test_cache_info_success(monkeypatch):
    """測試獲取快取資訊成功"""
    def mock_get_cache_info(*args, **kwargs):
        return {
            "hyde_cache": {"hits": 10, "misses": 5, "maxsize": 50, "currsize": 15},
            "embedding_cache": {"hits": 20, "misses": 10, "maxsize": 100, "currsize": 30}
        }
    
    monkeypatch.setattr(RAGService, "get_cache_info", mock_get_cache_info)
    
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        response = await ac.get("/api/v1/cache/info")
    
    assert response.status_code == 200
    response_json = response.json()
    assert response_json["status"] == "success"
    assert "cache_info" in response_json
    assert "hit_rate" in response_json["cache_info"]["hyde_cache"]
    assert "hit_rate" in response_json["cache_info"]["embedding_cache"]

@pytest.mark.asyncio
async def test_cache_clear_success(monkeypatch):
    """測試清除快取成功"""
    def mock_clear_cache(*args, **kwargs):
        pass
    
    monkeypatch.setattr(RAGService, "clear_cache", mock_clear_cache)
    
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        response = await ac.post("/api/v1/cache/clear")
    
    assert response.status_code == 200
    assert response.json()["status"] == "success"
    assert "cleared" in response.json()["message"]

@pytest.mark.asyncio
async def test_cache_clear_error(monkeypatch):
    """測試清除快取錯誤"""
    def mock_clear_cache(*args, **kwargs):
        raise CacheError("Failed to connect to cache backend")
    
    monkeypatch.setattr(RAGService, "clear_cache", mock_clear_cache)
    
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        response = await ac.post("/api/v1/cache/clear")
    
    assert response.status_code == 500
    assert "Failed to clear cache" in response.json()["detail"]
