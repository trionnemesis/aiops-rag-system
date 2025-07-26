import pytest
from httpx import AsyncClient
from src.main import app
import json

@pytest.mark.asyncio
async def test_root():
    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.get("/")
    assert response.status_code == 200
    assert "message" in response.json()

@pytest.mark.asyncio
async def test_health():
    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

@pytest.mark.asyncio
async def test_generate_report():
    monitoring_data = {
        "主機": "web-test-01",
        "採集時間": "2025-01-01T10:00:00Z",
        "CPU使用率": "50%",
        "RAM使用率": "70%",
        "磁碟I/O等待": "5%",
        "網路流出量": "100 Mbps"
    }
    
    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.post(
            "/api/v1/generate_report",
            json={"monitoring_data": monitoring_data}
        )
    
    # 這個測試可能需要 mock Gemini API
    # assert response.status_code == 200
    # assert "report" in response.json()