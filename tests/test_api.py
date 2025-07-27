import pytest
from httpx import AsyncClient
from src.main import app
from src.services.rag_service import RAGService
from src.models.schemas import InsightReport
from datetime import datetime
import json

# 測試 API 的基本端點，這部分通常不需要修改
@pytest.mark.asyncio
async def test_root():
    """測試根目錄 (/) 端點是否正常回應"""
    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.get("/")
    assert response.status_code == 200
    assert "message" in response.json()

@pytest.mark.asyncio
async def test_health():
    """測試健康檢查 (/health) 端點是否回報健康"""
    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

# --- 調整後的 test_generate_report ---
@pytest.mark.asyncio
async def test_generate_report_with_mock(monkeypatch):
    """
    測試報告生成 (/api/v1/generate_report) 端點。
    
    使用 monkeypatch 來模擬 (mock) RAGService 的 generate_report 方法，
    使其返回一個固定的假報告，從而避免在測試中實際呼叫 Gemini API。
    """
    # 1. 準備一個假的、可預期的報告物件，作為模擬函式的回傳值
    fake_report = InsightReport(
        insight_analysis="這是一個模擬的洞見分析。",
        recommendations="這是一條模擬的具體建議。",
        generated_at=datetime.now()
    )

    # 2. 定義一個非同步的模擬函式
    #    這個函式將會取代原始的 RAGService.generate_report
    #    它接收與原始函式相同的參數，並回傳我們預先準備好的假報告
    async def mock_generate_report(*args, **kwargs):
        return fake_report

    # 3. 使用 monkeypatch.setattr 來替換真實的函式
    #    參數: (目標類別, "目標方法名稱", 我們的模擬函式)
    monkeypatch.setattr(RAGService, "generate_report", mock_generate_report)

    # 4. 準備 API 請求的資料
    monitoring_data = {
        "主機": "web-test-01",
        "採集時間": "2025-01-01T10:00:00Z",
        "CPU使用率": "50%",
        "RAM使用率": "70%",
        "磁碟I/O等待": "5%",
        "網路流出量": "100 Mbps"
    }
    
    # 5. 像平常一樣發送 API 請求
    #    FastAPI 在處理這個請求時，呼叫到的將是我們的模擬函式，而不是真實的函式
    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.post(
            "/api/v1/generate_report",
            json={"monitoring_data": monitoring_data}
        )
    
    # 6. 驗證 API 回應是否符合預期
    #    現在我們可以安全地取消註解，進行完整的斷言 (assert)
    assert response.status_code == 200
    
    response_json = response.json()
    assert response_json["status"] == "success"
    assert "report" in response_json
    assert response_json["report"]["insight_analysis"] == "這是一個模擬的洞見分析。"
    assert response_json["report"]["recommendations"] == "這是一條模擬的具體建議。"
    assert response_json["monitoring_data"]["主機"] == "web-test-01"

