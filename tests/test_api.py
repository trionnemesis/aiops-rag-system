import pytest
from httpx import AsyncClient
from src.main import app  # 假設您的 FastAPI app 物件在 src/main.py
from src.services.rag_service import RAGService  # 假設您的服務類別路徑
from src.models.schemas import InsightReport  # 假設您的 Pydantic 模型路徑
from datetime import datetime

# --- 不需要修改的基礎測試 ---

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

# --- 修正並優化後的報告生成測試 ---

@pytest.mark.asyncio
async def test_generate_report_with_mock(monkeypatch):
    """
    測試報告生成 (/api/v1/generate_report) 端點，並優化以確保在 CI/CD 環境中的穩定性。
    
    使用 monkeypatch 來模擬 (mock) RAGService 的 generate_report 方法，
    使其返回一個固定的、可預測的報告，從而避免在測試中實際呼叫外部 API (如 Gemini)。
    """
    # 1. 準備一個固定的、可預期的報告物件。
    #    使用一個固定的 datetime 物件取代不確定的 datetime.now()，
    #    這可以消除測試的易變性，確保每次運行的結果都相同。
    fake_report_model = InsightReport(
        insight_analysis="這是一個模擬的洞見分析。",
        recommendations="這是一條模擬的具體建議。",
        generated_at=datetime(2025, 1, 1, 12, 30, 0) # 使用固定的時間
    )

    # 2. 定義一個非同步的模擬函式 (mock function)。
    #    這個函式將會取代原始的 RAGService.generate_report。
    #    它接收與原始函式相同的參數，並回傳我們預先準備好的假報告模型。
    async def mock_generate_report(*args, **kwargs):
        return fake_report_model

    # 3. 使用 monkeypatch.setattr 來用我們的模擬函式替換掉真實的函式。
    #    參數: (目標類別, "目標方法名稱", 我們的模擬函式)
    monkeypatch.setattr(RAGService, "generate_report", mock_generate_report)

    # 4. 準備 API 請求的 body 資料。
    monitoring_data = {
        "主機": "web-test-01",
        "採集時間": "2025-01-01T10:00:00Z",
        "CPU使用率": "50%",
        "RAM使用率": "70%",
        "磁碟I/O等待": "5%",
        "網路流出量": "100 Mbps"
    }
    
    # 5. 使用 AsyncClient 發送 API 請求。
    #    在處理這個請求時，FastAPI 內部呼叫到的將是我們的模擬函式。
    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.post(
            "/api/v1/generate_report",
            json={"monitoring_data": monitoring_data}
        )
    
    # 6. 驗證 API 回應的基本狀態和資料。
    assert response.status_code == 200
    response_json = response.json()
    
    assert response_json["status"] == "success"
    assert response_json["monitoring_data"] == monitoring_data

    # 7. 驗證報告內容是否完全符合預期。
    #    FastAPI 會自動將 Pydantic 模型序列化為字典，其中 datetime 物件會被轉換為 ISO 格式的字串。
    #    我們建立一個與預期 JSON 結構完全相同的字典來進行比對，這是最可靠的方法。
    expected_report_dict = {
        "insight_analysis": "這是一個模擬的洞見分析。",
        "recommendations": "這是一條模擬的具體建議。",
        "generated_at": "2025-01-01T12:30:00" # 這是 datetime(2025, 1, 1, 12, 30, 0) 序列化後的結果
    }

    assert "report" in response_json
    assert response_json["report"] == expected_report_dict
