# AIOps 智慧維運報告 RAG 系統

基於 LangChain LCEL 和 HyDE 技術的智慧維運報告生成系統，自動分析監控數據並生成專業的維運洞見報告。

## 🚀 核心特色

- **🤖 智慧分析**: HyDE + RAG 架構，生成深度維運洞見
- **🔗 LangChain LCEL**: 聲明式 RAG 流程，支援 fallback 機制
- **⚡ 高效能**: 85% API 成本節省，內建智慧快取
- **🛡️ 企業級**: 完整錯誤處理、測試覆蓋率 85%+
- **📊 即時監控**: 整合 Prometheus + Grafana

## 📦 快速開始

### 1. 使用 Docker Compose（推薦）

```bash
# Clone 專案
git clone https://github.com/your-username/aiops-rag-system.git
cd aiops-rag-system

# 設定環境變數
cp .env.example .env
# 編輯 .env，填入 Gemini API Key

# 一鍵啟動
docker-compose up -d
```

### 2. 測試 API

```bash
curl -X POST http://localhost:8000/api/v1/generate_report \
  -H "Content-Type: application/json" \
  -d '{
    "monitoring_data": {
      "主機": "web-prod-03",
      "CPU使用率": "75%",
      "RAM使用率": "95%"
    }
  }'
```

### 3. 存取服務

- **API 文檔**: http://localhost:8000/docs
- **Grafana**: http://localhost:3000 (admin/admin)
- **Prometheus**: http://localhost:9090

## 🏗️ 系統架構

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   FastAPI   │────▶│  LangChain   │────▶│   Gemini    │
│     API     │     │   RAG Chain  │     │  API (LLM)  │
└─────────────┘     └──────────────┘     └─────────────┘
       │                    │                     
       ▼                    ▼                    
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│ Prometheus  │     │  OpenSearch  │     │   Grafana   │
│  Metrics    │     │Vector Store  │     │ Dashboard   │
└─────────────┘     └──────────────┘     └─────────────┘
```

## 💡 主要功能

### 1. 智慧報告生成
- HyDE 技術生成假設性文檔，提升檢索品質
- 自動 fallback 機制，確保服務穩定性
- 支援多維度監控數據分析

### 2. 企業級錯誤處理
- 細分的業務邏輯例外（VectorDBError、GeminiAPIError 等）
- 全域錯誤處理器，統一回應格式
- 請求驗證與詳細錯誤訊息

### 3. 完整的 CI/CD
- 自動化測試（覆蓋率 85%+）
- 安全掃描（pip-audit、truffleHog）
- 程式碼品質檢查（black、pylint、mypy）
- 容器漏洞掃描（Trivy）

## 📚 API 端點

| 端點 | 方法 | 說明 |
|------|------|------|
| `/api/v1/generate_report` | POST | 生成維運報告 |
| `/api/v1/metrics/{hostname}` | GET | 取得主機指標 |
| `/api/v1/cache/info` | GET | 快取狀態資訊 |
| `/api/v1/cache/clear` | POST | 清除快取 |

## 🛠️ 開發指南

### 本地開發

```bash
# 安裝依賴
pip install -r requirements.txt

# 執行測試
pytest tests/ --cov=src --cov-fail-under=85

# 啟動開發伺服器
uvicorn src.main:app --reload
```

### 使用自定義例外

```python
from src.services.exceptions import VectorDBError, GeminiAPIError

# 在服務層拋出特定例外
if vector_db_connection_failed:
    raise VectorDBError("Failed to connect to OpenSearch")

# API 層會自動處理並回傳適當的 HTTP 狀態碼
```

## 📊 效能指標

- **API 成本**: 降低 85%
- **快取命中率**: 70%+  
- **回應時間**: < 5秒 (P95)
- **測試覆蓋率**: 85%+

## 📖 進階文檔

- [系統架構設計](./docs/architecture/system-design.md)
- [LangChain 整合指南](./docs/langchain_refactoring_report.md)
- [錯誤處理最佳實踐](./docs/development/error-handling.md)
- [效能優化指南](./docs/development/optimization-guide.md)

## 🤝 貢獻

歡迎提交 Issue 和 PR！請確保：
- 遵循程式碼規範
- 維持測試覆蓋率 85%+
- 更新相關文件

## 📝 授權

MIT License - 詳見 [LICENSE](LICENSE)

---

⭐ 覺得有幫助嗎？給個星星吧！
