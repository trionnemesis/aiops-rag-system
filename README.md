# AIOps 智慧維運報告 RAG 系統

> 基於 LangChain LCEL 和 HyDE 技術的智慧維運報告生成系統

## 🎯 系統簡介

本系統自動分析監控數據並生成專業的維運洞見報告，採用先進的 RAG (檢索增強生成) 架構，結合 HyDE 技術和多查詢檢索策略，為 DevOps 團隊提供精準的系統分析和可執行的優化建議。

## ⚡ 核心優勢

| 特色 | 說明 | 效益 |
|------|------|------|
| 🤖 **智慧分析** | HyDE + RAG-Fusion 架構 | 深度維運洞見 |
| 🔗 **LangChain LCEL** | 聲明式 RAG 流程 | 支援 fallback 機制 |
| ⚡ **高效能** | 智慧快取機制 | 85% API 成本節省 |
| 🛡️ **企業級** | 完整錯誤處理 | 85%+ 測試覆蓋率 |
| 📊 **即時監控** | Prometheus + Grafana | 即時系統狀態 |

## 🚀 快速開始

### 1. 一鍵部署

```bash
# Clone 專案
git clone https://github.com/your-username/aiops-rag-system.git
cd aiops-rag-system

# 設定環境變數
cp .env.example .env
# 編輯 .env，填入 Gemini API Key

# 啟動服務
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

## 📊 效能指標

- **API 成本**: 降低 85%
- **快取命中率**: 70%+  
- **回應時間**: < 5秒 (P95)
- **測試覆蓋率**: 85%+

## 📖 完整文檔

### 🏗️ 系統架構
- [系統設計](./docs/architecture/system-design.md) - 整體架構和核心組件

### 💻 開發指南
- [本地環境設置](./docs/development/local-setup.md) - 開發環境配置
- [錯誤處理最佳實踐](./docs/development/error-handling.md) - 錯誤處理機制
- [效能優化指南](./docs/development/optimization-guide.md) - RAG 系統優化
- [系統優化說明](./docs/development/optimizations.md) - 優化實作細節
- [優化總結](./docs/development/OPTIMIZATION_SUMMARY.md) - 優化成果總結

### 🚀 部署指南
- [Docker 部署](./docs/deployment/docker-guide.md) - 容器化部署完整指南

### 📡 API 文檔
- [端點參考](./docs/api/endpoints.md) - 詳細的 API 端點說明

### 🔗 LangChain 整合
- [重構報告](./docs/langchain_refactoring_report.md) - LangChain LCEL 重構詳細說明
- [遷移指南](./docs/langchain_migration_guide.md) - 從原實作遷移指南
- [GitHub Actions 變更](./docs/github-actions-changes.md) - CI/CD 配置更新

### 📚 文檔索引
- [文檔目錄](./docs/README.md) - 完整文檔導航和說明

## 🤝 貢獻

歡迎提交 Issue 和 PR！請確保：
- 遵循程式碼規範
- 維持測試覆蓋率 85%+
- 更新相關文件

## 📝 授權

MIT License - 詳見 [LICENSE](LICENSE)

---

⭐ 覺得有幫助嗎？給個星星吧！
