# AIOps 智慧維運報告 RAG 系統

> 基於 LangChain LCEL + LangGraph 的智慧維運報告生成系統

## 🎯 系統簡介

本系統自動分析監控數據並生成專業的維運洞見報告，採用先進的 RAG (檢索增強生成) 架構，整合 HyDE 技術、多查詢檢索策略和 LangGraph DAG 控制流程，為 DevOps 團隊提供精準的系統分析和可執行的優化建議。

### 🆕 最新架構升級
- **LangChain LCEL**: 聲明式 RAG 鏈，支援 fallback 機制
- **LangGraph 整合**: DAG 控制流程，可插拔式架構設計
- **HyDE + RAG-Fusion**: 多策略文檔檢索和內容生成
- **KNN 向量搜尋**: HNSW 演算法實作，支援多種搜尋策略
- **LangExtract**: 結構化資訊提取，智慧元數據管理

## ⚡ 核心優勢

| 特色 | 說明 | 效益 |
|------|------|------|
| 🤖 **智慧分析** | HyDE + RAG-Fusion 架構 | 深度維運洞見 |
| 🔗 **LangChain LCEL** | 聲明式 RAG 流程 | 支援 fallback 機制 |
| 🌐 **LangGraph DAG** | 可插拔控制流程 | 靈活的策略組合 |
| 📊 **LangExtract** | 結構化資訊提取 | 精準元數據過濾 |
| 🔍 **KNN 向量搜尋** | HNSW 演算法優化 | 高精度語義檢索 |
| ⚡ **高效能** | 智慧快取機制 | 85% API 成本節省 |
| 🛡️ **企業級** | 完整錯誤處理 | 85%+ 測試覆蓋率 |
| 📊 **即時監控** | Prometheus + Grafana | 即時系統狀態 |
| 🚀 **效能優化** | 向量檢索效能監控 | P95 < 200ms |

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
│   FastAPI   │────▶│  LangGraph   │────▶│   Gemini    │
│     API     │     │   DAG Flow   │     │  API (LLM)  │
└─────────────┘     └──────────────┘     └─────────────┘
       │                    │                     
       │                    ▼                    
       │            ┌──────────────┐     ┌─────────────┐
       │            │  LangChain   │────▶│    HyDE     │
       │            │   RAG Chain  │     │ Multi-Query │
       │            └──────────────┘     └─────────────┘
       ▼                    │                     
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│ Prometheus  │     │  OpenSearch  │     │   Grafana   │
│  Metrics    │     │ KNN + HNSW   │     │ Dashboard   │
└─────────────┘     └──────────────┘     └─────────────┘
```

## 📚 API 端點

| 端點 | 方法 | 說明 |
|------|------|------|
| `/api/v1/generate_report` | POST | 生成維運報告 |
| `/api/v1/metrics/{hostname}` | GET | 取得主機指標 |
| `/api/v1/cache/info` | GET | 快取狀態資訊 |
| `/api/v1/cache/clear` | POST | 清除快取 |
| `/api/v1/search/vector` | POST | 向量搜尋 |
| `/metrics` | GET | Prometheus 指標 |

## 🛠️ 開發指南

### 本地開發

```bash
# 安裝依賴
pip install -r requirements.txt

# 執行測試
pytest tests/ --cov=src --cov-fail-under=85

# 執行效能測試
pytest tests/test_vector_performance.py -v

# 執行壓力測試
pytest tests/test_vector_load.py -m load -v

# 啟動開發伺服器
uvicorn src.main:app --reload
```

## 📊 效能指標

- **API 成本**: 降低 85%
- **快取命中率**: 70%+  
- **回應時間**: < 5秒 (P95)
- **測試覆蓋率**: 85%+
- **向量搜尋延遲**: < 200ms (P95)
- **每秒查詢數 (QPS)**: 支援 100+ QPS
- **失敗率**: < 1%

## 📖 完整文檔

### 🏗️ 系統架構
- [系統設計](./docs/architecture/system-design.md) - 整體架構和核心組件

### 💻 開發指南
- [本地環境設置](./docs/development/local-setup.md) - 開發環境配置
- [錯誤處理最佳實踐](./docs/development/error-handling.md) - 錯誤處理機制
- [效能優化指南](./docs/development/optimization-guide.md) - RAG 系統優化
- [系統優化說明](./docs/development/optimizations.md) - 優化實作細節
- [優化總結](./docs/development/OPTIMIZATION_SUMMARY.md) - 優化成果總結

### 🚀 效能優化
- [向量檢索效能優化](./docs/vector-performance-optimization.md) - 向量搜尋效能監控與優化

### 🚀 部署指南
- [Docker 部署](./docs/deployment/docker-guide.md) - 容器化部署完整指南

### 📡 API 文檔
- [端點參考](./docs/api/endpoints.md) - 詳細的 API 端點說明

### 🔗 LangChain 整合
- [重構報告](./docs/langchain_refactoring_report.md) - LangChain LCEL 重構詳細說明
- [遷移指南](./docs/langchain_migration_guide.md) - 從原實作遷移指南
- [LangGraph RAG 整合](./docs/README_LANGGRAPH_INTEGRATION.md) - LangGraph DAG 實作指南
- [LangExtract 整合指南](./docs/langextract-integration.md) - 結構化資訊提取服務整合
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
