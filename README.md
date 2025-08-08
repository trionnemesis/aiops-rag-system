# AIOps 智慧維運報告 RAG 系統

> 基於 LangChain LCEL + LangGraph 的智慧維運報告生成系統，具備完整可觀測性

## 🎯 系統簡介

本系統自動分析監控數據並生成專業的維運洞見報告，採用先進的 RAG (檢索增強生成) 架構，整合 HyDE 技術、多查詢檢索策略和 LangGraph DAG 控制流程，為 DevOps 團隊提供精準的系統分析和可執行的優化建議。

### 🆕 最新架構升級
- **LangChain LCEL**: 聲明式 RAG 鏈，支援 fallback 機制
- **LangGraph 整合**: DAG 控制流程，可插拔式架構設計
- **HyDE + RAG-Fusion**: 多策略文檔檢索和內容生成
- **KNN 向量搜尋**: HNSW 演算法實作，支援多種搜尋策略
- **LangExtract**: 結構化資訊提取，智慧元數據管理
- **完整可觀測性**: 結構化日誌、分散式追蹤、度量指標收集
- **狀態持久化**: LangGraph 工作流程狀態管理與恢復
- **重試機制**: 智慧重試與錯誤處理策略

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
| 🔍 **可觀測性** | 結構化日誌 + 分散式追蹤 | 完整請求鏈路追蹤 |
| 💾 **狀態管理** | 工作流程狀態持久化 | 支援中斷恢復 |
| 🔄 **容錯機制** | 智慧重試策略 | 提升系統可靠性 |

## 🚀 快速開始

### 1. 一鍵部署

```bash
# Clone 專案
git clone https://github.com/[your-org]/aiops-rag-system.git
cd aiops-rag-system

# 設定環境變數
cp .env.example .env
# 編輯 .env，填入 Gemini API Key 和可觀測性配置

# 啟動服務
docker-compose up -d
```

### 2. 測試 API

```bash
curl -X POST http://localhost:8080/api/v1/rag/report \
  -H "Content-Type: application/json" \
  -d '{
    "query": "如何解決 Kubernetes Pod OOMKilled 問題"
  }'
```

### 3. 存取服務

- **API 文檔**: http://localhost:8080/docs
- **Grafana**: http://localhost:3000 (admin/admin)
- **Prometheus**: http://localhost:9090
- **Jaeger UI**: http://localhost:16686
- **OpenSearch Dashboards**: http://localhost:5601
- **Metrics**: http://localhost:8000/metrics

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
       │                    │                    │
       ▼                    ▼                    ▼
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   Loguru    │     │OpenTelemetry │     │   Jaeger    │
│ Structured  │     │  Tracing     │     │   Traces    │
│    Logs     │     └──────────────┘     └─────────────┘
└─────────────┘                                  
       │                                         
       ▼                                        
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│ LangExtract │     │    Redis     │     │  State DB   │
│  Metadata   │     │    Cache     │     │ Persistence │
└─────────────┘     └──────────────┘     └─────────────┘
```

## 🔍 可觀測性功能

### 結構化日誌
- **JSON 格式**：支援 Loki、Splunk、Elasticsearch
- **請求追蹤**：自動包含 request_id、node_name
- **上下文資訊**：執行時間、錯誤詳情、節點狀態

### 分散式追蹤
- **完整鏈路**：視覺化 LangGraph DAG 執行流程
- **節點耗時**：每個節點的詳細執行時間
- **錯誤定位**：快速找出失敗節點和原因

### 度量指標
- **系統健康**：QPS、延遲、錯誤率
- **資源使用**：Token 使用量、檢索文件數
- **業務指標**：答案品質分數、驗證結果

## 📚 API 端點

| 端點 | 方法 | 說明 |
|------|------|------|
| `/api/v1/rag/report` | POST | 生成 RAG 報告 |
| `/api/v1/rag/extract` | POST | 結構化資訊提取 |
| `/api/v1/health` | GET | 健康檢查 |
| `/api/v1/metrics` | GET | Prometheus 指標 |
| `/docs` | GET | Swagger API 文檔 |

## 🛠️ 開發指南

### 本地開發

```bash
# 安裝依賴
pip install -r requirements.txt

# 執行測試
pytest tests/ --cov=app --cov-fail-under=85

# 啟動開發伺服器
python -m app.main

# 檢視日誌（開發模式）
LOG_LEVEL=DEBUG JSON_LOGS=false python -m app.main
```

### 環境變數配置

```bash
# LLM 配置
GEMINI_API_KEY=your-api-key

# 日誌配置
LOG_LEVEL=INFO
JSON_LOGS=true
LOG_FILE=/var/log/rag/app.log

# 追蹤配置
JAEGER_ENDPOINT=localhost:6831
OTLP_ENDPOINT=localhost:4317
TRACE_CONSOLE=false

# 指標配置
METRICS_PORT=8000

# 快取配置
REDIS_URL=redis://localhost:6379
CACHE_TTL=3600

# 狀態持久化
STATE_DB_PATH=/data/state.db
ENABLE_STATE_PERSISTENCE=true
```

## 📊 效能指標

- **API 成本**: 降低 85%
- **快取命中率**: 70%+  
- **回應時間**: < 5秒 (P95)
- **測試覆蓋率**: 85%+
- **向量搜尋延遲**: < 200ms (P95)
- **每秒查詢數 (QPS)**: 支援 100+ QPS
- **失敗率**: < 1%
- **追蹤覆蓋率**: 100% 關鍵路徑
- **系統可用性**: 99.9%+

## 📖 完整文檔

### 🚀 快速開始
- [快速開始指南](./docs/quick-start.md) - 5分鐘內啟動系統

### 🏗️ 系統架構
- [系統設計](./docs/architecture/system-design.md) - 整體架構和核心組件
- [可觀測性指南](./docs/observability.md) - 結構化日誌、追蹤、指標詳細說明

### 💻 開發指南
- [本地環境設置](./docs/development/local-setup.md) - 開發環境配置
- [測試架構指南](./docs/development/test-architecture.md) - LangGraph 測試策略與實踐
- [錯誤處理最佳實踐](./docs/development/error-handling.md) - 錯誤處理機制
- [重試與錯誤處理](./docs/retry_and_error_handling.md) - 重試機制和錯誤處理策略
- [狀態持久化指南](./docs/state_persistence_guide.md) - LangGraph 狀態管理與持久化
- [效能優化指南](./docs/development/optimization-guide.md) - RAG 系統優化與實作細節
- [系統優化說明](./docs/development/optimizations.md) - 提示工程與監控優化

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

### 📝 其他資源
- [更新日誌](./docs/CHANGELOG.md) - 版本更新和功能變更記錄
- [環境設定範例](./.env.example) - 環境變數配置模板

## 🤝 貢獻

歡迎提交 Issue 和 PR！請確保：
- 遵循程式碼規範
- 維持測試覆蓋率 85%+
- 更新相關文件
- 包含適當的日誌和追蹤

## 📝 授權

MIT License - 詳見 [LICENSE](LICENSE)

---

⭐ 覺得有幫助嗎？給個星星吧！

📊 **最後更新**: 2024年1月
