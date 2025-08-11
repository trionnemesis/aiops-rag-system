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
- **強型別化**: 使用 Pydantic v2 BaseModel 進行狀態和輸入驗證
- **容器健康檢查**: Docker Compose 配置包含健康檢查和自動重啟

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

### 🎨 KNN 向量搜尋亮點功能

#### 多策略搜尋支援
系統實作了四種進階搜尋策略，可根據不同場景選擇最優方案：

1. **純向量搜尋 (KNN_ONLY)**
   - 基於 HNSW 演算法的高效向量檢索
   - 適合語義相似度匹配
   - 支援自訂 num_candidates 和 min_score 參數

2. **混合搜尋 (HYBRID)**
   - 結合向量搜尋與 BM25 文字搜尋
   - 同時考慮語義相似度和關鍵詞匹配
   - 支援結果高亮顯示

3. **多向量搜尋 (MULTI_VECTOR)**
   - 自動生成查詢變體進行多維度檢索
   - 提高召回率和搜尋全面性
   - 智慧去重和結果融合

4. **重新排序搜尋 (RERANK)**
   - 先寬鬆檢索後精準重排
   - 結合語義相似度和關鍵詞匹配度
   - 適合高精度需求場景

#### 強型別驗證與資料模型
使用 Pydantic v2 BaseModel 實現完整的型別安全：

- **搜尋參數驗證** (`KNNSearchParams`)：
  - 自動驗證 k 值範圍 (1-50)
  - 確保 num_candidates 合理性
  - 過濾條件結構化驗證

- **搜尋結果模型** (`SearchResult`)：
  - 標準化的結果格式
  - 包含分數、元數據、高亮等完整資訊
  - 便於下游處理和展示

- **RAG 狀態管理** (`RAGState`)：
  - LangGraph 工作流程狀態強型別化
  - 自動驗證查詢長度 (1-1000 字元)
  - 支援最多 100 個原始文本輸入
  - 內建欄位驗證器確保資料品質

#### 效能監控與可觀測性
- 每個搜尋策略的獨立延遲監控
- 結果數量和品質指標追蹤
- HNSW ef_search 參數動態調整
- 完整的 Prometheus 指標整合

## 🚀 快速開始

### 1. 一鍵部署

```bash
# Clone 專案
git clone https://github.com/trionnemesis/aiops-rag-system.git
cd aiops-rag-system

# 設定環境變數
cp .env.example .env
# 編輯 .env，填入 Gemini API Key 和可觀測性配置

# 啟動服務（docker-compose.yml 已包含健康檢查和重啟策略）
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

### 核心模組結構

```
aiops-rag-system/
├── app/                      # FastAPI 應用程式
│   ├── api/                  # API 路由與端點
│   │   ├── routes.py         # 主要 API 路由
│   │   ├── knn_langchain_bridge.py  # KNN 與 LangChain 整合
│   │   └── example_integration.py   # 整合範例
│   ├── graph/                # LangGraph 工作流程
│   │   └── build.py          # DAG 流程建構
│   └── observability/        # 可觀測性功能
│       ├── logging.py        # 結構化日誌
│       ├── metrics.py        # Prometheus 指標
│       └── tracing.py        # 分散式追蹤
├── src/                      # 核心服務層
│   ├── services/             # 商業邏輯服務
│   │   ├── knn_search_service.py    # KNN 向量搜尋
│   │   ├── opensearch_service.py    # OpenSearch 整合
│   │   ├── gemini_service.py        # Gemini LLM 服務
│   │   ├── rag_service.py           # RAG 核心邏輯
│   │   ├── prometheus_service.py    # 指標收集
│   │   └── langchain/               # LangChain 整合
│   ├── models/               # 資料模型
│   ├── config/               # 配置管理
│   └── utils/                # 工具函式
├── tests/                    # 測試套件
├── scripts/                  # 部署與管理腳本
├── examples/                 # 使用範例
├── configs/                  # 配置檔案
└── docs/                     # 專案文檔
    ├── api/                  # API 文檔
    ├── architecture/         # 架構設計文檔
    ├── development/          # 開發指南
    ├── deployment/           # 部署文檔
    └── testing/              # 測試相關文檔
```

### 系統架構圖

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
| `/api/v1/knn/search` | POST | KNN 向量搜尋（支援多策略） |
| `/api/v1/knn/explain` | POST | 解釋搜尋結果評分 |
| `/api/v1/health` | GET | 健康檢查 |
| `/metrics` | GET | Prometheus 指標 |
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

### 強型別化和輸入驗證

系統使用 Pydantic v2 BaseModel 進行狀態管理和 API 輸入驗證：

- **RAGState**: LangGraph 工作流程狀態使用 Pydantic BaseModel
  - 自動驗證輸入長度和格式
  - 提供預設值和欄位限制
  - 支援 LangGraph 的字典介面相容性

- **API 請求驗證**: 
  - 查詢最大長度: 1000 字元
  - 原始文本列表最大項目: 100
  - 自動清理和驗證輸入資料

### KNN 向量搜尋使用範例

```python
from src.services.knn_search_service import (
    KNNSearchService, 
    KNNSearchParams, 
    SearchStrategy
)

# 初始化服務
search_service = KNNSearchService(
    index_name="aiops_knowledge_base"
)

# 執行混合搜尋
results = await search_service.knn_search(
    query_text="Apache 記憶體洩漏問題",
    params=KNNSearchParams(
        k=10,
        num_candidates=100,
        min_score=0.7
    ),
    strategy=SearchStrategy.HYBRID
)

# 處理結果
for result in results:
    print(f"標題: {result.title}")
    print(f"分數: {result.score}")
    print(f"高亮: {result.highlights}")
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
- [KNN 向量搜尋架構](./docs/architecture/knn-vector-search.md) - 向量搜尋系統設計
- [可觀測性指南](./docs/observability.md) - 結構化日誌、追蹤、指標詳細說明

### 💻 開發指南
- [本地環境設置](./docs/development/local-setup.md) - 開發環境配置
- [測試架構指南](./docs/development/test-architecture.md) - LangGraph 測試策略與實踐
- [錯誤處理最佳實踐](./docs/development/error-handling.md) - 錯誤處理機制
- [重試與錯誤處理](./docs/retry_and_error_handling.md) - 重試機制和錯誤處理策略
- [狀態持久化指南](./docs/state_persistence_guide.md) - LangGraph 狀態管理與持久化
- [效能優化指南](./docs/development/optimization-guide.md) - RAG 系統優化與實作細節
- [系統優化說明](./docs/development/optimizations.md) - 提示工程與監控優化
- [KNN 索引指南](./docs/development/knn-index-guide.md) - KNN 索引建立與管理

### 🧪 測試文檔
- [測試架構](./docs/testing/README_new_tests.md) - 測試策略與實踐

### 🚀 效能優化
- [向量檢索效能優化](./docs/vector-performance-optimization.md) - 向量搜尋效能監控與優化

### 🚀 部署指南
- [Docker 部署](./docs/deployment/docker-guide.md) - 容器化部署完整指南

### 📡 API 文檔
- [端點參考](./docs/api/endpoints.md) - 詳細的 API 端點說明
- [KNN 搜尋 API](./docs/api/knn-search-api.md) - KNN 搜尋 API 詳細說明

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

📊 **最後更新**: 2024年12月
