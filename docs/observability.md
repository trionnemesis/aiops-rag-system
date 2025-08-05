# 可觀測性指南 (Observability Guide)

本文檔說明 LangGraph RAG 系統的可觀測性功能，包括結構化日誌、分散式追蹤和度量指標收集。

## 目錄

1. [概述](#概述)
2. [結構化日誌](#結構化日誌-structured-logging)
3. [分散式追蹤](#分散式追蹤-distributed-tracing)
4. [度量指標收集](#度量指標收集-metrics-collection)
5. [配置指南](#配置指南)
6. [監控儀表板](#監控儀表板)
7. [最佳實踐](#最佳實踐)

## 概述

我們的可觀測性解決方案提供三個關鍵功能：

- **結構化日誌**：使用 Loguru 實現 JSON 格式日誌，便於在 Loki、Splunk 或 Elasticsearch 中查詢
- **分散式追蹤**：使用 OpenTelemetry 追蹤請求在 LangGraph DAG 中的完整流程
- **度量指標**：使用 Prometheus 收集系統健康狀況和效能指標

## 結構化日誌 (Structured Logging)

### 功能特點

- JSON 格式輸出，支援 Loki、Splunk、Elasticsearch 等日誌收集系統
- 自動包含請求 ID、節點名稱、執行時間等上下文資訊
- 支援日誌輪替和壓縮
- 可配置的日誌級別和輸出格式

### 日誌格式範例

```json
{
  "timestamp": "2024-01-15T10:30:45.123Z",
  "level": "INFO",
  "message": "Processing RAG request",
  "module": "app.api.routes",
  "function": "rag_report",
  "line": 45,
  "request_id": "550e8400-e29b-41d4-a716-446655440000",
  "node_name": "retrieve",
  "extra": {
    "query": "如何解決 Kubernetes Pod OOMKilled 問題",
    "doc_count": 8
  }
}
```

### 關鍵日誌欄位

| 欄位 | 說明 | 範例 |
|-----|------|------|
| `request_id` | 唯一請求識別碼 | `550e8400-e29b-41d4-a716-446655440000` |
| `node_name` | LangGraph 節點名稱 | `extract`, `plan`, `retrieve`, `synthesize`, `validate` |
| `user_id` | 使用者識別碼（選填） | `user123` |
| `session_id` | 會話識別碼（選填） | `session456` |
| `error` | 錯誤訊息 | `Connection timeout` |
| `error_type` | 錯誤類型 | `TimeoutError` |

### 查詢範例

#### Loki 查詢
```logql
{job="langgraph-rag"} |= "request_id" |= "550e8400" | json
```

#### Elasticsearch 查詢
```json
{
  "query": {
    "bool": {
      "must": [
        {"term": {"request_id": "550e8400-e29b-41d4-a716-446655440000"}},
        {"range": {"timestamp": {"gte": "now-1h"}}}
      ]
    }
  }
}
```

## 分散式追蹤 (Distributed Tracing)

### 功能特點

- 視覺化請求在 LangGraph DAG 中的完整流程
- 每個節點的執行時間和依賴關係
- 支援 Jaeger 和 OTLP 協議
- 自動追蹤 FastAPI 和 HTTP 客戶端調用

### 追蹤架構

```
rag_report (API Entry)
├── extract (如果啟用)
│   └── LangExtract Service
├── plan
│   ├── HyDE Generation (如果啟用)
│   └── Multi-Query Generation (如果啟用)
├── retrieve
│   ├── Vector Search
│   └── BM25 Search (如果啟用)
├── synthesize
│   └── LLM Generation
└── validate
```

### 關鍵追蹤屬性

| 屬性 | 說明 | 範例 |
|-----|------|------|
| `node.name` | 節點名稱 | `retrieve` |
| `node.execution_time_ms` | 節點執行時間（毫秒） | `245` |
| `request.id` | 請求 ID | `550e8400-e29b-41d4-a716-446655440000` |
| `query.text` | 查詢文本（前100字符） | `如何解決 Kubernetes...` |
| `retrieve.doc_count` | 檢索到的文件數量 | `8` |
| `synthesize.answer_length` | 生成答案的長度 | `1024` |
| `llm.model` | 使用的 LLM 模型 | `gpt-4` |
| `llm.total_tokens` | Token 總使用量 | `2500` |

### Jaeger 查詢範例

1. 查找耗時超過 5 秒的請求：
   ```
   duration > 5s
   ```

2. 查找特定節點的錯誤：
   ```
   error=true AND node.name="retrieve"
   ```

3. 查找使用特定模型的請求：
   ```
   llm.model="gpt-4" AND duration > 2s
   ```

## 度量指標收集 (Metrics Collection)

### 可用指標

#### API 層級指標

| 指標名稱 | 類型 | 標籤 | 說明 |
|---------|------|------|------|
| `rag_api_requests_total` | Counter | `endpoint`, `method`, `status` | API 請求總數 |
| `rag_api_request_duration_seconds` | Histogram | `endpoint`, `method` | API 請求耗時 |
| `rag_active_requests` | Gauge | - | 當前活躍請求數 |

#### LangGraph 節點指標

| 指標名稱 | 類型 | 標籤 | 說明 |
|---------|------|------|------|
| `langgraph_node_execution_seconds` | Histogram | `node_name` | 節點執行時間 |
| `langgraph_node_errors_total` | Counter | `node_name`, `error_type` | 節點錯誤總數 |

#### LLM 相關指標

| 指標名稱 | 類型 | 標籤 | 說明 |
|---------|------|------|------|
| `llm_tokens_total` | Counter | `model`, `token_type` | Token 使用總量 |
| `llm_request_duration_seconds` | Histogram | `model`, `operation` | LLM 請求耗時 |
| `llm_errors_total` | Counter | `model`, `error_type` | LLM 錯誤總數 |

#### 檢索器指標

| 指標名稱 | 類型 | 標籤 | 說明 |
|---------|------|------|------|
| `retriever_documents_retrieved` | Histogram | `retriever_type` | 檢索文件數量分佈 |
| `retriever_relevance_score` | Histogram | `retriever_type` | 相關性分數分佈 |
| `retriever_duration_seconds` | Histogram | `retriever_type` | 檢索操作耗時 |

#### 驗證指標

| 指標名稱 | 類型 | 標籤 | 說明 |
|---------|------|------|------|
| `rag_validation_results_total` | Counter | `result` | 驗證結果計數 |
| `rag_validation_warnings_total` | Counter | `warning_type` | 警告類型計數 |
| `rag_answer_quality_score` | Histogram | - | 答案品質分數分佈 |

### Prometheus 查詢範例

1. 計算過去 5 分鐘的平均請求耗時：
   ```promql
   rate(rag_api_request_duration_seconds_sum[5m]) / rate(rag_api_request_duration_seconds_count[5m])
   ```

2. 查看每個節點的 P95 耗時：
   ```promql
   histogram_quantile(0.95, rate(langgraph_node_execution_seconds_bucket[5m]))
   ```

3. 計算 Token 使用率：
   ```promql
   sum(rate(llm_tokens_total[5m])) by (model, token_type)
   ```

4. 監控錯誤率：
   ```promql
   sum(rate(langgraph_node_errors_total[5m])) by (node_name) / sum(rate(rag_api_requests_total[5m]))
   ```

## 配置指南

### 環境變數配置

```bash
# 日誌配置
LOG_LEVEL=INFO                    # 日誌級別：DEBUG, INFO, WARNING, ERROR
JSON_LOGS=true                    # 是否使用 JSON 格式（生產環境建議開啟）
LOG_FILE=/var/log/rag/app.log    # 日誌文件路徑（選填）

# 追蹤配置
JAEGER_ENDPOINT=localhost:6831    # Jaeger 收集器端點
OTLP_ENDPOINT=localhost:4317      # OTLP 收集器端點（選填）
TRACE_CONSOLE=false               # 是否在控制台輸出追蹤（開發用）

# 指標配置
METRICS_PORT=8000                 # Prometheus 指標暴露端口
```

### Docker Compose 配置範例

```yaml
version: '3.8'

services:
  rag-api:
    image: langgraph-rag:latest
    environment:
      - LOG_LEVEL=INFO
      - JSON_LOGS=true
      - JAEGER_ENDPOINT=jaeger:6831
      - METRICS_PORT=8000
    ports:
      - "8080:8080"    # API 端口
      - "8000:8000"    # Metrics 端口

  jaeger:
    image: jaegertracing/all-in-one:latest
    ports:
      - "16686:16686"  # Jaeger UI
      - "6831:6831/udp" # Jaeger Agent

  prometheus:
    image: prom/prometheus:latest
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
    ports:
      - "9090:9090"

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
```

### Prometheus 配置

```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'langgraph-rag'
    static_configs:
      - targets: ['rag-api:8000']
```

## 監控儀表板

### Grafana 儀表板設計

建議創建以下面板：

1. **總覽面板**
   - 請求率（QPS）
   - 平均響應時間
   - 錯誤率
   - 活躍請求數

2. **節點效能面板**
   - 各節點執行時間分佈
   - 節點錯誤率
   - 節點吞吐量

3. **LLM 使用面板**
   - Token 使用趨勢
   - 模型調用頻率
   - LLM 響應時間

4. **檢索效能面板**
   - 檢索文件數量分佈
   - 相關性分數分佈
   - 檢索耗時趨勢

### 告警規則範例

```yaml
groups:
  - name: rag_alerts
    rules:
      - alert: HighErrorRate
        expr: rate(langgraph_node_errors_total[5m]) > 0.1
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "節點錯誤率過高"
          description: "{{ $labels.node_name }} 節點的錯誤率超過 10%"
      
      - alert: HighLatency
        expr: histogram_quantile(0.95, rate(rag_api_request_duration_seconds_bucket[5m])) > 5
        for: 10m
        labels:
          severity: critical
        annotations:
          summary: "API 響應時間過長"
          description: "P95 響應時間超過 5 秒"
      
      - alert: HighTokenUsage
        expr: sum(rate(llm_tokens_total[1h])) > 100000
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Token 使用量過高"
          description: "過去一小時 Token 使用量超過 100,000"
```

## 最佳實踐

### 日誌最佳實踐

1. **使用結構化日誌**：避免使用 print()，統一使用 logger
2. **包含上下文**：總是包含 request_id 和 node_name
3. **適當的日誌級別**：
   - DEBUG：詳細的調試資訊
   - INFO：正常的業務流程
   - WARNING：潛在問題但不影響功能
   - ERROR：錯誤和異常
4. **避免敏感資訊**：不要記錄密碼、API 金鑰等

### 追蹤最佳實踐

1. **合理的 Span 粒度**：不要為每個函數都創建 Span
2. **有意義的屬性**：添加有助於調試的屬性
3. **錯誤處理**：總是記錄異常和錯誤狀態
4. **取樣策略**：生產環境可考慮取樣以降低開銷

### 指標最佳實踐

1. **使用適當的指標類型**：
   - Counter：累計值（請求數、錯誤數）
   - Gauge：瞬時值（活躍連接數）
   - Histogram：分佈值（響應時間、文件數量）
2. **合理的標籤**：避免高基數標籤
3. **預聚合**：使用 Histogram 而非記錄每個值
4. **定期清理**：清理不再使用的指標

### 故障排查流程

1. **從指標開始**：查看 Grafana 儀表板識別問題
2. **追蹤定位**：使用 Jaeger 查看具體請求的執行流程
3. **日誌詳情**：根據 request_id 查詢詳細日誌
4. **關聯分析**：結合三者找出根本原因

## 整合範例

### 使用結構化日誌

```python
from app.observability import get_logger

logger = get_logger(__name__)

# 記錄帶上下文的日誌
logger.info("Processing document", 
           doc_id="doc123",
           doc_size=1024,
           processing_step="embedding")

# 記錄錯誤
try:
    process_document()
except Exception as e:
    logger.error("Document processing failed",
                error=str(e),
                error_type=type(e).__name__,
                doc_id="doc123")
```

### 添加自定義追蹤

```python
from app.observability import tracer

def custom_processing(data):
    with tracer.start_as_current_span("custom_processing") as span:
        span.set_attribute("data.size", len(data))
        span.set_attribute("processing.type", "batch")
        
        try:
            result = heavy_computation(data)
            span.set_attribute("result.size", len(result))
            return result
        except Exception as e:
            span.record_exception(e)
            span.set_status(Status(StatusCode.ERROR))
            raise
```

### 添加自定義指標

```python
from prometheus_client import Counter, Histogram

# 定義自定義指標
custom_counter = Counter(
    'custom_processing_total',
    'Custom processing operations',
    ['operation_type']
)

custom_histogram = Histogram(
    'custom_processing_duration',
    'Custom processing duration',
    ['operation_type']
)

# 使用指標
def process_custom_data(data, operation_type):
    with custom_histogram.labels(operation_type=operation_type).time():
        result = perform_operation(data)
        custom_counter.labels(operation_type=operation_type).inc()
        return result
```

## 結論

完整的可觀測性解決方案讓我們能夠：

- 快速定位和解決問題
- 優化系統效能
- 預測和預防故障
- 提供更好的用戶體驗

透過結構化日誌、分散式追蹤和度量指標的結合，我們可以全面了解系統的運行狀況，並持續改進服務品質。