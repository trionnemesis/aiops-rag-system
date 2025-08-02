# 向量索引與檢索效能優化指南

## 概述

本文檔描述了向量檢索系統效能優化的第一階段實施方案，包括建立效能基線、整合壓力測試和擴充監控指標。

## 第一階段：建立效能基線與監控

### 1. 建立測試集與評估指標

#### 1.1 標準查詢集合

我們定義了四類標準查詢來涵蓋不同的使用場景：

- **短查詢 (Short Queries)**：1-2個詞的簡單查詢
- **中等查詢 (Medium Queries)**：5-10個詞的一般查詢
- **長查詢 (Long Queries)**：15個詞以上的複雜查詢
- **模糊查詢 (Fuzzy Queries)**：包含拼寫錯誤或混合語言的查詢

#### 1.2 評估指標

**效能指標**：
- 平均延遲 (Average Latency)
- P50、P95、P99 延遲百分位數
- 最小/最大延遲
- 標準差

**品質指標**：
- 召回率 (Recall)：檢索到的相關文檔比例
- 準確率 (Precision)：檢索結果中相關文檔的比例
- F1 分數：召回率和準確率的調和平均

#### 1.3 使用方式

```python
# 執行效能基線測試
pytest tests/test_vector_performance.py::test_performance_baseline -v

# 測試 ef_search 參數影響
pytest tests/test_vector_performance.py::test_ef_search_impact -v

# 測試召回率和準確率
pytest tests/test_vector_performance.py::test_recall_precision -v
```

測試結果將保存為 CSV 文件：
- `vector_performance_baseline.csv`：基線測試結果
- `ef_search_impact.csv`：ef_search 參數影響
- `recall_precision_results.csv`：召回率和準確率結果

### 2. 整合壓力測試

#### 2.1 Locust 壓力測試配置

使用 Locust 進行分散式負載測試，支援多種測試場景：

**測試用戶行為**：
- 純向量搜尋 (權重 3)
- 混合搜尋 (權重 2)
- 帶過濾條件搜尋 (權重 1)

**負載測試類型**：
- **穩定負載測試**：固定用戶數持續測試
- **遞增負載測試**：階梯式增加用戶數
- **尖峰負載測試**：模擬突發流量

#### 2.2 執行壓力測試

```bash
# 使用 pytest 執行負載測試
pytest tests/test_vector_load.py -m load -v

# 或直接使用 Locust
locust -f tests/test_vector_load.py --host http://localhost:8000 \
       --users 50 --spawn-rate 5 --run-time 5m
```

**測試參數說明**：
- `--users`：模擬用戶數
- `--spawn-rate`：每秒新增用戶數
- `--run-time`：測試持續時間

#### 2.3 效能基準要求

- 平均響應時間 < 200ms
- P95 響應時間 < 500ms
- 失敗率 < 1%

### 3. Prometheus 監控指標

#### 3.1 新增的向量檢索指標

| 指標名稱 | 類型 | 說明 | 標籤 |
|---------|------|------|------|
| `vector_search_total` | Counter | 向量搜尋總次數 | strategy, index |
| `vector_search_duration_seconds` | Histogram | 搜尋延遲分布 | strategy, index |
| `vector_search_results_count` | Histogram | 返回結果數分布 | strategy, index |
| `opensearch_cluster_health` | Gauge | 叢集健康狀態 (0=red, 1=yellow, 2=green) | cluster |
| `opensearch_index_document_count` | Gauge | 索引文檔數量 | index |
| `opensearch_index_size_bytes` | Gauge | 索引大小（字節） | index |
| `opensearch_ef_search_value` | Gauge | 當前 ef_search 參數值 | index |
| `vector_search_recall` | Summary | 召回率 | query_type |
| `vector_search_precision` | Summary | 準確率 | query_type |

#### 3.2 查看監控指標

```bash
# 訪問 Prometheus 指標端點
curl http://localhost:8000/metrics

# 使用 Prometheus 查詢
# 查詢向量搜尋 P95 延遲
histogram_quantile(0.95, 
  sum(rate(vector_search_duration_seconds_bucket[5m])) by (le)
)

# 查詢每秒查詢率
sum(rate(vector_search_total[1m]))
```

#### 3.3 整合 Grafana Dashboard

建議創建以下 Dashboard 面板：

1. **效能概覽**
   - QPS (每秒查詢數)
   - 平均延遲趨勢
   - P95/P99 延遲趨勢

2. **搜尋策略分析**
   - 各策略使用比例
   - 各策略延遲對比
   - 返回結果數分布

3. **系統資源監控**
   - OpenSearch CPU/Memory 使用率
   - JVM Heap 使用率
   - 索引大小和文檔數

4. **品質指標**
   - 召回率趨勢
   - 準確率趨勢
   - F1 分數

## API 端點

### 向量搜尋端點

```http
POST /api/v1/search/vector
Content-Type: application/json

{
  "query": "深度學習模型訓練",
  "k": 10,
  "strategy": "hybrid",
  "filter": {
    "tags": ["machine-learning"]
  }
}
```

**參數說明**：
- `query`：搜尋查詢文字
- `k`：返回結果數量（預設 10）
- `strategy`：搜尋策略
  - `knn_only`：純向量搜尋
  - `hybrid`：混合搜尋（向量 + 文字）
  - `multi_vector`：多向量搜尋
  - `rerank`：重新排序
- `filter`：過濾條件（可選）

### Prometheus 指標端點

```http
GET /metrics
```

返回所有已註冊的 Prometheus 指標，格式符合 Prometheus 文本格式。

## 效能優化建議

### 1. ef_search 參數調優

根據測試結果，ef_search 參數對效能有顯著影響：

- **ef_search = 50**：最快但召回率較低
- **ef_search = 100**：平衡的預設值
- **ef_search = 200**：較好的召回率，延遲增加約 20%
- **ef_search = 500**：高召回率，延遲增加約 50%

建議根據業務需求在召回率和延遲之間權衡。

### 2. 索引優化參數

```json
{
  "settings": {
    "index": {
      "knn": true,
      "knn.algo_param.ef_search": 100,
      "knn.algo_param.ef_construction": 128,
      "knn.algo_param.m": 24
    }
  }
}
```

- `ef_construction`：構建時的搜尋寬度，越大索引品質越好但構建越慢
- `m`：每個節點的連接數，影響索引大小和搜尋效能

### 3. 查詢優化策略

1. **預過濾**：使用 filter 減少向量搜尋範圍
2. **分層搜尋**：先用較小的 k 值快速篩選，再精確排序
3. **快取策略**：對熱門查詢結果進行快取
4. **批次處理**：合併多個查詢減少開銷

## 下一步計劃

### 第二階段：查詢優化

1. 實現智慧查詢路由
2. 優化向量維度
3. 實現結果快取機制

### 第三階段：系統優化

1. 分散式部署
2. 索引分片優化
3. 硬體加速（GPU）

## 相關文件

- [OpenSearch k-NN 文檔](https://opensearch.org/docs/latest/search-plugins/knn/index/)
- [Prometheus 最佳實踐](https://prometheus.io/docs/practices/)
- [Locust 文檔](https://docs.locust.io/)