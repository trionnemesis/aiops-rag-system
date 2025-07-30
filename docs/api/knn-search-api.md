# KNN 搜尋 API

## 📋 概述

本文檔詳細說明 KNN 向量搜尋的 API 使用方式，包括端點定義、參數說明、使用範例和整合指南。

## 🚀 快速開始

### 基本搜尋請求

```bash
curl -X POST http://localhost:8000/api/v1/knn/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Apache 記憶體使用率過高",
    "k": 5,
    "strategy": "hybrid"
  }'
```

### Python 客戶端

```python
import requests

response = requests.post(
    "http://localhost:8000/api/v1/knn/search",
    json={
        "query": "MySQL 慢查詢優化",
        "k": 10,
        "strategy": "hybrid"
    }
)

results = response.json()
```

## 📡 API 端點

### POST /api/v1/knn/search

執行 KNN 向量搜尋

#### 請求參數

| 參數 | 類型 | 必填 | 預設值 | 說明 |
|------|------|------|---------|------|
| query | string | 是 | - | 搜尋查詢文字 |
| k | integer | 否 | 10 | 返回結果數量 |
| strategy | string | 否 | "hybrid" | 搜尋策略 |
| num_candidates | integer | 否 | k*10 | HNSW 候選數量 |
| min_score | float | 否 | null | 最低分數門檻 |
| filter | object | 否 | null | 過濾條件 |

#### 搜尋策略

- `knn_only`: 純向量搜尋
- `hybrid`: 混合搜尋（向量 + BM25）
- `multi_vector`: 多向量搜尋
- `rerank`: 重新排序搜尋

#### 回應格式

```json
{
  "status": "success",
  "query": "Apache 記憶體使用率過高",
  "strategy": "hybrid",
  "total_results": 5,
  "execution_time_ms": 85,
  "results": [
    {
      "doc_id": "E-2024-03-15",
      "title": "前台網站服務無回應事件",
      "content": "...",
      "score": 0.92,
      "tags": ["apache", "memory", "performance"],
      "category": "incident_report",
      "highlights": [
        "Apache 的 <em>記憶體使用率</em>持續在 90% 以上"
      ]
    }
  ]
}
```

### POST /api/v1/knn/explain

解釋搜尋結果評分

```bash
curl -X POST http://localhost:8000/api/v1/knn/explain \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Apache 效能問題",
    "doc_id": "E-2024-03-15"
  }'
```

#### 回應範例

```json
{
  "doc_id": "E-2024-03-15",
  "title": "前台網站服務無回應事件",
  "score": 0.92,
  "explanation": {
    "value": 0.92,
    "description": "knn similarity",
    "details": [
      {
        "value": 0.92,
        "description": "vector similarity score"
      }
    ]
  }
}
```

## 🔍 進階搜尋功能

### 過濾搜尋

#### 單一條件過濾

```json
{
  "query": "效能優化",
  "filter": {
    "term": {
      "tags": "mysql"
    }
  }
}
```

#### 複合條件過濾

```json
{
  "query": "記憶體問題",
  "filter": {
    "bool": {
      "must": [
        {"term": {"category": "incident_report"}},
        {"term": {"tags": "performance"}}
      ],
      "should": [
        {"term": {"tags": "apache"}},
        {"term": {"tags": "mysql"}}
      ],
      "minimum_should_match": 1
    }
  }
}
```

#### 範圍過濾

```json
{
  "query": "最近的效能問題",
  "filter": {
    "range": {
      "created_at": {
        "gte": "2024-01-01",
        "lte": "2024-12-31"
      }
    }
  }
}
```

### 自訂搜尋參數

```json
{
  "query": "資料庫索引優化",
  "k": 20,
  "num_candidates": 200,
  "min_score": 0.7,
  "strategy": "rerank"
}
```

## 🔗 LangChain 整合

### 使用 KNN Retriever

```python
from app.api.knn_langchain_bridge import KNNRetriever

# 建立 retriever
retriever = KNNRetriever(
    index_name="aiops_knowledge_base",
    embedding_model="gemini-embedding-001",
    search_strategy=SearchStrategy.HYBRID,
    k=10
)

# 在 LangChain 中使用
from langchain.chains import RetrievalQA

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)

result = qa_chain({"query": "如何優化 Apache 效能？"})
```

### 整合到 LangGraph

```python
from app.api.knn_langchain_bridge import create_knn_langraph_components

# 建立所有元件
components = create_knn_langraph_components(
    index_name="aiops_knowledge_base",
    embedding_model="gemini-embedding-001",
    vector_search_k=10,
    bm25_search_k=8
)

# 在 LangGraph 中使用
from app.graph.graph import build_graph

graph = build_graph(
    llm=llm,
    retriever=components["hybrid_retriever"],
    bm25_search_fn=components["bm25_search_fn"],
    build_context_fn=components["build_context_fn"],
    policy=policy
)
```

## 💻 程式化使用

### Python SDK

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

# 執行搜尋
results = await search_service.knn_search(
    query_text="MySQL 效能優化",
    params=KNNSearchParams(
        k=10,
        num_candidates=100,
        min_score=0.5
    ),
    strategy=SearchStrategy.HYBRID
)

# 處理結果
for result in results:
    print(f"標題: {result.title}")
    print(f"分數: {result.score}")
    print(f"標籤: {', '.join(result.metadata['tags'])}")
```

### 批次搜尋

```python
async def batch_search(queries: List[str]):
    """批次執行搜尋"""
    tasks = []
    
    for query in queries:
        task = search_service.knn_search(
            query_text=query,
            strategy=SearchStrategy.KNN_ONLY
        )
        tasks.append(task)
    
    results = await asyncio.gather(*tasks)
    return results
```

### 自訂搜尋策略

```python
# 根據查詢特徵選擇策略
def select_strategy(query: str) -> SearchStrategy:
    query_length = len(query.split())
    
    if query_length < 3:
        # 短查詢使用混合搜尋
        return SearchStrategy.HYBRID
    elif "如何" in query or "怎麼" in query:
        # 問題型查詢使用多向量
        return SearchStrategy.MULTI_VECTOR
    elif query_length > 10:
        # 長查詢使用重新排序
        return SearchStrategy.RERANK
    else:
        # 預設使用純向量搜尋
        return SearchStrategy.KNN_ONLY

# 使用自動策略選擇
strategy = select_strategy(query)
results = await search_service.knn_search(
    query_text=query,
    strategy=strategy
)
```

## 📊 效能優化建議

### 快取策略

```python
from functools import lru_cache
import hashlib

@lru_cache(maxsize=1000)
async def cached_search(query_hash: str, strategy: str):
    """快取搜尋結果"""
    # 實際搜尋邏輯
    pass

# 使用快取
query_hash = hashlib.md5(query.encode()).hexdigest()
results = await cached_search(query_hash, strategy.value)
```

### 並行請求

```python
async def parallel_multi_strategy_search(query: str):
    """並行執行多種策略搜尋"""
    strategies = [
        SearchStrategy.KNN_ONLY,
        SearchStrategy.HYBRID,
        SearchStrategy.MULTI_VECTOR
    ]
    
    tasks = [
        search_service.knn_search(
            query_text=query,
            strategy=strategy,
            params=KNNSearchParams(k=5)
        )
        for strategy in strategies
    ]
    
    results = await asyncio.gather(*tasks)
    
    # 合併結果
    all_results = []
    for strategy_results in results:
        all_results.extend(strategy_results)
    
    # 去重並排序
    unique_results = {r.doc_id: r for r in all_results}
    sorted_results = sorted(
        unique_results.values(),
        key=lambda x: x.score,
        reverse=True
    )
    
    return sorted_results[:10]
```

## 🔧 錯誤處理

### 常見錯誤碼

| 錯誤碼 | 說明 | 解決方案 |
|---------|------|----------|
| 400 | 無效的請求參數 | 檢查參數格式 |
| 404 | 索引不存在 | 確認索引名稱 |
| 422 | 查詢文字過長 | 縮短查詢文字 |
| 500 | 伺服器內部錯誤 | 檢查日誌 |
| 503 | 服務暫時不可用 | 稍後重試 |

### 錯誤處理範例

```python
try:
    results = await search_service.knn_search(
        query_text=query,
        strategy=SearchStrategy.HYBRID
    )
except ValueError as e:
    # 參數錯誤
    print(f"參數錯誤: {e}")
except ConnectionError as e:
    # 連線錯誤
    print(f"無法連接到 OpenSearch: {e}")
except Exception as e:
    # 其他錯誤
    print(f"搜尋失敗: {e}")
```

## 📈 監控指標

### Prometheus 指標

```python
# 搜尋延遲
search_latency = Histogram(
    'knn_search_latency_seconds',
    'KNN search latency',
    ['strategy']
)

# 搜尋數量
search_counter = Counter(
    'knn_search_total',
    'Total KNN searches',
    ['strategy', 'status']
)

# 使用範例
with search_latency.labels(strategy='hybrid').time():
    results = await search_service.knn_search(...)
    
search_counter.labels(
    strategy='hybrid',
    status='success'
).inc()
```

### 日誌記錄

```python
import structlog

logger = structlog.get_logger()

# 記錄搜尋請求
logger.info(
    "knn_search_request",
    query=query,
    strategy=strategy,
    k=k,
    filters=filters
)

# 記錄搜尋結果
logger.info(
    "knn_search_response",
    query=query,
    result_count=len(results),
    execution_time_ms=execution_time,
    top_score=results[0].score if results else 0
)
```

## 🔗 相關資源

- [KNN 向量搜尋架構](../architecture/knn-vector-search.md)
- [KNN 索引建置指南](../development/knn-index-guide.md)
- [API 端點參考](./endpoints.md)