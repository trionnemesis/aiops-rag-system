# KNN 向量索引建置指南

## 📋 概述

本指南詳細說明如何建立和管理 OpenSearch KNN 向量索引，包括索引設計、批次載入、參數調整和維護策略。

## 🚀 快速開始

### 1. 建立基本索引

```bash
# 使用預設參數建立索引
python scripts/build_knn_index.py --load-sample

# 指定索引名稱和維度
python scripts/build_knn_index.py \
    --index-name my_knowledge_base \
    --dimension 768
```

### 2. 自訂 HNSW 參數

```bash
# 高精度配置
python scripts/build_knn_index.py \
    --hnsw-m 48 \
    --hnsw-ef-construction 512 \
    --hnsw-ef-search 200

# 平衡配置
python scripts/build_knn_index.py \
    --hnsw-m 16 \
    --hnsw-ef-construction 128 \
    --hnsw-ef-search 100
```

## 📐 索引設計

### 映射結構

```json
{
  "mappings": {
    "properties": {
      "doc_id": {
        "type": "keyword"
      },
      "title": {
        "type": "text",
        "fields": {
          "keyword": {
            "type": "keyword"
          }
        }
      },
      "content": {
        "type": "text",
        "analyzer": "standard"
      },
      "embedding": {
        "type": "knn_vector",
        "dimension": 768,
        "method": {
          "name": "hnsw",
          "space_type": "l2",
          "engine": "nmslib",
          "parameters": {
            "ef_construction": 128,
            "m": 16
          }
        }
      },
      "tags": {
        "type": "keyword"
      },
      "category": {
        "type": "keyword"
      },
      "metadata": {
        "type": "object"
      },
      "created_at": {
        "type": "date"
      }
    }
  }
}
```

### 欄位說明

| 欄位 | 類型 | 用途 | 索引策略 |
|------|------|------|----------|
| doc_id | keyword | 唯一識別碼 | 精確匹配 |
| title | text/keyword | 文件標題 | 全文搜尋 + 精確匹配 |
| content | text | 文件內容 | 全文搜尋 |
| embedding | knn_vector | 向量表示 | HNSW 索引 |
| tags | keyword | 標籤分類 | 精確匹配、聚合 |
| category | keyword | 文件類別 | 過濾、聚合 |
| metadata | object | 擴展資訊 | 動態映射 |

## 🔧 程式化索引建立

### 基本範例

```python
from scripts.build_knn_index import KNNIndexBuilder

# 建立索引建置器
builder = KNNIndexBuilder(
    index_name="my_knowledge_base",
    embedding_dim=768,
    model_name="models/embedding-001"
)

# 建立索引
result = builder.create_knn_index(
    hnsw_m=16,
    hnsw_ef_construction=128,
    hnsw_ef_search=100,
    space_type="l2"
)
```

### 批次索引文件

```python
# 準備文件
documents = [
    {
        "doc_id": "DOC001",
        "title": "Apache 效能優化指南",
        "content": "詳細的 Apache 配置優化內容...",
        "tags": ["apache", "performance"],
        "category": "tutorial"
    },
    # 更多文件...
]

# 批次索引
result = await builder.bulk_index_documents(
    documents=documents,
    batch_size=100
)

print(f"成功索引: {result['success']} 個文件")
print(f"失敗: {result['failed']} 個文件")
```

### 單一文件索引

```python
# 索引單一文件
await builder.index_document(
    doc_id="DOC002",
    title="MySQL 索引優化",
    content="MySQL 索引建立和優化的最佳實踐...",
    tags=["mysql", "index", "optimization"],
    category="best_practice",
    metadata={
        "author": "DevOps Team",
        "version": "1.0",
        "last_updated": "2024-01-28"
    }
)
```

## 🎯 Embedding 模型配置

### 支援的模型

```python
from src.config.embedding_config import EMBEDDING_MODELS

# 查看支援的模型
for model_key, config in EMBEDDING_MODELS.items():
    print(f"{model_key}:")
    print(f"  維度: {config.dimension}")
    print(f"  類型: {config.model_type}")
```

### 切換模型

```python
# 使用 OpenAI 模型
builder_openai = KNNIndexBuilder(
    index_name="kb_openai",
    embedding_dim=1536,
    model_name="text-embedding-ada-002"
)

# 使用 HuggingFace 模型
builder_hf = KNNIndexBuilder(
    index_name="kb_hf",
    embedding_dim=384,
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
```

### 維度一致性檢查

```python
from src.config.embedding_config import validate_embedding_dimension

# 生成 embedding
embedding = await embeddings.aembed_query("測試文字")

# 驗證維度
try:
    validate_embedding_dimension(embedding, "gemini-embedding-001")
    print("維度檢查通過")
except ValueError as e:
    print(f"維度錯誤: {e}")
```

## 📊 索引管理

### 查看索引統計

```python
# 取得索引統計
stats = builder.get_index_stats()

print(f"索引名稱: {stats['index_name']}")
print(f"文件數量: {stats['document_count']}")
print(f"索引大小: {stats['size_in_bytes'] / 1024 / 1024:.2f} MB")
print(f"向量維度: {stats['embedding_dimension']}")
print(f"HNSW ef_search: {stats['ef_search']}")
```

### 索引健康檢查

```bash
# 檢查索引健康狀態
curl -X GET "localhost:9200/_cluster/health/my_knowledge_base?pretty"

# 查看索引設定
curl -X GET "localhost:9200/my_knowledge_base/_settings?pretty"

# 查看映射
curl -X GET "localhost:9200/my_knowledge_base/_mapping?pretty"
```

### 更新索引設定

```python
# 更新搜尋參數
client.indices.put_settings(
    index=index_name,
    body={
        "index": {
            "knn.algo_param.ef_search": 150
        }
    }
)
```

## 🔄 資料更新策略

### 增量更新

```python
async def incremental_update(new_documents):
    """增量更新索引"""
    # 檢查文件是否已存在
    existing_ids = set()
    for doc in new_documents:
        try:
            client.get(index=index_name, id=doc['doc_id'])
            existing_ids.add(doc['doc_id'])
        except:
            pass
    
    # 分離新文件和更新文件
    new_docs = [d for d in new_documents if d['doc_id'] not in existing_ids]
    update_docs = [d for d in new_documents if d['doc_id'] in existing_ids]
    
    # 索引新文件
    if new_docs:
        await builder.bulk_index_documents(new_docs)
    
    # 更新現有文件
    for doc in update_docs:
        await builder.index_document(**doc)
```

### 重建索引

```python
async def rebuild_index(old_index, new_index):
    """重建索引（零停機）"""
    # 1. 建立新索引
    new_builder = KNNIndexBuilder(index_name=new_index)
    new_builder.create_knn_index()
    
    # 2. 複製資料
    helpers.reindex(
        client=client,
        source_index=old_index,
        target_index=new_index
    )
    
    # 3. 建立別名切換
    client.indices.update_aliases(
        body={
            "actions": [
                {"remove": {"index": old_index, "alias": "kb_alias"}},
                {"add": {"index": new_index, "alias": "kb_alias"}}
            ]
        }
    )
    
    # 4. 刪除舊索引（可選）
    # client.indices.delete(index=old_index)
```

## ⚡ 效能優化

### 批次大小優化

```python
# 根據文件大小調整批次
def calculate_batch_size(avg_doc_size_kb):
    """計算最佳批次大小"""
    if avg_doc_size_kb < 10:
        return 200
    elif avg_doc_size_kb < 50:
        return 100
    else:
        return 50
```

### 並行索引

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

async def parallel_index(documents, num_workers=4):
    """並行索引文件"""
    # 分割文件
    chunk_size = len(documents) // num_workers
    chunks = [
        documents[i:i + chunk_size] 
        for i in range(0, len(documents), chunk_size)
    ]
    
    # 並行處理
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        tasks = []
        for chunk in chunks:
            task = asyncio.create_task(
                builder.bulk_index_documents(chunk)
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
    
    return results
```

### 記憶體優化

```python
# 串流處理大檔案
async def stream_index_from_file(file_path, chunk_size=1000):
    """從檔案串流索引"""
    documents = []
    
    with open(file_path, 'r') as f:
        for line in f:
            doc = json.loads(line)
            documents.append(doc)
            
            if len(documents) >= chunk_size:
                await builder.bulk_index_documents(documents)
                documents = []
        
        # 處理剩餘文件
        if documents:
            await builder.bulk_index_documents(documents)
```

## 🔍 測試索引品質

### 搜尋測試

```bash
# 執行搜尋測試
python scripts/test_knn_search.py --test accuracy

# 測試特定策略
python scripts/test_knn_search.py --test strategies
```

### 向量相似度驗證

```python
# 驗證向量品質
async def verify_vector_quality(sample_queries):
    """驗證向量搜尋品質"""
    for query in sample_queries:
        # KNN 搜尋
        knn_results = await search_service.knn_search(
            query_text=query,
            strategy=SearchStrategy.KNN_ONLY
        )
        
        # BM25 搜尋
        bm25_results = await bm25_search_fn(query)
        
        # 比較結果重疊度
        knn_ids = {r.doc_id for r in knn_results[:5]}
        bm25_ids = {r.metadata['doc_id'] for r in bm25_results[:5]}
        
        overlap = len(knn_ids & bm25_ids) / 5
        print(f"查詢: {query}")
        print(f"結果重疊度: {overlap * 100:.1f}%")
```

## 🛠️ 維護建議

### 定期任務

1. **每日**
   - 監控索引大小和文件數量
   - 檢查查詢延遲

2. **每週**
   - 分析慢查詢
   - 優化 HNSW 參數

3. **每月**
   - 評估向量品質
   - 考慮重建索引

### 監控指標

```python
# 建立監控指標
metrics = {
    "index_size_mb": stats['size_in_bytes'] / 1024 / 1024,
    "doc_count": stats['document_count'],
    "avg_query_time_ms": 85,
    "cache_hit_rate": 0.75,
    "vector_dimension": stats['embedding_dimension']
}

# 發送到 Prometheus
for metric, value in metrics.items():
    prometheus_gauge.labels(index=index_name).set(value)
```

## 🔗 相關資源

- [KNN 向量搜尋架構](../architecture/knn-vector-search.md)
- [KNN 搜尋 API](../api/knn-search-api.md)
- [OpenSearch KNN 外掛文檔](https://opensearch.org/docs/latest/search-plugins/knn/)