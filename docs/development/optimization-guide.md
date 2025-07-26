# RAG 系統優化指南

## 概述

本文檔說明針對 AIOps 智慧維運報告 RAG 系統的兩個主要優化措施：

1. **文件摘要整合 (Consolidated Document Summarization)**
2. **假設性事件快取機制 (Cache for Hypothetical Events)**

## 1. 文件摘要整合

### 問題描述

在原始實作中，系統會對從 OpenSearch 檢索回來的每一篇文件單獨呼叫 Gemini Flash 模型來產生摘要。如果 `top_k_results` 設定為 5，就會產生 5 次 API 呼叫，這會快速累積成本。

### 解決方案

將所有檢索到的文件內容合併成一個長文本，然後只呼叫一次 Gemini Flash 模型進行摘要。

### 實作細節

```python
# 原始方式（多次呼叫）
summaries = []
for doc in similar_docs:  # 假設有 5 個文件
    summary = await self.gemini.summarize_document(...)  # 5 次 API 呼叫
    summaries.append(summary)

# 優化後（單次呼叫）
all_docs_content = "\n\n--- 文件分隔 ---\n\n".join([
    f"文件 {i+1}:\n{doc['content']}" 
    for i, doc in enumerate(similar_docs)
])
consolidated_summary = await self.gemini.summarize_document(...)  # 1 次 API 呼叫
```

### 效益

- **成本降低**: API 呼叫次數從 k 次降低到 1 次（例如：從 5 次降到 1 次，節省 80% 的摘要成本）
- **效能提升**: 減少網路延遲，提高響應速度
- **上下文整合**: 模型可以同時看到所有文件，產生更整合的摘要

## 2. 假設性事件快取機制

### 問題描述

相似的監控數據可能會生成相同的假設性文件（HyDE）和嵌入向量，重複計算會浪費資源和成本。

### 解決方案

使用 `async-lru` 實作具有時間限制（TTL）的 LRU 快取，對 HyDE 生成和嵌入向量計算進行快取。

### 實作細節

```python
# HyDE 快取：最多 50 個項目，保留 30 分鐘
@alru_cache(maxsize=50, ttl=1800)
async def _get_cached_hyde(self, hyde_prompt: str) -> str:
    return await self.gemini.generate_hyde(hyde_prompt)

# 嵌入向量快取：最多 100 個項目，保留 1 小時
@alru_cache(maxsize=100, ttl=3600)
async def _get_cached_embedding(self, text: str) -> List[float]:
    return await self.gemini.generate_embedding(text)
```

### 快取鍵設計

快取鍵基於監控數據的關鍵指標：
- 主機名稱
- CPU 使用率
- RAM 使用率
- 磁碟使用率
- 服務名稱

### 效益

- **成本降低**: 重複請求不會產生額外的 API 呼叫
- **響應時間**: 快取命中時幾乎立即返回結果
- **資源優化**: 減少對 Gemini API 的負載

## API 端點

### 查看快取狀態

```bash
GET /api/v1/cache/info
```

響應範例：
```json
{
  "status": "success",
  "cache_info": {
    "hyde_cache": {
      "hits": 45,
      "misses": 10,
      "maxsize": 50,
      "currsize": 35,
      "hit_rate": "81.82%"
    },
    "embedding_cache": {
      "hits": 120,
      "misses": 30,
      "maxsize": 100,
      "currsize": 75,
      "hit_rate": "80.00%"
    }
  }
}
```

### 清除快取

```bash
POST /api/v1/cache/clear
```

響應範例：
```json
{
  "status": "success",
  "message": "All caches have been cleared"
}
```

## 使用建議

### 1. 監控快取效率

定期檢查快取命中率，理想情況下應該在 70% 以上：

```python
import httpx

async def monitor_cache_efficiency():
    async with httpx.AsyncClient() as client:
        response = await client.get("http://localhost:8000/api/v1/cache/info")
        cache_info = response.json()["cache_info"]
        
        for cache_name, info in cache_info.items():
            print(f"{cache_name}: {info['hit_rate']} hit rate")
```

### 2. 快取大小調整

根據實際使用情況調整快取大小：

```python
# 如果命中率低且 currsize 接近 maxsize，考慮增加 maxsize
@alru_cache(maxsize=200, ttl=3600)  # 增加到 200
```

### 3. TTL 調整

根據數據變化頻率調整 TTL：

- 穩定環境：可以設定較長的 TTL（例如 2-4 小時）
- 動態環境：使用較短的 TTL（例如 15-30 分鐘）

### 4. 清除快取時機

在以下情況下考慮清除快取：
- 知識庫有重大更新
- 系統維護後
- 測試新功能時

## 成本效益分析

假設每天處理 1000 個請求，其中 70% 是重複的：

### 文件摘要整合
- 原始：1000 請求 × 5 文件 = 5000 次摘要 API 呼叫
- 優化後：1000 請求 × 1 = 1000 次摘要 API 呼叫
- **節省：80% 的摘要成本**

### 快取機制（70% 命中率）
- HyDE 生成：節省 700 次 API 呼叫
- 嵌入向量：節省 700 次 API 呼叫
- **節省：70% 的 HyDE 和嵌入成本**

### 總體效益
結合兩個優化，在典型使用場景下可以節省超過 75% 的 API 成本。

## 注意事項

1. **快取一致性**: 當知識庫更新時，考慮清除相關快取
2. **記憶體使用**: 監控服務器記憶體使用情況，避免快取過大
3. **快取穿透**: 對於罕見的查詢，仍會產生 API 呼叫
4. **分散式環境**: 目前的快取是進程內的，在多實例部署時需要考慮共享快取方案（如 Redis）

## 未來優化方向

1. **分散式快取**: 使用 Redis 實現跨實例的快取共享
2. **智能快取預熱**: 基於歷史數據預先載入常用查詢的快取
3. **動態 TTL**: 根據數據變化頻率自動調整 TTL
4. **批次處理**: 累積多個請求後批次處理，進一步降低成本