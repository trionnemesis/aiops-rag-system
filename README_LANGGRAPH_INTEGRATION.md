# LangGraph RAG 整合指南

這個實作提供了一個最小可行的 LangGraph DAG，可以逐步替換你現有的 LCEL 鏈，同時保持系統穩定性。

## 架構特點

1. **小改動、可回退**：保留既有的 LLM/Embedder/Retriever 物件，只在上層新增 Graph 控制流程
2. **可插拔設計**：HyDE、多查詢、BM25/RRF 都是可選的，預設只跑「向量檢索 → 生成」
3. **內建穩定性**：錯誤處理和超時保護，失敗時降階到精簡回覆
4. **觀測能力**：回傳基礎指標（token 用量、檢索文檔數等），方便接入 Prometheus

## 快速開始

### 1. 安裝依賴

```bash
pip install langgraph langchain-core langchain-community opensearch-py
```

### 2. 整合你現有的元件

在 `app/api/routes.py` 中，替換 TODO 部分：

```python
# 注入你現有的 LLM
from langchain_google_genai import ChatGoogleGenerativeAI
llm = ChatGoogleGenerativeAI(
    model="gemini-pro",
    google_api_key="your-api-key",
    temperature=0.1
)

# 注入你現有的 retriever
from your_existing_code import get_opensearch_retriever
retriever = get_opensearch_retriever()

# 可選：提供 BM25 搜尋函式
def bm25_search_fn(query: str, top_k: int = 8):
    # 實作 OpenSearch BM25 搜尋
    return []  # List[Document]
```

### 3. 調整 Policy 設定

```python
policy = {
    "use_hyde": True,           # 啟用 HyDE（假設性文檔擴展）
    "use_multi_query": True,    # 啟用多查詢生成
    "multi_query_alts": 2,      # 生成 2 個替代查詢
    "use_rrf": False,          # 啟用 RRF 融合（需要 BM25）
    "top_k": 8,                # 檢索前 K 筆文檔
    "max_ctx_chars": 6000,     # 上下文最大字元數
    "strict_citation": True,   # 嚴格引用檢查
    "fallback_text": "...",   # 失敗時的降階文字
    "min_docs": 2,             # 最少文檔數警告閾值
    "min_answer_len": 40,      # 最短答案長度警告閾值
}
```

## 核心元件說明

### 1. State（狀態定義）

`app/graph/state.py` 定義了整個 DAG 的狀態流轉：

- `query`: 使用者查詢
- `route`: 路由決策（fast/deep）
- `queries`: 擴展後的查詢列表
- `docs`: 檢索到的文檔
- `context`: 組裝的上下文
- `answer`: 最終答案
- `metrics`: 指標數據

### 2. Nodes（節點實作）

`app/graph/nodes.py` 包含四個主要節點：

#### Plan Node（規劃節點）
- 決定走 fast 還是 deep 路徑
- 根據 policy 執行 HyDE 和多查詢生成
- 簡單啟發式：短查詢或含模糊詞彙時走 deep 路徑

#### Retrieve Node（檢索節點）
- 執行向量檢索（必須）
- 可選執行 BM25 文字檢索
- 可選使用 RRF 融合多個檢索結果
- 自動去重和排序

#### Synthesize Node（生成節點）
- 組裝上下文（可自訂函式）
- 呼叫 LLM 生成答案
- 檢查引用標註
- 錯誤時降階到簡短回覆

#### Validate Node（驗證節點）
- 檢查文檔數量是否足夠
- 檢查答案長度
- 產生警告標記

### 3. Graph Builder（圖建構器）

`app/graph/build.py` 負責組裝 LangGraph：

- 定義節點執行順序：Plan → Retrieve → Synthesize → Validate
- 提供預設的 RRF 融合函式
- 提供預設的上下文組裝函式
- 使用 MemorySaver 支援檢查點功能

## 進階設定

### 自訂 BM25 搜尋

```python
def bm25_search_fn(query: str, top_k: int = 8) -> List[Document]:
    body = {
        "query": {
            "match": {
                "content": {"query": query}
            }
        },
        "size": top_k
    }
    response = opensearch_client.search(index="your_index", body=body)
    # 轉換為 Document 物件
    return docs
```

### 自訂上下文組裝

```python
def custom_build_context(docs: List[Document], max_chars: int = 6000) -> str:
    # 實作你的組裝邏輯
    # 例如：加入特殊格式、過濾規則等
    return formatted_context
```

### 自訂 RRF 融合

```python
def advanced_rrf_fuse(runs: List[List[Document]], k: int = 8) -> List[Document]:
    # 實作更複雜的融合邏輯
    # 例如：考慮分數、來源權重等
    return fused_docs
```

## 監控整合

### Prometheus 指標

在 API handler 外層加入計時和計數：

```python
from prometheus_client import Counter, Histogram

rag_requests = Counter('rag_requests_total', 'Total RAG requests')
rag_latency = Histogram('rag_latency_seconds', 'RAG request latency')

@router.post("/rag/report")
def rag_report(req: RAGRequest):
    rag_requests.inc()
    with rag_latency.time():
        # 原本的處理邏輯
        result = graph_app.invoke(...)
```

### 回傳的 Metrics

每次請求都會回傳：

```json
{
  "metrics": {
    "queries": 3,          // 使用的查詢數
    "docs": 8,            // 檢索到的文檔數
    "rrf_on": false,      // 是否啟用 RRF
    "latency_ms": 1234,   // 處理延遲
    "warnings": ["low_docs"]  // 警告標記
  }
}
```

## 遷移策略

### 第一階段：並行測試
1. 保留原有的 LCEL 鏈
2. 新增 `/rag/report-v2` 路由使用 LangGraph
3. A/B 測試比較品質和性能

### 第二階段：逐步切換
1. 監控 metrics，確認穩定性
2. 逐步增加 LangGraph 的流量比例
3. 根據回饋調整 policy 參數

### 第三階段：完全遷移
1. 將原路由切換到 LangGraph
2. 保留原 LCEL 程式碼作為備份
3. 持續優化 Graph 節點

## 故障排除

### 常見問題

1. **檢索結果為空**
   - 檢查 retriever 設定
   - 確認索引名稱和欄位對應
   - 檢視 OpenSearch 連線狀態

2. **HyDE/多查詢沒有效果**
   - 確認 policy 中相關選項已開啟
   - 檢查 LLM 的 prompt 回應
   - 可能需要調整啟發式規則

3. **性能問題**
   - 減少 `top_k` 值
   - 關閉不必要的功能（如 RRF）
   - 使用更快的 LLM 模型

### Debug 模式

在開發時可以加入更詳細的日誌：

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# 在節點中加入日誌
def plan_node(state, ...):
    logging.debug(f"Planning for query: {state['query']}")
    # ... 節點邏輯
    logging.debug(f"Generated {len(state['queries'])} queries")
    return state
```

## 下一步優化

1. **加入快取層**：對常見查詢快取結果
2. **非同步處理**：使用 async 版本的 LangGraph
3. **更智慧的路由**：使用 LLM 判斷查詢類型
4. **動態 Policy**：根據查詢特徵調整參數
5. **更多檢索策略**：加入語義相似度重排序、時間衰減等