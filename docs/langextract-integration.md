# LangExtract 整合指南

## 概述

LangExtract 是一個專為 AIOps 場景設計的結構化資訊提取服務，能夠從非結構化的日誌和告警文本中自動提取關鍵實體資訊。本文檔說明如何將 LangExtract 整合到 LangGraph RAG 系統中。

## 架構設計

### 1. 整合位置

根據設計規則，LangExtract 被放置在資料預處理和知識庫建立的環節：

```
原始日誌/告警 → LangExtract (提取) → 分塊 → 向量化 → 存儲
                      ↓
                結構化元數據
```

### 2. LangGraph 流程

在 LangGraph 中，LangExtract 作為一個獨立的節點運行：

```
START → Extract Node → Plan Node → Retrieve Node → Synthesize Node → Validate Node → END
           ↓
    提取結構化資訊
```

## 核心組件

### 1. LangExtractService

負責從文本中提取 AIOps 相關的結構化資訊。

```python
from src.services.langchain.langextract_service import LangExtractService

# 初始化服務
extract_service = LangExtractService(llm=llm)

# 提取資訊
result = extract_service.extract(log_text)
```

#### 支援的實體類型

- **基本資訊**: 時間戳、日誌級別
- **系統資訊**: 主機名、服務名、組件、環境
- **錯誤資訊**: 錯誤碼、錯誤訊息、堆疊追蹤
- **效能指標**: CPU、記憶體、硬碟使用率、回應時間
- **網路相關**: IP 地址、埠號、HTTP 狀態碼、請求方法、端點

### 2. Extract Node

LangGraph 中的提取節點，負責批量處理原始文本。

```python
def extract_node(state, extract_service=None, policy=None):
    """提取結構化資訊節點"""
    raw_texts = state.get("raw_texts", [])
    extracted_data = extract_service.batch_extract(raw_texts)
    state["extracted_data"] = extracted_data
    return state
```

### 3. ChunkingService

支援結構化元數據的文檔分塊服務。

```python
from src.services.langchain.chunking_service import ChunkingService

chunking_service = ChunkingService(
    chunk_size=1000,
    chunk_overlap=200
)

# 分塊並附加元數據
documents = chunking_service.chunk_with_metadata(
    text=log_text,
    base_metadata={"source": "prometheus"},
    extracted_metadata=extracted_metadata
)
```

## 使用方式

### 1. 基本提取

```python
# 初始化
llm = ChatGoogleGenerativeAI(model="gemini-pro")
extract_service = LangExtractService(llm=llm)

# 提取
log_text = "2024-01-15 ERROR web-prod-03 CPU 95%"
result = extract_service.extract(log_text)

print(f"主機: {result.entities.hostname}")
print(f"CPU: {result.entities.cpu_usage}%")
print(f"信心分數: {result.confidence}")
```

### 2. 整合到 LangGraph

```python
from app.graph.build import build_graph

# 構建包含 LangExtract 的圖
app = build_graph(
    llm=llm,
    retriever=retriever,
    extract_service=extract_service,
    policy={
        "use_llm_extract": True,
        "use_metadata_filter": True
    }
)

# 執行
result = app.invoke({
    "query": "分析 CPU 異常",
    "raw_texts": ["ERROR: web-01 CPU 95%"]
})
```

### 3. 完整的資料攝入流程

```python
from src.services.langchain.chunking_service import ChunkingAndEmbeddingPipeline

# 創建管道
pipeline = ChunkingAndEmbeddingPipeline(
    chunking_service=chunking_service,
    embedding_service=embedding_service,
    extract_service=extract_service
)

# 處理日誌
vector_metadata_pairs = pipeline.process(
    texts=logs,
    base_metadata_list=metadata_list,
    use_extraction=True
)

# 存儲到向量資料庫
vector_store.add_embeddings(vector_metadata_pairs)
```

## 元數據過濾

### 1. 自動過濾

提取的結構化資訊會自動用於檢索過濾：

```python
# Retrieve Node 中的過濾邏輯
if extracted_data:
    metadata_filters = {
        "extracted_hostname": hostname,
        "extracted_service_name": service_name,
        "extracted_error_code": error_code
    }
    retriever.search_kwargs["filter"] = metadata_filters
```

### 2. 過濾效果

- **無過濾**: 檢索所有相關文檔
- **有過濾**: 優先檢索匹配提取實體的文檔

## 配置選項

### Policy 設定

```python
policy = {
    # LangExtract 相關
    "use_llm_extract": True,      # 是否使用 LLM 進行深度提取
    "use_metadata_filter": True,   # 是否啟用元數據過濾
    
    # 其他設定
    "use_hyde": True,             # HyDE 策略
    "use_rrf": False,             # RRF 融合
    "top_k": 8,                   # 檢索數量
}
```

## 效能優化

### 1. 混合提取策略

- **正則表達式**: 快速提取常見模式
- **LLM 提取**: 處理複雜或非標準格式

### 2. 批量處理

```python
# 批量提取多個日誌
results = extract_service.batch_extract(logs, use_llm=True)
```

### 3. 信心分數

根據信心分數決定是否使用提取結果：

```python
if result.confidence > 0.7:
    # 使用提取結果進行過濾
    apply_metadata_filter(result.entities)
```

## 實際案例

### 1. Prometheus 告警處理

```python
alert = """
FIRING: HighCPUUsage
Instance: web-prod-03:9100
Service: nginx
CPU Usage: 92.5%
"""

# 自動提取並生成報告
result = app.invoke({
    "query": "分析這個 CPU 告警",
    "raw_texts": [alert]
})
```

### 2. Elasticsearch 日誌分析

```python
log = """
{"@timestamp":"2024-01-15T10:30:00Z",
 "level":"ERROR",
 "host":"db-master-01",
 "error_code":"DB_TIMEOUT_001"}
"""

# 提取錯誤碼並查詢相關解決方案
result = app.invoke({
    "query": "如何解決這個資料庫超時錯誤",
    "raw_texts": [log]
})
```

## 最佳實踐

1. **預處理**: 在資料攝入時執行 LangExtract，而非查詢時
2. **元數據管理**: 使用統一的元數據命名規範（如 `extracted_` 前綴）
3. **信心閾值**: 根據場景調整信心分數閾值
4. **錯誤處理**: 當 LLM 提取失敗時，回退到正則表達式
5. **效能監控**: 追蹤提取時間和成功率

## 故障排除

### 常見問題

1. **提取結果為空**
   - 檢查日誌格式是否匹配
   - 確認 LLM API 可用
   - 查看正則表達式模式

2. **信心分數過低**
   - 增加關鍵欄位的提取
   - 優化正則表達式模式
   - 調整 LLM prompt

3. **元數據過濾無效**
   - 確認向量資料庫支援元數據過濾
   - 檢查元數據欄位名稱
   - 驗證過濾條件格式

## 總結

LangExtract 整合為 AIOps RAG 系統帶來以下優勢：

1. **精準檢索**: 基於結構化資訊的元數據過濾
2. **豐富上下文**: 提取的實體資訊增強生成品質
3. **自動化處理**: 減少人工標註和整理的工作
4. **可擴展性**: 易於添加新的實體類型和提取規則

透過將 LangExtract 整合到 LangGraph 流程中，系統能夠更智慧地理解和處理 AIOps 場景中的非結構化資料。