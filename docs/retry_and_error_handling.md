# 重試機制與錯誤處理實作說明

## 概述

本文檔說明了在 RAG 圖形中實作的節點級重試機制和精細錯誤處理分支。

## 1. 節點級重試機制 (Node-level Retries)

### 1.1 實作方式

使用 `tenacity` 函式庫實現指數退避（Exponential Backoff）的重試邏輯：

```python
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

@retry(
    stop=stop_after_attempt(3),  # 最多重試 3 次
    wait=wait_exponential(multiplier=1, min=2, max=10),  # 指數退避：2, 4, 8 秒
    retry=retry_if_exception_type((ConnectionError, TimeoutError)),  # 針對特定異常重試
    before_sleep=lambda retry_state: logger.warning(...)  # 重試前記錄日誌
)
def _api_call_with_retry():
    # 外部 API 調用
    pass
```

### 1.2 應用範圍

重試機制已應用於以下節點的外部服務調用：

1. **extract_node**: LangExtract 服務的批量提取操作
2. **plan_node**: LLM 調用（HyDE 和多查詢生成）
3. **retrieve_node**: 向量資料庫檢索和 BM25 搜尋
4. **synthesize_node**: LLM 生成答案

### 1.3 重試策略

- **重試次數**: 最多 3 次
- **等待時間**: 指數退避，從 2 秒開始，最多 10 秒
- **重試條件**: ConnectionError, TimeoutError 和其他 Exception
- **日誌記錄**: 每次重試前記錄警告日誌

## 2. 精細錯誤處理分支 (Granular Error Handling)

### 2.1 錯誤處理流程

```
START → extract → plan → retrieve → synthesize → validate → END
         ↓         ↓        ↓           ↓
         └─────────┴────────┴───────────┴──→ error_handler → END
```

### 2.2 錯誤檢測機制

每個關鍵節點執行後，使用條件邊（conditional edges）檢查是否有錯誤：

```python
def check_error(state):
    """檢查狀態中是否有錯誤"""
    if state.get("error"):
        return "error_handler"
    return "continue"

graph.add_conditional_edges(
    "node_name",
    check_error,
    {
        "continue": "next_node",
        "error_handler": "error_handler"
    }
)
```

### 2.3 錯誤處理節點

`error_handler_node` 負責：

1. **記錄詳細錯誤日誌**
2. **生成用戶友好的錯誤訊息**
3. **記錄錯誤指標**

錯誤類型對應的回應訊息：

- `extract_error`: "系統無法處理您提供的文本資料..."
- `plan_error`: "系統正在處理您的查詢，但遇到暫時性問題..."
- `retrieve_error`: "系統無法存取知識庫，可能是網路連線問題..."
- `synthesize_error`: "系統正在生成回答時遇到問題..."

## 3. 使用範例

### 3.1 正常情況（重試後成功）

```python
# 第一次調用失敗，重試後成功
result = app.invoke({"query": "系統異常分析"})
# 輸出：正常的分析結果
```

### 3.2 錯誤處理情況（重試失敗）

```python
# 所有重試都失敗，觸發錯誤處理
result = app.invoke({"query": "系統狀態查詢"})
# 輸出：友好的錯誤訊息 + 錯誤詳情
```

## 4. 配置選項

在 `policy` 字典中可配置：

```python
policy = {
    "retry_attempts": 3,  # 重試次數（預設：3）
    "retry_min_wait": 2,  # 最小等待時間（預設：2秒）
    "retry_max_wait": 10,  # 最大等待時間（預設：10秒）
    "fallback_text": "自定義錯誤回應文字",
}
```

## 5. 監控與日誌

### 5.1 日誌輸出

- 每次重試時記錄警告日誌
- 最終失敗時記錄錯誤日誌
- 錯誤處理節點記錄詳細錯誤信息

### 5.2 指標收集

在 `state["metrics"]` 中記錄：

- `error_handled`: 是否觸發錯誤處理
- `error_type`: 錯誤類型
- `retry_count`: 實際重試次數

## 6. 最佳實踐

1. **選擇性重試**: 只對暫時性錯誤（網路、超時）進行重試
2. **快速失敗**: 對於明確的錯誤（如參數錯誤）不進行重試
3. **降級策略**: 部分功能失敗時提供降級服務
4. **錯誤聚合**: 收集錯誤模式以改進系統穩定性

## 7. 測試

執行測試腳本驗證功能：

```bash
python test_retry_error_handling.py
```

測試涵蓋：
- 重試機制的正常運作
- 錯誤處理分支的觸發
- 用戶友好的錯誤訊息生成