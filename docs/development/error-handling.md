# 錯誤處理最佳實踐

本文檔說明 AIOps RAG 系統的錯誤處理架構和最佳實踐。

## 錯誤處理架構

### 1. 分層錯誤處理

系統採用分層的錯誤處理架構：

```
API 層 (FastAPI)
    ↓
服務層 (RAGService)
    ↓
LangChain 層 (RAGChainService)
    ↓
外部服務 (Gemini, OpenSearch, Prometheus)
```

每一層都有其專屬的錯誤類型和處理邏輯。

### 2. 自定義例外階層

```python
RAGServiceError (基礎例外)
├── VectorDBError         # 向量資料庫相關錯誤
├── GeminiAPIError        # Gemini API 呼叫錯誤
├── PrometheusError       # Prometheus 監控錯誤
├── HyDEGenerationError   # HyDE 生成錯誤
├── DocumentRetrievalError # 文檔檢索錯誤
├── ReportGenerationError # 報告生成錯誤
└── CacheError           # 快取操作錯誤
```

### 3. 全域錯誤處理器

FastAPI 應用程式設定了全域錯誤處理器：

```python
@app.exception_handler(RAGServiceError)
async def rag_service_exception_handler(request: Request, exc: RAGServiceError):
    """統一處理 RAG 服務相關錯誤"""
    # 根據錯誤類型返回適當的 HTTP 狀態碼
```

## 錯誤處理模式

### 1. 服務層錯誤拋出

在服務層明確拋出業務邏輯錯誤：

```python
# rag_chain_service.py
async def generate_report(self, monitoring_data: Dict[str, Any]) -> InsightReport:
    try:
        result = await self.full_rag_chain.ainvoke({
            "monitoring_data": monitoring_data
        })
        return InsightReport(...)
    except Exception as e:
        if isinstance(e, (HyDEGenerationError, DocumentRetrievalError)):
            raise  # 重新拋出已知錯誤
        raise ReportGenerationError(f"Failed to generate report: {str(e)}")
```

### 2. API 層錯誤捕捉

在 API 層捕捉並轉換為 HTTP 回應：

```python
# main.py
@app.post("/api/v1/generate_report")
async def generate_report(request: ReportRequest):
    try:
        report = await rag_service.generate_report(enriched_data)
        return ReportResponse(...)
    except RAGServiceError:
        raise  # 由全域處理器處理
    except HTTPException:
        raise  # 保留原有 HTTP 例外
    except Exception as e:
        # 未預期錯誤的通用處理
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal error")
```

### 3. Fallback 機制

對於非關鍵服務，實作 fallback 機制：

```python
# Prometheus 數據豐富（非關鍵）
if "主機" in enriched_data:
    try:
        enriched_data = await rag_service.enrich_with_prometheus(
            hostname, enriched_data
        )
    except PrometheusError as e:
        # 記錄警告但不中斷流程
        logger.warning(f"Prometheus enrichment failed: {str(e)}")
```

## 錯誤回應格式

統一的錯誤回應格式：

```json
{
    "status": "error",
    "message": "詳細的錯誤訊息",
    "error_type": "VectorDBError",
    "details": {
        // 額外的錯誤詳情（可選）
    }
}
```

## HTTP 狀態碼映射

| 錯誤類型 | HTTP 狀態碼 | 說明 |
|---------|------------|------|
| `RequestValidationError` | 422 | 請求驗證失敗 |
| `VectorDBError` | 503 | 向量資料庫服務不可用 |
| `GeminiAPIError` | 503 | AI 模型服務不可用 |
| `PrometheusError` | 503 | 監控服務不可用 |
| `HyDEGenerationError` | 500 | 內部處理錯誤 |
| `DocumentRetrievalError` | 500 | 內部處理錯誤 |
| `ReportGenerationError` | 500 | 內部處理錯誤 |
| `CacheError` | 500 | 快取操作失敗 |

## 最佳實踐

### 1. 錯誤訊息原則

- **具體明確**：說明發生了什麼錯誤
- **可操作性**：提供解決建議（如果適用）
- **安全性**：不洩露敏感資訊

```python
# 好的錯誤訊息
raise VectorDBError("Failed to connect to OpenSearch: Connection timeout after 30s")

# 不好的錯誤訊息
raise VectorDBError("Error")
```

### 2. 日誌記錄

- 使用適當的日誌等級
- 包含足夠的上下文資訊
- 對於預期錯誤使用 warning，未預期錯誤使用 error

```python
# 預期的外部服務錯誤
logger.warning(f"Prometheus service unavailable: {str(e)}")

# 未預期的系統錯誤
logger.error(f"Unexpected error in report generation: {str(e)}", exc_info=True)
```

### 3. 錯誤恢復

實作適當的錯誤恢復機制：

```python
# HyDE fallback 範例
def _safe_retrieval(self, x: Dict[str, Any]) -> List[Any]:
    try:
        # 嘗試使用 HyDE
        hyde_query = self.hyde_chain.invoke(...)
        documents = self.retriever.invoke(hyde_query)
        if documents:
            return documents
    except Exception as e:
        logging.warning(f"HyDE failed, using fallback: {str(e)}")
    
    # Fallback：直接使用原始查詢
    try:
        return self.retriever.invoke(x["monitoring_data_str"])
    except Exception as e:
        raise DocumentRetrievalError(f"All retrieval methods failed: {str(e)}")
```

## 測試錯誤處理

### 1. 單元測試範例

```python
@pytest.mark.asyncio
async def test_vector_db_error_handling(monkeypatch):
    """測試向量資料庫錯誤處理"""
    async def mock_generate_report(*args, **kwargs):
        raise VectorDBError("Connection failed")
    
    monkeypatch.setattr(RAGService, "generate_report", mock_generate_report)
    
    response = await client.post("/api/v1/generate_report", json={...})
    
    assert response.status_code == 503
    assert "Vector database service unavailable" in response.json()["message"]
```

### 2. 整合測試考量

- 測試各種錯誤情境
- 驗證錯誤不會導致資料洩露
- 確認錯誤恢復機制正常運作
- 檢查日誌記錄是否正確

## 監控和警報

### 1. 錯誤指標

透過 Prometheus 監控錯誤率：

```python
# 可以加入的指標
error_counter = Counter('api_errors_total', 'Total API errors', ['error_type'])

# 在錯誤處理器中
error_counter.labels(error_type=exc.__class__.__name__).inc()
```

### 2. 警報規則

設定關鍵錯誤的警報：

- VectorDBError 發生率 > 10/分鐘
- GeminiAPIError 連續失敗 > 5 次
- 總體錯誤率 > 5%

## 故障排除指南

### 常見錯誤和解決方案

1. **VectorDBError**
   - 檢查 OpenSearch 服務狀態
   - 驗證連線設定
   - 確認索引是否存在

2. **GeminiAPIError**
   - 檢查 API 金鑰
   - 確認 API 配額
   - 檢查網路連線

3. **PrometheusError**
   - 確認 Prometheus 服務運行中
   - 檢查目標主機是否存在
   - 驗證查詢語法

## 未來改進方向

1. **熔斷器模式**：對外部服務實作熔斷器
2. **重試機制**：加入智慧重試邏輯
3. **錯誤分析**：自動分析錯誤模式
4. **自我修復**：某些錯誤的自動恢復機制