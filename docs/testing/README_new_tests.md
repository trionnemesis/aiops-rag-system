# 新增的單元測試說明

本次更新新增了以下關鍵模組的單元測試，大幅提升了系統的測試覆蓋率和穩健性。

## 1. KNN 搜尋服務測試 (`test_knn_search_service.py`)

### 測試覆蓋範圍
- **各種搜尋策略的邏輯測試**
  - `KNN_ONLY`: 純向量搜尋
  - `HYBRID`: 混合搜尋（向量 + BM25）
  - `MULTI_VECTOR`: 多向量搜尋（查詢擴展）
  - `RERANK`: 重新排序搜尋

- **參數處理測試**
  - 過濾條件 (filter)
  - 最低分數閾值 (min_score)
  - 搜尋參數 (k, num_candidates, boost)

- **錯誤處理測試**
  - 無效策略處理
  - OpenSearch 連線錯誤
  - 異常情況恢復

- **輔助功能測試**
  - 餘弦相似度計算
  - 關鍵詞匹配分數
  - 結果去重和重新排序
  - LangChain Document 格式轉換
  - 搜尋結果解釋 (explain)

### 測試數量
- 總計 19 個測試案例
- 涵蓋正常流程、邊界條件和錯誤情況

## 2. 結構化日誌測試 (`test_observability_logging.py`)

### 測試覆蓋範圍
- **日誌序列化**
  - 基本日誌記錄序列化
  - 包含請求上下文的序列化
  - 包含異常資訊的序列化

- **日誌配置**
  - JSON 格式配置
  - 人類可讀格式配置
  - 文件輸出配置（含日誌輪替）

- **請求上下文管理**
  - 上下文設定和清除
  - 裝飾器模式的上下文注入
  - 嵌套上下文處理
  - 異常情況下的上下文恢復

- **進階功能**
  - ContextVar 的協程隔離性
  - 日誌系統整合測試

### 測試數量
- 總計 13 個測試案例
- 驗證結構化日誌的完整生命週期

## 3. Prometheus 指標測試 (`test_observability_metrics.py`)

### 測試覆蓋範圍
- **指標類型測試**
  - Counter（計數器）
  - Histogram（直方圖）
  - Gauge（量表）
  - Info（資訊）

- **裝飾器功能**
  - `@track_request_metrics`: API 請求追蹤
  - `@track_node_metrics`: LangGraph 節點追蹤
  - `@track_llm_metrics`: LLM 調用追蹤
  - `@track_retrieval_metrics`: 檢索操作追蹤

- **特殊節點指標**
  - retrieve 節點：文檔數量、相關性分數
  - validate 節點：驗證結果、警告、品質分數
  - LLM 調用：Token 使用量

- **進階測試**
  - 指標導出格式驗證
  - 標籤基數控制
  - 並發更新安全性
  - 百分位數計算

### 測試數量
- 總計 20 個測試案例
- 涵蓋所有指標類型和裝飾器

## 4. 分散式追蹤測試 (`test_observability_tracing.py`)

### 測試覆蓋範圍
- **追蹤配置**
  - 基本配置（控制台輸出）
  - Jaeger 整合配置
  - OTLP 整合配置
  - 自動儀表化（FastAPI、HTTPX）

- **節點追蹤裝飾器**
  - 成功執行的 Span 生成
  - 錯誤處理和異常記錄
  - 特殊節點屬性：
    - retrieve: 文檔數量
    - synthesize: 答案長度
    - validate: 驗證結果

- **LLM 追蹤**
  - Token 使用量記錄
  - 無使用量資訊處理
  - API 錯誤追蹤

- **檢索追蹤**
  - 結果數量和分數分佈
  - 空結果處理
  - 連線錯誤追蹤

- **資料處理**
  - 長查詢截斷（隱私保護）

### 測試數量
- 總計 18 個測試案例
- 完整覆蓋追蹤生命週期

## 執行測試

### 方式一：使用提供的腳本
```bash
python3 run_new_tests.py
```

### 方式二：使用 pytest（需要安裝）
```bash
# 安裝測試依賴
pip install pytest pytest-asyncio pytest-mock

# 執行所有新測試
pytest tests/test_knn_search_service.py tests/test_observability_*.py -v

# 執行單一測試檔案
pytest tests/test_knn_search_service.py -v
```

### 方式三：在 CI/CD 環境執行
新測試已整合到現有的 CI/CD 流程中，會在 GitHub Actions 中自動執行。

## 測試的重要性

1. **提升系統穩健性**
   - 確保核心搜尋邏輯的正確性
   - 驗證可觀測性模組的可靠性

2. **促進維護性**
   - 程式碼變更時能快速發現問題
   - 提供使用範例和文檔功能

3. **支援重構**
   - 有信心進行系統優化
   - 確保向後相容性

4. **品質保證**
   - 達到企業級的測試標準
   - 提升整體測試覆蓋率

## 下一步建議

1. **整合測試**
   - 新增端到端的整合測試
   - 測試各模組間的互動

2. **效能測試**
   - 新增負載測試案例
   - 監控記憶體使用

3. **Mock 服務**
   - 建立 OpenSearch 的 Mock 服務
   - 模擬各種網路狀況

4. **覆蓋率報告**
   - 整合 coverage.py
   - 設定覆蓋率目標（建議 80%+）