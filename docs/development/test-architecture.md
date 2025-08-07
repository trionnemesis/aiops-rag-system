# 測試架構

本文檔說明 RAG 服務的測試架構和策略，特別是在整合 LangGraph 後的測試方法。

## 概述

我們的測試架構專注於兩個主要層面：
1. **端到端流程測試**：測試完整的 LangGraph 流程
2. **節點單元測試**：測試各個節點的內部邏輯

## 測試結構

```
tests/
├── test_graph_flow.py          # 端到端的 LangGraph 流程測試
├── test_rag_chain_service.py   # 節點單元測試
├── test_vector_store_manager.py # 向量存儲管理器測試
├── test_model_manager.py       # 模型管理器測試
├── test_prompt_manager.py      # 提示管理器測試
└── ...                         # 其他組件測試
```

## 端到端流程測試 (test_graph_flow.py)

### 測試策略

端到端測試專注於驗證 LangGraph 的完整執行流程，包括：
- 不同路徑的觸發條件
- 節點之間的轉換
- 錯誤處理和 fallback 機制

### 主要測試案例

1. **快速路徑測試**
   ```python
   async def test_fast_path_flow():
       # 測試長查詢觸發快速路徑
       # 驗證：route="fast", 基本檢索, 答案生成
   ```

2. **深度路徑測試**
   ```python
   async def test_deep_path_with_hyde():
       # 測試短查詢或啟用 HyDE 時的深度路徑
       # 驗證：route="deep", HyDE 生成, 多重檢索
   ```

3. **多查詢擴展測試**
   ```python
   async def test_multi_query_expansion():
       # 測試查詢擴展功能
       # 驗證：生成多個查詢變體, 並行檢索
   ```

4. **BM25 + RRF 融合測試**
   ```python
   async def test_bm25_rrf_fusion():
       # 測試混合檢索策略
       # 驗證：向量檢索 + BM25, RRF 融合結果
   ```

5. **錯誤處理測試**
   ```python
   async def test_error_handling_in_retrieve():
       # 測試各節點的錯誤處理
       # 驗證：錯誤被捕獲, fallback 機制啟動
   ```

### Mock 策略

端到端測試 Mock 外部依賴，但不 Mock 圖的內部邏輯：
- Mock LLM API 調用
- Mock 向量資料庫查詢
- Mock 外部服務（如 LangExtract）

## 節點單元測試 (test_rag_chain_service.py)

### 測試策略

節點單元測試專注於各個節點函式的內部邏輯，確保每個節點獨立運作正常。

### 主要測試類別

1. **TestPlanNode**
   - 路由決策邏輯
   - HyDE 查詢生成
   - 多查詢擴展
   - 錯誤處理和 fallback

2. **TestRetrieveNode**
   - 基本檢索功能
   - 多查詢檢索
   - BM25 混合檢索
   - 空結果處理
   - 錯誤處理

3. **TestSynthesizeNode**
   - 有文件時的答案生成
   - 無文件時的 fallback
   - 上下文截斷
   - 錯誤處理

4. **TestValidateNode**
   - 答案驗證邏輯
   - 長度檢查
   - 錯誤狀態檢查

5. **TestExtractNode**
   - 結構化資訊提取
   - 批次處理
   - 重試機制
   - 錯誤處理

6. **TestErrorHandlerNode**
   - 錯誤訊息處理
   - Fallback 答案生成

## 測試最佳實踐

### 1. 使用適當的 Mock

```python
# 好的做法：Mock 外部依賴
mock_llm = Mock()
mock_llm.invoke = Mock(return_value=AIMessage(content="回答"))

# 避免：Mock 整個圖或節點
# graph_app = Mock()  # 不要這樣做
```

### 2. 測試真實流程

```python
# 好的做法：測試實際的圖執行
result = await graph_app.ainvoke(input_state)
assert result["route"] == "fast"

# 避免：只測試 Mock 的返回值
# mock_graph.invoke.return_value = {"answer": "test"}
```

### 3. 覆蓋邊界情況

- 空輸入
- 超長輸入
- 特殊字符
- 並發請求
- 服務失敗

### 4. 驗證狀態轉換

```python
# 驗證狀態在節點間的正確傳遞
assert "error" not in result  # 無錯誤狀態
assert len(result["queries"]) > 0  # 查詢已生成
assert result["context"] != ""  # 上下文已建立
```

## 測試環境設置

### 必要的 Fixtures

```python
@pytest.fixture
def mock_llm():
    """模擬 LLM"""
    llm = Mock()
    llm.invoke = Mock(return_value=AIMessage(content="測試回答"))
    return llm

@pytest.fixture
def mock_retriever():
    """模擬檢索器"""
    retriever = Mock()
    retriever.invoke = Mock(return_value=[
        Document(page_content="文件內容", metadata={"id": "1"})
    ])
    return retriever
```

### 測試配置

```python
# 測試用的策略配置
test_policy = {
    "use_hyde": False,
    "use_multi_query": False,
    "use_bm25": False,
    "top_k": 5,
    "min_answer_length": 10
}
```

## 執行測試

### 執行所有測試
```bash
pytest tests/
```

### 執行特定測試
```bash
# 只執行端到端測試
pytest tests/test_graph_flow.py

# 只執行節點單元測試
pytest tests/test_rag_chain_service.py

# 執行特定測試函式
pytest tests/test_graph_flow.py::TestGraphFlow::test_fast_path_flow
```

### 測試覆蓋率
```bash
pytest --cov=app --cov-report=html tests/
```

## 持續整合

在 CI/CD 管道中，測試按以下順序執行：
1. 單元測試（快速）
2. 整合測試（中等）
3. 端到端測試（較慢）

這確保快速反饋同時維持完整的測試覆蓋。

## 未來改進

1. **性能測試**：添加負載測試和基準測試
2. **屬性測試**：使用 Hypothesis 進行更全面的輸入測試
3. **視覺化測試**：生成圖執行的視覺化報告
4. **A/B 測試框架**：比較不同策略的效果