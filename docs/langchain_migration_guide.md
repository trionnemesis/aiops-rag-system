# LangChain 遷移指南

本指南說明如何從原有實作遷移到 LangChain 版本。

## 快速遷移

### 1. 無需改動的部分

如果您的程式碼使用 `RAGService`，**不需要任何改動**：

```python
from src.services.rag_service import RAGService

# 原有程式碼完全相容
rag_service = RAGService()
report = await rag_service.generate_report(monitoring_data)
```

### 2. 使用新功能

如果想使用 LangChain 的新功能：

```python
from src.services.langchain import RAGChainService

# 使用新的 LangChain 服務
rag_service = RAGChainService()

# 獲得更多功能
custom_chain = rag_service.create_custom_chain(
    retriever_kwargs={"search_kwargs": {"k": 10}},
    hyde_enabled=True
)
```

## 主要改進

### 1. 模型管理

**原本方式**：
```python
# 需要手動管理不同模型實例
self.flash_model = genai.GenerativeModel(settings.gemini_flash_model)
self.pro_model = genai.GenerativeModel(settings.gemini_pro_model)
```

**LangChain 方式**：
```python
from src.services.langchain import model_manager

# 統一的模型管理
model = model_manager.get_model("flash")  # 或 "pro"
```

### 2. 提示詞管理

**原本方式**：
```python
# 使用 Python 類別變數
prompt = PromptTemplates.HYDE_GENERATION.format(monitoring_data="...")
```

**LangChain 方式**：
```python
from src.services.langchain import prompt_manager

# 使用 LangChain PromptTemplate
prompt = prompt_manager.get_prompt("hyde_generation")
formatted = prompt.format(monitoring_data="...")
```

### 3. 向量資料庫

**原本方式**：
```python
# 直接使用 OpenSearch API
self.client.search(index=self.index_name, body=query)
```

**LangChain 方式**：
```python
from src.services.langchain import vector_store_manager

# 使用抽象化的介面
retriever = vector_store_manager.as_retriever()
docs = await retriever.ainvoke("查詢")
```

## 進階用法

### 1. 創建自定義 RAG 鏈

```python
from langchain_core.runnables import RunnablePassthrough
from src.services.langchain import model_manager, prompt_manager

# 構建自定義鏈
custom_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt_manager.get_prompt("rag_query")
    | model_manager.pro_model
    | StrOutputParser()
)

# 使用自定義鏈
result = await custom_chain.ainvoke("你的問題")
```

### 2. 添加新的提示詞

```python
# 動態添加提示詞
prompt_manager.add_custom_prompt(
    name="emergency_analysis",
    template="""
    緊急事件分析：
    監控數據：{monitoring_data}
    請提供立即行動建議。
    """
)

# 使用新提示詞
prompt = prompt_manager.get_prompt("emergency_analysis")
```

### 3. 切換向量資料庫

如果要從 OpenSearch 切換到其他向量資料庫，只需修改 `vector_store_manager.py`：

```python
# 例如切換到 Chroma
from langchain_community.vectorstores import Chroma

self._vector_store = Chroma(
    collection_name="aiops-knowledge",
    embedding_function=model_manager.embedding_model
)
```

## 性能對比

| 指標 | 原實作 | LangChain 實作 |
|------|--------|----------------|
| 程式碼行數 | 125 行 | 45 行（主要邏輯） |
| 可測試性 | 中等 | 高（組件解耦） |
| 擴展性 | 需要修改核心程式碼 | 輕鬆添加新功能 |
| 維護成本 | 高 | 低 |

## 常見問題

### Q: 為什麼要遷移到 LangChain？

A: LangChain 提供了：
- 更清晰的程式碼結構
- 標準化的組件介面
- 豐富的生態系統
- 更容易的測試和除錯

### Q: 遷移會影響現有功能嗎？

A: 不會。我們保持了完全的向後相容性，所有現有 API 都能正常運作。

### Q: 如何逐步遷移？

A: 建議步驟：
1. 先使用現有的 `RAGService`（已經在內部使用 LangChain）
2. 逐步學習 LangChain 組件
3. 在新功能中使用 LangChain 特性
4. 根據需要重構舊程式碼

## 下一步

- 閱讀 [LangChain 重構報告](./langchain_refactoring_report.md) 了解詳細實作
- 查看 `examples/langchain_rag_example.py` 學習使用方式
- 參考 [LangChain 官方文檔](https://python.langchain.com/docs/get_started/introduction) 深入學習