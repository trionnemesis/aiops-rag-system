# LangChain LCEL 重構報告

## 概述

本報告詳細說明了如何使用 LangChain 的表達式語言 (LCEL) 對 AIOps 智慧維運報告 RAG 系統進行重構，以提升程式碼的可讀性、可維護性和擴充性。

## 重構目標

1. **使用 LCEL 重構 RAG 流程** - 將原本手動串連的步驟改為聲明式的管道流程
2. **統一模型與提示詞管理** - 使用 LangChain 的標準化介面
3. **抽象化向量資料庫** - 提高系統的可擴充性
4. **保持向後相容** - 維持原有 API 介面不變

## 架構改進

### 1. 原有架構問題

原本的 RAG 流程在 `src/services/rag_service.py` 中實作：

```python
# 原本的實作方式
async def generate_report(self, monitoring_data):
    # Step 1: HyDE Generation
    hyde_prompt = self.prompts.HYDE_GENERATION.format(...)
    hypothetical_doc = await self.gemini.generate_hyde(hyde_prompt)
    
    # Step 2: Generate Embedding and Search
    query_embedding = await self.gemini.generate_embedding(hypothetical_doc)
    similar_docs = await self.opensearch.search_similar_documents(query_embedding)
    
    # Step 3: Summarize Documents
    # ... 手動處理每個步驟
    
    # Step 4: Generate Final Report
    # ... 更多手動處理
```

**問題：**
- 步驟之間緊密耦合
- 難以重用或修改單一步驟
- 缺乏聲明式的流程定義

### 2. LangChain LCEL 解決方案

使用 LCEL 後的實作：

```python
# 使用 LCEL 的新實作
self.full_rag_chain = (
    # 準備輸入
    RunnableParallel(
        monitoring_data=RunnablePassthrough(),
        monitoring_data_str=RunnableLambda(prepare_monitoring_data)
    )
    # HyDE 生成
    | RunnableParallel(
        monitoring_data=lambda x: x["monitoring_data"],
        hyde_query=lambda x: self.hyde_chain.invoke(...)
    )
    # 檢索相關文檔
    | RunnableParallel(
        documents=lambda x: self.retriever.invoke(x["hyde_query"])
    )
    # 生成最終報告
    | lambda x: self.report_chain.invoke(...)
)
```

**優勢：**
- 清晰的管道式流程定義
- 每個步驟都可以獨立測試和重用
- 易於修改和擴展

## 實作細節

### 1. 模型管理器 (`model_manager.py`)

統一管理所有 Gemini 模型實例：

```python
class ModelManager:
    @property
    def flash_model(self) -> BaseChatModel:
        """獲取 Gemini Flash 模型（快速任務）"""
        
    @property
    def pro_model(self) -> BaseChatModel:
        """獲取 Gemini Pro 模型（複雜任務）"""
        
    @property
    def embedding_model(self) -> Embeddings:
        """獲取嵌入模型"""
```

**特點：**
- 使用單例模式避免重複初始化
- 提供統一的模型介面
- 支援動態參數更新

### 2. 提示詞管理器 (`prompt_manager.py`)

使用 LangChain 的 `ChatPromptTemplate` 管理所有提示詞：

```python
class PromptManager:
    def _initialize_prompts(self):
        self._prompts["hyde_generation"] = ChatPromptTemplate.from_template(...)
        self._prompts["summary_refinement"] = ChatPromptTemplate.from_template(...)
        self._prompts["final_report"] = ChatPromptTemplate.from_template(...)
```

**特點：**
- 集中管理所有提示詞模板
- 支援動態添加和更新提示詞
- 自動處理變數傳入

### 3. 向量資料庫管理器 (`vector_store_manager.py`)

抽象化 OpenSearch 操作：

```python
class VectorStoreManager:
    @property
    def vector_store(self) -> VectorStore:
        """獲取向量資料庫實例"""
        return OpenSearchVectorSearch(
            opensearch_url=...,
            index_name=...,
            embedding_function=model_manager.embedding_model
        )
    
    def as_retriever(self, **kwargs):
        """獲取檢索器"""
        return self.vector_store.as_retriever(...)
```

**特點：**
- 使用 LangChain 的 `VectorStore` 介面
- 輕鬆切換不同的向量資料庫
- 內建檢索器功能

### 4. RAG 鏈服務 (`rag_chain_service.py`)

核心的 LCEL 實作：

```python
class RAGChainService:
    def _initialize_chains(self):
        # HyDE 生成鏈
        self.hyde_chain = (
            prompt_manager.get_prompt("hyde_generation")
            | model_manager.flash_model
            | StrOutputParser()
        )
        
        # 文檔摘要鏈
        self.summary_chain = (
            prompt_manager.get_prompt("summary_refinement")
            | model_manager.flash_model
            | StrOutputParser()
        )
        
        # 最終報告生成鏈
        self.report_chain = (
            prompt_manager.get_prompt("final_report")
            | model_manager.pro_model
            | StrOutputParser()
            | RunnableLambda(self._parse_report_output)
        )
```

## 使用方式

### 1. 基本使用（保持原有介面）

```python
from src.services.rag_service import RAGService

# 初始化服務
rag_service = RAGService()

# 生成報告
report = await rag_service.generate_report(monitoring_data)
```

### 2. 進階使用（直接使用 LangChain 組件）

```python
from src.services.langchain import model_manager, prompt_manager, vector_store_manager

# 使用模型
model = model_manager.pro_model
response = await model.ainvoke("你的提示詞")

# 使用提示詞模板
prompt = prompt_manager.get_prompt("hyde_generation")
formatted_prompt = prompt.format(monitoring_data="...")

# 使用向量資料庫
retriever = vector_store_manager.as_retriever(search_kwargs={"k": 5})
docs = await retriever.ainvoke("查詢文本")
```

### 3. 創建自定義 RAG 鏈

```python
# 創建自定義鏈
custom_chain = rag_chain_service.create_custom_chain(
    retriever_kwargs={"search_kwargs": {"k": 3}},
    hyde_enabled=True
)

# 使用自定義鏈
result = await custom_chain.ainvoke("你的查詢")
```

## 性能優化

### 1. 快取機制

保留原有的快取功能，整合到新架構中：

```python
@alru_cache(maxsize=100, ttl=3600)
async def _get_cached_embedding(self, text: str) -> List[float]:
    """帶快取的嵌入向量生成"""
    
@alru_cache(maxsize=50, ttl=1800)
async def _get_cached_hyde(self, monitoring_data_str: str) -> str:
    """帶快取的 HyDE 生成"""
```

### 2. 並行處理

使用 `RunnableParallel` 實現步驟的並行執行：

```python
RunnableParallel(
    monitoring_data=lambda x: x["monitoring_data"],
    monitoring_data_str=lambda x: x["monitoring_data_str"],
    hyde_query=lambda x: self.hyde_chain.invoke(...)
)
```

## 擴展性

### 1. 更換向量資料庫

只需修改 `vector_store_manager.py`：

```python
# 例如切換到 Pinecone
from langchain_community.vectorstores import Pinecone

self._vector_store = Pinecone(
    index_name="...",
    embedding_function=model_manager.embedding_model
)
```

### 2. 添加新的模型

在 `model_manager.py` 中添加：

```python
@property
def claude_model(self) -> BaseChatModel:
    """添加 Claude 模型"""
    return ChatAnthropic(model="claude-3-opus")
```

### 3. 自定義提示詞

使用 `prompt_manager` 動態添加：

```python
prompt_manager.add_custom_prompt(
    name="custom_analysis",
    template="你的自定義模板：{variable1} {variable2}"
)
```

## 測試和驗證

### 1. 單元測試

每個組件都可以獨立測試：

```python
# 測試模型管理器
def test_model_manager():
    model = model_manager.flash_model
    assert model.model == "gemini-1.5-flash"

# 測試提示詞管理器
def test_prompt_manager():
    prompts = prompt_manager.list_prompts()
    assert "hyde_generation" in prompts
```

### 2. 集成測試

測試完整的 RAG 流程：

```python
async def test_full_rag_chain():
    service = RAGChainService()
    result = await service.generate_report(test_monitoring_data)
    assert result.insight_analysis is not None
    assert result.recommendations is not None
```

## 結論

通過使用 LangChain LCEL 重構，我們實現了：

1. **更高的可讀性** - 聲明式的管道定義讓流程一目了然
2. **更好的可維護性** - 組件解耦，易於修改和測試
3. **更強的擴展性** - 輕鬆添加新模型、新的向量資料庫或自定義流程
4. **向後相容** - 保持原有 API 不變，無需修改現有程式碼

這次重構不僅改善了程式碼品質，也為未來的功能擴展奠定了堅實的基礎。