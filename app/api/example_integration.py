"""
示例：如何整合你現有的 LLM 和 Retriever 到新的 LangGraph RAG

這個檔案展示如何將你現有的元件注入到 graph 中
"""

from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import OpenSearchVectorSearch
from langchain_community.embeddings import OpenAIEmbeddings
from opensearchpy import OpenSearch
from langchain_core.documents import Document
from typing import List
import os

# ===== 1. 設定你現有的 LLM =====
# 範例：使用 Gemini
llm = ChatGoogleGenerativeAI(
    model="gemini-pro",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0.1,
)

# 或使用 OpenAI
# llm = ChatOpenAI(
#     model="gpt-4-turbo-preview",
#     api_key=os.getenv("OPENAI_API_KEY"),
#     temperature=0.1,
# )

# ===== 2. 設定你現有的 OpenSearch Vector Retriever =====
# OpenSearch 連線設定
opensearch_client = OpenSearch(
    hosts=[{'host': os.getenv("OPENSEARCH_HOST", "localhost"), 'port': 9200}],
    http_auth=(os.getenv("OPENSEARCH_USER", "admin"), os.getenv("OPENSEARCH_PASSWORD", "admin")),
    use_ssl=True,
    verify_certs=False,
    ssl_show_warn=False,
)

# Embeddings（與你建索引時用的一致）
embeddings = OpenAIEmbeddings(
    model="text-embedding-ada-002",
    openai_api_key=os.getenv("OPENAI_API_KEY"),
)

# Vector Store
vector_store = OpenSearchVectorSearch(
    opensearch_url=f"https://{os.getenv('OPENSEARCH_HOST', 'localhost')}:9200",
    index_name="your_index_name",  # 替換成你的索引名
    embedding_function=embeddings,
    opensearch_client=opensearch_client,
)

# 轉成 retriever
retriever = vector_store.as_retriever(
    search_kwargs={
        "k": 8,  # 預設取前 8 筆
        # 若你有特定的 knn 參數，可以加在這裡
        # "knn": {
        #     "field": "embedding",
        #     "k": 8,
        #     "num_candidates": 100,
        # }
    }
)

# ===== 3. （選配）實作 BM25 搜尋函式 =====
def bm25_search_fn(query: str, top_k: int = 8) -> List[Document]:
    """
    使用 OpenSearch 的標準文字搜尋（BM25）
    """
    # 構建 OpenSearch 查詢
    body = {
        "query": {
            "match": {
                "content": {  # 替換成你的文字欄位名稱
                    "query": query,
                    "operator": "or",
                }
            }
        },
        "size": top_k,
        "_source": ["content", "title", "metadata"],  # 調整成你的欄位
    }
    
    # 執行搜尋
    response = opensearch_client.search(
        index="your_index_name",  # 替換成你的索引名
        body=body
    )
    
    # 轉換成 Document 物件
    docs = []
    for hit in response["hits"]["hits"]:
        source = hit["_source"]
        doc = Document(
            page_content=source.get("content", ""),
            metadata={
                "_id": hit["_id"],
                "title": source.get("title", ""),
                "score": hit["_score"],
                **source.get("metadata", {}),
            }
        )
        docs.append(doc)
    
    return docs

# ===== 4. 自訂上下文組裝函式（可選） =====
def custom_build_context(docs: List[Document], max_chars: int = 6000) -> str:
    """
    你可以在這裡實作自己的上下文組裝邏輯
    """
    context_parts = []
    total_chars = 0
    
    for i, doc in enumerate(docs, 1):
        # 取得文件內容和標題
        content = doc.page_content.strip()
        title = doc.metadata.get("title") or doc.metadata.get("source") or f"文件{i}"
        
        # 格式化
        formatted = f"【{title}】\n{content}\n"
        
        # 檢查長度限制
        if total_chars + len(formatted) > max_chars:
            break
            
        context_parts.append(formatted)
        total_chars += len(formatted)
    
    return "\n".join(context_parts)

# ===== 5. 在 routes.py 中使用 =====
"""
在 app/api/routes.py 中，將上面的物件注入：

from app.api.example_integration import llm, retriever, bm25_search_fn, custom_build_context

# 然後在 build_graph 時使用：
graph_app = build_graph(
    llm=llm,
    retriever=retriever,
    bm25_search_fn=bm25_search_fn,  # 或 None
    build_context_fn=custom_build_context,  # 或使用預設
    policy=policy,
)
"""