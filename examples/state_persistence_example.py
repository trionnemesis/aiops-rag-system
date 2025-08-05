"""
狀態持久化與恢復範例
演示如何使用 Redis 來持久化 LangGraph 的執行狀態，
以便在中斷後能從最後成功的步驟恢復。
"""

import os
import time
import uuid
from langchain_google_genai import ChatGoogleGenerativeAI
from opensearchpy import OpenSearch
from app.graph.build import build_graph
from app.services.opensearch_service import OpenSearchService
from app.services.bm25_service import BM25Service
from dotenv import load_dotenv

# 載入環境變數
load_dotenv()

def create_retriever(opensearch_service):
    """創建一個簡單的檢索器"""
    class SimpleRetriever:
        def __init__(self, os_service):
            self.os_service = os_service
            
        def invoke(self, query: str, **kwargs):
            # 使用 OpenSearch 進行檢索
            results = self.os_service.search(
                query=query,
                index_name="aiops-knowledge",
                size=5
            )
            
            # 轉換為 Document 格式
            from langchain_core.documents import Document
            docs = []
            for hit in results.get('hits', {}).get('hits', []):
                doc = Document(
                    page_content=hit['_source'].get('content', ''),
                    metadata={
                        'id': hit['_id'],
                        'score': hit['_score'],
                        'title': hit['_source'].get('title', ''),
                        'source': hit['_source'].get('source', '')
                    }
                )
                docs.append(doc)
            return docs
    
    return SimpleRetriever(opensearch_service)

def simulate_interruption_example():
    """模擬中斷與恢復的範例"""
    
    # 初始化服務
    print("初始化服務...")
    
    # LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-pro",
        temperature=0.3,
        google_api_key=os.getenv("GEMINI_API_KEY")
    )
    
    # OpenSearch
    opensearch_service = OpenSearchService(
        host="localhost",
        port=9200,
        username=None,
        password=None,
        use_ssl=False
    )
    
    # Retriever
    retriever = create_retriever(opensearch_service)
    
    # BM25 (可選)
    bm25_service = BM25Service(opensearch_service)
    
    # 建構 Graph
    app = build_graph(
        llm=llm,
        retriever=retriever,
        bm25_search_fn=lambda q, k: bm25_service.search(q, k) if bm25_service else [],
        policy={
            "max_retries": 3,
            "retry_delay": 1.0,
            "require_plan": True,
            "require_validation": True
        }
    )
    
    # 產生唯一的 thread_id
    thread_id = f"demo-{uuid.uuid4()}"
    print(f"Thread ID: {thread_id}")
    
    # 第一次執行 - 模擬中斷
    print("\n=== 第一次執行（會在中途模擬中斷）===")
    
    try:
        # 使用一個會觸發多步驟處理的複雜查詢
        initial_state = {
            "query": "分析過去一週系統的效能問題，包括 CPU 使用率、記憶體洩漏和回應時間異常",
            "request_id": str(uuid.uuid4())
        }
        
        # 注入一個會在某個步驟失敗的條件（僅供演示）
        initial_state["_simulate_failure_at_step"] = "synthesize"  # 在合成步驟模擬失敗
        
        result = app.invoke(
            initial_state,
            config={
                "configurable": {
                    "thread_id": thread_id,
                    "run_id": initial_state["request_id"]
                }
            }
        )
        
        print("第一次執行完成（不應該到這裡）")
        
    except Exception as e:
        print(f"執行中斷: {e}")
        print("狀態已自動保存到 Redis")
    
    # 等待一下，模擬真實的中斷情況
    print("\n等待 3 秒後恢復執行...")
    time.sleep(3)
    
    # 第二次執行 - 從中斷處恢復
    print("\n=== 第二次執行（從中斷處恢復）===")
    
    try:
        # 使用相同的 thread_id，但新的 request_id
        resume_state = {
            "query": initial_state["query"],  # 保持相同的查詢
            "request_id": str(uuid.uuid4()),  # 新的請求 ID
            "_resume": True  # 標記為恢復執行
        }
        
        # 從上次中斷的地方繼續
        result = app.invoke(
            resume_state,
            config={
                "configurable": {
                    "thread_id": thread_id,  # 使用相同的 thread_id
                    "run_id": resume_state["request_id"]
                }
            }
        )
        
        print("\n執行成功恢復並完成！")
        print(f"最終答案: {result.get('answer', 'No answer')[:200]}...")
        print(f"檢索到的文檔數: {len(result.get('docs', []))}")
        
    except Exception as e:
        print(f"恢復執行失敗: {e}")

def check_state_example():
    """檢查保存的狀態範例"""
    print("\n=== 檢查保存的狀態 ===")
    
    # 這需要直接訪問 Redis
    import redis
    
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
    r = redis.from_url(redis_url)
    
    # 列出所有保存的 thread
    print("保存的 threads:")
    for key in r.scan_iter("checkpoint:*"):
        print(f"  - {key.decode()}")
    
    # 可以實現更詳細的狀態檢查功能

if __name__ == "__main__":
    print("狀態持久化與恢復範例")
    print("=" * 50)
    
    # 確保 Redis 可用
    redis_url = os.getenv("REDIS_URL")
    if not redis_url:
        print("警告: REDIS_URL 未設定，將使用記憶體 checkpoint")
        print("請設定 REDIS_URL 環境變數以啟用持久化功能")
    else:
        print(f"使用 Redis: {redis_url}")
    
    # 執行範例
    simulate_interruption_example()
    
    # 檢查狀態
    if redis_url:
        check_state_example()