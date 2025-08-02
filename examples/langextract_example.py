"""
LangExtract + LangGraph 整合範例
展示如何在 AIOps 場景中使用結構化資訊提取
"""
import os
from datetime import datetime
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import OpenSearchVectorSearch
from opensearchpy import OpenSearch

from src.services.langchain.langextract_service import LangExtractService
from src.services.langchain.chunking_service import (
    ChunkingService, EmbeddingService, ChunkingAndEmbeddingPipeline
)
from app.graph.build import build_graph


def setup_services():
    """設置所有必要的服務"""
    # 初始化 LLM
    api_key = os.getenv("GEMINI_API_KEY")
    llm = ChatGoogleGenerativeAI(
        model="gemini-pro",
        google_api_key=api_key,
        temperature=0.3
    )
    
    # 初始化 Embeddings
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=api_key
    )
    
    # 初始化 LangExtract
    extract_service = LangExtractService(llm=llm)
    
    # 初始化 Chunking
    chunking_service = ChunkingService(
        chunk_size=1000,
        chunk_overlap=200
    )
    
    # 初始化 Embedding
    embedding_service = EmbeddingService(embeddings=embeddings)
    
    # 初始化 OpenSearch
    opensearch_client = OpenSearch(
        hosts=[{"host": "localhost", "port": 9200}],
        http_auth=("admin", "admin"),
        use_ssl=False,
        verify_certs=False
    )
    
    # 創建向量存儲
    vector_store = OpenSearchVectorSearch(
        opensearch_url="http://localhost:9200",
        index_name="aiops-langextract",
        embedding_function=embeddings,
        http_auth=("admin", "admin"),
        use_ssl=False,
        verify_certs=False
    )
    
    # 獲取 retriever
    retriever = vector_store.as_retriever(search_kwargs={"k": 10})
    
    return {
        "llm": llm,
        "extract_service": extract_service,
        "chunking_service": chunking_service,
        "embedding_service": embedding_service,
        "vector_store": vector_store,
        "retriever": retriever
    }


def example_1_basic_extraction():
    """範例 1: 基本的結構化資訊提取"""
    print("\n=== 範例 1: 基本結構化資訊提取 ===\n")
    
    services = setup_services()
    extract_service = services["extract_service"]
    
    # 模擬 Prometheus 告警
    alert_text = """
    FIRING: HighCPUUsage
    Instance: web-prod-03.example.com:9100
    Job: node_exporter
    Service: nginx
    CPU Usage: 92.5%
    Memory Usage: 65%
    Timestamp: 2024-01-15T10:30:00Z
    Alert: CPU usage has been above 90% for more than 5 minutes
    """
    
    # 提取結構化資訊
    result = extract_service.extract(alert_text)
    
    print(f"原始文本:\n{alert_text}")
    print("\n提取的實體:")
    print(f"- 主機名: {result.entities.hostname}")
    print(f"- 服務: {result.entities.service_name}")
    print(f"- CPU 使用率: {result.entities.cpu_usage}%")
    print(f"- 記憶體使用率: {result.entities.memory_usage}%")
    print(f"- 時間戳: {result.entities.timestamp}")
    print(f"\n提取信心分數: {result.confidence}")


def example_2_ingestion_pipeline():
    """範例 2: 完整的資料攝入流程"""
    print("\n\n=== 範例 2: 資料攝入流程 (Extract -> Chunk -> Embed -> Store) ===\n")
    
    services = setup_services()
    
    # 創建管道
    pipeline = ChunkingAndEmbeddingPipeline(
        chunking_service=services["chunking_service"],
        embedding_service=services["embedding_service"],
        extract_service=services["extract_service"]
    )
    
    # 準備多個日誌樣本
    logs = [
        """
        2024-01-15 10:30:00 ERROR [api-gateway] Connection pool exhausted
        Host: api-prod-01.example.com
        Service: user-api
        Error Code: CONN_POOL_001
        Active Connections: 500/500
        Response Time: 5000ms
        """,
        """
        2024-01-15 10:31:00 WARN [database] Slow query detected
        Host: db-master-01.example.com
        Service: mysql
        Query Time: 3.5s
        CPU: 45%, Memory: 78%
        """,
        """
        2024-01-15 10:32:00 ERROR [cache] Redis connection timeout
        Host: cache-01.example.com
        Service: redis
        Error: Connection refused (ECONNREFUSED)
        Port: 6379
        """
    ]
    
    # 準備基礎元數據
    base_metadata_list = [
        {"source": "prometheus", "alert_type": "connection_pool"},
        {"source": "elasticsearch", "log_type": "slow_query"},
        {"source": "loki", "log_type": "connection_error"}
    ]
    
    # 處理日誌
    print("處理日誌並提取結構化資訊...")
    vector_metadata_pairs = pipeline.process(
        texts=logs,
        base_metadata_list=base_metadata_list,
        use_extraction=True
    )
    
    # 存儲到向量資料庫
    print(f"\n共生成 {len(vector_metadata_pairs)} 個向量塊")
    
    # 顯示部分結果
    for i, (vector, metadata) in enumerate(vector_metadata_pairs[:3]):
        print(f"\n塊 {i+1} 元數據:")
        print(f"- 來源: {metadata.get('source')}")
        print(f"- 提取的主機: {metadata.get('extracted_hostname', 'N/A')}")
        print(f"- 提取的服務: {metadata.get('extracted_service_name', 'N/A')}")
        print(f"- 提取的錯誤碼: {metadata.get('extracted_error_code', 'N/A')}")
        print(f"- 信心分數: {metadata.get('extraction_confidence', 'N/A')}")


def example_3_langgraph_rag():
    """範例 3: 使用 LangGraph 進行增強檢索"""
    print("\n\n=== 範例 3: LangGraph RAG with LangExtract ===\n")
    
    services = setup_services()
    
    # 構建 LangGraph
    app = build_graph(
        llm=services["llm"],
        retriever=services["retriever"],
        extract_service=services["extract_service"],
        policy={
            "use_llm_extract": True,
            "use_metadata_filter": True,
            "use_hyde": True,
            "use_rrf": False,
            "top_k": 5
        }
    )
    
    # 測試查詢
    queries = [
        {
            "query": "api-gateway 連線池耗盡的問題",
            "raw_texts": ["ERROR: api-gateway connection pool exhausted on api-prod-01"]
        },
        {
            "query": "分析 db-master-01 的慢查詢問題",
            "raw_texts": ["WARN: Slow query on db-master-01, CPU 45%, query time 3.5s"]
        }
    ]
    
    for i, test_case in enumerate(queries):
        print(f"\n--- 查詢 {i+1}: {test_case['query']} ---")
        
        # 執行 LangGraph
        result = app.invoke(test_case, {"configurable": {"thread_id": f"example-{i}"}})
        
        # 顯示結果
        print(f"\n提取的關鍵資訊:")
        if result.get("extracted_data"):
            for data in result["extracted_data"]:
                raw = data.get("_raw_extracted", {})
                if raw.get("hostname"):
                    print(f"- 主機: {raw['hostname']}")
                if raw.get("service_name"):
                    print(f"- 服務: {raw['service_name']}")
                if raw.get("error_code"):
                    print(f"- 錯誤碼: {raw['error_code']}")
        
        print(f"\n生成的回答:")
        print(result.get("answer", "無回答"))
        
        print(f"\n執行指標:")
        metrics = result.get("metrics", {})
        print(f"- 查詢數: {metrics.get('queries', 0)}")
        print(f"- 檢索文檔數: {metrics.get('docs', 0)}")
        print(f"- 元數據過濾: {metrics.get('metadata_filters', {})}")


def example_4_metadata_filtering():
    """範例 4: 元數據過濾的效果展示"""
    print("\n\n=== 範例 4: 元數據過濾效果 ===\n")
    
    services = setup_services()
    
    # 先攝入一些測試資料
    test_logs = [
        "ERROR: web-01 nginx CPU 95%",
        "ERROR: web-02 apache CPU 85%", 
        "ERROR: db-01 mysql CPU 75%",
        "WARN: cache-01 redis Memory 90%"
    ]
    
    # 為每個日誌添加元數據
    for i, log in enumerate(test_logs):
        extracted = services["extract_service"].extract(log)
        metadata = services["extract_service"].extract_to_metadata(log)
        metadata["id"] = f"log-{i}"
        
        # 存儲到向量資料庫
        services["vector_store"].add_texts(
            texts=[log],
            metadatas=[metadata]
        )
    
    print("已攝入測試資料")
    
    # 構建兩個 LangGraph：一個有過濾，一個沒有
    app_with_filter = build_graph(
        llm=services["llm"],
        retriever=services["retriever"],
        extract_service=services["extract_service"],
        policy={"use_metadata_filter": True}
    )
    
    app_without_filter = build_graph(
        llm=services["llm"],
        retriever=services["retriever"],
        extract_service=None,  # 不使用提取
        policy={"use_metadata_filter": False}
    )
    
    # 測試查詢
    test_query = {
        "query": "web-01 的 CPU 問題",
        "raw_texts": ["查詢 web-01 主機的 CPU 使用率問題"]
    }
    
    print("\n執行查詢（有元數據過濾）:")
    result_with = app_with_filter.invoke(test_query, {"configurable": {"thread_id": "filter-test-1"}})
    print(f"- 檢索到 {len(result_with.get('docs', []))} 個文檔")
    print(f"- 使用的過濾器: {result_with.get('metrics', {}).get('metadata_filters', {})}")
    
    print("\n執行查詢（無元數據過濾）:")
    result_without = app_without_filter.invoke(
        {"query": test_query["query"]}, 
        {"configurable": {"thread_id": "filter-test-2"}}
    )
    print(f"- 檢索到 {len(result_without.get('docs', []))} 個文檔")


def main():
    """執行所有範例"""
    print("=== LangExtract + LangGraph AIOps 整合範例 ===")
    print("確保 OpenSearch 和 Gemini API 已正確設置")
    
    try:
        example_1_basic_extraction()
        example_2_ingestion_pipeline()
        example_3_langgraph_rag()
        example_4_metadata_filtering()
    except Exception as e:
        print(f"\n錯誤: {e}")
        print("請確保：")
        print("1. OpenSearch 正在運行 (localhost:9200)")
        print("2. GEMINI_API_KEY 環境變數已設置")
        print("3. 所有相依套件已安裝")


if __name__ == "__main__":
    main()