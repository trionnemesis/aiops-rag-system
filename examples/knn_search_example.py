#!/usr/bin/env python3
"""
KNN 向量搜尋使用範例
展示如何使用 KNN 搜尋功能與 LangGraph RAG 整合
"""

import asyncio
import os
from dotenv import load_dotenv

# 載入環境變數
load_dotenv()

from app.api.knn_langchain_bridge import create_knn_langraph_components
from src.services.knn_search_service import SearchStrategy, KNNSearchParams
from app.graph.graph import build_graph
from langchain_google_genai import ChatGoogleGenerativeAI


async def basic_knn_search_example():
    """基本 KNN 搜尋範例"""
    print("\n=== 基本 KNN 搜尋範例 ===\n")
    
    # 建立 KNN 元件
    components = create_knn_langraph_components(
        index_name="aiops_knowledge_base",
        embedding_model="gemini-embedding-001"
    )
    
    # 取得搜尋服務
    search_service = components["search_service"]
    
    # 測試不同的搜尋策略
    query = "Apache 記憶體使用率過高怎麼辦？"
    
    print(f"查詢: {query}\n")
    
    # 1. 純向量搜尋
    print("1. 純向量搜尋結果:")
    knn_results = await search_service.knn_search(
        query_text=query,
        strategy=SearchStrategy.KNN_ONLY,
        params=KNNSearchParams(k=3)
    )
    
    for i, result in enumerate(knn_results, 1):
        print(f"   {i}. {result.title} (分數: {result.score:.3f})")
    
    # 2. 混合搜尋
    print("\n2. 混合搜尋結果 (向量 + BM25):")
    hybrid_results = await search_service.knn_search(
        query_text=query,
        strategy=SearchStrategy.HYBRID,
        params=KNNSearchParams(k=3)
    )
    
    for i, result in enumerate(hybrid_results, 1):
        print(f"   {i}. {result.title} (分數: {result.score:.3f})")
        if result.highlights:
            print(f"      高亮: {result.highlights[0]}")


async def langraph_integration_example():
    """LangGraph 整合範例"""
    print("\n=== LangGraph RAG 整合範例 ===\n")
    
    # 初始化 LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-pro",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0.1
    )
    
    # 建立 KNN 元件
    components = create_knn_langraph_components(
        index_name="aiops_knowledge_base",
        embedding_model="gemini-embedding-001",
        vector_search_k=10,
        bm25_search_k=8
    )
    
    # 定義 RAG 策略
    from app.graph.policy import RagPolicy
    policy = RagPolicy(
        enable_hyde=True,
        enable_rag_fusion=True,
        enable_rerank=True
    )
    
    # 建立 LangGraph
    graph = build_graph(
        llm=llm,
        retriever=components["hybrid_retriever"],
        bm25_search_fn=components["bm25_search_fn"],
        build_context_fn=components["build_context_fn"],
        policy=policy
    )
    
    # 執行查詢
    query = "MySQL 資料庫出現大量慢查詢，如何優化？"
    print(f"查詢: {query}\n")
    
    result = await graph.ainvoke({
        "query": query
    })
    
    print("RAG 回答:")
    print(result["final_answer"])
    
    print("\n參考文件:")
    for i, doc in enumerate(result.get("final_documents", [])[:3], 1):
        print(f"{i}. {doc.metadata.get('title', 'Unknown')} (分數: {doc.metadata.get('score', 0):.3f})")


async def advanced_search_example():
    """進階搜尋功能範例"""
    print("\n=== 進階搜尋功能範例 ===\n")
    
    components = create_knn_langraph_components()
    search_service = components["search_service"]
    
    # 1. 過濾搜尋
    print("1. 標籤過濾搜尋 (只搜尋 MySQL 相關):")
    filtered_results = await search_service.knn_search(
        query_text="效能優化",
        strategy=SearchStrategy.HYBRID,
        params=KNNSearchParams(
            k=5,
            filter={"term": {"tags": "mysql"}}
        )
    )
    
    for result in filtered_results[:3]:
        print(f"   - {result.title}")
        print(f"     標籤: {', '.join(result.metadata['tags'])}")
    
    # 2. 多向量搜尋
    print("\n2. 多向量搜尋 (查詢擴展):")
    multi_results = await search_service.knn_search(
        query_text="資料庫索引",
        strategy=SearchStrategy.MULTI_VECTOR,
        params=KNNSearchParams(k=5)
    )
    
    for result in multi_results[:3]:
        print(f"   - {result.title} (分數: {result.score:.3f})")
    
    # 3. 重新排序搜尋
    print("\n3. 重新排序搜尋:")
    rerank_results = await search_service.knn_search(
        query_text="高併發導致系統崩潰",
        strategy=SearchStrategy.RERANK,
        params=KNNSearchParams(k=5)
    )
    
    for result in rerank_results[:3]:
        print(f"   - {result.title} (分數: {result.score:.3f})")


async def performance_comparison():
    """效能比較範例"""
    print("\n=== 搜尋策略效能比較 ===\n")
    
    import time
    
    components = create_knn_langraph_components()
    search_service = components["search_service"]
    
    query = "如何處理記憶體洩漏問題"
    strategies = [
        (SearchStrategy.KNN_ONLY, "純向量搜尋"),
        (SearchStrategy.HYBRID, "混合搜尋"),
        (SearchStrategy.MULTI_VECTOR, "多向量搜尋"),
        (SearchStrategy.RERANK, "重新排序")
    ]
    
    print(f"測試查詢: {query}\n")
    
    for strategy, name in strategies:
        start_time = time.time()
        
        results = await search_service.knn_search(
            query_text=query,
            strategy=strategy,
            params=KNNSearchParams(k=5)
        )
        
        elapsed_time = (time.time() - start_time) * 1000  # 轉換為毫秒
        
        print(f"{name}:")
        print(f"  耗時: {elapsed_time:.2f} ms")
        print(f"  最高分數: {results[0].score:.3f}")
        print(f"  最佳結果: {results[0].title}\n")


async def main():
    """主程式"""
    print("🔍 KNN 向量搜尋示範程式")
    print("=" * 50)
    
    try:
        # 執行各種範例
        await basic_knn_search_example()
        await advanced_search_example()
        await performance_comparison()
        await langraph_integration_example()
        
    except Exception as e:
        print(f"\n❌ 錯誤: {e}")
        print("\n請確保:")
        print("1. OpenSearch 服務正在運行")
        print("2. 索引已建立並載入資料")
        print("3. 環境變數已正確設定")


if __name__ == "__main__":
    # 執行主程式
    asyncio.run(main())