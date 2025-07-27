"""
LangChain RAG 服務使用範例
展示如何使用 LCEL 重構後的 RAG 服務
"""
import asyncio
from datetime import datetime
from src.services.langchain import RAGChainService, model_manager, prompt_manager, vector_store_manager
from src.services.rag_service import RAGService


async def example_basic_usage():
    """基本使用範例"""
    print("=== 基本 RAG 服務使用範例 ===\n")
    
    # 初始化服務
    rag_service = RAGService()
    
    # 準備監控數據
    monitoring_data = {
        "主機": "web-server-01",
        "採集時間": datetime.now().isoformat(),
        "CPU使用率": "85%",
        "RAM使用率": "92%",
        "磁碟I/O等待": "高",
        "網路流出量": "異常增加",
        "作業系統Port流量": {
            "Port 80/443 流入連線數": 5000
        },
        "服務指標": {
            "Apache活躍工作程序": 250,
            "Nginx日誌錯誤率": {"4xx": 150, "5xx": 50}
        }
    }
    
    # 生成報告
    print("正在生成維運報告...")
    report = await rag_service.generate_report(monitoring_data)
    
    print(f"\n洞見分析:\n{report.insight_analysis}")
    print(f"\n具體建議:\n{report.recommendations}")
    print(f"\n生成時間: {report.generated_at}")


async def example_with_steps():
    """展示中間步驟的範例"""
    print("\n\n=== 帶中間步驟的 RAG 服務使用範例 ===\n")
    
    # 初始化服務
    rag_service = RAGService()
    
    # 準備監控數據
    monitoring_data = {
        "主機": "db-server-02",
        "採集時間": datetime.now().isoformat(),
        "CPU使用率": "45%",
        "RAM使用率": "78%",
        "磁碟I/O等待": "極高",
        "網路流出量": "正常"
    }
    
    # 生成報告並獲取中間步驟
    print("正在生成維運報告（包含中間步驟）...")
    result = await rag_service.generate_report_with_steps(monitoring_data)
    
    report = result["report"]
    steps = result["steps"]
    
    print(f"\n=== 中間步驟 ===")
    print(f"HyDE 查詢: {steps['hyde_query'][:100]}...")
    print(f"找到文檔數量: {steps['documents_found']}")
    print(f"上下文摘要: {steps['context_summary']}")
    
    print(f"\n=== 最終報告 ===")
    print(f"洞見分析:\n{report.insight_analysis}")
    print(f"具體建議:\n{report.recommendations}")


async def example_direct_langchain():
    """直接使用 LangChain 組件的範例"""
    print("\n\n=== 直接使用 LangChain 組件範例 ===\n")
    
    # 1. 使用模型管理器
    print("1. 模型管理器使用:")
    flash_model = model_manager.flash_model
    pro_model = model_manager.pro_model
    print(f"   - Flash 模型: {flash_model.model}")
    print(f"   - Pro 模型: {pro_model.model}")
    
    # 2. 使用提示詞管理器
    print("\n2. 提示詞管理器使用:")
    available_prompts = prompt_manager.list_prompts()
    print(f"   - 可用提示詞: {available_prompts}")
    
    # 3. 使用向量資料庫管理器
    print("\n3. 向量資料庫管理器使用:")
    retriever = vector_store_manager.as_retriever(search_kwargs={"k": 3})
    print(f"   - 檢索器已創建，設置返回前 3 個結果")
    
    # 4. 創建自定義 RAG 鏈
    print("\n4. 創建自定義 RAG 鏈:")
    rag_chain_service = RAGChainService()
    custom_chain = rag_chain_service.create_custom_chain(
        retriever_kwargs={"search_kwargs": {"k": 3}},
        hyde_enabled=True
    )
    print("   - 自定義 RAG 鏈已創建（啟用 HyDE，返回前 3 個結果）")


async def example_cache_management():
    """快取管理範例"""
    print("\n\n=== 快取管理範例 ===\n")
    
    rag_service = RAGService()
    
    # 查看初始快取狀態
    print("初始快取狀態:")
    cache_info = rag_service.get_cache_info()
    print(f"   - 嵌入快取: {cache_info['embedding_cache']}")
    print(f"   - HyDE 快取: {cache_info['hyde_cache']}")
    
    # 執行一些操作來填充快取
    monitoring_data = {
        "主機": "cache-test-01",
        "CPU使用率": "50%",
        "RAM使用率": "60%"
    }
    
    print("\n執行 RAG 操作...")
    await rag_service.generate_report(monitoring_data)
    
    # 再次查看快取狀態
    print("\n操作後快取狀態:")
    cache_info = rag_service.get_cache_info()
    print(f"   - 嵌入快取: {cache_info['embedding_cache']}")
    print(f"   - HyDE 快取: {cache_info['hyde_cache']}")
    
    # 清除快取
    print("\n清除快取...")
    rag_service.clear_cache()
    
    # 確認快取已清除
    print("\n清除後快取狀態:")
    cache_info = rag_service.get_cache_info()
    print(f"   - 嵌入快取: {cache_info['embedding_cache']}")
    print(f"   - HyDE 快取: {cache_info['hyde_cache']}")


async def main():
    """執行所有範例"""
    try:
        # 基本使用
        await example_basic_usage()
        
        # 帶中間步驟
        await example_with_steps()
        
        # 直接使用 LangChain 組件
        await example_direct_langchain()
        
        # 快取管理
        await example_cache_management()
        
    except Exception as e:
        print(f"\n錯誤: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())