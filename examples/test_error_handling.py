#!/usr/bin/env python3
"""
錯誤處理功能演示腳本

這個腳本展示了系統中各種錯誤處理機制的運作方式。
"""
import asyncio
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.services.exceptions import (
    VectorDBError, GeminiAPIError, PrometheusError,
    HyDEGenerationError, DocumentRetrievalError,
    ReportGenerationError, CacheError
)
from src.services.langchain.rag_chain_service import RAGChainService
from src.models.schemas import InsightReport
from datetime import datetime


async def demonstrate_error_handling():
    """演示各種錯誤處理情境"""
    
    print("=== AIOps RAG 系統錯誤處理演示 ===\n")
    
    # 1. 演示自定義例外類別
    print("1. 自定義例外類別階層：")
    print("   - RAGServiceError (基礎例外)")
    print("   - VectorDBError (向量資料庫錯誤)")
    print("   - GeminiAPIError (AI 模型錯誤)")
    print("   - PrometheusError (監控服務錯誤)")
    print("   - HyDEGenerationError (HyDE 生成錯誤)")
    print("   - DocumentRetrievalError (文檔檢索錯誤)")
    print("   - ReportGenerationError (報告生成錯誤)")
    print("   - CacheError (快取操作錯誤)\n")
    
    # 2. 演示錯誤拋出和捕捉
    print("2. 錯誤處理模式演示：")
    
    # 模擬 VectorDB 錯誤
    try:
        raise VectorDBError("Failed to connect to OpenSearch: Connection timeout")
    except VectorDBError as e:
        print(f"   ✓ 捕捉到 VectorDBError: {e}")
    
    # 模擬 Gemini API 錯誤
    try:
        raise GeminiAPIError("API quota exceeded")
    except GeminiAPIError as e:
        print(f"   ✓ 捕捉到 GeminiAPIError: {e}")
    
    # 3. 演示 Fallback 機制
    print("\n3. Fallback 機制演示：")
    
    class MockRAGChainService:
        """模擬的 RAG Chain 服務，用於演示 fallback"""
        
        async def retrieve_with_hyde(self, query: str):
            """模擬 HyDE 檢索失敗"""
            raise HyDEGenerationError("HyDE generation failed")
        
        async def retrieve_direct(self, query: str):
            """直接檢索（fallback）"""
            return ["Document 1", "Document 2"]
        
        async def safe_retrieval(self, query: str):
            """安全的檢索，包含 fallback"""
            try:
                # 嘗試使用 HyDE
                return await self.retrieve_with_hyde(query)
            except HyDEGenerationError as e:
                print(f"   - HyDE 失敗: {e}")
                print("   - 使用 fallback 直接檢索...")
                # Fallback 到直接檢索
                return await self.retrieve_direct(query)
    
    mock_service = MockRAGChainService()
    documents = await mock_service.safe_retrieval("test query")
    print(f"   ✓ Fallback 成功，檢索到 {len(documents)} 個文檔")
    
    # 4. 演示錯誤恢復策略
    print("\n4. 錯誤恢復策略：")
    
    async def enrich_with_prometheus_safe(data: dict) -> dict:
        """安全的 Prometheus 數據豐富（不會中斷主流程）"""
        try:
            # 模擬 Prometheus 服務不可用
            raise PrometheusError("Prometheus service is down")
        except PrometheusError as e:
            print(f"   - Prometheus 錯誤（非關鍵）: {e}")
            print("   - 繼續使用原始數據...")
            return data  # 返回原始數據，不中斷流程
    
    monitoring_data = {"主機": "test-01", "CPU使用率": "50%"}
    enriched_data = await enrich_with_prometheus_safe(monitoring_data)
    print(f"   ✓ 即使 Prometheus 失敗，仍可繼續處理")
    
    # 5. 演示錯誤訊息格式
    print("\n5. 統一的錯誤回應格式：")
    
    def format_error_response(exc: Exception) -> dict:
        """格式化錯誤回應"""
        return {
            "status": "error",
            "message": str(exc),
            "error_type": exc.__class__.__name__,
            "details": {
                "timestamp": datetime.now().isoformat(),
                "service": "aiops-rag"
            }
        }
    
    error = VectorDBError("Connection pool exhausted")
    error_response = format_error_response(error)
    print(f"   {error_response}")
    
    # 6. 演示錯誤日誌記錄最佳實踐
    print("\n6. 錯誤日誌記錄最佳實踐：")
    
    import logging
    logger = logging.getLogger(__name__)
    
    # 預期的外部服務錯誤
    try:
        raise PrometheusError("Connection timeout")
    except PrometheusError as e:
        logger.warning(f"Prometheus service unavailable: {str(e)}")
        print("   ✓ 預期錯誤使用 WARNING 等級")
    
    # 未預期的系統錯誤
    try:
        raise Exception("Unexpected internal error")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        print("   ✓ 未預期錯誤使用 ERROR 等級並包含 stack trace")
    
    print("\n=== 演示完成 ===")
    print("\n主要改進：")
    print("1. ✅ 定義了細分的業務邏輯例外")
    print("2. ✅ 實作了 HyDE fallback 機制")
    print("3. ✅ 非關鍵服務失敗不影響主流程")
    print("4. ✅ 統一的錯誤回應格式")
    print("5. ✅ 適當的日誌記錄策略")


if __name__ == "__main__":
    # 設定日誌
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 執行演示
    asyncio.run(demonstrate_error_handling())