from typing import Dict, Any, List
import json
from datetime import datetime
from async_lru import alru_cache

# 使用新的 LangChain 實作
from src.services.langchain.rag_chain_service import RAGChainService
from src.models.schemas import InsightReport


class RAGService:
    """
    RAG 服務的包裝類
    保持原有介面不變，但內部使用 LangChain 實作
    """
    def __init__(self):
        # 使用新的 LangChain RAG 服務
        self.rag_chain_service = RAGChainService()
        
    @staticmethod
    def _create_cache_key(monitoring_data: Dict[str, Any]) -> str:
        """根據監控數據生成一個穩定的快取鍵"""
        # 只取關鍵指標，並排序，避免順序問題
        key_data = {
            "host": monitoring_data.get("主機", ""),
            "cpu": monitoring_data.get("CPU使用率", 0),
            "ram": monitoring_data.get("RAM使用率", 0),
            "disk": monitoring_data.get("磁碟使用率", 0),
            "service": monitoring_data.get("服務名稱", "")
        }
        return json.dumps(key_data, sort_keys=True)
        
    async def generate_report(self, monitoring_data: Dict[str, Any]) -> InsightReport:
        """執行完整的 RAG 流程生成維運報告
        
        使用 LangChain LCEL 實現的 RAG 流程
        """
        return await self.rag_chain_service.generate_report(monitoring_data)
    
    async def generate_report_with_steps(self, monitoring_data: Dict[str, Any]) -> Dict[str, Any]:
        """生成報告並返回中間步驟（用於調試）"""
        return await self.rag_chain_service.generate_report_with_steps(monitoring_data)
    
    async def enrich_with_prometheus(self, hostname: str, 
                                   monitoring_data: Dict[str, Any]) -> Dict[str, Any]:
        """從 Prometheus 獲取實時數據豐富監控資料"""
        return await self.rag_chain_service.enrich_with_prometheus(hostname, monitoring_data)
    
    def clear_cache(self):
        """清除所有快取（用於測試或維護）"""
        self.rag_chain_service.clear_cache()
    
    def get_cache_info(self) -> Dict[str, Any]:
        """獲取快取狀態資訊"""
        return self.rag_chain_service.get_cache_info()
    
    def create_custom_chain(self, **kwargs):
        """創建自定義 RAG 鏈"""
        return self.rag_chain_service.create_custom_chain(**kwargs)