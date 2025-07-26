from typing import Dict, Any, List
import json
from datetime import datetime
from async_lru import alru_cache
from src.services.gemini_service import GeminiService
from src.services.opensearch_service import OpenSearchService
from src.services.prometheus_service import PrometheusService
from src.utils.prompts import PromptTemplates
from src.models.schemas import InsightReport

class RAGService:
    def __init__(self):
        self.gemini = GeminiService()
        self.opensearch = OpenSearchService()
        self.prometheus = PrometheusService()
        self.prompts = PromptTemplates()
        
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
    
    @alru_cache(maxsize=100, ttl=3600)  # 快取最多100個項目，每個項目保留1小時
    async def _get_cached_embedding(self, text: str) -> List[float]:
        """帶快取的嵌入向量生成"""
        return await self.gemini.generate_embedding(text)
    
    @alru_cache(maxsize=50, ttl=1800)  # 快取最多50個項目，每個項目保留30分鐘
    async def _get_cached_hyde(self, hyde_prompt: str) -> str:
        """帶快取的假設性文件生成"""
        return await self.gemini.generate_hyde(hyde_prompt)
        
    async def generate_report(self, monitoring_data: Dict[str, Any]) -> InsightReport:
        """執行完整的 RAG 流程生成維運報告"""
        
        # Step 1: HyDE Generation (使用快取)
        hyde_prompt = self.prompts.HYDE_GENERATION.format(
            monitoring_data=json.dumps(monitoring_data, ensure_ascii=False, indent=2)
        )
        hypothetical_doc = await self._get_cached_hyde(hyde_prompt)
        
        # Step 2: Generate Embedding and Search (使用快取)
        query_embedding = await self._get_cached_embedding(hypothetical_doc)
        similar_docs = await self.opensearch.search_similar_documents(query_embedding)
        
        # Step 3: Summarize Retrieved Documents (改善後：整合文件摘要)
        if similar_docs:
            # 將所有檢索到的文件內容合併成一個字串
            all_docs_content = "\n\n--- 文件分隔 ---\n\n".join([
                f"文件 {i+1}:\n{doc['content']}" 
                for i, doc in enumerate(similar_docs)
            ])
            
            # 僅呼叫一次摘要功能
            summary_prompt = self.prompts.SUMMARY_REFINEMENT.format(
                monitoring_data=json.dumps(monitoring_data, ensure_ascii=False),
                document=all_docs_content
            )
            consolidated_summary = await self.gemini.summarize_document(summary_prompt)
            
            # 為了保持後續處理邏輯不變，將單一摘要結果放入列表中
            summaries = [consolidated_summary]
        else:
            summaries = []
        
        # Step 4: Generate Final Report
        summaries_text = "\n- ".join(summaries) if summaries else "無相關文件摘要"
        final_prompt = self.prompts.FINAL_REPORT.format(
            monitoring_data=json.dumps(monitoring_data, ensure_ascii=False, indent=2),
            summaries=summaries_text
        )
        report_content = await self.gemini.generate_final_report(final_prompt)
        
        # Create Report Object
        report = InsightReport(
            insight_analysis=report_content["insight_analysis"],
            recommendations=report_content["recommendations"],
            generated_at=datetime.now()
        )
        
        return report
    
    async def enrich_with_prometheus(self, hostname: str, 
                                   monitoring_data: Dict[str, Any]) -> Dict[str, Any]:
        """從 Prometheus 獲取實時數據豐富監控資料"""
        try:
            prometheus_metrics = await self.prometheus.get_host_metrics(hostname)
            # 合併或更新監控數據
            for key, value in prometheus_metrics.items():
                if key not in monitoring_data:
                    monitoring_data[key] = value
        except Exception as e:
            print(f"Failed to enrich with Prometheus data: {e}")
        
        return monitoring_data
    
    def clear_cache(self):
        """清除所有快取（用於測試或維護）"""
        self._get_cached_embedding.cache_clear()
        self._get_cached_hyde.cache_clear()
    
    def get_cache_info(self) -> Dict[str, Any]:
        """獲取快取狀態資訊"""
        return {
            "embedding_cache": {
                "hits": self._get_cached_embedding.cache_info().hits,
                "misses": self._get_cached_embedding.cache_info().misses,
                "maxsize": self._get_cached_embedding.cache_info().maxsize,
                "currsize": self._get_cached_embedding.cache_info().currsize
            },
            "hyde_cache": {
                "hits": self._get_cached_hyde.cache_info().hits,
                "misses": self._get_cached_hyde.cache_info().misses,
                "maxsize": self._get_cached_hyde.cache_info().maxsize,
                "currsize": self._get_cached_hyde.cache_info().currsize
            }
        }