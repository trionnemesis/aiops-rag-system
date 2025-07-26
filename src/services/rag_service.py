from typing import Dict, Any, List
import json
from datetime import datetime
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
        
    async def generate_report(self, monitoring_data: Dict[str, Any]) -> InsightReport:
        """執行完整的 RAG 流程生成維運報告"""
        
        # Step 1: HyDE Generation
        hyde_prompt = self.prompts.HYDE_GENERATION.format(
            monitoring_data=json.dumps(monitoring_data, ensure_ascii=False, indent=2)
        )
        hypothetical_doc = await self.gemini.generate_hyde(hyde_prompt)
        
        # Step 2: Generate Embedding and Search
        query_embedding = await self.gemini.generate_embedding(hypothetical_doc)
        similar_docs = await self.opensearch.search_similar_documents(query_embedding)
        
        # Step 3: Summarize Retrieved Documents
        summaries = []
        for doc in similar_docs:
            summary_prompt = self.prompts.SUMMARY_REFINEMENT.format(
                monitoring_data=json.dumps(monitoring_data, ensure_ascii=False),
                document=doc["content"]
            )
            summary = await self.gemini.summarize_document(summary_prompt)
            summaries.append(summary)
        
        # Step 4: Generate Final Report
        summaries_text = "\n- ".join(summaries)
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