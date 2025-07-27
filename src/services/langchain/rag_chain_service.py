"""
LangChain RAG 鏈服務
使用 LCEL (LangChain Expression Language) 重構 RAG 流程
"""
from typing import Dict, Any, List, Optional
import json
from datetime import datetime
from langchain_core.runnables import (
    RunnablePassthrough, 
    RunnableParallel,
    RunnableLambda
)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from async_lru import alru_cache

from src.services.langchain.model_manager import model_manager
from src.services.langchain.prompt_manager import prompt_manager
from src.services.langchain.vector_store_manager import vector_store_manager
from src.services.prometheus_service import PrometheusService
from src.models.schemas import InsightReport
from src.config import settings


class RAGChainService:
    """使用 LCEL 實現的 RAG 服務"""
    
    def __init__(self):
        self.prometheus = PrometheusService()
        self._initialize_chains()
        
    def _initialize_chains(self):
        """初始化所有 LCEL 鏈"""
        
        # 1. HyDE 生成鏈
        self.hyde_chain = (
            prompt_manager.get_prompt("hyde_generation")
            | model_manager.flash_model
            | StrOutputParser()
        )
        
        # 2. 文檔檢索和格式化鏈
        self.retriever = vector_store_manager.as_retriever()
        
        # 3. 文檔摘要鏈（批次處理）
        self.summary_chain = (
            prompt_manager.get_prompt("summary_refinement")
            | model_manager.flash_model
            | StrOutputParser()
        )
        
        # 4. 最終報告生成鏈
        self.report_chain = (
            prompt_manager.get_prompt("final_report")
            | model_manager.pro_model
            | StrOutputParser()
            | RunnableLambda(self._parse_report_output)
        )
        
        # 5. 完整的 RAG 鏈
        self.full_rag_chain = self._build_full_rag_chain()
    
    def _build_full_rag_chain(self):
        """構建完整的 RAG 鏈"""
        
        def format_docs(docs: List[Document]) -> str:
            """格式化文檔列表為字符串"""
            if not docs:
                return "無相關文件"
            return "\n\n--- 文件分隔 ---\n\n".join([
                f"文件 {i+1}:\n{doc.page_content}" 
                for i, doc in enumerate(docs)
            ])
        
        def prepare_monitoring_data(input_dict: dict) -> str:
            """準備監控數據的 JSON 字符串"""
            return json.dumps(
                input_dict.get("monitoring_data", {}), 
                ensure_ascii=False, 
                indent=2
            )
        
        # 使用 LCEL 構建完整的 RAG 鏈
        chain = (
            # 準備輸入
            RunnableParallel(
                monitoring_data=RunnablePassthrough(),
                monitoring_data_str=RunnableLambda(prepare_monitoring_data)
            )
            # HyDE 生成
            | RunnableParallel(
                monitoring_data=lambda x: x["monitoring_data"],
                monitoring_data_str=lambda x: x["monitoring_data_str"],
                hyde_query=lambda x: self.hyde_chain.invoke({
                    "monitoring_data": x["monitoring_data_str"]
                })
            )
            # 檢索相關文檔
            | RunnableParallel(
                monitoring_data=lambda x: x["monitoring_data"],
                monitoring_data_str=lambda x: x["monitoring_data_str"],
                documents=lambda x: self.retriever.invoke(x["hyde_query"])
            )
            # 格式化文檔並生成摘要
            | RunnableParallel(
                monitoring_data=lambda x: x["monitoring_data"],
                monitoring_data_str=lambda x: x["monitoring_data_str"],
                context=lambda x: self._generate_summary_context(
                    x["documents"], 
                    x["monitoring_data_str"]
                )
            )
            # 生成最終報告
            | lambda x: self.report_chain.invoke({
                "monitoring_data": x["monitoring_data_str"],
                "context": x["context"]
            })
        )
        
        return chain
    
    def _generate_summary_context(self, documents: List[Document], 
                                 monitoring_data_str: str) -> str:
        """生成摘要上下文"""
        if not documents:
            return "無相關歷史經驗"
        
        # 合併所有文檔內容
        all_docs_content = "\n\n--- 文件分隔 ---\n\n".join([
            f"文件 {i+1}:\n{doc.page_content}" 
            for i, doc in enumerate(documents)
        ])
        
        # 使用摘要鏈處理
        summary = self.summary_chain.invoke({
            "monitoring_data": monitoring_data_str,
            "context": all_docs_content
        })
        
        return summary
    
    def _parse_report_output(self, report_text: str) -> Dict[str, str]:
        """解析報告輸出"""
        # 解析「洞見分析」和「具體建議」部分
        parts = report_text.split("具體建議")
        
        insight = ""
        recommendations = ""
        
        if len(parts) >= 1:
            insight = parts[0].replace("洞見分析", "").strip()
        if len(parts) >= 2:
            recommendations = parts[1].strip()
        
        return {
            "insight_analysis": insight,
            "recommendations": recommendations
        }
    
    @alru_cache(maxsize=100, ttl=3600)
    async def _get_cached_embedding(self, text: str) -> List[float]:
        """帶快取的嵌入向量生成"""
        embeddings = await model_manager.embedding_model.aembed_documents([text])
        return embeddings[0]
    
    @alru_cache(maxsize=50, ttl=1800)
    async def _get_cached_hyde(self, monitoring_data_str: str) -> str:
        """帶快取的 HyDE 生成"""
        return await self.hyde_chain.ainvoke({
            "monitoring_data": monitoring_data_str
        })
    
    async def generate_report(self, monitoring_data: Dict[str, Any]) -> InsightReport:
        """生成維運報告（主要介面）
        
        Args:
            monitoring_data: 監控數據字典
            
        Returns:
            InsightReport 物件
        """
        # 執行完整的 RAG 鏈
        result = await self.full_rag_chain.ainvoke({
            "monitoring_data": monitoring_data
        })
        
        # 創建報告物件
        report = InsightReport(
            insight_analysis=result["insight_analysis"],
            recommendations=result["recommendations"],
            generated_at=datetime.now()
        )
        
        return report
    
    async def generate_report_with_steps(self, monitoring_data: Dict[str, Any]) -> Dict[str, Any]:
        """生成報告並返回中間步驟（用於調試）
        
        Args:
            monitoring_data: 監控數據字典
            
        Returns:
            包含報告和中間步驟的字典
        """
        monitoring_data_str = json.dumps(monitoring_data, ensure_ascii=False, indent=2)
        
        # Step 1: HyDE 生成
        hyde_query = await self._get_cached_hyde(monitoring_data_str)
        
        # Step 2: 文檔檢索
        documents = await vector_store_manager.similarity_search(hyde_query)
        
        # Step 3: 生成摘要
        context = self._generate_summary_context(documents, monitoring_data_str)
        
        # Step 4: 生成最終報告
        report_result = await self.report_chain.ainvoke({
            "monitoring_data": monitoring_data_str,
            "context": context
        })
        
        # 創建報告物件
        report = InsightReport(
            insight_analysis=report_result["insight_analysis"],
            recommendations=report_result["recommendations"],
            generated_at=datetime.now()
        )
        
        return {
            "report": report,
            "steps": {
                "hyde_query": hyde_query,
                "documents_found": len(documents),
                "context_summary": context[:200] + "..." if len(context) > 200 else context
            }
        }
    
    async def enrich_with_prometheus(self, hostname: str, 
                                   monitoring_data: Dict[str, Any]) -> Dict[str, Any]:
        """從 Prometheus 獲取實時數據豐富監控資料
        
        Args:
            hostname: 主機名稱
            monitoring_data: 原始監控數據
            
        Returns:
            豐富後的監控數據
        """
        try:
            prometheus_metrics = await self.prometheus.get_host_metrics(hostname)
            # 合併或更新監控數據
            for key, value in prometheus_metrics.items():
                if key not in monitoring_data:
                    monitoring_data[key] = value
        except Exception as e:
            print(f"Failed to enrich with Prometheus data: {e}")
        
        return monitoring_data
    
    def create_custom_chain(self, 
                           retriever_kwargs: Optional[dict] = None,
                           hyde_enabled: bool = True) -> Any:
        """創建自定義 RAG 鏈
        
        Args:
            retriever_kwargs: 檢索器參數
            hyde_enabled: 是否啟用 HyDE
            
        Returns:
            自定義的 RAG 鏈
        """
        # 獲取檢索器
        if retriever_kwargs:
            retriever = vector_store_manager.as_retriever(**retriever_kwargs)
        else:
            retriever = self.retriever
        
        # 構建鏈
        if hyde_enabled:
            # 帶 HyDE 的鏈
            chain = (
                {"context": self.hyde_chain | retriever, 
                 "question": RunnablePassthrough()}
                | prompt_manager.get_prompt("rag_query")
                | model_manager.pro_model
                | StrOutputParser()
            )
        else:
            # 不帶 HyDE 的鏈
            chain = (
                {"context": retriever, 
                 "question": RunnablePassthrough()}
                | prompt_manager.get_prompt("rag_query")
                | model_manager.pro_model
                | StrOutputParser()
            )
        
        return chain
    
    def clear_cache(self):
        """清除所有快取"""
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