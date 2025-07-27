"""
LangChain RAG 鏈服務
使用 LCEL (LangChain Expression Language) 重構 RAG 流程
"""
import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import GoogleGenerativeAI
from async_lru import alru_cache

from src.services.exceptions import (
    HyDEGenerationError, DocumentRetrievalError, 
    ReportGenerationError, PrometheusError, GeminiAPIError
)
from src.services.langchain import (
    model_manager,
    prompt_manager,
    vector_store_manager
)
from src.services.prometheus_service import PrometheusService
from src.models.schemas import InsightReport


class RAGChainService:
    """
    使用 LangChain Expression Language (LCEL) 實現的 RAG 服務
    """
    
    def __init__(self):
        self.prometheus = PrometheusService()
        self._setup_chains()
    
    def _setup_chains(self):
        """初始化所有需要的鏈"""
        # 1. HyDE Chain
        self.hyde_chain = self._build_hyde_chain()
        
        # 2. Retriever
        self.retriever = vector_store_manager.as_retriever(
            search_kwargs={"k": 10}
        )
        
        # 3. Report Generation Chain
        self.report_chain = self._build_report_chain()
        
        # 4. Full RAG Chain
        self.full_rag_chain = self._build_full_rag_chain()
        
        # 5. Multi-Query Chain (RAG-Fusion)
        self.multi_query_chain = self._build_multi_query_chain()
    
    def _build_hyde_chain(self):
        """建立 HyDE (Hypothetical Document Embeddings) 鏈"""
        try:
            hyde_prompt = prompt_manager.get_prompt("hyde_generation")
            
            hyde_chain = (
                hyde_prompt
                | model_manager.pro_model
                | StrOutputParser()
            )
            
            return hyde_chain
        except Exception as e:
            raise HyDEGenerationError(f"Failed to build HyDE chain: {str(e)}")
    
    def _build_report_chain(self):
        """建立報告生成鏈"""
        try:
            report_prompt = prompt_manager.get_prompt("final_report")
            
            report_chain = (
                report_prompt
                | model_manager.flash_model
                | StrOutputParser()
                | self._parse_report_sections
            )
            
            return report_chain
        except Exception as e:
            raise ReportGenerationError(f"Failed to build report chain: {str(e)}")
    
    def _build_multi_query_chain(self):
        """建立多查詢生成鏈 (RAG-Fusion)"""
        try:
            multi_query_prompt = prompt_manager.get_prompt("multi_query_generation")
            
            multi_query_chain = (
                multi_query_prompt
                | model_manager.flash_model
                | StrOutputParser()
                | (lambda text: [q.strip() for q in text.strip().split('\n') if q.strip()])
            )
            
            return multi_query_chain
        except Exception as e:
            raise ReportGenerationError(f"Failed to build multi-query chain: {str(e)}")
    
    async def _multi_query_retrieval(self, monitoring_data_str: str) -> List[Any]:
        """使用多查詢進行檢索 (RAG-Fusion)"""
        try:
            # 生成多個查詢
            queries = await self.multi_query_chain.ainvoke({
                "monitoring_data": monitoring_data_str
            })
            
            # 對每個查詢進行檢索
            all_documents = []
            seen_contents = set()
            
            for query in queries[:3]:  # 限制最多3個查詢
                try:
                    docs = await vector_store_manager.similarity_search(query, k=5)
                    # 去重
                    for doc in docs:
                        if doc.page_content not in seen_contents:
                            seen_contents.add(doc.page_content)
                            all_documents.append(doc)
                except Exception as e:
                    import logging
                    logging.warning(f"Query '{query}' failed: {str(e)}")
            
            # 如果多查詢失敗，至少用原始數據查詢一次
            if not all_documents:
                all_documents = await vector_store_manager.similarity_search(monitoring_data_str, k=10)
            
            return all_documents[:10]  # 返回最多10個文檔
        except Exception as e:
            import logging
            logging.error(f"Multi-query retrieval failed: {str(e)}")
            # Fallback to single query
            return await vector_store_manager.similarity_search(monitoring_data_str, k=10)
    
    def _build_full_rag_chain(self):
        """建立完整的 RAG 鏈"""
        try:
            # 定義不使用 HyDE 的備用檢索邏輯
            def fallback_retrieval(x):
                """當 HyDE 失敗時的備用檢索"""
                try:
                    return self.retriever.invoke(x["monitoring_data_str"])
                except Exception as e:
                    raise DocumentRetrievalError(f"Fallback retrieval failed: {str(e)}")
            
            # 定義帶有 HyDE 的主要檢索邏輯
            def hyde_retrieval(x):
                """使用 HyDE 進行檢索"""
                try:
                    hyde_query = self.hyde_chain.invoke({
                        "monitoring_data": x["monitoring_data_str"]
                    })
                    return self.retriever.invoke(hyde_query)
                except Exception as e:
                    # 記錄錯誤但不中斷，讓 fallback 處理
                    import logging
                    logging.warning(f"HyDE retrieval failed: {str(e)}")
                    raise e
            
            # 建立具有 fallback 的完整鏈
            full_chain = (
                RunnableParallel(
                    monitoring_data=lambda x: x["monitoring_data"],
                    monitoring_data_str=lambda x: json.dumps(
                        x["monitoring_data"], 
                        ensure_ascii=False, 
                        indent=2
                    )
                ) |
                RunnableParallel(
                    monitoring_data=lambda x: x["monitoring_data"],
                    monitoring_data_str=lambda x: x["monitoring_data_str"],
                    documents=lambda x: self._safe_retrieval(x)
                ) |
                RunnableParallel(
                    monitoring_data_str=lambda x: x["monitoring_data_str"],
                    context=lambda x: self._generate_summary_context(
                        x["documents"], 
                        x["monitoring_data_str"]
                    )
                ) |
                self.report_chain
            )
            
            return full_chain
        except Exception as e:
            raise ReportGenerationError(f"Failed to build full RAG chain: {str(e)}")
    
    def _safe_retrieval(self, x: Dict[str, Any]) -> List[Any]:
        """安全的文檔檢索，包含多查詢、HyDE 和 fallback"""
        monitoring_data_str = x["monitoring_data_str"]
        
        # 1. 首先嘗試多查詢檢索 (RAG-Fusion)
        try:
            import asyncio
            loop = asyncio.get_event_loop()
            documents = loop.run_until_complete(
                self._multi_query_retrieval(monitoring_data_str)
            )
            if documents and len(documents) >= 3:  # 如果找到足夠的文檔
                return documents
        except Exception as e:
            import logging
            logging.warning(f"Multi-query retrieval failed: {str(e)}")
        
        # 2. 嘗試使用 HyDE
        try:
            hyde_query = self.hyde_chain.invoke({
                "monitoring_data": monitoring_data_str
            })
            documents = self.retriever.invoke(hyde_query)
            if documents:
                return documents
        except Exception as e:
            import logging
            logging.warning(f"HyDE retrieval failed, using fallback: {str(e)}")
        
        # 3. 使用 fallback（直接用監控數據檢索）
        try:
            return self.retriever.invoke(monitoring_data_str)
        except Exception as e:
            raise DocumentRetrievalError(f"All retrieval methods failed: {str(e)}")
    
    def _generate_summary_context(self, documents: List[Any], 
                                 monitoring_data_str: str) -> str:
        """從檢索到的文檔生成摘要上下文"""
        if not documents:
            return f"未找到相關文檔。監控數據：\n{monitoring_data_str}"
        
        # 將文檔內容合併
        context_parts = []
        for i, doc in enumerate(documents[:5]):  # 限制最多5個文檔
            context_parts.append(f"文檔 {i+1}:\n{doc.page_content}")
        
        return "\n\n".join(context_parts)
    
    def _parse_report_sections(self, report_text: str) -> Dict[str, str]:
        """解析報告文本，提取不同部分"""
        # 使用分隔符分割報告
        parts = report_text.split("具體建議")
        
        insight = ""
        recommendations = ""
        
        if len(parts) >= 1:
            insight = parts[0].replace("洞見分析", "").strip()
        if len(parts) >= 2:
            recommendations = parts[1].strip()
        
        # 確保建議部分包含結構化的標籤
        if recommendations and not any(tag in recommendations for tag in ["[緊急處理]", "[中期優化]", "[永久措施]"]):
            # 如果沒有結構化標籤，嘗試從原文本中找到
            if "緊急處理" in report_text or "中期優化" in report_text:
                recommendations = report_text.split("具體建議")[-1].strip()
        
        return {
            "insight_analysis": insight,
            "recommendations": recommendations
        }
    
    @alru_cache(maxsize=100, ttl=3600)
    async def _get_cached_embedding(self, text: str) -> List[float]:
        """帶快取的嵌入向量生成"""
        try:
            embeddings = await model_manager.embedding_model.aembed_documents([text])
            return embeddings[0]
        except Exception as e:
            raise GeminiAPIError(f"Failed to generate embeddings: {str(e)}")
    
    @alru_cache(maxsize=50, ttl=1800)
    async def _get_cached_hyde(self, monitoring_data_str: str) -> str:
        """帶快取的 HyDE 生成"""
        try:
            return await self.hyde_chain.ainvoke({
                "monitoring_data": monitoring_data_str
            })
        except Exception as e:
            raise HyDEGenerationError(f"Failed to generate HyDE query: {str(e)}")
    
    async def generate_report(self, monitoring_data: Dict[str, Any]) -> InsightReport:
        """生成維運報告（主要介面）
        
        Args:
            monitoring_data: 監控數據字典
            
        Returns:
            InsightReport 物件
        """
        try:
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
        except Exception as e:
            if isinstance(e, (HyDEGenerationError, DocumentRetrievalError, GeminiAPIError)):
                raise
            raise ReportGenerationError(f"Failed to generate report: {str(e)}")
    
    async def generate_report_with_steps(self, monitoring_data: Dict[str, Any]) -> Dict[str, Any]:
        """生成報告並返回中間步驟（用於調試）
        
        Args:
            monitoring_data: 監控數據字典
            
        Returns:
            包含報告和中間步驟的字典
        """
        monitoring_data_str = json.dumps(monitoring_data, ensure_ascii=False, indent=2)
        
        # Step 1: 多查詢生成 (RAG-Fusion)
        multi_queries = []
        try:
            multi_queries = await self.multi_query_chain.ainvoke({
                "monitoring_data": monitoring_data_str
            })
        except Exception as e:
            multi_queries = ["Multi-query generation failed"]
        
        # Step 2: 文檔檢索 (優先使用多查詢)
        documents = []
        retrieval_method = "unknown"
        
        # 嘗試多查詢檢索
        if multi_queries and multi_queries[0] != "Multi-query generation failed":
            try:
                documents = await self._multi_query_retrieval(monitoring_data_str)
                retrieval_method = "multi-query"
            except Exception:
                pass
        
        # 如果多查詢失敗或結果不足，嘗試 HyDE
        if not documents or len(documents) < 3:
            try:
                hyde_query = await self._get_cached_hyde(monitoring_data_str)
                if hyde_query != "HyDE generation failed, using fallback":
                    documents = await vector_store_manager.similarity_search(hyde_query)
                    retrieval_method = "hyde"
            except Exception:
                pass
        
        # 最後的 fallback
        if not documents:
            try:
                documents = await vector_store_manager.similarity_search(monitoring_data_str)
                retrieval_method = "direct"
            except Exception as e:
                raise DocumentRetrievalError(f"Failed to retrieve documents: {str(e)}")
        
        # Step 3: 生成摘要
        context = self._generate_summary_context(documents, monitoring_data_str)
        
        try:
            # Step 4: 生成最終報告
            report_result = await self.report_chain.ainvoke({
                "monitoring_data": monitoring_data_str,
                "context": context
            })
        except Exception as e:
            raise ReportGenerationError(f"Failed to generate final report: {str(e)}")
        
        # 創建報告物件
        report = InsightReport(
            insight_analysis=report_result["insight_analysis"],
            recommendations=report_result["recommendations"],
            generated_at=datetime.now()
        )
        
        return {
            "report": report,
            "steps": {
                "multi_queries": multi_queries if isinstance(multi_queries, list) else [multi_queries],
                "retrieval_method": retrieval_method,
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
            raise PrometheusError(f"Failed to enrich with Prometheus data: {str(e)}")
        
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