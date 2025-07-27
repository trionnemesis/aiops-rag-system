import pytest
from unittest.mock import Mock, AsyncMock, patch
from langchain_core.documents import Document
from src.services.langchain.rag_chain_service import RAGChainService
from src.services.exceptions import ReportGenerationError, DocumentRetrievalError
from src.models.schemas import InsightReport

class TestRAGChainService:
    """Test cases for RAGChainService"""

    @pytest.fixture
    def rag_service(self):
        """建立一個 RAGChainService 實例，並 mock 其內部鏈"""
        # patch 所有外部依賴
        with patch("src.services.langchain.rag_chain_service.PrometheusService"), \
             patch("src.services.langchain.rag_chain_service.model_manager"), \
             patch("src.services.langchain.rag_chain_service.prompt_manager"), \
             patch("src.services.langchain.rag_chain_service.vector_store_manager"):
            
            service = RAGChainService()
            # 初始化後，將鏈屬性替換為 Mock 物件
            service.hyde_chain = AsyncMock()
            service.retriever = AsyncMock()
            service.report_chain = AsyncMock()
            service.full_rag_chain = AsyncMock()
            yield service

    @pytest.mark.asyncio
    async def test_generate_report_success(self, rag_service):
        """測試成功的報告生成"""
        # 設定 mock 的 full_rag_chain 的回傳值
        rag_service.full_rag_chain.ainvoke.return_value = {
            "insight_analysis": "Test insight",
            "recommendations": "Test recommendations"
        }
        
        monitoring_data = {"host": "test-host"}
        report = await rag_service.generate_report(monitoring_data)
        
        assert isinstance(report, InsightReport)
        assert report.insight_analysis == "Test insight"
        # 驗證 mock 的鏈是否被呼叫
        rag_service.full_rag_chain.ainvoke.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_generate_report_failure(self, rag_service):
        """測試報告生成失敗"""
        rag_service.full_rag_chain.ainvoke.side_effect = Exception("Chain failed")
        
        with pytest.raises(ReportGenerationError, match="Failed to generate report: Chain failed"):
            await rag_service.generate_report({"host": "test-host"})
            
    def test_parse_report_sections_complete(self, rag_service):
        """測試報告文本解析功能"""
        # 這是私有方法，但測試它可以確保邏輯正確性
        report_text = "洞見分析\nAnalysis here\n建議與行動方案\nRecommendations here"
        parsed = rag_service._parse_report_sections(report_text)
        assert parsed["insight_analysis"] == "Analysis here"
        assert parsed["recommendations"] == "Recommendations here"

    def test_generate_summary_context_no_documents(self, rag_service):
        """測試沒有文件時的上下文生成"""
        context = rag_service._generate_summary_context([], '{"data": "empty"}')
        assert "未找到相關文檔" in context
        assert '{"data": "empty"}' in context

    def test_safe_retrieval_hyde_fails_fallback_succeeds(self, rag_service):
        """測試 HyDE 失敗後，fallback 檢索成功"""
        rag_service.hyde_chain.invoke = Mock(side_effect=Exception("HyDE error"))
        rag_service.retriever.invoke = Mock(return_value=[Document(page_content="fallback doc")])
        
        docs = rag_service._safe_retrieval({"monitoring_data_str": "some data"})
        
        assert len(docs) == 1
        assert docs[0].page_content == "fallback doc"
        rag_service.retriever.invoke.assert_called_with("some data")

    def test_safe_retrieval_all_fail(self, rag_service):
        """測試所有檢索方法都失敗的情況"""
        rag_service.hyde_chain.invoke = Mock(side_effect=Exception("HyDE error"))
        rag_service.retriever.invoke = Mock(side_effect=Exception("Retriever error"))
        
        with pytest.raises(DocumentRetrievalError):
            rag_service._safe_retrieval({"monitoring_data_str": "some data"})
    
    def test_generate_summary_context_with_documents(self, rag_service):
        """測試有文件時的上下文生成"""
        documents = [
            Document(page_content="First document content"),
            Document(page_content="Second document content"),
            Document(page_content="Third document content")
        ]
        monitoring_data_str = '{"host": "test-host"}'
        
        context = rag_service._generate_summary_context(documents, monitoring_data_str)
        
        assert "文檔 1:" in context
        assert "First document content" in context
        assert "文檔 2:" in context
        assert "Second document content" in context
        assert "文檔 3:" in context
        assert "Third document content" in context
    
    def test_generate_summary_context_max_five_documents(self, rag_service):
        """測試上下文生成最多只處理5個文檔"""
        documents = [Document(page_content=f"Document {i}") for i in range(10)]
        monitoring_data_str = '{"host": "test-host"}'
        
        context = rag_service._generate_summary_context(documents, monitoring_data_str)
        
        # 應該只包含前5個文檔
        assert "文檔 5:" in context
        assert "Document 4" in context
        assert "文檔 6:" not in context
        assert "Document 5" not in context
    
    def test_parse_report_sections_only_insight(self, rag_service):
        """測試只有洞見分析部分的報告解析"""
        report_text = "洞見分析\nOnly insight analysis here"
        parsed = rag_service._parse_report_sections(report_text)
        
        assert parsed["insight_analysis"] == "Only insight analysis here"
        assert parsed["recommendations"] == ""
    
    def test_parse_report_sections_empty(self, rag_service):
        """測試空報告文本的解析"""
        report_text = ""
        parsed = rag_service._parse_report_sections(report_text)
        
        assert parsed["insight_analysis"] == ""
        assert parsed["recommendations"] == ""
    
    @pytest.mark.asyncio
    async def test_generate_report_with_steps_success(self, rag_service):
        """測試帶步驟的報告生成（成功路徑）"""
        # Mock 必要的方法
        rag_service._get_cached_hyde = AsyncMock(return_value="Hypothetical document")
        
        with patch("src.services.langchain.rag_chain_service.vector_store_manager") as mock_vsm:
            mock_vsm.similarity_search = AsyncMock(return_value=[
                Document(page_content="Retrieved document")
            ])
            
            rag_service.report_chain.ainvoke.return_value = {
                "insight_analysis": "Test insight",
                "recommendations": "Test recommendations"
            }
            
            result = await rag_service.generate_report_with_steps({"host": "test-host"})
            
            assert "report" in result
            assert "steps" in result
            assert result["steps"]["hyde_query"] == "Hypothetical document"
            assert result["steps"]["documents_found"] == 1
            assert "Retrieved document" in result["steps"]["context_summary"]
    
    @pytest.mark.asyncio
    async def test_enrich_with_prometheus(self, rag_service):
        """測試使用 Prometheus 數據豐富監控資料"""
        # Mock PrometheusService
        rag_service.prometheus.get_host_metrics = AsyncMock(return_value={
            "cpu_usage": 85.5,
            "memory_usage": 70.2,
            "disk_usage": 60.0
        })
        
        original_data = {"host": "test-host", "existing_metric": 100}
        enriched = await rag_service.enrich_with_prometheus("test-host", original_data)
        
        # 檢查原始數據保留
        assert enriched["host"] == "test-host"
        assert enriched["existing_metric"] == 100
        
        # 檢查新增的 Prometheus 數據
        assert enriched["cpu_usage"] == 85.5
        assert enriched["memory_usage"] == 70.2
        assert enriched["disk_usage"] == 60.0
    
    def test_create_custom_chain_with_hyde(self, rag_service):
        """測試創建帶 HyDE 的自定義鏈"""
        with patch("src.services.langchain.rag_chain_service.prompt_manager") as mock_pm:
            mock_pm.get_prompt.return_value = Mock()
            
            chain = rag_service.create_custom_chain(hyde_enabled=True)
            
            assert chain is not None
            mock_pm.get_prompt.assert_called_with("rag_query")
    
    def test_create_custom_chain_without_hyde(self, rag_service):
        """測試創建不帶 HyDE 的自定義鏈"""
        with patch("src.services.langchain.rag_chain_service.prompt_manager") as mock_pm:
            mock_pm.get_prompt.return_value = Mock()
            
            chain = rag_service.create_custom_chain(hyde_enabled=False)
            
            assert chain is not None
            mock_pm.get_prompt.assert_called_with("rag_query")
    
    def test_clear_cache(self, rag_service):
        """測試清除快取"""
        # Mock cache clear methods
        rag_service._get_cached_embedding = Mock()
        rag_service._get_cached_embedding.cache_clear = Mock()
        rag_service._get_cached_hyde = Mock()
        rag_service._get_cached_hyde.cache_clear = Mock()
        
        rag_service.clear_cache()
        
        rag_service._get_cached_embedding.cache_clear.assert_called_once()
        rag_service._get_cached_hyde.cache_clear.assert_called_once()
    
    def test_get_cache_info(self, rag_service):
        """測試獲取快取資訊"""
        # Mock cache info
        mock_cache_info = Mock()
        mock_cache_info.hits = 10
        mock_cache_info.misses = 5
        mock_cache_info.maxsize = 100
        mock_cache_info.currsize = 15
        
        rag_service._get_cached_embedding = Mock()
        rag_service._get_cached_embedding.cache_info.return_value = mock_cache_info
        rag_service._get_cached_hyde = Mock()
        rag_service._get_cached_hyde.cache_info.return_value = mock_cache_info
        
        info = rag_service.get_cache_info()
        
        assert info["embedding_cache"]["hits"] == 10
        assert info["embedding_cache"]["misses"] == 5
        assert info["hyde_cache"]["maxsize"] == 100
        assert info["hyde_cache"]["currsize"] == 15
