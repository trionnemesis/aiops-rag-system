import pytest
from unittest.mock import Mock, AsyncMock, patch
from src.services.langchain.rag_chain_service import RAGChainService
from src.services.exceptions import ReportGenerationError
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
