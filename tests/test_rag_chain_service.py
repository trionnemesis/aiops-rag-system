import pytest
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from langchain_core.runnables import RunnablePassthrough
from datetime import datetime
from src.services.langchain.rag_chain_service import RAGChainService
from src.services.exceptions import HyDEGenerationError, DocumentRetrievalError, ReportGenerationError

class TestRAGChainService:
    @pytest.fixture
    def rag_service(self):
        # 完整地 mock 掉所有外部依賴
        with patch("src.services.langchain.rag_chain_service.PrometheusService"), \
             patch("src.services.langchain.rag_chain_service.model_manager") as mock_mm, \
             patch("src.services.langchain.rag_chain_service.prompt_manager") as mock_pm, \
             patch("src.services.langchain.rag_chain_service.vector_store_manager"):
            
            # 為了測試 create_custom_chain，需要讓 mock 物件支援 `|`
            mock_mm.pro_model = RunnablePassthrough()
            mock_pm.get_prompt.return_value = RunnablePassthrough()

            service = RAGChainService()
            # 替換掉會執行複雜邏輯的鏈
            service.hyde_chain = Mock()
            service.full_rag_chain = Mock()
            yield service
    
    # 關鍵修正：讓 retriever 的 mock 物件相容於 LCEL
    def test_create_custom_chain_with_hyde(self, rag_service):
        rag_service.retriever = RunnablePassthrough() # 使用 Runnable
        rag_service.hyde_chain = RunnablePassthrough() # 使用 Runnable
        chain = rag_service.create_custom_chain(hyde_enabled=True)
        assert chain is not None

    def test_create_custom_chain_without_hyde(self, rag_service):
        rag_service.retriever = RunnablePassthrough() # 使用 Runnable
        chain = rag_service.create_custom_chain(hyde_enabled=False)
        assert chain is not None

    def test_create_custom_chain_with_retriever_kwargs(self, rag_service):
        """測試使用自定義 retriever 參數"""
        with patch("src.services.langchain.rag_chain_service.vector_store_manager") as mock_vsm:
            mock_vsm.as_retriever.return_value = RunnablePassthrough()
            
            retriever_kwargs = {"search_kwargs": {"k": 5}}
            chain = rag_service.create_custom_chain(retriever_kwargs=retriever_kwargs, hyde_enabled=False)
            
            assert chain is not None
            mock_vsm.as_retriever.assert_called_once_with(**retriever_kwargs)

    def test_clear_cache(self, rag_service):
        """測試清除快取"""
        # 設置 mock
        rag_service._get_cached_embedding = Mock()
        rag_service._get_cached_embedding.cache_clear = Mock()
        rag_service._get_cached_hyde = Mock()
        rag_service._get_cached_hyde.cache_clear = Mock()
        
        rag_service.clear_cache()
        
        rag_service._get_cached_embedding.cache_clear.assert_called_once()
        rag_service._get_cached_hyde.cache_clear.assert_called_once()

    def test_get_cache_info(self, rag_service):
        """測試獲取快取資訊"""
        # 設置 mock
        cache_info_mock = Mock(hits=10, misses=5, maxsize=100, currsize=15)
        
        rag_service._get_cached_embedding = Mock()
        rag_service._get_cached_embedding.cache_info.return_value = cache_info_mock
        rag_service._get_cached_hyde = Mock()
        rag_service._get_cached_hyde.cache_info.return_value = cache_info_mock
        
        info = rag_service.get_cache_info()
        
        assert info["embedding_cache"]["hits"] == 10
        assert info["embedding_cache"]["misses"] == 5
        assert info["embedding_cache"]["maxsize"] == 100
        assert info["embedding_cache"]["currsize"] == 15
        assert info["hyde_cache"]["hits"] == 10
        assert info["hyde_cache"]["misses"] == 5

    @pytest.mark.asyncio
    async def test_generate_report_success(self, rag_service):
        """測試成功生成報告"""
        monitoring_data = {"cpu": 50, "memory": 80}
        
        # Mock full_rag_chain
        rag_service.full_rag_chain = AsyncMock()
        rag_service.full_rag_chain.ainvoke.return_value = {
            "insight_analysis": "系統資源使用正常",
            "recommendations": "建議持續監控"
        }
        
        report = await rag_service.generate_report(monitoring_data)
        
        assert report.insight_analysis == "系統資源使用正常"
        assert report.recommendations == "建議持續監控"
        assert isinstance(report.generated_at, datetime)
        
        rag_service.full_rag_chain.ainvoke.assert_called_once_with({
            "monitoring_data": monitoring_data
        })

    @pytest.mark.asyncio
    async def test_generate_report_error(self, rag_service):
        """測試生成報告時的錯誤處理"""
        monitoring_data = {"cpu": 50, "memory": 80}
        
        # Mock full_rag_chain 拋出錯誤
        rag_service.full_rag_chain = AsyncMock()
        rag_service.full_rag_chain.ainvoke.side_effect = Exception("Chain error")
        
        with pytest.raises(ReportGenerationError, match="Failed to generate report"):
            await rag_service.generate_report(monitoring_data)

    @pytest.mark.asyncio
    async def test_generate_report_with_steps_success(self, rag_service):
        """測試帶步驟的報告生成"""
        monitoring_data = {"cpu": 50, "memory": 80}
        
        # Mock 各個步驟
        rag_service._get_cached_hyde = AsyncMock(return_value="hyde query")
        
        with patch("src.services.langchain.rag_chain_service.vector_store_manager") as mock_vsm:
            mock_vsm.similarity_search = AsyncMock(return_value=[
                Mock(page_content="文檔內容1"),
                Mock(page_content="文檔內容2")
            ])
            
            rag_service.report_chain = AsyncMock()
            rag_service.report_chain.ainvoke.return_value = {
                "insight_analysis": "分析結果",
                "recommendations": "建議"
            }
            
            result = await rag_service.generate_report_with_steps(monitoring_data)
            
            assert result["report"].insight_analysis == "分析結果"
            assert result["report"].recommendations == "建議"
            assert result["steps"]["hyde_query"] == "hyde query"
            assert result["steps"]["documents_found"] == 2
            assert "文檔內容1" in result["steps"]["context_summary"]

    @pytest.mark.asyncio
    async def test_generate_report_with_steps_hyde_failure(self, rag_service):
        """測試 HyDE 失敗時的備用方案"""
        monitoring_data = {"cpu": 50, "memory": 80}
        
        # Mock HyDE 失敗
        rag_service._get_cached_hyde = AsyncMock(side_effect=Exception("HyDE error"))
        
        with patch("src.services.langchain.rag_chain_service.vector_store_manager") as mock_vsm:
            mock_vsm.similarity_search = AsyncMock(return_value=[
                Mock(page_content="文檔內容")
            ])
            
            rag_service.report_chain = AsyncMock()
            rag_service.report_chain.ainvoke.return_value = {
                "insight_analysis": "分析結果",
                "recommendations": "建議"
            }
            
            result = await rag_service.generate_report_with_steps(monitoring_data)
            
            assert result["steps"]["hyde_query"] == "HyDE generation failed, using fallback"
            # 確認使用了原始監控數據進行搜索
            mock_vsm.similarity_search.assert_called_with('{\n  "cpu": 50,\n  "memory": 80\n}')

    @pytest.mark.asyncio
    async def test_enrich_with_prometheus_success(self, rag_service):
        """測試成功使用 Prometheus 數據豐富監控資料"""
        hostname = "test-host"
        monitoring_data = {"cpu": 50}
        
        # Mock Prometheus service
        rag_service.prometheus = AsyncMock()
        rag_service.prometheus.get_host_metrics.return_value = {
            "memory": 80,
            "disk": 60
        }
        
        enriched_data = await rag_service.enrich_with_prometheus(hostname, monitoring_data)
        
        assert enriched_data["cpu"] == 50  # 原始數據
        assert enriched_data["memory"] == 80  # 新增數據
        assert enriched_data["disk"] == 60  # 新增數據

    @pytest.mark.asyncio
    async def test_enrich_with_prometheus_error(self, rag_service):
        """測試 Prometheus 豐富數據時的錯誤處理"""
        hostname = "test-host"
        monitoring_data = {"cpu": 50}
        
        # Mock Prometheus service 拋出錯誤
        rag_service.prometheus = AsyncMock()
        rag_service.prometheus.get_host_metrics.side_effect = Exception("Prometheus error")
        
        from src.services.exceptions import PrometheusError
        with pytest.raises(PrometheusError, match="Failed to enrich with Prometheus data"):
            await rag_service.enrich_with_prometheus(hostname, monitoring_data)

    def test_generate_summary_context_with_documents(self, rag_service):
        """測試有文檔時的摘要生成"""
        documents = [
            Mock(page_content="文檔1內容"),
            Mock(page_content="文檔2內容"),
            Mock(page_content="文檔3內容"),
            Mock(page_content="文檔4內容"),
            Mock(page_content="文檔5內容"),
            Mock(page_content="文檔6內容"),  # 第6個文檔應該被忽略
        ]
        monitoring_data_str = "監控數據"
        
        context = rag_service._generate_summary_context(documents, monitoring_data_str)
        
        # 確認只包含前5個文檔
        assert "文檔1內容" in context
        assert "文檔2內容" in context
        assert "文檔3內容" in context
        assert "文檔4內容" in context
        assert "文檔5內容" in context
        assert "文檔6內容" not in context
        assert "文檔 1:" in context
        assert "文檔 5:" in context

    def test_generate_summary_context_without_documents(self, rag_service):
        """測試無文檔時的摘要生成"""
        documents = []
        monitoring_data_str = "監控數據"
        
        context = rag_service._generate_summary_context(documents, monitoring_data_str)
        
        assert "未找到相關文檔" in context
        assert monitoring_data_str in context

    def test_parse_report_sections(self, rag_service):
        """測試報告部分的解析"""
        report_text = "洞見分析\n這是洞見內容\n建議與行動方案\n這是建議內容"
        
        result = rag_service._parse_report_sections(report_text)
        
        assert result["insight_analysis"] == "這是洞見內容"
        assert result["recommendations"] == "這是建議內容"

    def test_parse_report_sections_missing_parts(self, rag_service):
        """測試缺少部分時的報告解析"""
        report_text = "洞見分析\n只有洞見內容"
        
        result = rag_service._parse_report_sections(report_text)
        
        assert result["insight_analysis"] == "只有洞見內容"
        assert result["recommendations"] == ""  # 缺少建議部分

    def test_safe_retrieval_with_hyde_success(self, rag_service):
        """測試安全檢索 - HyDE 成功"""
        test_data = {"monitoring_data_str": "test data"}
        
        # Mock HyDE chain
        rag_service.hyde_chain = Mock()
        rag_service.hyde_chain.invoke.return_value = "hyde query"
        
        # Mock retriever
        rag_service.retriever = Mock()
        rag_service.retriever.invoke.return_value = ["doc1", "doc2"]
        
        result = rag_service._safe_retrieval(test_data)
        
        assert result == ["doc1", "doc2"]
        rag_service.hyde_chain.invoke.assert_called_once()
        rag_service.retriever.invoke.assert_called_once_with("hyde query")

    def test_safe_retrieval_hyde_fails_fallback_success(self, rag_service):
        """測試安全檢索 - HyDE 失敗但 fallback 成功"""
        test_data = {"monitoring_data_str": "test data"}
        
        # Mock HyDE chain 失敗
        rag_service.hyde_chain = Mock()
        rag_service.hyde_chain.invoke.side_effect = Exception("HyDE error")
        
        # Mock retriever
        rag_service.retriever = Mock()
        rag_service.retriever.invoke.return_value = ["fallback_doc"]
        
        with patch("logging.warning") as mock_warning:
            result = rag_service._safe_retrieval(test_data)
            
            assert result == ["fallback_doc"]
            mock_warning.assert_called_once()
            # 確認使用原始數據進行檢索
            rag_service.retriever.invoke.assert_called_once_with("test data")

    def test_safe_retrieval_all_fail(self, rag_service):
        """測試安全檢索 - 所有方法都失敗"""
        test_data = {"monitoring_data_str": "test data"}
        
        # Mock 所有檢索都失敗
        rag_service.hyde_chain = Mock()
        rag_service.hyde_chain.invoke.side_effect = Exception("HyDE error")
        
        rag_service.retriever = Mock()
        rag_service.retriever.invoke.side_effect = Exception("Retriever error")
        
        with pytest.raises(DocumentRetrievalError, match="All retrieval methods failed"):
            rag_service._safe_retrieval(test_data)
