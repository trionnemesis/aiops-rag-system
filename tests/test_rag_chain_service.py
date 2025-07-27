import pytest
from unittest.mock import Mock, patch
from langchain_core.runnables import RunnablePassthrough
from src.services.langchain.rag_chain_service import RAGChainService

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
