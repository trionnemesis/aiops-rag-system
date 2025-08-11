"""
重試機制和錯誤處理的自動化測試
使用 pytest 框架替代原始的示例腳本
"""
import pytest
import logging
from typing import List
from unittest.mock import Mock, patch
from langchain_core.documents import Document
from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.messages import AIMessage
from app.graph.build import build_graph

logger = logging.getLogger(__name__)

# 模擬一個會隨機失敗的 LLM
class FlakyLLM(BaseLanguageModel):
    """模擬不穩定的 LLM，有 50% 機率失敗"""
    _call_count = 0
    
    def invoke(self, prompt, **kwargs):
        self._call_count += 1
        if self._call_count % 2 == 1:  # 第1、3、5...次調用會失敗
            logger.info(f"LLM call #{self._call_count} - 模擬失敗")
            raise ConnectionError("模擬 LLM API 連線錯誤")
        logger.info(f"LLM call #{self._call_count} - 成功")
        return AIMessage(content="這是一個測試回應")
    
    @property
    def _llm_type(self):
        return "flaky"

# 模擬一個會失敗的檢索器
class FlakyRetriever:
    """模擬不穩定的檢索器"""
    _call_count = 0
    
    def get_relevant_documents(self, query: str) -> List[Document]:
        self._call_count += 1
        if self._call_count <= 2:  # 前2次調用會失敗
            logger.info(f"Retriever call #{self._call_count} - 模擬失敗")
            raise TimeoutError("模擬向量資料庫超時")
        logger.info(f"Retriever call #{self._call_count} - 成功")
        return [
            Document(
                page_content=f"測試文檔內容 for query: {query}",
                metadata={"id": "test-doc-1", "title": "測試文檔"}
            )
        ]

class TestRetryMechanism:
    """測試重試機制的自動化測試類"""
    
    def test_llm_retry_mechanism(self):
        """測試 LLM 的重試機制"""
        # 建立圖形
        app = build_graph(
            llm=FlakyLLM(),
            retriever=FlakyRetriever(),
            policy={
                "use_hyde": True,  # 啟用 HyDE 以測試 LLM 重試
                "use_rrf": False,  # 關閉 RRF 簡化測試
                "top_k": 5,
            }
        )
        
        # 測試正常查詢（應該在重試後成功）
        result = app.invoke({
            "query": "測試查詢：系統異常原因分析"
        })
        
        # 驗證結果
        assert result.get('answer') is not None
        assert result.get('error') is None or result.get('error') == ''
        assert 'metrics' in result
        
        # 驗證確實有重試發生（透過 FlakyLLM 的呼叫計數）
        assert FlakyLLM._call_count > 1
    
    def test_retriever_retry_mechanism(self):
        """測試檢索器的重試機制"""
        # 重置 FlakyRetriever 的呼叫計數
        FlakyRetriever._call_count = 0
        
        # 建立圖形
        app = build_graph(
            llm=Mock(invoke=Mock(return_value=AIMessage(content="測試回答"))),
            retriever=FlakyRetriever(),
            policy={
                "use_hyde": False,
                "use_rrf": False,
                "top_k": 5,
            }
        )
        
        # 測試查詢
        result = app.invoke({
            "query": "測試檢索器重試"
        })
        
        # 驗證結果
        assert result.get('answer') is not None
        assert result.get('documents') is not None
        assert len(result.get('documents', [])) > 0
        
        # 驗證重試發生
        assert FlakyRetriever._call_count > 1


class TestErrorHandling:
    """測試錯誤處理機制的自動化測試類"""
    
    def test_permanent_retriever_failure(self):
        """測試檢索器永久失敗時的錯誤處理"""
        # 建立一個永遠失敗的檢索器
        class AlwaysFailRetriever:
            def invoke(self, query: str) -> List[Document]:
                raise ConnectionError("永久性連線錯誤")
            
            def get_relevant_documents(self, query: str) -> List[Document]:
                raise ConnectionError("永久性連線錯誤")
        
        # 建立圖形
        app = build_graph(
            llm=FlakyLLM(),
            retriever=AlwaysFailRetriever(),
            policy={
                "use_hyde": False,
                "use_rrf": False,
                "top_k": 5,
            }
        )
        
        # 測試會觸發錯誤處理的查詢
        result = app.invoke({
            "query": "測試錯誤處理"
        })
        
        # 驗證錯誤處理
        assert result.get('error') is not None
        assert 'retrieve_error' in result.get('error', '')
        assert result.get('answer') is not None
        assert '系統無法存取知識庫' in result.get('answer', '')
        
        # 驗證錯誤指標
        metrics = result.get('metrics', {})
        assert metrics.get('error_handled') is True
        assert metrics.get('error_type') == 'retrieve_error'
    
    def test_llm_permanent_failure(self):
        """測試 LLM 永久失敗時的錯誤處理"""
        # 建立永遠失敗的 LLM
        class AlwaysFailLLM(BaseLanguageModel):
            def invoke(self, prompt, **kwargs):
                raise ConnectionError("永久性 LLM 連線錯誤")
            
            @property
            def _llm_type(self):
                return "always_fail"
        
        # 建立正常的檢索器
        normal_retriever = Mock()
        normal_retriever.invoke = Mock(return_value=[
            Document(page_content="測試文件", metadata={"id": "test"})
        ])
        
        # 建立圖形
        app = build_graph(
            llm=AlwaysFailLLM(),
            retriever=normal_retriever,
            policy={
                "use_hyde": False,
                "use_rrf": False,
                "top_k": 5,
            }
        )
        
        # 測試查詢
        result = app.invoke({
            "query": "測試 LLM 失敗"
        })
        
        # 驗證錯誤處理
        assert result.get('error') is not None
        assert 'synthesize_error' in result.get('error', '')
        assert result.get('answer') is not None
        assert '系統正在生成回答時遇到問題' in result.get('answer', '')
    
    @patch('app.graph.nodes.logger')
    def test_error_logging(self, mock_logger):
        """測試錯誤是否被正確記錄"""
        # 建立會失敗的檢索器
        class FailRetriever:
            def invoke(self, query: str) -> List[Document]:
                raise ValueError("測試錯誤")
            
            def get_relevant_documents(self, query: str) -> List[Document]:
                raise ValueError("測試錯誤")
        
        # 建立圖形
        app = build_graph(
            llm=Mock(),
            retriever=FailRetriever(),
            policy={"use_hyde": False, "use_rrf": False, "top_k": 5}
        )
        
        # 執行查詢
        result = app.invoke({
            "query": "測試錯誤記錄",
            "request_id": "test-error-log"
        })
        
        # 驗證錯誤被記錄
        assert mock_logger.error.called
        
        # 檢查錯誤日誌內容
        error_calls = [call for call in mock_logger.error.call_args_list]
        assert len(error_calls) > 0
        
        # 驗證包含正確的錯誤資訊
        error_logged = False
        for call in error_calls:
            args, kwargs = call
            if 'retrieve_error' in str(args) or 'retrieve_error' in str(kwargs):
                error_logged = True
                break
        assert error_logged