#!/usr/bin/env python3
"""
測試重試機制和錯誤處理的示例腳本
"""
import asyncio
import logging
from typing import List
from langchain_core.documents import Document
from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.messages import AIMessage
from app.graph.build import build_graph

# 設置日誌
logging.basicConfig(level=logging.INFO)
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

def test_retry_mechanism():
    """測試重試機制"""
    logger.info("=== 測試重試機制 ===")
    
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
    
    logger.info(f"最終結果：")
    logger.info(f"  答案: {result.get('answer', 'N/A')[:100]}...")
    logger.info(f"  錯誤: {result.get('error', 'None')}")
    logger.info(f"  指標: {result.get('metrics', {})}")

def test_error_handling():
    """測試錯誤處理機制"""
    logger.info("\n=== 測試錯誤處理機制 ===")
    
    # 建立一個永遠失敗的檢索器
    class AlwaysFailRetriever:
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
    
    logger.info(f"錯誤處理結果：")
    logger.info(f"  答案: {result.get('answer', 'N/A')}")
    logger.info(f"  錯誤: {result.get('error', 'None')}")
    logger.info(f"  指標: {result.get('metrics', {})}")

if __name__ == "__main__":
    # 測試重試機制
    test_retry_mechanism()
    
    # 測試錯誤處理
    test_error_handling()
    
    logger.info("\n測試完成！")