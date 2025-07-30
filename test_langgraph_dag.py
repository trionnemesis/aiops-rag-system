"""
測試 LangGraph DAG 的簡單腳本
可以在本地執行來驗證整合是否正確
"""

from langchain_core.documents import Document
from langchain_core.language_models import BaseLanguageModel
from typing import List
import json

# 模擬 LLM（測試用）
class MockLLM(BaseLanguageModel):
    def invoke(self, prompt: str, **kwargs):
        # 模擬不同類型的回應
        if "產生一段可能的說明" in prompt:
            return type('obj', (object,), {'content': '這是一個假設性的文檔內容，用於 HyDE 測試。'})()
        elif "產生" in prompt and "等義查詢" in prompt:
            return type('obj', (object,), {'content': '- 查詢變體 1\n- 查詢變體 2'})()
        else:
            return type('obj', (object,), {'content': '這是基於檢索結果的回答。[來源1] [來源2]'})()
    
    def _generate(self, *args, **kwargs):
        pass
    
    def _llm_type(self):
        return "mock"

# 模擬 Retriever（測試用）
class MockRetriever:
    def get_relevant_documents(self, query: str) -> List[Document]:
        # 回傳模擬文檔
        return [
            Document(
                page_content=f"關於 {query} 的文檔內容 1",
                metadata={"id": "doc1", "title": "文檔標題 1"}
            ),
            Document(
                page_content=f"關於 {query} 的文檔內容 2",
                metadata={"id": "doc2", "title": "文檔標題 2"}
            ),
            Document(
                page_content=f"關於 {query} 的文檔內容 3",
                metadata={"id": "doc3", "title": "文檔標題 3"}
            ),
        ]

def test_langgraph_dag():
    # 導入 graph 模組
    from app.graph.build import build_graph
    
    # 建立測試用的元件
    llm = MockLLM()
    retriever = MockRetriever()
    
    # 定義測試 policy
    policy = {
        "use_hyde": True,
        "use_multi_query": True,
        "multi_query_alts": 2,
        "use_rrf": False,
        "top_k": 5,
        "max_ctx_chars": 2000,
        "strict_citation": True,
        "fallback_text": "（系統暫時無法提供完整回答）",
        "min_docs": 2,
        "min_answer_len": 20,
    }
    
    # 建立 graph
    graph_app = build_graph(
        llm=llm,
        retriever=retriever,
        bm25_search_fn=None,
        policy=policy
    )
    
    # 測試案例
    test_cases = [
        "為什麼系統異常",  # 應該觸發 deep 路徑
        "如何配置 OpenSearch 向量檢索",  # 正常查詢
        "不穩定",  # 短查詢，應該觸發 deep 路徑
    ]
    
    for query in test_cases:
        print(f"\n{'='*60}")
        print(f"測試查詢: {query}")
        print(f"{'='*60}")
        
        # 執行 graph
        result = graph_app.invoke({"query": query})
        
        # 顯示結果
        print(f"\n路由決策: {result.get('route', 'N/A')}")
        print(f"生成的查詢數: {len(result.get('queries', []))}")
        print(f"查詢列表: {result.get('queries', [])}")
        print(f"檢索到的文檔數: {len(result.get('docs', []))}")
        print(f"\n最終答案:\n{result.get('answer', 'N/A')}")
        print(f"\n指標: {json.dumps(result.get('metrics', {}), indent=2, ensure_ascii=False)}")
        
        # 檢查警告
        warnings = result.get('metrics', {}).get('warnings', [])
        if warnings:
            print(f"警告: {warnings}")

if __name__ == "__main__":
    # 確保路徑正確
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    
    test_langgraph_dag()