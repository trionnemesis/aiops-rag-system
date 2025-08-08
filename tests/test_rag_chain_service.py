"""
節點單元測試

專注於測試 LangGraph 中各個節點的內部邏輯：
1. plan_node: 測試路由決策和查詢生成
2. retrieve_node: 測試檢索策略和 fallback 機制
3. synthesize_node: 測試答案生成和錯誤處理
4. validate_node: 測試驗證邏輯
5. extract_node: 測試結構化資訊提取
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from app.graph.nodes import (
    plan_node, retrieve_node, synthesize_node, validate_node, 
    extract_node, error_handler_node, _unique_by_id
)
from app.graph.state import RAGState


class TestNodeFunctions:
    """測試各個節點函式的單元測試"""

    def test_unique_by_id(self):
        """測試文件去重函式"""
        docs = [
            Document(page_content="內容1", metadata={"id": "doc1"}),
            Document(page_content="內容2", metadata={"id": "doc2"}),
            Document(page_content="內容1重複", metadata={"id": "doc1"}),  # 重複的 ID
            Document(page_content="內容3", metadata={"_id": "doc3"}),  # 使用 _id
            Document(page_content="內容4", metadata={}),  # 沒有 ID
        ]
        
        unique_docs = _unique_by_id(docs)
        
        # 應該只有4個唯一文件（doc1 的重複被移除）
        assert len(unique_docs) == 4
        
        # 檢查 ID
        ids = set()
        for doc in unique_docs:
            doc_id = doc.metadata.get("id") or doc.metadata.get("_id") or hash(doc.page_content)
            ids.add(doc_id)
        assert len(ids) == 4


class TestPlanNode:
    """測試 plan_node 的邏輯"""
    
    @pytest.fixture
    def mock_llm(self):
        """建立模擬的 LLM"""
        llm = Mock()
        llm.invoke = Mock(return_value=AIMessage(content="生成的內容"))
        return llm
    
    def test_plan_node_fast_route(self, mock_llm):
        """測試快速路徑判定：長查詢且無特殊策略"""
        state = {
            "query": "這是一個相對較長的查詢，不應該觸發深度路徑",
            "queries": []
        }
        policy = {
            "use_hyde": False,
            "use_multi_query": False
        }
        
        result = plan_node(state, llm=mock_llm, policy=policy)
        
        assert result["route"] == "fast"
        assert len(result["queries"]) == 1
        assert result["queries"][0] == state["query"]
        # LLM 不應該被調用（快速路徑不需要）
        mock_llm.invoke.assert_not_called()
    
    def test_plan_node_deep_route_short_query(self, mock_llm):
        """測試深度路徑判定：短查詢"""
        state = {
            "query": "錯誤",  # 短查詢
            "queries": []
        }
        policy = {
            "use_hyde": False,
            "use_multi_query": False
        }
        
        result = plan_node(state, llm=mock_llm, policy=policy)
        
        assert result["route"] == "deep"
        assert len(result["queries"]) == 1
    
    def test_plan_node_deep_route_ambiguous(self, mock_llm):
        """測試深度路徑判定：含模糊詞"""
        state = {
            "query": "為什麼系統會出現異常",
            "queries": []
        }
        policy = {
            "use_hyde": False,
            "use_multi_query": False
        }
        
        result = plan_node(state, llm=mock_llm, policy=policy)
        
        assert result["route"] == "deep"
    
    def test_plan_node_hyde_generation(self, mock_llm):
        """測試 HyDE 查詢生成"""
        mock_llm.invoke.return_value = AIMessage(
            content="這是一個假設性的文件，詳細說明了系統錯誤的原因..."
        )
        
        state = {
            "query": "系統錯誤原因",
            "queries": []
        }
        policy = {
            "use_hyde": True,
            "use_multi_query": False
        }
        
        result = plan_node(state, llm=mock_llm, policy=policy)
        
        assert result["route"] == "deep"
        assert len(result["queries"]) == 2  # 原始 + HyDE
        assert result["queries"][0] == state["query"]
        assert "假設性的文件" in result["queries"][1]
        mock_llm.invoke.assert_called_once()
    
    def test_plan_node_multi_query(self, mock_llm):
        """測試多查詢生成"""
        mock_llm.invoke.return_value = AIMessage(
            content="查詢變體1：系統故障分析\n查詢變體2：錯誤日誌查詢"
        )
        
        state = {
            "query": "系統錯誤",
            "queries": []
        }
        policy = {
            "use_hyde": False,
            "use_multi_query": True,
            "multi_query_alts": 2
        }
        
        result = plan_node(state, llm=mock_llm, policy=policy)
        
        assert result["route"] == "deep"
        assert len(result["queries"]) >= 3  # 原始 + 至少2個變體
        assert result["queries"][0] == state["query"]
        assert any("系統故障分析" in q for q in result["queries"])
    
    def test_plan_node_error_handling(self, mock_llm):
        """測試規劃錯誤處理"""
        mock_llm.invoke.side_effect = Exception("LLM 調用失敗")
        
        state = {
            "query": "測試錯誤",
            "queries": []
        }
        policy = {
            "use_hyde": True,  # 強制使用 LLM
            "use_multi_query": False
        }
        
        result = plan_node(state, llm=mock_llm, policy=policy)
        
        assert "error" in result
        assert "plan_error" in result["error"]
        # fallback 應該仍然設置基本的查詢
        assert len(result["queries"]) == 1
        assert result["queries"][0] == state["query"]


class TestRetrieveNode:
    """測試 retrieve_node 的邏輯"""
    
    @pytest.fixture
    def mock_retriever(self):
        """建立模擬的 retriever"""
        retriever = Mock()
        retriever.invoke = Mock(return_value=[
            Document(page_content="文件1", metadata={"id": "1"}),
            Document(page_content="文件2", metadata={"id": "2"})
        ])
        return retriever
    
    @pytest.fixture
    def mock_bm25_search(self):
        """建立模擬的 BM25 搜索"""
        def search(query, top_k=5):
            return [
                Document(page_content="BM25文件1", metadata={"id": "bm1"}),
                Document(page_content="BM25文件2", metadata={"id": "bm2"})
            ]
        return search
    
    def test_retrieve_node_basic(self, mock_retriever):
        """測試基本檢索"""
        state = {
            "queries": ["測試查詢"],
            "retrieved_docs": []
        }
        policy = {
            "use_bm25": False,
            "top_k": 5
        }
        
        result = retrieve_node(
            state, 
            retriever=mock_retriever,
            bm25_search_fn=None,
            rrf_fuse_fn=None,
            policy=policy
        )
        
        assert len(result["retrieved_docs"]) == 2
        assert result["metrics"]["retrieved_count"] == 2
        mock_retriever.invoke.assert_called_once_with("測試查詢")
    
    def test_retrieve_node_multiple_queries(self, mock_retriever):
        """測試多查詢檢索"""
        state = {
            "queries": ["查詢1", "查詢2", "查詢3"],
            "retrieved_docs": []
        }
        policy = {
            "use_bm25": False,
            "top_k": 5
        }
        
        # 為每個查詢返回不同的文件
        mock_retriever.invoke.side_effect = [
            [Document(page_content=f"查詢1文件{i}", metadata={"id": f"q1_{i}"}) for i in range(2)],
            [Document(page_content=f"查詢2文件{i}", metadata={"id": f"q2_{i}"}) for i in range(2)],
            [Document(page_content=f"查詢3文件{i}", metadata={"id": f"q3_{i}"}) for i in range(2)]
        ]
        
        result = retrieve_node(
            state,
            retriever=mock_retriever,
            bm25_search_fn=None,
            rrf_fuse_fn=None,
            policy=policy
        )
        
        # 應該有所有唯一文件
        assert len(result["retrieved_docs"]) == 6
        assert mock_retriever.invoke.call_count == 3
    
    def test_retrieve_node_with_bm25(self, mock_retriever, mock_bm25_search):
        """測試 BM25 + 向量檢索"""
        state = {
            "queries": ["測試查詢"],
            "retrieved_docs": []
        }
        policy = {
            "use_bm25": True,
            "top_k": 5
        }
        
        result = retrieve_node(
            state,
            retriever=mock_retriever,
            bm25_search_fn=mock_bm25_search,
            rrf_fuse_fn=None,  # 不使用 RRF
            policy=policy
        )
        
        # 應該包含兩種來源的文件
        assert len(result["retrieved_docs"]) == 4  # 2 向量 + 2 BM25
        doc_ids = {doc.metadata["id"] for doc in result["retrieved_docs"]}
        assert "1" in doc_ids  # 向量檢索
        assert "bm1" in doc_ids  # BM25 檢索
    
    def test_retrieve_node_error_handling(self, mock_bm25_search):
        """測試檢索錯誤處理"""
        # 建立會失敗的 retriever
        failing_retriever = Mock()
        failing_retriever.invoke.side_effect = Exception("檢索失敗")
        
        state = {
            "queries": ["測試查詢"],
            "retrieved_docs": []
        }
        policy = {
            "use_bm25": False,
            "top_k": 5
        }
        
        result = retrieve_node(
            state,
            retriever=failing_retriever,
            bm25_search_fn=mock_bm25_search,
            rrf_fuse_fn=None,
            policy=policy
        )
        
        assert "error" in result
        assert "retrieval_error" in result["error"]
        assert result["retrieved_docs"] == []
        assert result["metrics"]["retrieved_count"] == 0
    
    def test_retrieve_node_empty_results(self, mock_retriever):
        """測試空檢索結果"""
        mock_retriever.invoke.return_value = []
        
        state = {
            "queries": ["不存在的內容"],
            "retrieved_docs": []
        }
        policy = {
            "use_bm25": False,
            "top_k": 5
        }
        
        result = retrieve_node(
            state,
            retriever=mock_retriever,
            bm25_search_fn=None,
            rrf_fuse_fn=None,
            policy=policy
        )
        
        assert len(result["retrieved_docs"]) == 0
        assert result["metrics"]["retrieved_count"] == 0
        assert "error" not in result  # 空結果不是錯誤


class TestSynthesizeNode:
    """測試 synthesize_node 的邏輯"""
    
    @pytest.fixture
    def mock_llm(self):
        """建立模擬的 LLM"""
        llm = Mock()
        llm.invoke = Mock(return_value=AIMessage(content="這是基於文件的答案"))
        return llm
    
    @pytest.fixture
    def build_context_fn(self):
        """建立上下文構建函式"""
        def build_context(docs, max_chars=6000):
            return "\n".join([doc.page_content for doc in docs])
        return build_context
    
    def test_synthesize_with_documents(self, mock_llm, build_context_fn):
        """測試有文件時的合成"""
        state = {
            "query": "測試問題",
            "retrieved_docs": [
                Document(page_content="相關文件1"),
                Document(page_content="相關文件2")
            ],
            "context": "",
            "answer": ""
        }
        policy = {}
        
        result = synthesize_node(
            state,
            llm=mock_llm,
            build_context_fn=build_context_fn,
            policy=policy
        )
        
        assert result["context"] == "相關文件1\n相關文件2"
        assert result["answer"] == "這是基於文件的答案"
        mock_llm.invoke.assert_called_once()
    
    def test_synthesize_without_documents(self, mock_llm, build_context_fn):
        """測試無文件時的合成"""
        mock_llm.invoke.return_value = AIMessage(content="無法找到相關資訊的回答")
        
        state = {
            "query": "測試問題",
            "retrieved_docs": [],
            "context": "",
            "answer": ""
        }
        policy = {}
        
        result = synthesize_node(
            state,
            llm=mock_llm,
            build_context_fn=build_context_fn,
            policy=policy
        )
        
        assert result["context"] == ""
        assert result["answer"] == "無法找到相關資訊的回答"
        mock_llm.invoke.assert_called_once()
    
    def test_synthesize_error_handling(self, mock_llm, build_context_fn):
        """測試合成錯誤處理"""
        mock_llm.invoke.side_effect = Exception("LLM 錯誤")
        
        state = {
            "query": "測試問題",
            "retrieved_docs": [Document(page_content="文件")],
            "context": "",
            "answer": ""
        }
        policy = {}
        
        result = synthesize_node(
            state,
            llm=mock_llm,
            build_context_fn=build_context_fn,
            policy=policy
        )
        
        # 不應該有 error（synthesize 有內建 fallback）
        assert "error" not in result
        assert result["answer"] != ""  # 應該有 fallback 答案
        assert "抱歉" in result["answer"] or "錯誤" in result["answer"]
    
    def test_synthesize_context_truncation(self, mock_llm):
        """測試上下文截斷"""
        def build_context_with_limit(docs, max_chars=50):
            result = ""
            for doc in docs:
                if len(result) + len(doc.page_content) > max_chars:
                    break
                result += doc.page_content + "\n"
            return result.strip()
        
        state = {
            "query": "測試",
            "retrieved_docs": [
                Document(page_content="這是一個很長的文件內容" * 10),
                Document(page_content="這個不應該被包含")
            ],
            "context": "",
            "answer": ""
        }
        policy = {}
        
        result = synthesize_node(
            state,
            llm=mock_llm,
            build_context_fn=build_context_with_limit,
            policy=policy
        )
        
        # 確保上下文被截斷
        assert len(result["context"]) <= 50
        assert "這個不應該被包含" not in result["context"]


class TestValidateNode:
    """測試 validate_node 的邏輯"""
    
    def test_validate_node_success(self):
        """測試成功驗證"""
        state = {
            "answer": "這是一個完整的答案",
            "metrics": {"retrieved_count": 5}
        }
        policy = {
            "min_answer_length": 5
        }
        
        result = validate_node(state, policy=policy)
        
        assert result["metrics"]["validation_passed"] is True
        assert "error" not in result
    
    def test_validate_node_too_short(self):
        """測試答案過短"""
        state = {
            "answer": "短",
            "metrics": {}
        }
        policy = {
            "min_answer_length": 10
        }
        
        result = validate_node(state, policy=policy)
        
        assert result["metrics"]["validation_passed"] is False
        assert result["metrics"]["validation_reason"] == "answer_too_short"
    
    def test_validate_node_no_answer(self):
        """測試無答案"""
        state = {
            "answer": "",
            "metrics": {}
        }
        policy = {}
        
        result = validate_node(state, policy=policy)
        
        assert result["metrics"]["validation_passed"] is False
        assert result["metrics"]["validation_reason"] == "no_answer"
    
    def test_validate_node_with_error(self):
        """測試有錯誤時的驗證"""
        state = {
            "answer": "有答案但有錯誤",
            "error": "some_error",
            "metrics": {}
        }
        policy = {}
        
        result = validate_node(state, policy=policy)
        
        assert result["metrics"]["validation_passed"] is False
        assert result["metrics"]["validation_reason"] == "has_error"


class TestExtractNode:
    """測試 extract_node 的邏輯"""
    
    @pytest.fixture
    def mock_extract_service(self):
        """建立模擬的提取服務"""
        service = Mock()
        service.batch_extract = Mock(return_value=[
            Mock(
                raw_text="原始文本1",
                entities=Mock(dict=lambda: {
                    "alerts": ["CPU高", "記憶體不足"],
                    "systems": ["web-server", "database"],
                    "metrics": [{"name": "cpu_usage", "value": 95}]
                }),
                confidence=0.92
            ),
            Mock(
                raw_text="原始文本2",
                entities=Mock(dict=lambda: {
                    "alerts": ["磁碟滿"],
                    "systems": ["storage"],
                    "metrics": []
                }),
                confidence=0.88
            )
        ])
        service.extract_to_metadata = Mock(side_effect=lambda text, use_llm: {
            "alerts": ["CPU高"] if "文本1" in text else ["磁碟滿"],
            "systems": ["web-server"] if "文本1" in text else ["storage"],
            "confidence": 0.9
        })
        return service
    
    def test_extract_node_success(self, mock_extract_service):
        """測試成功提取"""
        state = {
            "query": "分析系統狀態",
            "raw_texts": ["原始文本1", "原始文本2"]
        }
        policy = {
            "use_llm_extract": True
        }
        
        result = extract_node(
            state,
            extract_service=mock_extract_service,
            policy=policy
        )
        
        assert "extracted_data" in result
        assert len(result["extracted_data"]) == 2
        assert result["metrics"]["extracted_count"] == 2
        
        # 驗證提取的數據結構
        first_extract = result["extracted_data"][0]
        assert "alerts" in first_extract
        assert "systems" in first_extract
        assert "_raw_extracted" in first_extract
        assert "_extraction_confidence" in first_extract
        
        mock_extract_service.batch_extract.assert_called_once_with(
            ["原始文本1", "原始文本2"],
            use_llm=True
        )
    
    def test_extract_node_no_texts(self, mock_extract_service):
        """測試無文本時的處理"""
        state = {
            "query": "測試",
            "raw_texts": []
        }
        policy = {}
        
        result = extract_node(
            state,
            extract_service=mock_extract_service,
            policy=policy
        )
        
        assert result["extracted_data"] == []
        mock_extract_service.batch_extract.assert_not_called()
    
    def test_extract_node_no_service(self):
        """測試無提取服務時的處理"""
        state = {
            "query": "測試",
            "raw_texts": ["一些文本"]
        }
        policy = {}
        
        result = extract_node(
            state,
            extract_service=None,
            policy=policy
        )
        
        assert result["extracted_data"] == []
    
    def test_extract_node_error_handling(self, mock_extract_service):
        """測試提取錯誤處理"""
        mock_extract_service.batch_extract.side_effect = Exception("提取服務失敗")
        
        state = {
            "query": "測試",
            "raw_texts": ["需要提取的文本"]
        }
        policy = {}
        
        result = extract_node(
            state,
            extract_service=mock_extract_service,
            policy=policy
        )
        
        assert "error" in result
        assert "extract_error" in result["error"]
        assert result["extracted_data"] == []
    
    def test_extract_node_retry_mechanism(self, mock_extract_service):
        """測試重試機制"""
        # 前兩次失敗，第三次成功
        mock_extract_service.batch_extract.side_effect = [
            ConnectionError("連接失敗"),
            TimeoutError("超時"),
            [Mock(
                raw_text="文本",
                entities=Mock(dict=lambda: {"alerts": ["測試"]}),
                confidence=0.9
            )]
        ]
        
        state = {
            "query": "測試",
            "raw_texts": ["文本"]
        }
        policy = {}
        
        result = extract_node(
            state,
            extract_service=mock_extract_service,
            policy=policy
        )
        
        # 應該成功（在第三次嘗試）
        assert "error" not in result
        assert len(result["extracted_data"]) == 1
        assert mock_extract_service.batch_extract.call_count == 3


class TestErrorHandlerNode:
    """測試 error_handler_node 的邏輯"""
    
    def test_error_handler_basic(self):
        """測試基本錯誤處理"""
        state = {
            "error": "test_error: Something went wrong",
            "answer": ""
        }
        policy = {}
        
        result = error_handler_node(state, policy=policy)
        
        assert result["answer"] != ""
        assert "抱歉" in result["answer"]
        assert result["metrics"]["error_handled"] is True
    
    def test_error_handler_with_existing_answer(self):
        """測試已有答案時的錯誤處理"""
        state = {
            "error": "minor_error",
            "answer": "已經有部分答案"
        }
        policy = {}
        
        result = error_handler_node(state, policy=policy)
        
        # 不應該覆蓋已有答案
        assert result["answer"] == "已經有部分答案"
        assert result["metrics"]["error_handled"] is True
    
    def test_error_handler_specific_errors(self):
        """測試特定錯誤類型的處理"""
        error_cases = [
            ("retrieval_error: 檢索失敗", "檢索"),
            ("plan_error: 規劃失敗", "處理"),
            ("extract_error: 提取失敗", "提取"),
        ]
        
        for error, expected_keyword in error_cases:
            state = {
                "error": error,
                "answer": ""
            }
            result = error_handler_node(state, policy={})
            
            assert result["answer"] != ""
            assert "抱歉" in result["answer"] or "錯誤" in result["answer"]
