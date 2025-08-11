"""
端到端的 LangGraph 流程測試

測試重點：
1. 驗證不同輸入是否能正確觸發 plan_node 的路由決策
2. 驗證 retrieve_node 是否能根據策略正確檢索文件
3. 驗證 synthesize_node 在不同場景下的行為
4. 驗證錯誤處理流程
"""

import pytest
from unittest.mock import Mock, MagicMock, patch, AsyncMock
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage
from app.graph.build import build_graph
from app.graph.state import RAGState
import asyncio


class TestGraphFlow:
    """端到端的 LangGraph 流程測試"""

    @pytest.fixture
    def mock_llm(self):
        """建立模擬的 LLM"""
        llm = Mock()
        # 模擬 invoke 方法
        llm.invoke = Mock(return_value=AIMessage(content="這是一個測試回答"))
        # 模擬 ainvoke 方法
        llm.ainvoke = AsyncMock(return_value=AIMessage(content="這是一個測試回答"))
        return llm

    @pytest.fixture
    def mock_retriever(self):
        """建立模擬的 retriever"""
        retriever = Mock()
        # 模擬 retrieve 方法
        retriever.invoke = Mock(return_value=[
            Document(
                page_content="測試文件內容 1",
                metadata={"id": "doc1", "title": "測試標題1"}
            ),
            Document(
                page_content="測試文件內容 2",
                metadata={"id": "doc2", "title": "測試標題2"}
            )
        ])
        retriever.ainvoke = AsyncMock(return_value=[
            Document(
                page_content="測試文件內容 1",
                metadata={"id": "doc1", "title": "測試標題1"}
            ),
            Document(
                page_content="測試文件內容 2",
                metadata={"id": "doc2", "title": "測試標題2"}
            )
        ])
        return retriever

    @pytest.fixture
    def mock_bm25_search(self):
        """建立模擬的 BM25 搜索函式"""
        def bm25_search(query, top_k=5):
            return [
                Document(
                    page_content="BM25 搜索結果 1",
                    metadata={"id": "bm25_1", "title": "BM25 標題1"}
                ),
                Document(
                    page_content="BM25 搜索結果 2",
                    metadata={"id": "bm25_2", "title": "BM25 標題2"}
                )
            ]
        return bm25_search

    @pytest.fixture
    def mock_extract_service(self):
        """建立模擬的 LangExtract 服務"""
        service = Mock()
        service.batch_extract = Mock(return_value=[
            Mock(
                raw_text="原始文本1",
                entities=Mock(dict=lambda: {"alerts": ["alert1"], "systems": ["system1"]}),
                confidence=0.95
            )
        ])
        service.extract_to_metadata = Mock(return_value={
            "alerts": ["alert1"],
            "systems": ["system1"],
            "confidence": 0.95
        })
        return service

    @pytest.fixture
    def graph_app(self, mock_llm, mock_retriever, mock_bm25_search):
        """建立測試用的 graph app"""
        return build_graph(
            llm=mock_llm,
            retriever=mock_retriever,
            bm25_search_fn=mock_bm25_search,
            policy={
                "use_hyde": False,
                "use_multi_query": False,
                "use_bm25": False,
                "top_k": 5
            }
        )

    @pytest.mark.asyncio
    async def test_fast_path_flow(self, graph_app, mock_llm, mock_retriever):
        """測試快速路徑流程：長查詢 -> fast route -> 基本檢索 -> 合成"""
        # 準備輸入狀態
        input_state = {
            "query": "這是一個相對較長的查詢，應該觸發快速路徑",
            "messages": [],
            "retrieved_docs": [],
            "context": "",
            "answer": "",
            "route": "",
            "queries": [],
            "metrics": {}
        }

        # 執行圖
        result = await graph_app.ainvoke(input_state)

        # 驗證流程
        assert result["route"] == "fast"  # 應該選擇快速路徑
        assert len(result["queries"]) == 1  # 只有原始查詢
        assert len(result["retrieved_docs"]) > 0  # 應該有檢索到的文件
        assert result["answer"] != ""  # 應該有生成的答案
        assert "error" not in result  # 不應該有錯誤

        # 驗證 retriever 被調用
        mock_retriever.ainvoke.assert_called()

    @pytest.mark.asyncio
    async def test_deep_path_with_hyde(self, mock_llm, mock_retriever, mock_bm25_search):
        """測試深度路徑流程：短查詢 + HyDE 策略"""
        # 建立啟用 HyDE 的圖
        graph_app = build_graph(
            llm=mock_llm,
            retriever=mock_retriever,
            bm25_search_fn=mock_bm25_search,
            policy={
                "use_hyde": True,
                "use_multi_query": False,
                "use_bm25": False,
                "top_k": 5
            }
        )

        # 模擬 HyDE 生成的假設性文件
        mock_llm.invoke.side_effect = [
            AIMessage(content="這是一個假設性的文件，詳細說明了錯誤的原因..."),
            AIMessage(content="基於檢索到的文件，錯誤的原因是...")
        ]

        input_state = {
            "query": "錯誤",  # 短查詢，應該觸發深度路徑
            "messages": [],
            "retrieved_docs": [],
            "context": "",
            "answer": "",
            "route": "",
            "queries": [],
            "metrics": {}
        }

        result = await graph_app.ainvoke(input_state)

        # 驗證流程
        assert result["route"] == "deep"  # 應該選擇深度路徑
        assert len(result["queries"]) > 1  # 應該有原始查詢 + HyDE 查詢
        assert result["answer"] != ""
        assert "error" not in result

    @pytest.mark.asyncio
    async def test_multi_query_expansion(self, mock_llm, mock_retriever, mock_bm25_search):
        """測試多查詢擴展功能"""
        # 建立啟用多查詢的圖
        graph_app = build_graph(
            llm=mock_llm,
            retriever=mock_retriever,
            bm25_search_fn=mock_bm25_search,
            policy={
                "use_hyde": False,
                "use_multi_query": True,
                "multi_query_alts": 2,
                "use_bm25": False,
                "top_k": 5
            }
        )

        # 模擬多查詢生成
        mock_llm.invoke.side_effect = [
            AIMessage(content="查詢變體1\n查詢變體2"),
            AIMessage(content="基於檢索到的文件，這是答案...")
        ]

        input_state = {
            "query": "系統異常",
            "messages": [],
            "retrieved_docs": [],
            "context": "",
            "answer": "",
            "route": "",
            "queries": [],
            "metrics": {}
        }

        result = await graph_app.ainvoke(input_state)

        # 驗證流程
        assert result["route"] == "deep"  # 多查詢應該走深度路徑
        assert len(result["queries"]) >= 3  # 原始 + 至少2個變體
        assert result["answer"] != ""

    @pytest.mark.asyncio
    async def test_bm25_rrf_fusion(self, mock_llm, mock_retriever, mock_bm25_search):
        """測試 BM25 + RRF 融合檢索"""
        # 建立啟用 BM25 的圖
        graph_app = build_graph(
            llm=mock_llm,
            retriever=mock_retriever,
            bm25_search_fn=mock_bm25_search,
            policy={
                "use_hyde": False,
                "use_multi_query": False,
                "use_bm25": True,
                "top_k": 5
            }
        )

        input_state = {
            "query": "測試 BM25 檢索",
            "messages": [],
            "retrieved_docs": [],
            "context": "",
            "answer": "",
            "route": "",
            "queries": [],
            "metrics": {}
        }

        result = await graph_app.ainvoke(input_state)

        # 驗證流程
        assert len(result["retrieved_docs"]) > 0
        # 應該包含來自兩個來源的文件
        doc_sources = set()
        for doc in result["retrieved_docs"]:
            if "bm25" in doc.metadata.get("id", ""):
                doc_sources.add("bm25")
            else:
                doc_sources.add("vector")
        assert len(doc_sources) == 2  # 應該有兩種來源

    @pytest.mark.asyncio
    async def test_error_handling_in_retrieve(self, mock_llm, mock_bm25_search):
        """測試檢索階段的錯誤處理"""
        # 建立會失敗的 retriever
        failing_retriever = Mock()
        failing_retriever.ainvoke = AsyncMock(side_effect=Exception("檢索失敗"))

        graph_app = build_graph(
            llm=mock_llm,
            retriever=failing_retriever,
            bm25_search_fn=mock_bm25_search,
            policy={
                "use_hyde": False,
                "use_multi_query": False,
                "use_bm25": False
            }
        )

        input_state = {
            "query": "測試錯誤處理",
            "messages": [],
            "retrieved_docs": [],
            "context": "",
            "answer": "",
            "route": "",
            "queries": [],
            "metrics": {}
        }

        result = await graph_app.ainvoke(input_state)

        # 驗證錯誤處理
        assert "error" in result
        assert "retrieval_error" in result["error"]
        # 即使檢索失敗，也應該有答案（來自 synthesize 的 fallback）
        assert result["answer"] != ""

    @pytest.mark.asyncio
    async def test_error_handling_in_plan(self, mock_retriever, mock_bm25_search):
        """測試規劃階段的錯誤處理"""
        # 建立會在 plan 階段失敗的 LLM
        failing_llm = Mock()
        failing_llm.invoke = Mock(side_effect=Exception("LLM 調用失敗"))
        failing_llm.ainvoke = AsyncMock(side_effect=Exception("LLM 調用失敗"))

        graph_app = build_graph(
            llm=failing_llm,
            retriever=mock_retriever,
            bm25_search_fn=mock_bm25_search,
            policy={
                "use_hyde": True,  # 強制使用 LLM
                "use_multi_query": False,
                "use_bm25": False
            }
        )

        input_state = {
            "query": "測試規劃錯誤",
            "messages": [],
            "retrieved_docs": [],
            "context": "",
            "answer": "",
            "route": "",
            "queries": [],
            "metrics": {}
        }

        result = await graph_app.ainvoke(input_state)

        # 驗證錯誤處理
        assert "error" in result
        assert "plan_error" in result["error"]

    @pytest.mark.asyncio
    async def test_extract_node_integration(self, mock_llm, mock_retriever, mock_bm25_search, mock_extract_service):
        """測試包含提取節點的完整流程"""
        # 建立包含提取服務的圖
        graph_app = build_graph(
            llm=mock_llm,
            retriever=mock_retriever,
            bm25_search_fn=mock_bm25_search,
            extract_service=mock_extract_service,
            policy={
                "use_llm_extract": True,
                "use_hyde": False,
                "use_multi_query": False,
                "use_bm25": False
            }
        )

        input_state = {
            "query": "分析系統告警",
            "raw_texts": ["系統出現告警：CPU 使用率過高"],
            "messages": [],
            "retrieved_docs": [],
            "context": "",
            "answer": "",
            "route": "",
            "queries": [],
            "metrics": {}
        }

        result = await graph_app.ainvoke(input_state)

        # 驗證流程
        assert "extracted_data" in result
        assert len(result["extracted_data"]) > 0
        assert "alerts" in result["extracted_data"][0]
        assert result["answer"] != ""

        # 驗證提取服務被調用
        mock_extract_service.batch_extract.assert_called_once()

    @pytest.mark.asyncio
    async def test_empty_retrieval_handling(self, mock_llm, mock_bm25_search):
        """測試無檢索結果時的處理"""
        # 建立返回空結果的 retriever
        empty_retriever = Mock()
        empty_retriever.ainvoke = AsyncMock(return_value=[])

        graph_app = build_graph(
            llm=mock_llm,
            retriever=empty_retriever,
            bm25_search_fn=lambda q, k: [],  # BM25 也返回空
            policy={
                "use_hyde": False,
                "use_multi_query": False,
                "use_bm25": True
            }
        )

        input_state = {
            "query": "完全不存在的內容",
            "messages": [],
            "retrieved_docs": [],
            "context": "",
            "answer": "",
            "route": "",
            "queries": [],
            "metrics": {}
        }

        result = await graph_app.ainvoke(input_state)

        # 驗證流程
        assert len(result["retrieved_docs"]) == 0
        assert result["context"] == ""
        # 即使沒有檢索結果，也應該有答案
        assert result["answer"] != ""
        assert "error" not in result

    @pytest.mark.asyncio
    async def test_conditional_routing_with_error(self, mock_llm, mock_retriever, mock_bm25_search, mock_extract_service):
        """測試條件路由中的錯誤處理"""
        # 模擬提取節點失敗
        mock_extract_service.batch_extract.side_effect = Exception("提取服務失敗")

        graph_app = build_graph(
            llm=mock_llm,
            retriever=mock_retriever,
            bm25_search_fn=mock_bm25_search,
            extract_service=mock_extract_service,
            policy={}
        )

        input_state = {
            "query": "測試提取錯誤",
            "raw_texts": ["需要提取的文本"],
            "messages": [],
            "retrieved_docs": [],
            "context": "",
            "answer": "",
            "route": "",
            "queries": [],
            "metrics": {}
        }

        result = await graph_app.ainvoke(input_state)

        # 驗證錯誤被正確處理
        assert "error" in result
        assert "extract_error" in result["error"]
        # 錯誤處理節點應該被執行
        assert result.get("extracted_data", []) == []

    @pytest.mark.asyncio
    async def test_metrics_tracking(self, graph_app):
        """測試指標追蹤功能"""
        input_state = {
            "query": "測試指標追蹤",
            "messages": [],
            "retrieved_docs": [],
            "context": "",
            "answer": "",
            "route": "",
            "queries": [],
            "metrics": {}
        }

        result = await graph_app.ainvoke(input_state)

        # 驗證指標被記錄
        assert "metrics" in result
        metrics = result["metrics"]
        assert "retrieved_count" in metrics
        assert metrics["retrieved_count"] > 0


class TestRetryAndErrorHandling:
    """測試重試機制和錯誤處理的整合"""
    
    @pytest.fixture
    def flaky_llm(self):
        """建立會隨機失敗的模擬 LLM"""
        llm = Mock()
        self.llm_call_count = 0
        
        def flaky_invoke(prompt, **kwargs):
            self.llm_call_count += 1
            if self.llm_call_count <= 2:  # 前兩次呼叫失敗
                raise ConnectionError("模擬 LLM API 連線錯誤")
            return AIMessage(content="第三次成功的回答")
        
        async def flaky_ainvoke(prompt, **kwargs):
            return flaky_invoke(prompt, **kwargs)
        
        llm.invoke = Mock(side_effect=flaky_invoke)
        llm.ainvoke = AsyncMock(side_effect=flaky_ainvoke)
        return llm
    
    @pytest.fixture
    def flaky_retriever(self):
        """建立會隨機失敗的模擬檢索器"""
        retriever = Mock()
        self.retriever_call_count = 0
        
        def flaky_retrieve(query):
            self.retriever_call_count += 1
            if self.retriever_call_count == 1:  # 第一次呼叫失敗
                raise TimeoutError("模擬向量資料庫超時")
            return [
                Document(
                    page_content="重試後成功檢索的文件",
                    metadata={"id": "retry-doc", "title": "重試文件"}
                )
            ]
        
        async def flaky_aretrieve(query):
            return flaky_retrieve(query)
        
        retriever.invoke = Mock(side_effect=flaky_retrieve)
        retriever.ainvoke = AsyncMock(side_effect=flaky_aretrieve)
        return retriever
    
    @pytest.fixture
    def always_fail_llm(self):
        """建立永遠失敗的模擬 LLM"""
        llm = Mock()
        
        def always_fail(*args, **kwargs):
            raise ConnectionError("永久性 LLM 連線錯誤")
        
        llm.invoke = Mock(side_effect=always_fail)
        llm.ainvoke = AsyncMock(side_effect=always_fail)
        return llm
    
    @pytest.fixture
    def always_fail_retriever(self):
        """建立永遠失敗的模擬檢索器"""
        retriever = Mock()
        
        def always_fail(*args, **kwargs):
            raise ConnectionError("永久性檢索器連線錯誤")
        
        retriever.invoke = Mock(side_effect=always_fail)
        retriever.ainvoke = AsyncMock(side_effect=always_fail)
        return retriever
    
    def test_llm_retry_mechanism_success(self, flaky_llm, mock_retriever):
        """測試 LLM 重試機制最終成功的情況"""
        # 重置計數器
        self.llm_call_count = 0
        
        # 建立圖形
        policy = {
            "use_rrf": False,
            "top_k": 5,
            "use_hyde": True  # 啟用 HyDE 以觸發 LLM 呼叫
        }
        
        app = build_graph(
            llm=flaky_llm,
            retriever=mock_retriever,
            policy=policy
        )
        
        # 執行查詢
        result = app.invoke({
            "query": "測試重試機制",
            "request_id": "test-retry-1"
        })
        
        # 驗證重試成功
        assert "error" not in result or result["error"] is None
        assert result["answer"] is not None
        assert "第三次成功的回答" in result["answer"]
        
        # 驗證 LLM 被呼叫了3次（2次失敗 + 1次成功）
        assert flaky_llm.invoke.call_count >= 3
    
    def test_retriever_retry_mechanism_success(self, mock_llm, flaky_retriever):
        """測試檢索器重試機制最終成功的情況"""
        # 重置計數器
        self.retriever_call_count = 0
        
        # 建立圖形
        policy = {
            "use_rrf": False,
            "top_k": 5,
            "use_hyde": False
        }
        
        app = build_graph(
            llm=mock_llm,
            retriever=flaky_retriever,
            policy=policy
        )
        
        # 執行查詢
        result = app.invoke({
            "query": "測試檢索重試",
            "request_id": "test-retry-2"
        })
        
        # 驗證重試成功
        assert "error" not in result or result["error"] is None
        assert result["answer"] is not None
        
        # 驗證檢索器被呼叫了2次（1次失敗 + 1次成功）
        assert flaky_retriever.invoke.call_count >= 2
        
        # 驗證使用了重試後的文件
        assert result.get("documents") is not None
        assert len(result["documents"]) > 0
        assert "重試後成功檢索的文件" in result["documents"][0].page_content
    
    def test_llm_retry_exhausted_triggers_error_handler(self, always_fail_llm, mock_retriever):
        """測試 LLM 重試次數耗盡後觸發錯誤處理節點"""
        # 建立圖形
        policy = {
            "use_rrf": False,
            "top_k": 5,
            "use_hyde": True  # 啟用 HyDE 以觸發 LLM 呼叫
        }
        
        app = build_graph(
            llm=always_fail_llm,
            retriever=mock_retriever,
            policy=policy
        )
        
        # 執行查詢
        result = app.invoke({
            "query": "測試 LLM 失敗",
            "request_id": "test-fail-1"
        })
        
        # 驗證進入錯誤處理流程
        assert result.get("error") is not None
        assert "plan_error" in result["error"] or "synthesize_error" in result["error"]
        
        # 驗證錯誤處理節點生成了適當的回應
        assert result["answer"] is not None
        assert "系統正在處理您的查詢" in result["answer"] or "系統正在生成回答時遇到問題" in result["answer"]
        
        # 驗證錯誤指標
        assert result.get("metrics", {}).get("error_handled") is True
        assert result.get("metrics", {}).get("error_type") is not None
    
    def test_retriever_retry_exhausted_triggers_error_handler(self, mock_llm, always_fail_retriever):
        """測試檢索器重試次數耗盡後觸發錯誤處理節點"""
        # 建立圖形
        policy = {
            "use_rrf": False,
            "top_k": 5,
            "use_hyde": False
        }
        
        app = build_graph(
            llm=mock_llm,
            retriever=always_fail_retriever,
            policy=policy
        )
        
        # 執行查詢
        result = app.invoke({
            "query": "測試檢索失敗",
            "request_id": "test-fail-2"
        })
        
        # 驗證進入錯誤處理流程
        assert result.get("error") is not None
        assert "retrieve_error" in result["error"]
        
        # 驗證錯誤處理節點生成了適當的回應
        assert result["answer"] is not None
        assert "系統無法存取知識庫" in result["answer"]
        
        # 驗證錯誤指標
        assert result.get("metrics", {}).get("error_handled") is True
        assert result.get("metrics", {}).get("error_type") == "retrieve_error"
    
    @patch('app.graph.nodes.retry')
    def test_custom_retry_configuration(self, mock_retry, mock_llm, mock_retriever):
        """測試自定義重試配置"""
        # 配置 mock_retry 來追蹤呼叫
        mock_retry.return_value = lambda func: func
        
        # 建立節點（這會觸發裝飾器）
        from app.graph import nodes
        
        # 驗證重試裝飾器被正確配置
        # 檢查 retry 被呼叫的參數
        retry_calls = mock_retry.call_args_list
        
        # 應該有多個重試配置（針對不同的節點）
        assert len(retry_calls) > 0
        
        # 檢查重試配置參數
        for call in retry_calls:
            kwargs = call[1]
            # 驗證 stop_after_attempt 參數
            if 'stop' in kwargs:
                # 確保設定了重試次數限制
                assert hasattr(kwargs['stop'], '__call__')
            # 驗證 wait_exponential 參數
            if 'wait' in kwargs:
                # 確保設定了指數退避
                assert hasattr(kwargs['wait'], '__call__')
            # 驗證 retry_if_exception_type 參數
            if 'retry' in kwargs:
                # 確保設定了特定的異常類型
                assert hasattr(kwargs['retry'], '__call__')
    
    def test_multiple_failures_cascade_handling(self, flaky_llm, flaky_retriever):
        """測試多個組件同時失敗時的級聯處理"""
        # 重置計數器
        self.llm_call_count = 0
        self.retriever_call_count = 0
        
        # 建立圖形
        policy = {
            "use_rrf": False,
            "top_k": 5,
            "use_hyde": True  # 啟用 HyDE
        }
        
        app = build_graph(
            llm=flaky_llm,
            retriever=flaky_retriever,
            policy=policy
        )
        
        # 執行查詢
        result = app.invoke({
            "query": "測試多重失敗",
            "request_id": "test-cascade-1"
        })
        
        # 驗證系統能夠處理多個組件的失敗和重試
        # 最終應該成功
        assert "error" not in result or result["error"] is None
        assert result["answer"] is not None
        
        # 驗證兩個組件都經歷了重試
        assert flaky_llm.invoke.call_count >= 3  # LLM 重試
        assert flaky_retriever.invoke.call_count >= 2  # 檢索器重試
    
    @pytest.mark.asyncio
    async def test_async_retry_mechanism(self, flaky_llm, mock_retriever):
        """測試非同步執行時的重試機制"""
        # 重置計數器
        self.llm_call_count = 0
        
        # 建立支持非同步的節點版本
        from app.graph.nodes import plan_node
        
        # 建立狀態
        state = RAGState(
            query="非同步測試查詢",
            request_id="async-test-1"
        )
        
        # 使用 flaky_llm 執行計劃節點
        with patch('app.graph.nodes.llm', flaky_llm):
            # 這應該會觸發重試機制
            result_state = await asyncio.to_thread(
                plan_node,
                state,
                llm=flaky_llm,
                policy={"use_hyde": True}
            )
        
        # 驗證非同步重試成功
        assert result_state.get("error") is None
        assert result_state.get("generated_query") is not None