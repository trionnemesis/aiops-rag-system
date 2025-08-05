"""
測試狀態持久化功能
"""

import os
import uuid
import pytest
import redis
from unittest.mock import Mock, patch
from langchain_core.documents import Document
from app.graph.build import build_graph
from app.graph.state import RAGState


class TestStatePersistence:
    """測試狀態持久化和恢復功能"""
    
    @pytest.fixture
    def mock_llm(self):
        """模擬 LLM"""
        llm = Mock()
        llm.invoke.return_value = Mock(content="Test response")
        return llm
    
    @pytest.fixture
    def mock_retriever(self):
        """模擬檢索器"""
        retriever = Mock()
        retriever.invoke.return_value = [
            Document(
                page_content="Test document 1",
                metadata={"id": "1", "title": "Doc 1"}
            ),
            Document(
                page_content="Test document 2", 
                metadata={"id": "2", "title": "Doc 2"}
            )
        ]
        return retriever
    
    @pytest.fixture
    def redis_client(self):
        """獲取 Redis 客戶端"""
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
        try:
            client = redis.from_url(redis_url)
            client.ping()
            return client
        except:
            pytest.skip("Redis not available")
    
    def test_redis_checkpoint_creation(self, mock_llm, mock_retriever):
        """測試 Redis checkpoint 的創建"""
        # 設置環境變數
        os.environ["REDIS_URL"] = "redis://localhost:6379"
        
        # 建構 graph
        app = build_graph(
            llm=mock_llm,
            retriever=mock_retriever,
            policy={"max_retries": 1}
        )
        
        # 確認使用了 RedisSaver
        assert app.checkpointer is not None
        assert "RedisSaver" in str(type(app.checkpointer))
    
    def test_state_persistence_and_recovery(self, mock_llm, mock_retriever, redis_client):
        """測試狀態持久化和恢復"""
        thread_id = f"test-{uuid.uuid4()}"
        
        # 建構 graph
        app = build_graph(
            llm=mock_llm,
            retriever=mock_retriever,
            policy={"max_retries": 1}
        )
        
        # 第一次執行
        initial_state = {
            "query": "Test query",
            "request_id": str(uuid.uuid4())
        }
        
        result1 = app.invoke(
            initial_state,
            config={"configurable": {"thread_id": thread_id}}
        )
        
        # 驗證結果
        assert "answer" in result1
        assert "docs" in result1
        
        # 檢查 Redis 中是否有 checkpoint
        checkpoint_keys = list(redis_client.scan_iter(f"*{thread_id}*"))
        assert len(checkpoint_keys) > 0
        
        # 使用相同的 thread_id 再次執行（應該從 checkpoint 恢復）
        resume_state = {
            "query": "Test query",
            "request_id": str(uuid.uuid4()),
            "_resume": True
        }
        
        result2 = app.invoke(
            resume_state,
            config={"configurable": {"thread_id": thread_id}}
        )
        
        # 結果應該相似（因為使用了保存的狀態）
        assert result2.get("docs") == result1.get("docs")
    
    def test_different_thread_ids(self, mock_llm, mock_retriever):
        """測試不同的 thread_id 產生獨立的執行"""
        app = build_graph(
            llm=mock_llm,
            retriever=mock_retriever,
            policy={"max_retries": 1}
        )
        
        thread_id1 = f"test-1-{uuid.uuid4()}"
        thread_id2 = f"test-2-{uuid.uuid4()}"
        
        # 兩個不同的查詢
        state1 = {"query": "Query 1", "request_id": str(uuid.uuid4())}
        state2 = {"query": "Query 2", "request_id": str(uuid.uuid4())}
        
        # 執行兩個獨立的流程
        result1 = app.invoke(
            state1,
            config={"configurable": {"thread_id": thread_id1}}
        )
        
        result2 = app.invoke(
            state2,
            config={"configurable": {"thread_id": thread_id2}}
        )
        
        # 應該是獨立的執行
        assert result1["query"] == "Query 1"
        assert result2["query"] == "Query 2"
    
    def test_checkpoint_with_error(self, mock_llm, mock_retriever):
        """測試錯誤情況下的 checkpoint"""
        # 讓 synthesize 步驟失敗
        mock_llm.invoke.side_effect = Exception("Simulated error")
        
        app = build_graph(
            llm=mock_llm,
            retriever=mock_retriever,
            policy={"max_retries": 1}
        )
        
        thread_id = f"test-error-{uuid.uuid4()}"
        state = {"query": "Test query", "request_id": str(uuid.uuid4())}
        
        # 執行應該失敗但狀態應該被保存
        with pytest.raises(Exception):
            app.invoke(
                state,
                config={"configurable": {"thread_id": thread_id}}
            )
        
        # 修復錯誤後恢復
        mock_llm.invoke.side_effect = None
        mock_llm.invoke.return_value = Mock(content="Recovered response")
        
        # 從 checkpoint 恢復
        result = app.invoke(
            {"query": "Test query", "_resume": True},
            config={"configurable": {"thread_id": thread_id}}
        )
        
        # 應該成功完成
        assert "answer" in result
    
    def test_fallback_to_memory_saver(self, mock_llm, mock_retriever):
        """測試當 Redis 不可用時回退到 MemorySaver"""
        # 移除 REDIS_URL
        original_redis_url = os.environ.pop("REDIS_URL", None)
        
        try:
            app = build_graph(
                llm=mock_llm,
                retriever=mock_retriever,
                policy={"max_retries": 1}
            )
            
            # 應該使用 MemorySaver
            assert "MemorySaver" in str(type(app.checkpointer))
            
            # 仍然可以正常執行
            result = app.invoke(
                {"query": "Test query"},
                config={"configurable": {"thread_id": "test"}}
            )
            assert "answer" in result
            
        finally:
            # 恢復環境變數
            if original_redis_url:
                os.environ["REDIS_URL"] = original_redis_url


if __name__ == "__main__":
    pytest.main([__file__, "-v"])