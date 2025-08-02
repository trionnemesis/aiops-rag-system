"""
測試 LangExtract 與 LangGraph 整合
"""
import pytest
from datetime import datetime
from unittest.mock import Mock, MagicMock, patch
from langchain_core.documents import Document
from langchain_core.language_models import BaseLanguageModel

from src.services.langchain.langextract_service import (
    LangExtractService, AIOpsEntity, ExtractedData
)
from src.services.langchain.chunking_service import (
    ChunkingService, EmbeddingService, ChunkingAndEmbeddingPipeline
)
from app.graph.build import build_graph
from app.graph.state import RAGState


class TestLangExtractService:
    """測試 LangExtract 服務"""
    
    @pytest.fixture
    def mock_llm(self):
        """模擬 LLM"""
        llm = Mock(spec=BaseLanguageModel)
        return llm
    
    @pytest.fixture
    def extract_service(self, mock_llm):
        """創建 LangExtract 服務實例"""
        return LangExtractService(llm=mock_llm)
    
    def test_regex_extract(self, extract_service):
        """測試正則表達式提取"""
        text = """
        2024-01-15 10:30:45 ERROR [web-prod-03] Service: api-gateway
        CPU usage: 85.5%, Memory usage: 92%
        Error code: E5001 - Connection timeout
        IP: 192.168.1.100, HTTP Status: 500
        """
        
        result = extract_service.extract(text, use_llm=False)
        
        assert isinstance(result, ExtractedData)
        assert result.entities.timestamp is not None
        assert result.entities.hostname == "web-prod-03"
        assert result.entities.service_name == "api-gateway"
        assert result.entities.cpu_usage == 85.5
        assert result.entities.memory_usage == 92.0
        assert result.entities.error_code == "E5001"
        assert result.entities.ip_address == "192.168.1.100"
        assert result.entities.http_status == 500
        assert result.entities.log_level == "ERROR"
    
    def test_llm_extract(self, extract_service, mock_llm):
        """測試 LLM 提取"""
        text = "主機 db-master-01 的 MySQL 服務發生記憶體溢出，使用率達到 98%"
        
        # 模擬 LLM 返回
        mock_llm.invoke.return_value = MagicMock(
            content='{"hostname": "db-master-01", "service_name": "MySQL", "memory_usage": 98.0}'
        )
        
        # 設置 parser 的行為
        extract_service.parser.parse = Mock(return_value=AIOpsEntity(
            hostname="db-master-01",
            service_name="MySQL",
            memory_usage=98.0
        ))
        
        result = extract_service.extract(text, use_llm=True)
        
        assert result.entities.hostname == "db-master-01"
        assert result.entities.service_name == "MySQL"
        assert result.entities.memory_usage == 98.0
    
    def test_confidence_calculation(self, extract_service):
        """測試信心分數計算"""
        # 完整的實體
        full_entity = AIOpsEntity(
            hostname="test-host",
            service_name="test-service",
            timestamp=datetime.now(),
            error_code="E001"
        )
        confidence = extract_service._calculate_confidence(full_entity)
        assert confidence > 0.5
        
        # 空實體
        empty_entity = AIOpsEntity()
        confidence = extract_service._calculate_confidence(empty_entity)
        assert confidence == 0.0
    
    def test_extract_to_metadata(self, extract_service):
        """測試元數據轉換"""
        text = "ERROR: Host web-01 CPU 75%"
        
        metadata = extract_service.extract_to_metadata(text, use_llm=False)
        
        assert isinstance(metadata, dict)
        assert "extracted_hostname" in metadata
        assert "extracted_cpu_usage" in metadata
        assert "extraction_confidence" in metadata
        assert "extraction_timestamp" in metadata


class TestChunkingService:
    """測試分塊服務"""
    
    @pytest.fixture
    def chunking_service(self):
        return ChunkingService(chunk_size=100, chunk_overlap=20)
    
    def test_chunk_with_metadata(self, chunking_service):
        """測試帶元數據的分塊"""
        text = "這是一段很長的文本。" * 20  # 創建長文本
        base_metadata = {"source": "test.log", "timestamp": "2024-01-15"}
        extracted_metadata = {
            "extracted_hostname": "test-host",
            "extracted_service_name": "test-service"
        }
        
        documents = chunking_service.chunk_with_metadata(
            text, base_metadata, extracted_metadata
        )
        
        assert len(documents) > 1  # 應該被分成多塊
        for doc in documents:
            assert doc.metadata["source"] == "test.log"
            assert doc.metadata["extracted_hostname"] == "test-host"
            assert "chunk_id" in doc.metadata
            assert "chunk_index" in doc.metadata


class TestLangGraphIntegration:
    """測試 LangGraph 整合"""
    
    @pytest.fixture
    def mock_components(self):
        """創建模擬組件"""
        llm = Mock(spec=BaseLanguageModel)
        retriever = Mock()
        extract_service = Mock(spec=LangExtractService)
        
        # 設置模擬行為
        llm.invoke.return_value = MagicMock(content="測試回答")
        retriever.get_relevant_documents.return_value = [
            Document(page_content="相關文檔1", metadata={"id": "1"}),
            Document(page_content="相關文檔2", metadata={"id": "2"})
        ]
        
        extract_service.batch_extract.return_value = [
            ExtractedData(
                entities=AIOpsEntity(hostname="test-host", cpu_usage=85.0),
                confidence=0.9,
                raw_text="原始日誌"
            )
        ]
        extract_service.extract_to_metadata.return_value = {
            "extracted_hostname": "test-host",
            "extracted_cpu_usage": 85.0,
            "_raw_extracted": {"hostname": "test-host", "cpu_usage": 85.0},
            "_extraction_confidence": 0.9
        }
        
        return {
            "llm": llm,
            "retriever": retriever,
            "extract_service": extract_service
        }
    
    def test_graph_with_extraction(self, mock_components):
        """測試包含提取的完整流程"""
        # 構建圖
        app = build_graph(
            llm=mock_components["llm"],
            retriever=mock_components["retriever"],
            extract_service=mock_components["extract_service"],
            policy={
                "use_llm_extract": True,
                "use_metadata_filter": True
            }
        )
        
        # 準備輸入
        input_state = {
            "query": "web-01 主機的 CPU 使用率異常",
            "raw_texts": ["ERROR: Host web-01 CPU 85%"]
        }
        
        # 執行圖
        config = {"configurable": {"thread_id": "test-thread"}}
        result = app.invoke(input_state, config)
        
        # 驗證結果
        assert "answer" in result
        assert "extracted_data" in result
        assert len(result["extracted_data"]) > 0
        assert result["extracted_data"][0]["extracted_hostname"] == "test-host"
    
    def test_graph_without_extraction(self, mock_components):
        """測試不包含提取的流程"""
        # 構建圖（不提供 extract_service）
        app = build_graph(
            llm=mock_components["llm"],
            retriever=mock_components["retriever"],
            extract_service=None
        )
        
        # 準備輸入
        input_state = {
            "query": "系統效能報告"
        }
        
        # 執行圖
        config = {"configurable": {"thread_id": "test-thread"}}
        result = app.invoke(input_state, config)
        
        # 驗證結果
        assert "answer" in result
        assert "extracted_data" not in result or result.get("extracted_data") == []
    
    def test_metadata_filtering(self, mock_components):
        """測試元數據過濾功能"""
        # 設置 retriever 支援元數據過濾
        mock_components["retriever"].search_kwargs = {}
        
        # 構建圖
        app = build_graph(
            llm=mock_components["llm"],
            retriever=mock_components["retriever"],
            extract_service=mock_components["extract_service"],
            policy={
                "use_metadata_filter": True
            }
        )
        
        # 準備輸入
        input_state = {
            "query": "查詢 test-host 的問題",
            "raw_texts": ["Host: test-host ERROR"]
        }
        
        # 執行圖
        config = {"configurable": {"thread_id": "test-thread"}}
        result = app.invoke(input_state, config)
        
        # 驗證元數據過濾被應用
        assert "metrics" in result
        assert "metadata_filters" in result["metrics"]
        assert "extracted_hostname" in result["metrics"]["metadata_filters"]


class TestEndToEndScenarios:
    """端到端場景測試"""
    
    @pytest.fixture
    def setup_real_services(self):
        """設置真實服務（使用模擬的 LLM）"""
        llm = Mock(spec=BaseLanguageModel)
        llm.invoke.return_value = MagicMock(content="分析結果：系統負載過高")
        
        extract_service = LangExtractService(llm=llm)
        
        retriever = Mock()
        retriever.get_relevant_documents.return_value = [
            Document(
                page_content="CPU 使用率持續超過 80% 可能導致系統回應緩慢",
                metadata={"source": "best-practices.md"}
            )
        ]
        retriever.search_kwargs = {}
        
        return {
            "llm": llm,
            "extract_service": extract_service,
            "retriever": retriever
        }
    
    def test_prometheus_alert_scenario(self, setup_real_services):
        """測試 Prometheus 告警場景"""
        # Prometheus 告警文本
        alert_text = """
        ALERT: High CPU Usage
        Instance: web-prod-03.example.com
        Service: nginx
        CPU: 95%
        Memory: 45%
        Time: 2024-01-15T10:30:00Z
        """
        
        # 構建圖
        app = build_graph(**setup_real_services)
        
        # 執行
        result = app.invoke({
            "query": "分析這個 CPU 高使用率告警",
            "raw_texts": [alert_text]
        }, {"configurable": {"thread_id": "alert-test"}})
        
        # 驗證
        assert "answer" in result
        assert "extracted_data" in result
        extracted = result["extracted_data"][0]
        assert extracted["extracted_hostname"] == "web-prod-03.example.com"
        assert extracted["extracted_service_name"] == "nginx"
        assert extracted["extracted_cpu_usage"] == 95.0
    
    def test_elasticsearch_log_scenario(self, setup_real_services):
        """測試 Elasticsearch 日誌場景"""
        # Elasticsearch 日誌
        log_text = """
        {"@timestamp":"2024-01-15T10:30:00.000Z","level":"ERROR","logger":"com.example.api",
        "message":"Database connection timeout","host":"db-master-01","service":"user-api",
        "error_code":"DB_TIMEOUT_001","response_time":5000}
        """
        
        # 構建圖
        app = build_graph(**setup_real_services)
        
        # 執行
        result = app.invoke({
            "query": "分析資料庫超時錯誤",
            "raw_texts": [log_text]
        }, {"configurable": {"thread_id": "log-test"}})
        
        # 驗證
        assert "answer" in result
        assert "extracted_data" in result
        extracted = result["extracted_data"][0]
        assert extracted.get("extracted_error_code") == "DB_TIMEOUT_001"
        assert extracted.get("extracted_response_time") == 5000


if __name__ == "__main__":
    pytest.main([__file__, "-v"])