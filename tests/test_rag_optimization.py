"""
測試 RAG 服務的優化功能
- 測試文件摘要整合
- 測試快取機制
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from src.services.rag_service import RAGService
from src.models.schemas import InsightReport
from datetime import datetime

@pytest.fixture
def mock_services():
    """模擬所有外部服務"""
    with patch('src.services.rag_service.GeminiService') as mock_gemini, \
         patch('src.services.rag_service.OpenSearchService') as mock_opensearch, \
         patch('src.services.rag_service.PrometheusService') as mock_prometheus, \
         patch('src.services.rag_service.PromptTemplates') as mock_prompts:
        
        # 設置模擬物件
        mock_gemini_instance = AsyncMock()
        mock_opensearch_instance = AsyncMock()
        mock_prometheus_instance = AsyncMock()
        mock_prompts_instance = MagicMock()
        
        # 配置返回值
        mock_gemini.return_value = mock_gemini_instance
        mock_opensearch.return_value = mock_opensearch_instance
        mock_prometheus.return_value = mock_prometheus_instance
        mock_prompts.return_value = mock_prompts_instance
        
        # 設置 prompt templates
        mock_prompts_instance.HYDE_GENERATION = "HyDE: {monitoring_data}"
        mock_prompts_instance.SUMMARY_REFINEMENT = "Summary: {monitoring_data} {document}"
        mock_prompts_instance.FINAL_REPORT = "Report: {monitoring_data} {summaries}"
        
        yield {
            'gemini': mock_gemini_instance,
            'opensearch': mock_opensearch_instance,
            'prometheus': mock_prometheus_instance,
            'prompts': mock_prompts_instance
        }

@pytest.mark.asyncio
async def test_consolidated_document_summarization(mock_services):
    """測試文件摘要整合功能"""
    # 初始化服務
    rag_service = RAGService()
    
    # 準備測試數據
    monitoring_data = {
        "主機": "server-01",
        "CPU使用率": 85,
        "RAM使用率": 90
    }
    
    # 設置模擬返回值
    mock_services['gemini'].generate_hyde.return_value = "假設性文件內容"
    mock_services['gemini'].generate_embedding.return_value = [0.1] * 768
    
    # 模擬多個檢索文件
    similar_docs = [
        {"content": "文件1內容", "event_id": "1", "title": "標題1"},
        {"content": "文件2內容", "event_id": "2", "title": "標題2"},
        {"content": "文件3內容", "event_id": "3", "title": "標題3"},
        {"content": "文件4內容", "event_id": "4", "title": "標題4"},
        {"content": "文件5內容", "event_id": "5", "title": "標題5"}
    ]
    mock_services['opensearch'].search_similar_documents.return_value = similar_docs
    
    # 設置摘要和報告生成的返回值
    mock_services['gemini'].summarize_document.return_value = "整合摘要結果"
    mock_services['gemini'].generate_final_report.return_value = {
        "insight_analysis": "分析結果",
        "recommendations": ["建議1", "建議2"]
    }
    
    # 執行測試
    report = await rag_service.generate_report(monitoring_data)
    
    # 驗證結果
    assert isinstance(report, InsightReport)
    assert report.insight_analysis == "分析結果"
    assert report.recommendations == ["建議1", "建議2"]
    
    # 驗證只調用了一次摘要功能（而不是5次）
    assert mock_services['gemini'].summarize_document.call_count == 1
    
    # 驗證摘要調用時包含了所有文件內容
    call_args = mock_services['gemini'].summarize_document.call_args[0][0]
    assert "文件1內容" in call_args
    assert "文件2內容" in call_args
    assert "文件3內容" in call_args
    assert "文件4內容" in call_args
    assert "文件5內容" in call_args

@pytest.mark.asyncio
async def test_cache_mechanism(mock_services):
    """測試快取機制"""
    # 初始化服務
    rag_service = RAGService()
    
    # 準備測試數據
    monitoring_data = {
        "主機": "server-01",
        "CPU使用率": 85,
        "RAM使用率": 90
    }
    
    # 設置模擬返回值
    mock_services['gemini'].generate_hyde.return_value = "假設性文件內容"
    mock_services['gemini'].generate_embedding.return_value = [0.1] * 768
    mock_services['opensearch'].search_similar_documents.return_value = []
    mock_services['gemini'].generate_final_report.return_value = {
        "insight_analysis": "分析結果",
        "recommendations": []
    }
    
    # 第一次調用
    await rag_service.generate_report(monitoring_data)
    
    # 驗證第一次調用
    assert mock_services['gemini'].generate_hyde.call_count == 1
    assert mock_services['gemini'].generate_embedding.call_count == 1
    
    # 第二次調用相同的數據
    await rag_service.generate_report(monitoring_data)
    
    # 驗證快取生效（HyDE 和 embedding 不應該再次調用）
    assert mock_services['gemini'].generate_hyde.call_count == 1
    assert mock_services['gemini'].generate_embedding.call_count == 1
    
    # 獲取快取資訊
    cache_info = rag_service.get_cache_info()
    
    # 驗證快取命中
    assert cache_info['hyde_cache']['hits'] > 0
    assert cache_info['embedding_cache']['hits'] > 0

@pytest.mark.asyncio
async def test_cache_with_different_data(mock_services):
    """測試不同數據的快取行為"""
    # 初始化服務
    rag_service = RAGService()
    
    # 準備不同的測試數據
    monitoring_data_1 = {
        "主機": "server-01",
        "CPU使用率": 85,
        "RAM使用率": 90
    }
    
    monitoring_data_2 = {
        "主機": "server-02",  # 不同的主機
        "CPU使用率": 85,
        "RAM使用率": 90
    }
    
    # 設置模擬返回值
    mock_services['gemini'].generate_hyde.return_value = "假設性文件內容"
    mock_services['gemini'].generate_embedding.return_value = [0.1] * 768
    mock_services['opensearch'].search_similar_documents.return_value = []
    mock_services['gemini'].generate_final_report.return_value = {
        "insight_analysis": "分析結果",
        "recommendations": []
    }
    
    # 調用不同的數據
    await rag_service.generate_report(monitoring_data_1)
    await rag_service.generate_report(monitoring_data_2)
    
    # 驗證由於數據不同，應該調用了兩次（沒有命中快取）
    assert mock_services['gemini'].generate_hyde.call_count == 2

@pytest.mark.asyncio
async def test_cache_clear(mock_services):
    """測試清除快取功能"""
    # 初始化服務
    rag_service = RAGService()
    
    # 準備測試數據
    monitoring_data = {
        "主機": "server-01",
        "CPU使用率": 85,
        "RAM使用率": 90
    }
    
    # 設置模擬返回值
    mock_services['gemini'].generate_hyde.return_value = "假設性文件內容"
    mock_services['gemini'].generate_embedding.return_value = [0.1] * 768
    mock_services['opensearch'].search_similar_documents.return_value = []
    mock_services['gemini'].generate_final_report.return_value = {
        "insight_analysis": "分析結果",
        "recommendations": []
    }
    
    # 第一次調用
    await rag_service.generate_report(monitoring_data)
    
    # 清除快取
    rag_service.clear_cache()
    
    # 第二次調用相同的數據
    await rag_service.generate_report(monitoring_data)
    
    # 驗證清除快取後，應該再次調用
    assert mock_services['gemini'].generate_hyde.call_count == 2
    assert mock_services['gemini'].generate_embedding.call_count == 2

@pytest.mark.asyncio
async def test_empty_document_handling(mock_services):
    """測試沒有檢索到文件時的處理"""
    # 初始化服務
    rag_service = RAGService()
    
    # 準備測試數據
    monitoring_data = {
        "主機": "server-01",
        "CPU使用率": 85,
        "RAM使用率": 90
    }
    
    # 設置模擬返回值
    mock_services['gemini'].generate_hyde.return_value = "假設性文件內容"
    mock_services['gemini'].generate_embedding.return_value = [0.1] * 768
    mock_services['opensearch'].search_similar_documents.return_value = []  # 沒有檢索到文件
    mock_services['gemini'].generate_final_report.return_value = {
        "insight_analysis": "分析結果",
        "recommendations": []
    }
    
    # 執行測試
    report = await rag_service.generate_report(monitoring_data)
    
    # 驗證結果
    assert isinstance(report, InsightReport)
    
    # 驗證沒有調用摘要功能
    assert mock_services['gemini'].summarize_document.call_count == 0
    
    # 驗證最終報告調用時使用了正確的摘要文本
    final_report_call = mock_services['gemini'].generate_final_report.call_args[0][0]
    assert "無相關文件摘要" in final_report_call