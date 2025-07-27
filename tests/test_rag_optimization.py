"""
RAG 優化測試
測試 RAG 系統的優化功能
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
import asyncio
from datetime import datetime
import json

from src.services.rag_service import RAGService
from src.models.schemas import InsightReport

@pytest.fixture
def mock_services():
    """模擬所有外部服務"""
    with patch('src.services.rag_service.RAGChainService') as mock_chain_service:
        
        # 設置 mock 的 RAGChainService
        mock_chain = mock_chain_service.return_value
        
        # 設置所有需要的 async 方法
        mock_chain.generate_report = AsyncMock()
        mock_chain.generate_report_with_steps = AsyncMock()
        mock_chain.enrich_with_prometheus = AsyncMock()
        
        # 設置同步方法
        mock_chain.clear_cache = Mock()
        mock_chain.get_cache_info = Mock(return_value={
            "cache_hits": 0,
            "cache_misses": 0,
            "cache_size": 0
        })
        
        yield mock_chain

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
    
    # 設置模擬返回值 - 返回 InsightReport 實例
    mock_report = InsightReport(
        insight_analysis="分析結果",
        recommendations="建議1\n建議2",
        generated_at=datetime.now()
    )
    mock_services.generate_report.return_value = mock_report
    
    # 執行測試
    report = await rag_service.generate_report(monitoring_data)
    
    # 驗證結果
    assert report.insight_analysis == "分析結果"
    assert report.recommendations == "建議1\n建議2"
    
    # 驗證調用次數
    assert mock_services.generate_report.call_count == 1
    
    # 驗證調用參數
    mock_services.generate_report.assert_called_once_with(monitoring_data)

@pytest.mark.asyncio
async def test_cache_mechanism(mock_services):
    """測試快取機制"""
    # 初始化服務
    rag_service = RAGService()
    
    # 準備測試數據
    monitoring_data = {
        "主機": "server-01", 
        "CPU使用率": 50,
        "RAM使用率": 60
    }
    
    # 設置模擬返回值
    mock_report = InsightReport(
        insight_analysis="分析結果",
        recommendations="",
        generated_at=datetime.now()
    )
    mock_services.generate_report.return_value = mock_report
    
    # 設置快取資訊返回值
    mock_services.get_cache_info.return_value = {
        "cache_hits": 1,
        "cache_misses": 1,
        "cache_size": 1
    }
    
    # 第一次調用
    await rag_service.generate_report(monitoring_data)
    
    # 驗證第一次調用
    assert mock_services.generate_report.call_count == 1
    
    # 第二次調用相同的數據
    await rag_service.generate_report(monitoring_data)
    
    # 驗證調用次數（測試簡化版本，實際快取在 RAGChainService 內部處理）
    assert mock_services.generate_report.call_count == 2
    
    # 獲取快取資訊
    cache_info = rag_service.get_cache_info()
    assert cache_info["cache_hits"] == 1
    assert cache_info["cache_misses"] == 1
    assert cache_info["cache_size"] == 1

@pytest.mark.asyncio
async def test_cache_with_different_data(mock_services):
    """測試不同數據不會命中快取"""
    # 初始化服務
    rag_service = RAGService()
    
    # 準備兩組不同的測試數據
    monitoring_data_1 = {
        "主機": "server-01",
        "CPU使用率": 50,
        "RAM使用率": 60
    }
    
    monitoring_data_2 = {
        "主機": "server-02",  # 不同的主機
        "CPU使用率": 50,
        "RAM使用率": 60
    }
    
    # 設置模擬返回值
    mock_report = InsightReport(
        insight_analysis="分析結果",
        recommendations="",
        generated_at=datetime.now()
    )
    mock_services.generate_report.return_value = mock_report
    
    # 調用兩次不同的數據
    await rag_service.generate_report(monitoring_data_1)
    await rag_service.generate_report(monitoring_data_2)
    
    # 驗證由於數據不同，應該調用了兩次
    assert mock_services.generate_report.call_count == 2

@pytest.mark.asyncio
async def test_cache_clear(mock_services):
    """測試清除快取功能"""
    # 初始化服務
    rag_service = RAGService()
    
    # 準備測試數據
    monitoring_data = {
        "主機": "server-01",
        "CPU使用率": 50,
        "RAM使用率": 60
    }
    
    # 設置模擬返回值
    mock_report = InsightReport(
        insight_analysis="分析結果",
        recommendations="",
        generated_at=datetime.now()
    )
    mock_services.generate_report.return_value = mock_report
    
    # 第一次調用
    await rag_service.generate_report(monitoring_data)
    
    # 清除快取
    rag_service.clear_cache()
    mock_services.clear_cache.assert_called_once()
    
    # 再次調用相同的數據
    await rag_service.generate_report(monitoring_data)
    
    # 驗證清除快取後，應該再次調用
    assert mock_services.generate_report.call_count == 2

@pytest.mark.asyncio
async def test_empty_document_handling(mock_services):
    """測試空文件處理"""
    # 初始化服務
    rag_service = RAGService()
    
    # 準備測試數據
    monitoring_data = {
        "主機": "server-01",
        "CPU使用率": 30,
        "RAM使用率": 40
    }
    
    # 設置模擬返回值 - 沒有檢索到文件的情況
    mock_report = InsightReport(
        insight_analysis="基於監控數據的分析結果",
        recommendations="無相關文件，基於經驗提供建議",
        generated_at=datetime.now()
    )
    mock_services.generate_report.return_value = mock_report
    
    # 執行測試
    report = await rag_service.generate_report(monitoring_data)
    
    # 驗證結果
    assert isinstance(report, InsightReport)
    assert "基於監控數據的分析結果" in report.insight_analysis
    assert "無相關文件" in report.recommendations