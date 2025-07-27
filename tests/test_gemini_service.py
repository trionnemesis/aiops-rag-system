import pytest
import asyncio
import os
from unittest.mock import Mock, AsyncMock, patch
from src.services.gemini_service import GeminiService
from src.config import settings


class TestGeminiService:
    """Test cases for GeminiService"""

    @pytest.fixture
    def gemini_service(self, monkeypatch):
        """Create a GeminiService instance for testing"""
        monkeypatch.setenv("TESTING", "true")
        return GeminiService()

    @pytest.fixture
    def real_gemini_service(self, monkeypatch):
        """Create a GeminiService instance with real API key"""
        monkeypatch.setenv("TESTING", "false")
        monkeypatch.setattr(settings, "gemini_api_key", "real-api-key")
        with patch("google.generativeai.configure"):
            with patch("google.generativeai.GenerativeModel") as mock_model:
                mock_model.return_value = Mock()
                service = GeminiService()
                return service

    @pytest.mark.asyncio
    async def test_init_testing_mode(self, monkeypatch):
        """Test initialization in testing mode"""
        monkeypatch.setenv("TESTING", "true")
        service = GeminiService()
        assert service.flash_model is None
        assert service.pro_model is None

    @pytest.mark.asyncio
    async def test_init_production_mode(self, monkeypatch):
        """Test initialization in production mode"""
        monkeypatch.setenv("TESTING", "false")
        monkeypatch.setattr(settings, "gemini_api_key", "real-api-key")
        
        with patch("google.generativeai.configure") as mock_configure:
            with patch("google.generativeai.GenerativeModel") as mock_model:
                service = GeminiService()
                mock_configure.assert_called_once_with(api_key="real-api-key")
                assert mock_model.call_count == 2

    @pytest.mark.asyncio
    async def test_generate_hyde_testing_mode(self, gemini_service):
        """Test HyDE generation in testing mode"""
        result = await gemini_service.generate_hyde("test prompt")
        assert result == "Test HyDE response"

    @pytest.mark.asyncio
    async def test_generate_hyde_production_mode(self, real_gemini_service):
        """Test HyDE generation in production mode"""
        # Mock the model's generate_content method
        mock_response = Mock()
        mock_response.text = "Generated HyDE content"
        real_gemini_service.flash_model = Mock()
        real_gemini_service.flash_model.generate_content = Mock(return_value=mock_response)
        
        with patch("asyncio.to_thread", new=AsyncMock(return_value=mock_response)):
            result = await real_gemini_service.generate_hyde("test prompt")
            assert result == "Generated HyDE content"

    @pytest.mark.asyncio
    async def test_generate_hyde_error(self, real_gemini_service):
        """Test HyDE generation error handling"""
        real_gemini_service.flash_model = Mock()
        real_gemini_service.flash_model.generate_content = Mock(side_effect=Exception("API Error"))
        
        with patch("asyncio.to_thread", side_effect=Exception("API Error")):
            with pytest.raises(Exception, match="HyDE generation failed: API Error"):
                await real_gemini_service.generate_hyde("test prompt")

    @pytest.mark.asyncio
    async def test_summarize_document_testing_mode(self, gemini_service):
        """Test document summarization in testing mode"""
        result = await gemini_service.summarize_document("test document")
        assert result == "Test summary"

    @pytest.mark.asyncio
    async def test_summarize_document_production_mode(self, real_gemini_service):
        """Test document summarization in production mode"""
        mock_response = Mock()
        mock_response.text = "A" * 1000  # Long text to test truncation
        real_gemini_service.flash_model = Mock()
        real_gemini_service.flash_model.generate_content = Mock(return_value=mock_response)
        
        with patch("asyncio.to_thread", new=AsyncMock(return_value=mock_response)):
            result = await real_gemini_service.summarize_document("test document")
            assert len(result) <= settings.max_summary_length

    @pytest.mark.asyncio
    async def test_summarize_document_error(self, real_gemini_service):
        """Test document summarization error handling"""
        real_gemini_service.flash_model = Mock()
        real_gemini_service.flash_model.generate_content = Mock(side_effect=Exception("API Error"))
        
        with patch("asyncio.to_thread", side_effect=Exception("API Error")):
            with pytest.raises(Exception, match="Document summarization failed: API Error"):
                await real_gemini_service.summarize_document("test document")

    @pytest.mark.asyncio
    async def test_generate_final_report_testing_mode(self, gemini_service):
        """Test final report generation in testing mode"""
        result = await gemini_service.generate_final_report("test prompt")
        assert result["insight_analysis"] == "Test insight analysis"
        assert result["recommendations"] == "Test recommendations"

    @pytest.mark.asyncio
    async def test_generate_final_report_production_mode(self, real_gemini_service):
        """Test final report generation in production mode"""
        mock_response = Mock()
        mock_response.text = "洞見分析\nThis is insight\n具體建議\nThese are recommendations"
        real_gemini_service.pro_model = Mock()
        real_gemini_service.pro_model.generate_content = Mock(return_value=mock_response)
        
        with patch("asyncio.to_thread", new=AsyncMock(return_value=mock_response)):
            result = await real_gemini_service.generate_final_report("test prompt")
            assert "This is insight" in result["insight_analysis"]
            assert "These are recommendations" in result["recommendations"]

    @pytest.mark.asyncio
    async def test_generate_final_report_no_recommendations(self, real_gemini_service):
        """Test final report generation without recommendations section"""
        mock_response = Mock()
        mock_response.text = "洞見分析\nThis is only insight"
        real_gemini_service.pro_model = Mock()
        real_gemini_service.pro_model.generate_content = Mock(return_value=mock_response)
        
        with patch("asyncio.to_thread", new=AsyncMock(return_value=mock_response)):
            result = await real_gemini_service.generate_final_report("test prompt")
            assert "This is only insight" in result["insight_analysis"]
            assert result["recommendations"] == ""

    @pytest.mark.asyncio
    async def test_generate_final_report_error(self, real_gemini_service):
        """Test final report generation error handling"""
        real_gemini_service.pro_model = Mock()
        real_gemini_service.pro_model.generate_content = Mock(side_effect=Exception("API Error"))
        
        with patch("asyncio.to_thread", side_effect=Exception("API Error")):
            with pytest.raises(Exception, match="Final report generation failed: API Error"):
                await real_gemini_service.generate_final_report("test prompt")

    @pytest.mark.asyncio
    async def test_generate_embedding_testing_mode(self, monkeypatch):
        """Test embedding generation in testing mode"""
        monkeypatch.setenv("TESTING", "true")
        service = GeminiService()
        result = await service.generate_embedding("test text")
        assert len(result) == settings.opensearch_embedding_dim
        assert all(x == 0.1 for x in result)

    @pytest.mark.asyncio
    async def test_generate_embedding_production_mode(self, monkeypatch):
        """Test embedding generation in production mode"""
        monkeypatch.setenv("TESTING", "false")
        service = GeminiService()
        
        mock_result = {"embedding": [0.1, 0.2, 0.3]}
        with patch("google.generativeai.embed_content", return_value=mock_result):
            with patch("asyncio.to_thread", new=AsyncMock(return_value=mock_result)):
                result = await service.generate_embedding("test text")
                assert result == [0.1, 0.2, 0.3]

    @pytest.mark.asyncio
    async def test_generate_embedding_error(self, monkeypatch):
        """Test embedding generation error handling"""
        monkeypatch.setenv("TESTING", "false")
        service = GeminiService()
        
        with patch("google.generativeai.embed_content", side_effect=Exception("API Error")):
            with patch("asyncio.to_thread", side_effect=Exception("API Error")):
                with pytest.raises(Exception, match="Embedding generation failed: API Error"):
                    await service.generate_embedding("test text")