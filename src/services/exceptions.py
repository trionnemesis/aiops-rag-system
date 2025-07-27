"""
Custom exceptions for AIOps RAG System
"""


class RAGServiceError(Exception):
    """Base exception for RAG service errors"""
    pass


class VectorDBError(RAGServiceError):
    """Error related to vector database operations"""
    pass


class GeminiAPIError(RAGServiceError):
    """Error related to Gemini API calls"""
    pass


class PrometheusError(RAGServiceError):
    """Error related to Prometheus data fetching"""
    pass


class HyDEGenerationError(RAGServiceError):
    """Error during HyDE query generation"""
    pass


class DocumentRetrievalError(RAGServiceError):
    """Error during document retrieval"""
    pass


class ReportGenerationError(RAGServiceError):
    """Error during report generation"""
    pass


class CacheError(RAGServiceError):
    """Error related to cache operations"""
    pass