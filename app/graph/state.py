from __future__ import annotations
from typing import List, Optional, Literal, Dict, Any
from pydantic import BaseModel, Field, field_validator
from langchain_core.documents import Document


class RAGState(BaseModel):
    """RAG workflow state with strong typing and validation"""
    
    # 輸入
    query: str = Field(..., min_length=1, max_length=1000, description="User query")
    raw_texts: Optional[List[str]] = Field(
        default=None, 
        max_items=100,
        description="原始日誌/告警文本列表"
    )
    
    # 提取階段
    extracted_data: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        max_items=100,
        description="LangExtract 提取的結構化資料"
    )
    
    # 規劃階段
    route: Literal["fast", "deep"] = Field(
        default="fast",
        description="fast: 直接向量檢索；deep: 觸發 HyDE/多查詢"
    )
    queries: List[str] = Field(
        default_factory=list,
        max_items=10,
        description="實際用來檢索的查詢（含 HyDE / multi-query）"
    )
    
    # 檢索階段
    docs: List[Document] = Field(
        default_factory=list,
        max_items=50,
        description="彙整後的候選文檔（已去重）"
    )
    
    # 生成階段
    context: str = Field(
        default="",
        max_length=10000,
        description="拼接後的上下文"
    )
    answer: str = Field(
        default="",
        max_length=5000,
        description="最終輸出"
    )
    
    # 控制/觀測
    error: Optional[str] = Field(
        default=None,
        max_length=1000,
        description="Error message if any"
    )
    metrics: Dict[str, Any] = Field(
        default_factory=dict,
        description="e.g. token_used, k, ef_search, rrf_on, num_docs, latency_ms"
    )
    
    # Additional fields that might be used
    request_id: Optional[str] = Field(
        default=None,
        description="Request ID for tracking"
    )
    
    class Config:
        arbitrary_types_allowed = True  # Allow Document type
        
    def dict(self, **kwargs) -> Dict[str, Any]:
        """Convert to dictionary for LangGraph compatibility"""
        # Ensure we can serialize to dict for LangGraph
        d = super().dict(**kwargs)
        # Convert Document objects to serializable format if needed
        if 'docs' in d and d['docs']:
            # Keep Document objects as-is for LangGraph
            pass
        return d
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RAGState':
        """Create from dictionary for LangGraph compatibility"""
        return cls(**data)
    
    @field_validator('query')
    @classmethod
    def validate_query(cls, v: str) -> str:
        """Validate query is not empty or just whitespace"""
        if not v or not v.strip():
            raise ValueError("Query cannot be empty or whitespace only")
        return v.strip()
    
    @field_validator('queries')
    @classmethod
    def validate_queries(cls, v: List[str]) -> List[str]:
        """Validate each query in queries list"""
        return [q.strip() for q in v if q and q.strip()]
    
    @field_validator('raw_texts')
    @classmethod
    def validate_raw_texts(cls, v: Optional[List[str]]) -> Optional[List[str]]:
        """Validate and clean raw texts"""
        if v is None:
            return v
        return [text.strip() for text in v if text and text.strip()]
    
    @field_validator('context')
    @classmethod
    def validate_context(cls, v: str) -> str:
        """Ensure context doesn't exceed maximum length"""
        if len(v) > 10000:
            return v[:10000]  # Truncate if too long
        return v