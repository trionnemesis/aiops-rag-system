from __future__ import annotations
from typing import TypedDict, List, Optional, Literal, Dict, Any
from langchain_core.documents import Document

class RAGState(TypedDict, total=False):
    # 輸入
    query: str
    raw_texts: Optional[List[str]]              # 原始日誌/告警文本列表
    # 提取階段
    extracted_data: Optional[List[Dict[str, Any]]]  # LangExtract 提取的結構化資料
    # 規劃階段
    route: Literal["fast", "deep"]          # fast: 直接向量檢索；deep: 觸發 HyDE/多查詢
    queries: List[str]                       # 實際用來檢索的查詢（含 HyDE / multi-query）
    # 檢索階段
    docs: List[Document]                     # 彙整後的候選文檔（已去重）
    # 生成階段
    context: str                             # 拼接後的上下文（可用你原本的組裝）
    answer: str                              # 最終輸出
    # 控制/觀測
    error: Optional[str]
    metrics: Dict[str, Any]                  # e.g. token_used, k, ef_search, rrf_on, num_docs, latency_ms