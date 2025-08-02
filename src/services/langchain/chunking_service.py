"""
Chunking and Embedding Service
支援結構化元數據的文檔分塊和向量化服務
"""
from typing import List, Dict, Any, Optional, Tuple
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.embeddings import Embeddings
from datetime import datetime
import hashlib


class ChunkingService:
    """
    文檔分塊服務
    負責將原始文本分割成適合向量化的塊，並附加結構化元數據
    """
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: Optional[List[str]] = None
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", " ", ""]
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=self.separators,
            length_function=len,
        )
    
    def chunk_with_metadata(
        self,
        text: str,
        base_metadata: Dict[str, Any],
        extracted_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """
        將文本分塊並附加元數據
        
        Args:
            text: 原始文本
            base_metadata: 基礎元數據（如來源、時間等）
            extracted_metadata: LangExtract 提取的結構化元數據
        
        Returns:
            List[Document]: 帶有元數據的文檔塊列表
        """
        # 分割文本
        chunks = self.text_splitter.split_text(text)
        
        # 準備元數據
        metadata = base_metadata.copy()
        if extracted_metadata:
            # 合併提取的元數據
            metadata.update(extracted_metadata)
        
        # 創建文檔列表
        documents = []
        for i, chunk in enumerate(chunks):
            # 為每個塊生成唯一 ID
            chunk_id = self._generate_chunk_id(text, i)
            
            # 創建塊特定的元數據
            chunk_metadata = metadata.copy()
            chunk_metadata.update({
                "chunk_id": chunk_id,
                "chunk_index": i,
                "total_chunks": len(chunks),
                "chunk_size": len(chunk),
                "chunking_timestamp": datetime.now().isoformat()
            })
            
            doc = Document(
                page_content=chunk,
                metadata=chunk_metadata
            )
            documents.append(doc)
        
        return documents
    
    def _generate_chunk_id(self, text: str, index: int) -> str:
        """生成塊的唯一 ID"""
        hash_input = f"{text[:100]}{index}"
        return hashlib.md5(hash_input.encode()).hexdigest()
    
    def batch_chunk_with_metadata(
        self,
        texts: List[str],
        base_metadata_list: List[Dict[str, Any]],
        extracted_metadata_list: Optional[List[Dict[str, Any]]] = None
    ) -> List[Document]:
        """
        批量處理多個文本的分塊
        
        Args:
            texts: 文本列表
            base_metadata_list: 基礎元數據列表
            extracted_metadata_list: 提取的元數據列表
        
        Returns:
            List[Document]: 所有文檔的塊列表
        """
        all_documents = []
        
        for i, text in enumerate(texts):
            base_metadata = base_metadata_list[i] if i < len(base_metadata_list) else {}
            extracted_metadata = extracted_metadata_list[i] if extracted_metadata_list and i < len(extracted_metadata_list) else None
            
            documents = self.chunk_with_metadata(text, base_metadata, extracted_metadata)
            all_documents.extend(documents)
        
        return all_documents


class EmbeddingService:
    """
    向量化服務
    負責將文檔塊轉換為向量表示，保留元數據
    """
    
    def __init__(self, embeddings: Embeddings):
        self.embeddings = embeddings
    
    def embed_documents_with_metadata(
        self,
        documents: List[Document]
    ) -> List[Tuple[List[float], Dict[str, Any]]]:
        """
        將文檔列表向量化，保留元數據
        
        Args:
            documents: 文檔列表
        
        Returns:
            List[Tuple[vector, metadata]]: 向量和元數據的配對列表
        """
        # 提取文本內容
        texts = [doc.page_content for doc in documents]
        
        # 批量向量化
        vectors = self.embeddings.embed_documents(texts)
        
        # 配對向量和元數據
        results = []
        for i, (vector, doc) in enumerate(zip(vectors, documents)):
            results.append((vector, doc.metadata))
        
        return results
    
    def embed_query(self, query: str) -> List[float]:
        """向量化查詢文本"""
        return self.embeddings.embed_query(query)


class ChunkingAndEmbeddingPipeline:
    """
    完整的分塊和向量化管道
    整合 LangExtract、分塊和向量化的完整流程
    """
    
    def __init__(
        self,
        chunking_service: ChunkingService,
        embedding_service: EmbeddingService,
        extract_service=None
    ):
        self.chunking_service = chunking_service
        self.embedding_service = embedding_service
        self.extract_service = extract_service
    
    def process(
        self,
        texts: List[str],
        base_metadata_list: List[Dict[str, Any]],
        use_extraction: bool = True
    ) -> List[Tuple[List[float], Dict[str, Any]]]:
        """
        處理文本列表的完整管道
        
        Args:
            texts: 原始文本列表
            base_metadata_list: 基礎元數據列表
            use_extraction: 是否使用 LangExtract 提取結構化資訊
        
        Returns:
            List[Tuple[vector, metadata]]: 向量和元數據的配對列表
        """
        # 步驟 1: 提取結構化資訊（如果啟用）
        extracted_metadata_list = None
        if use_extraction and self.extract_service:
            extracted_metadata_list = []
            for text in texts:
                metadata = self.extract_service.extract_to_metadata(text)
                extracted_metadata_list.append(metadata)
        
        # 步驟 2: 分塊並附加元數據
        documents = self.chunking_service.batch_chunk_with_metadata(
            texts,
            base_metadata_list,
            extracted_metadata_list
        )
        
        # 步驟 3: 向量化並保留元數據
        vector_metadata_pairs = self.embedding_service.embed_documents_with_metadata(documents)
        
        return vector_metadata_pairs
    
    def process_single(
        self,
        text: str,
        base_metadata: Dict[str, Any],
        use_extraction: bool = True
    ) -> List[Tuple[List[float], Dict[str, Any]]]:
        """處理單個文本"""
        return self.process([text], [base_metadata], use_extraction)