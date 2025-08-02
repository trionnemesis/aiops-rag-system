"""
LangExtract Service for AIOps
結構化資訊提取服務，用於從日誌和告警文本中提取關鍵實體
"""
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, asdict
from datetime import datetime
import json
import re
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field


class AIOpsEntity(BaseModel):
    """AIOps 實體 Schema"""
    # 基本資訊
    timestamp: Optional[datetime] = Field(None, description="事件時間戳")
    log_level: Optional[str] = Field(None, description="日誌級別 (ERROR, WARN, INFO)")
    
    # 系統資訊
    hostname: Optional[str] = Field(None, description="主機名稱")
    service_name: Optional[str] = Field(None, description="服務名稱")
    component: Optional[str] = Field(None, description="組件名稱")
    environment: Optional[str] = Field(None, description="環境 (prod, staging, dev)")
    
    # 錯誤資訊
    error_code: Optional[str] = Field(None, description="錯誤碼")
    error_message: Optional[str] = Field(None, description="錯誤訊息")
    stack_trace: Optional[str] = Field(None, description="堆疊追蹤")
    
    # 效能指標
    cpu_usage: Optional[float] = Field(None, description="CPU 使用率")
    memory_usage: Optional[float] = Field(None, description="記憶體使用率")
    disk_usage: Optional[float] = Field(None, description="硬碟使用率")
    response_time: Optional[float] = Field(None, description="回應時間 (ms)")
    
    # 網路相關
    ip_address: Optional[str] = Field(None, description="IP 地址")
    port: Optional[int] = Field(None, description="埠號")
    http_status: Optional[int] = Field(None, description="HTTP 狀態碼")
    request_method: Optional[str] = Field(None, description="HTTP 方法")
    endpoint: Optional[str] = Field(None, description="API 端點")
    
    # 額外標籤
    tags: Optional[List[str]] = Field(default_factory=list, description="標籤列表")
    custom_fields: Optional[Dict[str, Any]] = Field(default_factory=dict, description="自訂欄位")


class ExtractedData(BaseModel):
    """提取結果資料結構"""
    entities: AIOpsEntity
    confidence: float = Field(description="提取信心分數", ge=0.0, le=1.0)
    raw_text: str = Field(description="原始文本")
    extraction_timestamp: datetime = Field(default_factory=datetime.now, description="提取時間")


class LangExtractService:
    """
    LangExtract 服務
    負責從非結構化文本中提取結構化的 AIOps 實體資訊
    """
    
    def __init__(self, llm: BaseLanguageModel):
        self.llm = llm
        self.parser = JsonOutputParser(pydantic_object=AIOpsEntity)
        
        # 預編譯常用的正則表達式
        self._compile_patterns()
        
    def _compile_patterns(self):
        """編譯常用的正則表達式模式"""
        self.patterns = {
            'timestamp': re.compile(r'\d{4}-\d{2}-\d{2}[T\s]\d{2}:\d{2}:\d{2}'),
            'ip': re.compile(r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b'),
            'error_code': re.compile(r'(?:ERROR|ERR)[-_]?\d+|E\d{3,}'),
            'percentage': re.compile(r'(\d+(?:\.\d+)?)\s*%'),
            'http_status': re.compile(r'\b[1-5]\d{2}\b'),
            'hostname': re.compile(r'(?:host(?:name)?|server)[:=\s]+([a-zA-Z0-9\-\.]+)'),
            'service': re.compile(r'(?:service|app(?:lication)?)[:=\s]+([a-zA-Z0-9\-_]+)'),
        }
    
    def extract(self, text: str, use_llm: bool = True) -> ExtractedData:
        """
        從文本中提取結構化資訊
        
        Args:
            text: 原始日誌或告警文本
            use_llm: 是否使用 LLM 進行智慧提取（預設為 True）
        
        Returns:
            ExtractedData: 提取結果
        """
        # 第一步：使用正則表達式進行快速提取
        entities = self._regex_extract(text)
        
        # 第二步：如果啟用 LLM，使用 LLM 進行深度提取
        if use_llm:
            llm_entities = self._llm_extract(text)
            # 合併結果，LLM 結果優先
            entities = self._merge_entities(entities, llm_entities)
        
        # 計算信心分數
        confidence = self._calculate_confidence(entities)
        
        return ExtractedData(
            entities=entities,
            confidence=confidence,
            raw_text=text
        )
    
    def _regex_extract(self, text: str) -> AIOpsEntity:
        """使用正則表達式進行基本提取"""
        data = {}
        
        # 提取時間戳
        timestamp_match = self.patterns['timestamp'].search(text)
        if timestamp_match:
            try:
                data['timestamp'] = datetime.fromisoformat(timestamp_match.group())
            except:
                pass
        
        # 提取 IP 地址
        ip_match = self.patterns['ip'].search(text)
        if ip_match:
            data['ip_address'] = ip_match.group()
        
        # 提取錯誤碼
        error_code_match = self.patterns['error_code'].search(text)
        if error_code_match:
            data['error_code'] = error_code_match.group()
        
        # 提取 HTTP 狀態碼
        http_status_match = self.patterns['http_status'].search(text)
        if http_status_match:
            data['http_status'] = int(http_status_match.group())
        
        # 提取百分比（CPU、記憶體等）
        percentage_matches = self.patterns['percentage'].findall(text)
        if percentage_matches:
            # 簡單的啟發式規則來判斷是哪種使用率
            text_lower = text.lower()
            for i, percent in enumerate(percentage_matches):
                percent_val = float(percent)
                if 'cpu' in text_lower:
                    data['cpu_usage'] = percent_val
                elif 'memory' in text_lower or 'mem' in text_lower:
                    data['memory_usage'] = percent_val
                elif 'disk' in text_lower:
                    data['disk_usage'] = percent_val
        
        # 提取主機名
        hostname_match = self.patterns['hostname'].search(text)
        if hostname_match:
            data['hostname'] = hostname_match.group(1)
        
        # 提取服務名
        service_match = self.patterns['service'].search(text)
        if service_match:
            data['service_name'] = service_match.group(1)
        
        # 提取日誌級別
        if 'ERROR' in text.upper():
            data['log_level'] = 'ERROR'
        elif 'WARN' in text.upper():
            data['log_level'] = 'WARN'
        elif 'INFO' in text.upper():
            data['log_level'] = 'INFO'
        
        return AIOpsEntity(**data)
    
    def _llm_extract(self, text: str) -> AIOpsEntity:
        """使用 LLM 進行智慧提取"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一個 AIOps 日誌分析專家。請從以下文本中提取結構化資訊。
            
輸出格式：{format_instructions}

注意事項：
1. 只提取文本中明確存在的資訊
2. 對於不確定的資訊，請留空（null）
3. 時間戳請轉換為 ISO 格式
4. 百分比數值請去掉 % 符號，只保留數字"""),
            ("user", "請分析以下日誌文本並提取結構化資訊：\n\n{text}")
        ])
        
        chain = prompt | self.llm | self.parser
        
        try:
            result = chain.invoke({
                "format_instructions": self.parser.get_format_instructions(),
                "text": text
            })
            return result
        except Exception as e:
            # 如果 LLM 提取失敗，返回空實體
            return AIOpsEntity()
    
    def _merge_entities(self, base: AIOpsEntity, override: AIOpsEntity) -> AIOpsEntity:
        """合併兩個實體，override 的值優先"""
        base_dict = base.dict()
        override_dict = override.dict()
        
        # 合併邏輯：如果 override 中的值不為 None，則使用它
        for key, value in override_dict.items():
            if value is not None:
                base_dict[key] = value
        
        return AIOpsEntity(**base_dict)
    
    def _calculate_confidence(self, entities: AIOpsEntity) -> float:
        """計算提取的信心分數"""
        # 統計非空欄位的數量
        entity_dict = entities.dict()
        total_fields = len(entity_dict)
        filled_fields = sum(1 for v in entity_dict.values() if v is not None)
        
        # 基礎信心分數
        base_confidence = filled_fields / total_fields if total_fields > 0 else 0
        
        # 關鍵欄位加權
        key_fields = ['hostname', 'service_name', 'timestamp', 'error_code']
        key_field_bonus = sum(0.1 for field in key_fields if entity_dict.get(field) is not None)
        
        # 計算最終信心分數
        confidence = min(base_confidence + key_field_bonus, 1.0)
        
        return round(confidence, 2)
    
    def batch_extract(self, texts: List[str], use_llm: bool = True) -> List[ExtractedData]:
        """批量提取多個文本"""
        results = []
        for text in texts:
            result = self.extract(text, use_llm=use_llm)
            results.append(result)
        return results
    
    def extract_to_metadata(self, text: str, use_llm: bool = True) -> Dict[str, Any]:
        """
        提取並轉換為適合作為向量資料庫元數據的格式
        
        Returns:
            Dict[str, Any]: 扁平化的元數據字典
        """
        extracted = self.extract(text, use_llm=use_llm)
        
        # 將實體轉換為扁平的元數據格式
        metadata = {}
        entity_dict = extracted.entities.dict()
        
        for key, value in entity_dict.items():
            if value is not None:
                if isinstance(value, datetime):
                    metadata[f"extracted_{key}"] = value.isoformat()
                elif isinstance(value, (list, dict)):
                    metadata[f"extracted_{key}"] = json.dumps(value)
                else:
                    metadata[f"extracted_{key}"] = value
        
        # 添加提取元數據
        metadata['extraction_confidence'] = extracted.confidence
        metadata['extraction_timestamp'] = extracted.extraction_timestamp.isoformat()
        
        return metadata