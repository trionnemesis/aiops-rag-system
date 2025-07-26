from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
from datetime import datetime

class PortTraffic(BaseModel):
    port_80_443_connections: Optional[int] = Field(None, alias="Port 80/443 流入連線數")

class ServiceMetrics(BaseModel):
    apache_workers: Optional[int] = Field(None, alias="Apache活躍工作程序")
    nginx_error_rate: Optional[Dict[str, int]] = Field(None, alias="Nginx日誌錯誤率")

class MonitoringData(BaseModel):
    host: str = Field(..., alias="主機")
    collection_time: str = Field(..., alias="採集時間")
    cpu_usage: str = Field(..., alias="CPU使用率")
    ram_usage: str = Field(..., alias="RAM使用率")
    disk_io_wait: str = Field(..., alias="磁碟I/O等待")
    network_outbound: str = Field(..., alias="網路流出量")
    port_traffic: Optional[PortTraffic] = Field(None, alias="作業系統Port流量")
    service_metrics: Optional[ServiceMetrics] = Field(None, alias="服務指標")
    
    class Config:
        populate_by_name = True

class ReportRequest(BaseModel):
    monitoring_data: Dict[str, Any]

class InsightReport(BaseModel):
    insight_analysis: str
    recommendations: str
    generated_at: datetime
    
class ReportResponse(BaseModel):
    status: str
    report: InsightReport
    monitoring_data: Dict[str, Any]