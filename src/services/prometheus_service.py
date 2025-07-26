import aiohttp
from typing import Dict, Any, List
from datetime import datetime, timedelta
from src.config import settings
import json

class PrometheusService:
    def __init__(self):
        self.base_url = f"http://{settings.prometheus_host}:{settings.prometheus_port}"
        
    async def query(self, promql: str) -> Dict[str, Any]:
        """執行 Prometheus 查詢"""
        url = f"{self.base_url}/api/v1/query"
        params = {"query": promql}
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as response:
                data = await response.json()
                if data["status"] == "success":
                    return data["data"]
                else:
                    raise Exception(f"Prometheus query failed: {data}")
    
    async def query_range(self, promql: str, start: datetime, end: datetime, 
                         step: str = "15s") -> Dict[str, Any]:
        """執行 Prometheus 範圍查詢"""
        url = f"{self.base_url}/api/v1/query_range"
        params = {
            "query": promql,
            "start": start.timestamp(),
            "end": end.timestamp(),
            "step": step
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as response:
                data = await response.json()
                if data["status"] == "success":
                    return data["data"]
                else:
                    raise Exception(f"Prometheus range query failed: {data}")
    
    async def get_host_metrics(self, hostname: str) -> Dict[str, Any]:
        """獲取主機的各項指標"""
        metrics = {}
        
        # CPU 使用率
        cpu_query = f'100 - (avg(rate(node_cpu_seconds_total{{mode="idle",instance=~"{hostname}.*"}}[5m])) * 100)'
        cpu_data = await self.query(cpu_query)
        if cpu_data["result"]:
            metrics["CPU使用率"] = f"{float(cpu_data['result'][0]['value'][1]):.1f}%"
        
        # 記憶體使用率
        mem_query = f'(1 - (node_memory_MemAvailable_bytes{{instance=~"{hostname}.*"}} / node_memory_MemTotal_bytes{{instance=~"{hostname}.*"}})) * 100'
        mem_data = await self.query(mem_query)
        if mem_data["result"]:
            metrics["RAM使用率"] = f"{float(mem_data['result'][0]['value'][1]):.1f}%"
        
        # 磁碟 I/O 等待
        io_query = f'rate(node_disk_io_time_seconds_total{{instance=~"{hostname}.*"}}[5m]) * 100'
        io_data = await self.query(io_query)
        if io_data["result"]:
            metrics["磁碟I/O等待"] = f"{float(io_data['result'][0]['value'][1]):.1f}%"
        
        # 網路流量
        net_query = f'rate(node_network_transmit_bytes_total{{instance=~"{hostname}.*",device!="lo"}}[5m]) * 8 / 1000000'
        net_data = await self.query(net_query)
        if net_data["result"]:
            total_mbps = sum(float(r['value'][1]) for r in net_data['result'])
            metrics["網路流出量"] = f"{total_mbps:.0f} Mbps"
        
        return metrics