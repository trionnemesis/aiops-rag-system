import aiohttp
from typing import Dict, Any, List
from datetime import datetime, timedelta
from src.config import settings
import json
import time
from prometheus_client import Counter, Histogram, Gauge, Summary

# 定義向量檢索相關的 Prometheus 指標
vector_search_counter = Counter(
    'vector_search_total',
    'Total number of vector searches',
    ['strategy', 'index']
)

vector_search_latency = Histogram(
    'vector_search_duration_seconds',
    'Vector search duration in seconds',
    ['strategy', 'index'],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0)
)

vector_search_results = Histogram(
    'vector_search_results_count',
    'Number of results returned by vector search',
    ['strategy', 'index'],
    buckets=(0, 1, 5, 10, 20, 50, 100, 200)
)

opensearch_cluster_health = Gauge(
    'opensearch_cluster_health',
    'OpenSearch cluster health status (0=red, 1=yellow, 2=green)',
    ['cluster']
)

opensearch_index_docs = Gauge(
    'opensearch_index_document_count',
    'Number of documents in OpenSearch index',
    ['index']
)

opensearch_index_size = Gauge(
    'opensearch_index_size_bytes',
    'Size of OpenSearch index in bytes',
    ['index']
)

ef_search_value = Gauge(
    'opensearch_ef_search_value',
    'Current ef_search parameter value',
    ['index']
)

vector_search_recall = Summary(
    'vector_search_recall',
    'Vector search recall rate',
    ['query_type']
)

vector_search_precision = Summary(
    'vector_search_precision',
    'Vector search precision rate',
    ['query_type']
)

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
        
        # 增加 1 分鐘系統負載查詢
        load_query = f'node_load1{{instance=~"{hostname}.*"}}'
        load_data = await self.query(load_query)
        if load_data["result"]:
            metrics["系統一分鐘負載"] = f"{float(load_data['result'][0]['value'][1]):.2f}"
        
        # 增加 TCP 連線數查詢
        tcp_conn_query = f'node_netstat_Tcp_CurrEstab{{instance=~"{hostname}.*"}}'
        tcp_conn_data = await self.query(tcp_conn_query)
        if tcp_conn_data["result"]:
            metrics["TCP當前連線數"] = f"{int(float(tcp_conn_data['result'][0]['value'][1]))}"
        
        # 增加磁碟讀取速率 (IOPS)
        disk_read_query = f'rate(node_disk_reads_completed_total{{instance=~"{hostname}.*"}}[5m])'
        disk_read_data = await self.query(disk_read_query)
        if disk_read_data["result"]:
            total_read_iops = sum(float(r['value'][1]) for r in disk_read_data['result'])
            metrics["磁碟讀取IOPS"] = f"{total_read_iops:.0f}"
        
        # 增加磁碟寫入速率 (IOPS)
        disk_write_query = f'rate(node_disk_writes_completed_total{{instance=~"{hostname}.*"}}[5m])'
        disk_write_data = await self.query(disk_write_query)
        if disk_write_data["result"]:
            total_write_iops = sum(float(r['value'][1]) for r in disk_write_data['result'])
            metrics["磁碟寫入IOPS"] = f"{total_write_iops:.0f}"
        
        # 增加磁碟讀取吞吐量 (MB/s)
        disk_read_bytes_query = f'rate(node_disk_read_bytes_total{{instance=~"{hostname}.*"}}[5m]) / 1024 / 1024'
        disk_read_bytes_data = await self.query(disk_read_bytes_query)
        if disk_read_bytes_data["result"]:
            total_read_mbps = sum(float(r['value'][1]) for r in disk_read_bytes_data['result'])
            metrics["磁碟讀取速率"] = f"{total_read_mbps:.1f} MB/s"
        
        # 增加磁碟寫入吞吐量 (MB/s)
        disk_write_bytes_query = f'rate(node_disk_written_bytes_total{{instance=~"{hostname}.*"}}[5m]) / 1024 / 1024'
        disk_write_bytes_data = await self.query(disk_write_bytes_query)
        if disk_write_bytes_data["result"]:
            total_write_mbps = sum(float(r['value'][1]) for r in disk_write_bytes_data['result'])
            metrics["磁碟寫入速率"] = f"{total_write_mbps:.1f} MB/s"
        
        return metrics
    
    async def get_opensearch_metrics(self, cluster_name: str = "opensearch") -> Dict[str, Any]:
        """獲取 OpenSearch 叢集的各項指標"""
        metrics = {}
        
        # OpenSearch JVM Heap 使用率
        jvm_heap_query = f'opensearch_jvm_memory_heap_used_percent{{cluster="{cluster_name}"}}'
        jvm_data = await self.query(jvm_heap_query)
        if jvm_data["result"]:
            metrics["JVM Heap使用率"] = f"{float(jvm_data['result'][0]['value'][1]):.1f}%"
        
        # OpenSearch CPU 使用率
        cpu_query = f'opensearch_os_cpu_percent{{cluster="{cluster_name}"}}'
        cpu_data = await self.query(cpu_query)
        if cpu_data["result"]:
            metrics["OpenSearch CPU使用率"] = f"{float(cpu_data['result'][0]['value'][1]):.1f}%"
        
        # 索引查詢速率
        query_rate = f'rate(opensearch_index_search_query_total{{cluster="{cluster_name}"}}[5m])'
        query_data = await self.query(query_rate)
        if query_data["result"]:
            metrics["查詢速率"] = f"{float(query_data['result'][0]['value'][1]):.2f} QPS"
        
        # 索引延遲
        latency_query = f'opensearch_index_search_query_time_seconds{{cluster="{cluster_name}"}}'
        latency_data = await self.query(latency_query)
        if latency_data["result"]:
            metrics["平均查詢延遲"] = f"{float(latency_data['result'][0]['value'][1]) * 1000:.2f} ms"
        
        # 向量檢索指標
        vector_latency_query = 'histogram_quantile(0.95, vector_search_duration_seconds_bucket)'
        vector_data = await self.query(vector_latency_query)
        if vector_data["result"]:
            metrics["向量檢索P95延遲"] = f"{float(vector_data['result'][0]['value'][1]) * 1000:.2f} ms"
        
        # ef_search 當前值
        ef_query = 'opensearch_ef_search_value'
        ef_data = await self.query(ef_query)
        if ef_data["result"]:
            metrics["ef_search參數"] = int(float(ef_data['result'][0]['value'][1]))
        
        return metrics
    
    async def get_vector_search_stats(self, time_range: str = "5m") -> Dict[str, Any]:
        """獲取向量檢索統計資訊"""
        stats = {}
        
        # 總查詢數
        total_query = f'sum(increase(vector_search_total[{time_range}]))'
        total_data = await self.query(total_query)
        if total_data["result"]:
            stats["總查詢數"] = int(float(total_data['result'][0]['value'][1]))
        
        # 按策略分組的查詢數
        strategy_query = f'sum by (strategy) (increase(vector_search_total[{time_range}]))'
        strategy_data = await self.query(strategy_query)
        if strategy_data["result"]:
            stats["策略分布"] = {
                r['metric']['strategy']: int(float(r['value'][1]))
                for r in strategy_data['result']
            }
        
        # 延遲百分位數
        for percentile in [50, 95, 99]:
            latency_query = f'histogram_quantile({percentile/100}, sum(rate(vector_search_duration_seconds_bucket[{time_range}])) by (le))'
            latency_data = await self.query(latency_query)
            if latency_data["result"]:
                stats[f"P{percentile}延遲"] = f"{float(latency_data['result'][0]['value'][1]) * 1000:.2f} ms"
        
        # 平均返回結果數
        results_query = f'avg(vector_search_results_count)'
        results_data = await self.query(results_query)
        if results_data["result"]:
            stats["平均返回結果數"] = f"{float(results_data['result'][0]['value'][1]):.1f}"
        
        # 召回率和準確率
        recall_query = 'avg(vector_search_recall)'
        recall_data = await self.query(recall_query)
        if recall_data["result"]:
            stats["平均召回率"] = f"{float(recall_data['result'][0]['value'][1]) * 100:.2f}%"
        
        precision_query = 'avg(vector_search_precision)'
        precision_data = await self.query(precision_query)
        if precision_data["result"]:
            stats["平均準確率"] = f"{float(precision_data['result'][0]['value'][1]) * 100:.2f}%"
        
        return stats