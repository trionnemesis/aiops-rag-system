"""
Fixtures and helper functions for integration testing.
"""

import pytest
import time
import json
import os
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from contextlib import contextmanager
import requests
from opensearchpy import OpenSearch
import redis
import docker
from pathlib import Path


@dataclass
class ServiceEndpoints:
    """Container for service endpoints."""
    api_url: str = "http://localhost:8000"
    opensearch_url: str = "http://localhost:9200"
    redis_url: str = "redis://localhost:6379"
    prometheus_url: str = "http://localhost:9090"


@dataclass
class TestDocument:
    """Test document structure."""
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None


class IntegrationTestHelper:
    """Helper class for integration testing operations."""
    
    def __init__(self, endpoints: ServiceEndpoints):
        self.endpoints = endpoints
        self.opensearch_client = None
        self.redis_client = None
        self._setup_clients()
    
    def _setup_clients(self):
        """Initialize client connections."""
        self.opensearch_client = OpenSearch(
            hosts=[{'host': 'localhost', 'port': 9200}],
            http_compress=True,
            use_ssl=False,
            verify_certs=False,
            ssl_assert_hostname=False,
            ssl_show_warn=False,
        )
        self.redis_client = redis.from_url(self.endpoints.redis_url)
    
    def wait_for_service(self, service_name: str, timeout: int = 60) -> bool:
        """Wait for a specific service to be ready."""
        start_time = time.time()
        
        check_functions = {
            'api': self._check_api_health,
            'opensearch': self._check_opensearch_health,
            'redis': self._check_redis_health,
        }
        
        check_fn = check_functions.get(service_name)
        if not check_fn:
            raise ValueError(f"Unknown service: {service_name}")
        
        while time.time() - start_time < timeout:
            if check_fn():
                return True
            time.sleep(1)
        
        return False
    
    def _check_api_health(self) -> bool:
        """Check API health."""
        try:
            resp = requests.get(f"{self.endpoints.api_url}/health", timeout=2)
            return resp.status_code == 200
        except:
            return False
    
    def _check_opensearch_health(self) -> bool:
        """Check OpenSearch health."""
        try:
            health = self.opensearch_client.cluster.health()
            return health['status'] in ['green', 'yellow']
        except:
            return False
    
    def _check_redis_health(self) -> bool:
        """Check Redis health."""
        try:
            return self.redis_client.ping()
        except:
            return False
    
    def create_test_index(self, index_name: str = "test-aiops-docs"):
        """Create a test index with proper mappings."""
        if self.opensearch_client.indices.exists(index=index_name):
            self.opensearch_client.indices.delete(index=index_name)
        
        index_body = {
            "settings": {
                "number_of_shards": 1,
                "number_of_replicas": 0,
                "index": {
                    "knn": True,
                    "knn.space_type": "cosinesimil"
                }
            },
            "mappings": {
                "properties": {
                    "content": {"type": "text"},
                    "metadata": {"type": "object"},
                    "embedding": {
                        "type": "knn_vector",
                        "dimension": 768,
                        "method": {
                            "name": "hnsw",
                            "space_type": "cosinesimil",
                            "engine": "nmslib",
                            "parameters": {
                                "ef_construction": 128,
                                "m": 24
                            }
                        }
                    }
                }
            }
        }
        
        self.opensearch_client.indices.create(index=index_name, body=index_body)
        return index_name
    
    def index_test_documents(self, documents: List[TestDocument], index_name: str):
        """Index test documents."""
        for i, doc in enumerate(documents):
            doc_body = {
                "content": doc.content,
                "metadata": doc.metadata,
                "embedding": doc.embedding or [0.1] * 768  # Dummy embedding
            }
            
            self.opensearch_client.index(
                index=index_name,
                body=doc_body,
                id=f"test-doc-{i}",
                refresh=True
            )
    
    def clear_redis_data(self, pattern: str = "*"):
        """Clear Redis data matching pattern."""
        for key in self.redis_client.scan_iter(match=pattern):
            self.redis_client.delete(key)
    
    @contextmanager
    def temporary_index(self, index_name: str = "temp-test-index"):
        """Context manager for temporary test index."""
        try:
            self.create_test_index(index_name)
            yield index_name
        finally:
            if self.opensearch_client.indices.exists(index=index_name):
                self.opensearch_client.indices.delete(index=index_name)
    
    def make_rag_request(self, query: str, **kwargs) -> Dict[str, Any]:
        """Make a RAG query request."""
        request_data = {
            "query": query,
            **kwargs
        }
        
        response = requests.post(
            f"{self.endpoints.api_url}/rag/query",
            json=request_data,
            timeout=30
        )
        
        return {
            "status_code": response.status_code,
            "data": response.json() if response.status_code == 200 else None,
            "error": response.text if response.status_code != 200 else None
        }
    
    def verify_response_structure(self, response_data: Dict[str, Any]) -> bool:
        """Verify RAG response has expected structure."""
        required_fields = ["answer", "sources", "request_id", "processing_time"]
        return all(field in response_data for field in required_fields)
    
    def get_redis_state(self, key: str) -> Optional[Dict[str, Any]]:
        """Get state from Redis."""
        data = self.redis_client.get(key)
        return json.loads(data) if data else None
    
    def get_metrics(self) -> str:
        """Get Prometheus metrics from API."""
        resp = requests.get(f"{self.endpoints.api_url}/metrics")
        return resp.text if resp.status_code == 200 else ""


# Pytest fixtures
@pytest.fixture(scope="session")
def service_endpoints():
    """Provide service endpoints configuration."""
    return ServiceEndpoints()


@pytest.fixture(scope="session")
def test_helper(service_endpoints):
    """Provide test helper instance."""
    return IntegrationTestHelper(service_endpoints)


@pytest.fixture
def clean_redis(test_helper):
    """Ensure Redis is clean before test."""
    test_helper.clear_redis_data()
    yield
    test_helper.clear_redis_data()


@pytest.fixture
def test_documents():
    """Provide sample test documents."""
    return [
        TestDocument(
            content="Kubernetes pod crash due to OOMKilled. Memory limit exceeded.",
            metadata={"source": "k8s", "severity": "error"}
        ),
        TestDocument(
            content="Database connection timeout. Pool exhausted.",
            metadata={"source": "app", "severity": "error"}
        ),
        TestDocument(
            content="Network latency spike detected between regions.",
            metadata={"source": "network", "severity": "warning"}
        ),
    ]


# Helper functions for common test scenarios
def assert_rag_response_valid(response: Dict[str, Any]):
    """Assert that a RAG response is valid."""
    assert response["status_code"] == 200
    assert response["data"] is not None
    
    data = response["data"]
    assert "answer" in data
    assert "sources" in data
    assert "request_id" in data
    assert "processing_time" in data
    
    assert len(data["answer"]) > 0
    assert isinstance(data["sources"], list)
    assert isinstance(data["processing_time"], (int, float))


def assert_contains_relevant_content(response: Dict[str, Any], keywords: List[str]):
    """Assert response contains relevant content."""
    answer = response["data"]["answer"].lower()
    sources_text = " ".join(
        source.get("content", "").lower() 
        for source in response["data"]["sources"]
    )
    
    combined_text = f"{answer} {sources_text}"
    
    for keyword in keywords:
        assert keyword.lower() in combined_text, \
            f"Expected keyword '{keyword}' not found in response"


def wait_for_eventual_consistency(
    check_fn, 
    timeout: int = 10, 
    interval: float = 0.5
) -> bool:
    """Wait for eventual consistency with timeout."""
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        if check_fn():
            return True
        time.sleep(interval)
    
    return False


class MockEmbeddingGenerator:
    """Generate mock embeddings for testing."""
    
    @staticmethod
    def generate(text: str, dimension: int = 768) -> List[float]:
        """Generate deterministic mock embedding from text."""
        # Simple hash-based approach for consistent embeddings
        import hashlib
        
        hash_obj = hashlib.md5(text.encode())
        hash_hex = hash_obj.hexdigest()
        
        # Convert to floats between -1 and 1
        embedding = []
        for i in range(0, len(hash_hex), 2):
            value = int(hash_hex[i:i+2], 16) / 127.5 - 1.0
            embedding.append(value)
        
        # Pad or truncate to desired dimension
        if len(embedding) < dimension:
            embedding.extend([0.0] * (dimension - len(embedding)))
        else:
            embedding = embedding[:dimension]
        
        return embedding


@pytest.fixture
def mock_embeddings():
    """Provide mock embedding generator."""
    return MockEmbeddingGenerator()


# Performance testing utilities
class PerformanceMonitor:
    """Monitor performance metrics during tests."""
    
    def __init__(self):
        self.metrics = []
    
    @contextmanager
    def measure(self, operation_name: str):
        """Measure operation time."""
        start_time = time.time()
        try:
            yield
        finally:
            duration = time.time() - start_time
            self.metrics.append({
                "operation": operation_name,
                "duration": duration,
                "timestamp": time.time()
            })
    
    def get_stats(self, operation_name: str = None) -> Dict[str, float]:
        """Get performance statistics."""
        filtered = self.metrics
        if operation_name:
            filtered = [m for m in self.metrics if m["operation"] == operation_name]
        
        if not filtered:
            return {}
        
        durations = [m["duration"] for m in filtered]
        return {
            "count": len(durations),
            "min": min(durations),
            "max": max(durations),
            "avg": sum(durations) / len(durations),
            "total": sum(durations)
        }


@pytest.fixture
def performance_monitor():
    """Provide performance monitor."""
    return PerformanceMonitor()