"""
Integration tests for the RAG system with real OpenSearch, Redis and API interactions.

These tests use docker-compose to spin up a complete testing environment and verify
the entire RAG pipeline end-to-end.
"""

import os
import time
import pytest
import requests
import redis
from opensearchpy import OpenSearch
from typing import Dict, Any, List
import docker
import subprocess
from pathlib import Path

# Import our custom fixtures and helpers
from tests.fixtures.integration_fixtures import (
    ServiceEndpoints, 
    IntegrationTestHelper,
    TestDocument,
    assert_rag_response_valid,
    assert_contains_relevant_content,
    wait_for_eventual_consistency,
    service_endpoints,
    test_helper,
    clean_redis,
    test_documents,
    mock_embeddings,
    performance_monitor
)

# Test configuration
API_BASE_URL = "http://localhost:8000"
OPENSEARCH_URL = "http://localhost:9200"
REDIS_URL = "redis://localhost:6379"
TEST_TIMEOUT = 300  # 5 minutes for container startup


class IntegrationTestEnvironment:
    """Manages the docker-compose test environment lifecycle."""
    
    def __init__(self):
        self.docker_client = docker.from_env()
        self.compose_file = Path(__file__).parent.parent / "docker-compose.test.yml"
        self.project_name = "rag-integration-test"
        
    def start(self):
        """Start the test environment using docker-compose."""
        cmd = [
            "docker-compose",
            "-f", str(self.compose_file),
            "-p", self.project_name,
            "up", "-d"
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        
        # Wait for services to be healthy
        self._wait_for_services()
        
    def stop(self):
        """Stop and clean up the test environment."""
        cmd = [
            "docker-compose",
            "-f", str(self.compose_file),
            "-p", self.project_name,
            "down", "-v"
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        
    def _wait_for_services(self):
        """Wait for all services to be healthy."""
        start_time = time.time()
        
        while time.time() - start_time < TEST_TIMEOUT:
            if self._check_api_health() and self._check_opensearch_health() and self._check_redis_health():
                return
            time.sleep(2)
            
        raise TimeoutError("Services did not become healthy within timeout period")
        
    def _check_api_health(self) -> bool:
        """Check if the API service is healthy."""
        try:
            response = requests.get(f"{API_BASE_URL}/health", timeout=5)
            return response.status_code == 200
        except:
            return False
            
    def _check_opensearch_health(self) -> bool:
        """Check if OpenSearch is healthy."""
        try:
            client = OpenSearch(
                hosts=[{'host': 'localhost', 'port': 9200}],
                http_compress=True,
                use_ssl=False,
                verify_certs=False,
                ssl_assert_hostname=False,
                ssl_show_warn=False,
            )
            health = client.cluster.health()
            return health['status'] in ['green', 'yellow']
        except:
            return False
            
    def _check_redis_health(self) -> bool:
        """Check if Redis is healthy."""
        try:
            r = redis.from_url(REDIS_URL)
            return r.ping()
        except:
            return False


@pytest.fixture(scope="session")
def test_environment():
    """Set up and tear down the integration test environment."""
    env = IntegrationTestEnvironment()
    
    # Start environment
    print("Starting integration test environment...")
    env.start()
    
    yield env
    
    # Cleanup
    print("Stopping integration test environment...")
    env.stop()


@pytest.fixture
def api_client(test_environment):
    """Provides a configured requests session for API calls."""
    session = requests.Session()
    session.headers.update({
        "Content-Type": "application/json"
    })
    return session


@pytest.fixture
def opensearch_client(test_environment):
    """Provides an OpenSearch client for direct database operations."""
    return OpenSearch(
        hosts=[{'host': 'localhost', 'port': 9200}],
        http_compress=True,
        use_ssl=False,
        verify_certs=False,
        ssl_assert_hostname=False,
        ssl_show_warn=False,
    )


@pytest.fixture
def redis_client(test_environment):
    """Provides a Redis client for state verification."""
    return redis.from_url(REDIS_URL)


class TestRAGIntegration:
    """Integration tests for the complete RAG pipeline."""
    
    def test_health_check(self, api_client):
        """Test that all services are properly connected."""
        response = api_client.get(f"{API_BASE_URL}/health")
        assert response.status_code == 200
        
        health_data = response.json()
        assert health_data["status"] == "healthy"
        assert "opensearch" in health_data
        assert "redis" in health_data
        
    def test_document_indexing_flow(self, api_client, opensearch_client):
        """Test document indexing through the API."""
        # Prepare test documents
        test_documents = [
            {
                "content": "Kubernetes pod OOMKilled due to memory limit exceeded. Container memory usage reached 2Gi while limit was set to 1Gi.",
                "metadata": {
                    "source": "k8s-logs",
                    "timestamp": "2024-01-15T10:30:00Z",
                    "severity": "error"
                }
            },
            {
                "content": "Application performance degraded. Response time increased from 200ms to 2s. Database connection pool exhausted.",
                "metadata": {
                    "source": "apm-metrics",
                    "timestamp": "2024-01-15T10:35:00Z",
                    "severity": "warning"
                }
            }
        ]
        
        # Index documents via API
        response = api_client.post(
            f"{API_BASE_URL}/documents/index",
            json={"documents": test_documents}
        )
        assert response.status_code == 200
        
        # Wait for indexing
        time.sleep(2)
        
        # Verify documents in OpenSearch
        index_name = "aiops-docs"  # Adjust based on your actual index name
        search_response = opensearch_client.search(
            index=index_name,
            body={"query": {"match_all": {}}}
        )
        
        assert search_response["hits"]["total"]["value"] >= 2
        
    def test_rag_query_flow(self, api_client, redis_client):
        """Test complete RAG query flow with state persistence."""
        # Send RAG query
        query_data = {
            "query": "Why is my Kubernetes pod getting OOMKilled?",
            "top_k": 5,
            "use_hyde": True,
            "use_multi_query": True
        }
        
        response = api_client.post(
            f"{API_BASE_URL}/rag/query",
            json=query_data
        )
        
        assert response.status_code == 200
        result = response.json()
        
        # Verify response structure
        assert "answer" in result
        assert "sources" in result
        assert "request_id" in result
        assert "processing_time" in result
        
        # Verify answer quality
        assert len(result["answer"]) > 50
        assert "memory" in result["answer"].lower() or "oom" in result["answer"].lower()
        
        # Verify sources
        assert len(result["sources"]) > 0
        for source in result["sources"]:
            assert "content" in source
            assert "metadata" in source
            assert "relevance_score" in source
            
        # Verify state was persisted in Redis
        request_id = result["request_id"]
        state_key = f"rag_state:{request_id}"
        
        state_data = redis_client.get(state_key)
        assert state_data is not None
        
    def test_multi_query_expansion(self, api_client):
        """Test that multi-query expansion works correctly."""
        query_data = {
            "query": "database connection issues",
            "use_multi_query": True,
            "top_k": 3
        }
        
        response = api_client.post(
            f"{API_BASE_URL}/rag/query",
            json=query_data
        )
        
        assert response.status_code == 200
        result = response.json()
        
        # Check if expanded queries were used (this might be in metadata)
        if "metadata" in result and "expanded_queries" in result["metadata"]:
            assert len(result["metadata"]["expanded_queries"]) > 1
            
    def test_hyde_query_enhancement(self, api_client):
        """Test HyDE (Hypothetical Document Embeddings) functionality."""
        query_data = {
            "query": "application slow response time",
            "use_hyde": True,
            "top_k": 5
        }
        
        response = api_client.post(
            f"{API_BASE_URL}/rag/query",
            json=query_data
        )
        
        assert response.status_code == 200
        result = response.json()
        
        # Verify enhanced retrieval results
        assert len(result["sources"]) > 0
        
        # Check if any sources mention performance issues
        performance_related = any(
            "performance" in source["content"].lower() or 
            "response time" in source["content"].lower()
            for source in result["sources"]
        )
        assert performance_related
        
    def test_concurrent_requests(self, api_client):
        """Test system behavior under concurrent load."""
        import concurrent.futures
        
        def make_request(query: str) -> Dict[str, Any]:
            response = api_client.post(
                f"{API_BASE_URL}/rag/query",
                json={"query": query}
            )
            return response.json() if response.status_code == 200 else None
            
        queries = [
            "Kubernetes pod crash",
            "Database connection timeout",
            "High CPU usage alert",
            "Memory leak detection",
            "Network latency issues"
        ]
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(make_request, query) for query in queries]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]
            
        # Verify all requests succeeded
        assert all(result is not None for result in results)
        
        # Verify each request has unique request_id
        request_ids = [r["request_id"] for r in results if r]
        assert len(request_ids) == len(set(request_ids))
        
    def test_error_handling(self, api_client):
        """Test API error handling for invalid requests."""
        # Test empty query
        response = api_client.post(
            f"{API_BASE_URL}/rag/query",
            json={"query": ""}
        )
        assert response.status_code == 422  # Validation error
        
        # Test oversized query
        response = api_client.post(
            f"{API_BASE_URL}/rag/query",
            json={"query": "x" * 2000}  # Exceeds max length
        )
        assert response.status_code == 422
        
        # Test invalid parameters
        response = api_client.post(
            f"{API_BASE_URL}/rag/query",
            json={
                "query": "test query",
                "top_k": 100  # Exceeds limit
            }
        )
        assert response.status_code == 422
        
    def test_observability_metrics(self, api_client):
        """Test that observability metrics are being collected."""
        # Make a few requests to generate metrics
        for i in range(3):
            api_client.post(
                f"{API_BASE_URL}/rag/query",
                json={"query": f"test query {i}"}
            )
            
        # Check Prometheus metrics endpoint
        response = api_client.get(f"{API_BASE_URL}/metrics")
        assert response.status_code == 200
        
        metrics_text = response.text
        
        # Verify key metrics are present
        assert "request_duration_seconds" in metrics_text
        assert "request_count" in metrics_text
        assert "rag_query" in metrics_text
        
    def test_vector_search_accuracy(self, api_client, opensearch_client):
        """Test vector search returns relevant results."""
        # First, ensure we have some test data indexed
        test_data = [
            {
                "content": "Redis cluster failover occurred. Master node became unreachable. Sentinel promoted replica to master.",
                "metadata": {"source": "redis-logs", "component": "redis"}
            },
            {
                "content": "PostgreSQL replication lag increased to 5 minutes. Replica falling behind master due to heavy write load.",
                "metadata": {"source": "postgres-logs", "component": "postgresql"}
            },
            {
                "content": "Nginx rate limiting triggered. Too many requests from single IP address. 429 responses being sent.",
                "metadata": {"source": "nginx-logs", "component": "nginx"}
            }
        ]
        
        # Index test data
        api_client.post(
            f"{API_BASE_URL}/documents/index",
            json={"documents": test_data}
        )
        time.sleep(2)  # Wait for indexing
        
        # Test specific query
        response = api_client.post(
            f"{API_BASE_URL}/rag/query",
            json={
                "query": "Redis replication issues",
                "top_k": 3
            }
        )
        
        assert response.status_code == 200
        result = response.json()
        
        # Verify Redis-related content is prioritized
        redis_found = False
        for source in result["sources"]:
            if "redis" in source["content"].lower():
                redis_found = True
                # Check relevance score is high
                assert source["relevance_score"] > 0.7
                break
                
        assert redis_found, "Redis-related content should be found for Redis query"
        
    def test_state_persistence_and_recovery(self, api_client, redis_client):
        """Test that conversation state persists and can be recovered."""
        # Initial query
        response1 = api_client.post(
            f"{API_BASE_URL}/rag/query",
            json={
                "query": "What causes high memory usage?",
                "session_id": "test-session-123"
            }
        )
        assert response1.status_code == 200
        
        # Follow-up query in same session
        response2 = api_client.post(
            f"{API_BASE_URL}/rag/query",
            json={
                "query": "How can I fix it?",
                "session_id": "test-session-123"
            }
        )
        assert response2.status_code == 200
        
        result2 = response2.json()
        
        # Verify context was maintained
        # The answer should reference memory-related solutions
        assert "memory" in result2["answer"].lower() or "ram" in result2["answer"].lower()
        
        # Verify session state in Redis
        session_state = redis_client.get("session:test-session-123")
        assert session_state is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])