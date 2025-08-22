#!/usr/bin/env python3
"""
Initialize test data for integration testing.
This script sets up sample documents in OpenSearch for testing the RAG pipeline.
"""

import time
import json
from opensearchpy import OpenSearch
import requests
from typing import List, Dict, Any

# Configuration
OPENSEARCH_HOST = "opensearch-test"
OPENSEARCH_PORT = 9200
API_URL = "http://app-test:8000"

# Test data
TEST_DOCUMENTS = [
    {
        "content": "Kubernetes pod OOMKilled error occurred. The container exceeded its memory limit of 1Gi. Current usage was 1.5Gi. Consider increasing memory limits or optimizing application memory usage.",
        "metadata": {
            "source": "k8s-logs",
            "timestamp": "2024-01-15T10:30:00Z",
            "severity": "error",
            "component": "kubernetes",
            "cluster": "prod-cluster-1"
        }
    },
    {
        "content": "Database connection pool exhausted. Maximum connections (100) reached. Active connections: 100, Idle: 0. Consider increasing pool size or optimizing connection usage.",
        "metadata": {
            "source": "app-logs",
            "timestamp": "2024-01-15T10:35:00Z",
            "severity": "error",
            "component": "database",
            "database": "postgresql"
        }
    },
    {
        "content": "Redis cluster failover initiated. Master node 192.168.1.10:6379 became unreachable. Sentinel promoted replica 192.168.1.11:6379 to master. Cluster is now healthy.",
        "metadata": {
            "source": "redis-logs",
            "timestamp": "2024-01-15T10:40:00Z",
            "severity": "warning",
            "component": "redis",
            "action": "failover"
        }
    },
    {
        "content": "High CPU usage detected on node worker-03. CPU utilization: 95%. Top processes: java (45%), python (30%), nginx (20%). Consider scaling horizontally or optimizing workloads.",
        "metadata": {
            "source": "metrics",
            "timestamp": "2024-01-15T10:45:00Z",
            "severity": "warning",
            "component": "infrastructure",
            "node": "worker-03"
        }
    },
    {
        "content": "Application response time degraded. P95 latency increased from 200ms to 2000ms. Database query time accounts for 80% of request time. Review slow queries and add indexes.",
        "metadata": {
            "source": "apm",
            "timestamp": "2024-01-15T10:50:00Z",
            "severity": "warning",
            "component": "application",
            "service": "api-gateway"
        }
    },
    {
        "content": "SSL certificate expiring in 7 days for domain api.example.com. Certificate fingerprint: SHA256:1234567890abcdef. Renew certificate to avoid service disruption.",
        "metadata": {
            "source": "cert-monitor",
            "timestamp": "2024-01-15T11:00:00Z",
            "severity": "warning",
            "component": "security",
            "domain": "api.example.com"
        }
    },
    {
        "content": "Disk space critical on /var/log. Usage: 95% (95GB/100GB). Log rotation failed due to insufficient space. Clean up old logs or increase disk size.",
        "metadata": {
            "source": "system-logs",
            "timestamp": "2024-01-15T11:05:00Z",
            "severity": "critical",
            "component": "storage",
            "mount": "/var/log"
        }
    },
    {
        "content": "Network latency spike detected between regions. us-east to eu-west latency increased from 50ms to 500ms. Possible network congestion or routing issue.",
        "metadata": {
            "source": "network-monitor",
            "timestamp": "2024-01-15T11:10:00Z",
            "severity": "warning",
            "component": "network",
            "regions": ["us-east", "eu-west"]
        }
    }
]


def wait_for_opensearch():
    """Wait for OpenSearch to be ready."""
    client = OpenSearch(
        hosts=[{'host': OPENSEARCH_HOST, 'port': OPENSEARCH_PORT}],
        http_compress=True,
        use_ssl=False,
        verify_certs=False,
        ssl_assert_hostname=False,
        ssl_show_warn=False,
    )
    
    max_retries = 30
    for i in range(max_retries):
        try:
            health = client.cluster.health()
            if health['status'] in ['green', 'yellow']:
                print(f"OpenSearch is ready. Status: {health['status']}")
                return client
        except Exception as e:
            print(f"Waiting for OpenSearch... ({i+1}/{max_retries})")
            time.sleep(2)
    
    raise Exception("OpenSearch failed to become ready")


def wait_for_api():
    """Wait for the API to be ready."""
    max_retries = 30
    for i in range(max_retries):
        try:
            response = requests.get(f"{API_URL}/health", timeout=5)
            if response.status_code == 200:
                print("API is ready")
                return
        except Exception:
            print(f"Waiting for API... ({i+1}/{max_retries})")
            time.sleep(2)
    
    raise Exception("API failed to become ready")


def create_index(client: OpenSearch):
    """Create the test index with appropriate mappings."""
    index_name = "aiops-docs"
    
    # Check if index already exists
    if client.indices.exists(index=index_name):
        print(f"Index {index_name} already exists, deleting...")
        client.indices.delete(index=index_name)
    
    # Create index with mappings
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
                    "dimension": 768,  # Adjust based on your embedding model
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
    
    client.indices.create(index=index_name, body=index_body)
    print(f"Created index: {index_name}")


def index_documents_via_api():
    """Index documents through the API to test the complete flow."""
    # First, let's check if the API has a document indexing endpoint
    try:
        # Try to index documents via API
        response = requests.post(
            f"{API_URL}/documents/index",
            json={"documents": TEST_DOCUMENTS},
            timeout=30
        )
        
        if response.status_code == 200:
            print(f"Successfully indexed {len(TEST_DOCUMENTS)} documents via API")
            return True
        else:
            print(f"Failed to index documents via API: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"Error indexing documents via API: {e}")
        return False


def index_documents_directly(client: OpenSearch):
    """Index documents directly to OpenSearch (fallback method)."""
    index_name = "aiops-docs"
    
    # Note: In a real scenario, you would generate embeddings here
    # For testing, we'll use dummy embeddings
    dummy_embedding = [0.1] * 768  # 768-dimensional dummy vector
    
    for i, doc in enumerate(TEST_DOCUMENTS):
        doc_with_embedding = {
            "content": doc["content"],
            "metadata": doc["metadata"],
            "embedding": dummy_embedding
        }
        
        client.index(
            index=index_name,
            body=doc_with_embedding,
            id=f"test-doc-{i}",
            refresh=True
        )
    
    print(f"Indexed {len(TEST_DOCUMENTS)} documents directly to OpenSearch")


def verify_data(client: OpenSearch):
    """Verify that test data was indexed correctly."""
    index_name = "aiops-docs"
    
    # Get document count
    count_response = client.count(index=index_name)
    doc_count = count_response["count"]
    print(f"Total documents in index: {doc_count}")
    
    # Sample search
    search_response = client.search(
        index=index_name,
        body={
            "query": {"match": {"content": "kubernetes"}},
            "size": 5
        }
    )
    
    hits = search_response["hits"]["total"]["value"]
    print(f"Sample search for 'kubernetes' returned {hits} hits")
    
    return doc_count > 0


def main():
    """Main initialization routine."""
    print("Starting test data initialization...")
    
    # Wait for services
    print("Waiting for services to be ready...")
    opensearch_client = wait_for_opensearch()
    wait_for_api()
    
    # Create index
    print("Creating OpenSearch index...")
    create_index(opensearch_client)
    
    # Try to index via API first
    print("Attempting to index documents via API...")
    api_indexed = index_documents_via_api()
    
    if not api_indexed:
        # Fallback to direct indexing
        print("Falling back to direct OpenSearch indexing...")
        index_documents_directly(opensearch_client)
    
    # Verify data
    print("Verifying indexed data...")
    if verify_data(opensearch_client):
        print("Test data initialization completed successfully!")
    else:
        print("Warning: Test data verification failed")
        exit(1)


if __name__ == "__main__":
    main()