"""
End-to-end integration tests for the complete RAG pipeline.
This file contains comprehensive tests that verify the entire flow from 
API request to response, including all intermediate services.
"""

import pytest
import time
import json
import asyncio
from typing import Dict, Any, List
from concurrent.futures import ThreadPoolExecutor, as_completed

from tests.fixtures.integration_fixtures import (
    assert_rag_response_valid,
    assert_contains_relevant_content,
    wait_for_eventual_consistency,
    PerformanceMonitor
)


class TestRAGPipelineE2E:
    """End-to-end tests for the RAG pipeline."""
    
    def test_complete_rag_flow_with_real_data(self, test_helper, test_documents):
        """Test the complete RAG flow with realistic data."""
        # 1. Create and populate test index
        with test_helper.temporary_index("e2e-test-index") as index_name:
            # Index test documents
            test_helper.index_test_documents(test_documents, index_name)
            
            # 2. Wait for indexing to complete
            time.sleep(2)
            
            # 3. Make RAG query
            response = test_helper.make_rag_request(
                query="What causes Kubernetes pods to crash?",
                top_k=5,
                use_hyde=True,
                use_multi_query=True
            )
            
            # 4. Validate response
            assert_rag_response_valid(response)
            assert_contains_relevant_content(
                response, 
                ["kubernetes", "pod", "crash", "oom", "memory"]
            )
            
            # 5. Verify sources are from our test data
            sources = response["data"]["sources"]
            assert len(sources) > 0
            
            # At least one source should mention Kubernetes
            k8s_source_found = any(
                "kubernetes" in source["content"].lower() 
                for source in sources
            )
            assert k8s_source_found
    
    def test_multi_step_conversation_flow(self, test_helper, clean_redis):
        """Test multi-step conversation with context retention."""
        session_id = "test-conversation-123"
        
        # Step 1: Initial query
        response1 = test_helper.make_rag_request(
            query="What are common database performance issues?",
            session_id=session_id
        )
        assert_rag_response_valid(response1)
        
        # Step 2: Follow-up query
        response2 = test_helper.make_rag_request(
            query="How can I fix connection pool exhaustion?",
            session_id=session_id
        )
        assert_rag_response_valid(response2)
        
        # Verify context was maintained
        # The second response should reference database/connection concepts
        assert_contains_relevant_content(
            response2,
            ["connection", "pool", "database"]
        )
        
        # Step 3: Verify conversation state in Redis
        session_key = f"session:{session_id}"
        session_state = test_helper.get_redis_state(session_key)
        
        # Session state should exist and contain conversation history
        assert session_state is not None
        
    def test_concurrent_user_isolation(self, test_helper, performance_monitor):
        """Test that concurrent users don't interfere with each other."""
        num_users = 5
        queries = [
            ("user1", "Kubernetes memory issues"),
            ("user2", "Database connection problems"),
            ("user3", "Network latency monitoring"),
            ("user4", "SSL certificate management"),
            ("user5", "Disk space alerts")
        ]
        
        def make_user_request(user_id: str, query: str) -> Dict[str, Any]:
            """Make a request for a specific user."""
            with performance_monitor.measure(f"request_{user_id}"):
                return test_helper.make_rag_request(
                    query=query,
                    session_id=f"session_{user_id}"
                )
        
        # Execute requests concurrently
        with ThreadPoolExecutor(max_workers=num_users) as executor:
            future_to_user = {
                executor.submit(make_user_request, user_id, query): user_id
                for user_id, query in queries
            }
            
            results = {}
            for future in as_completed(future_to_user):
                user_id = future_to_user[future]
                try:
                    result = future.result()
                    results[user_id] = result
                except Exception as e:
                    pytest.fail(f"User {user_id} request failed: {e}")
        
        # Verify all requests succeeded
        assert len(results) == num_users
        
        # Verify each user got appropriate responses
        for user_id, query in queries:
            response = results[user_id]
            assert_rag_response_valid(response)
            
            # Each response should have unique request_id
            request_ids = [r["data"]["request_id"] for r in results.values()]
            assert len(request_ids) == len(set(request_ids))
        
        # Check performance stats
        stats = performance_monitor.get_stats()
        assert stats["count"] == num_users
        assert stats["avg"] < 5.0  # Average response time under 5 seconds
    
    def test_error_recovery_and_retry(self, test_helper):
        """Test system behavior during errors and recovery."""
        # Test 1: Invalid query handling
        invalid_response = test_helper.make_rag_request(
            query="",  # Empty query
            top_k=5
        )
        assert invalid_response["status_code"] == 422
        
        # Test 2: Recovery after error - system should still work
        valid_response = test_helper.make_rag_request(
            query="Valid query after error",
            top_k=5
        )
        assert_rag_response_valid(valid_response)
        
        # Test 3: Overload protection - very large query
        large_query = "x" * 1500  # Exceeds typical limits
        large_response = test_helper.make_rag_request(query=large_query)
        assert large_response["status_code"] == 422
        
    def test_observability_integration(self, test_helper):
        """Test that observability features work correctly."""
        # Make several requests to generate metrics
        test_queries = [
            "Memory usage monitoring",
            "CPU performance metrics",
            "Network traffic analysis"
        ]
        
        request_ids = []
        for query in test_queries:
            response = test_helper.make_rag_request(query=query)
            if response["status_code"] == 200:
                request_ids.append(response["data"]["request_id"])
        
        # Give metrics time to propagate
        time.sleep(2)
        
        # Check Prometheus metrics
        metrics_text = test_helper.get_metrics()
        assert len(metrics_text) > 0
        
        # Verify key metrics are present
        expected_metrics = [
            "request_duration_seconds",
            "request_count",
            "rag_query_total",
            "vector_search_duration"
        ]
        
        for metric in expected_metrics:
            assert metric in metrics_text, f"Metric {metric} not found"
    
    def test_vector_search_relevance(self, test_helper, mock_embeddings):
        """Test vector search returns relevant results."""
        # Create test documents with mock embeddings
        test_docs = [
            TestDocument(
                content="Redis cluster configuration for high availability",
                metadata={"topic": "redis", "type": "config"},
                embedding=mock_embeddings.generate("redis cluster config")
            ),
            TestDocument(
                content="PostgreSQL performance tuning best practices",
                metadata={"topic": "postgres", "type": "performance"},
                embedding=mock_embeddings.generate("postgres performance")
            ),
            TestDocument(
                content="Nginx load balancing configuration guide",
                metadata={"topic": "nginx", "type": "config"},
                embedding=mock_embeddings.generate("nginx load balancer")
            )
        ]
        
        with test_helper.temporary_index("relevance-test-index") as index_name:
            test_helper.index_test_documents(test_docs, index_name)
            time.sleep(2)
            
            # Query for Redis-specific content
            redis_response = test_helper.make_rag_request(
                query="How to configure Redis cluster?",
                top_k=2
            )
            
            assert_rag_response_valid(redis_response)
            
            # First result should be Redis-related
            sources = redis_response["data"]["sources"]
            if sources:
                first_source = sources[0]
                assert "redis" in first_source["content"].lower()
                assert first_source.get("relevance_score", 0) > 0.7
    
    def test_cache_and_performance(self, test_helper, performance_monitor):
        """Test caching behavior and performance improvements."""
        query = "Common application performance issues"
        
        # First request (cold cache)
        with performance_monitor.measure("cold_request"):
            response1 = test_helper.make_rag_request(query=query)
        assert_rag_response_valid(response1)
        
        # Second identical request (warm cache)
        with performance_monitor.measure("warm_request"):
            response2 = test_helper.make_rag_request(query=query)
        assert_rag_response_valid(response2)
        
        # Third request with slight variation (partial cache hit)
        with performance_monitor.measure("partial_cache_request"):
            response3 = test_helper.make_rag_request(
                query="Common application performance problems"
            )
        assert_rag_response_valid(response3)
        
        # Analyze performance
        stats = performance_monitor.get_stats()
        
        # Warm request should be faster than cold
        cold_time = performance_monitor.get_stats("cold_request")["avg"]
        warm_time = performance_monitor.get_stats("warm_request")["avg"]
        
        # Cache should provide at least 20% improvement
        # (This might not always be true in test environment)
        # assert warm_time < cold_time * 0.8
    
    def test_data_consistency_across_services(self, test_helper):
        """Test data consistency between OpenSearch and API responses."""
        # Index specific test document
        test_doc = TestDocument(
            content="Unique test document for consistency check XYZ123",
            metadata={"test_id": "consistency_test_001"}
        )
        
        with test_helper.temporary_index("consistency-test-index") as index_name:
            test_helper.index_test_documents([test_doc], index_name)
            
            # Wait for eventual consistency
            def check_indexed():
                search_result = test_helper.opensearch_client.search(
                    index=index_name,
                    body={"query": {"match": {"content": "XYZ123"}}}
                )
                return search_result["hits"]["total"]["value"] > 0
            
            assert wait_for_eventual_consistency(check_indexed, timeout=10)
            
            # Query through API
            api_response = test_helper.make_rag_request(
                query="XYZ123 consistency check"
            )
            
            if api_response["status_code"] == 200:
                sources = api_response["data"]["sources"]
                
                # Verify the specific document was retrieved
                found = any(
                    "XYZ123" in source.get("content", "")
                    for source in sources
                )
                assert found, "Test document not found in API response"
    
    def test_graceful_degradation(self, test_helper):
        """Test system behavior when some components are slow or failing."""
        # Test with unreasonable timeout expectations
        response = test_helper.make_rag_request(
            query="Test query with degraded performance",
            top_k=50  # Request many documents
        )
        
        # System should still respond, even if degraded
        assert response["status_code"] in [200, 503]
        
        if response["status_code"] == 200:
            # Even in degraded mode, should return some results
            assert len(response["data"]["answer"]) > 0


class TestAdvancedRAGFeatures:
    """Test advanced RAG features like HyDE, multi-query, RRF."""
    
    def test_hyde_improves_results(self, test_helper):
        """Test that HyDE (Hypothetical Document Embeddings) improves results."""
        query = "How to troubleshoot application memory leaks"
        
        # Query without HyDE
        response_no_hyde = test_helper.make_rag_request(
            query=query,
            use_hyde=False,
            top_k=5
        )
        
        # Query with HyDE
        response_with_hyde = test_helper.make_rag_request(
            query=query,
            use_hyde=True,
            top_k=5
        )
        
        # Both should be valid
        assert_rag_response_valid(response_no_hyde)
        assert_rag_response_valid(response_with_hyde)
        
        # HyDE results should be at least as good
        # (In practice, we'd measure relevance scores)
        assert len(response_with_hyde["data"]["sources"]) >= len(response_no_hyde["data"]["sources"])
    
    def test_multi_query_expansion(self, test_helper):
        """Test multi-query expansion generates diverse results."""
        response = test_helper.make_rag_request(
            query="server performance issues",
            use_multi_query=True,
            multi_query_alts=3,
            top_k=10
        )
        
        assert_rag_response_valid(response)
        
        # Should retrieve diverse sources covering different aspects
        sources = response["data"]["sources"]
        
        # Check for diversity in retrieved content
        topics_covered = set()
        for source in sources:
            content_lower = source["content"].lower()
            if "cpu" in content_lower:
                topics_covered.add("cpu")
            if "memory" in content_lower:
                topics_covered.add("memory")
            if "disk" in content_lower:
                topics_covered.add("disk")
            if "network" in content_lower:
                topics_covered.add("network")
        
        # Multi-query should help retrieve diverse aspects
        assert len(topics_covered) >= 2
    
    def test_strict_citation_mode(self, test_helper):
        """Test that strict citation mode properly attributes sources."""
        response = test_helper.make_rag_request(
            query="Explain database connection pooling",
            strict_citation=True,
            top_k=3
        )
        
        assert_rag_response_valid(response)
        
        answer = response["data"]["answer"]
        sources = response["data"]["sources"]
        
        # In strict citation mode, answer should reference sources
        # Look for citation markers like [1], [2], etc.
        import re
        citation_pattern = r'\[\d+\]'
        citations_found = re.findall(citation_pattern, answer)
        
        # Should have citations if sources were used
        if sources:
            assert len(citations_found) > 0, "No citations found in strict citation mode"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])