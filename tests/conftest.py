"""
Test configuration file for pytest
Sets up environment variables and common fixtures
"""
import os
import pytest

# Set TESTING environment variable before importing any modules
os.environ['TESTING'] = 'true'

# Ensure settings use test values
os.environ['GEMINI_API_KEY'] = 'test-api-key'

@pytest.fixture(autouse=True)
def set_test_environment():
    """Automatically set test environment for all tests"""
    # Ensure TESTING flag is set
    os.environ['TESTING'] = 'true'
    yield
    # Clean up after test if needed