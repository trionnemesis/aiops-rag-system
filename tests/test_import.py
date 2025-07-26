"""Test module imports to ensure they work correctly."""

def test_can_import_main():
    """Test that we can import the main module."""
    from src.main import app
    assert app is not None

def test_can_import_services():
    """Test that we can import service modules."""
    from src.services import gemini_service
    from src.services import opensearch_service
    from src.services import prometheus_service
    from src.services import rag_service
    assert True  # If we get here, imports worked

def test_can_import_models():
    """Test that we can import model modules."""
    from src.models import schemas
    assert True  # If we get here, imports worked

def test_can_import_utils():
    """Test that we can import util modules."""
    from src.utils import prompts
    assert True  # If we get here, imports worked