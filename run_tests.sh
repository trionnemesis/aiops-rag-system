#!/bin/bash
# Script to run tests without requiring google_api_key
set -e

# Set environment variables for testing
export TESTING=true
export GEMINI_API_KEY=test-api-key

# Create and activate virtual environment
VENV_DIR=".venv"
if [ ! -d "$VENV_DIR" ]; then
  python3 -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"

# Ensure dependencies for testing are installed
python -m pip install --upgrade pip
python -m pip install -r requirements.txt pytest pytest-asyncio pytest-cov httpx

# Run all tests
echo "Running all tests..."
python -m pytest tests/ -v

# Exit with the same code as pytest
exit $?