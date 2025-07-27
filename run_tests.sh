#!/bin/bash
# Script to run tests without requiring google_api_key

# Set environment variables for testing
export TESTING=true
export GEMINI_API_KEY=test-api-key

# Run all tests
echo "Running all tests..."
python3 -m pytest tests/ -v

# Exit with the same code as pytest
exit $?