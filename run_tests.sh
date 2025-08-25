#!/bin/bash
# Script to run tests without requiring google_api_key

# Set environment variables for testing
export TESTING=true
export GEMINI_API_KEY=test-api-key

# Set minimum coverage threshold
MIN_COVERAGE=${MIN_COVERAGE:-50}

# Run all tests with coverage
echo "Running all tests with minimum coverage requirement: ${MIN_COVERAGE}%..."
python3 -m pytest tests/ \
    --cov=src \
    --cov-report=term-missing \
    --cov-fail-under=${MIN_COVERAGE} \
    -v

# Exit with the same code as pytest
exit $?