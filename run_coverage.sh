#!/bin/bash
# Script to run test coverage report

echo "========================================"
echo "Running Test Coverage Report"
echo "========================================"

# Set environment variables for testing
export TESTING=true
export GEMINI_API_KEY=test-api-key

# Run tests with coverage
echo "Running pytest with coverage..."
python3 -m pytest tests/ -v --cov=. --cov-report=term-missing --cov-report=html --cov-report=xml

# Check if coverage report was generated
if [ -f .coverage ]; then
    echo ""
    echo "========================================"
    echo "Coverage Summary:"
    echo "========================================"
    python3 -m coverage report --show-missing | head -20
    
    echo ""
    echo "Full report available in:"
    echo "- Terminal: Run 'python3 -m coverage report'"
    echo "- HTML: Open htmlcov/index.html in browser"
    echo "- Detailed analysis: See coverage_report.md"
else
    echo "Coverage report generation failed!"
    exit 1
fi

# Display current coverage percentage
echo ""
echo "========================================"
COVERAGE_PCT=$(python3 -m coverage report | grep TOTAL | awk '{print $4}')
echo "Current Total Coverage: $COVERAGE_PCT"
echo "Target Coverage: 85%+"
echo "========================================"