#!/bin/bash
# Script to run integration tests locally

set -e

echo "=== Running Integration Tests ==="
echo "This script will start the test environment and run integration tests."
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

print_error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

# Check if docker and docker-compose are installed
check_dependencies() {
    print_status "Checking dependencies..."
    
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    if ! command -v docker-compose &> /dev/null; then
        print_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
    
    print_status "Dependencies check passed."
}

# Start test environment
start_test_env() {
    print_status "Starting test environment..."
    
    # Stop any existing containers
    docker-compose -f docker-compose.test.yml down -v 2>/dev/null || true
    
    # Build images
    print_status "Building Docker images..."
    docker-compose -f docker-compose.test.yml build --parallel
    
    # Start services
    print_status "Starting services..."
    docker-compose -f docker-compose.test.yml up -d
    
    # Wait for services to be healthy
    print_status "Waiting for services to be healthy..."
    local max_attempts=30
    local attempt=0
    
    while [ $attempt -lt $max_attempts ]; do
        if docker-compose -f docker-compose.test.yml ps | grep -q "unhealthy\|starting"; then
            echo -n "."
            sleep 5
            attempt=$((attempt + 1))
        else
            echo ""
            print_status "All services are healthy!"
            break
        fi
    done
    
    if [ $attempt -eq $max_attempts ]; then
        print_error "Services failed to become healthy within timeout."
        docker-compose -f docker-compose.test.yml ps
        exit 1
    fi
}

# Run tests
run_tests() {
    print_status "Running integration tests..."
    
    # Install test dependencies
    pip install -q pytest pytest-asyncio pytest-timeout docker opensearch-py redis requests
    
    # Run different test suites
    local test_suites=(
        "tests/test_integration.py"
        "tests/test_rag_integration_e2e.py"
    )
    
    local all_passed=true
    
    for test_suite in "${test_suites[@]}"; do
        if [ -f "$test_suite" ]; then
            print_status "Running $test_suite..."
            
            if pytest "$test_suite" -v -s --tb=short; then
                print_status "✓ $test_suite passed"
            else
                print_error "✗ $test_suite failed"
                all_passed=false
            fi
        else
            print_warning "Test suite $test_suite not found, skipping..."
        fi
    done
    
    if [ "$all_passed" = true ]; then
        print_status "All integration tests passed!"
        return 0
    else
        print_error "Some integration tests failed."
        return 1
    fi
}

# Cleanup
cleanup() {
    print_status "Cleaning up test environment..."
    
    # Collect logs if tests failed
    if [ "$1" != "0" ]; then
        print_status "Collecting logs..."
        mkdir -p test-artifacts/logs
        
        for container in $(docker-compose -f docker-compose.test.yml ps -q); do
            container_name=$(docker inspect -f '{{.Name}}' $container | sed 's/\///')
            docker logs $container > "test-artifacts/logs/${container_name}.log" 2>&1
        done
        
        print_warning "Logs collected in test-artifacts/logs/"
    fi
    
    # Stop and remove containers
    docker-compose -f docker-compose.test.yml down -v
    
    print_status "Cleanup completed."
}

# Main execution
main() {
    # Trap to ensure cleanup runs
    trap 'cleanup $?' EXIT
    
    print_status "Starting integration test run..."
    
    # Check dependencies
    check_dependencies
    
    # Start test environment
    start_test_env
    
    # Run tests
    run_tests
    
    # Exit with test result
    exit $?
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --keep-env)
            # Don't cleanup after tests
            trap - EXIT
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --keep-env    Keep test environment running after tests"
            echo "  --help        Show this help message"
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Run main function
main