#!/bin/bash
# Batch testing script for the email processor system
# This script runs multiple test configurations and collects results

# Default settings
BASE_DIR="$(cd "$(dirname "$0")" && pwd)/.."
RESULTS_DIR="$BASE_DIR/test_results"
TEST_SCRIPT="$BASE_DIR/test_large_batch.py"
ANALYZE_SCRIPT="$BASE_DIR/scripts/analyze_results.py"
DEFAULT_EMAILS=1000
WORKER_COUNTS=(1 3 5 10)
BATCH_SIZES=(20 50 100)
RUN_TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Create results directory
mkdir -p "$RESULTS_DIR/$RUN_TIMESTAMP"

# Print header
echo "=========================================="
echo "Email Processor Batch Testing"
echo "Started at: $(date)"
echo "Results directory: $RESULTS_DIR/$RUN_TIMESTAMP"
echo "=========================================="

# Function to run a single test configuration
run_test() {
    worker_count=$1
    batch_size=$2
    email_count=$3
    
    echo ""
    echo "----------------------------------------"
    echo "Running test with configuration:"
    echo "  - Worker count: $worker_count"
    echo "  - Batch size: $batch_size"
    echo "  - Email count: $email_count"
    echo "----------------------------------------"
    
    # Create test directory
    test_dir="$RESULTS_DIR/$RUN_TIMESTAMP/workers${worker_count}_batch${batch_size}_emails${email_count}"
    mkdir -p "$test_dir"
    
    # Configure system
    echo "Configuring system with worker_count=$worker_count, batch_size=$batch_size..."
    redis-cli set "system:config:worker_count" "$worker_count"
    redis-cli set "system:config:batch_size" "$batch_size"
    
    # Run the test
    echo "Running test..."
    start_time=$(date +%s)
    
    # Modify the test script command line to set the email count and output file path
    python "$TEST_SCRIPT" --total-emails "$email_count" --result-file "$test_dir/results.csv" | tee "$test_dir/test_log.txt"
    
    end_time=$(date +%s)
    duration=$((end_time - start_time))
    
    echo "Test completed in $duration seconds"
    
    # Run analysis
    echo "Analyzing results..."
    python "$ANALYZE_SCRIPT" -f "$test_dir/results.csv" -o "$test_dir" --export-json "$test_dir/stats.json" | tee "$test_dir/analysis_log.txt"
    
    # Save test metadata
    cat > "$test_dir/metadata.json" <<EOL
{
  "worker_count": $worker_count,
  "batch_size": $batch_size,
  "email_count": $email_count,
  "duration_seconds": $duration,
  "timestamp": "$(date)",
  "emails_per_second": $(echo "scale=2; $email_count / $duration" | bc)
}
EOL
    
    echo "Test results saved to $test_dir"
    return 0
}

# Process command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --emails|-e)
            DEFAULT_EMAILS="$2"
            shift 2
            ;;
        --workers|-w)
            IFS=',' read -ra WORKER_COUNTS <<< "$2"
            shift 2
            ;;
        --batches|-b)
            IFS=',' read -ra BATCH_SIZES <<< "$2"
            shift 2
            ;;
        --single-test|-s)
            # Format: worker_count,batch_size,email_count
            IFS=',' read -r worker batch emails <<< "$2"
            WORKER_COUNTS=("$worker")
            BATCH_SIZES=("$batch")
            DEFAULT_EMAILS="$emails"
            echo "Running single test configuration: workers=$worker, batch=$batch, emails=$emails"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --emails, -e NUMBER      Number of emails to test (default: $DEFAULT_EMAILS)"
            echo "  --workers, -w LIST       Comma-separated list of worker counts (default: ${WORKER_COUNTS[*]})"
            echo "  --batches, -b LIST       Comma-separated list of batch sizes (default: ${BATCH_SIZES[*]})"
            echo "  --single-test, -s CONFIG Run a single test with specific configuration"
            echo "                           Format: worker_count,batch_size,email_count"
            echo "  --help, -h               Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Run tests with all configurations
total_tests=$((${#WORKER_COUNTS[@]} * ${#BATCH_SIZES[@]}))
current_test=0

echo "Running $total_tests test configurations..."

for worker_count in "${WORKER_COUNTS[@]}"; do
    for batch_size in "${BATCH_SIZES[@]}"; do
        current_test=$((current_test + 1))
        echo ""
        echo "========================================"
        echo "Test $current_test of $total_tests"
        echo "========================================"
        
        run_test "$worker_count" "$batch_size" "$DEFAULT_EMAILS"
    done
done

# Generate summary report
echo ""
echo "=========================================="
echo "Generating summary report..."

# Collect all metrics
echo "Test Configuration,Email Count,Duration (s),Emails/sec,Accuracy,Avg Confidence" > "$RESULTS_DIR/$RUN_TIMESTAMP/summary.csv"

for worker_count in "${WORKER_COUNTS[@]}"; do
    for batch_size in "${BATCH_SIZES[@]}"; do
        test_dir="$RESULTS_DIR/$RUN_TIMESTAMP/workers${worker_count}_batch${batch_size}_emails${DEFAULT_EMAILS}"
        
        if [[ -f "$test_dir/metadata.json" && -f "$test_dir/stats.json" ]]; then
            # Extract metrics
            duration=$(jq -r '.duration_seconds' "$test_dir/metadata.json")
            eps=$(jq -r '.emails_per_second' "$test_dir/metadata.json")
            accuracy=$(jq -r '.overall_accuracy' "$test_dir/stats.json")
            confidence=$(jq -r '.avg_confidence' "$test_dir/stats.json")
            
            # Calculate accuracy percentage
            accuracy_pct=$(echo "scale=2; $accuracy * 100" | bc)
            
            # Add to summary
            echo "Workers: $worker_count, Batch: $batch_size,$DEFAULT_EMAILS,$duration,$eps,$accuracy_pct%,$confidence" >> "$RESULTS_DIR/$RUN_TIMESTAMP/summary.csv"
        fi
    done
done

echo "Summary report generated: $RESULTS_DIR/$RUN_TIMESTAMP/summary.csv"
echo ""
echo "=========================================="
echo "Batch testing completed at: $(date)"
echo "Results available in: $RESULTS_DIR/$RUN_TIMESTAMP"
echo "=========================================="