#!/bin/bash

# Configuration
OUTPUT_DIR="./benchmark/results"
DATA_DIR="./data/testing"
PROCESSED_DIR="./processed_data"

DURATION_DIR="$DATA_DIR/duration_tests"
AMP_DIR="$DATA_DIR/amplitude_tests"
FREQ_DIR="$DATA_DIR/frequency_tests"
COMP_DIR="$DATA_DIR/complexity_tests"

# Ensure directories exist
mkdir -p "$OUTPUT_DIR"
mkdir -p "$DATA_DIR"
mkdir -p "$PROCESSED_DIR"

# Clean previous test data
rm -f "$DATA_DIR"/*.wav
rm -f "$OUTPUT_DIR"/*.json
rm -f "$PROCESSED_DIR"/*.wav

# Run each test case
echo "Running duration tests..."
python -m benchmark.run_test_set --test_case duration_test --output "$OUTPUT_DIR/duration_test.json"

echo "Running amplitude tests..."
python -m benchmark.run_test_set --test_case amplitude_test --output "$OUTPUT_DIR/amplitude_test.json"

echo "Running frequency tests..."
python -m benchmark.run_test_set --test_case frequency_test --output "$OUTPUT_DIR/frequency_test.json"

echo "Running complex signal tests..."
python -m benchmark.run_test_set --test_case complex_signal_mod_freq_test --output "$OUTPUT_DIR/complex_test.json"

# Generate analysis
echo "Analyzing results..."
python -m benchmark.analyze_results --input "$OUTPUT_DIR" --output "$OUTPUT_DIR/analysis"