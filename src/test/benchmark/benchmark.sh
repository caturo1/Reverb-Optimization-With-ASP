#!/bin/bash

MAX_DURATION=10
MAX_FREQ=20000
N_RUNS=20

# set up the directory structure
BENCHMARK_DIR="./benchmark"
SIGNAL_DIR="$BENCHMARK_DIR/signals"
AUDIO_DIR="$BENCHMARK_DIR/audio"
RESULT_DIR="$BENCHMARK_DIR/results"

mkdir -p "$BENCHMARK_DIR"
mkdir -p "$SIGNAL_DIR"
mkdir -p "$AUDIO_DIR"
mkdir -p "$RESULT_DIR"

echo "Starting benchmark..."

# Warmup runs
for i in {1..3}; do
    echo "Warmup run $i/3"
    python3 -m src.test signal_generator.py 1000 1 2 0 "${SIGNAL_DIR}/warmup$i.wav"
    python3 -m src.application --audio-file="warmup$i.wav"
done
rm "${SIGNAL_DIR}"/warmup*.wav

# Main benchmark
for ((i=1; i<=N_RUNS; i++)); do
    freq=$(( 20 + (i * MAX_FREQ / N_RUNS) ))
    duration=$(( 1 + (i * MAX_DURATION / N_RUNS) ))
  
    echo "Run $i/$N_RUNS - Frequency: $freq Hz, Duration: $duration s"
    python3 -m src.test $freq $duration 2 0 "${SIGNAL_DIR}/test$i.wav"
    python3 -m src.application --audio-file="test$i.wav"
    mv stats.json "${RESULT_DIR}/stats_test$i.json"
done

# Analysis
python3 analyze_results.py