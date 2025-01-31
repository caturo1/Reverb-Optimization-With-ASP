#!/bin/bash -

MAX_DURATION=10
MAX_FREQ=20000
N_RUNS=20
PI=3.141

# set up the directory structure
BENCHMARK_DIR="./benchmark"
AUDIO_DIR="../data"
SIGNAL_DIR="$BENCHMARK_DIR/signals"
AUDIO_DIR="$AUDIO_DIR/testing"
RESULT_DIR="$BENCHMARK_DIR/results"

mkdir -p "$BENCHMARK_DIR"
mkdir -p "$SIGNAL_DIR"
mkdir -p "$AUDIO_DIR"
mkdir -p "$RESULT_DIR"

echo "Starting benchmark..."

# Warmup runs to set up data structures and
# python cache
for i in {1..3}
do
    echo "Warmup run $i/3"
    python3 -m ./benchmark/signal_generator.py 1000 1 2 0 "${SIGNAL_DIR}/warmup$i.wav"
    python3 -m src.application --audio-file="warmup$i.wav"
done
rm "${SIGNAL_DIR}"/warmup*.wav

# Main benchmark
# run over frequency ranges
# with different 
for ((i=1; i<=N_RUNS; i++))
do
    freq=$(( 20 + (i * MAX_FREQ / N_RUNS) ))
    duration=$(( 1 + (i * MAX_DURATION / N_RUNS) ))
    # the larger i the smaller the phase offset
    phase_offset=($PI / $i)
    # convert radians to degree for clarity in stats
    deg=$($phase_offset * (180 / $PI))
    # still experimental
    mf=($freq / 10)
    md=$i / 10
    name="${SIGNAL_DIR}/test$i\_$freq\_Hz\_$duration\s.wav"
    echo "Run $i out of $N_RUNS - Frequency: $freq Hz, Duration: $duration s"
    python3 ./signal_generator.py -f=$freq -d=$duration -p=$phase_offset -mf=$mf -mf=$md -name=$name
    # write values to 
    python3 -m src.application --audio-file="test$i.wav" > /dev/null
    mv ../stats.json "${RESULT_DIR}/stats_test$i.json"
done

# Analysis
python3 ./benchmark/analyze_results.py