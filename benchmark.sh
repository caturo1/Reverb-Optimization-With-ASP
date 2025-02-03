#!/bin/bash -

MAX_DURATION=10
MAX_FREQ=2000
current=110
N_RUNS=20
PI=3.141

# set up the directory structure
BASE_DIR="."  # Add this to make paths clearer
AUDIO_DIR="$BASE_DIR/data"
RESULT_DIR="$BASE_DIR/benchmark/results"
#RESULT_DIR_CLINGO="$RESULT_DIR/clingo"
#RESULT_DIR_SIGNALS="$RESULT_DIR/signals"
SIGNAL_DIR="$AUDIO_DIR/testing"

mkdir -p "$RESULT_DIR"
#mkdir -p "$RESULT_DIR_CLINGO"
#mkdir -p "$RESULT_DIR_SIGNALS"

echo "Starting benchmark..."

# Warmup runs
for i in {1..3}
do
    echo "Warmup run $i/3"
    freq=$((100 * $i))
    duration=$((10 / $i))
    
    python -m benchmark \
        -f=$freq \
        -d=$duration \
        -mf=2 \
        -md=1 \
        -n="warmup$i" \
        -r=$i
    
    python -m src.application --audio-file="testing/warmup$i.wav"
    rm ./clingo_stats.json

done

rm "$RESULT_DIR"/audio_test*.json
rm "$SIGNAL_DIR"/warmup*.wav
#rm "$RESULT_DIR_CLINGO"/*.json
#rm "$RESULT_DIR_SIGNALS"/*.json

# Main benchmark
# using a simple waveform (no FM synthesis)
mode="True"
loop=1

while (($loop==1))
do
    freq=$(($current * 2))

    #duration=$(( 1 + (i * $MAX_DURATION / $N_RUNS) ))
    mf=$freq
    md=$i
    name="${mode}_${freq}Hz_${duration}s_${i}run"
    echo $name
    echo "Run $i out of $N_RUNS - Frequency: $freq Hz, Duration: $duration s"
    
    # Generate signal
    python -m benchmark \
        -f=$freq \
        -d=$duration \
        -mf=$mf \
        -md=$md \
        -n="$name" \
        -r=$i
    
    sed -i '$d' "$RESULT_DIR/audio_test_${i}.json"
    sed -i '$ s/$/,/' "$RESULT_DIR/audio_test_${i}.json"

    # Process with application
    python -m src.application --audio-file="testing/${name}.wav" > /dev/null
    
    # Append the clingo stats to the audio stats
    sed '1d' ./clingo_stats.json >> "$RESULT_DIR/audio_test_${i}.json"

    $current=$freq
    
    if (($current > 4000)):
        loop=0

done

# using a complex signal (FM)
mode="complex"
for ((i=1; i<=$N_RUNS; i++))
do
    freq=$(($current * 2))
    
    #duration=$(( 1 + (i * $MAX_DURATION / $N_RUNS) ))
    mf=$freq
    md=$i
    name="${mode}_${freq}Hz_${duration}s_${i}run"
    echo $name
    echo "Run $i out of $N_RUNS - Frequency: $freq Hz, Duration: $duration s"
    
    # Generate signal
    python -m benchmark \
        -f=$freq \
        -d=$duration \
        -mf=$mf \
        -md=$md \
        -n="$name" \
        -r=$i
    
    sed -i '$d' "$RESULT_DIR/audio_test_${i}.json"
    sed -i '$ s/$/,/' "$RESULT_DIR/audio_test_${i}.json"

    # Process with application
    python -m src.application --audio-file="testing/${name}.wav" > /dev/null
    
    # Append the clingo stats to the audio stats
    sed '1d' ./clingo_stats.json >> "$RESULT_DIR/audio_test_${i}.json"

    $current=$freq 

done

# Analysis
python ./benchmark/analyze_results.py