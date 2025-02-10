#!/bin/bash -

MAX_DURATION=10
MAX_FREQ=10000
# we can use the current variable and double it consistently to iterate over frequencies in octaves
current=110
N_RUNS=10
AMPLITUDES=(
    "(-0.9,0.9)"    # Max dynamic range
    "(-0.5,0.5)"    # Medium dynamic range  
    "(-0.2,0.2)"    # Low dynamic range
    "(-0.9,-0.9)"   # High static
    "(-0.5,-0.5)"   # Medium static
    "(-0.2,-0.2)"   # Low static
)

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
        -dr="(-0.5,0.5)" \
        -r=$i
    
    python -m src.application --audio-file="testing/warmup$i.wav"
    rm ./clingo_stats.json

done

echo "Cleaning warmup files..."

rm "$RESULT_DIR"/audio_test*.json
rm "$SIGNAL_DIR"/warmup*.wav
#rm "$RESULT_DIR_CLINGO"/*.json
#rm "$RESULT_DIR_SIGNALS"/*.json

# Main benchmark
# using a simple waveform (no FM synthesis)
mode=0
loop=1
# we can swap the max frequency check again the upper bound of amplitude

for duration in $(seq 1 2 10)
do
    for amp in "${AMPLITUDES[@]}"
    do
        #duration=$(( 1 + (i * $MAX_DURATION / $N_RUNS) ))
        frequency=440
        mf=$frequency
        md=$duration
        name="${mode}_${amp}Hz_${duration}s_${loop}run"
        
        echo $name
        echo "Run $loop out of 10 - Frequency: $frequency Hz, Duration: $duration s"
        
        # Generate signal
        python -m benchmark \
            -f=$frequency \
            -d=$duration \
            -mf=$mf \
            -md=$md \
            -n="$name" \
            -r=$loop \
            -m=$mode \
        
        sed -i '$d' "$RESULT_DIR/audio_test_${loop}.json"
        sed -i '$ s/$/,/' "$RESULT_DIR/audio_test_${loop}.json"

        # Process with application
        python -m src.application --audio-file="testing/${name}.wav" > /dev/null
        
        # Append the clingo stats to the audio stats
        sed '1d' ./clingo_stats.json >> "$RESULT_DIR/audio_test_${loop}.json"
        mv "$RESULT_DIR/audio_test_${loop}.json" "$RESULT_DIR/audio_test_${loop}_model_${mode}.json"

        loop=$(($loop + 1))
        echo $loop
    done
done


# using a complex signal (FM)
mode=1
loop=1

for duration in $(seq 1 2 10)
do
    for amp in "${AMPLITUDES[@]}"
    do
        #duration=$(( 1 + (i * $MAX_DURATION / $N_RUNS) ))
        frequency=440
        mf=$frequency
        md=$duration
        name="${mode}_${amp}Hz_${duration}s_${loop}run"
        
        echo $name
        echo "Run $loop out of 10 - Frequency: $frequency Hz, Duration: $duration s"
        
        # Generate signal
        python -m benchmark \
            -f=$frequency \
            -d=$duration \
            -mf=$mf \
            -md=$md \
            -n="$name" \
            -r=$loop \
            -m=$mode \
        
        sed -i '$d' "$RESULT_DIR/audio_test_${loop}.json"
        sed -i '$ s/$/,/' "$RESULT_DIR/audio_test_${loop}.json"

        # Process with application
        python -m src.application --audio-file="testing/${name}.wav" > /dev/null
        
        # Append the clingo stats to the audio stats
        sed '1d' ./clingo_stats.json >> "$RESULT_DIR/audio_test_${loop}.json"
        mv "$RESULT_DIR/audio_test_${loop}.json" "$RESULT_DIR/audio_test_${loop}_model_${mode}.json"

        loop=$(($loop + 1))
    done
done

# Analysis
python ./benchmark/analyze_results.py