# benchmark/run_test_set.py
import sys
import json
import argparse
import numpy as np
from .__main__ import SignalGenerator, main

# Parameter sets for distinct test cases
TEST_CASES = {
    "duration_test": [
        {"frequency": 440, "duration": 1, "phase_offset": 0.0, "dynamic_range": (0.5, 0.5), "mode": 0},
        {"frequency": 440, "duration": 5, "phase_offset": 0.0, "dynamic_range": (0.5, 0.5), "mode": 0},
        {"frequency": 440, "duration": 10, "phase_offset": 0.1, "dynamic_range": (-0.5, 0.5), "mode": 0},
        {"frequency": 440, "duration": 20, "phase_offset": 0.1, "dynamic_range": (-0.5, 0.5), "mode": 0},
        {"frequency": 440, "duration": 30, "phase_offset": 0.1, "dynamic_range": (-0.5, 0.5), "mode": 0},
        {"frequency": 440, "duration": 45, "phase_offset": 0.1, "dynamic_range": (-0.5, 0.5), "mode": 0},
        {"frequency": 440, "duration": 60, "phase_offset": 0.1, "dynamic_range": (-0.5, 0.5), "mode": 0},
        {"frequency": 440, "duration": 75, "phase_offset": 0.1, "dynamic_range": (-0.5, 0.5), "mode": 0},
        {"frequency": 440, "duration": 90, "phase_offset": 0.1, "dynamic_range": (-0.5, 0.5), "mode": 0},
    ],
    
    "amplitude_test": [
        {"frequency": 440, "duration": 8, "phase_offset": 0.1, "dynamic_range": (-0.1, 0.1), "mode": 0},
        {"frequency": 440, "duration": 8, "phase_offset": 0.1, "dynamic_range": (-0.2, 0.2), "mode": 0},
        {"frequency": 440, "duration": 8, "phase_offset": 0.1, "dynamic_range": (-0.3, 0.3), "mode": 0},
        {"frequency": 440, "duration": 8, "phase_offset": 0.1, "dynamic_range": (-0.4, 0.4), "mode": 0},
        {"frequency": 440, "duration": 8, "phase_offset": 0.1, "dynamic_range": (-0.5, 0.5), "mode": 0},
        {"frequency": 440, "duration": 8, "phase_offset": 0.1, "dynamic_range": (-0.6, 0.6), "mode": 0},
        {"frequency": 440, "duration": 8, "phase_offset": 0.1, "dynamic_range": (-0.7, 0.7), "mode": 0},
        {"frequency": 440, "duration": 8, "phase_offset": 0.1, "dynamic_range": (-0.8, 0.8), "mode": 0},
        {"frequency": 440, "duration": 8, "phase_offset": 0.1, "dynamic_range": (-0.9, 0.9), "mode": 0},
    ],
    
    "constant_amplitude_test": [
        {"frequency": 440, "duration": 8, "phase_offset": 0.1, "dynamic_range": (-0.9, -0.9), "mode": 0},
        {"frequency": 440, "duration": 8, "phase_offset": 0.1, "dynamic_range": (-0.7, -0.7), "mode": 0},
        {"frequency": 440, "duration": 8, "phase_offset": 0.1, "dynamic_range": (-0.5, -0.5), "mode": 0},
        {"frequency": 440, "duration": 8, "phase_offset": 0.1, "dynamic_range": (-0.3, -0.3), "mode": 0},
        {"frequency": 440, "duration": 8, "phase_offset": 0.1, "dynamic_range": (-0.1, -0.1), "mode": 0},
    ],
    
    "frequency_test": [
        {"frequency": 55, "duration": 8, "phase_offset": 0.1, "dynamic_range": (-0.5, 0.5), "mode": 0},
        {"frequency": 110, "duration": 8, "phase_offset": 0.1, "dynamic_range": (-0.5, 0.5), "mode": 0},
        {"frequency": 220, "duration": 8, "phase_offset": 0.1, "dynamic_range": (-0.5, 0.5), "mode": 0},
        {"frequency": 440, "duration": 8, "phase_offset": 0.1, "dynamic_range": (-0.5, 0.5), "mode": 0},
        {"frequency": 880, "duration": 8, "phase_offset": 0.1, "dynamic_range": (-0.5, 0.5), "mode": 0},
        {"frequency": 1760, "duration": 8, "phase_offset": 0.1, "dynamic_range": (-0.5, 0.5), "mode": 0},
        {"frequency": 3520, "duration": 8, "phase_offset": 0.1, "dynamic_range": (-0.5, 0.5), "mode": 0},
    ],
    
    "phase_offset_test": [
        {"frequency": 440, "duration": 8, "phase_offset": 0.0, "dynamic_range": (-0.5, 0.5), "mode": 0},
        {"frequency": 440, "duration": 8, "phase_offset": 0.1, "dynamic_range": (-0.5, 0.5), "mode": 0},
        {"frequency": 440, "duration": 8, "phase_offset": 0.2, "dynamic_range": (-0.5, 0.5), "mode": 0},
        {"frequency": 440, "duration": 8, "phase_offset": 0.3, "dynamic_range": (-0.5, 0.5), "mode": 0},
        {"frequency": 440, "duration": 8, "phase_offset": 0.4, "dynamic_range": (-0.5, 0.5), "mode": 0},
        {"frequency": 440, "duration": 8, "phase_offset": 0.5, "dynamic_range": (-0.5, 0.5), "mode": 0},
        {"frequency": 440, "duration": 8, "phase_offset": 0.75, "dynamic_range": (-0.5, 0.5), "mode": 0},
        {"frequency": 440, "duration": 8, "phase_offset": 1.0, "dynamic_range": (-0.5, 0.5), "mode": 0},
    ],
    
    "complex_signal_mod_freq_test": [
        {"frequency": 440, "duration": 8, "phase_offset": 0.1, "dynamic_range": (-0.5, 0.5), "mode": 1, "mod_frequency": 0.02, "mod_depth": 1.0},
        {"frequency": 440, "duration": 8, "phase_offset": 0.1, "dynamic_range": (-0.5, 0.5), "mode": 1, "mod_frequency": 0.05, "mod_depth": 1.0},
        {"frequency": 440, "duration": 8, "phase_offset": 0.1, "dynamic_range": (-0.5, 0.5), "mode": 1, "mod_frequency": 0.1, "mod_depth": 1.0},
        {"frequency": 440, "duration": 8, "phase_offset": 0.1, "dynamic_range": (-0.5, 0.5), "mode": 1, "mod_frequency": 0.2, "mod_depth": 1.0},
        {"frequency": 440, "duration": 8, "phase_offset": 0.1, "dynamic_range": (-0.5, 0.5), "mode": 1, "mod_frequency": 0.3, "mod_depth": 1.0},
        {"frequency": 440, "duration": 8, "phase_offset": 0.1, "dynamic_range": (-0.5, 0.5), "mode": 1, "mod_frequency": 0.5, "mod_depth": 1.0},
    ],
    
    "complex_signal_mod_depth_test": [
        {"frequency": 440, "duration": 8, "phase_offset": 0.1, "dynamic_range": (-0.5, 0.5), "mode": 1, "mod_frequency": 0.1, "mod_depth": 0.2},
        {"frequency": 440, "duration": 8, "phase_offset": 0.1, "dynamic_range": (-0.5, 0.5), "mode": 1, "mod_frequency": 0.1, "mod_depth": 0.5},
        {"frequency": 440, "duration": 8, "phase_offset": 0.1, "dynamic_range": (-0.5, 0.5), "mode": 1, "mod_frequency": 0.1, "mod_depth": 1.0},
        {"frequency": 440, "duration": 8, "phase_offset": 0.1, "dynamic_range": (-0.5, 0.5), "mode": 1, "mod_frequency": 0.1, "mod_depth": 2.0},
        {"frequency": 440, "duration": 8, "phase_offset": 0.1, "dynamic_range": (-0.5, 0.5), "mode": 1, "mod_frequency": 0.1, "mod_depth": 3.0},
        {"frequency": 440, "duration": 8, "phase_offset": 0.1, "dynamic_range": (-0.5, 0.5), "mode": 1, "mod_frequency": 0.1, "mod_depth": 5.0},
    ],
    
    "complex_signal_combined_test": [
        {"frequency": 440, "duration": 8, "phase_offset": 0.1, "dynamic_range": (-0.3, 0.3), "mode": 1, "mod_frequency": 0.05, "mod_depth": 0.5},
        {"frequency": 440, "duration": 8, "phase_offset": 0.1, "dynamic_range": (-0.5, 0.5), "mode": 1, "mod_frequency": 0.1, "mod_depth": 1.0},
        {"frequency": 440, "duration": 8, "phase_offset": 0.1, "dynamic_range": (-0.7, 0.7), "mode": 1, "mod_frequency": 0.2, "mod_depth": 2.0},
        {"frequency": 440, "duration": 8, "phase_offset": 0.1, "dynamic_range": (-0.9, 0.9), "mode": 1, "mod_frequency": 0.3, "mod_depth": 3.0},
    ],
    
    "performance_scaling_test": [
        # Test with increasingly complex signals to see how performance scales
        {"frequency": 440, "duration": 5, "phase_offset": 0.1, "dynamic_range": (-0.5, 0.5), "mode": 0},  # Simple
        {"frequency": 440, "duration": 5, "phase_offset": 0.1, "dynamic_range": (-0.5, 0.5), "mode": 1, "mod_frequency": 0.05, "mod_depth": 0.5},  # Basic modulation
        {"frequency": 220, "duration": 10, "phase_offset": 0.3, "dynamic_range": (-0.7, 0.7), "mode": 1, "mod_frequency": 0.1, "mod_depth": 1.0},  # Medium complexity
        {"frequency": 110, "duration": 15, "phase_offset": 0.5, "dynamic_range": (-0.9, 0.9), "mode": 1, "mod_frequency": 0.2, "mod_depth": 2.0},  # High complexity
        {"frequency": 55, "duration": 30, "phase_offset": 0.7, "dynamic_range": (-0.9, 0.9), "mode": 1, "mod_frequency": 0.3, "mod_depth": 3.0},  # Very high complexity
    ]
}

def run_test_set(test_case_name, output_file):
    """Run a predefined set of tests and collect results"""
    
    if test_case_name not in TEST_CASES:
        print(f"Error: Unknown test case '{test_case_name}'")
        sys.exit(1)
        
    results = []
    for i, params in enumerate(TEST_CASES[test_case_name]):
        print(f"Running test {i}/{len(TEST_CASES[test_case_name])}")
        
        # Create a unique name for this test
        test_name = f"{test_case_name}_{i}"
        
        # Generate signal with these parameters
        generator = SignalGenerator(output_dir="./data/testing")
        sig, sr = generator.generate_signal(
            name=test_name,
            amplitude=params.get("dynamic_range"),
            frequency=params.get("frequency"),
            duration=params.get("duration"),
            phase_offset=params.get("phase_offset"),
            modulated=params.get("mode"),
            mod_frequency=params.get("mod_frequency"),
            mod_depth=params.get("mod_depth")
        )
        
        # Run your application on the generated signal
        audio_path = f"./data/testing/{test_name}.wav"
        
        # This is where you need to call your application
        # For example:
        import subprocess
        subprocess.run([
            "python", "-m", "src.application", 
            "--audio-file", audio_path
        ], check=True)
        
        # Collect results from your application
        try:
            with open('clingo_stats.json', 'r') as f:
                clingo_stats = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            print(f"Warning: Could not load results for test {test_name}")
            clingo_stats = {}
            
        # Add parameters and results to our collection
        result = {
            "test_name": test_name,
            "parameters": params,
            "performance": clingo_stats,
            "signal_info": {
                "max_amplitude": float(np.max(np.abs(sig))),
                "rms": float(np.sqrt(np.mean(sig**2)))
            }
        }
        results.append(result)
    
    # Save all results
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run benchmark test sets")
    parser.add_argument("--test_case", required=True, help="Name of test case to run")
    parser.add_argument("--output", required=True, help="Output JSON file")
    
    args = parser.parse_args()
    run_test_set(args.test_case, args.output)