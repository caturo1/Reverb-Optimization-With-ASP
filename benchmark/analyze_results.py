# benchmark/analyze.py
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse

def analyze_results(input_dir, output_dir):
    """Analyze benchmark results and create visualizations"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Load all test results
    results = {}
    for filename in os.listdir(input_dir):
        if filename.endswith('.json'):
            with open(os.path.join(input_dir, filename), 'r') as f:
                results[filename.replace('.json', '')] = json.load(f)
    
    # Process each test case
    for test_name, test_results in results.items():
        if 'duration' in test_name:
            plot_parameter_impact(test_results, 'duration', 
                                 output_dir, 'Duration (s)')
        elif 'amplitude' in test_name:
            # Extract dynamic range values
            for result in test_results:
                result['parameters']['amplitude'] = result['parameters']['dynamic_range'][1]
            plot_parameter_impact(test_results, 'amplitude', 
                                 output_dir, 'Amplitude')
        elif 'frequency' in test_name:
            plot_parameter_impact(test_results, 'frequency', 
                                 output_dir, 'Frequency (Hz)')
        # Add more cases as needed

def plot_parameter_impact(results, param_name, output_dir, axis_label):
    """Create a set of plots showing the impact of a parameter"""
    param_values = [r['parameters'][param_name] for r in results]
    
    # Sort results by parameter value
    sorted_indices = np.argsort(param_values)
    param_values = [param_values[i] for i in sorted_indices]
    results = [results[i] for i in sorted_indices]
    
    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Parameter vs. Solving Time
    axes[0, 0].plot(param_values, [r['performance']['time_solve'] for r in results], 'o-b')
    axes[0, 0].set_xlabel(axis_label)
    axes[0, 0].set_ylabel('Solving Time (s)')
    axes[0, 0].set_title(f'Impact of {param_name} on Solving Time')
    
    # Add more plots showing other metrics
    # ...
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{param_name}_impact.png'))
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze benchmark results")
    parser.add_argument("--input", required=True, help="Input directory with JSON results")
    parser.add_argument("--output", required=True, help="Output directory for analysis")
    
    args = parser.parse_args()
    analyze_results(args.input, args.output)