# benchmark/analyze.py
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

def analyze_results(input_dir, output_dir):
    """Analyze benchmark results and create visualizations"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Load all test results
    results = {}
    for filename in os.listdir(input_dir):
        if filename.endswith('.json'):
            with open(os.path.join(input_dir, filename), 'r') as f:
                results[filename.replace('.json', '')] = json.load(f)
    
    # Count how many test types we have
    num_tests = len(results)
    
    # Create a single figure with subplots for all test types
    if num_tests <= 3:
        fig, axes = plt.subplots(1, num_tests, figsize=(6*num_tests, 5))
    else:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Handle single subplot case
    if num_tests == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if num_tests > 3 else axes
    
    plot_idx = 0
    
    # Process each test case
    for test_name, test_results in results.items():
        if plot_idx >= len(axes):
            break
            
        if 'duration' in test_name:
            plot_single_parameter(test_results, 'duration', axes[plot_idx], 
                                'Duration (s)', 'Duration Test')
        elif 'amplitude' in test_name:
            # Extract dynamic range values
            for result in test_results:
                result['parameters']['amplitude'] = abs(result['parameters']['dynamic_range'][1])
            plot_single_parameter(test_results, 'amplitude', axes[plot_idx], 
                                'Amplitude', 'Amplitude Test')
        elif 'frequency' in test_name and 'mod_freq' not in test_name:  # Exclude mod_freq test
            plot_single_parameter(test_results, 'frequency', axes[plot_idx], 
                                'Frequency (Hz)', 'Frequency Test')
        elif 'complex' in test_name or 'mod_freq' in test_name:
            # Extract modulation frequency (it might be stored as mod_frequency)
            for result in test_results:
                mod_freq = result['parameters'].get('mod_frequency', 
                          result['parameters'].get('modulation_frequency', 0))
                result['parameters']['mod_freq'] = mod_freq
            plot_single_parameter(test_results, 'mod_freq', axes[plot_idx], 
                                'Modulation Frequency (Hz)', 'Complex Signal Test')
        else:
            continue  # Skip unknown test types
            
        plot_idx += 1
    
    # Hide any unused subplots
    if num_tests > 3:
        for idx in range(plot_idx, len(axes)):
            axes[idx].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'all_results.png'))
    plt.close()
    
    # Also create a combined figure showing all parameters' impact on solving time
    create_combined_plot(results, output_dir)

def plot_single_parameter(results, param_name, ax, axis_label, title):
    """Plot the impact of a single parameter on solving time with artifact count coloring"""
    param_values = [r['parameters'][param_name] for r in results]
    
    # Sort results by parameter value
    sorted_indices = np.argsort(param_values)
    param_values = [param_values[i] for i in sorted_indices]
    results = [results[i] for i in sorted_indices]
    
    # Extract solving times and artifact counts
    solve_times = [r['performance']['time_solve'] for r in results]
    artifact_counts = [r['performance'].get('num_artifacts', 0) for r in results]
    
    # Create colormap for artifact counts
    if max(artifact_counts) > 0:
        norm = Normalize(vmin=0, vmax=max(artifact_counts))
        cmap = plt.cm.viridis
        colors = [cmap(norm(count)) for count in artifact_counts]
    else:
        colors = ['blue'] * len(artifact_counts)
    
    # Plot lines first
    ax.plot(param_values, solve_times, '-', color='gray', linewidth=1, alpha=0.5)
    
    # Plot points with colors based on artifact count
    scatter = ax.scatter(param_values, solve_times, c=artifact_counts, 
                        cmap='viridis', s=64, edgecolors='black', linewidth=1)
    
    ax.set_xlabel(axis_label)
    ax.set_ylabel('Solving Time (s)')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    # Add colorbar if we have artifacts
    if max(artifact_counts) > 0:
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Number of Artifacts')
    
    # Add value labels for better readability
    for i, (x, y, count) in enumerate(zip(param_values, solve_times, artifact_counts)):
        if i % max(1, len(param_values) // 5) == 0:  # Only label every few points
            ax.annotate(f'{y:.2f}s', (x, y), textcoords="offset points", 
                       xytext=(0,10), ha='center', fontsize=8)

def create_combined_plot(all_results, output_dir):
    """Create a single plot showing all parameters on the same graph"""
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    colors = ['b', 'r', 'g', 'c', 'm']
    markers = ['o', 's', '^', 'D', 'v']
    
    color_idx = 0
    max_artifacts = 0
    
    # First pass to find maximum artifact count
    for test_results in all_results.values():
        for result in test_results:
            max_artifacts = max(max_artifacts, result['performance'].get('num_artifacts', 0))
    
    # Set up colormap for artifact counts
    norm = Normalize(vmin=0, vmax=max_artifacts) if max_artifacts > 0 else None
    cmap = plt.cm.viridis
    
    for test_name, test_results in all_results.items():
        if color_idx >= len(colors):
            break
            
        param_name = None
        param_label = None
        
        if 'duration' in test_name:
            param_name = 'duration'
            param_label = 'Duration (s)'
        elif 'amplitude' in test_name:
            for result in test_results:
                result['parameters']['amplitude'] = abs(result['parameters']['dynamic_range'][1])
            param_name = 'amplitude'
            param_label = 'Amplitude'
        elif 'frequency' in test_name and 'mod_freq' not in test_name:
            param_name = 'frequency'
            param_label = 'Frequency (Hz)'
        elif 'complex' in test_name or 'mod_freq' in test_name:
            for result in test_results:
                mod_freq = result['parameters'].get('mod_frequency', 
                          result['parameters'].get('modulation_frequency', 0))
                result['parameters']['mod_freq'] = mod_freq
            param_name = 'mod_freq'
            param_label = 'Mod Frequency (Hz)'
        
        if param_name:
            param_values = [r['parameters'][param_name] for r in test_results]
            solve_times = [r['performance']['time_solve'] for r in test_results]
            artifact_counts = [r['performance'].get('num_artifacts', 0) for r in test_results]
            
            # Sort by parameter value
            sorted_indices = np.argsort(param_values)
            param_values = [param_values[i] for i in sorted_indices]
            solve_times = [solve_times[i] for i in sorted_indices]
            artifact_counts = [artifact_counts[i] for i in sorted_indices]
            
            # Normalize parameter values to [0, 1] for comparison
            if len(param_values) > 1:
                norm_values = (np.array(param_values) - min(param_values)) / (max(param_values) - min(param_values))
            else:
                norm_values = param_values
            
            # Plot line
            ax.plot(norm_values, solve_times, 
                   color=colors[color_idx], 
                   linewidth=1, 
                   alpha=0.5)
            
            # Plot points with artifact count coloring
            if max_artifacts > 0:
                point_colors = [cmap(norm(count)) for count in artifact_counts]
            else:
                point_colors = colors[color_idx]
            
            scatter = ax.scatter(norm_values, solve_times, 
                               c=artifact_counts,
                               cmap='viridis',
                               s=100,
                               marker=markers[color_idx],
                               edgecolors='black',
                               linewidth=1,
                               label=param_label,
                               vmin=0,
                               vmax=max_artifacts)
            
            color_idx += 1
    
    ax.set_xlabel('Normalized Parameter Value (0-1)')
    ax.set_ylabel('Solving Time (s)')
    ax.set_title('Impact of All Parameters on Solving Time (Normalized)')
    
    # Create custom legend for parameters
    handles = []
    for i, (test_name, _) in enumerate(all_results.items()):
        if i >= len(colors):
            break
        label = None
        if 'duration' in test_name:
            label = 'Duration (s)'
        elif 'amplitude' in test_name:
            label = 'Amplitude'
        elif 'frequency' in test_name and 'mod_freq' not in test_name:
            label = 'Frequency (Hz)'
        elif 'complex' in test_name or 'mod_freq' in test_name:
            label = 'Mod Frequency (Hz)'
        
        if label:
            handles.append(plt.Line2D([0], [0], marker=markers[i], color='w', 
                                    markerfacecolor=colors[i], markersize=10, 
                                    label=label, markeredgecolor='black'))
    
    ax.legend(handles=handles, loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # Add colorbar for artifact counts
    if max_artifacts > 0:
        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label('Number of Artifacts')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'combined_analysis.png'))
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze benchmark results")
    parser.add_argument("--input", required=True, help="Input directory with JSON results")
    parser.add_argument("--output", required=True, help="Output directory for analysis")
    
    args = parser.parse_args()
    analyze_results(args.input, args.output)