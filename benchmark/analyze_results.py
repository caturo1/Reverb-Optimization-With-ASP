# benchmark/analyze.py
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

# Set default font sizes for LaTeX scaling (0.7 linewidth) with larger figure
plt.rcParams.update({
    'font.size': 34,           # Base font size
    'axes.titlesize': 38,      # Title font size
    'axes.labelsize': 34,      # Axis label font size
    'xtick.labelsize': 32,     # X-axis tick label size
    'ytick.labelsize': 32,     # Y-axis tick label size
    'legend.fontsize': 32,     # Legend font size
    'figure.titlesize': 42,    # Figure title font size
})

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
        fig, axes = plt.subplots(1, num_tests, figsize=(10*num_tests, 8))
    else:
        fig, axes = plt.subplots(2, 2, figsize=(24, 20))
    
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
    plt.savefig(os.path.join(output_dir, 'all_results.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Also create a combined figure showing all parameters' impact on solving time
    create_combined_plot(results, output_dir)

def plot_single_parameter(results, param_name, ax, axis_label, title):
    """Plot the impact of a single parameter on solving time with artifact count coloring"""
    param_values = [r['parameters'][param_name] for r in results]
    
    # Sort results by solving time instead of parameter value
    solve_times = [r['performance']['time_solve'] for r in results]
    sorted_indices = np.argsort(solve_times)
    param_values = [param_values[i] for i in sorted_indices]
    solve_times = [solve_times[i] for i in sorted_indices]
    results = [results[i] for i in sorted_indices]
    
    # Extract artifact counts
    artifact_counts = [r['performance'].get('num_artifacts', 0) for r in results]
    
    # Create colormap for artifact counts
    if max(artifact_counts) > 0:
        norm = Normalize(vmin=0, vmax=max(artifact_counts))
        cmap = plt.cm.viridis
        colors = [cmap(norm(count)) for count in artifact_counts]
    else:
        colors = ['blue'] * len(artifact_counts)
    
    # Plot lines first (swapped axes)
    ax.plot(solve_times, param_values, '-', color='gray', linewidth=3, alpha=0.5)
    
    # Plot points with colors based on artifact count (swapped axes)
    scatter = ax.scatter(solve_times, param_values, c=artifact_counts, 
                            cmap='viridis', s=120, edgecolors='black', linewidth=2)
    
    ax.set_xlabel('Solving Time (s)', fontsize=34)  # Swapped
    ax.set_ylabel(axis_label, fontsize=34)           # Swapped
    ax.set_title(title, fontsize=38)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=32)
    
    # Add colorbar if we have artifacts
    if max(artifact_counts) > 0:
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Number of Artifacts', fontsize=32)
        cbar.ax.tick_params(labelsize=30)
    
    # Add value labels for better readability (swapped coordinates)
    for i, (x, y, count) in enumerate(zip(solve_times, param_values, artifact_counts)):
        if i % max(1, len(param_values) // 3) == 0:  # Only label every 3rd of points
            ax.annotate(f'{y:.2f}', (x, y), textcoords="offset points", 
                            xytext=(15,0), ha='left', fontsize=22)  # Adjusted offset direction

def create_combined_plot(all_results, output_dir):
    """Create a single plot showing all parameters on the same graph"""
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    
    # Define a consistent mapping for parameter names to colors and markers
    # Now including Amplitude again
    param_styles = {
        'duration': {'color': 'g', 'marker': '^', 'label': 'Duration (s)'},
        'amplitude': {'color': 'r', 'marker': 's', 'label': 'Amplitude'}, # Re-added
        'frequency': {'color': 'c', 'marker': 'D', 'label': 'Frequency (Hz)'},
        'mod_freq': {'color': 'b', 'marker': 'o', 'label': 'Mod Frequency (Hz)'},
    }
    
    max_artifacts = 0
    
    # First pass to find maximum artifact count
    for test_results in all_results.values():
        for result in test_results:
            max_artifacts = max(max_artifacts, result['performance'].get('num_artifacts', 0))
    
    # Set up colormap for artifact counts
    norm = Normalize(vmin=0, vmax=max_artifacts) if max_artifacts > 0 else None
    cmap = plt.cm.viridis
    
    # List to hold legend handles
    legend_elements = []

    for test_name, test_results in all_results.items():
        param_name = None
        
        if 'duration' in test_name:
            param_name = 'duration'
        elif 'amplitude' in test_name:
            param_name = 'amplitude' # No longer skipping
        elif 'frequency' in test_name and 'mod_freq' not in test_name:
            param_name = 'frequency'
        elif 'complex' in test_name or 'mod_freq' in test_name:
            param_name = 'mod_freq'
        
        if param_name and param_name in param_styles: # Ensure it's a known parameter type with a style
            param_values = []
            for result in test_results:
                # Handle specific parameter extraction logic
                if param_name == 'amplitude':
                    param_values.append(abs(result['parameters']['dynamic_range'][1]))
                elif param_name == 'mod_freq':
                    mod_freq = result['parameters'].get('mod_frequency', 
                                      result['parameters'].get('modulation_frequency', 0))
                    param_values.append(mod_freq)
                else:
                    param_values.append(result['parameters'][param_name])

            solve_times = [r['performance']['time_solve'] for r in test_results]
            artifact_counts = [r['performance'].get('num_artifacts', 0) for r in test_results]
            
            # Sort by solving time to make lines connect meaningfully on the x-axis
            sorted_indices = np.argsort(solve_times)
            param_values = [param_values[i] for i in sorted_indices]
            solve_times = [solve_times[i] for i in sorted_indices]
            artifact_counts = [artifact_counts[i] for i in sorted_indices]
            
            # Normalize parameter values to [0, 1] for comparison
            if len(param_values) > 1 and max(param_values) != min(param_values):
                norm_param_values = (np.array(param_values) - min(param_values)) / (max(param_values) - min(param_values))
            else:
                norm_param_values = np.zeros_like(param_values)  # Handle constant values
            
            style = param_styles[param_name]

            # Plot line (actual solving times vs normalized parameters)
            ax.plot(solve_times, norm_param_values, 
                    color=style['color'], 
                    linewidth=3, 
                    alpha=0.5)
            
            # Plot points with artifact count coloring
            scatter = ax.scatter(solve_times, norm_param_values, 
                                 c=artifact_counts,
                                 cmap='viridis',
                                 s=160,
                                 marker=style['marker'],
                                 edgecolors='black',
                                 linewidth=2,
                                 vmin=0,
                                 vmax=max_artifacts)
            
            # Add element for this parameter type to the legend_elements list
            legend_elements.append(plt.Line2D([0], [0], marker=style['marker'], color='w', 
                                             markerfacecolor=style['color'], markersize=16, 
                                             label=style['label'], markeredgecolor='black'))
    
    ax.set_xlabel('Solving Time (s)', fontsize=34)
    ax.set_ylabel('Normalized Parameter Value (0-1)', fontsize=34)
    ax.set_title('Impact of Normalized Parameters on Solving Time', fontsize=38)
    ax.tick_params(labelsize=32)
    
    # Use the collected legend_elements
    ax.legend(handles=legend_elements, loc='upper left', fontsize=32)
    ax.grid(True, alpha=0.3)
    
    # Add colorbar for artifact counts
    if max_artifacts > 0:
        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label('Number of Artifacts', fontsize=32)
        cbar.ax.tick_params(labelsize=30)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'combined_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze benchmark results")
    parser.add_argument("--input", required=True, help="Input directory with JSON results")
    parser.add_argument("--output", required=True, help="Output directory for analysis")
    
    args = parser.parse_args()
    analyze_results(args.input, args.output)