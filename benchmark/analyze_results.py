import sys
import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Load all JSON stats files
benchmark_dir = Path("./benchmark/results/")

# Combine into DataFrame
df_0 = []
df_1 = []

run = sys.argv[1]

for file in benchmark_dir.glob("*_model_0.json"):
    try:
        # Read JSON file contents
        with open(file, 'r') as f:
            data = json.load(f)
            
        # Convert to DataFrame
        df_temp = pd.DataFrame([data])
        df_0.append(df_temp)
        
    except json.JSONDecodeError as e:
        print(f"Error reading {file}: {e}")
        continue

for file in benchmark_dir.glob("*_model_1.json"):
    try:
        # Read JSON file contents
        with open(file, 'r') as f:
            data = json.load(f)
            
        # Convert to DataFrame
        df_temp = pd.DataFrame([data])
        df_1.append(df_temp)
        
    except json.JSONDecodeError as e:
        print(f"Error reading {file}: {e}")
        continue

if df_0 and df_1:
    df_fin_0 = pd.concat(df_0, ignore_index=True)#
    df_fin_1 = pd.concat(df_1, ignore_index=True)
else:
    sys.exit(1)
# Create plots
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(df_fin_0)
    print(df_fin_1)

plt.figure(figsize=(30,10))
# duration vs time_solve
plt.subplot(2,3,1)
sns.scatterplot(data=df_fin_0, x='av_amplitude', y='time_solve', size="phase_offset")
plt.title('amplitude vs time_solve; model=0; size -> phase_offset')

# frequency vs conflicts 
plt.subplot(2,3,2)
sns.scatterplot(data=df_fin_0, x='duration', y='time_solve', size="phase_offset")
plt.title('duration vs time_solve; model=0')

plt.subplot(2,3,3)
sns.scatterplot(data=df_fin_0, x='dynamic_range', y='time_solve', size="phase_offset")
plt.title('dynamic_range vs time_solve; model=0')
"""
# modulation_index vs time_solve
plt.subplot(2,5,7)
sns.scatterplot(data=df_fin_0, x='modulation_index', y='time_solve', hue="modulation_index", size="phase_offset")
plt.title('modulation_index vs time_solve; model=0')

# phase_offset vs time_sat
plt.subplot(2,5,3)
sns.scatterplot(data=df_fin_0, x='phase_offset', y='time_solve', hue="modulation_index", size="phase_offset")
plt.title('phase_offset vs time_solve; model=0')
"""
# same for complex signals
plt.subplot(2,3,4)
sns.scatterplot(data=df_fin_1, x='av_amplitude', y='time_solve', hue="modulation_index", size="phase_offset")
plt.title('amplitude vs time_solve; model=1; color -> modulation depth; size -> phase_offset')

plt.subplot(2,3,5)
sns.scatterplot(data=df_fin_1, x='duration', y='time_solve', hue="modulation_index", size="phase_offset")
plt.title('duration vs time_solve; model=1')

plt.subplot(2,3,6)
sns.scatterplot(data=df_fin_1, x='dynamic_range', y='time_solve', hue="modulation_index", size="phase_offset")
plt.title('dynamic_range vs time_solve; model=1')
"""
# modulation_index vs time_solve
plt.subplot(2,5,9)
sns.scatterplot(data=df_fin_1, x='modulation_index', y='time_solve', hue="modulation_index", size="phase_offset")
plt.title('modulation_index vs time_solve; model=1')

# phase_offset vs time_sat
plt.subplot(2,5,10)
sns.scatterplot(data=df_fin_1, x='phase_offset', y='time_solve', hue="modulation_index", size="phase_offset")
plt.title('phase_offset vs time_solve; model=1')
"""
#save figure maybe automatically
plt.tight_layout()
plt.savefig(benchmark_dir / f'benchmark_analysis_{run}.png', 
            dpi=300,
            bbox_inches='tight',
            pad_inches=0.1)
plt.close()
