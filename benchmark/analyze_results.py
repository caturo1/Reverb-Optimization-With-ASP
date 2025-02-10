import sys
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Load all JSON stats files
benchmark_dir = Path("./benchmark/results/")

# Combine into DataFrame
df_0 = []
df_1 = []

for file in benchmark_dir.glob("*0.json"):
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

for file in benchmark_dir.glob("*1.json"):
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

# duration vs time_solve
plt.subplot(2,3,1)
sns.scatterplot(data=df_fin_0, x='max_amplitude', y='time_solve')
plt.title('amplitude vs time_solve; model=0')

# frequency vs conflicts 
plt.subplot(2,3,3)
sns.scatterplot(data=df_fin_0, x='duration', y='time_solve')
plt.title('duration vs time_solve; model=0')

# duration vs rules
plt.subplot(2,3,2)
sns.scatterplot(data=df_fin_1, x='max_amplitude', y='time_solve')
plt.title('amplitude vs time_solve; model=1')

plt.subplot(2,3,4)
sns.scatterplot(data=df_fin_1, x='duration', y='time_solve')
plt.title('duration vs time_solve; model=1')

plt.subplot(2,3,5)
sns.scatterplot(data=df_fin_0, x='dynamic_range', y='time_solve')
plt.title('dynamic_range vs time_solve; model=0')

plt.subplot(2,3,6)
sns.scatterplot(data=df_fin_1, x='dynamic_range', y='time_solve')
plt.title('dynamic_range vs time_solve; model=1')

"""# frequency vs choices
plt.subplot(2,4,4)
sns.scatterplot(data=df_fin_0, x='frequency', y='time_solve')
plt.title('frequency vs choices')
# modulation_index vs choices

# modulation_index vs time_solve
plt.subplot(2,4,6)
sns.scatterplot(data=df_fin_1, x='duration', y='time_solve')
plt.title('modulation_index vs time_solve')

# phase_offset vs time_sat
plt.subplot(2,4,7)
sns.scatterplot(data=df_fin_1, x='max_amplitude', y='time_solve')
plt.title('phase_offset vs time_sat')

# duration vs rules
plt.subplot(2,4,8)
sns.scatterplot(data=df_fin_1, x='modulation_index', y='time_solve')
plt.title('duration vs rules')
"""

plt.tight_layout()
plt.show()

#save figure maybe automatically