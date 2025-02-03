import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Load all JSON stats files
benchmark_dir = Path("./benchmark/results/")
stats = list(benchmark_dir.glob("*.json"))

# Combine into DataFrame
df = []
for file in benchmark_dir.glob("*.json"):
    try:
        # Read JSON file contents
        with open(file, 'r') as f:
            data = json.load(f)
            
        # Convert to DataFrame
        df_temp = pd.DataFrame([data])
        df.append(df_temp)
        
    except json.JSONDecodeError as e:
        print(f"Error reading {file}: {e}")
        continue

if df:
    df_fin = pd.concat(df, ignore_index=True)
# Create plots
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(df_fin)

print(df_fin)
# duration vs time_solve
plt.subplot(3,4,1)
sns.scatterplot(data=df_fin, x='duration', y='time_solve')
plt.title('duration vs time_solve')

# frequency vs conflicts 
plt.subplot(3,4,2)
sns.scatterplot(data=df_fin, x='frequency', y='conflicts')
plt.title('frequency vs conflicts')

# duration vs rules
plt.subplot(3,4,3)
sns.scatterplot(data=df_fin, x='duration', y='rules')
plt.title('duration vs rules')

# frequency vs choices
plt.subplot(3,4,5)
sns.scatterplot(data=df_fin, x='frequency', y='choices')
plt.title('frequency vs choices')

# modulation_index vs choices
plt.subplot(3,4,6)
sns.scatterplot(data=df_fin, x='modulation_index', y='choices')
plt.title('modulation_index vs choices')

# modulation_index vs time_solve
plt.subplot(3,4,7)
sns.scatterplot(data=df_fin, x='modulation_index', y='time_solve')
plt.title('modulation_index vs time_solve')

# phase_offset vs time_sat
plt.subplot(3,4,8)
sns.scatterplot(data=df_fin, x='phase_offset', y='time_sat')
plt.title('phase_offset vs time_sat')

# duration vs rules
plt.subplot(3,4,9)
sns.scatterplot(data=df_fin, x='duration', y='rules')
plt.title('duration vs rules')

# modulating_frequency vs rules
plt.subplot(3,4,10)
sns.scatterplot(data=df_fin, x='modulating_frequency', y='rules')
plt.title('modulating_frequency vs rules')

# modulation_index vs rules
plt.subplot(3,4,11)
sns.scatterplot(data=df_fin, x='modulation_index', y='rules')
plt.title('modulation_index vs rules')

# duration vs time_total
plt.subplot(3,4,12)
sns.scatterplot(data=df_fin, x='duration', y='time_total')
plt.title('duration vs time_total')

# phase_offset vs conflicts
plt.subplot(3,4,4)
sns.scatterplot(data=df_fin, x='phase_offset', y='conflicts')
plt.title('phase_offset vs conflicts')

plt.tight_layout()
plt.show()

#save figure maybe automatically