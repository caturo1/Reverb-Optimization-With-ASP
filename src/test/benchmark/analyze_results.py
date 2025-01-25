import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Load all JSON stats files
benchmark_dir = Path("./benchmark/results")
stats_files = list(benchmark_dir.glob("stats_test*.json"))

# Combine into DataFrame
df = pd.concat([pd.read_json(f) for f in stats_files])

# Create plots
plt.figure(figsize=(15,10))

# Plot 1: Solving time vs signal duration
plt.subplot(2,2,1)
sns.scatterplot(data=df, x='length', y='solving_time')
plt.title('Solving Time vs Signal Duration')

# Plot 2: Conflicts vs frequency 
plt.subplot(2,2,2)
sns.scatterplot(data=df, x='frequency', y='conflicts')
plt.title('Conflicts vs Signal Frequency')

# Plot 3: Rules generated vs length
plt.subplot(2,2,3)
sns.scatterplot(data=df, x='length', y='rules')
plt.title('Rules vs Signal Length')

# Plot 4: Choices vs frequency
plt.subplot(2,2,4)
sns.scatterplot(data=df, x='frequency', y='choices')
plt.title('Choices vs Signal Frequency')

plt.tight_layout()
plt.show()
plt.savefig('benchmark_results/benchmark_analysis.png')