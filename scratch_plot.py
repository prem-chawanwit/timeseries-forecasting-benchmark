import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Load data
df = pd.read_csv('experiments/run_20260516_101629/data/2_etl/processed.csv', index_col='timestamp', parse_dates=True)

# 1. Correlation Heatmap
plt.figure(figsize=(12, 10))
corr = df.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1)
plt.title('Feature Correlation Heatmap')
plt.tight_layout()
plt.savefig('experiments/run_20260516_101629/data/2_etl/correlation_heatmap.png', dpi=300)
plt.close()

# 2. Scatter plots of Top 4 features vs Target
target_col = 'target'
# Get absolute correlation with target, drop target itself
top_features = corr[target_col].abs().drop(target_col).sort_values(ascending=False).head(4).index

plt.figure(figsize=(15, 10))
for i, feature in enumerate(top_features, 1):
    plt.subplot(2, 2, i)
    sns.scatterplot(data=df, x=feature, y=target_col, alpha=0.5, color='teal')
    plt.title(f'{feature} vs Target (Actual)')
    plt.xlabel(feature)
    plt.ylabel('Target (TP2 Future)')

plt.tight_layout()
plt.savefig('experiments/run_20260516_101629/data/2_etl/feature_scatter.png', dpi=300)
plt.close()
print("Plots generated successfully!")
