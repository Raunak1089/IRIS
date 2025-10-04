# eda_and_stats.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from scipy import stats

# Load Iris
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target, name='species')
df = pd.concat([X, y], axis=1)
df['species_name'] = df['species'].map(dict(enumerate(iris.target_names)))

print(df.head())

# Pairplot
sns.pairplot(df, hue='species_name', corner=True)
plt.suptitle("Iris pairplot", y=1.02)
plt.show()

# Summary statistics
print(df.groupby('species_name').agg(['mean','std']).round(3))

# Statistical tests: ANOVA for each feature across species
for col in iris.feature_names:
    groups = [df[df['species'] == i][col] for i in sorted(df['species'].unique())]
    f, p = stats.f_oneway(*groups)
    print(f"ANOVA {col}: F={f:.3f}, p={p:.3e}")

# If normality not assumed -> Kruskal-Wallis
for col in iris.feature_names:
    groups = [df[df['species'] == i][col] for i in sorted(df['species'].unique())]
    h, p = stats.kruskal(*groups)
    print(f"Kruskal-Wallis {col}: H={h:.3f}, p={p:.3e}")
