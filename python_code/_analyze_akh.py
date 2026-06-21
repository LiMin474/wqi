import numpy as np
import pandas as pd
import sys, os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
data = np.load(os.path.join(SCRIPT_DIR, 'datasets', '4_akh_wqi.npz'), allow_pickle=True)
X = data['X']
y = data['y'].ravel()

print(f"Samples: {len(y)}, Features: {X.shape[1]}")
print(f"Target name: {data['target_name']}")
print(f"Dataset: {data['name']}")
print()

# Target distribution
print(f"WQI range: [{y.min():.2f}, {y.max():.2f}]")
print(f"WQI mean ± std: {y.mean():.2f} ± {y.std():.2f}")
print(f"WQI median: {np.median(y):.2f}")
print()

# Feature statistics
print("Feature stats:")
for i in range(X.shape[1]):
    print(f"  F{i+1}: mean={X[:,i].mean():.2f}, std={X[:,i].std():.2f}, "
          f"min={X[:,i].min():.2f}, max={X[:,i].max():.2f}, "
          f"missing={np.isnan(X[:,i]).sum()}")

print()
# Correlation with target
corrs = []
for i in range(X.shape[1]):
    mask = ~(np.isnan(X[:,i]) | np.isnan(y))
    corr = np.corrcoef(X[mask,i], y[mask])[0,1]
    corrs.append(corr)
    print(f"  F{i+1}-WQI corr: {corr:.4f}")

print()
# Check for NaNs
print(f"X NaNs: {np.isnan(X).sum()}")
print(f"y NaNs: {np.isnan(y).sum()}")

# Check for duplicates
df = pd.DataFrame(X)
dup = df.duplicated().sum()
print(f"Duplicate rows: {dup}")
print()

# Nearest neighbor analysis - is there useful structure?
from sklearn.neighbors import NearestNeighbors
nbrs = NearestNeighbors(n_neighbors=6).fit(X)
distances, indices = nbrs.kneighbors(X)
# For each point, check if neighbors have similar y values
y_diff = []
for i in range(len(y)):
    neighbor_ys = y[indices[i, 1:]]  # skip self
    y_diff.append(np.abs(y[i] - neighbor_ys.mean()))
print(f"Mean |y - neighbor_mean|: {np.mean(y_diff):.4f}")
print(f"y.std(): {y.std():.4f}")
print(f"Ratio: {np.mean(y_diff) / y.std():.2f} (lower = more locally predictable)")

# Check if target is essentially random noise
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

rf = Pipeline([
    ('scaler', StandardScaler()),
    ('rf', RandomForestRegressor(n_estimators=200, random_state=1))
])
scores = cross_val_score(rf, X, y, cv=5, scoring='r2')
print(f"\nRandomForest R2CV: {scores.mean():.4f} ± {scores.std():.4f}")

# Linear baseline
from sklearn.linear_model import LinearRegression
lr = Pipeline([
    ('scaler', StandardScaler()),
    ('lr', LinearRegression())
])
scores_lr = cross_val_score(lr, X, y, cv=5, scoring='r2')
print(f"LinearRegression R2CV: {scores_lr.mean():.4f} ± {scores_lr.std():.4f}")