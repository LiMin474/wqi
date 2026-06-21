import numpy as np
import pandas as pd
import os, sys, warnings
warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'datasets')
SCRIPT_DATA_DIR = os.path.join(SCRIPT_DIR, 'datasets')

# Load original CSV
csv_path = os.path.join(DATA_DIR, '4_akh_wqi.csv')
df = pd.read_csv(csv_path)
print(f"Original: {df.shape}")
print(f"Coliforms range: [{df['Coliforms'].min()}, {df['Coliforms'].max()}]")

# Log-transform Coliforms
df['Coliforms_log'] = np.log1p(df['Coliforms'])
print(f"After log1p: [{df['Coliforms_log'].min():.2f}, {df['Coliforms_log'].max():.2f}]")

# Build X, y - replace Coliforms with log version
feature_cols = ['PH','Temp','Turbidity','TSS','BOD5','COD','DO','Amoni','Phosphat','Coliforms_log']
X = df[feature_cols].values.astype(np.float64)
y = df['WQI'].values.astype(np.float64).reshape(-1, 1)

# Check new correlations
for i, col in enumerate(feature_cols):
    corr = np.corrcoef(X[:,i], y.ravel())[0,1]
    print(f"  {col:15s} vs WQI: r={corr:.4f}")

# Save as temp npz (in datasets folder - for _run_multi_dataset.py compatibility)
for sp in [os.path.join(DATA_DIR, '4_akh_wqi_logcoliforms.npz'),
           os.path.join(SCRIPT_DATA_DIR, '4_akh_wqi_logcoliforms.npz')]:
    np.savez_compressed(sp, X=X, y=y, name='AKH_WQI (log-Coliforms)', 
                        target_name='WQI', n_features=10)
    print(f"Saved: {sp}")

# Quick RandomForest baseline
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score

rf = Pipeline([
    ('scaler', StandardScaler()),
    ('rf', RandomForestRegressor(n_estimators=200, random_state=1))
])
scores = cross_val_score(rf, X, y.ravel(), cv=5, scoring='r2')
print(f"RandomForest R2CV: {scores.mean():.4f} ± {scores.std():.4f}")

# Linear baseline
from sklearn.linear_model import LinearRegression
lr = Pipeline([
    ('scaler', StandardScaler()),
    ('lr', LinearRegression())
])
scores_lr = cross_val_score(lr, X, y.ravel(), cv=5, scoring='r2')
print(f"LinearRegression R2CV: {scores_lr.mean():.4f} ± {scores_lr.std():.4f}")