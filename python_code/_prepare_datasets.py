import numpy as np
import pandas as pd
import os
import sys
import warnings
warnings.filterwarnings('ignore')

SAVE_DIR = os.path.dirname(os.path.abspath(__file__))
os.makedirs(os.path.join(SAVE_DIR, 'datasets'), exist_ok=True)


def prepare_jajpur():
    print('=' * 60)
    print('Dataset 1: Jajpur Groundwater (Original)')
    print('=' * 60)

    sys.path.insert(0, SAVE_DIR)
    from data_loader import load_X0, load_GQ

    X0 = load_X0()
    GQ = load_GQ()

    if X0 is None:
        csv_path = os.path.join(SAVE_DIR, 'X0_data.csv')
        X0 = pd.read_csv(csv_path)

    if isinstance(X0, pd.DataFrame):
        X = X0.values.astype(float)
    else:
        X = np.asarray(X0, dtype=float)

    y = np.asarray(GQ, dtype=float).ravel()

    print(f'  Samples: {len(y)}')
    print(f'  Features: {X.shape[1]}')
    print(f'  Target (WQI): mean={y.mean():.2f}, std={y.std():.2f}, min={y.min():.2f}, max={y.max():.2f}')

    np.savez(os.path.join(SAVE_DIR, 'datasets', '1_jajpur.npz'), X=X, y=y,
             name='Jajpur Groundwater', target_name='GWQI', n_features=X.shape[1])
    print(f'  Saved: datasets/1_jajpur.npz')
    return X, y


def prepare_indian_water_quality():
    print()
    print('=' * 60)
    print('Dataset 2: Indian River Water Quality (IJERPH 2022)')
    print('=' * 60)

    kaggle_path = os.path.join(os.path.expanduser('~'), '.cache', 'kagglehub',
                               'datasets', 'anbarivan', 'indian-water-quality-data',
                               'versions', '4', 'water_dataX.csv')

    df = pd.read_csv(kaggle_path, encoding='latin-1')

    param_cols = ['Temp', 'D.O. (mg/l)', 'PH', 'CONDUCTIVITY (\xb5mhos/cm)',
                  'B.O.D. (mg/l)', 'NITRATENAN N+ NITRITENANN (mg/l)',
                  'FECAL COLIFORM (MPN/100ml)', 'TOTAL COLIFORM (MPN/100ml)Mean']

    df_clean = df[param_cols].copy()
    for col in param_cols:
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')

    df_clean = df_clean.dropna()

    X_raw = df_clean.values.astype(float)
    print(f'  Raw samples: {X_raw.shape[0]}, features: {X_raw.shape[1]}')

    BIS = np.array([25.0, 5.0, 8.5, 500.0, 5.0, 45.0, 500.0, 5000.0])
    wt = 1.0 / BIS
    wt = wt / wt.sum()

    pH = X_raw[:, 2]
    DO = X_raw[:, 1]
    qj = (X_raw / BIS) * 100
    qj[:, 2] = np.abs((pH - 7) / (8.5 - 7)) * 100
    qj[:, 1] = np.clip((14.6 - DO) / (14.6 - 5), 0, None) * 100
    qj = np.clip(qj, 0, 100)

    QI = np.sum(qj * wt, axis=1)

    y = QI
    X = X_raw

    print(f'  Calculated WQI: mean={y.mean():.2f}, std={y.std():.2f}, min={y.min():.2f}, max={y.max():.2f}')

    np.savez(os.path.join(SAVE_DIR, 'datasets', 'indian_river.npz'), X=X, y=y,
             name='Indian River Water Quality', target_name='WQI', n_features=X.shape[1],
             feature_names=param_cols)
    print(f'  Saved: datasets/indian_river.npz')
    return X, y


def prepare_uci_water_quality():
    print()
    print('=' * 60)
    print('Dataset 3: UCI Water Quality Prediction (pH)')
    print('=' * 60)

    import scipy.io as sio

    kaggle_path = os.path.join(os.path.expanduser('~'), '.cache', 'kagglehub',
                               'datasets', 'parmajha', 'water-quality-prediction-dataset',
                               'versions', '1', 'water_dataset.mat')

    mat = sio.loadmat(kaggle_path, squeeze_me=True)
    X_tr = mat['X_tr']
    Y_tr = mat['Y_tr']
    X_te = mat['X_te']
    Y_te = mat['Y_te']
    location_ids = mat['location_ids']

    feature_names = [str(f) for f in mat['features']]

    def flatten_data(X, Y, location_ids):
        rows = []
        for day_idx in range(len(X)):
            day_matrix = X[day_idx]
            for loc_idx, loc_id in enumerate(location_ids):
                features = day_matrix[loc_idx]
                if hasattr(features, 'tolist'):
                    features = features.tolist()
                elif np.isscalar(features):
                    features = [float(features)]
                else:
                    features = list(features)
                target = float(Y[loc_idx, day_idx])
                row = [int(loc_id), int(day_idx)] + features + [target]
                rows.append(row)
        return rows

    rows_tr = flatten_data(X_tr, Y_tr, location_ids)
    rows_te = flatten_data(X_te, Y_te, location_ids)
    all_rows = rows_tr + rows_te

    columns = ['location_id', 'day_index'] + feature_names + ['pH']
    df_all = pd.DataFrame(all_rows, columns=columns)

    feature_cols = feature_names
    X = df_all[feature_cols].values.astype(float)
    y = df_all['pH'].values.astype(float)

    y = y * 14.0

    print(f'  Samples: {len(y)}')
    print(f'  Features: {X.shape[1]}')
    print(f'  Target (pH): mean={y.mean():.2f}, std={y.std():.2f}, min={y.min():.2f}, max={y.max():.2f}')

    np.savez(os.path.join(SAVE_DIR, 'datasets', 'uci_water_quality.npz'), X=X, y=y,
             name='UCI Water Quality Prediction', target_name='pH', n_features=X.shape[1],
             feature_names=feature_names)
    print(f'  Saved: datasets/uci_water_quality.npz')
    return X, y


if __name__ == '__main__':
    prepare_jajpur()
    prepare_indian_water_quality()
    prepare_uci_water_quality()
    print()
    print('All datasets prepared successfully!')