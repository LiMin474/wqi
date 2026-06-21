"""
Data loader for MATLAB .mat files.
Handles loading of arrays and tables from .mat files.
Tables saved from MATLAB as MCOS objects cannot be read by scipy.io.
Use export_data_to_csv.m in MATLAB to convert table data to CSV files.
"""
import scipy.io as sio
import numpy as np
import pandas as pd
import os
import warnings

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DATA_CACHE = {}


def _resolve_path(relative_path):
    full_path = os.path.join(_BASE_DIR, relative_path)
    if os.path.exists(full_path):
        return full_path
    alt_path = os.path.join(_BASE_DIR, os.path.basename(relative_path))
    if os.path.exists(alt_path):
        return alt_path
    return full_path


def _find_in_paths(filename):
    for root, dirs, files in os.walk(_BASE_DIR):
        if filename in files:
            return os.path.join(root, filename)
    return None


def load_GQ():
    b0_X_GQ_path = _find_in_paths('b0_X_GQ.mat')
    if b0_X_GQ_path is None:
        raise FileNotFoundError("b0_X_GQ.mat not found. Ensure it is in the project directory.")

    mat = sio.loadmat(b0_X_GQ_path, squeeze_me=True)
    if 'GQ' not in mat:
        raise KeyError("Variable 'GQ' not found in b0_X_GQ.mat")

    GQ = np.asarray(mat['GQ'], dtype=float).flatten()
    return GQ


def load_wqdata():
    csv_path = _resolve_path(os.path.join('python_code', 'wqdata.csv'))
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)

    csv_alt = os.path.join(_BASE_DIR, 'wqdata.csv')
    if os.path.exists(csv_alt):
        return pd.read_csv(csv_alt)

    warnings.warn("wqdata.csv not found. "
                  "Run export_data_to_csv.m in MATLAB to generate CSV files. "
                  "Falling back to DataFrame from available .mat data.")
    return None


def load_stdwt():
    csv_path = _resolve_path(os.path.join('python_code', 'stdwt.csv'))
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)

    csv_alt = os.path.join(_BASE_DIR, 'stdwt.csv')
    if os.path.exists(csv_alt):
        return pd.read_csv(csv_alt)

    warnings.warn("stdwt.csv not found. "
                  "Run export_data_to_csv.m in MATLAB to generate CSV files. "
                  "Returning None - BIS standards not available.")
    return None


def load_BISd():
    stdwt = load_stdwt()
    if stdwt is not None:
        return stdwt.values.astype(float)
    return None


def load_X0():
    csv_path = _resolve_path(os.path.join('python_code', 'X0_data.csv'))
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)

    wqdata = load_wqdata()
    if wqdata is not None:
        # Paper's 12 parameters: pH through HCO3 (indices 3:15)
        X0 = wqdata.iloc[:, 3:15]
        return X0

    warnings.warn("X0_data.csv not found and wqdata not available. "
                  "Cannot extract X0 predictor variables.")
    return None


def load_all_data():
    GQ = load_GQ()
    wqdata = load_wqdata()
    stdwt = load_stdwt()
    X0 = load_X0() if os.path.exists(_resolve_path(os.path.join('python_code', 'X0_data.csv'))) else None
    return GQ, X0, wqdata, stdwt


def save_model_results(name, data_dict):
    save_dir = _resolve_path(os.path.join('python_code', 'saved_models'))
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, f'{name}.npz')
    np.savez(filepath, **data_dict)
    return filepath


def load_model_results(name):
    save_dir = _resolve_path(os.path.join('python_code', 'saved_models'))
    filepath = os.path.join(save_dir, f'{name}.npz')
    if os.path.exists(filepath):
        return np.load(filepath, allow_pickle=True)
    return None