import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis


def a0_statistics_X(X):
    avg = np.mean(X, axis=0)
    MIN = np.min(X, axis=0)
    MAX = np.max(X, axis=0)
    Std_Dev = np.std(X, axis=0, ddof=1)
    Skewn = skew(X, axis=0, bias=False)
    kurt = kurtosis(X, axis=0, bias=False)
    Table = np.column_stack([MIN, MAX, avg, Std_Dev, Skewn, kurt])
    return Table