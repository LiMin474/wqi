import numpy as np
import pandas as pd


def a9_classified_confusion(X):
    edges = [0, 50, 100, 150, 200, np.inf]
    labels = [1, 2, 3, 4, 5]
    WQIclass = np.zeros_like(X)
    for i in range(X.shape[1]):
        WQIclass[:, i] = np.digitize(X[:, i], edges[1:-1]) + 1

    unique_rows, counts = np.unique(WQIclass, axis=0, return_counts=True)
    print('Unique classification patterns and their counts:')
    print(pd.DataFrame(np.column_stack([unique_rows, counts]),
                       columns=[f'Method_{j+1}' for j in range(X.shape[1])] + ['Count']))

    numMethods = X.shape[1]
    count_mat = np.zeros((len(labels), numMethods), dtype=int)
    for m in range(numMethods):
        count_mat[:, m] = np.histogram(X[:, m], bins=edges)[0]

    rowNames = [f"Class_{l}" for l in labels]
    colNames = [f"Method_{m+1}" for m in range(numMethods)]
    T = pd.DataFrame(count_mat, index=rowNames, columns=colNames)
    print(T)

    return WQIclass