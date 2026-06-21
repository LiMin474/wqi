import numpy as np


def a2_GWQI(X, standard):
    valid = ~np.any(np.isnan(standard), axis=0)
    BIS = standard[0, valid]
    wt = standard[1, valid]
    X = X[:, valid]

    wj = wt / np.sum(wt)
    qj = (X / BIS) * 100

    pos1 = 0
    pos2 = 2
    pH = X[:, pos1]
    DO = X[:, pos2]

    qj[:, pos1] = np.abs((pH - 7) / (8.5 - 7)) * 100
    qj[:, pos2] = ((14.6 - DO) / (14.6 - 5)) * 100

    QI = qj * wj
    QI = np.sum(QI, axis=1)
    return QI