import numpy as np
import os


def a3_bayesvarselect_repetation(GQ, X0):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.join(base_dir, 'variable_selection'))

    data = np.column_stack([X0, GQ])
    n = data.shape[0]
    rng = np.random.default_rng(42)

    for i in range(5):
        idx = rng.permutation(n)
        new_data = data[idx[:round(0.9 * n)], :]

        from variable_selection.c0_RF_bayesian_20_times import c0_RF_bayesian_20_times
        c0_RF_bayesian_20_times(new_data[:, :12], new_data[:, 12])
    print('')