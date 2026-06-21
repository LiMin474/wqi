import numpy as np
import pandas as pd


def a8_statcalculator(WQIs):
    exp = WQIs[:, 0]
    pred = WQIs[:, 1:]
    AE = np.abs(exp - pred[:, None] if pred.ndim == 1 else exp[:, None] - pred)
    AE = np.abs(exp[:, np.newaxis] - pred)
    MAE = np.mean(AE, axis=0)
    SE = AE ** 2
    MSE = np.mean(SE, axis=0)
    RMSE = MSE ** 0.5

    Res = np.vstack([MAE, RMSE])
    Res = pd.DataFrame(Res, index=['MAE', 'RMSE'], columns=['LWQI', 'AWQI', 'RWQI'])
    return Res