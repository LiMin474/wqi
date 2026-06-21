import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
import statsmodels.api as sm
import os


def c4_stepwise_lm_model(Pred, Resp):
    columnNames = ['Model', 'R2', 'adj-R2', 'AIC', 'R2CV']
    dataTable = pd.DataFrame(columns=columnNames)

    X = sm.add_constant(Pred)
    mdl_lm = sm.OLS(Resp, X).fit()

    kf = KFold(n_splits=5, shuffle=True, random_state=1)
    yPredCV = np.zeros(len(Resp))
    for train_idx, test_idx in kf.split(Pred):
        X_train, X_test = Pred[train_idx], Pred[test_idx]
        y_train = Resp[train_idx]
        mdl_cv = LinearRegression().fit(X_train, y_train)
        yPredCV[test_idx] = mdl_cv.predict(X_test)

    SSE = np.sum((yPredCV - Resp)**2)
    SST = np.sum((np.mean(Resp) - Resp)**2)
    R2cv = 1 - (SSE / SST)

    data = {'Model': 'linear', 'R2': mdl_lm.rsquared, 'adj-R2': mdl_lm.rsquared_adj,
            'AIC': mdl_lm.aic, 'R2CV': R2cv}
    dataTable = pd.concat([dataTable, pd.DataFrame([data])], ignore_index=True)

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    save_dir = os.path.join(base_dir, 'saved_models')
    os.makedirs(save_dir, exist_ok=True)
    np.savez(os.path.join(save_dir, 'stepwiselm_model.npz'),
             model=mdl_lm, perf=dataTable.to_records())

    return mdl_lm, dataTable