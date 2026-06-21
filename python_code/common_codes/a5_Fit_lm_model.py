import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
import statsmodels.api as sm


def a5_Fit_lm_model(Pred, Resp):
    Resp = Resp.ravel()
    columnNames = ['Model', 'R2', 'adj-R2', 'AIC', 'R2CV']
    performance = pd.DataFrame(columns=columnNames)

    X = sm.add_constant(Pred)
    lm_sm = sm.OLS(Resp, X).fit()
    R2 = lm_sm.rsquared
    adj_R2 = lm_sm.rsquared_adj
    AIC = lm_sm.aic

    kf = KFold(n_splits=5, shuffle=True, random_state=1)
    yPredCV = np.zeros(len(Resp))
    for train_idx, test_idx in kf.split(Pred):
        X_train, X_test = Pred[train_idx], Pred[test_idx]
        y_train = Resp[train_idx]
        mdl_cv = LinearRegression().fit(X_train, y_train)
        yPredCV[test_idx] = mdl_cv.predict(X_test).ravel()

    SSE = np.sum((yPredCV - Resp)**2)
    SST = np.sum((np.mean(Resp) - Resp)**2)
    R2CV = 1 - (SSE / SST)

    performance.loc[0] = ['linear', R2, adj_R2, AIC, R2CV]

    Modells_flm = LinearRegression()
    Modells_flm.fit(Pred, Resp)

    return Modells_flm, performance