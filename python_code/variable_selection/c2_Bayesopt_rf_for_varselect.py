import numpy as np
from skopt import gp_minimize
from skopt.space import Integer, Categorical
from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import KFold, cross_val_score
import warnings
warnings.filterwarnings('ignore')


def SumSqr(params, XX, YY, cvss, Mtype, cond):
    n_estimators = int(params[0])
    min_samples_leaf = int(params[1])
    max_depth = int(params[2])
    max_features = int(params[3])

    base_tree = DecisionTreeRegressor(
        min_samples_leaf=min_samples_leaf,
        max_depth=max_depth,
        max_features=max_features,
        random_state=1
    )

    if Mtype == 'Bag':
        Mdl = BaggingRegressor(
            estimator=base_tree,
            n_estimators=n_estimators,
            random_state=1,
            bootstrap=True
        )
    else:
        Mdl = BaggingRegressor(
            estimator=base_tree,
            n_estimators=n_estimators,
            random_state=1,
            bootstrap=True
        )

    Mdl.fit(XX, YY)

    SST = np.sum((YY - np.mean(YY))**2)
    y_pred = Mdl.predict(XX)
    SSEmdl = np.sum((YY - y_pred)**2)

    cv_scores = cross_val_score(Mdl, XX, YY, cv=cvss, scoring='neg_mean_squared_error')
    SSEcv = -np.mean(cv_scores) * len(YY) * 5
    R2CV = 1 - (SSEcv / SST)

    if cond == 'final':
        R2 = 1 - (SSEmdl / SST)
        output = {'R2': R2, 'R2CV': R2CV, 'Mdl': Mdl}
        return 0.0, output
    else:
        target = 1 - R2CV
        return target, None


def Optopt(XX, YY, Modeltype):
    dimensions = [
        Integer(10, 100, name='NumLearningCycles'),
        Integer(1, 37, name='MinLeafSize'),
        Integer(1, 73, name='MaxNumSplits'),
        Integer(1, 12, name='NumVariablesToSample')
    ]
    return dimensions


def c2_Bayesopt_rf_for_varselect(X, Y):
    Mtype = "Bag"
    nfolds = 5
    np.random.seed(1)

    kf = KFold(n_splits=nfolds, shuffle=True, random_state=1)
    cvss = list(kf.split(X))

    dimensions = Optopt(X, Y, Mtype)

    def objective(params):
        target, _ = SumSqr(params, X, Y, cvss, Mtype, 'ita')
        return target

    res = gp_minimize(
        objective,
        dimensions,
        acq_func='EI',
        n_calls=60,
        random_state=1,
        verbose=False
    )

    best_params = res.x
    target, output = SumSqr(best_params, X, Y, cvss, Mtype, 'final')

    return output['R2'], output['R2CV'], output['Mdl']