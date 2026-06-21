import numpy as np
from skopt import gp_minimize
from skopt.space import Integer, Categorical
from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import KFold, cross_val_score
import os
import warnings
warnings.filterwarnings('ignore')


def SumSqr(params, XX, YY, cvss, j):
    np.random.seed(j)
    ind = np.array([True] * XX.shape[1])

    var_indices_str = [f'var{i+1}' for i in range(12)]
    ind = np.array([params[i] == 'true' if i < 12 else True for i in range(len(params))])
    ind = ind[:12] if len(ind) > 12 else ind

    if np.sum(ind) < 2:
        return 1.0, None

    n_estimators = int(params[-1]) if len(params) > 15 else int(params[3])
    min_leaf = int(params[12]) if len(params) > 12 else 5
    max_splits = int(params[13]) if len(params) > 13 else 10
    n_features = int(params[14]) if len(params) > 14 else 6

    base_tree = DecisionTreeRegressor(
        min_samples_leaf=min_leaf,
        max_depth=max_splits,
        max_features=n_features,
        random_state=j
    )

    Mdl = BaggingRegressor(
        estimator=base_tree,
        n_estimators=n_estimators,
        random_state=j,
        bootstrap=True
    )

    Mdl.fit(XX[:, ind], YY)

    SST = np.sum((YY - np.mean(YY))**2)
    y_pred = Mdl.predict(XX[:, ind])
    SSEmdl = np.sum((YY - y_pred)**2)

    cv_scores = cross_val_score(Mdl, XX[:, ind], YY, cv=cvss, scoring='neg_mean_squared_error')
    SSEcv = -np.mean(cv_scores) * len(YY) * 5
    R2CV = 1 - (SSEcv / SST)
    R2 = 1 - (SSEmdl / SST)

    output = {'R2': R2, 'R2CV': R2CV, 'Mdl': Mdl}
    target = 1 - R2CV
    return target, output


def Optopt():
    dimensions = []
    for i in range(12):
        dimensions.append(Categorical(['true', 'false'], name=f'var{i+1}'))
    dimensions.append(Integer(1, 37, name='MinLeafSize'))
    dimensions.append(Integer(1, 73, name='MaxNumSplits'))
    dimensions.append(Integer(1, 12, name='NumVariablesToSample'))
    dimensions.append(Integer(10, 100, name='NumLearningCycles'))
    dimensions.append(Categorical(['Bag', 'LSBoost'], name='Method'))
    dimensions.append(Categorical(['0.001'], name='LearnRate'))
    return dimensions


def c0_RF_bayesian_20_times(Pred, Resp):
    numFolds = 5
    fname = 'RF_bayesian_predictor_result.npz'
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    save_dir = os.path.join(base_dir, 'saved_models')
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, fname)

    all_tabs = []

    for j in range(20):
        np.random.seed(j)
        kf = KFold(n_splits=numFolds, shuffle=True, random_state=j)
        cvss = list(kf.split(Pred))

        dimensions = Optopt()

        def objective(params):
            target, _ = SumSqr(params, Pred, Resp, cvss, j)
            return target

        res = gp_minimize(
            objective,
            dimensions,
            acq_func='EI',
            n_calls=60,
            random_state=j,
            verbose=False
        )

        best_params = res.x
        target, output = SumSqr(best_params, Pred, Resp, cvss, j)

        Tab = {}
        for i in range(12):
            Tab[f'var{i+1}'] = best_params[i]
        Tab['MinLeafSize'] = int(best_params[12])
        Tab['MaxNumSplits'] = int(best_params[13])
        Tab['NumVariablesToSample'] = int(best_params[14])
        Tab['NumLearningCycles'] = int(best_params[15])
        Tab['Method'] = best_params[16]
        Tab['R2'] = output['R2']
        Tab['CVR2'] = output['R2CV']
        all_tabs.append(Tab)

    mytable = np.array(all_tabs)
    np.savez(filepath, mytable=mytable)
    print('')