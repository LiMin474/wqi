import numpy as np
from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import KFold, cross_val_score
import os


def SumSqr(z, XX, YY, cvss, j):
    np.random.seed(j)
    ind = np.array([z[f'var{j+1}'] == 'true' for j in range(12)])

    if np.sum(ind) < 2:
        return 1.0, None

    base_tree = DecisionTreeRegressor(
        min_samples_leaf=z['MinLeafSize'],
        max_depth=z['MaxNumSplits'],
        max_features=z['NumVariablesToSample'],
        random_state=j
    )

    Mdl = BaggingRegressor(
        estimator=base_tree,
        n_estimators=z['NumLearningCycles'],
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

    output = (R2, R2CV, Mdl)
    target = 1 - R2CV
    return target, output


def c6_RF_baysmodel_generate(Pred, Resp):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    save_dir = os.path.join(base_dir, 'saved_models')
    os.makedirs(save_dir, exist_ok=True)

    data = np.load(os.path.join(save_dir, 'RF_bayesian_predictor_result.npz'), allow_pickle=True)
    mytable = data['mytable']

    cvr_idx = mytable.dtype.names.index('CVR2') if 'CVR2' in mytable.dtype.names else -1
    sort_idx = np.argsort(-mytable['CVR2']) if cvr_idx >= 0 else np.arange(len(mytable))
    mytable = mytable[sort_idx]

    z = mytable[0]

    j = 1
    np.random.seed(j)
    kf = KFold(n_splits=5, shuffle=True, random_state=j)
    cvss = list(kf.split(Pred))

    target, output = SumSqr(z, Pred, Resp, cvss, j)

    output_dict = {'R2': output[0], 'R2CV': output[1], 'Mdl': output[2]}
    np.savez(os.path.join(save_dir, 'RF_mdl_beys.npz'), output=output_dict)

    return output_dict