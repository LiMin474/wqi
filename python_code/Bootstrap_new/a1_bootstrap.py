import numpy as np
import os
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor


def find_RMSE(mdl, X, Y):
    pred = mdl.predict(X)
    AE = np.abs(Y - pred)
    SE = AE ** 2
    MSE = np.mean(SE)
    RMSE = MSE ** 0.5
    return RMSE


def createmdl(j, Xdata, Ydata, Mdl, Opttable_rf):
    if j == 0:
        mdl = LinearRegression()
        mdl.fit(Xdata, Ydata)
    elif j == 1:
        layer_sizes = Mdl['LayerSizes']
        activation = Mdl['Activations']
        act_map = {'tanh': 'tanh', 'sigmoid': 'logistic', 'relu': 'relu'}
        mdl = MLPRegressor(
            hidden_layer_sizes=layer_sizes,
            activation=act_map.get(activation, 'relu'),
            max_iter=2000,
            random_state=1,
            early_stopping=True
        )
        mdl.fit(Xdata, Ydata)
    elif j == 2:
        base_tree = DecisionTreeRegressor(
            min_samples_leaf=Opttable_rf.get('MinLeafSize', 5),
            max_depth=Opttable_rf.get('MaxNumSplits', 10),
            max_features=Opttable_rf.get('NumVariablesToSample', 6),
            random_state=1
        )
        mdl = BaggingRegressor(
            estimator=base_tree,
            n_estimators=Opttable_rf.get('NumLearningCycles', 50),
            random_state=1,
            bootstrap=True
        )
        mdl.fit(Xdata, Ydata)
    return mdl


def a1_bootstrap(GQ, X0, Modells_flm, Modells_ann, Modells_rf, Opttable_rf):
    nB = 100
    nResp = 3
    Mdl = [Modells_flm, Modells_ann, Modells_rf]
    Boot_main = [None] * nResp

    for j in range(nResp):
        XX = X0
        YY = GQ

        nc = XX.shape[0]
        ns = XX.shape[0]

        YB_sample = np.zeros((ns, nB))
        RMSE_Boots = np.zeros(nB)

        rng = np.random.default_rng(42)
        for i in range(nB):
            idx = rng.choice(nc, size=round(0.9 * nc), replace=False)
            Xdata = XX[idx, :]
            Ydata = YY[idx]
            boot_model = createmdl(j, Xdata, Ydata, Mdl[j], Opttable_rf)
            RMSE = find_RMSE(boot_model, Xdata, Ydata)
            RMSE_Boots[i] = RMSE
            Yb_sample = boot_model.predict(XX)
            YB_sample[:, i] = Yb_sample

        Boot_main[j] = YB_sample

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    save_dir = os.path.join(base_dir, 'saved_models')
    os.makedirs(save_dir, exist_ok=True)
    np.savez(os.path.join(save_dir, 'Bootconf.npz'),
             Boot_main=np.array(Boot_main, dtype=object))
    return Boot_main