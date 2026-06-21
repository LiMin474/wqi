import numpy as np
import joblib
import os


def a7_all_mdl_prediction(X, Modells_flm=None, Modells_ann=None, Modells_rf=None):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    save_dir = os.path.join(base_dir, 'saved_models')

    if Modells_flm is None:
        path_lm = os.path.join(save_dir, 'Modells_flm.pkl')
        if os.path.exists(path_lm):
            Modells_flm = joblib.load(path_lm)
        else:
            raise FileNotFoundError("Modells_flm not found. Run a5_Fit_lm_model first.")

    if Modells_ann is None:
        path_ann = os.path.join(save_dir, 'Modells_ann.pkl')
        if os.path.exists(path_ann):
            Modells_ann = joblib.load(path_ann)
        else:
            raise FileNotFoundError("Modells_ann not found. Run a4_Bayesian_fitrnet_opt first.")

    if Modells_rf is None:
        path_rf = os.path.join(save_dir, 'Modells_rf.pkl')
        if os.path.exists(path_rf):
            Modells_rf = joblib.load(path_rf)
        else:
            raise FileNotFoundError("Modells_rf not found. Run a3_Bayesopt_rf_model first.")

    LQ = Modells_flm.predict(X)
    AQ = Modells_ann.predict(X)
    RQ = Modells_rf.predict(X)

    return LQ, AQ, RQ