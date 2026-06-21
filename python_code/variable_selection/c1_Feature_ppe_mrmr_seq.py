import numpy as np
from sklearn.feature_selection import mutual_info_regression
import os
import sys


def mysequential(Xtr, Ytr):
    from variable_selection.c2_Bayesopt_rf_for_varselect import c2_Bayesopt_rf_for_varselect

    output = c2_Bayesopt_rf_for_varselect(Xtr, Ytr)
    R2cvf = output[1]
    n = Xtr.shape[1]
    selected = list(range(n))

    R2cvi = 0
    filename = 'Sequential.npz'
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    save_dir = os.path.join(base_dir, 'saved_models')
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, filename)

    rf_history = []

    while R2cvf > 0.90 * R2cvi:
        count = len(selected)
        result = []

        for i in range(count):
            v = selected.copy()
            v.pop(i)
            output = c2_Bayesopt_rf_for_varselect(Xtr[:, v], Ytr)
            result.append({'v': v, 'R2': output[0], 'CVR2': output[1]})

        result.sort(key=lambda x: x['CVR2'], reverse=True)
        Newv = result[0]['v']
        newR2 = result[0]['R2']
        newCVR2 = result[0]['CVR2']

        if newCVR2 > 0.90 * R2cvf:
            R2cvi = R2cvf
            R2cvf = newCVR2
            selected = Newv
            newHistory = {'selected': selected, 'R2': newR2, 'CVR2': newCVR2}
            rf_history.append(newHistory)
        else:
            break

    rf_history_arr = np.array(rf_history)
    np.savez(filepath, rf_history=rf_history_arr)
    return rf_history


def feature_selection_ppe_mrmr(ppe, X0, GQ, Method, filename):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    save_dir = os.path.join(base_dir, 'saved_models')
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, filename)

    from variable_selection.c2_Bayesopt_ann_for_varselect import c2_Bayesopt_ann_for_varselect
    from variable_selection.c2_Bayesopt_rf_for_varselect import c2_Bayesopt_rf_for_varselect

    if ppe.ndim == 2 and ppe.shape[1] == 2:
        id_ann = ppe[:, 0]
        id_RF = ppe[:, 1]
    else:
        id_ann = ppe.flatten()
        id_RF = ppe.flatten()

    try:
        data = np.load(filepath, allow_pickle=True)
        myT_RF = data['myT_RF'].tolist()
        myT_ann = data['myT_ann'].tolist()
    except (FileNotFoundError, IOError):
        myT_RF = []
        myT_ann = []

    X0_arr = X0 if isinstance(X0, np.ndarray) else X0.values

    for i in range(10):
        vars_ann = id_ann[:2 + i]
        vars_ann = vars_ann[vars_ann < X0_arr.shape[1]]
        if len(vars_ann) == 0:
            continue
        output_ann = c2_Bayesopt_ann_for_varselect(X0_arr[:, vars_ann], GQ)
        newRow_ann = {'Method': Method, 'Model': output_ann[2], 'Size': len(vars_ann),
                       'variables': vars_ann, 'R2': output_ann[0], 'R2CV': output_ann[1]}
        myT_ann.append(newRow_ann)

        vars_RF = id_RF[:2 + i]
        vars_RF = vars_RF[vars_RF < X0_arr.shape[1]]
        if len(vars_RF) == 0:
            continue
        output_rf = c2_Bayesopt_rf_for_varselect(X0_arr[:, vars_RF], GQ)
        newRow_rf = {'Method': Method, 'Model': output_rf[2], 'Size': len(vars_RF),
                      'variables': vars_RF, 'R2': output_rf[0], 'R2CV': output_rf[1]}
        myT_RF.append(newRow_rf)

    np.savez(filepath, myT_RF=np.array(myT_RF, dtype=object),
             myT_ann=np.array(myT_ann, dtype=object))


def c1_Feature_ppe_mrmr_seq(X0, GQ):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    save_dir = os.path.join(base_dir, 'saved_models')

    X0_arr = X0 if isinstance(X0, np.ndarray) else X0.values

    ppe_data = np.load(os.path.join(save_dir, 'PPE_mean.npz'), allow_pickle=True)
    meanloss = ppe_data['meanloss']

    idxmrmr = np.argsort(mutual_info_regression(X0_arr, GQ))[::-1]

    id_ann = np.argsort(meanloss[1])[::-1]
    id_RF = np.argsort(meanloss[2])[::-1]

    mysequential(X0_arr, GQ)

    feature_selection_ppe_mrmr(idxmrmr, X0_arr, GQ, 'mrmr', 'features_mrmr.npz')

    ppe_indices = np.column_stack([id_ann, id_RF])
    feature_selection_ppe_mrmr(ppe_indices, X0_arr, GQ, 'ppe', 'features_ppe.npz')