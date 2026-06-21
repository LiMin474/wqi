import numpy as np
import pandas as pd
import os


def a4_bayesselect_variables(GQ, X0, vars_list):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    save_dir = os.path.join(base_dir, 'saved_models')
    data = np.load(os.path.join(save_dir, 'RF_bayesian_predictor_result.npz'), allow_pickle=True)
    mytable = data['mytable']

    table1 = mytable[:20]
    table2 = mytable[20:40]
    table3 = mytable[40:60]
    table4 = mytable[60:80]
    table5 = mytable[80:100]
    table_list = [table1, table2, table3, table4, table5]

    ret_feature = []
    parameters = []

    for i in range(5):
        tab = sorted(table_list[i], key=lambda x: x['CVR2'], reverse=True)
        z = tab[0]
        ind = np.array([z[f'var{j+1}'] == 'true' for j in range(12)])
        para = {'MinLeafSize': z['MinLeafSize'], 'MaxNumSplits': z['MaxNumSplits'],
                'NumVariablesToSample': z['NumVariablesToSample'], 'NumLearningCycles': z['NumLearningCycles']}
        features = [vars_list[j] for j in range(12) if ind[j]]
        ret_feature.append(features)
        parameters.append(para)

    np.savez(os.path.join(save_dir, 'slected_features.npz'),
             ret_feature=np.array(ret_feature, dtype=object),
             parameters=np.array(parameters, dtype=object))
    print('')