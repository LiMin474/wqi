import numpy as np
import matplotlib.pyplot as plt
import os


def plotheatmap(mytable_dict, rept=20):
    gg = plt.figure()
    gg.set_facecolor('white')
    name_labels = 'ab'

    for idx, key in enumerate(mytable_dict.keys()):
        A = mytable_dict[key][:rept]
        cvr_idx = A.dtype.names.index('CVR2') if 'CVR2' in A.dtype.names else -1
        if cvr_idx >= 0:
            sort_idx = np.argsort(-A['CVR2'])
            A = A[sort_idx]

        X = np.zeros((len(A), 12), dtype=float)
        for i in range(len(A)):
            row = A[i]
            for j in range(12):
                if f'var{j+1}' in row.dtype.names:
                    val = row[f'var{j+1}']
                    X[i, j] = 0.0 if val in ['false', False, 0] else 1.0

        CVR = A['CVR2']

        ax = gg.add_subplot(1, 2, idx + 1)
        im = ax.imshow(X, aspect='auto', cmap='Blues', interpolation='nearest')
        ax.set_xticks(range(12))
        ax.set_xticklabels([f'v$_{{{j+1}}}$' for j in range(12)], fontsize=8)
        y_ticks = np.linspace(0, len(CVR) - 1, min(10, len(CVR)), dtype=int)
        ax.set_yticks(y_ticks)
        ax.set_yticklabels([f'{CVR[i]:.2f}' for i in y_ticks], fontsize=8)
        ax.set_title(f'({name_labels[idx]})')
        ax.set_xlabel('Predictors')
        ax.set_ylabel('R$^2_{CV}$')

    plt.tight_layout()
    return gg


def plotscatter(models):
    titles = 'abcdefgh'
    ff = plt.figure()
    ff.set_facecolor('white')

    for i in range(len(models)):
        ax = ff.add_subplot(2, 4, i + 1)
        tb = models[i]
        if len(tb) == 0:
            continue
        tb_arr = np.array(tb)
        ax.plot(tb_arr[:, 0], tb_arr[:, 2], '-o', label='R$^2_{cv}$')
        ax.plot(tb_arr[:, 0], tb_arr[:, 1], '-o', label='R$^2$')
        ax.set_xlim(ax.get_xlim()[::-1])
        ax.set_ylim([0.6, 1.1])
        ax.set_xlabel('Feature set size')
        ax.set_ylabel('Model performance')
        ax.set_title(f'({titles[i]})')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return ff


def c3_figure_feature_selection():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    save_dir = os.path.join(base_dir, 'saved_models')
    os.makedirs(save_dir, exist_ok=True)

    try:
        data_rf = np.load(os.path.join(save_dir, 'RF_bayesian_predictor_result.npz'), allow_pickle=True)
        resp_rf = data_rf['mytable']
    except (FileNotFoundError, IOError):
        resp_rf = np.array([])

    try:
        data_ann = np.load(os.path.join(save_dir, 'Fitrnet_bayesian_predictor_result.npz'), allow_pickle=True)
        resp_ann = data_ann['mytable']
    except (FileNotFoundError, IOError):
        resp_ann = np.array([])

    str_dict = {'resp1': resp_ann, 'resp2': resp_rf}

    if len(resp_ann) > 0:
        final_A = np.column_stack([
            np.array([np.sum([row[f'var{j+1}'] == 'true' for j in range(12)]) for row in resp_ann]),
            np.array([row['R2'] if 'R2' in row.dtype.names else 0 for row in resp_ann]),
            np.array([row['CVR2'] if 'CVR2' in row.dtype.names else 0 for row in resp_ann])
        ])
        sort_idx = np.argsort(-final_A[:, 2])
        fin_A = final_A[sort_idx]
        _, id1 = np.unique(fin_A[:, 0], return_index=True)
        ANN_beys = fin_A[np.sort(id1)]
    else:
        ANN_beys = np.empty((0, 3))

    if len(resp_rf) > 0:
        final_R = np.column_stack([
            np.array([np.sum([row[f'var{j+1}'] == 'true' for j in range(12)]) for row in resp_rf]),
            np.array([row['R2'] if 'R2' in row.dtype.names else 0 for row in resp_rf]),
            np.array([row['CVR2'] if 'CVR2' in row.dtype.names else 0 for row in resp_rf])
        ])
        sort_idx = np.argsort(-final_R[:, 2])
        fin_R = final_R[sort_idx]
        _, id2 = np.unique(fin_R[:, 0], return_index=True)
        RF_beys = fin_R[np.sort(id2)]
    else:
        RF_beys = np.empty((0, 3))

    try:
        data_ppe = np.load(os.path.join(save_dir, 'features_ppe.npz'), allow_pickle=True)
        PPE_rf = data_ppe['myT_RF'].tolist()
        PPE_ann = data_ppe['myT_ann'].tolist()
        ANN_PPE = np.array([[r['Size'], r['R2'], r['R2CV']] for r in PPE_ann]) if len(PPE_ann) > 0 else np.empty((0, 3))
        RF_PPE = np.array([[r['Size'], r['R2'], r['R2CV']] for r in PPE_rf]) if len(PPE_rf) > 0 else np.empty((0, 3))
    except (FileNotFoundError, IOError):
        ANN_PPE = np.empty((0, 3))
        RF_PPE = np.empty((0, 3))

    try:
        data_mr = np.load(os.path.join(save_dir, 'features_mrmr.npz'), allow_pickle=True)
        ANN_MR = np.array([[r['Size'], r['R2'], r['R2CV']] for r in data_mr['myT_ann'].tolist()]) if len(data_mr['myT_ann'].tolist()) > 0 else np.empty((0, 3))
        RF_MR = np.array([[r['Size'], r['R2'], r['R2CV']] for r in data_mr['myT_RF'].tolist()]) if len(data_mr['myT_RF'].tolist()) > 0 else np.empty((0, 3))
    except (FileNotFoundError, IOError):
        ANN_MR = np.empty((0, 3))
        RF_MR = np.empty((0, 3))

    models = [ANN_beys, ANN_MR, ANN_PPE, RF_beys, RF_MR, RF_PPE]
    fig1 = plotscatter(models)

    if len(resp_ann) > 0 and len(resp_rf) > 0:
        fig2 = plotheatmap(str_dict, rept=20)
    else:
        fig2 = None

    return fig1, fig2