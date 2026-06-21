import numpy as np
import matplotlib.pyplot as plt
import os


def c5_model_predict_regression():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    save_dir = os.path.join(base_dir, 'saved_models')

    data_output = np.load(os.path.join(save_dir, 'RF_mdl_beys.npz'), allow_pickle=True)
    output = data_output['output'].item()
    model = output['Mdl']

    from data_loader import load_GQ, load_X0
    GQ = load_GQ()
    X0 = load_X0()
    X = X0 if isinstance(X0, np.ndarray) else X0.values

    data_rf = np.load(os.path.join(save_dir, 'RF_bayesian_predictor_result.npz'), allow_pickle=True)
    mytable = data_rf['mytable']

    cvr_idx = mytable.dtype.names.index('CVR2') if 'CVR2' in mytable.dtype.names else -1
    sort_idx = np.argsort(-mytable['CVR2']) if cvr_idx >= 0 else np.arange(len(mytable))
    mytable = mytable[sort_idx]

    z = mytable[0]
    ind = np.array([z[f'var{j+1}'] == 'true' for j in range(12)])
    pred = model.predict(X[:, ind])
    R2 = output['R2']

    fig = plt.figure()
    fig.set_facecolor('white')
    ft = 12

    ax1 = fig.add_subplot(1, 2, 1)
    xylim = [min(np.min(GQ), np.min(pred)), max(np.max(GQ), np.max(pred))]
    ax1.scatter(GQ, pred, c='green', alpha=0.7)
    ax1.plot(xylim, xylim, 'r-', linewidth=1)
    ax1.set_xlim(xylim)
    ax1.set_ylim(xylim)
    ax1.set_xlabel("Original GWQI", fontsize=ft)
    ax1.set_ylabel("RF Model WQI", fontsize=ft)
    ax1.set_title('(a)', fontsize=ft)
    txt = f'R$^2$: {np.round(R2, 3)}'
    ax1.text(np.mean(ax1.get_xlim()), 0.75 * np.mean(ax1.get_ylim()), txt, fontsize=ft)
    ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(1, 2, 2)
    resu = pred - GQ
    MAE = resu / np.std(resu, ddof=1)
    ax2.stem(range(len(GQ)), MAE)
    ax2.set_ylim([-4, 4])
    ax2.set_xlabel('Observations', fontsize=ft)
    ax2.set_ylabel('Residuals (standardized)', fontsize=ft)
    ax2.axhline(3, color='r', linestyle='--')
    ax2.axhline(-3, color='r', linestyle='--')
    ax2.set_title('(b)', fontsize=ft)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig