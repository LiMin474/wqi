import numpy as np
import matplotlib.pyplot as plt
import os
import joblib


def MDL_PPE(MDL, X, Y, rept=100):
    num_pred = X.shape[1]
    lossmatrix = np.zeros((num_pred, rept))
    for i in range(num_pred):
        Xtemp = X.copy()
        val = X[:, i].copy()
        for j in range(rept):
            Xtemp[:, i] = np.random.permutation(val)
            pred = MDL.predict(Xtemp)
            loss = np.mean((pred - Y) ** 2)
            lossmatrix[i, j] = loss
    return lossmatrix


def MDL_tol(mdl, pred):
    A = []
    for i in range(100):
        err = 0.95 + 0.1 * np.random.rand(pred.shape[0], 1)
        newmat = pred * err
        A.append(mdl.predict(newmat))
    A = np.column_stack(A)
    alt_model = np.mean(A, axis=1)
    Ori_mod = mdl.predict(pred)
    Tol = ((alt_model - Ori_mod) / Ori_mod) * 100
    return Tol


def plotPPEsens(meanloss, stdloss, Tol, vars_list):
    var_labels = vars_list.copy()
    if len(var_labels) >= 12:
        var_labels[3] = 'F$^-$'
        var_labels[4] = 'Cl$^-$'
        var_labels[5] = 'NO$_3^-$'
        var_labels[6] = 'SO$_4^{2-}$'
        var_labels[7] = 'PO$_4^{3-}$'
        var_labels[11] = 'HCO$_3^-$'

    s = plt.figure()
    s.set_facecolor('white')
    titleLabels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']

    for i in range(3):
        ax1 = s.add_subplot(3, 2, i * 2 + 1)
        ax2 = s.add_subplot(3, 2, i * 2 + 2)

        x_pos = np.arange(len(vars_list))
        ax1.bar(x_pos, meanloss[i], color='steelblue')
        ax1.errorbar(x_pos, meanloss[i], yerr=stdloss[i], fmt='k.', capsize=3)
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(var_labels, fontsize=8, rotation=45, ha='right')
        ax1.set_ylabel('Increase in model MSE')
        ax1.set_ylim([0, 75])
        ax1.set_title(titleLabels[i * 2])
        ax1.grid(True, alpha=0.3)

        ax2.stem(range(len(Tol[i])), Tol[i])
        ax2.set_xlabel('Observations')
        ax2.set_ylabel('Deviation from model (%)')
        ax2.set_ylim([-10, 10])
        ax2.axhline(5, color='r', linestyle='--')
        ax2.axhline(-5, color='r', linestyle='--')
        ax2.legend(['Perturbed Model'], loc='lower left')
        ax2.set_title(titleLabels[i * 2 + 1])
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return s


def a12_Predimp(pred, Resp, vars_list, Modells_flm=None, Modells_ann=None, Modells_rf=None):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    save_dir = os.path.join(base_dir, 'saved_models')
    fname = os.path.join(save_dir, 'PPE_mean.npz')

    if not os.path.exists(fname):
        if Modells_flm is None:
            path_lm = os.path.join(save_dir, 'Modells_flm.pkl')
            Modells_flm = joblib.load(path_lm)
        if Modells_ann is None:
            path_ann = os.path.join(save_dir, 'Modells_ann.pkl')
            Modells_ann = joblib.load(path_ann)
        if Modells_rf is None:
            path_rf = os.path.join(save_dir, 'Modells_rf.pkl')
            Modells_rf = joblib.load(path_rf)

        mdl = [Modells_flm, Modells_ann, Modells_rf]
        n = len(mdl)
        loss = []
        meanloss = []
        stdloss = []
        Tol = []

        for i in range(n):
            A = MDL_PPE(mdl[i], pred, Resp, 100)
            B = np.mean(A, axis=1)
            C = np.std(A, axis=1, ddof=1)
            D = MDL_tol(mdl[i], pred)
            loss.append(A)
            meanloss.append(B)
            stdloss.append(C)
            Tol.append(D)

        np.savez(fname, loss=np.array(loss, dtype=object),
                 meanloss=np.array(meanloss, dtype=object),
                 stdloss=np.array(stdloss, dtype=object),
                 Tol=np.array(Tol, dtype=object),
                 vars=np.array(vars_list, dtype=object))
    else:
        data = np.load(fname, allow_pickle=True)
        meanloss = data['meanloss']
        stdloss = data['stdloss']
        Tol = data['Tol']
        vars_list = data['vars'].tolist()

    fig = plotPPEsens(meanloss, stdloss, Tol, vars_list)
    return fig