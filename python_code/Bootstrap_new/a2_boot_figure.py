import numpy as np
import matplotlib.pyplot as plt
import os


def a2_boot_figure(GQ, X0):
    from common_codes.a7_all_mdl_prediction import a7_all_mdl_prediction
    pred1, pred2, pred3 = a7_all_mdl_prediction(X0)
    pred_all = np.column_stack([pred1, pred2, pred3])
    a = np.arange(1, pred_all.shape[0] + 1)

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    save_dir = os.path.join(base_dir, 'saved_models')
    data = np.load(os.path.join(save_dir, 'Bootconf.npz'), allow_pickle=True)
    Boot_main = data['Boot_main']

    std_lin = np.std(Boot_main[0], axis=1, ddof=1)
    std_ann = np.std(Boot_main[1], axis=1, ddof=1)
    std_rf = np.std(Boot_main[2], axis=1, ddof=1)
    std_all = np.column_stack([std_lin, std_ann, std_rf])

    s = plt.figure()
    s.set_facecolor('white')
    names = ['(a)', '(b)', '(c)']

    for i in range(pred_all.shape[1]):
        ax = s.add_subplot(1, 3, i + 1)
        ax.scatter(a, pred_all[:, i], marker='o', color='b', alpha=0.7)
        ax.errorbar(a, pred_all[:, i], yerr=std_all[:, i],
                     fmt='none', capsize=3, color='r', linewidth=1)
        ax.set_title(names[i])
        ax.set_xlabel('Sample ID')
        ax.set_ylabel('Model predicted GWQI')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return s