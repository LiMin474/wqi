import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import zscore


def a1_heatmap_boxplot(X, vars_list):
    s = plt.figure()
    s.set_facecolor('white')

    ax1 = s.add_subplot(1, 2, 1)
    C = np.round(np.corrcoef(X.T), 3)
    im = ax1.imshow(C, cmap='jet', vmin=-1, vmax=1)
    ax1.set_xticks(range(len(vars_list)))
    ax1.set_yticks(range(len(vars_list)))
    var_labels = vars_list.copy()
    if len(var_labels) >= 12:
        var_labels[5] = 'NO$_3^-$'
        var_labels[3] = 'F$^-$'
        var_labels[4] = 'Cl$^-$'
        var_labels[6] = 'SO$_4^{2-}$'
        var_labels[7] = 'PO$_4^{3-}$'
        var_labels[11] = 'HCO$_3^-$'
    ax1.set_xticklabels(var_labels, fontsize=8)
    ax1.set_yticklabels(var_labels, fontsize=8)
    plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
    ax1.set_title('(a)')
    cbar = plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
    for i in range(len(vars_list)):
        for j in range(len(vars_list)):
            ax1.text(j, i, f'{C[i,j]:.2f}', ha='center', va='center', fontsize=6,
                     color='white' if abs(C[i,j]) > 0.5 else 'black')
    ax1.grid(False)

    ax2 = s.add_subplot(1, 2, 2)
    zWQ = zscore(X, axis=0, ddof=1)
    id_labels = ['pH', 'EC', 'DO', 'F$^-$', 'Cl$^-$', 'NO$_3^-$', 'SO$_4^{2-}$',
                 'PO$_4^{3-}$', 'U', 'Ca H', 'Mg H', 'HCO$_3^-$']
    if len(vars_list) == len(id_labels):
        display_vars = id_labels
    else:
        display_vars = vars_list
    bp = ax2.boxplot(zWQ, vert=False, patch_artist=True)
    ax2.set_yticklabels(display_vars, fontsize=9)
    ax2.set_title('(b)')
    ax2.set_xlabel('Z-score')
    ax2.grid(True, alpha=0.3)
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')

    plt.tight_layout()
    return s