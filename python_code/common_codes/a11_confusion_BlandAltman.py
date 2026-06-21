import numpy as np
import matplotlib.pyplot as plt


def a11_confusion_BlandAltman(data):
    edges = [0, 50, 100, 150, 200, np.inf]
    WQIclass = np.zeros_like(data)
    for i in range(data.shape[1]):
        WQIclass[:, i] = np.digitize(data[:, i], edges[1:-1]) + 1

    s = plt.figure()
    s.set_facecolor('white')
    name_labels = 'abcdefghij'

    n_methods = data.shape[1]
    n_plots = n_methods - 1 + n_methods - 1
    plot_idx = 1

    for i in range(1, n_methods):
        ax = s.add_subplot(2, 3, plot_idx)
        g1 = WQIclass[:, 0]
        g2 = WQIclass[:, i]
        n_classes = max(g1.max(), g2.max()) + 1
        C = np.zeros((int(n_classes), int(n_classes)), dtype=int)
        for ii in range(len(g1)):
            C[int(g1[ii]), int(g2[ii])] += 1
        im = ax.imshow(C, cmap='Blues', aspect='auto')
        for ii in range(C.shape[0]):
            for jj in range(C.shape[1]):
                ax.text(jj, ii, str(C[ii, jj]), ha='center', va='center')
        ax.set_title(f'({name_labels[plot_idx-1]})')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Observed')
        plot_idx += 1

    plot_idx = 3
    for j in range(1, n_methods):
        plot_idx += 1
        if plot_idx > 6:
            continue
        ax = s.add_subplot(2, 3, plot_idx)
        data1 = data[:, 0]
        data2 = data[:, j]
        data_mean = np.mean([data1, data2], axis=0)
        data_diff = data1 - data2
        md = np.mean(data_diff)
        sd = np.std(data_diff, ddof=1)

        ax.scatter(data_mean, data_diff, alpha=0.7)
        ax.axhline(md, color='k', linestyle='-')
        ax.axhline(md + 1.96 * sd, color='r', linestyle='--')
        ax.axhline(md - 1.96 * sd, color='r', linestyle='--')
        ax.set_title(f'({name_labels[plot_idx-1]})')
        ax.set_xlabel('Mean of two measures')
        ax.set_ylabel('Diff. in methods')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return s