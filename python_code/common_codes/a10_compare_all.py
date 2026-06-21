import numpy as np
import matplotlib.pyplot as plt


def a10_compare_all(All, name):
    a = np.arange(1, All.shape[0] + 1)
    s = plt.figure()
    s.set_facecolor('white')
    colors = ['r', 'g', 'b', 'y', 'c', 'm', 'k']

    for i in range(All.shape[1]):
        plt.plot(a, All[:, i], color=colors[i % len(colors)], marker='o',
                 linestyle=':', markerfacecolor=colors[i % len(colors)],
                 label=name[i] if i < len(name) else f'Method {i+1}')

    plt.legend()
    plt.axhline(y=50, color='g', linestyle='--', alpha=0.5)
    plt.axhline(y=100, color='g', linestyle='--', alpha=0.5)
    plt.axhline(y=150, color='g', linestyle='--', alpha=0.5)
    plt.xlabel('Sample ID')
    plt.ylabel('WQI value')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    return s