import numpy as np
import matplotlib.pyplot as plt
import os


def cart_map(mypolyLat, mypolyLon, mod_Lat, mod_Lon, conc, explat, explon):
    list_vars = ['Cl', 'SO4', 'PO4', 'U', 'CaH', 'MgH', 'HCO3']
    name_labels = 'abcdefg'

    s = plt.figure(figsize=(12, 10))
    s.set_facecolor('white')

    for j in range(7):
        ax = s.add_subplot(3, 3, j + 1)
        var_name = list_vars[j]

        if isinstance(conc, dict) and var_name in conc:
            Z = conc[var_name]
        else:
            Z = np.full(mod_Lat.shape, np.nan)

        pc = ax.pcolormesh(mod_Lon, mod_Lat, Z, cmap='jet', shading='flat')
        plt.colorbar(pc, ax=ax, fraction=0.046, pad=0.04)

        if len(mypolyLat) > 2:
            ax.plot(mypolyLon, mypolyLat, 'k-', linewidth=1)

        ax.scatter(explon, explat, s=10, c='black', marker='o', zorder=5)

        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_title(f'({name_labels[j]})')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return s


def b3_IDW_plots():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    save_dir = os.path.join(base_dir, 'saved_models')

    try:
        data = np.load(os.path.join(save_dir, 'geoplott_2.npz'), allow_pickle=True)
        jprLat = data['jprLat']
        jprLon = data['jprLon']
        mod_Lat = data['mod_Lat']
        mod_Lon = data['mod_Lon']
        mod_conc = data['mod_conc'].item()
        explat = data['explat']
        explon = data['explon']
    except FileNotFoundError:
        data = np.load(os.path.join(save_dir, 'geoplott_3.npz'), allow_pickle=True)
        jprLat = data['jprLat']
        jprLon = data['jprLon']
        mod_Lat = data['mod_Lat']
        mod_Lon = data['mod_Lon']
        mod_conc = data['mod_conc'].item()
        explat = data['explat']
        explon = data['explon']

    fig = cart_map(jprLat, jprLon, mod_Lat, mod_Lon, mod_conc, explat, explon)
    return fig