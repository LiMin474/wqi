import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import os


def b1_sampling_plot(filename, wqdata):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    map_dir = os.path.dirname(base_dir)
    shapefile_path = os.path.join(map_dir, 'Map Codes', filename)
    shp_path = shapefile_path.replace('.dbf', '.shp')

    if not os.path.exists(shp_path):
        shp_path = os.path.join(map_dir, 'Map Codes', 'myshape1.shp')

    GT0 = gpd.read_file(shp_path)

    if 'District' in GT0.columns:
        GT = GT0[GT0['District'].str.upper().str.contains('JAJAPUR|J>JAPUR', na=False)]
    else:
        GT = GT0

    GT = GT.to_crs('EPSG:4326')

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection': ccrs.PlateCarree()})
    fig.set_facecolor('white')

    ax.add_feature(cfeature.LAND, facecolor='white')
    ax.add_feature(cfeature.OCEAN, facecolor='lightblue')
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)

    GT.boundary.plot(ax=ax, color='black', linewidth=1, transform=ccrs.PlateCarree())

    if wqdata is not None:
        if isinstance(wqdata, np.ndarray):
            lat_col = 0 if 'Latitude' not in str(wqdata.dtype) else None
            lon_col = 1
            ax.scatter(wqdata[:, lon_col], wqdata[:, lat_col], c='red',
                       marker='o', s=30, transform=ccrs.PlateCarree(), edgecolors='blue')
        else:
            ax.scatter(wqdata['Longitude'], wqdata['Latitude'], c='red',
                       marker='o', s=30, transform=ccrs.PlateCarree(), edgecolors='blue',
                       zorder=5)

            labels = np.arange(1, len(wqdata) + 1)
            for i in range(len(wqdata)):
                ax.text(wqdata['Longitude'].iloc[i], wqdata['Latitude'].iloc[i],
                        str(labels[i]), fontsize=8, ha='right',
                        transform=ccrs.PlateCarree())

    if 'TEHSIL' in GT.columns:
        centroids = GT.geometry.centroid
        for idx, row in GT.iterrows():
            name = str(row['TEHSIL']).replace('J>JAPUR', 'JAJAPUR')
            ax.text(centroids[idx].x, centroids[idx].y, name,
                    fontsize=10, color='blue', ha='center',
                    transform=ccrs.PlateCarree())

    ax.set_title('Sampling Locations')
    gl = ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
    gl.top_labels = False
    gl.right_labels = False

    return fig