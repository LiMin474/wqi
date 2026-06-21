import numpy as np
import os


def idw1(X0, F0, Xint, p=2, rad=np.inf, L=2):
    N = X0.shape[0]
    Q = Xint.shape[0]
    Fint = np.zeros(Q)

    for ipos in range(Q):
        DeltaX = X0 - Xint[ipos, :]
        DabsL = np.sum(np.abs(DeltaX) ** L, axis=1) ** (1 / L)
        DabsL[DabsL == 0] = np.finfo(float).eps
        DabsL[DabsL > rad] = np.inf

        W = 1.0 / (DabsL ** p)
        Fint[ipos] = np.sum(W * F0) / np.sum(W)

    return Fint


def gridpoint(jprLat, jprLon, mod_Lat, mod_Lon, expconc, explat, explon):
    list_vars = ['GQ', 'LQ', 'AQ', 'RQ']
    projct = {}

    for j in range(len(list_vars)):
        col_name = list_vars[j]
        if isinstance(expconc, dict) and col_name in expconc:
            col_data = expconc[col_name]
        elif hasattr(expconc, 'shape') and expconc.ndim == 2 and j < expconc.shape[1]:
            col_data = expconc[:, j]
        else:
            col_data = expconc if isinstance(expconc, np.ndarray) and expconc.ndim == 1 else np.zeros(mod_Lat.size)

        X0 = np.column_stack([explat, explon])
        Xint = np.column_stack([mod_Lat.ravel(), mod_Lon.ravel()])
        Fint = idw1(X0, col_data, Xint)
        intpconc = Fint.reshape(mod_Lat.shape)

        inPoly = np.zeros(mod_Lat.shape, dtype=bool)
        if len(jprLat) > 2:
            from matplotlib.path import Path
            poly_path = Path(np.column_stack([jprLon, jprLat]))
            points = np.column_stack([mod_Lon.ravel(), mod_Lat.ravel()])
            inPoly_flat = poly_path.contains_points(points)
            inPoly = inPoly_flat.reshape(mod_Lat.shape)

        intpconc[~inPoly] = np.nan
        projct[col_name] = intpconc

    return projct


def b2_IDW_extrapolation(wqdata):
    if wqdata is None:
        print("wqdata not available")
        return

    if isinstance(wqdata, np.ndarray):
        explat = wqdata[:, 0]
        explon = wqdata[:, 1]
    else:
        explat = wqdata['Latitude'].values
        explon = wqdata['Longitude'].values

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    map_dir = os.path.dirname(base_dir)
    shp_path = os.path.join(map_dir, 'Map Codes', 'myshape1.shp')

    try:
        import geopandas as gpd
        GT = gpd.read_file(shp_path)
        jajpur = GT.to_crs('EPSG:4326')
        bounds = jajpur.total_bounds
        boundary = jajpur.geometry.unary_union
        from shapely.geometry import Polygon

        if boundary.geom_type == 'Polygon':
            jprLat = np.array([boundary.exterior.xy[1]]).flatten()
            jprLon = np.array([boundary.exterior.xy[0]]).flatten()
        else:
            jprLat = np.array([boundary.convex_hull.exterior.xy[1]]).flatten()
            jprLon = np.array([boundary.convex_hull.exterior.xy[0]]).flatten()
    except Exception as e:
        print(f"Error reading shapefile: {e}")
        jprLat = np.array([20.5, 21.0, 21.5, 21.0, 20.5])
        jprLon = np.array([86.0, 86.0, 86.5, 86.5, 86.0])

    mod_Lat, mod_Lon = np.meshgrid(
        np.linspace(np.min(jprLat), np.max(jprLat), 500),
        np.linspace(np.min(jprLon), np.max(jprLon), 500)
    )

    mod_conc = gridpoint(jprLat, jprLon, mod_Lat, mod_Lon, wqdata, explat, explon)

    save_dir = os.path.join(base_dir, 'saved_models')
    os.makedirs(save_dir, exist_ok=True)
    np.savez(os.path.join(save_dir, 'geoplott_2.npz'),
             jprLat=jprLat, jprLon=jprLon, mod_Lat=mod_Lat, mod_Lon=mod_Lon,
             mod_conc=mod_conc, explat=explat, explon=explon)

    return mod_conc, jprLat, jprLon, mod_Lat, mod_Lon, explat, explon