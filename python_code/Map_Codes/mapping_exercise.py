import os
import sys


def mapping_exercise(data_dir=None):
    if data_dir is None:
        data_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))

    samplefilename = "ODISHA_SUBDISTRICT_BDY.dbf"

    try:
        import scipy.io as sio
        mat_path = os.path.join(os.path.dirname(data_dir), 'a0_Postmonsoon_JAJAPUR.mat')
        mat = sio.loadmat(mat_path, squeeze_me=True)
    except Exception as e:
        print(f"Could not load wqdata: {e}")
        return

    csv_path = os.path.join(data_dir, 'python_code', 'wqdata.csv')
    if os.path.exists(csv_path):
        import pandas as pd
        wqdata = pd.read_csv(csv_path)
    else:
        print("wqdata.csv not found. Run export_data_to_csv.m in MATLAB first.")
        return

    from Map_Codes.b1_sampling_plot import b1_sampling_plot
    from Map_Codes.b2_IDW_extrapolation import b2_IDW_extrapolation
    from Map_Codes.b3_IDW_plots import b3_IDW_plots

    b1_sampling_plot(samplefilename, wqdata)
    b2_IDW_extrapolation(wqdata)
    b3_IDW_plots()