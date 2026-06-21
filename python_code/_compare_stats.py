"""
Compare data statistics between extracted CSV and paper's Table 3.
"""
import pandas as pd
import numpy as np
import os

base = r"d:\study\project\water-quality\reference\origin - 副本\Machine Learning Modelling of Groundwater Water Qu\python_code"

# Load our data
wq = pd.read_csv(os.path.join(base, 'wqdata.csv'))
print(f"Columns: {list(wq.columns)}")
print(f"Shape: {wq.shape}")
print()

# Paper Table 3 data
paper_stats = {
    'pH':       [5.320, 7.820, 6.663, 0.487],
    'EC':       [53.400, 3000.000, 554.090, 483.560],
    'DO':       [1.070, 7.630, 3.839, 1.344],
    'F':        [0.040, 1.680, 0.523, 0.339],
    'Cl':       [0.000, 299.900, 57.759, 68.515],
    'NO3':      [4.800, 52.500, 27.640, 9.882],
    'SO4':      [2.140, 104.850, 18.793, 24.400],
    'PO4':      [0.000, 3.270, 0.651, 0.614],
    'U':        [0.000, 18.800, 0.974, 2.937],
    'CaH':      [0.000, 140.000, 50.811, 25.253],
    'MgH':      [0.000, 100.000, 33.243, 25.272],
    'HCO3':     [20.000, 360.000, 136.760, 88.024]
}

# Our data stats for each column [min, max, mean, std]
param_map = {
    'pH': 'pH', 'EC': 'EC', 'DO': 'DO', 'F': 'F', 'Cl': 'Cl',
    'NO3': 'NO3', 'SO4': 'SO4', 'PO4': 'PO4', 'U': 'U',
    'CaH': 'CaH', 'MgH': 'MgH', 'HCO3': 'HCO3', 'TDS': 'TDS',
    'Na': 'Na'
}

for pname, col in param_map.items():
    if col not in wq.columns:
        print(f"{pname:>5s}: COLUMN NOT FOUND")
        continue
    cmin, cmax, cmean, cstd = wq[col].min(), wq[col].max(), wq[col].mean(), wq[col].std()
    
    if pname in paper_stats:
        pmin, pmax, pmean, pstd = paper_stats[pname]
        match = (abs(cmin-pmin)/max(abs(pmin),0.01) < 0.01 and 
                 abs(cmax-pmax)/max(abs(pmax),0.01) < 0.01)
        flag = " ✓" if match else " ✗"
        print(f"{pname:>5s}: min={cmin:8.3f}(paper={pmin:8.3f})  max={cmax:8.3f}(paper={pmax:8.3f})  mean={cmean:8.3f}(paper={pmean:8.3f})  std={cstd:8.3f}(paper={pstd:8.3f}){flag}")
    else:
        print(f"{pname:>5s}: min={cmin:>8.2f}  max={cmax:>8.2f}  mean={cmean:>8.2f}  std={cstd:>8.2f}  (not in paper)")