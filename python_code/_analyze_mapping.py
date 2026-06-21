"""
Detailed analysis of column mapping discrepancies between extracted CSV and paper.
"""
import pandas as pd
import numpy as np
import os

base = r"d:\study\project\water-quality\reference\origin - 副本\Machine Learning Modelling of Groundwater Water Qu\python_code"
wq = pd.read_csv(os.path.join(base, 'wqdata.csv'))

# Paper Table 3 stats [min, max, mean, std]
PAPER = {
    'pH':   [5.320, 7.820, 6.663, 0.487],
    'EC':   [53.400, 3000.000, 554.090, 483.560],
    'DO':   [1.070, 7.630, 3.839, 1.344],
    'F':    [0.040, 1.680, 0.523, 0.339],
    'Cl':   [0.000, 299.900, 57.759, 68.515],
    'NO3':  [4.800, 52.500, 27.640, 9.882],
    'SO4':  [2.140, 104.850, 18.793, 24.400],
    'PO4':  [0.000, 3.270, 0.651, 0.614],
    'U':    [0.000, 18.800, 0.974, 2.937],
    'CaH':  [0.000, 140.000, 50.811, 25.253],
    'MgH':  [0.000, 100.000, 33.243, 25.272],
    'HCO3': [20.000, 360.000, 136.760, 88.024]
}

# Our extracted columns (numeric only)
our_cols = [c for c in wq.columns if c not in ['Station', 'Latitude', 'Longitude', 'GWQI', 'LWQI', 'AWQI', 'RWQI']]

print(f"{'Our Column':>10s} | {'Our [min,max,mean,std]':>45s} | {'Best Paper Match':>18s} | {'Match?':>6s}")
print("="*85)

for col in our_cols:
    cmin, cmax, cmean, cstd = wq[col].min(), wq[col].max(), wq[col].mean(), wq[col].std()
    our_str = f"[{cmin:>8.2f},{cmax:>8.2f},{cmean:>8.3f},{cstd:>8.3f}]"
    
    # Find best paper match by comparing std (most robust)
    best_match = None
    best_std_diff = 1e9
    for pname, pstats in PAPER.items():
        c, s = [cmin, cmax, cmean, cstd], [pstats[0], pstats[1], pstats[2], pstats[3]]
        # Check min, max, mean all within 5% or absolute difference small
        diffs = sum(abs(a-b)/max(abs(b), 0.01) for a,b in zip(c, s))
        if diffs < best_std_diff:
            best_std_diff = diffs
            best_match = pname
    
    # Check if good match
    if best_match:
        pstats = PAPER[best_match]
        close = (abs(cmin-pstats[0])/max(abs(pstats[0]),0.01) < 0.05 and 
                 abs(cmax-pstats[1])/max(abs(pstats[1]),0.01) < 0.05 and
                 abs(cmean-pstats[2])/max(abs(pstats[2]),0.01) < 0.05)
        match_str = " ✓" if close else "  "
        print(f"{col:>10s} | {our_str} | {best_match:>18s} | {match_str:>6s}")

# Also show paper columns not matched
print(f"\n\nPaper columns and their mapping:")
print("="*60)
paper_order = ['pH', 'EC', 'DO', 'F', 'Cl', 'NO3', 'SO4', 'PO4', 'U', 'CaH', 'MgH', 'HCO3']
for pn in paper_order:
    pstats = PAPER[pn]
    pmin, pmax, pmean, pstd = pstats
    found_in = []
    for col in our_cols:
        cmin, cmax, cmean, cstd = wq[col].min(), wq[col].max(), wq[col].mean(), wq[col].std()
        if (abs(cmin-pmin)/max(abs(pmin),0.01) < 0.05 and 
            abs(cmax-pmax)/max(abs(pmax),0.01) < 0.05 and
            abs(cmean-pmean)/max(abs(pmean),0.01) < 0.05):
            found_in.append(col)
    print(f"Paper {pn:>5s} [{pmin:>8.2f},{pmax:>8.2f},{pmean:>8.3f}] => Our column(s): {found_in if found_in else '*** NOT FOUND ***'}")