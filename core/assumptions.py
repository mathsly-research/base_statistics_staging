# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

try:
    from scipy import stats as spstats
except Exception:
    spstats = None

# ----------------- util -----------------
def _drop_nan_pairs(a: np.ndarray, g: np.ndarray):
    m = ~np.isnan(a) & pd.notna(g)
    return a[m], g[m]

# ----------------- normalità -----------------
def shapiro_test(x: np.ndarray):
    if spstats is None or x.size < 3:
        return None
    W, p = spstats.shapiro(x if x.size <= 5000 else x[:5000])
    return {"W": float(W), "p": float(p)}

def anderson_test(x: np.ndarray):
    if spstats is None or x.size < 3:
        return None
    ad = spstats.anderson(x, dist="norm")
    return {
        "stat": float(ad.statistic),
        "crit_5": float(ad.critical_values[list(ad.significance_level).index(5.0)]) if 5.0 in ad.significance_level else None
    }

# ----------------- omoscedasticità -----------------
def levene_test(*groups):
    if spstats is None:
        return None
    stat, p = spstats.levene(*groups, center="median")
    return {"Levene W": float(stat), "p": float(p)}

def bartlett_test(*groups):
    if spstats is None:
        return None
    stat, p = spstats.bartlett(*groups)
    return {"Bartlett χ²": float(stat), "p": float(p)}

# ----------------- chi-quadrato expected counts -----------------
def chi2_expected_table(x: pd.Series, g: pd.Series):
    if spstats is None:
        return None
    tbl = pd.crosstab(g, x)
    chi2, p, dof, expected = spstats.chi2_contingency(tbl.values)
    expected_df = pd.DataFrame(expected, index=tbl.index, columns=tbl.columns)
    min_exp = expected.min()
    return {
        "expected": expected_df,
        "chi2": float(chi2),
        "p": float(p),
        "dof": dof,
        "min_expected": float(min_exp)
    }
