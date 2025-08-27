# -*- coding: utf-8 -*-
import pandas as pd

try:
    import statsmodels.stats.multicomp as mc
except Exception:
    mc = None

try:
    import scikit_posthocs as sp
except Exception:
    sp = None


def tukey_hsd(data: pd.Series, group: pd.Series):
    """
    Tukey HSD post-hoc dopo ANOVA.
    Ritorna un DataFrame con confronti a coppie.
    """
    if mc is None:
        return None
    df = pd.DataFrame({"y": data, "g": group}).dropna()
    comp = mc.MultiComparison(df["y"], df["g"])
    res = comp.tukeyhsd()
    return pd.DataFrame(
        data=res._results_table.data[1:],
        columns=res._results_table.data[0]
    )


def dunn_test(data: pd.Series, group: pd.Series, p_adjust="bonferroni"):
    """
    Dunn post-hoc dopo Kruskal–Wallis.
    Ritorna una matrice di p-values corretti.
    """
    if sp is None:
        return None
    df = pd.DataFrame({"y": data, "g": group}).dropna()
    res = sp.posthoc_dunn(df, val_col="y", group_col="g", p_adjust=p_adjust)
    return res
