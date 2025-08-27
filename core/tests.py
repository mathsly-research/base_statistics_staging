# -*- coding: utf-8 -*-
from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple, List

try:
    from scipy import stats as spstats
except Exception:
    spstats = None

# ----------------------------- util -----------------------------

def _drop_nan_pairs(a: np.ndarray, g: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    m = ~np.isnan(a) & pd.notna(g)
    return a[m], g[m]

def _mean_ci_diff(a: np.ndarray, b: np.ndarray, alpha: float = 0.05) -> Tuple[float, float, float]:
    """
    CI per differenza di medie (Welch): media(b)-media(a) ± t * SE.
    Ritorna (diff, lo, hi).
    """
    ma, mb = np.mean(a), np.mean(b)
    va, vb = np.var(a, ddof=1), np.var(b, ddof=1)
    na, nb = a.size, b.size
    se = np.sqrt(va/na + vb/nb)
    diff = mb - ma
    if se == 0:
        return diff, diff, diff
    # df Welch–Satterthwaite
    df = (va/na + vb/nb)**2 / ((va/na)**2/(na-1) + (vb/nb)**2/(nb-1))
    tcrit = spstats.t.ppf(1 - alpha/2, df) if spstats is not None else 1.96
    lo, hi = diff - tcrit*se, diff + tcrit*se
    return float(diff), float(lo), float(hi)

def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = a.size, b.size
    sa2, sb2 = np.var(a, ddof=1), np.var(b, ddof=1)
    sp2 = ((na-1)*sa2 + (nb-1)*sb2) / (na + nb - 2) if (na + nb - 2) > 0 else 0.0
    if sp2 <= 0:
        return 0.0
    d = (np.mean(b) - np.mean(a)) / np.sqrt(sp2)
    return float(d)

def _hedges_g(d: float, na: int, nb: int) -> float:
    J = 1 - 3/(4*(na+nb) - 9) if (na+nb) > 2 else 1.0
    return float(J * d)

def _cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    # definizione O(n log n)
    a_sorted = np.sort(a)
    b_sorted = np.sort(b)
    i = j = more = less = 0
    na, nb = a_sorted.size, b_sorted.size
    while i < na and j < nb:
        if a_sorted[i] > b_sorted[j]:
            more += (na - i)
            j += 1
        elif a_sorted[i] < b_sorted[j]:
            less += (nb - j)
            i += 1
        else:
            # tie: avanza entrambi
            i += 1; j += 1
    delta = (more - less) / (na * nb) if na*nb > 0 else 0.0
    return float(delta)

def _cramers_v(chi2: float, n: int, r: int, c: int) -> float:
    denom = n * (min(r-1, c-1))
    if denom <= 0:
        return 0.0
    return float(np.sqrt(chi2 / denom))

def _two_proportions_wald_ci(x1: int, n1: int, x2: int, n2: int, alpha: float = 0.05) -> Tuple[float, float, float]:
    p1, p2 = x1/n1, x2/n2
    diff = p2 - p1
    z = 1.96 if spstats is None else spstats.norm.ppf(1 - alpha/2)
    se = np.sqrt(p1*(1-p1)/n1 + p2*(1-p2)/n2)
    lo, hi = diff - z*se, diff + z*se
    return float(diff), float(lo), float(hi)

# --------------------------- risultati --------------------------

@dataclass
class TestResult:
    test_name: str
    stat: Optional[float]
    pvalue: Optional[float]
    df: Optional[float] = None
    details: Dict[str, Any] = None
    effect_name: Optional[str] = None
    effect_value: Optional[float] = None
    effect_ci: Optional[Tuple[float, float]] = None
    estimate_name: Optional[str] = None
    estimate_value: Optional[float] = None
    estimate_ci: Optional[Tuple[float, float]] = None
    note: Optional[str] = None

# --------------------------- test continui -----------------------

def ttest_welch(a: np.ndarray, b: np.ndarray) -> TestResult:
    if spstats is None:
        return TestResult("Welch t-test", None, None, note="SciPy non disponibile")
    na, nb = a.size, b.size
    t, p = spstats.ttest_ind(b, a, equal_var=False, nan_policy="omit")
    d = _cohens_d(a, b)
    g = _hedges_g(d, na, nb)
    diff, lo, hi = _mean_ci_diff(a, b)
    return TestResult(
        test_name="Welch t-test (medie, varianze disuguali)",
        stat=float(t), pvalue=float(p),
        effect_name="Hedges g", effect_value=g,
        estimate_name="Δ media (gruppo2 - gruppo1)", estimate_value=diff,
        estimate_ci=(lo, hi)
    )

def mannwhitney(a: np.ndarray, b: np.ndarray) -> TestResult:
    if spstats is None:
        return TestResult("Mann–Whitney U", None, None, note="SciPy non disponibile")
    # usa 'two-sided' e continuity correction
    U, p = spstats.mannwhitneyu(b, a, alternative="two-sided", method="auto")
    delta = _cliffs_delta(a, b)
    return TestResult(
        test_name="Mann–Whitney U (non parametrico)",
        stat=float(U), pvalue=float(p),
        effect_name="Cliff’s delta", effect_value=delta
    )

def anova_welch(groups: List[np.ndarray]) -> TestResult:
    if spstats is None:
        return TestResult("Welch ANOVA", None, None, note="SciPy non disponibile")
    # Welch ANOVA approssimata (spstats.oneway è classica). Usiamo oneway per la statistica F classica, e segnaliamo var disuguali.
    F, p = spstats.f_oneway(*groups)
    # omega² (classico, approx)
    k = len(groups)
    n = sum(len(g) for g in groups)
    # SS_between approx da F
    # omega² ≈ (F*(k-1) - (k-1)) / (F*(k-1) + (n - k))
    omega2 = ((F*(k-1)) - (k-1)) / ((F*(k-1)) + (n - k)) if (n - k) > 0 else 0.0
    return TestResult(
        test_name="ANOVA (F-test) — considerare Welch se varianze disuguali",
        stat=float(F), pvalue=float(p),
        effect_name="ω² (omega squared)", effect_value=float(omega2)
    )

def kruskal(groups: List[np.ndarray]) -> TestResult:
    if spstats is None:
        return TestResult("Kruskal–Wallis", None, None, note="SciPy non disponibile")
    H, p = spstats.kruskal(*groups, nan_policy="omit")
    # epsilon² ≈ (H - k + 1) / (n - k)
    k = len(groups)
    n = sum(len(g) for g in groups)
    eps2 = ((H - k + 1) / (n - k)) if (n - k) > 0 else 0.0
    return TestResult(
        test_name="Kruskal–Wallis (non parametrico)",
        stat=float(H), pvalue=float(p),
        effect_name="ε² (epsilon squared)", effect_value=float(eps2)
    )

# ----------------------- test categorici -------------------------

def chi_square_of_independence(x: pd.Series, g: pd.Series) -> TestResult:
    if spstats is None:
        return TestResult("Chi-quadrato di indipendenza", None, None, note="SciPy non disponibile")
    tbl = pd.crosstab(g, x)  # righe=gruppo, colonne=categoria
    chi2, p, dof, expected = spstats.chi2_contingency(tbl.values)
    v = _cramers_v(chi2, n=int(tbl.values.sum()), r=tbl.shape[0], c=tbl.shape[1])
    return TestResult(
        test_name="Chi-quadrato di indipendenza",
        stat=float(chi2), pvalue=float(p), df=float(dof),
        effect_name="Cramer’s V", effect_value=float(v),
        details={"table": tbl}
    )

def two_proportions(x: pd.Series, g: pd.Series, success: Optional[Any] = None) -> TestResult:
    # atteso: x binaria, g binaria
    s = x
    if success is not None:
        s = (x == success).astype(int)
    if s.dropna().nunique() != 2 or pd.Series(g).dropna().nunique() != 2:
        return TestResult("Confronto due proporzioni", None, None, note="Variabili non binarie")
    df = pd.DataFrame({"x": s, "g": g}).dropna()
    gvals = sorted(df["g"].unique())
    x1 = int(df.loc[df["g"] == gvals[0], "x"].sum())
    n1 = int((df["g"] == gvals[0]).sum())
    x2 = int(df.loc[df["g"] == gvals[1], "x"].sum())
    n2 = int((df["g"] == gvals[1]).sum())
    if spstats is None:
        # z test manuale (Wald) con CI
        diff, lo, hi = _two_proportions_wald_ci(x1, n1, x2, n2)
        return TestResult(
            test_name="Confronto due proporzioni (Wald approx.)",
            stat=None, pvalue=None,
            estimate_name="Δ proporzioni (gruppo2 - gruppo1)",
            estimate_value=diff, estimate_ci=(lo, hi)
        )
    # test z pooled
    p_pool = (x1 + x2) / (n1 + n2)
    se = np.sqrt(p_pool*(1-p_pool)*(1/n1 + 1/n2))
    z = (x2/n2 - x1/n1) / se if se > 0 else 0.0
    p = 2*(1 - spstats.norm.cdf(abs(z)))
    diff, lo, hi = _two_proportions_wald_ci(x1, n1, x2, n2)
    return TestResult(
        test_name="Confronto due proporzioni (z test)",
        stat=float(z), pvalue=float(p),
        estimate_name="Δ proporzioni (gruppo2 - gruppo1)",
        estimate_value=diff, estimate_ci=(lo, hi)
    )

# --------------------- decisione automatica ----------------------

def decide_and_test(df: pd.DataFrame, y: str, group: Optional[str]) -> Dict[str, Any]:
    """
    Decide il test in base ai tipi:
    - y continua & gruppo binario → Welch t-test + Mann–Whitney
    - y continua & gruppo con >2 livelli → ANOVA + Kruskal
    - y categorica & gruppo categorico → Chi-quadrato (se 2x2: mostra anche test due proporzioni)
    - y binaria & gruppo binario → aggiunge 'two_proportions'
    """
    out: Dict[str, Any] = {"tests": [], "note": None}

    s = df[y]
    if group is None:
        out["note"] = "Selezionare una variabile di raggruppamento per il confronto."
        return out
    g = df[group]

    # continuous vs groups?
    is_cont = pd.api.types.is_numeric_dtype(s)
    is_cat_y = not is_cont

    if is_cont:
        a, gg = _drop_nan_pairs(s.to_numpy(dtype=float), g)
        # livelli gruppo
        levels = pd.Series(gg).astype("category").cat.categories.tolist()
        groups = [a[pd.Series(gg).astype("category").cat.codes == i] for i in range(len(levels))]

        if len(levels) == 2:
            res1 = ttest_welch(groups[0], groups[1])
            res2 = mannwhitney(groups[0], groups[1])
            out["tests"] = [res1, res2]
            out["design"] = {"y_type": "continuous", "groups": levels}
        elif len(levels) > 2:
            res1 = anova_welch(groups)
            res2 = kruskal(groups)
            out["tests"] = [res1, res2]
            out["design"] = {"y_type": "continuous", "groups": levels}
        else:
            out["note"] = "La variabile di raggruppamento deve avere almeno 2 livelli."
    else:
        # categorica
        if pd.api.types.is_bool_dtype(s) or s.dropna().nunique() == 2:
            # binaria
            res_chi = chi_square_of_independence(s, g)
            out["tests"] = [res_chi]
            out["design"] = {"y_type": "binary", "groups": None}
            # se anche g è binaria, aggiungi confronto due proporzioni
            if pd.Series(g).dropna().nunique() == 2:
                out["tests"].append(two_proportions(s, g))
        else:
            # multinomiale
            res_chi = chi_square_of_independence(s, g)
            out["tests"] = [res_chi]
            out["design"] = {"y_type": "categorical", "groups": None}

    return out

