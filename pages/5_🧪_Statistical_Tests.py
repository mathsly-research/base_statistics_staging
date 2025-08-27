# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from core.state import init_state
from core.validate import dataset_diagnostics
from core.ui import quality_alert

# Assunzioni
from core.assumptions import shapiro_test, levene_test, chi2_expected_table

# Test
from core.tests import (
    ttest_welch, mannwhitney,
    anova_welch, kruskal,
    chi_square_of_independence, two_proportions,
    TestResult
)

# Post-hoc
from core.posthoc import tukey_hsd, dunn_test

try:
    from scipy import stats as spstats
except Exception:
    spstats = None

# -----------------------------
# Init & dataset check
# -----------------------------
init_state()
st.title("🧪 Step 6 — Test statistici (guidati dalle assunzioni)")

if st.session_state.df is None:
    st.warning("Nessun dataset in memoria. Carichi i dati in **Upload Dataset**.")
    st.page_link("pages/1_📂_Upload_Dataset.py", label="➡️ Vai a Upload Dataset", icon="📂")
    st.stop()

df: pd.DataFrame = st.session_state.df

with st.expander("🔎 Controllo rapido qualità", expanded=False):
    diag = st.session_state.diagnostics or dataset_diagnostics(df)
    st.session_state.diagnostics = diag
    quality_alert(diag)

# -----------------------------
# Selezione variabili
# -----------------------------
st.subheader("Selezione variabili")
all_cols = list(df.columns)

c1, c2 = st.columns(2)
with c1:
    y = st.selectbox("Variabile di interesse (outcome)", options=all_cols)
with c2:
    group = st.selectbox("Raggruppamento (gruppi)", options=[None] + all_cols,
                         format_func=lambda x: "— nessuno —" if x is None else x)

if group is None:
    st.info("Selezioni una variabile di **raggruppamento** per procedere.")
    st.stop()

# -----------------------------
# Assunzioni
# -----------------------------
def assess_assumptions(df: pd.DataFrame, y: str, group: str):
    out = {"design": None, "normality_ok": None, "levene_ok": None, "chi2_min_expected": None}
    s = df[y]; g = df[group]
    is_cont = pd.api.types.is_numeric_dtype(s)
    levels = pd.Series(g).dropna().unique().tolist()
    if is_cont:
        out["design"] = f"Continua vs Gruppi ({len(levels)} livelli)"
        sh_p = []
        for lv in levels:
            vals = df.loc[g == lv, y].dropna().to_numpy(dtype=float)
            sh = shapiro_test(vals)
            if sh: sh_p.append(sh["p"])
        out["normality_ok"] = all(p >= 0.05 for p in sh_p) if sh_p else None
        groups = [df.loc[g == lv, y].dropna().to_numpy(dtype=float) for lv in levels]
        groups = [arr for arr in groups if arr.size >= 2]
        if len(groups) >= 2:
            lev = levene_test(*groups)
            out["levene_ok"] = (lev and lev["p"] >= 0.05)
    else:
        out["design"] = "Categorica vs Gruppi"
        res = chi2_expected_table(df[y], g)
        if res: out["chi2_min_expected"] = res["min_expected"]
    return out

assess = assess_assumptions(df, y, group)
is_cont = pd.api.types.is_numeric_dtype(df[y])

# -----------------------------
# Suggerimento percorso
# -----------------------------
if is_cont:
    if assess["normality_ok"] and (assess["levene_ok"] in (True, None)):
        recommended = "Parametrici"
        st.success("Percorso consigliato: **Parametrici** (normalità e varianze ok).")
    else:
        recommended = "Non parametrici"
        st.error("Percorso consigliato: **Non parametrici** (assunzioni non rispettate).")
    path = st.radio("Selezionare percorso:", ["Parametrici", "Non parametrici"],
                    index=0 if recommended == "Parametrici" else 1, horizontal=True)
else:
    recommended = "Chi-quadrato"
    path = "Chi-quadrato"
    if assess["chi2_min_expected"] is not None:
        if assess["chi2_min_expected"] < 5:
            st.error("Attesi < 5 → Fisher consigliato se tabella 2×2.")
        else:
            st.success("Attesi ok → Chi-quadrato applicabile.")

# -----------------------------
# Esecuzione test
# -----------------------------
st.subheader("Risultati")

def show_result(tr: TestResult, idx: int):
    with st.container(border=True):
        st.markdown(f"**{idx}. {tr.test_name}**")
        if tr.stat is not None: st.write(f"Statistica = `{tr.stat:.4f}`")
        if tr.df is not None: st.write(f"Gradi di libertà = `{tr.df:.2f}`")
        if tr.pvalue is not None: st.write(f"p-value = `{tr.pvalue:.4g}`")
        if tr.estimate_name and tr.estimate_value is not None:
            if tr.estimate_ci:
                lo, hi = tr.estimate_ci
                st.write(f"{tr.estimate_name}: **{tr.estimate_value:.4g}** (CI95% [{lo:.4g}, {hi:.4g}])")
            else:
                st.write(f"{tr.estimate_name}: **{tr.estimate_value:.4g}**")
        if tr.effect_name and tr.effect_value is not None:
            st.write(f"{tr.effect_name}: **{tr.effect_value:.4g}**")
        if tr.details and "table" in tr.details:
            st.markdown("**Tabella di contingenza**")
            st.dataframe(tr.details["table"], use_container_width=True)
        if tr.note: st.info(tr.note)

results: list[TestResult] = []

if is_cont:
    levels = pd.Series(df[group]).dropna().astype("category").cat.categories.tolist()
    codes = pd.Series(df[group]).astype("category").cat.codes
    yvals = df[y].to_numpy(dtype=float)
    arrays = [yvals[codes == i] for i in range(len(levels))]

    if len(levels) == 2:
        a, b = arrays
        results.append(ttest_welch(a, b) if path == "Parametrici" else mannwhitney(a, b))
    elif len(levels) > 2:
        results.append(anova_welch(arrays) if path == "Parametrici" else kruskal(arrays))
    else:
        st.info("La variabile di raggruppamento deve avere almeno 2 livelli.")
else:
    res_chi = chi_square_of_independence(df[y], df[group])
    results.append(res_chi)
    if df[y].dropna().nunique() == 2 and pd.Series(df[group]).dropna().nunique() == 2:
        ct = pd.crosstab(df[group], df[y])
        if ct.shape == (2, 2) and spstats:
            oddsratio, p = spstats.fisher_exact(ct.values)
            tr = TestResult("Fisher exact test (2×2)", stat=float(oddsratio), pvalue=float(p), details={"table": ct})
            results.append(tr)
        results.append(two_proportions(df[y], df[group]))

for i, tr in enumerate(results, start=1):
    show_result(tr, i)

# -----------------------------
# Post-hoc se globale significativo
# -----------------------------
if is_cont and len(set(df[group].dropna())) > 2:
    global_test = results[0]
    if global_test.pvalue and global_test.pvalue < 0.05:
        st.subheader("🔎 Confronti post-hoc")
        if path == "Parametrici":
            tukey = tukey_hsd(df[y], df[group])
            if tukey is not None:
                st.write("**Tukey HSD**")
                st.dataframe(tukey, use_container_width=True)
                # Grafico lettere significative
                st.write("**Grafico post-hoc (Tukey HSD)**")
                plt.figure(figsize=(6, 4))
                sns.boxplot(x=group, y=y, data=df, showfliers=False)
                # lettere: usiamo reject da MultiComparison
                from statsmodels.stats.multicomp import MultiComparison
                comp = MultiComparison(df[y], df[group])
                res = comp.tukeyhsd()
                groups_letters = comp.groupsunique
                letters = res.groupsunique
                st.pyplot(plt.gcf())
            else:
                st.info("Tukey HSD non disponibile (statsmodels mancante).")
        else:
            dunn = dunn_test(df[y], df[group], p_adjust="bonferroni")
            if dunn is not None:
                st.write("**Dunn test (Bonferroni)**")
                st.dataframe(dunn, use_container_width=True)
                # Heatmap
                st.write("**Heatmap p-values (Dunn test)**")
                plt.figure(figsize=(6, 4))
                sns.heatmap(dunn, annot=True, fmt=".3f", cmap="coolwarm", cbar=False)
                st.pyplot(plt.gcf())
            else:
                st.info("Dunn test non disponibile (scikit-posthocs mancante).")

# -----------------------------
# Guida interpretativa
# -----------------------------
with st.expander("ℹ️ Quando usare questi test", expanded=True):
    st.markdown("""
**Parametrici (t-test Welch, ANOVA):**
- Outcome continuo, gruppi indipendenti.
- Assunzioni di normalità e omoscedasticità rispettate.
- Post-hoc: Tukey HSD.

**Non parametrici (Mann–Whitney, Kruskal–Wallis):**
- Quando le assunzioni non sono rispettate.
- Confrontano mediane/ranghi, robusti ad outlier.
- Post-hoc: Dunn test (con correzione multipla).

**Categoriche (Chi-quadrato, Fisher, due proporzioni):**
- Chi-quadrato valido se tutti gli attesi ≥5.
- In tabelle 2×2 con attesi piccoli → Fisher.
- Due proporzioni: utile per confrontare frequenze binarie con CI.
""")

# -----------------------------
# Add to Results Summary
# -----------------------------
st.divider()
if st.button("➕ Aggiungi risultati al Results Summary"):
    for tr in results:
        entry = {
            "type": "text",
            "title": tr.test_name,
            "content": {
                "stat": tr.stat,
                "pvalue": tr.pvalue,
                "df": tr.df,
                "estimate_name": tr.estimate_name,
                "estimate_value": tr.estimate_value,
                "estimate_ci": tr.estimate_ci,
                "effect_name": tr.effect_name,
                "effect_value": tr.effect_value,
                "effect_ci": tr.effect_ci,
            }
        }
        st.session_state.report_items.append(entry)
    st.success("Risultati aggiunti al Results Summary.")
