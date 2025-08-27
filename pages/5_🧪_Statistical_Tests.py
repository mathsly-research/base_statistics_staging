# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np

from core.state import init_state
from core.validate import dataset_diagnostics
from core.ui import quality_alert

# Assunzioni
from core.assumptions import shapiro_test, levene_test, chi2_expected_table

# Test principali
from core.tests import (
    ttest_welch, mannwhitney,
    anova_welch, kruskal,
    chi_square_of_independence, two_proportions,
    TestResult
)

# Post-hoc
try:
    from core.posthoc import tukey_hsd, dunn_test
    POSTHOC_OK = True
except Exception as e:
    POSTHOC_OK = False
    POSTHOC_ERR = repr(e)

# SciPy opzionale per Fisher
try:
    from scipy import stats as spstats
except Exception:
    spstats = None

# Plotly
import plotly.express as px
import plotly.graph_objects as go

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
# Assunzioni → raccomandazione
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
        out["normality_ok"] = (all(p >= 0.05 for p in sh_p) if sh_p else None)
        # Levene
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
            st.error("Attesi < 5 → **Fisher** consigliato se tabella 2×2.")
        else:
            st.success("Attesi ok → **Chi-quadrato** applicabile.")

# -----------------------------
# Esecuzione test principali
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
            try:
                oddsratio, p = spstats.fisher_exact(ct.values)
                tr = TestResult("Fisher exact test (2×2)", stat=float(oddsratio), pvalue=float(p), details={"table": ct})
                results.append(tr)
            except Exception:
                pass
        results.append(two_proportions(df[y], df[group]))

for i, tr in enumerate(results, start=1):
    show_result(tr, i)

# -----------------------------
# Post-hoc (solo se >2 gruppi e p<0.05)
# -----------------------------
def _pairs_to_matrix_from_tukey(tukey_df: pd.DataFrame) -> pd.DataFrame:
    """Costruisce una matrice simmetrica di p-value dalla tabella Tukey HSD di statsmodels."""
    # statsmodels columns: group1, group2, meandiff, p-adj, lower, upper, reject
    groups = sorted(set(tukey_df['group1']).union(set(tukey_df['group2'])))
    mat = pd.DataFrame(np.nan, index=groups, columns=groups, dtype=float)
    for _, r in tukey_df.iterrows():
        g1, g2, p = r['group1'], r['group2'], float(r['p-adj'])
        mat.loc[g1, g2] = p
        mat.loc[g2, g1] = p
        mat.loc[g1, g1] = 0.0
        mat.loc[g2, g2] = 0.0
    # riempi diagonale
    for g in groups:
        mat.loc[g, g] = 0.0
    return mat

if is_cont and len(set(df[group].dropna())) > 2:
    global_test = results[0]
    if global_test.pvalue is not None and global_test.pvalue < 0.05:
        st.subheader("🔎 Confronti post-hoc")
        if not POSTHOC_OK:
            st.info("Modulo post-hoc non disponibile. Installare `statsmodels` (per Tukey) e `scikit-posthocs` (per Dunn).")
        else:
            if path == "Parametrici":
                tukey = tukey_hsd(df[y], df[group])
                if tukey is not None and not tukey.empty:
                    st.write("**Tukey HSD (tabella)**")
                    st.dataframe(tukey, use_container_width=True)
                    # Heatmap delle p-values
                    try:
                        pmat = _pairs_to_matrix_from_tukey(tukey)
                        fig = go.Figure(data=go.Heatmap(
                            z=pmat.values, x=pmat.columns, y=pmat.index,
                            text=np.round(pmat.values, 3), texttemplate="%{text}", colorscale="RdBu_r",
                            reversescale=True, zmin=0, zmax=0.1, colorbar_title="p-value"
                        ))
                        fig.update_layout(title="Tukey HSD — p-values (più scuro = più significativo)",
                                          xaxis_title="Gruppo", yaxis_title="Gruppo", template="plotly_white")
                        st.plotly_chart(fig, use_container_width=True, key="tukey_heatmap")
                    except Exception:
                        pass
                    # Boxplot per gruppi
                    st.write("**Boxplot per gruppi**")
                    figb = px.box(df, x=group, y=y, points="outliers", title="Distribuzioni per gruppo")
                    st.plotly_chart(figb, use_container_width=True, key="tukey_box")
                else:
                    st.info("Tukey HSD non disponibile (verificare `statsmodels`).")
            else:
                dunn = dunn_test(df[y], df[group], p_adjust="bonferroni")
                if dunn is not None and not dunn.empty:
                    st.write("**Dunn test (Bonferroni) — matrice p-values**")
                    st.dataframe(dunn, use_container_width=True)
                    fig = go.Figure(data=go.Heatmap(
                        z=dunn.values, x=dunn.columns, y=dunn.index,
                        text=np.round(dunn.values, 3), texttemplate="%{text}", colorscale="RdBu_r",
                        reversescale=True, zmin=0, zmax=0.1, colorbar_title="p-value"
                    ))
                    fig.update_layout(title="Dunn post-hoc — p-values (più scuro = più significativo)",
                                      xaxis_title="Gruppo", yaxis_title="Gruppo", template="plotly_white")
                    st.plotly_chart(fig, use_container_width=True, key="dunn_heatmap")
                    # Boxplot per gruppi (utile per interpretazione)
                    st.write("**Boxplot per gruppi**")
                    figb = px.box(df, x=group, y=y, points="outliers", title="Distribuzioni per gruppo")
                    st.plotly_chart(figb, use_container_width=True, key="dunn_box")
                else:
                    st.info("Dunn test non disponibile (verificare `scikit-posthocs`).")

# -----------------------------
# Guida interpretativa
# -----------------------------
with st.expander("ℹ️ Quando usare questi test", expanded=True):
    st.markdown("""
**Parametrici (t-test Welch, ANOVA)**  
Outcome continuo, gruppi indipendenti, normalità e varianze simili (o campioni grandi). Post-hoc: **Tukey HSD**.

**Non parametrici (Mann–Whitney, Kruskal–Wallis)**  
Assunzioni violate o outlier importanti. Confrontano ranghi/mediane. Post-hoc: **Dunn con correzione**.

**Categoriche (Chi-quadrato, Fisher, due proporzioni)**  
Tabelle di contingenza; Chi-quadrato valido con attesi ≥ 5. Se 2×2 con attesi piccoli → **Fisher**.  
Per due proporzioni, riportare sempre **Δ p** con **CI 95%**.
""")

# -----------------------------
# Add to Results Summary
# -----------------------------
st.divider()
if st.button("➕ Aggiungi risultati al Results Summary"):
    for tr in results:
        st.session_state.report_items.append({
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
        })
    st.success("Risultati aggiunti al Results Summary.")
