# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np

from core.state import init_state
from core.validate import dataset_diagnostics
from core.ui import quality_alert

# Assunzioni (riuso funzioni Step 5)
from core.assumptions import shapiro_test, levene_test, chi2_expected_table

# Test e misure d’effetto
from core.tests import (
    decide_and_test,               # utile come fallback/riassunto
    ttest_welch, mannwhitney,      # 2 gruppi continui
    anova_welch, kruskal,          # >2 gruppi continui
    chi_square_of_independence,    # categoriche
    two_proportions,               # 2x2 proporzioni
    TestResult
)

# opzionale: Fisher exact 2x2
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
num_cols = list(df.select_dtypes(include="number").columns)

c1, c2 = st.columns(2)
with c1:
    y = st.selectbox("Variabile di interesse (outcome)", options=all_cols,
                     help="Continua o categorica.")
with c2:
    group = st.selectbox("Raggruppamento (gruppi)", options=[None] + all_cols,
                         format_func=lambda x: "— nessuno —" if x is None else x,
                         help="Obbligatorio per i confronti.")

if group is None:
    st.info("Selezioni una variabile di **raggruppamento** per procedere.")
    st.stop()

# -----------------------------
# Valutazione assunzioni (lettura da sessione o ricalcolo)
# -----------------------------
def assess_assumptions(df: pd.DataFrame, y: str, group: str):
    out = {
        "design": None,
        "normality_ok": None,
        "levene_ok": None,
        "chi2_min_expected": None,
        "notes": []
    }
    s = df[y]
    g = df[group]
    is_cont = pd.api.types.is_numeric_dtype(s)

    # livelli
    levels = pd.Series(g).dropna().unique().tolist()
    levels = [lv for lv in levels if pd.notna(lv)]

    if is_cont:
        out["design"] = f"Continua vs Gruppi ({len(levels)} livelli)"
        # Normalità per gruppo con Shapiro
        sh_pvals = []
        for lv in levels:
            vals = df.loc[g == lv, y].dropna().to_numpy(dtype=float)
            sh = shapiro_test(vals)
            if sh is not None:
                sh_pvals.append(sh["p"])
        normality_ok = (len(sh_pvals) >= 1) and all(p >= 0.05 for p in sh_pvals)
        out["normality_ok"] = normality_ok

        # Levene per varianze uguali
        groups = [df.loc[g == lv, y].dropna().to_numpy(dtype=float) for lv in levels]
        groups = [arr for arr in groups if arr.size >= 2]
        if len(groups) >= 2:
            lev = levene_test(*groups)
            out["levene_ok"] = (lev is not None and lev["p"] >= 0.05)
        else:
            out["levene_ok"] = None
            out["notes"].append("Gruppi troppo piccoli per verificare l’omoscedasticità con Levene.")

    else:
        out["design"] = "Categorica vs Gruppi"
        res = chi2_expected_table(df[y], g)
        if res is not None:
            out["chi2_min_expected"] = res["min_expected"]

    return out

# prova a leggere da sessione (se Step 5 l’ha salvato), altrimenti calcola
assess_key = f"assess::{y}::{group}"
if "assess_cache" not in st.session_state:
    st.session_state.assess_cache = {}
if assess_key in st.session_state.assess_cache:
    assess = st.session_state.assess_cache[assess_key]
else:
    assess = assess_assumptions(df, y, group)
    st.session_state.assess_cache[assess_key] = assess

# -----------------------------
# Suggerimento percorso
# -----------------------------
is_cont = pd.api.types.is_numeric_dtype(df[y])

if is_cont:
    msg = f"**Disegno:** {assess['design']}."
    st.write(msg)

    # raccomandazione
    if assess["normality_ok"] is True and (assess["levene_ok"] in (True, None)):
        recommended = "Parametrici"
        reason = []
        if assess["normality_ok"]: reason.append("normalità per gruppo OK")
        if assess["levene_ok"] is True: reason.append("varianze simili (Levene OK)")
        elif assess["levene_ok"] is None: reason.append("varianze non verificabili (campioni piccoli)")
        reason_text = "; ".join(reason)
        st.success(f"Percorso consigliato: **Parametrici** ({reason_text}).")
    else:
        recommended = "Non parametrici"
        reason = []
        if assess["normality_ok"] is False: reason.append("deviazioni dalla normalità")
        if assess["levene_ok"] is False: reason.append("varianze diverse (Levene non OK)")
        reason_text = "; ".join(reason) if reason else "assunzioni non soddisfatte"
        st.error(f"Percorso consigliato: **Non parametrici** ({reason_text}).")

    path = st.radio("Selezionare il percorso da eseguire:",
                    options=["Parametrici", "Non parametrici"],
                    index=0 if recommended == "Parametrici" else 1,
                    horizontal=True)

else:
    st.write(f"**Disegno:** {assess['design']}.")
    recommended = "Chi-quadrato"
    path = "Chi-quadrato"

    if assess["chi2_min_expected"] is not None:
        if assess["chi2_min_expected"] < 5:
            st.error(f"Attesi minimi = {assess['chi2_min_expected']:.2f} → **Fisher** consigliato (se 2×2).")
        else:
            st.success(f"Attesi minimi = {assess['chi2_min_expected']:.2f} → **Chi-quadrato** applicabile.")

# -----------------------------
# Esecuzione test (in base al percorso)
# -----------------------------
st.subheader("Risultati")

def show_result(tr: TestResult, idx: int):
    with st.container(border=True):
        st.markdown(f"**{idx}. {tr.test_name}**")
        if tr.stat is not None:
            st.write(f"Statistica = `{tr.stat:.4f}`")
        if tr.df is not None:
            st.write(f"Gradi di libertà = `{tr.df:.2f}`")
        if tr.pvalue is not None:
            st.write(f"p-value = `{tr.pvalue:.4g}`")
        if tr.estimate_name and tr.estimate_value is not None:
            if tr.estimate_ci:
                lo, hi = tr.estimate_ci
                st.write(f"{tr.estimate_name}: **{tr.estimate_value:.4g}**  (CI 95% [{lo:.4g}, {hi:.4g}])")
            else:
                st.write(f"{tr.estimate_name}: **{tr.estimate_value:.4g}**")
        if tr.effect_name and tr.effect_value is not None:
            if tr.effect_ci:
                elo, ehi = tr.effect_ci
                st.write(f"{tr.effect_name}: **{tr.effect_value:.4g}**  (CI 95% [{elo:.4g}, {ehi:.4g}])")
            else:
                st.write(f"{tr.effect_name}: **{tr.effect_value:.4g}**")
        if tr.details and "table" in tr.details:
            st.markdown("**Tabella di contingenza**")
            st.dataframe(tr.details["table"], use_container_width=True)
        if tr.note:
            st.info(tr.note)

results: list[TestResult] = []

if is_cont:
    # prepara gruppi
    levels = pd.Series(df[group]).dropna().astype("category").cat.categories.tolist()
    codes = pd.Series(df[group]).astype("category").cat.codes
    yvals = df[y].to_numpy(dtype=float)

    arrays = [yvals[codes == i] for i in range(len(levels))]

    if len(levels) == 2:
        a, b = arrays[0], arrays[1]
        if path == "Parametrici":
            results.append(ttest_welch(a, b))
        else:
            results.append(mannwhitney(a, b))
    elif len(levels) > 2:
        if path == "Parametrici":
            results.append(anova_welch(arrays))
        else:
            results.append(kruskal(arrays))
    else:
        st.info("La variabile di raggruppamento deve avere almeno 2 livelli.")
else:
    # categoriche
    res_chi = chi_square_of_independence(df[y], df[group])
    results.append(res_chi)

    # Se 2×2 e attesi piccoli → prova Fisher (se SciPy disponibile)
    if df[y].dropna().nunique() == 2 and pd.Series(df[group]).dropna().nunique() == 2:
        ct = pd.crosstab(df[group], df[y])
        if ct.shape == (2, 2) and spstats is not None:
            try:
                oddsratio, p = spstats.fisher_exact(ct.values)
                tr = TestResult(
                    test_name="Fisher exact test (2×2)",
                    stat=float(oddsratio), pvalue=float(p),
                    details={"table": ct}
                )
                results.append(tr)
            except Exception:
                pass
        # confronta anche due proporzioni (con CI)
        results.append(two_proportions(df[y], df[group]))

# Mostra i risultati
if results:
    for i, tr in enumerate(results, start=1):
        show_result(tr, i)

# -----------------------------
# Spiegazione del contesto d’uso
# -----------------------------
with st.expander("ℹ️ Quando usare questi test", expanded=True):
    st.markdown("""
**Parametrici (t-test Welch, ANOVA):**  
- Outcome **continuo**, gruppi **indipendenti**.  
- Normalità **approssimativamente** valida (per gruppo) e varianze **simili** (Levene OK).  
- Vantaggi: stime efficienti, IC e misure d’effetto noti (Hedges g, ω²).

**Non parametrici (Mann–Whitney, Kruskal–Wallis):**  
- Quando la normalità **non tiene** o sono presenti **outlier/code**.  
- Confrontano posizioni/mediane o ranghi: **robusti** alle deviazioni.  
- Effetti: **Cliff’s δ**, **ε²**.

**Categoriche (Chi-quadrato, due proporzioni, Fisher):**  
- Tabelle di contingenza; Chi-quadrato richiede attesi **≥ 5**.  
- Se tabella **2×2** con attesi bassi → **Fisher exact**.  
- Per due proporzioni: differenza con **CI 95%** (z-test o Wilson/Wald).
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
