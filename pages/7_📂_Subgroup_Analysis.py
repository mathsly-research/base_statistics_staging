# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from scipy import stats

# Post-hoc: import sicuri
try:
    from statsmodels.stats.multicomp import pairwise_tukeyhsd
    _HAS_STATSMODELS = True
except Exception:
    _HAS_STATSMODELS = False

try:
    import scikit_posthocs as sp
    _HAS_SCIKIT_POSTHOCS = True
except Exception:
    _HAS_SCIKIT_POSTHOCS = False

from core.state import init_state

# -----------------------------
# Helper: effect size
# -----------------------------
def eta_squared_anova(groups, F, df_between, df_within):
    """
    Eta squared per ANOVA = SS_between / SS_total
    """
    # Calcolo SS totali da F e varianze
    # formula: η² = (F * df_between) / (F * df_between + df_within)
    return (F * df_between) / (F * df_between + df_within)

def epsilon_squared_kw(H, k, n):
    """
    Epsilon squared per Kruskal–Wallis
    """
    return (H - k + 1) / (n - k)

# -----------------------------
# Init & check dataset
# -----------------------------
init_state()
st.title("📂 Step 7 — Analisi di sottogruppo")

if "df" not in st.session_state or st.session_state.df is None:
    st.warning("Nessun dataset disponibile. Carichi i dati in Step 0 — Upload Dataset.")
    st.page_link("pages/0_📂_Upload_Dataset.py", label="➡️ Vai a Upload Dataset", icon="📂")
    st.stop()

df = st.session_state.df_working if "df_working" in st.session_state and st.session_state.df_working is not None else st.session_state.df
if df is None or df.empty:
    st.error("Il dataset attivo è vuoto.")
    st.stop()

# -----------------------------
# Selezione variabili
# -----------------------------
cat_vars = df.select_dtypes(exclude="number").columns.tolist()
num_vars = df.select_dtypes(include="number").columns.tolist()

if not cat_vars:
    st.error("Non ci sono variabili categoriche per definire sottogruppi.")
    st.stop()
if not num_vars:
    st.error("Non ci sono variabili numeriche da analizzare.")
    st.stop()

group_var = st.selectbox("Variabile di raggruppamento (sottogruppi):", cat_vars)
target_vars = st.multiselect("Variabili numeriche da analizzare:", num_vars, default=num_vars[:1])

alpha = st.slider("Soglia di significatività (α)", 0.001, 0.10, 0.05, step=0.001)

# -----------------------------
# Analisi per ogni variabile numerica
# -----------------------------
for target in target_vars:
    st.subheader(f"🔎 {target} per sottogruppi di {group_var}")

    desc = df.groupby(group_var, dropna=False)[target].agg(
        N="count", Mean="mean", Median="median", SD="std"
    ).round(3)
    st.dataframe(desc, use_container_width=True)

    fig = px.box(df, x=group_var, y=target, points="all", color=group_var,
                 title=f"Distribuzione di {target} per {group_var}")
    st.plotly_chart(fig, use_container_width=True)

    groups_series = [df.loc[df[group_var] == g, target].dropna() for g in df[group_var].dropna().unique()]
    has_enough = all(len(g) >= 2 for g in groups_series) and len(groups_series) >= 2

    posthoc_results = {}
    any_test_shown = False

    st.markdown("**Test statistici tra sottogruppi**")

    # ANOVA
    if has_enough:
        try:
            F, p_anova = stats.f_oneway(*groups_series)
            any_test_shown = True
            df_between = len(groups_series) - 1
            df_within = sum(len(g) for g in groups_series) - len(groups_series)
            eta2 = eta_squared_anova(groups_series, F, df_between, df_within)

            st.write(f"ANOVA: **F = {F:.3f}**, **p = {p_anova:.3g}**, **η² = {eta2:.3f}**")
            st.caption("**Interpretazione η²:** 0.01 ≈ piccolo, 0.06 ≈ medio, 0.14 ≈ grande (Cohen).")

            if p_anova < alpha and _HAS_STATSMODELS:
                st.success("ANOVA significativa → eseguo **Tukey HSD**")
                tukey = pairwise_tukeyhsd(endog=df[target].dropna(), groups=df.loc[df[target].notna(), group_var], alpha=alpha)
                tukey_df = pd.DataFrame(tukey.summary().data[1:], columns=tukey.summary().data[0])
                st.dataframe(tukey_df, use_container_width=True)
                st.caption("**Come leggere Tukey HSD:** group1 vs group2, differenza medie, p-adj corretto, reject=True se significativo.")
                posthoc_results["Tukey HSD"] = tukey_df
        except Exception as e:
            st.info(f"ANOVA non calcolabile: {e}")

    # Kruskal–Wallis
    if has_enough:
        try:
            H, p_kw = stats.kruskal(*groups_series)
            any_test_shown = True
            k = len(groups_series)
            n = sum(len(g) for g in groups_series)
            eps2 = epsilon_squared_kw(H, k, n)

            st.write(f"Kruskal–Wallis: **H = {H:.3f}**, **p = {p_kw:.3g}**, **ε² = {eps2:.3f}**")
            st.caption("**Interpretazione ε²:** 0.01 ≈ piccolo, 0.08 ≈ medio, 0.26 ≈ grande (Tomczak & Tomczak, 2014).")

            if p_kw < alpha and _HAS_SCIKIT_POSTHOCS:
                st.success("Kruskal–Wallis significativo → eseguo **Dunn test** (Bonferroni)")
                dunn = sp.posthoc_dunn(df[[group_var, target]].dropna(), val_col=target, group_col=group_var, p_adjust='bonferroni')
                st.dataframe(dunn.round(4), use_container_width=True)
                st.caption("**Come leggere Dunn:** matrice di p-value corretti; celle < α indicano differenze significative.")
                posthoc_results["Dunn (Bonferroni)"] = dunn.round(4)
        except Exception as e:
            st.info(f"Kruskal–Wallis non calcolabile: {e}")

    if not any_test_shown:
        st.info("Gruppi troppo piccoli o dati insufficienti per i test.")

    # Salvataggio nel Results Summary
    if st.button(f"➕ Aggiungi {target} a Results Summary", key=f"add_{group_var}_{target}"):
        item = {
            "type": "subgroup",
            "title": f"Analisi sottogruppi — {target} per {group_var}",
            "content": desc.to_dict()
        }
        if posthoc_results:
            item["posthoc"] = {k: (v.to_dict() if isinstance(v, pd.DataFrame) else v) for k, v in posthoc_results.items()}
        st.session_state.report_items.append(item)
        st.success(f"Analisi {target} aggiunta al Results Summary.")

# ---------------------------------
# Guida generale
# ---------------------------------
with st.expander("ℹ️ Guida — come leggere i risultati"):
    st.markdown("""
**Test globali**
- **ANOVA**: F misura il rapporto tra varianza tra gruppi e dentro gruppi.  
  - **η²** = proporzione di varianza spiegata dal fattore (0.01 piccolo, 0.06 medio, 0.14 grande).  
- **Kruskal–Wallis**: confronta i ranghi tra gruppi.  
  - **ε²** = proporzione di varianza spiegata dai gruppi (0.01 piccolo, 0.08 medio, 0.26 grande).  

**Post-hoc**
- **Tukey HSD**: mostra differenza di medie tra gruppi, p-value corretto, reject=True se significativo.  
- **Dunn (Bonferroni)**: matrice di p-value corretti; p < α indica differenza significativa.  

⚠️ Ricordare che significatività statistica ≠ importanza clinica. Interpretare sempre nel contesto.
""")
