# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from scipy import stats

# Post-hoc: import sicuri (opzionali)
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
def eta_squared_anova(F: float, df_between: int, df_within: int) -> float:
    """
    Eta squared (η²) per ANOVA: SS_between / SS_total.
    Con F-test: η² = (F * df_between) / (F * df_between + df_within)
    """
    num = F * df_between
    den = num + df_within
    return float(num / den) if den > 0 else np.nan

def epsilon_squared_kw(H: float, k: int, n: int) -> float:
    """
    Epsilon squared (ε²) per Kruskal–Wallis: (H - k + 1) / (n - k)
    """
    den = (n - k)
    return float((H - k + 1) / den) if den > 0 else np.nan

# -----------------------------
# Init & check dataset
# -----------------------------
init_state()
st.title("📂 Step 7 — Analisi di sottogruppo")

if "df" not in st.session_state or st.session_state.df is None:
    st.warning("Nessun dataset disponibile. Carichi i dati in Step 0 — Upload Dataset.")
    st.page_link("pages/0_📂_Upload_Dataset.py", label="➡️ Vai a Upload Dataset", icon="📂")
    st.stop()

# Dataset attivo: preferire il working se presente
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

    # Descrittive per sottogruppo
    desc = df.groupby(group_var, dropna=False)[target].agg(
        N="count", Mean="mean", Median="median", SD="std"
    ).round(3)
    st.dataframe(desc, use_container_width=True)

    # Boxplot comparativo
    fig = px.box(df, x=group_var, y=target, points="all", color=group_var,
                 title=f"Distribuzione di {target} per {group_var}")
    st.plotly_chart(fig, use_container_width=True)

    # Prepara gruppi per test
    group_levels = df[group_var].dropna().unique().tolist()
    groups_series = [df.loc[df[group_var] == g, target].dropna() for g in group_levels]
    has_enough = all(len(g) >= 2 for g in groups_series) and len(groups_series) >= 2

    # Contenitori per salvataggio nel Results Summary
    posthoc_results = {}
    global_tests = {}
    any_test_shown = False

    st.markdown("**Test statistici tra sottogruppi**")

    # ---- ANOVA (parametrico)
    if has_enough:
        try:
            F, p_anova = stats.f_oneway(*groups_series)
            any_test_shown = True
            df_between = len(groups_series) - 1
            df_within = sum(len(g) for g in groups_series) - len(groups_series)
            eta2 = eta_squared_anova(F, df_between, df_within)

            st.write(f"ANOVA: **F = {F:.3f}**, **p = {p_anova:.3g}**, **η² = {eta2:.3f}**")
            st.caption("Interpretazione η²: ~0.01 piccolo, ~0.06 medio, ~0.14 grande (Cohen).")

            global_tests["ANOVA"] = {
                "stat": float(F), "pvalue": float(p_anova),
                "df_between": int(df_between), "df_within": int(df_within),
                "effect_name": "eta_squared", "effect_value": float(eta2)
            }

            if p_anova < alpha:
                if _HAS_STATSMODELS:
                    st.success("ANOVA significativa → eseguo **Tukey HSD** (correzione multipla).")
                    valid = df[[group_var, target]].dropna()
                    tukey = pairwise_tukeyhsd(endog=valid[target], groups=valid[group_var], alpha=alpha)
                    tukey_df = pd.DataFrame(tukey.summary().data[1:], columns=tukey.summary().data[0])
                    # Colonne tipiche: group1, group2, meandiff, p-adj, lower, upper, reject
                    st.dataframe(tukey_df, use_container_width=True)
                    st.caption("Come leggere Tukey: differenza medie, p-adj già corretto, reject=True se significativo.")
                    posthoc_results["Tukey HSD"] = tukey_df.round(4)
                else:
                    st.warning("Post-hoc Tukey non disponibile: installare `statsmodels`.")
        except Exception as e:
            st.info(f"ANOVA non calcolabile: {e}")

    # ---- Kruskal–Wallis (non parametrico)
    if has_enough:
        try:
            H, p_kw = stats.kruskal(*groups_series)
            any_test_shown = True
            k = len(groups_series)
            n = sum(len(g) for g in groups_series)
            eps2 = epsilon_squared_kw(H, k, n)

            st.write(f"Kruskal–Wallis: **H = {H:.3f}**, **p = {p_kw:.3g}**, **ε² = {eps2:.3f}**")
            st.caption("Interpretazione ε²: ~0.01 piccolo, ~0.08 medio, ~0.26 grande (Tomczak & Tomczak, 2014).")

            global_tests["Kruskal–Wallis"] = {
                "stat": float(H), "pvalue": float(p_kw),
                "k_groups": int(k), "n_total": int(n),
                "effect_name": "epsilon_squared", "effect_value": float(eps2)
            }

            if p_kw < alpha:
                if _HAS_SCIKIT_POSTHOCS:
                    st.success("Kruskal–Wallis significativo → eseguo **Dunn test** (Bonferroni).")
                    valid = df[[group_var, target]].dropna()
                    dunn = sp.posthoc_dunn(valid, val_col=target, group_col=group_var, p_adjust='bonferroni')
                    st.dataframe(dunn.round(4), use_container_width=True)
                    st.caption("Come leggere Dunn: matrice di p-value corretti; celle < α indicano differenze significative.")
                    posthoc_results["Dunn (Bonferroni)"] = dunn.round(4)
                else:
                    st.warning("Post-hoc Dunn non disponibile: installare `scikit-posthocs`.")
        except Exception as e:
            st.info(f"Kruskal–Wallis non calcolabile: {e}")

    if not any_test_shown:
        st.info("Gruppi troppo piccoli o dati insufficienti per eseguire i test globali.")

    # ---- Salvataggio nel Results Summary (con effect size)
    if st.button(f"➕ Aggiungi {target} a Results Summary", key=f"add_{group_var}_{target}"):
        item = {
            "type": "subgroup",
            "title": f"Analisi sottogruppi — {target} per {group_var}",
            "content": desc.to_dict(),
            "tests": global_tests  # ← include anche effect size
        }
        if posthoc_results:
            # Converti DataFrame in dict serializzabile
            item["posthoc"] = {k: (v.to_dict() if isinstance(v, pd.DataFrame) else v)
                               for k, v in posthoc_results.items()}
        # Inizializza contenitore report se non presente
        if "report_items" not in st.session_state:
            st.session_state.report_items = []
        st.session_state.report_items.append(item)
        st.success(f"Analisi {target} (con effect size) aggiunta al Results Summary.")

# -----------------------------
# Guida — interpretazione risultati
# -----------------------------
with st.expander("ℹ️ Guida — come leggere i risultati"):
    st.markdown("""
**Test globali**  
- **ANOVA**: verifica differenze tra **medie**.  
  - **F** elevato e **p < α** → almeno una media differisce.  
  - **η²**: quota di varianza spiegata dal fattore (≈0.01 piccolo, ≈0.06 medio, ≈0.14 grande).  
- **Kruskal–Wallis**: confronto **non parametrico** dei ranghi.  
  - **H** elevato e **p < α** → almeno una distribuzione differisce.  
  - **ε²**: misura dell’effetto per Kruskal (≈0.01 piccolo, ≈0.08 medio, ≈0.26 grande).

**Post-hoc**  
- **Tukey HSD** (dopo ANOVA significativa): per ogni coppia, mostra differenza di medie, **p-adj** (già corretto) e **reject** (True se significativo).  
- **Dunn (Bonferroni)** (dopo Kruskal significativo): matrice di p-value **corretti**; celle < α indicano coppie significative.

**Buone pratiche**  
- Integrare sempre il **p-value** con una **dimensione dell’effetto** (η² / ε²).  
- Valutare grandezze campionarie, outlier e contesto clinico/decisionale.  
- Ricordare che la significatività statistica **non** implica importanza pratica.
""")
