# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import plotly.express as px
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import scikit_posthocs as sp

from core.state import init_state

# -----------------------------
# Init & check dataset
# -----------------------------
init_state()
st.title("📂 Step 7 — Analisi di sottogruppo")

if "df" not in st.session_state or st.session_state.df is None:
    st.warning("Nessun dataset disponibile. Carichi i dati in Step 0 — Upload Dataset.")
    st.page_link("pages/0_📂_Upload_Dataset.py", label="➡️ Vai a Upload Dataset", icon="📂")
    st.stop()

# Dataset attivo
df = st.session_state.df_working if st.session_state.df_working is not None else st.session_state.df

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

group_var = st.selectbox("Seleziona variabile di raggruppamento (sottogruppi):", cat_vars)
target_vars = st.multiselect("Seleziona variabili numeriche da analizzare:", num_vars, default=num_vars[:1])

# -----------------------------
# Analisi per ogni variabile numerica
# -----------------------------
for target in target_vars:
    st.subheader(f"🔎 {target} per sottogruppi di {group_var}")

    # Tabella descrittive
    desc = df.groupby(group_var)[target].agg(
        N="count", Mean="mean", Median="median", SD="std"
    ).round(2)
    st.dataframe(desc, use_container_width=True)

    # Boxplot comparativo
    fig = px.box(df, x=group_var, y=target, points="all", color=group_var,
                 title=f"Distribuzione di {target} per {group_var}")
    st.plotly_chart(fig, use_container_width=True)

    # Test statistici globali
    groups = [df[df[group_var] == g][target].dropna() for g in df[group_var].unique()]
    if len(groups) > 1:
        st.write("**Test statistici tra sottogruppi**")
        try:
            fval, p_anova = stats.f_oneway(*groups)
            st.write(f"ANOVA: F = {fval:.3f}, p = {p_anova:.3g}")
            if p_anova < 0.05:
                st.success("ANOVA significativa → eseguo Tukey HSD")
                tukey = pairwise_tukeyhsd(endog=df[target], groups=df[group_var], alpha=0.05)
                tukey_df = pd.DataFrame(data=tukey.summary().data[1:], columns=tukey.summary().data[0])
                st.dataframe(tukey_df, use_container_width=True)
        except Exception as e:
            st.info(f"ANOVA non calcolabile: {e}")

        try:
            hval, p_kw = stats.kruskal(*groups)
            st.write(f"Kruskal–Wallis: H = {hval:.3f}, p = {p_kw:.3g}")
            if p_kw < 0.05:
                st.success("Kruskal–Wallis significativo → eseguo Dunn test")
                dunn = sp.posthoc_dunn(df, val_col=target, group_col=group_var, p_adjust='bonferroni')
                st.dataframe(dunn.round(4), use_container_width=True)
        except Exception as e:
            st.info(f"Kruskal–Wallis non calcolabile: {e}")

    # Aggiunta opzionale al Results Summary
    if st.button(f"➕ Aggiungi {target} a Results Summary", key=f"add_{target}"):
        st.session_state.report_items.append({
            "type": "subgroup",
            "title": f"Analisi sottogruppi — {target} per {group_var}",
            "content": desc.to_dict()
        })
        st.success(f"Analisi {target} aggiunta al Results Summary.")

# -----------------------------
# Guida interpretativa
# -----------------------------
with st.expander("ℹ️ Guida"):
    st.markdown("""
**Analisi per sottogruppi**  
- Permette di confrontare una variabile numerica tra i livelli di una variabile categorica.  
- Fornisce descrittive, grafici e test statistici globali.  
- Se i test globali sono significativi, vengono eseguiti **post-hoc**:  
  - **Tukey HSD** (dopo ANOVA) → confronti a coppie tra tutti i gruppi.  
  - **Dunn test** (dopo Kruskal–Wallis) → confronti multipli con correzione di Bonferroni.  

⚠️ Importante: i post-hoc vanno interpretati nel contesto (numerosità dei gruppi, varianze, distribuzioni).
""")

