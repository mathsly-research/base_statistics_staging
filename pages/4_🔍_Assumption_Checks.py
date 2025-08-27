# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd

from core.state import init_state
from core.validate import dataset_diagnostics
from core.ui import quality_alert
from core.assumptions import shapiro_test, anderson_test, levene_test, bartlett_test, chi2_expected_table

# -----------------------------
# Init & check dataset
# -----------------------------
init_state()
st.title("🔍 Step 5 — Assumption Checks")

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
cols_num = list(df.select_dtypes(include="number").columns)
cols_all = list(df.columns)

c1, c2 = st.columns(2)
with c1:
    y = st.selectbox("Variabile di interesse (outcome)", options=cols_all)
with c2:
    group = st.selectbox("Raggruppamento (per confronto)", options=[None] + cols_all, format_func=lambda x: "— nessuno —" if x is None else x)

# -----------------------------
# Normalità nei gruppi
# -----------------------------
if group is not None:
    st.subheader("1️⃣ Normalità nei gruppi")
    levels = df[group].dropna().unique()
    for lev in levels:
        vals = df.loc[df[group] == lev, y].dropna()
        st.markdown(f"**Gruppo {lev}** — n={len(vals)}")
        sh = shapiro_test(vals.to_numpy(dtype=float))
        if sh:
            if sh["p"] >= 0.05:
                st.success(f"Shapiro–Wilk: W={sh['W']:.3f}, p={sh['p']:.3f} → compatibile con normalità ✅")
            else:
                st.error(f"Shapiro–Wilk: W={sh['W']:.3f}, p={sh['p']:.3f} → devia da normalità ❌")

# -----------------------------
# Omoscedasticità
# -----------------------------
if group is not None:
    st.subheader("2️⃣ Omoscedasticità (varianze uguali)")
    groups = [df.loc[df[group] == lev, y].dropna().to_numpy(dtype=float) for lev in levels if len(df.loc[df[group]==lev,y].dropna())>1]
    if len(groups) >= 2:
        lev = levene_test(*groups)
        bart = bartlett_test(*groups)
        if lev and bart:
            if lev["p"] >= 0.05:
                st.success(f"Levene: W={lev['Levene W']:.3f}, p={lev['p']:.3f} → varianze simili ✅")
            else:
                st.error(f"Levene: W={lev['Levene W']:.3f}, p={lev['p']:.3f} → varianze diverse ❌")
            if bart["p"] >= 0.05:
                st.success(f"Bartlett: χ²={bart['Bartlett χ²']:.3f}, p={bart['p']:.3f} → varianze simili ✅")
            else:
                st.warning(f"Bartlett: χ²={bart['Bartlett χ²']:.3f}, p={bart['p']:.3f} → varianze diverse ⚠️ (sensibile a non normalità)")
    else:
        st.info("Gruppi troppo piccoli per test di omoscedasticità.")

# -----------------------------
# Tabella attesi per chi-quadrato
# -----------------------------
if not pd.api.types.is_numeric_dtype(df[y]) and group is not None:
    st.subheader("3️⃣ Tabella attesi per Chi-quadrato")
    res = chi2_expected_table(df[y], df[group])
    if res:
        st.write("**Tabella dei valori attesi**")
        st.dataframe(res["expected"])
        if res["min_expected"] < 5:
            st.error(f"Attesi minimi = {res['min_expected']:.2f} → non adatto al χ², usare Test esatto di Fisher ❌")
        else:
            st.success(f"Attesi minimi = {res['min_expected']:.2f} → ok per χ² ✅")

# -----------------------------
# Guida interpretativa
# -----------------------------
with st.expander("ℹ️ Come interpretare gli assumption checks", expanded=True):
    st.markdown("""
- **Shapiro–Wilk**: se p ≥ 0.05 → dati compatibili con normalità.  
- **Levene / Bartlett**: se p ≥ 0.05 → varianze simili nei gruppi.  
- **Chi-quadrato attesi**: se tutti ≥ 5 → test valido; se no → usare Fisher.  

👉 **Quando usare i test**:
- Se **normalità + omoscedasticità rispettate** → test parametrici (t-test, ANOVA).  
- Se **normalità violata** o **outlier forti** → test non parametrici (Mann–Whitney, Kruskal–Wallis).  
- Se **attesi piccoli** nelle tabelle → Fisher invece di χ².
""")

