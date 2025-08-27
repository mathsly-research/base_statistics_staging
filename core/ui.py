
import streamlit as st
import pandas as pd

def quality_alert(diag: dict):
    score = diag.get("quality_score", 0)
    badge = "🟢 Buona" if score >= 80 else "🟠 Discreta" if score >= 60 else "🔴 Critica"
    msg = f"**Qualità dataset:** {badge} (score {score}/100)"
    if score >= 80:
        st.success(msg)
    elif score >= 60:
        st.warning(msg)
    else:
        st.error(msg)

    st.caption(f"Righe: {diag['n_rows']} • Colonne: {diag['n_cols']} • Memoria: {diag['memory_mb']:.2f} MB")

    if diag["n_duplicates"] > 0:
        st.info(f"🔁 Righe duplicate: {diag['n_duplicates']}")
    if len(diag["constant_cols"]) > 0:
        st.info(f"📏 Colonne costanti: {', '.join(diag['constant_cols'][:5])}{'…' if len(diag['constant_cols'])>5 else ''}")
    if diag["high_cardinality"]:
        st.info("🔢 Cardinalità elevata (categoriche): " + ", ".join([f"{k} ({v})" for k,v in list(diag['high_cardinality'].items())[:5]]))

def show_missing_table(diag: dict):
    miss = diag.get("missing_by_col")
    if isinstance(miss, pd.Series):
        df = miss.to_frame(name="% missing").mul(100).round(1)
        st.dataframe(df, use_container_width=True, height=min(400, 30+24*len(df)))
