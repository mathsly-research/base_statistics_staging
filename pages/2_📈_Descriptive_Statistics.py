# pages/2_📈_Descriptive_Statistics.py
from __future__ import annotations
import streamlit as st
import pandas as pd
import numpy as np

try:
    import plotly.express as px
except Exception:
    px = None

# ──────────────────────────────────────────────────────────────────────────────
# Data store (preferito) + fallback
# ──────────────────────────────────────────────────────────────────────────────
try:
    from data_store import ensure_initialized, get_active, stamp_meta
except Exception:
    def ensure_initialized():
        st.session_state.setdefault("ds_active_df", None)
        st.session_state.setdefault("ds_meta", {"version": 0, "updated_at": None, "source": None, "note": ""})
    def get_active(required: bool = True):
        ensure_initialized(); df = st.session_state.get("ds_active_df")
        if required and (df is None or df.empty):
            st.error("Nessun dataset attivo. Esegua Upload/Pulizia dati."); st.stop()
        return df
    def stamp_meta():
        meta = st.session_state["ds_meta"]; ver = meta.get("version", 0); src = meta.get("source") or "-"
        ts = meta.get("updated_at"); when = "-"
        if ts:
            from datetime import datetime; when = datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
        c1, c2, c3 = st.columns(3)
        with c1: st.metric("Versione dati", ver)
        with c2: st.metric("Origine", src)
        with c3: st.metric("Ultimo aggiornamento", when)

# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="📈 Statistiche descrittive", layout="wide")
try:
    from nav import sidebar; sidebar()
except Exception:
    pass

KEY = "ds"
def k(name: str) -> str: return f"{KEY}_{name}"

# ──────────────────────────────────────────────────────────────────────────────
st.title("📈 Statistiche descrittive")
ensure_initialized()
df = get_active(required=True)

with st.expander("Stato dati", expanded=False):
    stamp_meta()

# ──────────────────────────────────────────────────────────────────────────────
# Passo 1 · Selezione variabili
# ──────────────────────────────────────────────────────────────────────────────
st.subheader("Passo 1 · Selezione variabili")

num_vars = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
cat_vars = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]

c1, c2 = st.columns(2)
with c1:
    sel_num = st.multiselect("Variabili numeriche", options=num_vars, default=num_vars[:3], key=k("sel_num"))
with c2:
    sel_cat = st.multiselect("Variabili categoriali", options=cat_vars, default=cat_vars[:2], key=k("sel_cat"))

if not sel_num and not sel_cat:
    st.warning("Selezioni almeno una variabile (numerica o categoriale).")
    st.stop()

# ──────────────────────────────────────────────────────────────────────────────
# Passo 2 · Tabelle descrittive
# ──────────────────────────────────────────────────────────────────────────────
st.subheader("Passo 2 · Tabelle descrittive")

if sel_num:
    st.markdown("**Statistiche per variabili numeriche**")
    desc_num = df[sel_num].describe().T
    desc_num["missing"] = df[sel_num].isna().sum()
    desc_num["unique"] = df[sel_num].nunique()
    st.dataframe(desc_num, use_container_width=True)

if sel_cat:
    st.markdown("**Distribuzione variabili categoriali**")
    for c in sel_cat:
        vc = df[c].value_counts(dropna=False)
        tab = vc.reset_index(name="count").rename(columns={"index": c})
        tab[c] = tab[c].astype(str)
        tab["percent"] = (tab["count"] / len(df) * 100).round(2)
        st.markdown(f"**{c}**")
        st.dataframe(tab, use_container_width=True)

# ──────────────────────────────────────────────────────────────────────────────
# Passo 3 · Visualizzazioni
# ──────────────────────────────────────────────────────────────────────────────
st.subheader("Passo 3 · Visualizzazioni")

if px is None:
    st.info("Plotly non disponibile nell'ambiente.")
else:
    tabs = st.tabs(["🔢 Numeriche", "🔠 Categoriali"])

    # ==============================
    # NUMERICHE
    # ==============================
    with tabs[0]:
        if sel_num:
            for var_num in sel_num:
                st.markdown(f"### 🔢 {var_num}")

                left, right = st.columns(2, vertical_alignment="top")

                with left:
                    nbins = st.slider(
                        f"Numero bin per {var_num}", 10, 100, 30, 5, key=k(f"bins_{var_num}")
                    )
                    hue = st.selectbox(
                        f"Colore per sottogruppo ({var_num})",
                        options=["(nessuno)"] + cat_vars,
                        key=k(f"hue_{var_num}")
                    )
                    hue = None if hue == "(nessuno)" else hue

                    fig_hist = px.histogram(
                        df,
                        x=var_num,
                        color=hue,
                        nbins=nbins,
                        template="simple_white",
                    )
                    fig_hist.update_layout(
                        title=f"Istogramma di {var_num}",
                        height=400,
                        margin=dict(l=10, r=10, t=50, b=10),
                        font=dict(size=14)
                    )
                    st.plotly_chart(fig_hist, use_container_width=True)

                with right:
                    style = st.selectbox(
                        f"Tipo grafico ({var_num})",
                        ["Boxplot", "Violin"],
                        key=k(f"style_{var_num}")
                    )
                    if style == "Boxplot":
                        fig_box = px.box(
                            df, y=var_num, color=hue, points="all", template="simple_white"
                        )
                        fig_box.update_layout(
                            title=f"Boxplot di {var_num}",
                            height=400,
                            margin=dict(l=10, r=10, t=50, b=10),
                            font=dict(size=14)
                        )
                        st.plotly_chart(fig_box, use_container_width=True)
                    else:
                        fig_violin = px.violin(
                            df, y=var_num, color=hue, box=True, points="all", template="simple_white"
                        )
                        fig_violin.update_layout(
                            title=f"Violin plot di {var_num}",
                            height=400,
                            margin=dict(l=10, r=10, t=50, b=10),
                            font=dict(size=14)
                        )
                        st.plotly_chart(fig_violin, use_container_width=True)
        else:
            st.info("Nessuna variabile numerica selezionata.")

    # ==============================
    # CATEGORIALI
    # ==============================
    with tabs[1]:
        if sel_cat:
            top_row = st.columns([2, 1, 1, 1])
            with top_row[0]:
                var_cat = st.selectbox("Variabile categoriale", options=sel_cat, key=k("plot_cat"))
            with top_row[1]:
                show_mode = st.selectbox("Misura", ["Conteggi", "Percentuali"], key=k("measure"))
            with top_row[2]:
                top_n = st.number_input("Top N (0 = tutte)", min_value=0, value=0, step=1, key=k("topn"))
            with top_row[3]:
                orient = st.selectbox("Orientamento", ["Verticale", "Orizzontale"], key=k("orient"))

            order_mode = st.selectbox("Ordina per", ["Frequenza ↓", "Etichetta A→Z"], key=k("ord"))

            # Frequenze
            ser = df[var_cat]
            freq = ser.value_counts(dropna=False).reset_index(name="count").rename(columns={"index": var_cat})
            freq[var_cat] = freq[var_cat].astype(str)
            freq["percent"] = freq["count"] / len(df) * 100

            # Ordine
            if order_mode == "Frequenza ↓":
                freq = freq.sort_values("count", ascending=False)
            else:
                freq = freq.sort_values(var_cat, ascending=True)

            # Top N
            if top_n and top_n > 0:
                freq = freq.head(top_n)

            if show_mode == "Percentuali":
                y_col = "percent"; y_title = "Percentuale"; textfmt = "%{text:.1f}%"
            else:
                y_col = "count"; y_title = "Frequenza"; textfmt = "%{text}"

            if orient == "Verticale":
                fig_cat = px.bar(freq, x=var_cat, y=y_col, text=y_col, template="simple_white")
                fig_cat.update_traces(texttemplate=textfmt, textposition="outside", cliponaxis=False)
            else:
                fig_cat = px.bar(freq, x=y_col, y=var_cat, text=y_col, orientation="h", template="simple_white")
                fig_cat.update_traces(texttemplate=textfmt, textposition="outside", cliponaxis=False)

            fig_cat.update_layout(
                title=f"Distribuzione di {var_cat}",
                yaxis_title=y_title if orient == "Verticale" else None,
                xaxis_title=None if orient == "Verticale" else y_title,
                height=520,
                margin=dict(l=10, r=10, t=60, b=10),
                font=dict(size=16),
            )
            st.plotly_chart(fig_cat, use_container_width=True)
        else:
            st.info("Nessuna variabile categoriale selezionata.")

# ──────────────────────────────────────────────────────────────────────────────
# Passo 4 · Esportazione
# ──────────────────────────────────────────────────────────────────────────────
st.subheader("Passo 4 · Esportazione")
csv_bytes = df.to_csv(index=False).encode("utf-8")
st.download_button(
    "⬇️ Scarica CSV corrente",
    data=csv_bytes,
    file_name="dataset_corrente.csv",
    mime="text/csv",
    key=k("download")
)
