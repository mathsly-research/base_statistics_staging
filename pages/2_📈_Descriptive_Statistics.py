# pages/2_📈_Descriptive_Statistics.py
from __future__ import annotations
import streamlit as st
import pandas as pd
import numpy as np

try:
    import plotly.express as px
    import plotly.graph_objects as go
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
st.set_page_config(page_title="Statistiche descrittive", layout="wide")
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
# Passo 2 · Tabelle descrittive (sintesi)
# ──────────────────────────────────────────────────────────────────────────────
st.subheader("Passo 2 · Tabelle descrittive")

if sel_num:
    st.markdown("**Statistiche per variabili numeriche**")
    desc_num = df[sel_num].describe().T
    desc_num["missing"] = df[sel_num].isna().sum()
    desc_num["unique"] = df[sel_num].nunique()
    st.dataframe(desc_num, use_container_width=True)

if sel_cat:
    st.markdown("**Distribuzione variabili categoriali (tabella)**")
    for c in sel_cat:
        vc = df[c].value_counts(dropna=False)
        tab = vc.reset_index(name="count").rename(columns={"index": c})
        tab[c] = tab[c].astype(str)
        tab["percent"] = (tab["count"] / len(df) * 100).round(2)
        st.markdown(f"**{c}**")
        st.dataframe(tab, use_container_width=True)

# ──────────────────────────────────────────────────────────────────────────────
# Passo 3 · Visualizzazioni migliorate
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
            left, right = st.columns([2, 1], vertical_alignment="top")
            with left:
                var_num = st.selectbox("Variabile numerica", options=sel_num, key=k("plot_num"))
            with right:
                hue = st.selectbox("Colore per sottogruppo (opz.)", options=["(nessuno)"] + cat_vars, key=k("hue"))
                hue = None if hue == "(nessuno)" else hue

            with st.expander("Opzioni grafico", expanded=True):
                r1c1, r1c2, r1c3 = st.columns(3)
                with r1c1:
                    nbins = st.slider("Numero di bin", 10, 100, 30, 5, key=k("nbins"))
                with r1c2:
                    marginal = st.selectbox("Margine", ["box", "violin", "rug", "(nessuno)"], index=0, key=k("marg"))
                with r1c3:
                    show_ecdf = st.checkbox("Mostra ECDF (curva cumulata)", value=False, key=k("ecdf"))

                r2c1, r2c2, r2c3 = st.columns(3)
                with r2c1:
                    log_x = st.checkbox("Scala logaritmica X", value=False, key=k("logx"))
                with r2c2:
                    facet_col = st.selectbox("Facet per colonna (opz.)", ["(nessuno)"] + cat_vars, key=k("facet"))
                    facet_col = None if facet_col == "(nessuno)" else facet_col
                with r2c3:
                    barmode = st.selectbox("Sovrapposizione colori", ["overlay", "stack"], key=k("bmode"))

            # Istogramma migliorato
            fig = px.histogram(
                df,
                x=var_num,
                color=hue,
                nbins=nbins,
                barmode=barmode,
                facet_col=facet_col,
                marginal=None if marginal == "(nessuno)" else marginal,
                template="simple_white",
            )
            fig.update_layout(
                title=f"Distribuzione di {var_num}",
                height=480,
                margin=dict(l=10, r=10, t=60, b=10),
                font=dict(size=16),
            )
            if log_x:
                fig.update_xaxes(type="log")

            # Hover/etichette leggibili
            fig.update_traces(hovertemplate="%{x}<br>count=%{y}")

            st.plotly_chart(fig, use_container_width=True)

            # ECDF (opzionale) — grafico cumulativo
            if show_ecdf:
                ecdf_fig = px.ecdf(
                    df, x=var_num, color=hue, facet_col=facet_col, template="simple_white"
                )
                ecdf_fig.update_layout(
                    title=f"ECDF di {var_num}",
                    height=420,
                    margin=dict(l=10, r=10, t=50, b=10),
                    font=dict(size=16),
                )
                st.plotly_chart(ecdf_fig, use_container_width=True)
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

            opt_row = st.columns([1, 1, 1])
            with opt_row[0]:
                order_mode = st.selectbox("Ordina per", ["Frequenza ↓", "Etichetta A→Z"], key=k("ord"))
            with opt_row[1]:
                other_thr = st.slider("Soglia 'Altro' (%)", 0, 20, 0, 1, key=k("other_thr"))
            with opt_row[2]:
                color_by = st.selectbox("Colore per sottogruppo (opz.)", ["(nessuno)"] + cat_vars, key=k("cat_hue"))
                color_by = None if color_by == "(nessuno)" else color_by

            # Frequenze robuste
            ser = df[var_cat]
            freq = ser.value_counts(dropna=False).reset_index(name="count").rename(columns={"index": var_cat})
            freq[var_cat] = freq[var_cat].astype(str)
            freq["percent"] = freq["count"] / len(df) * 100

            # Raggruppa in "Altro" sotto una soglia %
            if other_thr and other_thr > 0:
                major = freq[freq["percent"] >= other_thr]
                minor = freq[freq["percent"] < other_thr]
                if not minor.empty:
                    other_row = pd.DataFrame({var_cat: ["Altro"], 
                                              "count": [int(minor["count"].sum())],
                                              "percent": [minor["percent"].sum()]})
                    freq = pd.concat([major, other_row], ignore_index=True)

            # Ordine
            if order_mode == "Frequenza ↓":
                freq = freq.sort_values("count", ascending=False)
            else:
                freq = freq.sort_values(var_cat, ascending=True)

            # Top N
            if top_n and top_n > 0:
                freq = freq.head(top_n)

            # Preparazione figure
            if show_mode == "Percentuali":
                y_col = "percent"
                y_title = "Percentuale"
                textfmt = "%{text:.1f}%"
            else:
                y_col = "count"
                y_title = "Frequenza"
                textfmt = "%{text}"

            if color_by is None:
                # Bar semplice
                if orient == "Verticale":
                    fig_cat = px.bar(freq, x=var_cat, y=y_col, text=y_col, template="simple_white")
                    fig_cat.update_traces(texttemplate=textfmt, textposition="outside", cliponaxis=False)
                else:
                    fig_cat = px.bar(freq, x=y_col, y=var_cat, text=y_col, orientation="h", template="simple_white")
                    fig_cat.update_traces(texttemplate=textfmt, textposition="outside", cliponaxis=False)
            else:
                # Bar per sottogruppo: si ricostruisce una tabella long (categoria × colore)
                # Calcolo tabella pivot counts
                crosstab = (
                    df.assign(__cat__=df[var_cat].astype(str))
                      .pivot_table(index="__cat__", columns=color_by, values=color_by, aggfunc="count", fill_value=0)
                      .reset_index().rename(columns={"__cat__": var_cat})
                )
                # Eventuale 'Altro' non è ricalcolato per sottogruppo per semplicità
                melt = crosstab.melt(id_vars=[var_cat], var_name=color_by, value_name="count")
                if show_mode == "Percentuali":
                    # Percentuale sul totale di ciascuna categoria
                    tot = melt.groupby(var_cat)["count"].transform("sum")
                    melt["percent"] = melt["count"] / tot * 100
                    y_col = "percent"; y_title = "Percentuale"
                # Ordinamento coerente con ordine delle categorie principali
                cat_order = freq[var_cat].tolist()
                melt[var_cat] = pd.Categorical(melt[var_cat], categories=cat_order, ordered=True)
                if orient == "Verticale":
                    fig_cat = px.bar(melt, x=var_cat, y=y_col, color=color_by, barmode="group", template="simple_white")
                else:
                    fig_cat = px.bar(melt, x=y_col, y=var_cat, color=color_by, barmode="group",
                                     orientation="h", template="simple_white")
                fig_cat.update_traces(hovertemplate=f"%{{x}}<br>%{{y}}")
            
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
