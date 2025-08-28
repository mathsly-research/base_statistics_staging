# pages/3_📊_Explore_Distributions.py
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
# Data Store centralizzato (+ fallback)
# ──────────────────────────────────────────────────────────────────────────────
try:
    from data_store import ensure_initialized, get_active, stamp_meta
except Exception:
    def ensure_initialized():
        st.session_state.setdefault("ds_active_df", None)
        st.session_state.setdefault("ds_meta", {"version": 0, "updated_at": None, "source": None, "note": ""})
    def get_active(required: bool = True):
        ensure_initialized()
        df = st.session_state.get("ds_active_df")
        if required and (df is None or df.empty):
            st.error("Nessun dataset attivo. Importi i dati e completi la pulizia.")
            st.stop()
        return df
    def stamp_meta():
        ensure_initialized()
        meta = st.session_state["ds_meta"]
        ver = meta.get("version", 0)
        src = meta.get("source") or "-"
        ts = meta.get("updated_at")
        when = "-"
        if ts:
            from datetime import datetime
            when = datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
        c1, c2, c3 = st.columns(3)
        with c1: st.metric("Versione dati", ver)
        with c2: st.metric("Origine", src)
        with c3: st.metric("Ultimo aggiornamento", when)

# ──────────────────────────────────────────────────────────────────────────────
# Config pagina + nav
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Explore Distributions", layout="wide")
try:
    from nav import sidebar
    sidebar()
except Exception:
    pass

KEY = "ed"  # explore distributions
def k(name: str) -> str: return f"{KEY}_{name}"

# ──────────────────────────────────────────────────────────────────────────────
# Header
# ──────────────────────────────────────────────────────────────────────────────
st.title("📊 Explore Distributions")
st.caption("Esplora le distribuzioni **univariate** e le relazioni **bivariate** con opzioni semplici e guidate.")

ensure_initialized()
df = get_active(required=True)

with st.expander("Stato dati", expanded=False):
    stamp_meta()

# Preparazione liste variabili
num_vars = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
cat_vars = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]

if px is None:
    st.info("Plotly non è disponibile nell'ambiente. Aggiornare l'ambiente per visualizzazioni interattive.")
    st.stop()

# ──────────────────────────────────────────────────────────────────────────────
# Tabs principali
# ──────────────────────────────────────────────────────────────────────────────
tab_uni_num, tab_uni_cat, tab_bi = st.tabs(["🔢 Univariate · Continue", "🔠 Univariate · Categoriali", "🔗 Bivariate"])

# =============================================================================
# TAB 1 · UNIVARIATE CONTINUE
# =============================================================================
with tab_uni_num:
    st.subheader("Univariate · Variabili continue")
    if not num_vars:
        st.info("Non sono state rilevate variabili numeriche.")
    else:
        # Scelta variabile (una alla volta per evitare pagine lunghe)
        csel1, csel2, csel3 = st.columns([2, 1, 1])
        with csel1:
            x_var = st.selectbox("Variabile continua", options=num_vars, key=k("x_var"))
        with csel2:
            color_by = st.selectbox("Colore per sottogruppo (opz.)", options=["(nessuno)"] + cat_vars, key=k("color_num"))
            color_by = None if color_by == "(nessuno)" else color_by
        with csel3:
            facet_col = st.selectbox("Facet colonna (opz.)", options=["(nessuno)"] + cat_vars, key=k("facet_num"))
            facet_col = None if facet_col == "(nessuno)" else facet_col

        opt1, opt2, opt3, opt4 = st.columns(4)
        with opt1:
            nbins = st.slider("Numero di bin (istogramma)", 10, 100, 30, 5, key=k("nbins"))
        with opt2:
            add_kde = st.checkbox("Mostra densità (KDE)", value=True, key=k("kde"))
        with opt3:
            add_ecdf = st.checkbox("Mostra ECDF", value=False, key=k("ecdf"))
        with opt4:
            log_x = st.checkbox("Scala log X", value=False, key=k("logx"))

        # Due grafici affiancati: Istogramma + Densità/Box
        col_left, col_right = st.columns(2, vertical_alignment="top")

        with col_left:
            # Istogramma
            fig_h = px.histogram(
                df, x=x_var, color=color_by, nbins=nbins, facet_col=facet_col,
                template="simple_white", barmode="overlay"
            )
            fig_h.update_layout(
                title=f"Istogramma di {x_var}",
                height=420, margin=dict(l=10, r=10, t=60, b=10), font=dict(size=14)
            )
            if log_x:
                fig_h.update_xaxes(type="log")
            st.plotly_chart(fig_h, use_container_width=True)

        with col_right:
            # Densità (kde) oppure Boxplot (se si preferisce un’alternativa)
            mode = st.radio("Secondo grafico", ["Densità (KDE)", "Boxplot", "Violin"], index=0, key=k("second_plot"))
            if mode == "Densità (KDE)":
                fig_d = px.density_curve(df, x=x_var, color=color_by, facet_col=facet_col, template="simple_white") \
                        if hasattr(px, "density_curve") else px.density_contour(df, x=x_var, color=color_by, facet_col=facet_col, template="simple_white")
                # Se density_curve non esiste in questa versione di Plotly Express, fallback a histogram + density using histogram with marginal?
                try:
                    # Preferito: density estimate 1D
                    fig_kde = px.density_contour(df, x=x_var, color=color_by, facet_col=facet_col, template="simple_white")
                    fig_kde.update_traces(contours_coloring="fill", contours_showlabels=False)
                    fig_kde.update_layout(
                        title=f"Densità (KDE) di {x_var}",
                        height=420, margin=dict(l=10, r=10, t=60, b=10), font=dict(size=14)
                    )
                    if log_x:
                        fig_kde.update_xaxes(type="log")
                    st.plotly_chart(fig_kde, use_container_width=True)
                except Exception:
                    # Fallback semplice: histogram normalized
                    fig_k = px.histogram(df, x=x_var, color=color_by, facet_col=facet_col, histnorm="probability density", nbins=nbins, template="simple_white")
                    fig_k.update_layout(
                        title=f"Densità (stima) di {x_var}",
                        height=420, margin=dict(l=10, r=10, t=60, b=10), font=dict(size=14)
                    )
                    if log_x:
                        fig_k.update_xaxes(type="log")
                    st.plotly_chart(fig_k, use_container_width=True)
            elif mode == "Boxplot":
                fig_b = px.box(df, y=x_var, color=color_by, facet_col=facet_col, points="outliers", template="simple_white")
                fig_b.update_layout(
                    title=f"Boxplot di {x_var}",
                    height=420, margin=dict(l=10, r=10, t=60, b=10), font=dict(size=14)
                )
                st.plotly_chart(fig_b, use_container_width=True)
            else:
                fig_v = px.violin(df, y=x_var, color=color_by, facet_col=facet_col, box=True, points="outliers", template="simple_white")
                fig_v.update_layout(
                    title=f"Violin plot di {x_var}",
                    height=420, margin=dict(l=10, r=10, t=60, b=10), font=dict(size=14)
                )
                st.plotly_chart(fig_v, use_container_width=True)

        # ECDF aggiuntiva (intera larghezza)
        if add_ecdf:
            try:
                fig_e = px.ecdf(df, x=x_var, color=color_by, facet_col=facet_col, template="simple_white")
                fig_e.update_layout(
                    title=f"ECDF di {x_var}",
                    height=380, margin=dict(l=10, r=10, t=60, b=10), font=dict(size=14)
                )
                if log_x:
                    fig_e.update_xaxes(type="log")
                st.plotly_chart(fig_e, use_container_width=True)
            except Exception:
                st.info("ECDF non supportata da questa versione di Plotly.")

# =============================================================================
# TAB 2 · UNIVARIATE CATEGORIALI
# =============================================================================
with tab_uni_cat:
    st.subheader("Univariate · Variabili categoriali")
    if not cat_vars:
        st.info("Non sono state rilevate variabili categoriali.")
    else:
        cl1, cl2, cl3, cl4 = st.columns([2, 1, 1, 1])
        with cl1:
            c_var = st.selectbox("Variabile categoriale", options=cat_vars, key=k("c_var"))
        with cl2:
            measure = st.selectbox("Misura", ["Conteggi", "Percentuali"], key=k("c_measure"))
        with cl3:
            orient = st.selectbox("Orientamento", ["Verticale", "Orizzontale"], key=k("c_orient"))
        with cl4:
            top_n = st.number_input("Top N (0 = tutte)", min_value=0, value=0, step=1, key=k("c_topn"))

        r1, r2 = st.columns(2)
        with r1:
            order_mode = st.selectbox("Ordina per", ["Frequenza ↓", "Etichetta A→Z"], key=k("c_order"))
        with r2:
            other_thr = st.slider("Soglia 'Altro' (%)", 0, 20, 0, 1, key=k("c_other"))

        # Frequenze robuste
        ser = df[c_var]
        freq = ser.value_counts(dropna=False).reset_index(name="count").rename(columns={"index": c_var})
        freq[c_var] = freq[c_var].astype(str)
        freq["percent"] = freq["count"] / len(df) * 100

        # Raggruppa in "Altro"
        if other_thr and other_thr > 0:
            major = freq[freq["percent"] >= other_thr]
            minor = freq[freq["percent"] < other_thr]
            if not minor.empty:
                other_row = pd.DataFrame({c_var: ["Altro"], "count": [int(minor["count"].sum())], "percent": [minor["percent"].sum()]})
                freq = pd.concat([major, other_row], ignore_index=True)

        # Ordine
        if order_mode == "Frequenza ↓":
            freq = freq.sort_values("count", ascending=False)
        else:
            freq = freq.sort_values(c_var, ascending=True)

        # Top N
        if top_n and top_n > 0:
            freq = freq.head(top_n)

        # Scelta metrica
        if measure == "Percentuali":
            y_col = "percent"; y_title = "Percentuale"; textfmt = "%{text:.1f}%"
        else:
            y_col = "count"; y_title = "Frequenza"; textfmt = "%{text}"

        # Grafico
        if orient == "Verticale":
            fig_c = px.bar(freq, x=c_var, y=y_col, text=y_col, template="simple_white")
            fig_c.update_traces(texttemplate=textfmt, textposition="outside", cliponaxis=False)
        else:
            fig_c = px.bar(freq, x=y_col, y=c_var, text=y_col, orientation="h", template="simple_white")
            fig_c.update_traces(texttemplate=textfmt, textposition="outside", cliponaxis=False)

        fig_c.update_layout(
            title=f"Distribuzione di {c_var}",
            yaxis_title=y_title if orient == "Verticale" else None,
            xaxis_title=None if orient == "Verticale" else y_title,
            height=500, margin=dict(l=10, r=10, t=60, b=10), font=dict(size=16),
        )
        st.plotly_chart(fig_c, use_container_width=True)

# =============================================================================
# TAB 3 · BIVARIATE (rapide)
# =============================================================================
with tab_bi:
    st.subheader("Bivariate rapide")

    # Selettori principali
    b1, b2, b3 = st.columns(3)
    with b1:
        y_num = st.selectbox("Numerica (Y, opz.)", options=["(nessuna)"] + num_vars, key=k("bi_y"))
        y_num = None if y_num == "(nessuna)" else y_num
    with b2:
        x_cat = st.selectbox("Categoriale (X, opz.)", options=["(nessuna)"] + cat_vars, key=k("bi_xcat"))
        x_cat = None if x_cat == "(nessuna)" else x_cat
    with b3:
        color = st.selectbox("Colore (opz.)", options=["(nessuno)"] + cat_vars, key=k("bi_color"))
        color = None if color == "(nessuno)" else color

    # Caso 1: Numerica ~ Categoriale  → box/violin per gruppi
    if y_num and x_cat:
        st.markdown(f"**{y_num} in funzione di {x_cat}**")
        opt = st.radio("Tipo grafico", ["Boxplot", "Violin"], index=0, key=k("bi_type"))
        if opt == "Boxplot":
            fig = px.box(df, x=x_cat, y=y_num, color=color, points="outliers", template="simple_white")
        else:
            fig = px.violin(df, x=x_cat, y=y_num, color=color, box=True, points="outliers", template="simple_white")
        fig.update_layout(height=520, margin=dict(l=10, r=10, t=60, b=10), font=dict(size=15))
        st.plotly_chart(fig, use_container_width=True)

    # Caso 2: Categoriale ~ Categoriale  → barre grouped/stacked (+ normalizzazione %)
    elif x_cat and color:
        st.markdown(f"**Distribuzione di {x_cat} per {color}**")
        mode = st.selectbox("Modalità barre", ["group", "stack", "percent"], key=k("bi_barmode"))
        ct = (
            df.assign(__x__=df[x_cat].astype(str), __c__=df[color].astype(str))
              .pivot_table(index="__x__", columns="__c__", values=color, aggfunc="count", fill_value=0)
              .reset_index().rename(columns={"__x__": x_cat})
        )
        melted = ct.melt(id_vars=[x_cat], var_name=color, value_name="count")
        if mode == "percent":
            melted["total"] = melted.groupby(x_cat)["count"].transform("sum").replace(0, np.nan)
            melted["percent"] = melted["count"] / melted["total"] * 100
            y_col, y_title = "percent", "Percentuale"
        else:
            y_col, y_title = "count", "Frequenza"
        fig = px.bar(melted, x=x_cat, y=y_col, color=color,
                     barmode=("stack" if mode in ["stack", "percent"] else "group"),
                     template="simple_white", text=y_col if mode != "percent" else None)
        if mode == "percent":
            fig.update_yaxes(range=[0, 100])
        fig.update_layout(height=520, margin=dict(l=10, r=10, t=60, b=10), font=dict(size=15), yaxis_title=y_title)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Selezioni una combinazione tra *Numerica (Y)* e/o *Categoriale (X)* per vedere i grafici bivariati.")

# ──────────────────────────────────────────────────────────────────────────────
# Navigazione
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("---")
nav1, nav2 = st.columns(2)
with nav1:
    if st.button("⬅️ Torna: Statistiche descrittive", use_container_width=True, key=k("go_prev")):
        st.switch_page("pages/2_📈_Descriptive_Statistics.py")
with nav2:
    if st.button("➡️ Vai: Assumption Checks", use_container_width=True, key=k("go_next")):
        st.switch_page("pages/4_🔎_Assumption_Checks.py")
