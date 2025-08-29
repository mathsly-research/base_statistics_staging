# pages/6_🔗_Correlation_Analysis.py
from __future__ import annotations
import math
import streamlit as st
import pandas as pd
import numpy as np

# Plotting (opzionale)
try:
    import plotly.express as px
    import plotly.graph_objects as go
except Exception:
    px = None
    go = None

# Statistiche (opzionale)
try:
    from scipy import stats
    from scipy.cluster.hierarchy import linkage, leaves_list
    from scipy.spatial.distance import squareform
except Exception:
    stats = None
    linkage = None
    leaves_list = None
    squareform = None

try:
    import statsmodels.api as sm
except Exception:
    sm = None

# ──────────────────────────────────────────────────────────────────────────────
# Data store centralizzato (+ fallback)
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
st.set_page_config(page_title="Correlation Analysis", layout="wide")
try:
    from nav import sidebar
    sidebar()
except Exception:
    pass

KEY = "ca"  # correlation analysis
def k(name: str) -> str: return f"{KEY}_{name}"

# ──────────────────────────────────────────────────────────────────────────────
# Header
# ──────────────────────────────────────────────────────────────────────────────
st.title("🔗 Correlation Analysis")
st.caption("Matrice delle correlazioni, analisi puntuale delle coppie e correlazioni parziali, con guida alla lettura.")

ensure_initialized()
df = get_active(required=True)

with st.expander("Stato dati", expanded=False):
    stamp_meta()

# Variabili
num_vars = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
cat_vars = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]

if not num_vars:
    st.warning("Non sono state rilevate variabili numeriche.")
    st.stop()

if px is None:
    st.info("Plotly non è disponibile nell'ambiente. Le visualizzazioni interattive potrebbero non comparire.")

# ──────────────────────────────────────────────────────────────────────────────
# Helper
# ──────────────────────────────────────────────────────────────────────────────
def _safe_num_pair(x: pd.Series, y: pd.Series):
    """Allinea X e Y e rimuove NA pairwise."""
    xy = pd.DataFrame({"x": pd.to_numeric(x, errors="coerce"),
                       "y": pd.to_numeric(y, errors="coerce")}).dropna()
    return xy["x"].values, xy["y"].values

def _corr_and_p(x: np.ndarray, y: np.ndarray, method: str):
    """Ritorna (rho, p, n, ci_lo, ci_hi) – CI solo per Pearson."""
    n = min(len(x), len(y))
    if stats is None or n < 3:
        return (np.nan, np.nan, n, np.nan, np.nan)
    if method == "Pearson":
        r, p = stats.pearsonr(x, y)
        # CI via Fisher z
        if n > 3:
            z = 0.5 * np.log((1+r)/(1-r))
            se = 1 / math.sqrt(n - 3)
            zcrit = stats.norm.ppf(1 - 0.05/2)
            lo_z, hi_z = z - zcrit*se, z + zcrit*se
            lo = (np.exp(2*lo_z)-1)/(np.exp(2*lo_z)+1)
            hi = (np.exp(2*hi_z)-1)/(np.exp(2*hi_z)+1)
        else:
            lo = hi = np.nan
        return (r, p, n, lo, hi)
    elif method == "Spearman":
        r, p = stats.spearmanr(x, y)
        return (r, p, n, np.nan, np.nan)
    else:
        r, p = stats.kendalltau(x, y)
        return (r, p, n, np.nan, np.nan)

def _p_adjust(pvals: np.ndarray, method: str):
    """Holm / BH (FDR) / None"""
    p = np.array(pvals, dtype=float)
    m = len(p)
    if m == 0 or method == "Nessuna":
        return p
    order = np.argsort(p)
    inv = np.empty_like(order); inv[order] = np.arange(m)
    if method == "Holm":
        sorted_p = p[order]
        adj_sorted = (m - np.arange(m)) * sorted_p
        for i in range(m-2, -1, -1):
            adj_sorted[i] = max(adj_sorted[i], adj_sorted[i+1])
        adj = np.minimum(adj_sorted[inv], 1.0)
        return adj
    elif method in ("BH (FDR)", "Benjamini–Hochberg"):
        sorted_p = p[order]
        adj_sorted = m / (np.arange(1, m+1)) * sorted_p
        for i in range(m-2, -1, -1):
            adj_sorted[i] = min(adj_sorted[i], adj_sorted[i+1])
        adj = np.minimum(adj_sorted[inv], 1.0)
        return adj
    else:
        return p

def _cluster_order(corr: pd.DataFrame):
    """Ordina variabili tramite clustering gerarchico su 1-|r| (se SciPy disponibile)."""
    try:
        if linkage is None or leaves_list is None or squareform is None:
            score = corr.abs().sum().sort_values(ascending=False)
            return score.index.tolist()
        d = 1 - corr.abs()
        np.fill_diagonal(d.values, 0.0)
        condensed = squareform(d.values, checks=False)
        Z = linkage(condensed, method="average")
        order_idx = leaves_list(Z)
        return corr.index[order_idx].tolist()
    except Exception:
        return corr.columns.tolist()

def _partial_corr(x: pd.Series, y: pd.Series, covars: list[str], method: str = "Pearson"):
    """Correlazione parziale: regressa X e Y su covariate e correla i residui.
       Per Spearman, applica rank-transform prima di regressione."""
    if len(covars) == 0:
        xx, yy = _safe_num_pair(x, y)
        return _corr_and_p(xx, yy, "Pearson" if method == "Pearson" else "Spearman")
    X = pd.DataFrame({"_x": x, "_y": y}, copy=True)
    for c in covars: X[c] = df[c]
    if method == "Spearman":
        X = X.apply(lambda s: s.rank(method="average") if pd.api.types.is_numeric_dtype(s) else s)
    X = X.apply(pd.to_numeric, errors="coerce").dropna()
    if X.shape[0] < 5:
        return (np.nan, np.nan, X.shape[0], np.nan, np.nan)
    try:
        if sm is None:
            Z = X[covars].to_numpy()
            Z = np.column_stack([np.ones(len(Z)), Z])
            beta_x, *_ = np.linalg.lstsq(Z, X["_x"].to_numpy(), rcond=None)
            beta_y, *_ = np.linalg.lstsq(Z, X["_y"].to_numpy(), rcond=None)
            rx = X["_x"].to_numpy() - Z.dot(beta_x)
            ry = X["_y"].to_numpy() - Z.dot(beta_y)
        else:
            Z = sm.add_constant(X[covars].astype(float))
            rx = sm.OLS(X["_x"].astype(float), Z).fit().resid
            ry = sm.OLS(X["_y"].astype(float), Z).fit().resid
        return _corr_and_p(np.asarray(rx), np.asarray(ry), "Pearson" if method == "Pearson" else "Spearman")
    except Exception:
        return (np.nan, np.nan, X.shape[0], np.nan, np.nan)

# ──────────────────────────────────────────────────────────────────────────────
# Tabs principali
# ──────────────────────────────────────────────────────────────────────────────
tab_matrix, tab_pair, tab_partial = st.tabs(["🧩 Matrice", "🔍 Coppia", "🧭 Parziali"])

# =============================================================================
# TAB 1 · MATRICE
# =============================================================================
with tab_matrix:
    st.subheader("🧩 Matrice delle correlazioni")
    top1, top2, top3 = st.columns([2,2,2])
    with top1:
        method = st.selectbox("Metodo", ["Pearson", "Spearman", "Kendall"], key=k("m_method"))
        vars_sel = st.multiselect("Variabili (numeriche)", options=num_vars, default=num_vars[: min(10, len(num_vars))], key=k("m_vars"))
    with top2:
        miss = st.selectbox("Gestione NA", ["pairwise (consigliato)", "listwise"], key=k("m_na"))
        reorder = st.selectbox("Ordinamento", ["A–Z", "Clustering (|r|)"], key=k("m_ord"))
    with top3:
        padj = st.selectbox("Correzione p-value", ["Nessuna", "Holm", "BH (FDR)"], key=k("m_padj"))
        show_vals = st.checkbox("Mostra valori r nella heatmap", value=True, key=k("m_showvals"))

    if not vars_sel:
        st.info("Selezioni almeno una variabile.")
    else:
        # Calcolo matrice r e p
        V = vars_sel
        r_mat = pd.DataFrame(np.eye(len(V)), index=V, columns=V, dtype=float)
        p_mat = pd.DataFrame(np.zeros((len(V), len(V))), index=V, columns=V, dtype=float)
        n_mat = pd.DataFrame(np.zeros((len(V), len(V))), index=V, columns=V, dtype=int)

        for i, a in enumerate(V):
            for j, b in enumerate(V[i+1:], start=i+1):
                if miss.startswith("pairwise"):
                    xa, yb = _safe_num_pair(df[a], df[b])
                else:
                    pair_df = df[V].dropna()
                    xa, yb = pair_df[a].values, pair_df[b].values
                r, p, n, _, _ = _corr_and_p(xa, yb, method)
                r_mat.loc[a, b] = r_mat.loc[b, a] = r
                p_mat.loc[a, b] = p_mat.loc[b, a] = p
                n_mat.loc[a, b] = n_mat.loc[b, a] = n
        # p-adjust (sopra diagonale)
        tril_idx = np.triu_indices(len(V), k=1)
        pvals = p_mat.values[tril_idx]
        padj_vals = _p_adjust(pvals, padj)
        P = p_mat.copy()
        P.values[tril_idx] = padj_vals
        P.values[(tril_idx[1], tril_idx[0])] = padj_vals  # copia simmetrica

        # Reordering
        if reorder.startswith("Clustering"):
            order = _cluster_order(r_mat)
            r_plot = r_mat.loc[order, order]
            P_plot = P.loc[order, order]
        else:
            order = sorted(V)
            r_plot = r_mat.loc[order, order]
            P_plot = P.loc[order, order]

        # Heatmap (FIX: niente text_auto=array; uso text + texttemplate)
        if px is not None:
            fig = px.imshow(r_plot, aspect="auto", origin="lower",
                            labels=dict(color="r"), template="simple_white",
                            title=f"Matrice di correlazione ({method})")
            if show_vals:
                text_vals = np.round(r_plot.values, 2)
                fig.update_traces(text=text_vals, texttemplate="%{text}", 
                                  hovertemplate="r=%{z:.3f}<extra></extra>")
            st.plotly_chart(fig, use_container_width=True, theme=None)

        col_l, col_r = st.columns([3,2])
        with col_l:
            st.markdown("**Dettagli (r, p-adj, n)**")
            show_tri = st.radio("Formato tabella", ["Superiore", "Completa"], horizontal=True, key=k("m_tri"))
            rows = []
            for i, a in enumerate(r_plot.index):
                rng = range(i+1, len(r_plot.columns)) if show_tri == "Superiore" else range(len(r_plot.columns))
                for j in rng:
                    b = r_plot.columns[j]
                    if a == b and show_tri == "Completa":
                        rows.append([a, b, 1.0, 0.0, int(n_mat.loc[a, b])])
                    elif a != b:
                        rows.append([a, b, r_plot.loc[a, b], P_plot.loc[a, b], int(n_mat.loc[a, b])])
            out = pd.DataFrame(rows, columns=["Var A", "Var B", "r", "p (adj)", "n"]).sort_values(["Var A","Var B"])
            st.dataframe(out, use_container_width=True, height=360)
            csv = out.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Scarica tabella (CSV)", csv, file_name=f"correlation_matrix_{method}.csv", mime="text/csv", key=k("m_dl"))
        with col_r:
            with st.expander("ℹ️ Come leggere", expanded=False):
                st.markdown(f"""
- **Metodo**: {method}.  
- **r** ∈ [−1, +1]; |r|≈0.1/0.3/0.5 → piccolo/medio/grande (indicativo).  
- **p (adj)**: p-value **corretto** ({padj}) per confronti multipli.  
- **Clustering (|r|)** raggruppa variabili con profili simili di correlazione.  
- **NA**: *pairwise* usa tutte le coppie disponibili; *listwise* rimuove righe con NA su **tutte** le variabili selezionate.  
""")

# =============================================================================
# TAB 2 · COPPIA
# =============================================================================
with tab_pair:
    st.subheader("🔍 Analisi di una coppia")
    c1, c2, c3 = st.columns([2,2,2])
    with c1:
        x_var = st.selectbox("Variabile X (numerica)", options=num_vars, key=k("p_x"))
    with c2:
        y_var = st.selectbox("Variabile Y (numerica)", options=[c for c in num_vars if c != x_var], key=k("p_y"))
    with c3:
        method_p = st.selectbox("Metodo", ["Pearson", "Spearman", "Kendall"], key=k("p_meth"))

    color_by = st.selectbox("Colore per sottogruppo (opzionale)", options=["(nessuno)"] + cat_vars, key=k("p_color"))
    color_by = None if color_by == "(nessuno)" else color_by

    x_arr, y_arr = _safe_num_pair(df[x_var], df[y_var])
    r, p, n, lo, hi = _corr_and_p(x_arr, y_arr, method_p)

    st.markdown(
        f"**n = {n}**, **r = {r:.3f}**, **p = {p:.4f}**"
        + (f", **CI95% r = ({lo:.3f}, {hi:.3f})**" if method_p == "Pearson" and not (np.isnan(lo) or np.isnan(hi)) else "")
    )

    if px is not None:
        plot_df = pd.DataFrame({x_var: df[x_var], y_var: df[y_var]})
        if color_by:
            plot_df[color_by] = df[color_by].astype(str)
        fig = px.scatter(plot_df, x=x_var, y=y_var, color=color_by if color_by else None,
                         template="simple_white", title=f"{y_var} vs {x_var}")
        if method_p == "Pearson" and sm is not None:
            try:
                fig_tr = px.scatter(plot_df.dropna(), x=x_var, y=y_var, color=color_by if color_by else None,
                                    template="simple_white", trendline="ols")
                for tr in fig_tr.data:
                    if "trendline" in tr.name:
                        fig.add_trace(tr)
            except Exception:
                pass
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("ℹ️ Come leggere", expanded=False):
        st.markdown(f"""
- **{method_p}**: misura la relazione tra **{x_var}** e **{y_var}** ({'lineare' if method_p=='Pearson' else 'monotona'}).  
- **r** indica forza e direzione; **p** la significatività; per Pearson è mostrata la **CI95%**.  
- Verifichi **outlier** e **non linearità** dal grafico; il trendline è mostrato solo per Pearson.  
""")

# =============================================================================
# TAB 3 · PARZIALI
# =============================================================================
with tab_partial:
    st.subheader("🧭 Correlazioni parziali")
    if len(num_vars) < 3:
        st.info("Servono almeno due variabili numeriche più ≥1 covariata.")
    else:
        c1, c2 = st.columns([2,2])
        with c1:
            x_var2 = st.selectbox("Variabile X", options=num_vars, key=k("pc_x"))
            y_candidates = [c for c in num_vars if c != x_var2]
            y_var2 = st.selectbox("Variabile Y", options=y_candidates, key=k("pc_y"))
        with c2:
            covars = st.multiselect("Covariate (controllo)", options=[c for c in num_vars if c not in (x_var2, y_var2)], key=k("pc_cov"))
            method_pc = st.selectbox("Metodo", ["Pearson", "Spearman"], key=k("pc_method"))

        r, p, n, lo, hi = _partial_corr(df[x_var2], df[y_var2], covars, method_pc)
        st.markdown(
            f"**n = {n}**, **r_parz = {r:.3f}**, **p = {p:.4f}**"
            + (f", **CI95% r (appross.) = ({lo:.3f}, {hi:.3f})**" if method_pc == "Pearson" and not (np.isnan(lo) or np.isnan(hi)) else "")
        )

        if px is not None and len(covars) > 0:
            try:
                data = pd.DataFrame({"_x": df[x_var2], "_y": df[y_var2]}, copy=True)
                for c in covars: data[c] = df[c]
                data = data.apply(pd.to_numeric, errors="coerce").dropna()
                if sm is not None and data.shape[0] > 3:
                    Z = sm.add_constant(data[covars].astype(float))
                    rx = sm.OLS(data["_x"].astype(float), Z).fit().resid
                    ry = sm.OLS(data["_y"].astype(float), Z).fit().resid
                    plot_df2 = pd.DataFrame({"Residuo_X": rx, "Residuo_Y": ry})
                    fig2 = px.scatter(plot_df2, x="Residuo_X", y="Residuo_Y", template="simple_white",
                                      title=f"Residui: {x_var2}|cov vs {y_var2}|cov")
                    st.plotly_chart(fig2, use_container_width=True)
            except Exception:
                pass

        with st.expander("ℹ️ Come leggere", expanded=False):
            st.markdown("""
- **Correlazione parziale**: relazione tra X e Y **al netto** delle covariate.  
- Se r_parz rimane simile a r grezzo → l’associazione non è spiegata dalle covariate; se diminuisce molto → probabile **confondimento**.  
- Per **Spearman**, la parziale è calcolata su **ranghi** (robusta a outlier e non linearità monotone).
""")

# ──────────────────────────────────────────────────────────────────────────────
# Navigazione
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("---")
nav1, nav2 = st.columns(2)
with nav1:
    if st.button("⬅️ Torna: Test statistici", use_container_width=True, key=k("go_prev")):
        st.switch_page("pages/5_🧪_Statistical_Tests.py")
with nav2:
    if st.button("➡️ Vai: (Prossimo modulo)", use_container_width=True, key=k("go_next")):
        # Adegui questo percorso al nome reale del file successivo nel suo progetto
        st.switch_page("pages/7_📐_Regression.py")
