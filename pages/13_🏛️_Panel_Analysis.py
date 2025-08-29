# -*- coding: utf-8 -*-
# pages/13_🏛️_Panel_Analysis.py
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

# Plotly (diagnostica)
try:
    import plotly.express as px
    import plotly.graph_objects as go
except Exception:
    px = None
    go = None

# Econometria/panel
try:
    from linearmodels import PanelOLS, RandomEffects, PooledOLS
    _has_lm = True
except Exception:
    _has_lm = False

# Statsmodels per fallback e utilità
try:
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    _has_sm = True
except Exception:
    _has_sm = False

try:
    from scipy import stats as sps
    _has_scipy = True
except Exception:
    _has_scipy = False

# ──────────────────────────────────────────────────────────────────────────────
# Data store comune
# ──────────────────────────────────────────────────────────────────────────────
try:
    from data_store import ensure_initialized, get_active, stamp_meta
except Exception:
    def ensure_initialized():
        st.session_state.setdefault("ds_active_df", None)
        st.session_state.setdefault("ds_meta", {"version": 0, "updated_at": None, "source": None, "note": ""})
    def get_active(required: bool = True):
        ensure_initialized()
        df_ = st.session_state.get("ds_active_df")
        if required and (df_ is None or df_.empty):
            st.error("Nessun dataset attivo. Importi i dati e completi la pulizia.")
            st.stop()
        return df_
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
# Config + nav
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="🏛️ Panel Analysis (Econometria)", layout="wide")
try:
    from nav import sidebar
    sidebar()
except Exception:
    pass

KEY = "panel"
def k(name: str) -> str:
    return f"{KEY}_{name}"

# ──────────────────────────────────────────────────────────────────────────────
# Utility
# ──────────────────────────────────────────────────────────────────────────────
def fmt_p(p: float | None) -> str:
    if p is None or p != p:
        return "—"
    if p < 1e-4:
        return "< 1e-4"
    return f"{p:.4f}"

def standardize_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if pd.api.types.is_numeric_dtype(out[c]):
            s = pd.to_numeric(out[c], errors="coerce")
            mu, sd = float(s.mean()), float(s.std(ddof=1))
            if sd and sd > 0:
                out[c] = (s - mu) / sd
    return out

def make_panel_index(df: pd.DataFrame, entity_col: str, time_col: str) -> pd.DataFrame:
    d = df.copy()
    d[entity_col] = d[entity_col].astype(str)
    # time può essere numerico, datetime o stringa
    if pd.api.types.is_datetime64_any_dtype(d[time_col]):
        pass
    elif pd.api.types.is_numeric_dtype(d[time_col]):
        pass
    else:
        d[time_col] = d[time_col].astype(str)
    d = d.set_index([entity_col, time_col]).sort_index()
    return d

def balance_summary(df: pd.DataFrame, entity_col: str, time_col: str) -> dict:
    n_entities = df[entity_col].nunique()
    n_periods = df[time_col].nunique()
    counts = df.groupby(entity_col)[time_col].nunique()
    balanced = (counts.min() == counts.max())
    return {
        "entities": int(n_entities),
        "periods": int(n_periods),
        "min_T": int(counts.min()),
        "max_T": int(counts.max()),
        "balanced": bool(balanced)
    }

def hausman(fe_params: pd.Series, fe_cov: pd.DataFrame,
            re_params: pd.Series, re_cov: pd.DataFrame) -> tuple[float, float, int]:
    """
    Hausman test: H = (b_FE - b_RE)' [Var_FE - Var_RE]^{-1} (b_FE - b_RE)
    Usa solo regressori comuni (esclude FE assorbiti).
    """
    common = fe_params.index.intersection(re_params.index)
    if len(common) == 0:
        return (np.nan, np.nan, 0)
    bfe = fe_params.loc[common].values
    bre = re_params.loc[common].values
    Vfe = fe_cov.loc[common, common].values
    Vre = re_cov.loc[common, common].values
    V = Vfe - Vre
    try:
        Vinv = np.linalg.pinv(V)
        diff = bfe - bre
        H = float(diff.T @ Vinv @ diff)
        df = int(len(common))
        if _has_scipy:
            p = float(sps.chi2.sf(H, df))
        else:
            p = np.nan
        return (H, p, df)
    except Exception:
        return (np.nan, np.nan, len(common))

def residual_plots(fitted, resid, title_prefix=""):
    if go is None:
        return None
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=fitted, y=resid, mode="markers", name="Residui",
                              marker=dict(size=6, opacity=0.7)))
    fig1.add_hline(y=0, line=dict(color="black", width=1))
    fig1.update_layout(template="simple_white", height=360,
                       title=f"{title_prefix} Residui vs Fitted",
                       xaxis_title="Fitted", yaxis_title="Residuo")
    fig2 = go.Figure()
    fig2.add_trace(go.Histogram(x=resid, nbinsx=40, name="Residui"))
    fig2.update_layout(template="simple_white", height=360,
                       title=f"{title_prefix} Distribuzione residui",
                       xaxis_title="Residuo", yaxis_title="Frequenza")
    return fig1, fig2

def lm_covargs_from_se(se_type: str) -> tuple[str, dict]:
    cov_type = "unadjusted"; cov_kw = {}
    if se_type == "HC1 (etero-robusta)":
        cov_type = "robust"
    elif se_type == "Cluster per entità":
        cov_type = "clustered"; cov_kw = {"cluster_entity": True}
    elif se_type == "Cluster per tempo":
        cov_type = "clustered"; cov_kw = {"cluster_time": True}
    elif se_type == "Cluster entità & tempo":
        cov_type = "clustered"; cov_kw = {"cluster_entity": True, "cluster_time": True}
    elif se_type == "Driscoll–Kraay (se disponibile)":
        cov_type = "kernel"; cov_kw = {"kernel": "bartlett", "bandwidth": 3}
    return cov_type, cov_kw

def sm_covargs_from_se(se_type: str, data: pd.DataFrame, entity_col: str, time_col: str):
    # Statsmodels: single clustering only
    if se_type == "HC1 (etero-robusta)":
        return {"cov_type": "HC1", "cov_kwds": None}
    elif se_type == "Cluster per entità":
        return {"cov_type": "cluster", "cov_kwds": {"groups": data[entity_col]}}
    elif se_type == "Cluster per tempo":
        return {"cov_type": "cluster", "cov_kwds": {"groups": data[time_col]}}
    else:
        return {"cov_type": "nonrobust", "cov_kwds": None}

def se_explanation_block():
    st.markdown(
        """
**Come scegliere gli errori standard (SE)**  
- **Classici**: assumono omoschedasticità e indipendenza ⇒ *da evitare* nei panel reali.  
- **HC1 (etero-robusta)**: corregge l’eteroschedasticità ma **non** l’autocorrelazione entro entità/tempo.  
- **Cluster per entità**: consigliato quando c’è correlazione seriale **entro entità** (caso tipico nei panel).  
- **Cluster per tempo**: utile quando shock comuni colpiscono **tutte** le entità in un periodo (correlazione cross-section nello stesso t).  
- **Cluster entità & tempo** *(solo `linearmodels`)*: robusto **simultaneamente** a entrambe le dipendenze (scelta predefinita prudente se disponibile).  
- **Driscoll–Kraay** *(solo `linearmodels`)*: robusto a eteroschedasticità, autocorrelazione e **dipendenza cross-section**; raccomandato con **T medio-alto** (indicativamente T≥20).
"""
    )
    st.caption(
        "Regola pratica: se ha dubbi, usi **Cluster per entità**; se sospetta anche shock comuni nel tempo e ha `linearmodels`, usi **Cluster entità & tempo**; "
        "se **T** è ampio e c’è dipendenza cross-section, consideri **Driscoll–Kraay**."
    )

# ──────────────────────────────────────────────────────────────────────────────
# Header e dati
# ──────────────────────────────────────────────────────────────────────────────
st.title("🏛️ Panel Analysis (Econometria)")
st.caption("Pooled OLS, Effetti Fissi (entità/tempo), Effetti Casuali, Interazioni, Difference-in-Differences, "
           "SE robuste (HC1, Cluster 1-via/2-vie, Driscoll–Kraay) e Hausman. Interfaccia guidata.")

ensure_initialized()
DF = get_active(required=True)
with st.expander("Stato dati", expanded=False):
    stamp_meta()
if DF is None or DF.empty:
    st.stop()

# ──────────────────────────────────────────────────────────────────────────────
# 📌 Come impostare correttamente l’analisi — (SEZIONE IN ALTO)
# ──────────────────────────────────────────────────────────────────────────────
with st.expander("📌 Come impostare correttamente l’analisi (prima di iniziare)", expanded=True):
    st.markdown(
        "- **Identificatori**: verifichi che esistano colonne per **entità (id)** e **tempo (t)**; ogni riga deve essere una coppia (id, t).  \n"
        "- **Bilanciamento**: un panel *non* deve per forza essere bilanciato; FE/RE funzionano anche con T variabile tra entità.  \n"
        "- **Scelta del modello**:  \n"
        "  • **FE** se gli effetti non osservati sono plausibilmente **correlati** con X;  \n"
        "  • **RE** se tale correlazione è plausibilmente **nulla** (verificare con **Hausman**);  \n"
        "  • **DiD** se ha un trattamento applicato in un periodo a un sottoinsieme di entità e vuole **ATT**; includa FE entità+tempo.  \n"
        "- **Errori standard**: scelga SE **cluster per entità** come default prudente; consideri **due-vie** o **Driscoll–Kraay** in presenza di shock comuni e/o dipendenza cross-section.  \n"
        "- **Controlli**: includa covariate esogene e pertinenti; eviti controlli *post-trattamento* in DiD.  \n"
        "- **Interazioni**: utili per effetti condizionali (es. X che varia con Z); centri le variabili numeriche per ridurre collinearità.  \n"
        "- **Scaling**: opzionale (z-score) per confrontare coefficienti su scale diverse e migliorare stabilità numerica."
    )

# Dati e colonne
all_cols = list(DF.columns)

# ──────────────────────────────────────────────────────────────────────────────
# STEP 1 — Struttura del panel e variabili
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("### 1) Struttura del panel e variabili")

c1, c2, c3 = st.columns([1.1, 1.1, 1.2])
with c1:
    entity_col = st.selectbox("Colonna **Entità (id)**", options=all_cols, key=k("entity"))
with c2:
    time_col = st.selectbox("Colonna **Tempo (t)**", options=[c for c in all_cols if c != entity_col], key=k("time"))
with c3:
    y_col = st.selectbox("Variabile **dipendente (y)**", options=[c for c in all_cols if c not in {entity_col, time_col}], key=k("y"))

x1, x2 = st.columns([1.5, 1.5])
with x1:
    x_main = st.multiselect("Regressori principali (X)", options=[c for c in all_cols if c not in {entity_col, time_col, y_col}], key=k("X"))
with x2:
    x_ctrl = st.multiselect("**Controlli** aggiuntivi", options=[c for c in all_cols if c not in {entity_col, time_col, y_col} and c not in x_main], key=k("C"))

pre1, pre2, pre3 = st.columns([1.1, 1.1, 1.2])
with pre1:
    std_num = st.checkbox("Standardizza variabili numeriche (z-score)", value=False, key=k("z"))
with pre2:
    make_dummies = st.checkbox("Dummies per categoriche (drop first)", value=True, key=k("dummies"))
with pre3:
    drop_na = st.checkbox("Rimuovi righe con NA in y/X", value=True, key=k("dropna"))

# Prepara dataset di lavoro base
work_cols = [entity_col, time_col, y_col] + x_main + x_ctrl
work = DF[work_cols].copy()
if make_dummies:
    cat_for_dummies = [c for c in work.columns if c not in {entity_col, time_col, y_col} and not pd.api.types.is_numeric_dtype(work[c])]
    if cat_for_dummies:
        work = pd.get_dummies(work, columns=cat_for_dummies, drop_first=True)
if std_num:
    num_for_std = [c for c in work.columns if c not in {entity_col, time_col, y_col} and pd.api.types.is_numeric_dtype(work[c])]
    work = standardize_numeric(work, num_for_std)
if drop_na:
    work = work.dropna(subset=[y_col] + [c for c in work.columns if c not in {entity_col, time_col}])

# ──────────────────────────────────────────────────────────────────────────────
# STEP 2 — Interazioni tra regressori (opzionale)
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("### 2) Interazioni (opzionale)")
st.session_state.setdefault(k("interactions"), [])
regressors_current = [c for c in work.columns if c not in {entity_col, time_col, y_col}]

ci1, ci2, ci3 = st.columns([1.2, 1.2, 1.0])
with ci1:
    ia = st.selectbox("Variabile A", options=(regressors_current or ["—"]), key=k("ia"))
with ci2:
    ib = st.selectbox("Variabile B", options=([c for c in regressors_current if c != ia] or ["—"]), key=k("ib"))
with ci3:
    center_int = st.checkbox("Centra variabili numeriche (− media)", value=True, key=k("center"))
add_int = st.button("➕ Aggiungi interazione A×B", key=k("add_int"))
if add_int and ia in work.columns and ib in work.columns:
    a = work[ia]
    b = work[ib]
    if center_int:
        if pd.api.types.is_numeric_dtype(a): a = a - a.mean()
        if pd.api.types.is_numeric_dtype(b): b = b - b.mean()
    new_name = f"{ia}_x_{ib}"
    work[new_name] = pd.to_numeric(a, errors="coerce") * pd.to_numeric(b, errors="coerce")
    st.session_state[k("interactions")] = sorted(set(st.session_state[k("interactions")] + [new_name]))
    st.success(f"Interazione creata: **{new_name}**")

if st.session_state[k("interactions")]:
    st.caption("Interazioni attive: " + ", ".join(st.session_state[k("interactions")]))

# Aggiorna regressori dopo interazioni
regressors = [c for c in work.columns if c not in {entity_col, time_col, y_col}]

# Info panel
info = balance_summary(DF[[entity_col, time_col]].dropna(), entity_col, time_col)
i1, i2, i3, i4, i5 = st.columns(5)
i1.metric("Entità (N)", info["entities"])
i2.metric("Periodi (T)", info["periods"])
i3.metric("Min T per entità", info["min_T"])
i4.metric("Max T per entità", info["max_T"])
i5.metric("Balanced", "Sì" if info["balanced"] else "No")

st.markdown("#### Anteprima dati")
st.dataframe(work.head(10), use_container_width=True)

# ──────────────────────────────────────────────────────────────────────────────
# STEP 3 — Modello standard (Pooled / FE / RE)
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("### 3) Modello standard (Pooled / FE / RE)")

m1, m2, m3 = st.columns([1.2, 1.2, 1.2])
with m1:
    model_type = st.radio("Tipo di modello", ["Pooled OLS", "Effetti Fissi (FE)", "Effetti Casuali (RE)"], horizontal=False, key=k("model"))
with m2:
    fe_entity = st.checkbox("Effetti fissi **entità**", value=(model_type == "Effetti Fissi (FE)"), key=k("fe_e"))
    fe_time = st.checkbox("Effetti fissi **tempo**", value=False, key=k("fe_t"))
with m3:
    se_type = st.selectbox("Errori standard", options=[
        "Classici",
        "HC1 (etero-robusta)",
        "Cluster per entità",
        "Cluster per tempo",
        "Cluster entità & tempo",
        "Driscoll–Kraay (se disponibile)"
    ], index=2, key=k("se"))

# Guida sintetica dinamica sugli SE (in base alla scelta)
with st.expander("ℹ️ Come scegliere gli errori standard — guida rapida", expanded=True):
    se_explanation_block()
    # Suggerimento contestuale
    hint = ""
    if se_type == "Classici":
        hint = "Scelta fragile nei panel: preferisca **Cluster per entità** almeno."
    elif se_type == "HC1 (etero-robusta)":
        hint = "Corregge l’eteroschedasticità ma **non** l’autocorrelazione: meglio **Cluster per entità** se sospetta serial correlation."
    elif se_type == "Cluster per entità":
        hint = "Scelta **consigliata** nella maggior parte dei panel con autocorrelazione intra-entità."
    elif se_type == "Cluster per tempo":
        hint = "Utile con **shock comuni** per periodo; valuti due-vie se possibile."
    elif se_type == "Cluster entità & tempo":
        hint = "Robusto a dipendenze **intra-entità e intra-tempo** (richiede `linearmodels`)."
    elif se_type == "Driscoll–Kraay (se disponibile)":
        hint = "Robusto a dipendenza cross-section; indicato con **T medio-alto (≈≥20)**."
    st.info(f"Scelto: **{se_type}** → {hint}")

# Avvertenze quando la scelta non è supportata dal backend usato
if not _has_lm and se_type in {"Cluster entità & tempo", "Driscoll–Kraay (se disponibile)"}:
    st.warning("L’opzione selezionata richiede **linearmodels**. Verrà usata un’alternativa compatibile (statsmodels non supporta due-vie né Driscoll–Kraay).")

if len(regressors) == 0:
    st.warning("Selezionare almeno un regressore/controllo (o creare un'interazione).")
    st.stop()

# Costruzione panel per linearmodels (se disponibile)
panel = None
if _has_lm:
    try:
        panel = make_panel_index(work[[entity_col, time_col, y_col] + regressors], entity_col, time_col)
    except Exception as e:
        st.error(f"Errore nella costruzione dell'indice panel: {e}")
        _has_lm = False

# ──────────────────────────────────────────────────────────────────────────────
# Stima modelli standard
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("#### Stima e risultati (modello standard)")
results = {}

def show_lm_table(res, title: str):
    st.markdown(f"**{title}**")
    # Tabella coefficienti
    try:
        df_out = pd.DataFrame({
            "coef": res.params,
            "se": res.std_errors if hasattr(res, "std_errors") else res.bse,
            "t": res.tstats if hasattr(res, "tstats") else res.tvalues,
            "p": res.pvalues
        })
        st.dataframe(df_out.round(4), use_container_width=True)
    except Exception:
        try:
            df_out = pd.DataFrame({"coef": res.params, "se": res.bse, "t": res.tvalues, "p": res.pvalues})
            st.dataframe(df_out.round(4), use_container_width=True)
        except Exception:
            st.text(str(res.summary()))

    # metriche
    m1, m2, m3, m4 = st.columns(4)
    try:
        r2 = float(res.rsquared) if hasattr(res, "rsquared") else (float(res.rsquared_overall) if hasattr(res, "rsquared_overall") else np.nan)
    except Exception:
        r2 = np.nan
    try:
        r2_within = float(res.rsquared_within) if hasattr(res, "rsquared_within") else np.nan
    except Exception:
        r2_within = np.nan
    try:
        nobs = int(res.nobs) if hasattr(res, "nobs") else int(res.n)
    except Exception:
        nobs = np.nan
    try:
        aic = float(res.aic) if hasattr(res, "aic") else np.nan
    except Exception:
        aic = np.nan
    with m1: st.metric("R² (overall)", f"{r2:.3f}" if r2 == r2 else "—")
    with m2: st.metric("R² (within)", f"{r2_within:.3f}" if r2_within == r2_within else "—")
    with m3: st.metric("N osservazioni", f"{nobs}")
    with m4: st.metric("AIC", f"{aic:.1f}" if aic == aic else "—")

try:
    if model_type == "Pooled OLS":
        if _has_lm:
            Y = panel[y_col]
            X = panel[regressors]
            X = sm.add_constant(X) if _has_sm else X.assign(const=1.0)
            cov_type, cov_kw = lm_covargs_from_se(se_type)
            mod = PooledOLS(Y, X)
            res = mod.fit(cov_type=cov_type, **cov_kw)
            results["Pooled OLS"] = res
        elif _has_sm:
            formula = y_col + " ~ " + " + ".join(regressors)
            data = work.copy()
            kw = sm_covargs_from_se(se_type, data, entity_col, time_col)
            model = smf.ols(formula, data=data).fit(cov_type=kw["cov_type"], cov_kwds=kw["cov_kwds"])
            results["Pooled OLS"] = model
        else:
            st.error("Né linearmodels né statsmodels disponibili per Pooled OLS.")

    elif model_type == "Effetti Fissi (FE)":
        if _has_lm:
            Y = panel[y_col]; X = panel[regressors]
            cov_type, cov_kw = lm_covargs_from_se(se_type)
            mod_fe = PanelOLS(Y, X, entity_effects=fe_entity, time_effects=fe_time)
            res_fe = mod_fe.fit(cov_type=cov_type, **cov_kw)
            results["FE"] = res_fe
        elif _has_sm:
            pieces = [" + ".join(regressors)] if regressors else []
            if fe_entity: pieces.append(f"C({entity_col})")
            if fe_time: pieces.append(f"C({time_col})")
            rhs = " + ".join([p for p in pieces if p]) if pieces else "1"
            formula = f"{y_col} ~ {rhs}"
            data = work.copy()
            kw = sm_covargs_from_se(se_type, data, entity_col, time_col)
            model = smf.ols(formula, data=data).fit(cov_type=kw["cov_type"], cov_kwds=kw["cov_kwds"])
            results["FE"] = model
        else:
            st.error("Né linearmodels né statsmodels disponibili per FE.")

    else:  # Effetti Casuali (RE)
        if _has_lm:
            Y = panel[y_col]; X = panel[regressors]
            cov_type, cov_kw = lm_covargs_from_se(se_type)
            mod_re = RandomEffects(Y, X)
            res_re = mod_re.fit(cov_type=cov_type, **cov_kw)
            results["RE"] = res_re
        elif _has_sm:
            st.info("Effetti Casuali richiedono `linearmodels`. Con statsmodels consideri FE.")
        else:
            st.error("Né linearmodels né statsmodels disponibili per RE.")
except Exception as e:
    st.error(f"Errore in stima modello: {e}")

for name, res in results.items():
    show_lm_table(res, name)

# Interpretazione risultati standard
with st.expander("📝 Interpretazione (modello standard)"):
    st.markdown(
        "- **Coefficienti**: effetto marginale medio di ciascun regressore su **y**, a parità degli altri.  \n"
        "- **p-value**: < α ⇒ evidenza di effetto diverso da 0.  \n"
        "- **FE**: interpretazione **within** (variazione intra-entità nel tempo).  \n"
        "- **RE**: più efficiente se l’effetto casuale è **non correlato** con X; altrimenti preferire **FE**.  \n"
        "- **R² within**: capacità esplicativa sulle variazioni **entro entità** (rilevante per FE).  \n"
        "- **SE cluster**: correggono per eteroschedasticità/autocorrelazione; due-vie per robustezza a shock comuni nel tempo."
    )

# ──────────────────────────────────────────────────────────────────────────────
# STEP 4 — Hausman test (FE vs RE)
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("### 4) Hausman test (FE vs RE)")
if _has_lm:
    try:
        Y = panel[y_col]; X = panel[regressors]
        fe_mod = PanelOLS(Y, X, entity_effects=True, time_effects=False).fit(cov_type="robust")
        re_mod = RandomEffects(Y, X).fit(cov_type="robust")
        H, pH, dfH = hausman(fe_mod.params, fe_mod.cov, re_mod.params, re_mod.cov)
        c1, c2, c3 = st.columns(3)
        c1.metric("Hausman χ²", f"{H:.3f}" if H == H else "—")
        c2.metric("df", f"{dfH}")
        c3.metric("p-value", fmt_p(pH))
        st.caption("p **piccolo** ⇒ preferire **FE** (ipotesi RE non valida). p **grande** ⇒ **RE** plausibile.")
    except Exception as e:
        st.info(f"Hausman non calcolabile: {e}")
else:
    st.info("Hausman richiede **linearmodels**. Installarlo per abilitare il test.")

# ──────────────────────────────────────────────────────────────────────────────
# STEP 5 — Difference-in-Differences (DiD)
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("### 5) Difference-in-Differences (DiD)")
did_enable = st.checkbox("Abilita analisi DiD", value=False, key=k("did_on"))

if did_enable:
    all_cols = list(DF.columns)
    dc1, dc2 = st.columns([1.2, 1.8])
    with dc1:
        treat_col = st.selectbox("Colonna **Treatment** (entità: 0/1 o categoria)", options=[c for c in all_cols if c not in {entity_col, time_col, y_col}], key=k("treat"))
        treat_is_binary = st.checkbox("Treatment già binario 0/1", value=True, key=k("treat_bin"))
        treat_level = None
        if not treat_is_binary:
            lvl = sorted(DF[treat_col].dropna().astype(str).unique().tolist())
            treat_level = st.selectbox("Valore che indica **trattato**", options=lvl, key=k("treat_lvl"))
    with dc2:
        post_mode = st.radio("Come definire **Post**", ["Colonna esistente (0/1)", "Da soglia sul tempo"], horizontal=True, key=k("post_mode"))
        if post_mode == "Colonna esistente (0/1)":
            post_col = st.selectbox("Colonna **Post** (0/1)", options=[c for c in all_cols if c not in {entity_col, time_col, y_col}], key=k("post_col"))
            post_cut = None
        else:
            post_col = None
            tvals = DF[time_col]
            if pd.api.types.is_numeric_dtype(tvals):
                tmin, tmax = float(np.nanmin(tvals)), float(np.nanmax(tvals))
                post_cut = st.slider("Soglia tempo per **Post** (t ≥ soglia ⇒ Post=1)", min_value=float(tmin), max_value=float(tmax), value=float(np.nanmedian(tvals)))
            elif pd.api.types.is_datetime64_any_dtype(tvals):
                tmin, tmax = pd.to_datetime(tvals.min()), pd.to_datetime(tvals.max())
                post_cut = st.date_input("Data soglia per **Post** (t ≥ soglia ⇒ Post=1)", value=(tmin if pd.isna(tmin) else tmin))
            else:
                levels = sorted(DF[time_col].dropna().astype(str).unique().tolist())
                post_cut = st.selectbox("Livello soglia per **Post** (≥ livello ⇒ Post=1)", options=levels, index=max(0, len(levels)//2))

    # Costruzione dataset DiD
    did_df = DF[[entity_col, time_col, y_col] + regressors].copy()
    if treat_is_binary:
        treat_series = DF.groupby(entity_col)[treat_col].transform(lambda s: int(pd.to_numeric(s, errors="coerce").fillna(0).iloc[0]))
    else:
        treat_series = (DF[treat_col].astype(str) == str(treat_level)).astype(int)
    did_df["TREAT"] = treat_series.astype(int)

    if post_mode == "Colonna esistente (0/1)":
        did_df["POST"] = pd.to_numeric(DF[post_col], errors="coerce").fillna(0).astype(int)
    else:
        if pd.api.types.is_numeric_dtype(DF[time_col]):
            did_df["POST"] = (DF[time_col].astype(float) >= float(post_cut)).astype(int)
        elif pd.api.types.is_datetime64_any_dtype(DF[time_col]):
            did_df["POST"] = (pd.to_datetime(DF[time_col]) >= pd.to_datetime(post_cut)).astype(int)
        else:
            levels = sorted(DF[time_col].dropna().astype(str).unique().tolist())
            order = {lev: i for i, lev in enumerate(levels)}
            thresh = order.get(str(post_cut), 0)
            did_df["POST"] = did_df[time_col].astype(str).map(order).fillna(-1).astype(int).rsub(thresh).le(0).astype(int)

    did_df["DID"] = did_df["TREAT"] * did_df["POST"]

    # Stima DiD: FE entità + tempo
    st.markdown("#### Stima DiD (FE entità+tempo)")
    try:
        if _has_lm:
            pnl = make_panel_index(did_df, entity_col, time_col)
            Y = pnl[y_col]
            X = pnl.drop(columns=[y_col])
            cov_type, cov_kw = lm_covargs_from_se(se_type)
            mod_did = PanelOLS(Y, X, entity_effects=True, time_effects=True)
            res_did = mod_did.fit(cov_type=cov_type, **cov_kw)
            st.dataframe(pd.DataFrame({
                "coef": res_did.params,
                "se": res_did.std_errors,
                "t": res_did.tstats,
                "p": res_did.pvalues
            }).round(4), use_container_width=True)
            did_coef = float(res_did.params.get("DID", np.nan))
            did_se = float(res_did.std_errors.get("DID", np.nan))
            ci_low = did_coef - 1.96 * did_se if did_se == did_se else np.nan
            ci_high = did_coef + 1.96 * did_se if did_se == did_se else np.nan
            c1, c2, c3 = st.columns(3)
            c1.metric("DiD (TREAT×POST)", f"{did_coef:.4f}" if did_coef == did_coef else "—")
            c2.metric("IC 95%", f"[{ci_low:.4f}, {ci_high:.4f}]" if ci_low == ci_low and ci_high == ci_high else "—")
            c3.metric("p-value", fmt_p(float(res_did.pvalues.get('DID', np.nan))))
            did_res = res_did
        elif _has_sm:
            rhs = "DID + TREAT + POST" + (" + " + " + ".join([c for c in regressors if c not in {'TREAT','POST','DID'}]) if regressors else "")
            formula = f"{y_col} ~ {rhs} + C({entity_col}) + C({time_col})"
            model = smf.ols(formula, data=did_df).fit(cov_type="HC1")
            st.dataframe(pd.DataFrame({"coef": model.params, "se": model.bse, "t": model.tvalues, "p": model.pvalues}).round(4), use_container_width=True)
            did_coef = float(model.params.get("DID", np.nan))
            did_se = float(model.bse.get("DID", np.nan))
            ci_low = did_coef - 1.96 * did_se if did_se == did_se else np.nan
            ci_high = did_coef + 1.96 * did_se if did_se == did_se else np.nan
            c1, c2, c3 = st.columns(3)
            c1.metric("DiD (TREAT×POST)", f"{did_coef:.4f}" if did_coef == did_coef else "—")
            c2.metric("IC 95%", f"[{ci_low:.4f}, {ci_high:.4f}]" if ci_low == ci_low and ci_high == ci_high else "—")
            c3.metric("p-value", fmt_p(float(model.pvalues.get('DID', np.nan))))
            did_res = model
        else:
            st.error("Né linearmodels né statsmodels disponibili per la stima DiD.")
            did_res = None
    except Exception as e:
        st.error(f"Errore nella stima DiD: {e}")
        did_res = None

    with st.expander("📝 Come interpretare la DiD", expanded=True):
        st.markdown(
            "- Il coefficiente **DID = TREAT×POST** stima l’**ATT** sotto l’ipotesi di **trend paralleli**.  \n"
            "- **Segno**: positivo ⇒ aumento di **y** dovuto al trattamento; negativo ⇒ diminuzione.  \n"
            "- **IC 95% e p-value**: misurano precisione ed evidenza statistica.  \n"
            "- FE **entità** e **tempo** controllano eterogeneità invariante e shock comuni.  \n"
            "- Preferire **SE cluster per entità** o **due-vie** per robustezza."
        )

# ──────────────────────────────────────────────────────────────────────────────
# Diagnostica (residui)
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("### 6) Diagnostica grafica")
diag_opts = list(results.keys())
if did_enable and 'did_res' in locals() and did_res is not None:
    diag_opts = ["DiD"] + diag_opts
sel_model = st.selectbox("Seleziona il risultato da diagnosticare", options=diag_opts, key=k("diag"))
if sel_model and go is not None:
    try:
        res = did_res if sel_model == "DiD" else results[sel_model]
        fitted = res.fitted_values if hasattr(res, "fitted_values") else res.fittedvalues
        resid = res.resids if hasattr(res, "resids") else res.resid
        f1, f2 = residual_plots(np.asarray(fitted), np.asarray(resid), title_prefix=f"{sel_model} —")
        if f1 and f2:
            c1, c2 = st.columns(2)
            with c1: st.plotly_chart(f1, use_container_width=True)
            with c2: st.plotly_chart(f2, use_container_width=True)
            st.caption("Verificare assenza di pattern sistematici (linearità, omoschedasticità) e simmetria dei residui.")
    except Exception as e:
        st.info(f"Diagnostica non disponibile: {e}")

# ──────────────────────────────────────────────────────────────────────────────
# Checklist di impostazione (richiamo finale opzionale)
# ──────────────────────────────────────────────────────────────────────────────
with st.expander("Checklist rapida (richiamo)"):
    st.markdown(
        "- Definisca chiaramente **id** e **t**; una riga = una coppia (id,t).  \n"
        "- Scegliere tra **FE** (se correlazione con X) e **RE** (se no; confermare con **Hausman**).  \n"
        "- Usare **SE cluster per entità** come default; valutare **due-vie** o **Driscoll–Kraay** secondo le dipendenze.  \n"
        "- In **DiD**: FE entità+tempo, trend paralleli, nessun controllo post-trattamento."
    )

# ──────────────────────────────────────────────────────────────────────────────
# Navigazione
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("---")
nav1, nav2 = st.columns(2)
with nav1:
    if st.button("⬅️ Torna: Longitudinale — Misure ripetute", use_container_width=True, key=k("go_prev")):
        try:
            st.switch_page("pages/12_📈_Longitudinale_Misure_Ripetute.py")
        except Exception:
            pass
with nav2:
    if st.button("➡️ Vai: Report / Export", use_container_width=True, key=k("go_next")):
        for target in [
            "pages/14_🧾_Report_Automatico.py",
            "pages/14_📤_Export_Risultati.py",
        ]:
            try:
                st.switch_page(target)
                break
            except Exception:
                continue
