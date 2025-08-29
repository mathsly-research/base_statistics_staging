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
    from linearmodels.panel import compare
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
    # consente time numerico o stringa
    if pd.api.types.is_datetime64_any_dtype(d[time_col]):
        pass
    elif pd.api.types.is_numeric_dtype(d[time_col]):
        pass
    else:
        # stringa
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
    Usa solo i regressori comuni (esclude effetti fissi/dummies assorbite).
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

def residual_plots(y, fitted, resid, title_prefix=""):
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

# ──────────────────────────────────────────────────────────────────────────────
# Header e dati
# ──────────────────────────────────────────────────────────────────────────────
st.title("🏛️ Panel Analysis (Econometria)")
st.caption("Pooled OLS, Effetti Fissi (entità/tempo) ed Effetti Casuali con SE robuste, Hausman test e diagnostica. Interfaccia coerente e guidata.")

ensure_initialized()
DF = get_active(required=True)
with st.expander("Stato dati", expanded=False):
    stamp_meta()
if DF is None or DF.empty:
    st.stop()

all_cols = list(DF.columns)
num_cols = [c for c in all_cols if pd.api.types.is_numeric_dtype(DF[c])]
cat_cols = [c for c in all_cols if not pd.api.types.is_numeric_dtype(DF[c])]

# ──────────────────────────────────────────────────────────────────────────────
# STEP 1 — Definizione struttura panel e variabili
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

# Prepara dataset di lavoro
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
# STEP 2 — Specifica modello e opzioni
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("### 2) Specifica del modello")
m1, m2, m3 = st.columns([1.2, 1.2, 1.2])
with m1:
    model_type = st.radio("Tipo di modello", ["Pooled OLS", "Effetti Fissi (FE)", "Effetti Casuali (RE)"], horizontal=False, key=k("model"))
with m2:
    fe_entity = st.checkbox("Effetti fissi **entità**", value=True if model_type == "Effetti Fissi (FE)" else False, key=k("fe_e"))
    fe_time = st.checkbox("Effetti fissi **tempo**", value=False, key=k("fe_t"))
with m3:
    se_type = st.selectbox("Errori standard", options=[
        "Classici",
        "HC1 (etero-robusta)",
        "Cluster per entità",
        "Cluster per tempo",
        "Driscoll–Kraay (se disponibile)"
    ], index=1, key=k("se"))

st.caption(
    "- FE rimuove **eterogeneità non osservata** fissa per entità/tempo.  \n"
    "- RE assume effetti casuali **non correlati** con i regressori.  \n"
    "- **Hausman** aiuta a scegliere tra FE e RE.  \n"
    "- Scelga SE **cluster** per entità o tempo in presenza di autocorrelazione intra-cluster; **Driscoll–Kraay** per T moderato/alto e cross-corr."
)

# Costruzione matrici
if len(x_main) + len(x_ctrl) == 0:
    st.warning("Selezionare almeno un regressore o controllo.")
    st.stop()

regressors = [c for c in work.columns if c not in {entity_col, time_col, y_col}]
y = work[y_col]

# Indice panel per linearmodels
panel = None
if _has_lm:
    try:
        panel = make_panel_index(work[[entity_col, time_col, y_col] + regressors], entity_col, time_col)
    except Exception as e:
        st.error(f"Errore nella costruzione dell'indice panel: {e}")
        _has_lm = False

# ──────────────────────────────────────────────────────────────────────────────
# STEP 3 — Stima modelli
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("### 3) Stima e risultati")

results = {}
hausman_tuple = (np.nan, np.nan, 0)

try:
    if model_type == "Pooled OLS":
        if _has_lm:
            Y = panel[y_col]
            X = panel[regressors]
            X = sm.add_constant(X) if _has_sm else X.assign(const=1.0)
            cov_kw = {}
            cov_type = "unadjusted"
            if se_type == "HC1 (etero-robusta)":
                cov_type = "robust"
            elif se_type == "Cluster per entità":
                cov_type = "clustered"; cov_kw = {"cluster_entity": True}
            elif se_type == "Cluster per tempo":
                cov_type = "clustered"; cov_kw = {"cluster_time": True}
            elif se_type == "Driscoll–Kraay (se disponibile)":
                cov_type = "kernel"; cov_kw = {"kernel": "bartlett", "bandwidth": 3}
            mod = PooledOLS(Y, X)
            res = mod.fit(cov_type=cov_type, **cov_kw)
            results["Pooled OLS"] = res

        elif _has_sm:
            # statsmodels OLS con dummies opzionali (nessun FE assorbito qui)
            formula = y_col + " ~ " + " + ".join(regressors)
            data = work.copy()
            model = smf.ols(formula, data=data).fit(
                cov_type="HC1" if se_type == "HC1 (etero-robusta)" else "nonrobust"
            )
            res = model
            results["Pooled OLS"] = res
        else:
            st.error("Né linearmodels né statsmodels disponibili per Pooled OLS.")

    elif model_type == "Effetti Fissi (FE)":
        if _has_lm:
            Y = panel[y_col]
            X = panel[regressors]
            cov_kw = {}
            cov_type = "unadjusted"
            if se_type == "HC1 (etero-robusta)":
                cov_type = "robust"
            elif se_type == "Cluster per entità":
                cov_type = "clustered"; cov_kw = {"cluster_entity": True}
            elif se_type == "Cluster per tempo":
                cov_type = "clustered"; cov_kw = {"cluster_time": True}
            elif se_type == "Driscoll–Kraay (se disponibile)":
                cov_type = "kernel"; cov_kw = {"kernel": "bartlett", "bandwidth": 3}

            mod_fe = PanelOLS(Y, X, entity_effects=fe_entity, time_effects=fe_time)
            res_fe = mod_fe.fit(cov_type=cov_type, **cov_kw)
            results["FE"] = res_fe

        elif _has_sm:
            # FE via dummies: C(entity) e/o C(time)
            pieces = [" + ".join(regressors)] if regressors else []
            if fe_entity:
                pieces.append(f"C({entity_col})")
            if fe_time:
                pieces.append(f"C({time_col})")
            rhs = " + ".join(pieces) if pieces else "1"
            formula = f"{y_col} ~ {rhs}"
            data = work.copy()
            model = smf.ols(formula, data=data).fit(
                cov_type=("cluster" if se_type.startswith("Cluster") else ("HC1" if se_type == "HC1 (etero-robusta)" else "nonrobust")),
                cov_kwds=({"groups": data[entity_col]} if se_type == "Cluster per entità" else ({"groups": data[time_col]} if se_type == "Cluster per tempo" else None))
            )
            results["FE"] = model
        else:
            st.error("Né linearmodels né statsmodels disponibili per FE.")

    else:  # Effetti Casuali (RE)
        if _has_lm:
            Y = panel[y_col]
            X = panel[regressors]
            cov_kw = {}
            cov_type = "unadjusted"
            if se_type == "HC1 (etero-robusta)":
                cov_type = "robust"
            elif se_type == "Cluster per entità":
                cov_type = "clustered"; cov_kw = {"cluster_entity": True}
            elif se_type == "Cluster per tempo":
                cov_type = "clustered"; cov_kw = {"cluster_time": True}
            elif se_type == "Driscoll–Kraay (se disponibile)":
                cov_type = "kernel"; cov_kw = {"kernel": "bartlett", "bandwidth": 3}

            mod_re = RandomEffects(Y, X)
            res_re = mod_re.fit(cov_type=cov_type, **cov_kw)
            results["RE"] = res_re

        elif _has_sm:
            st.info("Effetti Casuali non supportati nativamente in statsmodels OLS. Usi linearmodels per RE o consideri FE.")
        else:
            st.error("Né linearmodels né statsmodels disponibili per RE.")

except Exception as e:
    st.error(f"Errore in stima modello: {e}")

# ──────────────────────────────────────────────────────────────────────────────
# Output tabelle + indice di qualità
# ──────────────────────────────────────────────────────────────────────────────
def show_lm_table(res, title: str):
    st.markdown(f"**{title}**")
    try:
        summ = res.summary.tables[1]  # linearmodels ha tabelle già formattate; ma preferiamo dataframe
    except Exception:
        summ = None
    try:
        # linearmodels: params, std_errors, tstats, pvalues
        df_out = pd.DataFrame({
            "coef": res.params,
            "se": res.std_errors if hasattr(res, "std_errors") else res.bse,
            "t": res.tstats if hasattr(res, "tstats") else res.tvalues,
            "p": res.pvalues
        })
        st.dataframe(df_out.round(4), use_container_width=True)
    except Exception:
        try:
            # statsmodels RegressionResults
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

# Visualizza risultati
for name, res in results.items():
    show_lm_table(res, name)

# ──────────────────────────────────────────────────────────────────────────────
# Hausman test (FE vs RE) — solo se disponibili entrambi in linearmodels
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("### 4) Hausman test (FE vs RE)")
hausman_msg = ""
if _has_lm:
    try:
        # Stimo FE (entity FE) e RE sulla stessa specifica per il test
        Y = panel[y_col]; X = panel[regressors]
        fe_mod = PanelOLS(Y, X, entity_effects=True, time_effects=False).fit(cov_type="robust")
        re_mod = RandomEffects(Y, X).fit(cov_type="robust")
        H, pH, dfH = hausman(fe_mod.params, fe_mod.cov, re_mod.params, re_mod.cov)
        st.metric("Hausman χ²", f"{H:.3f}" if H == H else "—")
        st.metric("df", f"{dfH}")
        st.metric("p-value", fmt_p(pH))
        hausman_msg = ("Preferire **FE** (p piccolo ⇒ RE non valido)" if (pH == pH and pH < 0.05) else
                       "Preferire **RE** (p grande ⇒ ipotesi RE plausibile)")
        st.caption(hausman_msg)
    except Exception as e:
        st.info(f"Hausman non calcolabile: {e}")
else:
    st.info("Hausman richiede **linearmodels**. Installarlo per abilitare il test.")

# ──────────────────────────────────────────────────────────────────────────────
# Diagnostica (residui)
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("### 5) Diagnostica grafica")
sel_model = st.selectbox("Seleziona il risultato da diagnosticare", options=list(results.keys()) if results else [], key=k("diag"))
if sel_model and go is not None:
    res = results[sel_model]
    try:
        fitted = res.fitted_values if hasattr(res, "fitted_values") else res.fittedvalues
        resid = res.resids if hasattr(res, "resids") else res.resid
        f1, f2 = residual_plots(None, np.asarray(fitted), np.asarray(resid), title_prefix=f"{sel_model} —")
        if f1 and f2:
            c1, c2 = st.columns(2)
            with c1: st.plotly_chart(f1, use_container_width=True)
            with c2: st.plotly_chart(f2, use_container_width=True)
            st.caption("Controllare linearità (residui ~ 0 senza pattern) e simmetria/assenza di code estreme nella distribuzione.")
    except Exception as e:
        st.info(f"Diagnostica non disponibile: {e}")

# ──────────────────────────────────────────────────────────────────────────────
# Come leggere i risultati
# ──────────────────────────────────────────────────────────────────────────────
with st.expander("ℹ️ Come leggere modelli panel e test"):
    st.markdown(
        "- **Pooled OLS**: ignora la struttura panel; può essere distorto se esiste eterogeneità fissa tra entità.  \n"
        "- **Effetti Fissi (FE)**: controlla per **eterogeneità invariante** (per entità/tempo). Coefficienti si interpretano come effetti **within**.  \n"
        "- **Effetti Casuali (RE)**: più efficienti se gli effetti casuali sono **non correlati** con X.  \n"
        "- **Hausman**: p-value **piccolo** ⇒ preferire **FE** (l’ipotesi di RE è violata); p **grande** ⇒ RE plausibile.  \n"
        "- **SE robuste**: usare **HC1** con eteroschedasticità; **cluster** per correlazione intra-entità/tempo; **Driscoll–Kraay** (se disponibile) per dipendenze cross-sezionali con T moderato/alto.  \n"
        "- **R² within** misura capacità esplicativa **entro entità** (rilevante per FE); **R² overall** è complessivo."
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

