# -*- coding: utf-8 -*-
# pages/8_🧮_Regression.py
from __future__ import annotations

import math
import re
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

# Statistiche / Modelli (opzionale)
try:
    from scipy import stats
except Exception:
    stats = None

try:
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    from statsmodels.stats.outliers_influence import variance_inflation_factor
except Exception:
    sm = None
    smf = None
    variance_inflation_factor = None

# ──────────────────────────────────────────────────────────────────────────────
# Data store centralizzato (+ fallback)
# ──────────────────────────────────────────────────────────────────────────────
try:
    from data_store import ensure_initialized, get_active, stamp_meta
except Exception:
    def ensure_initialized() -> None:
        st.session_state.setdefault("ds_active_df", None)
        st.session_state.setdefault("ds_meta", {"version": 0, "updated_at": None, "source": None, "note": ""})

    def get_active(required: bool = True) -> pd.DataFrame | None:
        ensure_initialized()
        _df = st.session_state.get("ds_active_df")
        if required and (_df is None or _df.empty):
            st.error("Nessun dataset attivo. Importi i dati e completi la pulizia.")
            st.stop()
        return _df

    def stamp_meta() -> None:
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
st.set_page_config(page_title="Regression", layout="wide")
try:
    from nav import sidebar
    sidebar()
except Exception:
    pass

KEY = "reg"
def k(name: str) -> str:
    return f"{KEY}_{name}"

# ──────────────────────────────────────────────────────────────────────────────
# Helper generali: formule, label, tabelle
# ──────────────────────────────────────────────────────────────────────────────
def fq(s: str) -> str:
    """Escape sicuro per Patsy (nomi con spazi/simboli/emoji)."""
    return s.replace("\\", "\\\\").replace("'", "\\'")

_latex_esc = {"_": r"\_", "%": r"\%", "&": r"\&", "$": r"\$", "#": r"\#", "{": r"\{", "}": r"\}",
              "~": r"\textasciitilde{}", "^": r"\^{}", "\\": r"\textbackslash{}"}
def latex_escape(text: str) -> str:
    return "".join(_latex_esc.get(ch, ch) for ch in text)

def clean_param_name(name: str) -> str:
    if name == "Intercept": return "Intercetta"
    m = re.match(r"""C\(Q\('(.+?)'\)\)\[T\.(.+)\]""", name)
    if m: return f"{m.group(1)} = {m.group(2)}"
    m2 = re.match(r"""Q\('(.+?)'\)""", name)
    if m2: return m2.group(1)
    if ":" in name: return " × ".join(clean_param_name(p) for p in name.split(":"))
    return name

def build_formula(y: str, X: list[str], df_: pd.DataFrame) -> str:
    terms: list[str] = []
    for v in X:
        if pd.api.types.is_numeric_dtype(df_[v]):
            terms.append(f"Q('{fq(v)}')")
        else:
            terms.append(f"C(Q('{fq(v)}'))")
    rhs = " + ".join(terms) if terms else "1"
    return f"Q('{fq(y)}') ~ {rhs}"

def pretty_formula_latex_linear(y: str, formula_patsy: str, data: pd.DataFrame) -> str:
    try:
        import patsy
        _, X = patsy.dmatrices(formula_patsy, data=data, return_type="dataframe")
        cols = [c for c in X.columns if c != "Intercept"]
        terms = " + ".join([rf"\beta_{{{i+1}}}\cdot {latex_escape(clean_param_name(c))}" for i, c in enumerate(cols)])
        return rf"\mathbb{{E}}\!\left[{latex_escape(y)}\right] = \beta_0" + (f" + {terms}" if terms else "")
    except Exception:
        return rf"\mathbb{{E}}[{latex_escape(y)}] = \beta_0 + \sum_j \beta_j x_j"

def pretty_formula_latex_logit(y: str, formula_patsy: str, data: pd.DataFrame, success_label: str | int) -> str:
    try:
        import patsy
        _, X = patsy.dmatrices(formula_patsy.replace("_y", "dummyY"),
                               data=data.assign(dummyY=np.random.randint(0, 2, size=len(data))),
                               return_type="dataframe")
        cols = [c for c in X.columns if c != "Intercept"]
        terms = " + ".join([rf"\beta_{{{i+1}}}\cdot {latex_escape(clean_param_name(c))}" for i, c in enumerate(cols)])
        return rf"\log\!\left(\frac{{\Pr({latex_escape(y)}={latex_escape(str(success_label))})}}{{1-\Pr({latex_escape(y)}={latex_escape(str(success_label))})}}\right) = \beta_0" + (f" + {terms}" if terms else "")
    except Exception:
        return rf"\log\!\left(\frac{{\Pr({latex_escape(y)}=1)}}{{1-\Pr({latex_escape(y)}=1)}}\right) = \beta_0 + \sum_j \beta_j x_j"

def coef_table_from_fit(fit, or_scale: bool = False) -> pd.DataFrame:
    params = fit.params.copy(); bse = fit.bse.copy(); pvals = fit.pvalues.copy()
    ci = fit.conf_int(); ci.columns = ["CI 2.5%", "CI 97.5%"]
    out = pd.concat([params, bse, pvals, ci], axis=1)
    out.columns = ["β", "SE", "p", "CI 2.5%", "CI 97.5%"]
    stat = getattr(fit, "tvalues", None) if hasattr(fit, "tvalues") else getattr(fit, "zvalues", None)
    if stat is not None: out.insert(2, "z/t", stat)
    out.index = [clean_param_name(str(i)) for i in out.index]
    def star(p):
        try:
            p = float(p)
            return "****" if p < 1e-4 else ("***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "")))
        except Exception:
            return ""
    out["Sig."] = [star(p) for p in out["p"]]
    cols = ["β", "SE"]; 
    if "z/t" in out.columns: cols += ["z/t"]
    cols += ["p", "Sig.", "CI 2.5%", "CI 97.5%"]
    out = out[cols]
    if or_scale:
        or_df = pd.DataFrame({
            "OR": np.exp(params),
            "OR CI 2.5%": np.exp(ci["CI 2.5%"]),
            "OR CI 97.5%": np.exp(ci["CI 97.5%"])
        }, index=out.index)
        out = pd.concat([out, or_df], axis=1)
    for c in out.columns:
        if c == "Sig.": continue
        out[c] = pd.to_numeric(out[c], errors="coerce").round(3)
    return out

def confusion_table_counts_percent(TN, FP, FN, TP) -> pd.DataFrame:
    row0 = TN + FP; row1 = FN + TP
    def cell(count, row_sum):
        perc = (count / row_sum * 100.0) if row_sum > 0 else np.nan
        return f"{int(count)} ({perc:.1f}%)" if perc == perc else f"{int(count)}"
    return pd.DataFrame({"Pred 0": [cell(TN, row0), cell(FN, row1)],
                         "Pred 1": [cell(FP, row0), cell(TP, row1)]},
                        index=["Vera 0", "Vera 1"])

# ── ROC helpers (AUC, curva step, smoothing morbido) ─────────────────────────
def auc_mann_whitney(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """AUC con U di Mann–Whitney = Pr(score_pos > score_neg)."""
    try:
        from scipy.stats import rankdata
        y_true = np.asarray(y_true).astype(int)
        y_score = np.asarray(y_score).astype(float)
        n1 = int((y_true == 1).sum()); n0 = int((y_true == 0).sum())
        if n1 == 0 or n0 == 0: return float("nan")
        ranks = rankdata(y_score)
        U = float(ranks[y_true == 1].sum()) - n1 * (n1 + 1) / 2.0
        return float(U / (n1 * n0))
    except Exception:
        return float("nan")

def roc_curve_strict(y_true: np.ndarray, y_score: np.ndarray):
    """ROC corretta con soglie ai valori distinti, tie gestiti; include (0,0) e (1,1)."""
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    P = int((y_true == 1).sum()); N = int((y_true == 0).sum())
    if P == 0 or N == 0: return np.array([0.0, 1.0]), np.array([0.0, 1.0])

    order = np.argsort(-y_score, kind="mergesort")
    y_sorted = y_true[order]; s_sorted = y_score[order]
    tpr = [0.0]; fpr = [0.0]; tp = 0; fp = 0; i = 0; n = len(y_sorted)
    while i < n:
        thr = s_sorted[i]
        tp_inc = fp_inc = 0
        while i < n and s_sorted[i] == thr:
            if y_sorted[i] == 1: tp_inc += 1
            else: fp_inc += 1
            i += 1
        tp += tp_inc; fp += fp_inc
        tpr.append(tp / P); fpr.append(fp / N)
    if tpr[-1] != 1.0 or fpr[-1] != 1.0:
        tpr.append(1.0); fpr.append(1.0)
    return np.array(fpr, dtype=float), np.array(tpr, dtype=float)

def _uniq_monotone(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rende FPR unico crescente e TPR non decrescente (max per FPR duplicati)."""
    df = pd.DataFrame({"x": x, "y": y}).groupby("x", as_index=False)["y"].max().sort_values("x")
    xx = df["x"].to_numpy()
    yy = np.maximum.accumulate(df["y"].to_numpy())
    return xx, yy

def smooth_roc(fpr: np.ndarray, tpr: np.ndarray, grid_points: int = 300) -> tuple[np.ndarray, np.ndarray]:
    """Smussa la ROC con interpolazione monotòna (PCHIP se disponibile)."""
    fpr = np.asarray(fpr, float); tpr = np.asarray(tpr, float)
    pts = np.vstack([[0.0, 0.0], np.vstack([fpr, tpr]).T, [1.0, 1.0]])
    x, y = _uniq_monotone(pts[:, 0], pts[:, 1])
    grid = np.linspace(0.0, 1.0, grid_points)
    try:
        from scipy.interpolate import PchipInterpolator
        y_s = PchipInterpolator(x, y)(grid)
    except Exception:
        y_s = np.interp(grid, x, y)
    y_s = np.clip(y_s, 0.0, 1.0)
    y_s = np.maximum.accumulate(y_s)
    return grid, y_s

def make_roc_figure(fpr, tpr, auc_value, sens_at_thr, spec_at_thr, thr_label: str = "",
                    style: str = "classic", smooth: bool = False):
    """
    ROC:
    - style='classic' → verde, linea piena; style='soft' → rosso, tratteggiata (morbida).
    - smooth=True → curva smussata monotòna.
    """
    if smooth:
        x_plot, y_plot = smooth_roc(fpr, tpr)
    else:
        x_plot, y_plot = _uniq_monotone(np.array(fpr, float), np.array(tpr, float))
        if x_plot[0] > 0: x_plot = np.insert(x_plot, 0, 0.0); y_plot = np.insert(y_plot, 0, 0.0)
        if x_plot[-1] < 1: x_plot = np.append(x_plot, 1.0);  y_plot = np.append(y_plot, 1.0)

    if style == "soft":
        line_color = "#8e1b1b"; line_dash = "dot"; fill_color = "rgba(200,0,0,0.20)"; marker_color = "#8e1b1b"
    else:
        line_color = "#2ecc71"; line_dash = None;    fill_color = "rgba(128,128,128,0.35)"; marker_color = "#2ecc71"

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_plot, y=y_plot, mode="lines",
                             line=dict(color=line_color, width=3, dash=line_dash),
                             fill="tozeroy", fillcolor=fill_color,
                             name=f"ROC (AUC={auc_value:.3f})"))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                             line=dict(color="rgba(0,0,0,0.75)", dash="dash", width=2),
                             name="No-skill"))
    fig.add_trace(go.Scatter(x=[1 - spec_at_thr], y=[sens_at_thr], mode="markers",
                             marker=dict(symbol="x", size=11, color=marker_color),
                             name=(f"Soglia {thr_label}" if thr_label else "Soglia")))
    fig.update_layout(template="simple_white", title="Curva ROC",
                      xaxis_title="FPR (1 − Specificità)", yaxis_title="TPR (Sensibilità)",
                      legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="left", x=0),
                      height=420)
    fig.update_xaxes(range=[0, 1], showline=True, linewidth=2, linecolor="black")
    fig.update_yaxes(range=[0, 1], showline=True, linewidth=2, linecolor="black",
                     scaleanchor="x", scaleratio=1)
    return fig

# ──────────────────────────────────────────────────────────────────────────────
# Header
# ──────────────────────────────────────────────────────────────────────────────
st.title("🧮 Regression")
st.caption("Regressione lineare (outcome continuo) e logistica (outcome binario), con diagnostica e guida alla lettura.")

ensure_initialized()
df = get_active(required=True)
with st.expander("Stato dati", expanded=False):
    stamp_meta()
if df is None or df.empty:
    st.stop()

num_vars = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

if px is None:
    st.info("Plotly non è disponibile nell'ambiente. Alcune visualizzazioni potrebbero non comparire.")

# ──────────────────────────────────────────────────────────────────────────────
# Pannelli
# ──────────────────────────────────────────────────────────────────────────────
tab_lin, tab_logit = st.tabs(["📈 Regressione lineare", "⚖️ Regressione logistica"])

# ===========================
# TAB: Regressione lineare
# ===========================
with tab_lin:
    if not num_vars:
        st.info("Serve un outcome numerico.")
    else:
        c1, c2, c3 = st.columns(3)
        with c1:
            y_lin = st.selectbox("Outcome (numerico)", options=num_vars, key=k("lin_y"))
        with c2:
            X_lin = st.multiselect("Predittori", options=[c for c in df.columns if c != y_lin], key=k("lin_X"))
        with c3:
            zscore = st.checkbox("Standardizza predittori numerici (z-score)", value=False, key=k("lin_z"))
        c4, c5, c6 = st.columns(3)
        with c4:
            robust = st.selectbox("Errori standard", ["Classici", "Robusti (HC3)"], key=k("lin_rob"))
        with c5:
            _ = st.selectbox("Gestione NA", ["listwise (consigliato)"], key=k("lin_na"))
        with c6:
            show_anova = st.checkbox("Mostra ANOVA Type II", value=True, key=k("lin_anova"))

        if not X_lin:
            st.info("Selezioni almeno un predittore.")
        else:
            df_fit = df[[y_lin] + X_lin].copy()
            if zscore:
                for c in X_lin:
                    if pd.api.types.is_numeric_dtype(df_fit[c]):
                        s = pd.to_numeric(df_fit[c], errors="coerce")
                        mu, sd = float(s.mean()), float(s.std(ddof=1))
                        if sd and sd > 0:
                            df_fit.loc[:, c] = (s - mu) / sd
            df_fit = df_fit.dropna()

            if smf is None:
                st.error("`statsmodels` non disponibile nell'ambiente.")
                st.stop()

            formula_lin = build_formula(y_lin, X_lin, df_fit)
            try:
                model = smf.ols(formula=formula_lin, data=df_fit)
                fit = model.fit()
                if robust.startswith("Robusti"):
                    fit = fit.get_robustcov_results(cov_type="HC3")

                st.markdown("### Formula del modello")
                st.code(formula_lin, language="text")
                with st.expander("📐 Formula leggibile (LaTeX)"):
                    st.latex(pretty_formula_latex_linear(y_lin, formula_lin, df_fit))

                st.markdown("### Coefficienti")
                tbl = coef_table_from_fit(fit, or_scale=False)
                st.dataframe(tbl, use_container_width=True)

                if hasattr(sm, "stats") and show_anova:
                    try:
                        an = sm.stats.anova_lm(fit, typ=2)
                        st.markdown("### ANOVA (Type II)")
                        st.dataframe(an.round(3), use_container_width=True)
                    except Exception as e:
                        st.caption(f"ANOVA non disponibile: {e}")

                st.markdown("### Diagnostica")
                resid = np.asarray(fit.resid)
                fitted = np.asarray(fit.fittedvalues)
                c1p, c2p = st.columns(2)
                with c1p:
                    if px is not None:
                        fig1 = px.scatter(x=fitted, y=resid, labels={"x": "Fitted", "y": "Residui"},
                                          template="simple_white", title="Residui vs Fitted")
                        fig1.add_hline(y=0, line_dash="dash")
                        st.plotly_chart(fig1, use_container_width=True)
                with c2p:
                    if stats is not None and resid.size >= 3 and go is not None:
                        p = (np.arange(1, resid.size + 1) - 0.5) / resid.size
                        q = stats.norm.ppf(p); qy = np.sort(pd.Series(resid).dropna().values)
                        fig2 = go.Figure()
                        fig2.add_trace(go.Scatter(x=q, y=qy, mode="markers", name="Residui"))
                        mn = float(np.nanmin([q.min(), qy.min()])); mx = float(np.nanmax([q.max(), qy.max()]))
                        fig2.add_shape(type="line", x0=mn, y0=mn, x1=mx, y1=mx, line=dict(dash="dash"))
                        fig2.update_layout(template="simple_white", title="QQ-plot residui",
                                           xaxis_title="Quantili teorici", yaxis_title="Residui")
                        st.plotly_chart(fig2, use_container_width=True)

                st.markdown("### Indicatori di adattamento")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("R²", f"{fit.rsquared:.3f}")
                m1.caption("Quota di varianza spiegata (0–1). Più alto è meglio.")
                m2.metric("R² adj.", f"{fit.rsquared_adj:.3f}")
                m2.caption("R² corretto per il numero di predittori; confronta modelli con diversa complessità.")
                m3.metric("AIC", f"{fit.aic:.1f}")
                m3.caption("Criterio informativo (più basso è meglio). Penalizza la complessità.")
                m4.metric("BIC", f"{fit.bic:.1f}")
                m4.caption("Come AIC ma penalizza di più i modelli complessi.")

            except Exception as e:
                st.error(f"Errore nella stima OLS: {e}")

# ===========================
# TAB: Regressione logistica
# ===========================
with tab_logit:
    all_vars = list(df.columns)
    y_logit = st.selectbox("Outcome (binario o categoriale)", options=all_vars, key=k("log_y"))

    y_series = df[y_logit]
    if pd.api.types.is_numeric_dtype(y_series) and set(pd.unique(y_series.dropna())) <= {0, 1}:
        success_label = 1
        y_encoded = (y_series == 1).astype(int)
        st.caption("Outcome interpretato come binario {0,1} con '1' = successo.")
    else:
        lvls = sorted(y_series.dropna().astype(str).unique().tolist())
        success_label = st.selectbox("Categoria considerata 'successo'", options=lvls, key=k("log_succ"))
        y_encoded = (y_series.astype(str) == success_label).astype(int)

    X_opts = [c for c in df.columns if c != y_logit]
    X_log = st.multiselect("Predittori", options=X_opts, key=k("log_X"))

    colA, colB, colC = st.columns(3)
    with colA:
        zscore_log = st.checkbox("Standardizza predittori numerici (z-score)", value=False, key=k("log_z"))
    with colB:
        robust_log = st.selectbox("Errori standard", ["Classici", "Robusti (HC3)"], key=k("log_rob"))
    with colC:
        thr = st.slider("Soglia di classificazione", min_value=0.05, max_value=0.95, value=0.50, step=0.05, key=k("log_thr"))

    if not X_log:
        st.info("Selezioni almeno un predittore.")
    else:
        df_fit = df[X_log].copy()
        df_fit["_y"] = y_encoded.values
        if zscore_log:
            for c in X_log:
                if pd.api.types.is_numeric_dtype(df_fit[c]):
                    s = pd.to_numeric(df_fit[c], errors="coerce")
                    mu, sd = float(s.mean()), float(s.std(ddof=1))
                    if sd and sd > 0:
                        df_fit.loc[:, c] = (s - mu) / sd
        df_fit = df_fit.dropna()

        if df_fit["_y"].nunique() != 2:
            st.error("L'outcome non risulta binario dopo la preparazione dei dati.")
            st.stop()

        if smf is None or sm is None:
            st.error("`statsmodels` non disponibile nell'ambiente.")
            st.stop()

        try:
            formula_log = build_formula("_y", X_log, df_fit)
            model = smf.glm(formula=formula_log, data=df_fit, family=sm.families.Binomial())
            fit = model.fit()
            if robust_log.startswith("Robusti"):
                fit = model.fit(cov_type="HC3")

            st.markdown("### Formula del modello")
            st.code(formula_log, language="text")
            with st.expander("📐 Formula leggibile (LaTeX)"):
                st.latex(pretty_formula_latex_logit(y_logit, formula_log, df_fit, success_label))

            st.markdown("### Coefficienti")
            tbl = coef_table_from_fit(fit, or_scale=True)
            st.dataframe(tbl, use_container_width=True)

            # Prestazioni
            p_hat = np.asarray(fit.predict(df_fit))
            y_true = df_fit["_y"].astype(int).values
            y_pred = (p_hat >= thr).astype(int)

            TP = int(((y_true == 1) & (y_pred == 1)).sum())
            TN = int(((y_true == 0) & (y_pred == 0)).sum())
            FP = int(((y_true == 0) & (y_pred == 1)).sum())
            FN = int(((y_true == 1) & (y_pred == 0)).sum())

            acc = (TP + TN) / max(len(y_true), 1)
            sens = TP / max((TP + FN), 1)
            spec = TN / max((TN + FP), 1)
            auc = auc_mann_whitney(y_true, p_hat)

            mcfR2 = None
            try:
                llf = float(fit.llf); llnull = float(fit.llnull) if hasattr(fit, "llnull") else None
                if llnull is not None and llnull != 0:
                    mcfR2 = 1.0 - (llf / llnull)
            except Exception:
                pass

            st.markdown("### Matrice di confusione")
            cm_tp = confusion_table_counts_percent(TN, FP, FN, TP)
            st.dataframe(cm_tp, use_container_width=True)

            # Metriche con spiegazione
            st.markdown("### Indicatori di prestazione")
            c1m, c2m, c3m, c4m, c5m = st.columns(5)
            c1m.metric("Accuracy", f"{acc:.3f}")
            c1m.caption("Quota di classificazioni corrette complessive (dipende dal bilanciamento).")
            c2m.metric("Sensibilità (TPR)", f"{sens:.3f}")
            c2m.caption("Pr(Y=1 | Y vero=1): capacità di cogliere i positivi.")
            c3m.metric("Specificità (TNR)", f"{spec:.3f}")
            c3m.caption("Pr(Y=0 | Y vero=0): capacità di evitare falsi positivi.")
            c4m.metric("AUC (ROC)", f"{auc:.3f}" if auc == auc else "—")
            c4m.caption("Discriminazione globale: 0.5=casuale; >0.8=ottima; 1=perfetta.")
            c5m.metric("McFadden R²", f"{mcfR2:.3f}" if mcfR2 is not None and mcfR2 == mcfR2 else "—")
            c5m.caption("Pseudo-R² per modelli logit; utile per confronti tra modelli.")

            # ── Grafici: ROC e distribuzione p̂ affiancati ─────────────────────
            if px is not None and go is not None:
                left_col, right_col = st.columns(2)

                with left_col:
                    try:
                        fpr, tpr = roc_curve_strict(y_true, p_hat)
                    except Exception:
                        # fallback grezzo
                        thr_grid = np.unique(np.round(p_hat, 6))[::-1]
                        fpr, tpr = [0.0], [0.0]
                        for t in thr_grid:
                            yp = (p_hat >= t).astype(int)
                            TPt = ((y_true == 1) & (yp == 1)).sum()
                            TNt = ((y_true == 0) & (yp == 0)).sum()
                            FPt = ((y_true == 0) & (yp == 1)).sum()
                            FNt = ((y_true == 1) & (yp == 0)).sum()
                            tpr.append(TPt / max(TPt + FNt, 1))
                            fpr.append(FPt / max(FPt + TNt, 1))
                        fpr.append(1.0); tpr.append(1.0)
                        fpr, tpr = np.array(fpr), np.array(tpr)

                    opt1, opt2 = st.columns([1.3, 1.7])
                    with opt1:
                        roc_smooth = st.checkbox("Andamento morbido (interpolato)", value=True, key=k("log_roc_smooth"))
                    with opt2:
                        roc_style = st.selectbox("Stile curva",
                                                 ["Classico (verde)", "Morbido (rosso)"],
                                                 index=1 if roc_smooth else 0, key=k("log_roc_style"))
                    style = "soft" if roc_style.startswith("Morbido") else "classic"

                    figroc = make_roc_figure(
                        fpr=fpr, tpr=tpr,
                        auc_value=auc if auc == auc else float("nan"),
                        sens_at_thr=sens, spec_at_thr=spec,
                        thr_label=f"{thr:.2f}",
                        style=style, smooth=roc_smooth
                    )
                    st.plotly_chart(figroc, use_container_width=True)

                with right_col:
                    try:
                        df_plot = pd.DataFrame({"p_hat": p_hat, "y_true": y_true})
                        df_plot["Classe"] = df_plot["y_true"].map({0: "Classe 0", 1: "Classe 1"})
                        figd = px.histogram(df_plot, x="p_hat", color="Classe",
                                            barmode="overlay", nbins=30, template="simple_white",
                                            title="Probabilità stimate per classe")
                        figd.add_vline(x=thr, line_dash="dash")
                        figd.update_layout(xaxis_title="p̂", yaxis_title="Frequenza")
                        st.plotly_chart(figd, use_container_width=True)
                    except Exception:
                        st.info("Impossibile disegnare l’istogramma delle probabilità stimate.")

            with st.expander("ℹ️ Come leggere", expanded=False):
                st.markdown(
                    "- **β/OR**: β è su log-odds; `OR = exp(β)` (CI su scala OR).  \n"
                    "- **Accuracy** riflette il bilanciamento delle classi; Sensibilità/Specificità gestiscono i due tipi di errore.  \n"
                    "- **AUC** valuta la discriminazione indipendentemente dalla soglia; **McFadden R²** confronta modelli logit.  \n"
                    "- Le distribuzioni di **p̂** dovrebbero sovrapporsi poco se il modello discrimina bene.  \n"
                    "- **Curva ROC morbida**: interpolazione monotòna per leggibilità; i punti estremi (0,0) e (1,1) sono sempre inclusi."
                )

        except Exception as e:
            st.error(f"Errore nella stima logistica: {e}")

# ──────────────────────────────────────────────────────────────────────────────
# Navigazione
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("---")
nav1, nav2 = st.columns(2)
with nav1:
    if st.button("⬅️ Torna: Subgroup Analysis", use_container_width=True, key=k("go_prev")):
        st.switch_page("pages/7_📂_Subgroup_Analysis.py")
with nav2:
    if st.button("➡️ Vai: Analisi Test Diagnostici", use_container_width=True, key=k("go_next")):
        st.switch_page("pages/9_🔬_Analisi_Test_Diagnostici.py")
