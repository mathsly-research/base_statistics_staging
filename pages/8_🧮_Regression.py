# -*- coding: utf-8 -*-
# pages/8_🧮_Regression.py
from __future__ import annotations
import math
import streamlit as st
import pandas as pd
import numpy as np

try:
    import plotly.express as px
    import plotly.graph_objects as go
except Exception:
    px = None
    go = None

try:
    from scipy import stats
except Exception:
    stats = None

try:
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    from statsmodels.stats.diagnostic import het_breuschpagan
except Exception:
    sm = None
    smf = None
    variance_inflation_factor = None
    het_breuschpagan = None

# ───────── Data store (fallback) ─────────
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

# ───────── Config ─────────
st.set_page_config(page_title="Regression", layout="wide")
try:
    from nav import sidebar
    sidebar()
except Exception:
    pass

KEY = "reg"
def k(name: str) -> str: return f"{KEY}_{name}"

st.title("🧮 Regression")
st.caption("Regressione lineare (outcome continuo) e logistica (outcome binario), con diagnostica e guida alla lettura.")

ensure_initialized()
df = get_active(required=True)
with st.expander("Stato dati", expanded=False):
    stamp_meta()

num_vars = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
cat_vars = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]

if px is None:
    st.info("Plotly non è disponibile nell'ambiente. Le visualizzazioni interattive potrebbero non comparire.")

# ───────── Helper ─────────
def fq(s: str) -> str:
    return s.replace("\\", "\\\\").replace("'", "\\'")

def build_formula(y: str, X: list[str], df_: pd.DataFrame):
    terms = []
    for v in X:
        if pd.api.types.is_numeric_dtype(df_[v]):
            terms.append(f"Q('{fq(v)}')")
        else:
            terms.append(f"C(Q('{fq(v)}'))")
    rhs = " + ".join(terms) if terms else "1"
    return f"Q('{fq(y)}') ~ {rhs}"

def standardize_inplace(df_: pd.DataFrame, cols: list[str]):
    for c in cols:
        if c in df_.columns and pd.api.types.is_numeric_dtype(df_[c]):
            s = pd.to_numeric(df_[c], errors="coerce")
            mu, sd = float(s.mean()), float(s.std(ddof=1))
            if sd and sd > 0:
                df_.loc[:, c] = (s - mu) / sd

def qq_points(resid: np.ndarray):
    s = pd.Series(resid).dropna().values
    n = len(s)
    if n < 3: return np.array([]), np.array([])
    p = (np.arange(1, n+1) - 0.5) / n
    q = stats.norm.ppf(p) if stats else np.sort(s)
    return q, np.sort(s)

def calc_vif_from_formula(formula: str, data: pd.DataFrame):
    try:
        import patsy
        y, X = patsy.dmatrices(formula, data=data, return_type="dataframe")
        X_ = X.drop(columns=["Intercept"], errors="ignore")
        if X_.shape[1] == 0 or variance_inflation_factor is None:
            return None
        vifs = []
        for i, col in enumerate(X_.columns):
            try:
                v = float(variance_inflation_factor(X_.values, i))
            except Exception:
                v = float("nan")
            vifs.append((str(col), v))
        return pd.DataFrame(vifs, columns=["Termine", "VIF"]).sort_values("VIF", ascending=False)
    except Exception:
        return None

def mcfadden_r2(model_fit):
    try:
        llf = float(model_fit.llf)
        llnull = float(model_fit.llnull) if hasattr(model_fit, "llnull") else float(model_fit.null_deviance) / -2.0
        return 1.0 - (llf / llnull)
    except Exception:
        return float("nan")

def auc_fast(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """AUC via U di Mann–Whitney (equivalente alla trapezoidale su ROC corretta)."""
    try:
        from scipy.stats import rankdata
        y_true = np.asarray(y_true).astype(int)
        y_score = np.asarray(y_score).astype(float)
        n1 = int((y_true == 1).sum())
        n0 = int((y_true == 0).sum())
        if n1 == 0 or n0 == 0: return float("nan")
        ranks = rankdata(y_score)
        sum_r_pos = float(ranks[y_true == 1].sum())
        U = sum_r_pos - n1 * (n1 + 1) / 2.0
        return float(U / (n1 * n0))
    except Exception:
        return float("nan")

def roc_curve_strict(y_true: np.ndarray, y_score: np.ndarray):
    """ROC corretta: soglie ai valori distinti dello score (desc), gestione tie, include (0,0) e (1,1)."""
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    P = int((y_true == 1).sum())
    N = int((y_true == 0).sum())
    if P == 0 or N == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0])
    order = np.argsort(-y_score, kind="mergesort")  # stabile per gestire tie
    y_sorted = y_true[order]
    s_sorted = y_score[order]
    tpr = [0.0]; fpr = [0.0]
    tp = 0; fp = 0
    i = 0; n = len(y_sorted)
    while i < n:
        thr = s_sorted[i]
        # accumula tutti i casi con lo stesso score
        tp_inc = 0; fp_inc = 0
        while i < n and s_sorted[i] == thr:
            if y_sorted[i] == 1: tp_inc += 1
            else: fp_inc += 1
            i += 1
        tp += tp_inc; fp += fp_inc
        tpr.append(tp / P)
        fpr.append(fp / N)
    # assicura punto (1,1)
    if tpr[-1] != 1.0 or fpr[-1] != 1.0:
        tpr.append(1.0); fpr.append(1.0)
    return np.array(fpr, dtype=float), np.array(tpr, dtype=float)

# ───────── Tabs ─────────
tab_lin, tab_logit = st.tabs(["📈 Regressione lineare", "⚖️ Regressione logistica"])

# ===== LINEARE =====
with tab_lin:
    st.subheader("📈 Regressione lineare (OLS)")
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
            st.selectbox("Gestione NA", ["listwise (consigliato)"], key=k("lin_na"))
        with c6:
            show_anova = st.checkbox("Mostra ANOVA Type II", value=True, key=k("lin_anova"))

        if not X_lin:
            st.info("Selezioni almeno un predittore.")
        else:
            df_fit = df[[y_lin] + X_lin].copy()
            if zscore: standardize_inplace(df_fit, X_lin)
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

                st.markdown("### Risultati del modello")
                left, right = st.columns([3,2])
                with left:
                    st.write("**Formula**:"); st.code(formula_lin, language="text")
                    try:
                        st.dataframe(fit.summary2().tables[1], use_container_width=True)
                    except Exception:
                        st.text(fit.summary().as_text())
                with right:
                    st.metric("R²", f"{fit.rsquared:.3f}")
                    st.metric("R² adj.", f"{fit.rsquared_adj:.3f}")
                    st.metric("AIC", f"{fit.aic:.1f}")
                    st.metric("BIC", f"{fit.bic:.1f}")

                if show_anova and hasattr(sm, "stats"):
                    try:
                        an = sm.stats.anova_lm(fit, typ=2)
                        st.markdown("**ANOVA Type II**")
                        st.dataframe(an, use_container_width=True)
                    except Exception as e:
                        st.caption(f"ANOVA non disponibile: {e}")

                st.markdown("### Diagnostica")
                resid = np.asarray(fit.resid); fitted = np.asarray(fit.fittedvalues)
                c1p, c2p = st.columns(2)
                with c1p:
                    if px is not None:
                        fig1 = px.scatter(x=fitted, y=resid, labels={"x":"Fitted","y":"Residui"},
                                          template="simple_white", title="Residui vs Fitted")
                        fig1.add_hline(y=0, line_dash="dash")
                        st.plotly_chart(fig1, use_container_width=True)
                with c2p:
                    qx, qy = qq_points(resid)
                    if px is not None and qx.size > 0:
                        fig2 = go.Figure()
                        fig2.add_trace(go.Scatter(x=qx, y=qy, mode="markers", name="Residui"))
                        mn = float(np.nanmin([qx.min(), qy.min()])); mx = float(np.nanmax([qx.max(), qy.max()]))
                        fig2.add_shape(type="line", x0=mn, y0=mn, x1=mx, y1=mx, line=dict(dash="dash"))
                        fig2.update_layout(template="simple_white", title="QQ-plot residui",
                                           xaxis_title="Quantili teorici", yaxis_title="Residui")
                        st.plotly_chart(fig2, use_container_width=True)

                with st.expander("🧪 Verifiche sui residui", expanded=False):
                    if het_breuschpagan is not None:
                        try:
                            import patsy
                            _, Xdm = patsy.dmatrices(formula_lin, data=df_fit, return_type="dataframe")
                            if "Intercept" not in Xdm.columns:
                                Xdm = sm.add_constant(Xdm, prepend=True, has_constant="raise")
                            lm, lm_p, _, _ = het_breuschpagan(fit.resid, Xdm)
                            st.markdown(f"**Breusch–Pagan**: LM={lm:.2f}, p={lm_p:.4f} (H₀: omoscedasticità)")
                        except Exception as e:
                            st.caption(f"Breusch–Pagan non calcolabile: {e}")
                    if stats is not None and len(resid) >= 3:
                        try:
                            W, p_sh = stats.shapiro(pd.Series(resid).dropna())
                            st.markdown(f"**Shapiro–Wilk**: W={W:.3f}, p={p_sh:.4f} (H₀: normalità residui)")
                        except Exception as e:
                            st.caption(f"Shapiro non calcolabile: {e}")

                with st.expander("📦 Multicollinearità (VIF)", expanded=False):
                    vif = calc_vif_from_formula(formula_lin, df_fit)
                    st.dataframe(vif, use_container_width=True) if vif is not None else st.caption("VIF non calcolabile.")

                with st.expander("ℹ️ Come leggere", expanded=False):
                    st.markdown(
                        "- **β**: effetto atteso sull’outcome per +1 unità (o rispetto alla categoria di riferimento).  \n"
                        "- **p < 0.05**: coefficiente ≠ 0; osservare **CI95%**.  \n"
                        "- **R²/R² adj., AIC/BIC** per confronto modelli.  \n"
                        "- **Residui vs Fitted**: ventaglio → possibile eteroschedasticità; **QQ-plot** per normalità.  \n"
                        "- **VIF > 5–10**: potenziale multicollinearità."
                    )
            except Exception as e:
                st.error(f"Errore OLS: {e}")

# ===== LOGISTICA =====
with tab_logit:
    st.subheader("⚖️ Regressione logistica (binaria)")
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
        thr = st.slider("Soglia di classificazione", 0.05, 0.95, 0.50, 0.05, key=k("log_thr"))

    if not X_log:
        st.info("Selezioni almeno un predittore.")
    else:
        df_fit = df[X_log].copy()
        df_fit["_y"] = y_encoded.values
        if zscore_log: standardize_inplace(df_fit, X_log)
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

            st.markdown("### Risultati del modello")
            left, right = st.columns([3,2])
            with left:
                st.write("**Formula**:"); st.code(formula_log, language="text")
                try:
                    coefs = fit.summary2().tables[1].copy()
                    if {"Coef.", "[0.025", "0.975]"}.issubset(coefs.columns):
                        coefs["OR"] = np.exp(coefs["Coef."])
                        coefs["OR_low"] = np.exp(coefs["[0.025"])
                        coefs["OR_hi"] = np.exp(coefs["0.975]"])
                    else:
                        if "Coef." in coefs.columns: coefs["OR"] = np.exp(coefs["Coef."])
                    st.dataframe(coefs, use_container_width=True)
                except Exception:
                    st.text(fit.summary().as_text())
            with right:
                st.metric("McFadden R²", f"{mcfadden_r2(fit):.3f}")
                st.metric("AIC", f"{fit.aic:.1f}")
                st.metric("BIC", f"{getattr(fit,'bic',float('nan')):.1f}" if hasattr(fit,'bic') else "—")

            # Prestazioni + ROC corretta
            st.markdown("### Prestazioni di classificazione")
            try:
                p_hat = np.asarray(fit.predict(df_fit))
                y_true = df_fit["_y"].astype(int).values
                y_pred = (p_hat >= thr).astype(int)
                TP = int(((y_true==1)&(y_pred==1)).sum())
                TN = int(((y_true==0)&(y_pred==0)).sum())
                FP = int(((y_true==0)&(y_pred==1)).sum())
                FN = int(((y_true==1)&(y_pred==0)).sum())
                acc = (TP+TN)/max(len(y_true),1); sens = TP/max((TP+FN),1); spec = TN/max((TN+FP),1)
                auc = auc_fast(y_true, p_hat)

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Accuracy", f"{acc:.3f}")
                c2.metric("Sensibilità", f"{sens:.3f}")
                c3.metric("Specificità", f"{spec:.3f}")
                c4.metric("AUC (ROC)", f"{auc:.3f}" if auc==auc else "—")

                cm = pd.DataFrame([[TN, FP],[FN, TP]], index=["Vera 0","Vera 1"], columns=["Pred 0","Pred 1"])
                st.dataframe(cm, use_container_width=True)

                # ROC corretta (step) + diagonale
                if px is not None:
                    fpr, tpr = roc_curve_strict(y_true, p_hat)
                    figroc = go.Figure()
                    figroc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", line_shape="hv", name="ROC"))
                    figroc.add_shape(type="line", x0=0, x1=1, y0=0, y1=1, line=dict(dash="dash"))
                    figroc.update_layout(template="simple_white", title="Curva ROC (corretta)",
                                         xaxis_title="FPR", yaxis_title="TPR", yaxis=dict(range=[0,1]), xaxis=dict(range=[0,1]))
                    st.plotly_chart(figroc, use_container_width=True)

                    # Distribuzione p̂ per classe
                    try:
                        df_plot = pd.DataFrame({"p_hat": p_hat, "y_true": y_true})
                        df_plot["Classe"] = df_plot["y_true"].map({0:"Classe 0", 1:"Classe 1"})
                        figd = px.histogram(df_plot, x="p_hat", color="Classe", barmode="overlay",
                                            nbins=30, template="simple_white", title="Distribuzione delle probabilità stimate per classe")
                        st.plotly_chart(figd, use_container_width=True)
                    except Exception:
                        pass
            except Exception as e:
                st.caption(f"Valutazione prestazioni non disponibile: {e}")

            with st.expander("📦 Multicollinearità (VIF)", expanded=False):
                vif = calc_vif_from_formula(formula_log, df_fit)
                st.dataframe(vif, use_container_width=True) if vif is not None else st.caption("VIF non calcolabile.")

            with st.expander("ℹ️ Come leggere", expanded=False):
                st.markdown(
                    "- Coefficienti su **log-odds**; `OR = exp(β)` (con CI).  \n"
                    "- **AUC** misura la discriminazione complessiva (0.5=casuale, 1=perfetta).  \n"
                    "- La **ROC** è uno **step-plot** da (0,0) a (1,1): ogni salto corrisponde a un valore distinto di p̂.  \n"
                    "- La **soglia** imposta il compromesso **sensibilità/specificità**."
                )
        except Exception as e:
            st.error(f"Errore nella stima logistica: {e}")

# ───────── Navigazione ─────────
st.markdown("---")
nav1, nav2 = st.columns(2)
with nav1:
    if st.button("⬅️ Torna: Subgroup Analysis", use_container_width=True, key=k("go_prev")):
        st.switch_page("pages/7_📂_Subgroup_Analysis.py")
with nav2:
    if st.button("➡️ Vai: Analisi Test Diagnostici", use_container_width=True, key=k("go_next")):
        st.switch_page("pages/9_🔬_Analisi_Test_Diagnostici.py")
