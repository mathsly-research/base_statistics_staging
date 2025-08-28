# pages/8_🧮_Regression.py
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

# Statistiche / Modelli (opzionale)
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
st.set_page_config(page_title="🧮 Regression", layout="wide")
try:
    from nav import sidebar
    sidebar()
except Exception:
    pass

KEY = "reg"
def k(name: str) -> str: return f"{KEY}_{name}"

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

# Variabili
num_vars = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
cat_vars = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]

if px is None:
    st.info("Plotly non è disponibile nell'ambiente. Alcune visualizzazioni potrebbero non comparire.")

# ──────────────────────────────────────────────────────────────────────────────
# Helper
# ──────────────────────────────────────────────────────────────────────────────
def fq(s: str) -> str:
    """Escape sicuro per formule Patsy/Statsmodels (nomi con spazi/simboli/emoji)."""
    return s.replace("\\", "\\\\").replace("'", "\\'")

def build_formula(y: str, X: list[str], df_: pd.DataFrame):
    """Costruisce formula con quoting sicuro e C() per categoriali."""
    terms = []
    for v in X:
        if pd.api.types.is_numeric_dtype(df_[v]):
            terms.append(f"Q('{fq(v)}')")
        else:
            terms.append(f"C(Q('{fq(v)}'))")
    rhs = " + ".join(terms) if terms else "1"
    return f"Q('{fq(y)}') ~ {rhs}"

def standardize_inplace(df_: pd.DataFrame, cols: list[str]):
    """Standardizza z-score solo le colonne numeriche presenti."""
    for c in cols:
        if c in df_.columns and pd.api.types.is_numeric_dtype(df_[c]):
            s = df_[c].astype(float)
            mu, sd = s.mean(), s.std(ddof=1)
            if sd and sd > 0:
                df_.loc[:, c] = (s - mu) / sd

def qq_points(resid: np.ndarray):
    """Restituisce quantili teorici e campionari per QQ."""
    resid = pd.Series(resid).dropna().values
    n = len(resid)
    if n < 3: return np.array([]), np.array([])
    sr = np.sort(resid)
    # quantili teorici standard normali a (i-0.5)/n
    p = (np.arange(1, n+1) - 0.5) / n
    q = stats.norm.ppf(p) if stats else np.sort(sr)  # fallback
    return q, sr

def calc_vif_from_formula(formula: str, data: pd.DataFrame):
    """Calcola VIF sul design matrix (escludendo l'intercetta)."""
    try:
        import patsy
        y, X = patsy.dmatrices(formula, data=data, return_type="dataframe")
        if "Intercept" in X.columns:
            X_ = X.drop(columns=["Intercept"])
        else:
            X_ = X
        if X_.shape[1] == 0 or variance_inflation_factor is None:
            return None
        vifs = []
        for i, col in enumerate(X_.columns):
            try:
                v = variance_inflation_factor(X_.values, i)
            except Exception:
                v = np.nan
            vifs.append((col, v))
        return pd.DataFrame(vifs, columns=["Termine", "VIF"]).sort_values("VIF", ascending=False)
    except Exception:
        return None

def mcfadden_r2(model_fit):
    try:
        llf = model_fit.llf
        llnull = model_fit.null_deviance / -2 if hasattr(model_fit, "null_deviance") else model_fit.llnull
        return 1 - (llf / llnull)
    except Exception:
        return np.nan

def auc_fast(y_true: np.ndarray, y_score: np.ndarray):
    """AUC via ranghi (equivalente a U/(n1*n0))."""
    try:
        from scipy.stats import rankdata
        y_true = np.asarray(y_true).astype(int)
        y_score = np.asarray(y_score).astype(float)
        n1 = (y_true == 1).sum()
        n0 = (y_true == 0).sum()
        if n1 == 0 or n0 == 0: return np.nan
        ranks = rankdata(y_score)
        sum_ranks_pos = ranks[y_true == 1].sum()
        u = sum_ranks_pos - n1 * (n1 + 1) / 2
        return u / (n1 * n0)
    except Exception:
        return np.nan

# ──────────────────────────────────────────────────────────────────────────────
# Tabs principali
# ──────────────────────────────────────────────────────────────────────────────
tab_lin, tab_logit = st.tabs(["📈 Regressione lineare", "⚖️ Regressione logistica"])

# =============================================================================
# LINEARE
# =============================================================================
with tab_lin:
    st.subheader("📈 Regressione lineare (OLS)")
    if not num_vars:
        st.info("Serve un outcome numerico.")
    else:
        c1, c2, c3 = st.columns([2,2,2])
        with c1:
            y_lin = st.selectbox("Outcome (numerico)", options=num_vars, key=k("lin_y"))
        with c2:
            X_lin = st.multiselect("Predittori", options=[c for c in df.columns if c != y_lin], key=k("lin_X"))
        with c3:
            zscore = st.checkbox("Standardizza predittori numerici (z-score)", value=False, key=k("lin_z"))
        c4, c5, c6 = st.columns([2,2,2])
        with c4:
            robust = st.selectbox("Errori standard", ["Classici", "Robusti (HC3)"], key=k("lin_rob"))
        with c5:
            na_hand = st.selectbox("Gestione NA", ["listwise (consigliato)", "pairwise (non per OLS)"], key=k("lin_na"))
        with c6:
            show_anova = st.checkbox("Mostra ANOVA Type II", value=True, key=k("lin_anova"))

        if not X_lin:
            st.info("Selezioni almeno un predittore.")
        else:
            df_fit = df[[y_lin] + X_lin].copy()
            if zscore:
                standardize_inplace(df_fit, X_lin)
            df_fit = df_fit.apply(pd.to_numeric, errors="ignore")  # non coerce per categoriali
            df_fit = df_fit.dropna()  # listwise

            try:
                formula_lin = build_formula(y_lin, X_lin, df_fit)
                if smf is None:
                    st.error("`statsmodels` non disponibile nell'ambiente.")
                    st.stop()
                model = smf.ols(formula=formula_lin, data=df_fit)
                fit = model.fit()
                if robust.startswith("Robusti"):
                    fit = fit.get_robustcov_results(cov_type="HC3")

                # Riepilogo sintetico
                st.markdown("### Risultati del modello")
                left, right = st.columns([3,2])
                with left:
                    st.write(f"**Formula**: `{formula_lin}`")
                    st.dataframe(fit.summary2().tables[1], use_container_width=True)
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

                # Diagnostica
                st.markdown("### Diagnostica")
                resid = fit.resid
                fitted = fit.fittedvalues
                c1, c2 = st.columns(2)
                with c1:
                    if px is not None:
                        fig1 = px.scatter(x=fitted, y=resid, labels={"x":"Fitted", "y":"Residui"},
                                          template="simple_white", title="Residui vs Fitted")
                        fig1.add_hline(y=0, line_dash="dash")
                        st.plotly_chart(fig1, use_container_width=True)
                with c2:
                    qx, qy = qq_points(resid)
                    if px is not None and qx.size > 0:
                        fig2 = go.Figure()
                        fig2.add_trace(go.Scatter(x=qx, y=qy, mode="markers", name="Residui"))
                        # retta 45°
                        minv = float(np.nanmin([qx.min(), qy.min()]))
                        maxv = float(np.nanmax([qx.max(), qy.max()]))
                        fig2.add_shape(type="line", x0=minv, y0=minv, x1=maxv, y1=maxv, line=dict(dash="dash"))
                        fig2.update_layout(template="simple_white", title="QQ-plot residui", xaxis_title="Quantili teorici", yaxis_title="Residui")
                        st.plotly_chart(fig2, use_container_width=True)

                # Test residui (opzionali)
                with st.expander("🧪 Verifiche sui residui", expanded=False):
                    if het_breuschpagan is not None:
                        try:
                            # het_breuschpagan richiede design matrix con colonna costante
                            import patsy
                            y_dm, X_dm = patsy.dmatrices(formula_lin, data=df_fit, return_type="dataframe")
                            if "Intercept" not in X_dm.columns:
                                X_dm = sm.add_constant(X_dm, prepend=True, has_constant="raise")
                            lm, lm_pvalue, fvalue, f_pvalue = het_breuschpagan(resid, X_dm)
                            st.markdown(f"**Breusch–Pagan**: LM={lm:.2f}, p={lm_pvalue:.4f} (H₀: omoscedasticità)")
                        except Exception as e:
                            st.caption(f"Breusch–Pagan non calcolabile: {e}")
                    if stats is not None and len(resid) >= 3:
                        try:
                            W, p_sh = stats.shapiro(pd.Series(resid).dropna())
                            st.markdown(f"**Shapiro–Wilk**: W={W:.3f}, p={p_sh:.4f} (H₀: normalità residui)")
                        except Exception as e:
                            st.caption(f"Shapiro non calcolabile: {e}")

                # VIF
                with st.expander("📦 Multicollinearità (VIF)", expanded=False):
                    vif = calc_vif_from_formula(formula_lin, df_fit)
                    if vif is not None:
                        st.dataframe(vif, use_container_width=True)
                    else:
                        st.caption("VIF non calcolabile (patsy/statsmodels non disponibili o solo intercetta).")

                # Come leggere
                with st.expander("ℹ️ Come leggere", expanded=False):
                    st.markdown("""
- **β**: effetto atteso sull’outcome per **+1 unità** del predittore (o per passaggio di categoria rispetto alla **reference**).  
- **p < 0.05**: evidenza che il coefficiente ≠ 0; **CI95%** quantifica l’incertezza.  
- **R² / R² adj.**: quota di varianza spiegata; **AIC/BIC** per confronto tra modelli.  
- **Residui vs Fitted**: pattern a ventaglio → possibile **eteroschedasticità** (valuti robusti HC3).  
- **QQ-plot**: deviazioni forti dalla diagonale → residui non normali.  
- **VIF > 5–10**: possibile **multicollinearità** (valuti accorpamenti/penalizzazioni).
""")

# =============================================================================
# LOGISTICA
# =============================================================================
with tab_logit:
    st.subheader("⚖️ Regressione logistica (binaria)")
    # Outcome: numerico 0/1 oppure categoriale con scelta del "successo"
    all_vars = list(df.columns)
    y_logit = st.selectbox("Outcome (binario o categoriale)", options=all_vars, key=k("log_y"))

    # Determino se è già binario 0/1
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
    colA, colB, colC = st.columns([2,2,2])
    with colA:
        zscore_log = st.checkbox("Standardizza predittori numerici (z-score)", value=False, key=k("log_z"))
    with colB:
        robust_log = st.selectbox("Errori standard", ["Classici", "Robusti (HC3)"], key=k("log_rob"))
    with colC:
        thr = st.slider("Soglia di classificazione", min_value=0.05, max_value=0.95, value=0.5, step=0.05, key=k("log_thr"))

    if not X_log:
        st.info("Selezioni almeno un predittore.")
    else:
        df_fit = df[X_log].copy()
        df_fit["_y"] = y_encoded.values
        if zscore_log:
            standardize_inplace(df_fit, X_log)
        df_fit = df_fit.dropna()

        if df_fit["_y"].nunique() != 2:
            st.error("L'outcome non risulta binario dopo la preparazione dei dati.")
            st.stop()

        try:
            formula_log = build_formula("_y", X_log, df_fit)
            if smf is None or sm is None:
                st.error("`statsmodels` non disponibile nell'ambiente.")
                st.stop()

            model = smf.glm(formula=formula_log, data=df_fit, family=sm.families.Binomial())
            fit = model.fit()
            if robust_log.startswith("Robusti"):
                fit = model.fit(cov_type="HC3")

            st.markdown("### Risultati del modello")
            left, right = st.columns([3,2])
            with left:
                st.write(f"**Formula**: `{formula_log}`")
                coefs = fit.summary2().tables[1].copy()
                # Aggiungo Odds Ratio
                try:
                    coefs["OR"] = np.exp(coefs["Coef."])
                    coefs["OR_low"] = np.exp(coefs["[0.025"])
                    coefs["OR_hi"] = np.exp(coefs["0.975]"])
                except Exception:
                    pass
                st.dataframe(coefs, use_container_width=True)
            with right:
                try:
                    llf = fit.llf
                    aic = fit.aic
                    bic = fit.bic if hasattr(fit, "bic") else np.nan
                except Exception:
                    llf = aic = bic = np.nan
                st.metric("McFadden R²", f"{mcfadden_r2(fit):.3f}")
                st.metric("AIC", f"{aic:.1f}")
                st.metric("BIC", f"{bic:.1f}" if not np.isnan(bic) else "—")

            # Prestazioni di classificazione
            st.markdown("### Prestazioni di classificazione")
            try:
                p_hat = fit.predict(df_fit)
                y_true = df_fit["_y"].astype(int).values
                y_pred = (p_hat >= thr).astype(int)
                TP = int(((y_true == 1) & (y_pred == 1)).sum())
                TN = int(((y_true == 0) & (y_pred == 0)).sum())
                FP = int(((y_true == 0) & (y_pred == 1)).sum())
                FN = int(((y_true == 1) & (y_pred == 0)).sum())
                acc = (TP + TN) / max(len(y_true), 1)
                sens = TP / max((TP+FN), 1)
                spec = TN / max((TN+FP), 1)
                prec = TP / max((TP+FP), 1)
                auc = auc_fast(y_true, p_hat)

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Accuracy", f"{acc:.3f}")
                c2.metric("Sensibilità", f"{sens:.3f}")
                c3.metric("Specificità", f"{spec:.3f}")
                c4.metric("AUC (ROC)", f"{auc:.3f}" if auc==auc else "—")

                cm = pd.DataFrame([[TN, FP], [FN, TP]], index=["Vera 0", "Vera 1"], columns=["Pred 0", "Pred 1"])
                st.dataframe(cm, use_container_width=True)

                if px is not None:
                    # Curva ROC (semplice)
                    try:
                        # costruiamo punti ROC
                        thr_grid = np.linspace(0, 1, 101)
                        TPR, FPR = [], []
                        for t in thr_grid:
                            yp = (p_hat >= t).astype(int)
                            TP = ((y_true==1)&(yp==1)).sum()
                            TN = ((y_true==0)&(yp==0)).sum()
                            FP = ((y_true==0)&(yp==1)).sum()
                            FN = ((y_true==1)&(yp==0)).sum()
                            tpr = TP / max((TP+FN), 1)
                            fpr = FP / max((FP+TN), 1)
                            TPR.append(tpr); FPR.append(fpr)
                        figroc = px.area(x=FPR, y=TPR, labels={"x":"FPR", "y":"TPR"},
                                         title="Curva ROC (approssimata)", template="simple_white")
                        figroc.add_shape(type="line", x0=0, x1=1, y0=0, y1=1, line=dict(dash="dash"))
                        st.plotly_chart(figroc, use_container_width=True)
                    except Exception:
                        pass

                    # Distribuzione p̂ per classe
                    try:
                        df_plot = pd.DataFrame({"p_hat": p_hat, "y_true": y_true})
                        figd = px.histogram(df_plot, x="p_hat", color=df_plot["y_true"].map({0:"Classe 0", 1:"Classe 1"}),
                                            barmode="overlay", nbins=30, template="simple_white",
                                            title="Distribuzione delle probabilità stimate per classe")
                        st.plotly_chart(figd, use_container_width=True)
                    except Exception:
                        pass
            except Exception as e:
                st.caption(f"Valutazione prestazioni non disponibile: {e}")

            # VIF
            with st.expander("📦 Multicollinearità (VIF)", expanded=False):
                vif = calc_vif_from_formula(formula_log, df_fit)
                if vif is not None:
                    st.dataframe(vif, use_container_width=True)
                else:
                    st.caption("VIF non calcolabile (patsy/statsmodels non disponibili o solo intercetta).")

            # Come leggere
            with st.expander("ℹ️ Come leggere", expanded=False):
                st.markdown(f"""
- **Log-odds/OR**: i coefficienti esprimono l’effetto sui **log-odds**; `OR = exp(β)` (con CI).  
- **p < 0.05**: evidenza che il coefficiente ≠ 0; controlli anche l’**ampiezza** (OR e CI).  
- **McFadden R²**, **AIC/BIC** per confronto tra modelli; **AUC** valuta discriminazione.  
- **Soglia {thr:.2f}**: determina la classificazione; valuti trade-off sensibilità/specificità.  
- **VIF** alto → possibile multicollinearità.
""")

# ──────────────────────────────────────────────────────────────────────────────
# Navigazione
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("---")
nav1, nav2 = st.columns(2)
with nav1:
    if st.button("⬅️ Torna: Subgroup Analysis", use_container_width=True, key=k("go_prev")):
        st.switch_page("pages/7_📂_Subgroup_Analysis.py")
with nav2:
    if st.button("➡️ Vai: (Prossimo modulo)", use_container_width=True, key=k("go_next")):
        # Adegui questo percorso al nome reale del file successivo nel suo progetto
        st.switch_page("pages/9_📦_Report_Export.py")
