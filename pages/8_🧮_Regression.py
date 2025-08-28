# -*- coding: utf-8 -*-
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
st.set_page_config(page_title="🧮 Regression", layout="wide")
try:
    from nav import sidebar
    sidebar()
except Exception:
    pass

KEY = "reg"
def k(name: str) -> str:
    return f"{KEY}_{name}"

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

def build_formula(y: str, X: list[str], df_: pd.DataFrame) -> str:
    """Costruisce formula con quoting sicuro e C() per categoriali."""
    terms: list[str] = []
    for v in X:
        if pd.api.types.is_numeric_dtype(df_[v]):
            terms.append(f"Q('{fq(v)}')")
        else:
            terms.append(f"C(Q('{fq(v)}'))")
    rhs = " + ".join(terms) if terms else "1"
    return f"Q('{fq(y)}') ~ {rhs}"

def standardize_inplace(df_: pd.DataFrame, cols: list[str]) -> None:
    """Standardizza z-score solo le colonne numeriche presenti."""
    for c in cols:
        if c in df_.columns and pd.api.types.is_numeric_dtype(df_[c]):
            s = pd.to_numeric(df_[c], errors="coerce")
            mu = float(s.mean())
            sd = float(s.std(ddof=1))
            if sd and sd > 0:
                df_.loc[:, c] = (s - mu) / sd

def qq_points(resid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Restituisce quantili teorici e campionari per QQ."""
    sr = pd.Series(resid).dropna().values
    n = len(sr)
    if n < 3:
        return np.array([]), np.array([])
    p = (np.arange(1, n + 1) - 0.5) / n
    if stats is not None:
        q = stats.norm.ppf(p)
    else:
        q = np.sort(sr)
    return q, np.sort(sr)

def calc_vif_from_formula(formula: str, data: pd.DataFrame) -> pd.DataFrame | None:
    """Calcola VIF sul design matrix (escludendo l'intercetta)."""
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

def mcfadden_r2(model_fit) -> float:
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
        if n1 == 0 or n0 == 0:
            return float("nan")
        ranks = rankdata(y_score)
        sum_r_pos = float(ranks[y_true == 1].sum())
        u = sum_r_pos - n1 * (n1 + 1) / 2.0
        return float(u / (n1 * n0))
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
    order = np.argsort(-y_score, kind="mergesort")  # stabile
    y_sorted = y_true[order]
    s_sorted = y_score[order]
    tpr = [0.0]
    fpr = [0.0]
    tp = 0
    fp = 0
    i = 0
    n = len(y_sorted)
    while i < n:
        thr = s_sorted[i]
        tp_inc = 0
        fp_inc = 0
        while i < n and s_sorted[i] == thr:
            if y_sorted[i] == 1:
                tp_inc += 1
            else:
                fp_inc += 1
            i += 1
        tp += tp_inc
        fp += fp_inc
        tpr.append(tp / P)
        fpr.append(fp / N)
    if tpr[-1] != 1.0 or fpr[-1] != 1.0:
        tpr.append(1.0)
        fpr.append(1.0)
    return np.array(fpr, dtype=float), np.array(tpr, dtype=float)

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
                standardize_inplace(df_fit, X_lin)
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
                left, right = st.columns([3, 2])
                with left:
                    st.write("**Formula**:")
                    st.code(formula_lin, language="text")
                    try:
                        tbl = fit.summary2().tables[1]
                        st.dataframe(tbl, use_container_width=True)
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

                # Diagnostica
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
                    qx, qy = qq_points(resid)
                    if px is not None and qx.size > 0:
                        fig2 = go.Figure()
                        fig2.add_trace(go.Scatter(x=qx, y=qy, mode="markers", name="Residui"))
                        minv = float(np.nanmin([qx.min(), qy.min()]))
                        maxv = float(np.nanmax([qx.max(), qy.max()]))
                        fig2.add_shape(type="line", x0=minv, y0=minv, x1=maxv, y1=maxv, line=dict(dash="dash"))
                        fig2.update_layout(template="simple_white", title="QQ-plot residui",
                                           xaxis_title="Quantili teorici", yaxis_title="Residui")
                        st.plotly_chart(fig2, use_container_width=True)

                # Verifiche sui residui
                with st.expander("🧪 Verifiche sui residui", expanded=False):
                    if het_breuschpagan is not None:
                        try:
                            import patsy
                            y_dm, X_dm = patsy.dmatrices(formula_lin, data=df_fit, return_type="dataframe")
                            if "Intercept" not in X_dm.columns:
                                X_dm = sm.add_constant(X_dm, prepend=True, has_constant="raise")
                            lm, lm_pvalue, fvalue, f_pvalue = het_breuschpagan(fit.resid, X_dm)
                            st.markdown(f"**Breusch–Pagan**: LM={lm:.2f}, p={lm_pvalue:.4f} (H₀: omoscesasticità)")
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
                    st.markdown(
                        "- **β**: effetto atteso sull’outcome per +1 unità del predittore "
                        "(o per passaggio di categoria rispetto alla reference).\n"
                        "- **p < 0.05**: coefficiente diverso da 0; osservi anche **CI95%**.\n"
                        "- **R² / R² adj.**: quota di varianza spiegata; **AIC/BIC** per confronto modelli.\n"
                        "- **Residui vs Fitted**: ventaglio → possibile eteroschedasticità (consideri robusti HC3).\n"
                        "- **QQ-plot**: deviazioni forti dalla diagonale → residui non normali.\n"
                        "- **VIF > 5–10**: possibile multicollinearità."
                    )

            except Exception as e:
                st.error(f"Errore nella stima OLS: {e}")

# =============================================================================
# LOGISTICA
# =============================================================================
with tab_logit:
    st.subheader("⚖️ Regressione logistica (binaria)")
    all_vars = list(df.columns)
    y_logit = st.selectbox("Outcome (binario o categoriale)", options=all_vars, key=k("log_y"))

    # Outcome binario / categoriale
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
            standardize_inplace(df_fit, X_log)
        df_fit = df_fit.dropna()

        if df_fit["_y"].nunique() != 2:
            st.error("L'outcome non risulta binario dopo la preparazione dei dati.")
            st.stop()

        if smf is None or sm is None:
            st.error("`statsmodels` non disponibile nell'ambiente.")
            st.stop()

        try:
            # Modello
            formula_log = build_formula("_y", X_log, df_fit)
            model = smf.glm(formula=formula_log, data=df_fit, family=sm.families.Binomial())
            fit = model.fit()
            if robust_log.startswith("Robusti"):
                fit = model.fit(cov_type="HC3")

            st.markdown("### Risultati del modello")
            left, right = st.columns([3, 2])
            with left:
                st.write("**Formula**:")
                st.code(formula_log, language="text")
                coefs = None
                try:
                    coefs = fit.summary2().tables[1].copy()
                except Exception:
                    pass
                if coefs is not None:
                    if {"Coef.", "[0.025", "0.975]"}.issubset(coefs.columns):
                        coefs["OR"] = np.exp(coefs["Coef."])
                        coefs["OR_low"] = np.exp(coefs["[0.025"])
                        coefs["OR_hi"] = np.exp(coefs["0.975]"])
                    elif "Coef." in coefs.columns:
                        coefs["OR"] = np.exp(coefs["Coef."])
                    st.dataframe(coefs, use_container_width=True)
                else:
                    st.text(fit.summary().as_text())
            with right:
                try:
                    aic_val = float(fit.aic)
                except Exception:
                    aic_val = float("nan")
                try:
                    bic_val = float(fit.bic) if hasattr(fit, "bic") else float("nan")
                except Exception:
                    bic_val = float("nan")
                st.metric("McFadden R²", f"{mcfadden_r2(fit):.3f}")
                st.metric("AIC", f"{aic_val:.1f}" if aic_val == aic_val else "—")
                st.metric("BIC", f"{bic_val:.1f}" if bic_val == bic_val else "—")

            # Prestazioni di classificazione
            st.markdown("### Prestazioni di classificazione")
            try:
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
                auc = auc_fast(y_true, p_hat)

                c1m, c2m, c3m, c4m = st.columns(4)
                c1m.metric("Accuracy", f"{acc:.3f}")
                c2m.metric("Sensibilità", f"{sens:.3f}")
                c3m.metric("Specificità", f"{spec:.3f}")
                c4m.metric("AUC (ROC)", f"{auc:.3f}" if auc == auc else "—")

                cm = pd.DataFrame([[TN, FP], [FN, TP]],
                                  index=["Vera 0", "Vera 1"],
                                  columns=["Pred 0", "Pred 1"])
                st.dataframe(cm, use_container_width=True)

                # ── ROC classica + distribuzione p-hat affiancata ──
                if px is not None and go is not None:
                    left_col, right_col = st.columns(2)

                    # ROC corretta con step "hv", diagonale e punto alla soglia
                    with left_col:
                        try:
                            fpr, tpr = roc_curve_strict(y_true, p_hat)
                        except Exception:
                            # fallback (non ideale)
                            thr_grid = np.unique(np.round(p_hat, 6))[::-1]
                            fpr, tpr = [], []
                            for t in thr_grid:
                                yp = (p_hat >= t).astype(int)
                                TPt = ((y_true == 1) & (yp == 1)).sum()
                                TNt = ((y_true == 0) & (yp == 0)).sum()
                                FPt = ((y_true == 0) & (yp == 1)).sum()
                                FNt = ((y_true == 1) & (yp == 0)).sum()
                                tpr.append(TPt / max(TPt + FNt, 1))
                                fpr.append(FPt / max(FPt + TNt, 1))
                            fpr = np.array([0.0] + fpr + [1.0])
                            tpr = np.array([0.0] + tpr + [1.0])

                        figroc = go.Figure()
                        figroc.add_trace(go.Scatter(
                            x=fpr, y=tpr, mode="lines", line_shape="hv",
                            name=f"ROC (AUC={auc:.3f})"
                        ))
                        # linea no-skill
                        figroc.add_shape(type="line", x0=0, x1=1, y0=0, y1=1, line=dict(dash="dash"))
                        # punto alla soglia corrente
                        fpr_thr = 1 - spec
                        tpr_thr = sens
                        figroc.add_trace(go.Scatter(
                            x=[fpr_thr], y=[tpr_thr], mode="markers",
                            marker=dict(size=10, symbol="x"),
                            name=f"Soglia {thr:.2f}"
                        ))
                        figroc.update_layout(
                            template="simple_white", title="Curva ROC",
                            xaxis_title="FPR (1 − Specificità)",
                            yaxis_title="TPR (Sensibilità)",
                            legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="left", x=0)
                        )
                        figroc.update_xaxes(range=[0, 1])
                        # aspetto quadrato classico
                        figroc.update_yaxes(range=[0, 1], scaleanchor="x", scaleratio=1)
                        st.plotly_chart(figroc, use_container_width=True)

                    # Distribuzione di p̂ per classe con linea di soglia
                    with right_col:
                        try:
                            df_plot = pd.DataFrame({"p_hat": p_hat, "y_true": y_true})
                            df_plot["Classe"] = df_plot["y_true"].map({0: "Classe 0", 1: "Classe 1"})
                            figd = px.histogram(
                                df_plot, x="p_hat", color="Classe",
                                barmode="overlay", nbins=30, template="simple_white",
                                title="Probabilità stimate per classe"
                            )
                            figd.add_vline(x=thr, line_dash="dash")
                            figd.update_layout(xaxis_title="p̂", yaxis_title="Frequenza")
                            st.plotly_chart(figd, use_container_width=True)
                        except Exception:
                            st.info("Impossibile disegnare l’istogramma delle probabilità stimate.")

                    # Guida alla lettura dei grafici
                    with st.expander("ℹ️ Come interpretare i grafici", expanded=False):
                        st.markdown(
                            "- **Curva ROC**: step dalla coordinata (0,0) a (1,1). Ogni salto corrisponde a un valore "
                            "distinto della probabilità stimata p̂. La **diagonale** rappresenta un classificatore casuale; "
                            "quanto più la curva sta **sopra** la diagonale, tanto migliore è la discriminazione.\n"
                            "- **AUC**: area sotto la ROC (0.5 = casuale, 1 = perfetta).\n"
                            "- **Punto di soglia**: il marcatore indica la coppia (FPR, TPR) alla **soglia corrente**; "
                            "spostando la soglia ci si muove lungo la curva (più sensibilità ↔ meno specificità).\n"
                            "- **Distribuzione di p̂ per classe**: le due distribuzioni dovrebbero essere **separate** se il modello "
                            "discrimina bene. La **linea tratteggiata** è la soglia: per la **classe 1** l’area a destra della soglia "
                            "produce i **TP**, a sinistra i **FN**; per la **classe 0**, a destra i **FP** e a sinistra i **TN**.\n"
                            "- Elevata **sovrapposizione** tra le distribuzioni indica bassa discriminazione; valuti soglie alternative in base a **PPV/NPV** o ai **costi** degli errori."
                        )

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
                st.markdown(
                    "- **Log-odds / Odds Ratio (OR)**: i coefficienti sono su scala log-odds; `OR = exp(β)`.\n"
                    "- **p < 0.05**: evidenza che il coefficiente ≠ 0; consideri anche l’**ampiezza** (OR e CI).\n"
                    "- **McFadden R²**, **AIC/BIC**: confronto tra modelli; **AUC** valuta la discriminazione complessiva.\n"
                    f"- **Soglia {thr:.2f}**: determina il trade-off tra **sensibilità** e **specificità**.\n"
                    "- **VIF** elevato → possibile multicollinearità."
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
