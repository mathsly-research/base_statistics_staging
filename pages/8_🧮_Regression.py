# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np

from core.state import init_state

# Stats / Modeling
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.tools.tools import add_constant

# Optional: sklearn for ROC/AUC, confusion matrix, regularized logistic
try:
    from sklearn.metrics import (
        roc_curve, auc, confusion_matrix,
        precision_score, recall_score, f1_score, accuracy_score
    )
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.utils import resample
    _HAS_SKLEARN = True
except Exception:
    _HAS_SKLEARN = False

# Optional: SciPy for Shapiro & QQ theoretical quantiles
try:
    from scipy import stats as spstats
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

# Plotly
import plotly.express as px
import plotly.graph_objects as go

# -----------------------------
# Utility
# -----------------------------
def _use_active_df() -> pd.DataFrame:
    """Usa il dataset working se esiste, altrimenti l'originale df."""
    if "df_working" in st.session_state and st.session_state.df_working is not None:
        return st.session_state.df_working.copy()
    return st.session_state.df.copy()

def _make_design_matrix(df: pd.DataFrame, y: str, X_cols: list[str], dropna=True):
    """
    Crea y, X con dummies (drop_first=True), costante e cast a float.
    Rimuove le righe con NA se dropna=True.
    """
    X = df[X_cols].copy()
    X = pd.get_dummies(X, drop_first=True)          # codifica categoriche
    X = X.astype(float)                              # <- evita dtype object
    Xc = add_constant(X, has_constant="add")
    yv = df[y].copy()
    if dropna:
        data = pd.concat([yv, Xc], axis=1).dropna()
        yv = data[y].astype(float)                   # outcome numerico
        Xc = data.drop(columns=[y]).astype(float)    # predittori numerici
    return yv, Xc

def _is_binary_series(s: pd.Series) -> bool:
    vals = s.dropna().unique()
    return len(vals) == 2

def _compute_vif(X: pd.DataFrame) -> pd.DataFrame:
    """Calcola VIF escludendo la costante."""
    Xn = X.drop(columns=["const"], errors="ignore").copy()
    cols = list(Xn.columns)
    if len(cols) == 0:
        return pd.DataFrame({"feature": [], "VIF": []})
    vif = []
    for i, c in enumerate(cols):
        try:
            v = variance_inflation_factor(Xn.values, i)
        except Exception:
            v = np.nan
        vif.append({"feature": c, "VIF": v})
    return pd.DataFrame(vif).sort_values("VIF", ascending=False)

def _qq_plot(residuals: np.ndarray):
    """Restituisce una figura Plotly del Q-Q plot se SciPy è disponibile; altrimenti None."""
    if not _HAS_SCIPY or residuals is None or len(residuals) < 3:
        return None
    osm, osr = spstats.probplot(residuals, dist="norm", sparams=())
    theo = np.array(osm[0], dtype=float)
    samp = np.array(osr, dtype=float)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=theo, y=samp, mode="markers", name="Residui"))
    minv = float(np.nanmin([np.nanmin(theo), np.nanmin(samp)]))
    maxv = float(np.nanmax([np.nanmax(theo), np.nanmax(samp)]))
    fig.add_trace(go.Scatter(x=[minv, maxv], y=[minv, maxv], mode="lines", name="45°", line=dict(dash="dash")))
    fig.update_layout(title="Q-Q plot dei residui", xaxis_title="Quantili teorici", yaxis_title="Quantili residui")
    return fig

# -----------------------------
# Bootstrap per CI metriche (stratificato)
# -----------------------------
def _bootstrap_ci_metric(y_true: np.ndarray, y_pred: np.ndarray, func, n_boot: int = 500, alpha: float = 0.05, seed: int = 42):
    """
    CI bootstrap stratificato per una metrica di classificazione.
    y_true, y_pred: array binari (0/1)
    func: callable(y_true, y_pred) -> float
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    boot = []
    for _ in range(n_boot):
        idx = resample(np.arange(n), replace=True, stratify=y_true, random_state=int(rng.integers(1e9)))
        boot.append(func(y_true[idx], y_pred[idx]))
    lo, hi = np.percentile(boot, [100*alpha/2.0, 100*(1-alpha/2.0)])
    return float(lo), float(hi)

def _metrics_with_ci(y_true: np.ndarray, y_pred: np.ndarray, n_boot: int = 500, alpha: float = 0.05):
    """Restituisce DataFrame con Value e CI 95% per Precision, Recall, F1, Accuracy (bootstrap stratificato)."""
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    acc  = accuracy_score(y_true, y_pred)
    # Spinner dedicato al bootstrap (potrebbe richiedere tempo)
    with st.spinner("Calcolo degli intervalli di confidenza (bootstrap stratificato)…"):
        p_lo, p_hi = _bootstrap_ci_metric(y_true, y_pred, lambda yt, yp: precision_score(yt, yp, zero_division=0), n_boot=n_boot, alpha=alpha)
        r_lo, r_hi = _bootstrap_ci_metric(y_true, y_pred, lambda yt, yp: recall_score(yt, yp, zero_division=0), n_boot=n_boot, alpha=alpha)
        f_lo, f_hi = _bootstrap_ci_metric(y_true, y_pred, lambda yt, yp: f1_score(yt, yp, zero_division=0), n_boot=n_boot, alpha=alpha)
        a_lo, a_hi = _bootstrap_ci_metric(y_true, y_pred, lambda yt, yp: accuracy_score(yt, yp), n_boot=n_boot, alpha=alpha)

    df = pd.DataFrame([
        {"Metric": "Precision", "Value": prec, "CI 2.5%": p_lo, "CI 97.5%": p_hi},
        {"Metric": "Recall (Sensibilità)", "Value": rec, "CI 2.5%": r_lo, "CI 97.5%": r_hi},
        {"Metric": "F1", "Value": f1, "CI 2.5%": f_lo, "CI 97.5%": f_hi},
        {"Metric": "Accuracy", "Value": acc, "CI 2.5%": a_lo, "CI 97.5%": a_hi},
    ])
    return df

# -----------------------------
# Init & checks
# -----------------------------
init_state()
st.title("🧮 Step 8 — Regressione lineare, logistica e Poisson")

if "df" not in st.session_state or st.session_state.df is None:
    st.warning("Nessun dataset disponibile. Carichi i dati in **Step 0 — Upload Dataset**.")
    st.page_link("pages/0_📂_Upload_Dataset.py", label="➡️ Vai a Upload Dataset", icon="📂")
    st.stop()

df = _use_active_df()
if df is None or df.empty:
    st.error("Il dataset attivo è vuoto.")
    st.stop()

# -----------------------------
# Selettori base
# -----------------------------
st.subheader("Selezione modello")
target = st.selectbox("Variabile di outcome:", options=list(df.columns))

model_type = st.radio(
    "Tipo di regressione:",
    options=["Lineare (OLS)", "Logistica (binaria)", "Logistica regolarizzata (L1/L2)", "Poisson (conteggi)"],
    horizontal=True
)

with st.expander("ℹ️ Quale modello scegliere?", expanded=False):
    st.markdown("""
- **Lineare (OLS)** → outcome **continuo** (es. pressione, peso); assunzioni: linearità, normalità dei residui, omoscedasticità.  
- **Logistica (binaria)** → outcome **0/1** o due categorie (es. evento sì/no); output come **Odds Ratio**.  
- **Logistica regolarizzata (L1/L2)** → come la logistica, con **penalizzazione** per ridurre overfitting/collinearità.  
- **Poisson (GLM)** → outcome = **conteggio** di eventi (0,1,2,...) con varianza ≈ media; output come **IRR** (Incidence Rate Ratio).  
""")

# Vincoli specifici dei modelli
if model_type in ("Logistica (binaria)", "Logistica regolarizzata (L1/L2)"):
    if not _is_binary_series(df[target]) and df[target].dropna().nunique() > 2:
        st.error("Per la regressione logistica l’outcome deve essere **binario** (esattamente due livelli).")
        st.stop()

if model_type == "Lineare (OLS)":
    # outcome deve essere numerico
    if not pd.api.types.is_numeric_dtype(df[target]):
        st.error("Per la regressione lineare l'outcome deve essere **numerico**.")
        st.stop()

if model_type == "Poisson (conteggi)":
    # outcome deve essere intero (conteggi)
    if not pd.api.types.is_integer_dtype(df[target]):
        st.error("Per la regressione di Poisson l'outcome deve essere **numerico intero (conteggi)**.")
        st.stop()

# predittori
candidate_preds = [c for c in df.columns if c != target]
X_sel = st.multiselect("Seleziona i predittori (può scegliere più variabili):", candidate_preds)
if not X_sel:
    st.info("Selezioni almeno un predittore per stimare il modello.")
    st.stop()

# ===========================================================
#                     LINEARE (OLS)
# ===========================================================
if model_type == "Lineare (OLS)":
    with st.spinner("Preparo la matrice del modello…"):
        y, X = _make_design_matrix(df, target, X_sel, dropna=True)
    if X.shape[0] < 5 or X.shape[1] < 2:
        st.error("Dati insufficienti per stimare il modello (pochi casi o nessun predittore dopo la codifica).")
        st.stop()

    # OLS con errori standard robusti (HC3)
    with st.spinner("Stimo il modello OLS con errori standard robusti (HC3)…"):
        model = sm.OLS(y, X).fit(cov_type="HC3")

    st.subheader("Risultati modello (OLS con SE robusti HC3)")
    info = {
        "N": int(model.nobs),
        "R²": float(model.rsquared),
        "R² adj.": float(model.rsquared_adj),
        "AIC": float(model.aic),
        "BIC": float(model.bic)
    }
    st.write(pd.DataFrame(info, index=["Valore"]).T)
    st.caption("**Come leggere — Metriche globali (OLS):** **N**=numero osservazioni; **R²**=quota di varianza spiegata; **R² adj.** penalizza la complessità; **AIC/BIC** per confronto modelli (più basso è meglio).")

    # Coefficienti con CI e p-value (robusti)
    with st.spinner("Calcolo coefficienti, intervalli di confidenza e p-value…"):
        params = model.params
        conf = model.conf_int()
        pvals = model.pvalues
        df_coefs = pd.DataFrame({
            "term": params.index,
            "coef": params.values,
            "CI 2.5%": conf[0].values,
            "CI 97.5%": conf[1].values,
            "p-value": pvals.values
        })
    st.markdown("**Coefficienti (IC95% e p-value)**")
    st.dataframe(df_coefs.round(4), use_container_width=True)
    st.caption("**Come leggere — Tabella coefficienti (OLS):** ogni **coef** è la variazione media dell’outcome per +1 del predittore (a parità degli altri). Se l’**IC95%** non include 0, l’effetto è significativo (coerente col p-value).")

    # Diagnostica affiancata
    st.subheader("Diagnostica modello")
    resid = model.resid.values
    fitted = model.fittedvalues.values

    col1, col2 = st.columns(2)

    with col1:
        with st.spinner("Genero grafico Residui vs Fitted…"):
            fig1 = px.scatter(x=fitted, y=resid, labels={"x": "Fitted", "y": "Residui"}, title="Residui vs Fitted")
            fig1.add_hline(y=0, line_dash="dash")
            st.plotly_chart(fig1, use_container_width=True)
        st.caption("**Come leggere — Residui vs Fitted:** punti casuali attorno a 0 senza pattern → assunzioni plausibili; ventaglio/curve → possibili violazioni (non linearità/eteroscedasticità).")

    with col2:
        with st.spinner("Genero Q-Q plot dei residui…"):
            figqq = _qq_plot(resid)
        if figqq is not None:
            st.plotly_chart(figqq, use_container_width=True)
            st.caption("**Come leggere — Q-Q residui:** punti sulla diagonale → normalità dei residui; deviazioni sistematiche → non normalità.")

    # Shapiro-Wilk residui (se SciPy)
    if _HAS_SCIPY and len(resid) >= 3:
        try:
            with st.spinner("Eseguo test di normalità (Shapiro–Wilk) sui residui…"):
                W, p_sh = spstats.shapiro(resid if len(resid) <= 5000 else resid[:5000])
            st.write(f"Shapiro–Wilk residui: W={W:.3f}, p={p_sh:.3g}  "
                     f"{'→ compatibile con normalità ✅' if p_sh>=0.05 else '→ devia da normalità ❌'}")
        except Exception:
            pass

    # Breusch–Pagan per omoscedasticità
    try:
        with st.spinner("Eseguo test di omoscedasticità (Breusch–Pagan)…"):
            bp_stat, bp_p, _, _ = het_breuschpagan(resid, X)
        st.write(f"Breusch–Pagan: stat={bp_stat:.3f}, p={bp_p:.3g}  "
                 f"{'→ omoscedasticità compatibile ✅' if bp_p>=0.05 else '→ eteroscedasticità sospetta ❌'}")
    except Exception:
        pass

    # VIF per multicollinearità
    st.markdown("**Multicollinearità (VIF)**")
    with st.spinner("Calcolo VIF (potrebbe richiedere alcuni secondi con molte variabili)…"):
        vif_df = _compute_vif(X)
    st.dataframe(vif_df.round(3), use_container_width=True)
    st.caption("**Come leggere — VIF:** ≈1 nessuna collinearità; >5 moderata; >10 problematica. Collinearità alta rende instabili i coefficienti.")

    # ➕ Salva nel Results Summary
    if st.button("➕ Aggiungi risultati OLS al Results Summary"):
        with st.spinner("Salvo i risultati nel Results Summary…"):
            if "report_items" not in st.session_state:
                st.session_state.report_items = []
            st.session_state.report_items.append({
                "type": "regression_ols",
                "title": f"Regressione OLS — {target}",
                "content": {
                    "nobs": int(model.nobs),
                    "r2": float(model.rsquared),
                    "r2_adj": float(model.rsquared_adj),
                    "aic": float(model.aic),
                    "bic": float(model.bic),
                    "coefficients": df_coefs.round(6).to_dict(orient="records"),
                    "vif": vif_df.round(6).to_dict(orient="records")
                }
            })
        st.success("Modello OLS aggiunto al Results Summary.")

    with st.expander("ℹ️ Approfondimento — Analisi dei residui"):
        st.markdown("""
I **residui** sono differenze tra osservato e predetto. Se il modello è adeguato:
- pattern **non strutturati** → **linearità** plausibile;
- **varianza** circa costante → **omoscedasticità**;
- **distribuzione** circa normale → inferenza classica più affidabile.
Violazioni suggeriscono trasformazioni, termini non lineari (es. polinomiali/spline) o modelli alternativi.
""")

# ===========================================================
#                 LOGISTICA (BINARIA)
# ===========================================================
elif model_type == "Logistica (binaria)":
    y_raw = df[target]
    unique_vals = sorted(y_raw.dropna().unique().tolist(), key=lambda x: str(x))
    if len(unique_vals) != 2:
        st.error("L’outcome per la logistica deve avere **esattamente due** livelli (dopo gestione NA).")
        st.stop()

    positive_class = st.selectbox("Classe positiva (codificata come 1):", options=unique_vals)
    y_bin = (y_raw == positive_class).astype(int)

    # design matrix
    with st.spinner("Preparo la matrice del modello…"):
        y, X = _make_design_matrix(df.assign(__y__=y_bin), "__y__", X_sel, dropna=True)
    if X.shape[0] < 10 or X.shape[1] < 2:
        st.error("Dati insufficienti per stimare il modello (pochi casi utili o nessun predittore dopo la codifica).")
        st.stop()

    # statsmodels Logit + (tentativo) covarianze robuste
    try:
        with st.spinner("Stimo il modello logit…"):
            logit = sm.Logit(y, X).fit(disp=False)
        try:
            with st.spinner("Calcolo errori standard robusti (HC3)…"):
                logit = logit.get_robustcov_results(cov_type="HC3")
        except Exception:
            pass
    except Exception as e:
        st.error(f"Impossibile stimare il modello logit: {e}")
        st.stop()

    st.subheader("Risultati modello (Logistica binaria)")

    # Coefficienti → Odds Ratio con IC95%
    with st.spinner("Calcolo Odds Ratio, IC95% e p-value…"):
        params = logit.params
        conf = logit.conf_int()
        or_ = np.exp(params)
        or_lo = np.exp(conf[0])
        or_hi = np.exp(conf[1])
        pvals = logit.pvalues

        df_or = pd.DataFrame({
            "term": params.index,
            "OR": or_.values,
            "CI 2.5%": or_lo.values,
            "CI 97.5%": or_hi.values,
            "p-value": pvals.values
        })
    st.markdown("**Odds Ratio (IC95%) e p-value**")
    st.dataframe(df_or.round(4), use_container_width=True)
    st.caption(f"**Come leggere — Tabella coefficienti (Logistica):** l’**OR** è l’effetto moltiplicativo sul rapporto di odds della classe positiva (**{positive_class}**). OR>1 ↑ probabilità relativa, OR<1 ↓. Se l’IC95% non include 1, l’effetto è significativo.")

    # Pseudo-R² (McFadden)
    llf = float(getattr(logit, "llf", np.nan))
    llnull = float(getattr(logit, "llnull", np.nan)) if hasattr(logit, "llnull") else np.nan
    pseudo_r2 = 1 - (llf / llnull) if np.isfinite(llf) and np.isfinite(llnull) and llnull != 0 else np.nan

    info = {
        "N": int(getattr(logit, "nobs", len(y))),
        "LogLik (modello)": llf,
        "LogLik (null)": llnull,
        "Pseudo-R² (McFadden)": float(pseudo_r2) if np.isfinite(pseudo_r2) else np.nan,
        "AIC": float(getattr(logit, "aic", np.nan)),
        "BIC": float(getattr(logit, "bic", np.nan)),
    }
    st.write(pd.DataFrame(info, index=["Valore"]).T)
    st.caption("**Come leggere — Metriche globali (Logit):** **N**=osservazioni; **LogLik**=verosimiglianza; **Pseudo-R²** misura adattamento (0=nullo; ~0.2–0.4 spesso accettabile); **AIC/BIC** per confronto modelli.")

    # Valutazione predittiva — grafici affiancati
    st.subheader("Valutazione predittiva")
    with st.spinner("Calcolo probabilità predette…"):
        try:
            y_pred_prob = logit.predict(X)
        except Exception:
            # fallback per alcuni oggetti robusti
            y_pred_prob = 1 / (1 + np.exp(-(X @ params)))

    thresh = st.slider("Soglia di classificazione", 0.05, 0.95, 0.50, step=0.01)
    y_pred = (y_pred_prob >= thresh).astype(int)

    if _HAS_SKLEARN:
        col1, col2 = st.columns(2)

        with col1:
            with st.spinner("Traccio curva ROC e calcolo AUC…"):
                fpr, tpr, _ = roc_curve(y, y_pred_prob)
                auc_val = auc(fpr, tpr)
                figroc = go.Figure()
                figroc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"ROC (AUC={auc_val:.3f})"))
                figroc.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines", name="No skill", line=dict(dash="dash")))
                figroc.update_layout(title="ROC curve", xaxis_title="FPR", yaxis_title="TPR")
                st.plotly_chart(figroc, use_container_width=True)
            st.caption("**Come leggere — ROC curve:** **TPR** (=Recall=TP/(TP+FN)) vs **FPR** (=FP/(FP+TN)). Più la curva si allontana dalla diagonale, migliore la discriminazione. **AUC**: 0.5 casuale; >0.7 accettabile; >0.8 buona; >0.9 eccellente.")

        with col2:
            with st.spinner("Genero confusion matrix…"):
                cm = confusion_matrix(y, y_pred, labels=[1,0])
                figcm = go.Figure(data=go.Heatmap(
                    z=cm, x=["Pred 1","Pred 0"], y=["True 1","True 0"],
                    text=cm, texttemplate="%{text}", colorscale="Blues"))
                figcm.update_layout(title="Confusion matrix", xaxis_title="", yaxis_title="")
                st.plotly_chart(figcm, use_container_width=True)
            st.caption("**Acronimi:** **TP**=Vero Positivo, **TN**=Vero Negativo, **FP**=Falso Positivo, **FN**=Falso Negativo. Regolando la **soglia** bilancia il trade-off tra FP e FN.")

        with st.spinner("Calcolo metriche di classificazione con IC bootstrap…"):
            metrics_df = _metrics_with_ci(y.values if isinstance(y, pd.Series) else y, y_pred, n_boot=500, alpha=0.05).round(3)
        st.dataframe(metrics_df, use_container_width=True)
        st.caption("**Metriche di classificazione:** **Precision**=TP/(TP+FP); **Recall/Sensibilità (TPR)**=TP/(TP+FN); **Accuracy**=(TP+TN)/Totale; **F1**=media armonica di Precision e Recall. **CI 95%** con **bootstrap stratificato (n=500)**.")
    else:
        st.info("Per ROC/AUC, confusion matrix e metriche avanzate installare `scikit-learn`.")
        st.write(pd.DataFrame({"Accuracy":[(y==y_pred).mean()]}).round(3))

    # ➕ Salva nel Results Summary
    if st.button("➕ Aggiungi risultati Logit al Results Summary"):
        with st.spinner("Salvo i risultati nel Results Summary…"):
            if "report_items" not in st.session_state:
                st.session_state.report_items = []
            st.session_state.report_items.append({
                "type": "regression_logit",
                "title": f"Regressione logistica — {target} (positiva: {positive_class})",
                "content": {
                    "nobs": int(getattr(logit, "nobs", len(y))),
                    "loglik": llf,
                    "loglik_null": llnull,
                    "pseudo_r2_mcfadden": float(pseudo_r2) if np.isfinite(pseudo_r2) else None,
                    "aic": float(getattr(logit, "aic", np.nan)),
                    "bic": float(getattr(logit, "bic", np.nan)),
                    "odds_ratios": df_or.round(6).to_dict(orient="records")
                }
            })
        st.success("Modello logit aggiunto al Results Summary.")

# ===========================================================
#          LOGISTICA REGOLARIZZATA (L1/L2)
# ===========================================================
elif model_type == "Logistica regolarizzata (L1/L2)":
    if not _HAS_SKLEARN:
        st.error("Per la logistica regolarizzata è necessario `scikit-learn`.")
        st.stop()

    # outcome binario
    y_raw = df[target]
    unique_vals = sorted(y_raw.dropna().unique().tolist(), key=lambda x: str(x))
    if len(unique_vals) != 2:
        st.error("L’outcome per la logistica regolarizzata deve avere **esattamente due** livelli (dopo gestione NA).")
        st.stop()

    positive_class = st.selectbox("Classe positiva (codificata come 1):", options=unique_vals)
    y_bin = (y_raw == positive_class).astype(int)

    penalty = st.radio("Tipo di penalizzazione:", ["l1", "l2"], horizontal=True)
    C_val = st.slider("Forza regolarizzazione (C)", 0.001, 10.0, 1.0, step=0.01)
    solver = "liblinear" if penalty == "l1" else "lbfgs"
    max_iter = st.number_input("Max iterazioni", 50, 5000, 1000, step=50)

    with st.spinner("Preparo la matrice del modello…"):
        y, X = _make_design_matrix(df.assign(__y__=y_bin), "__y__", X_sel, dropna=True)
    if X.shape[0] < 10 or X.shape[1] < 2:
        st.error("Dati insufficienti per stimare il modello (pochi casi utili o nessun predittore dopo la codifica).")
        st.stop()

    with st.spinner("Stimo la logistica regolarizzata…"):
        pipe = make_pipeline(
            StandardScaler(with_mean=False),  # compatibile con sparse/dummies
            LogisticRegression(penalty=penalty, C=C_val, solver=solver, max_iter=int(max_iter))
        )
        pipe.fit(X, y)

    st.subheader("Risultati modello (Logistica regolarizzata)")
    with st.spinner("Estraggo e trasformo i coefficienti…"):
        lr = pipe.named_steps["logisticregression"]
        coefs = lr.coef_[0]
        ORs = np.exp(coefs)
        coef_df = pd.DataFrame({"term": X.columns, "coef": coefs, "OR (exp coef)": ORs}).round(4)
    st.dataframe(coef_df, use_container_width=True)
    st.caption("**Come leggere — Coefficienti penalizzati:** la penalizzazione riduce overfitting/collinearità. Con **L1** alcuni coefficienti possono diventare 0 (selezione di variabili).")

    # Prestazioni — grafici affiancati
    st.subheader("Valutazione predittiva")
    with st.spinner("Calcolo probabilità e classi predette…"):
        y_pred_prob = pipe.predict_proba(X)[:, 1]
    thresh = st.slider("Soglia di classificazione", 0.05, 0.95, 0.50, step=0.01, key="thresh_reg")
    y_pred = (y_pred_prob >= thresh).astype(int)

    if _HAS_SKLEARN:
        col1, col2 = st.columns(2)

        with col1:
            with st.spinner("Traccio curva ROC e calcolo AUC…"):
                fpr, tpr, _ = roc_curve(y, y_pred_prob)
                auc_val = auc(fpr, tpr)
                figroc = go.Figure()
                figroc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"ROC (AUC={auc_val:.3f})"))
                figroc.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines", name="No skill", line=dict(dash="dash")))
                figroc.update_layout(title="ROC curve", xaxis_title="FPR", yaxis_title="TPR")
                st.plotly_chart(figroc, use_container_width=True)
            st.caption("**Come leggere — ROC curve:** TPR (Recall) vs FPR; l’**AUC** riassume la discriminazione del modello su tutte le soglie.")

        with col2:
            with st.spinner("Genero confusion matrix…"):
                cm = confusion_matrix(y, y_pred, labels=[1,0])
                figcm = go.Figure(data=go.Heatmap(
                    z=cm, x=["Pred 1","Pred 0"], y=["True 1","True 0"],
                    text=cm, texttemplate="%{text}", colorscale="Blues"))
                figcm.update_layout(title="Confusion matrix", xaxis_title="", yaxis_title="")
                st.plotly_chart(figcm, use_container_width=True)
            st.caption("**Acronimi:** **TP**=Vero Positivo, **TN**=Vero Negativo, **FP**=Falso Positivo, **FN**=Falso Negativo. **TPR**=TP/(TP+FN); **FPR**=FP/(FP+TN).")

        with st.spinner("Calcolo metriche di classificazione con IC bootstrap…"):
            metrics_df = _metrics_with_ci(y.values if isinstance(y, pd.Series) else y, y_pred, n_boot=500, alpha=0.05).round(3)
        st.dataframe(metrics_df, use_container_width=True)
        st.caption("**Metriche di classificazione:** **Precision**=TP/(TP+FP); **Recall/Sensibilità (TPR)**=TP/(TP+FN); **Accuracy**=(TP+TN)/Totale; **F1**=media armonica di Precision e Recall. **CI 95%** con bootstrap stratificato (n=500).")
    else:
        st.info("Per ROC/AUC e confusion matrix installare `scikit-learn`.")
        st.write(pd.DataFrame({"Accuracy":[(y==y_pred).mean()]}).round(3))

    # ➕ Salva nel Results Summary
    if st.button("➕ Aggiungi risultati Logistica regolarizzata al Results Summary"):
        with st.spinner("Salvo i risultati nel Results Summary…"):
            if "report_items" not in st.session_state:
                st.session_state.report_items = []
            st.session_state.report_items.append({
                "type": "regression_logit_regularized",
                "title": f"Logistica regolarizzata — {target} (positiva: {positive_class})",
                "content": {
                    "penalty": penalty,
                    "C": float(C_val),
                    "coefficients": coef_df.to_dict(orient="records")
                }
            })
        st.success("Modello logit regolarizzato aggiunto al Results Summary.")

# ===========================================================
#                     POISSON (GLM)
# ===========================================================
else:  # "Poisson (conteggi)"
    # y intero (conteggi) già verificato sopra
    with st.spinner("Preparo la matrice del modello…"):
        y, X = _make_design_matrix(df, target, X_sel, dropna=True)
    if X.shape[0] < 10 or X.shape[1] < 2:
        st.error("Dati insufficienti per stimare il modello di Poisson.")
        st.stop()

    try:
        with st.spinner("Stimo il modello GLM di Poisson…"):
            model = sm.GLM(y, X, family=sm.families.Poisson()).fit()
    except Exception as e:
        st.error(f"Errore nella stima del modello di Poisson: {e}")
        st.stop()

    st.subheader("Risultati modello (GLM Poisson)")

    # Coefficienti → IRR con IC95%
    with st.spinner("Calcolo IRR, IC95% e p-value…"):
        params = model.params
        conf = model.conf_int()
        irr = np.exp(params)
        irr_lo = np.exp(conf[0])
        irr_hi = np.exp(conf[1])
        pvals = model.pvalues

        df_irr = pd.DataFrame({
            "term": params.index,
            "IRR": irr.values,
            "CI 2.5%": irr_lo.values,
            "CI 97.5%": irr_hi.values,
            "p-value": pvals.values
        })
    st.markdown("**Incidence Rate Ratio (IRR, IC95%) e p-value**")
    st.dataframe(df_irr.round(4), use_container_width=True)
    st.caption("**Come leggere — Tabella coefficienti (Poisson):** l’**IRR** è l’effetto moltiplicativo sul **tasso atteso di eventi**. IRR>1 → aumento; IRR<1 → diminuzione. Se l’IC95% non include 1, l’effetto è significativo.")

    # Info generali
    info = {
        "N": int(model.nobs),
        "Deviance": float(model.deviance),
        "Pearson Chi2": float(model.pearson_chi2),
        "AIC": float(model.aic)
    }
    st.write(pd.DataFrame(info, index=["Valore"]).T)
    st.caption("**Come leggere — Metriche globali (Poisson):** **Deviance** e **Pearson Chi2** misurano l’adattamento; valori molto elevati suggeriscono **overdispersione** (considerare quasi-Poisson / Negative Binomial). **AIC** per confronto modelli.")

    # ➕ Salva nel Results Summary
    if st.button("➕ Aggiungi risultati Poisson al Results Summary"):
        with st.spinner("Salvo i risultati nel Results Summary…"):
            if "report_items" not in st.session_state:
                st.session_state.report_items = []
            st.session_state.report_items.append({
                "type": "regression_poisson",
                "title": f"Regressione Poisson — {target}",
                "content": {
                    "nobs": int(model.nobs),
                    "deviance": float(model.deviance),
                    "pearson_chi2": float(model.pearson_chi2),
                    "aic": float(model.aic),
                    "irr": df_irr.round(6).to_dict(orient="records")
                }
            })
        st.success("Modello di Poisson aggiunto al Results Summary.")

    with st.expander("ℹ️ Approfondimento — Poisson e overdispersione"):
        st.markdown("""
La regressione di **Poisson** assume varianza ≈ media dei conteggi. Se la **varianza >> media** (overdispersione),
i SE possono essere sottostimati: considerare **quasi-Poisson** o **Negative Binomial** e l’uso di **offset** per tassi (es. log del tempo a rischio).
""")
