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
    from sklearn.metrics import roc_curve, auc, confusion_matrix, precision_recall_fscore_support
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
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
    y, X = _make_design_matrix(df, target, X_sel, dropna=True)
    if X.shape[0] < 5 or X.shape[1] < 2:
        st.error("Dati insufficienti per stimare il modello (pochi casi o nessun predittore dopo la codifica).")
        st.stop()

    # OLS con errori standard robusti (HC3)
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

    # Coefficienti con CI e p-value (robusti)
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

    # Diagnostica affiancata
    st.subheader("Diagnostica modello")
    resid = model.resid.values
    fitted = model.fittedvalues.values

    col1, col2 = st.columns(2)

    with col1:
        fig1 = px.scatter(x=fitted, y=resid, labels={"x": "Fitted", "y": "Residui"}, title="Residui vs Fitted")
        fig1.add_hline(y=0, line_dash="dash")
        st.plotly_chart(fig1, use_container_width=True)
        st.caption("**Come leggere — Residui vs Fitted:** i punti dovrebbero distribuirsi casualmente attorno a 0 senza pattern. "
                   "Strutture a ventaglio o curve indicano possibili violazioni (non linearità/eteroscedasticità).")

    with col2:
        figqq = _qq_plot(resid)
        if figqq is not None:
            st.plotly_chart(figqq, use_container_width=True)
            st.caption("**Come leggere — Q-Q plot dei residui:** i punti dovrebbero allinearsi alla diagonale. "
                       "Deviazioni sistematiche indicano non normalità dei residui.")

    # Shapiro-Wilk residui (se SciPy)
    if _HAS_SCIPY and len(resid) >= 3:
        try:
            W, p_sh = spstats.shapiro(resid if len(resid) <= 5000 else resid[:5000])
            st.write(f"Shapiro–Wilk residui: W={W:.3f}, p={p_sh:.3g}  "
                     f"{'→ compatibile con normalità ✅' if p_sh>=0.05 else '→ devia da normalità ❌'}")
        except Exception:
            pass

    # Breusch–Pagan per omoscedasticità
    try:
        bp_stat, bp_p, _, _ = het_breuschpagan(resid, X)
        st.write(f"Breusch–Pagan: stat={bp_stat:.3f}, p={bp_p:.3g}  "
                 f"{'→ omoscedasticità compatibile ✅' if bp_p>=0.05 else '→ eteroscedasticità sospetta ❌'}")
    except Exception:
        pass

    # VIF per multicollinearità
    st.markdown("**Multicollinearità (VIF)**")
    vif_df = _compute_vif(X)
    st.dataframe(vif_df.round(3), use_container_width=True)
    st.caption("**Come leggere — VIF:** ≈1 nessuna collinearità; >5 moderata; >10 problematica. "
               "Collinearità elevata rende instabili i coefficienti.")

    # ➕ Salva nel Results Summary
    if st.button("➕ Aggiungi risultati OLS al Results Summary"):
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

    # Guida interpretativa (riepilogo)
    with st.expander("ℹ️ Come leggere i risultati (OLS)", expanded=False):
        st.markdown("""
- **R² / R² adj.**: quota di varianza spiegata (R² adj. penalizza modelli troppo complessi).  
- **Coefficienti**: effetto marginale; IC95% e p-value quantificano l’incertezza.  
- **Residui vs Fitted**: pattern ≠ casuale → non linearità o eteroscedasticità.  
- **Q-Q residui**: deviazioni dalla diagonale → non normalità.  
- **Breusch–Pagan**: p<0.05 → eteroscedasticità (qui SE robusti HC3 già applicati).  
- **VIF**: collinearità alta inflaziona varianze dei coefficienti.
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
    y, X = _make_design_matrix(df.assign(__y__=y_bin), "__y__", X_sel, dropna=True)
    if X.shape[0] < 10 or X.shape[1] < 2:
        st.error("Dati insufficienti per stimare il modello (pochi casi utili o nessun predittore dopo la codifica).")
        st.stop()

    # statsmodels Logit + (tentativo) covarianze robuste
    try:
        logit = sm.Logit(y, X).fit(disp=False)
        try:
            logit = logit.get_robustcov_results(cov_type="HC3")
        except Exception:
            pass
    except Exception as e:
        st.error(f"Impossibile stimare il modello logit: {e}")
        st.stop()

    st.subheader("Risultati modello (Logistica binaria)")

    # Coefficienti → Odds Ratio con IC95%
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

    # Valutazione predittiva — grafici affiancati
    st.subheader("Valutazione predittiva")
    try:
        y_pred_prob = logit.predict(X)
    except Exception:
        # fallback per alcuni oggetti robusti
        y_pred_prob = 1 / (1 + np.exp(-(X @ params)))

    thresh = st.slider("Soglia di classificazione", 0.05, 0.95, 0.50, step=0.01)
    y_pred = (y_pred_prob >= thresh).astype(int)

    if _HAS_SKLEARN:
        fpr, tpr, _ = roc_curve(y, y_pred_prob)
        auc_val = auc(fpr, tpr)

        col1, col2 = st.columns(2)

        with col1:
            figroc = go.Figure()
            figroc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"ROC (AUC={auc_val:.3f})"))
            figroc.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines", name="No skill", line=dict(dash="dash")))
            figroc.update_layout(title="ROC curve", xaxis_title="FPR", yaxis_title="TPR")
            st.plotly_chart(figroc, use_container_width=True)
            st.caption("**Come leggere — ROC curve:** più la curva si allontana dalla diagonale, migliore la discriminazione. "
                       "L’AUC riassume l’accuratezza indipendente dalla soglia (0.5 casuale; >0.7 accettabile; >0.8 buona; >0.9 eccellente).")

        with col2:
            cm = confusion_matrix(y, y_pred, labels=[1,0])
            figcm = go.Figure(data=go.Heatmap(
                z=cm, x=["Pred 1","Pred 0"], y=["True 1","True 0"],
                text=cm, texttemplate="%{text}", colorscale="Blues"))
            figcm.update_layout(title="Confusion matrix", xaxis_title="", yaxis_title="")
            st.plotly_chart(figcm, use_container_width=True)
            st.caption("**Come leggere — Confusion matrix:** TP (pred 1 | true 1), TN (pred 0 | true 0). "
                       "FP e FN quantificano errori; osservare Precision, Recall e F1 per bilanciare la soglia.")

        prec, rec, f1, _ = precision_recall_fscore_support(y, y_pred, average="binary", zero_division=0)
        st.write(pd.DataFrame({"Precision": [prec], "Recall": [rec], "F1": [f1], "Accuracy":[(y==y_pred).mean()]}).round(3))
    else:
        st.info("Per ROC/AUC e confusion matrix installare `scikit-learn`.")
        st.write(pd.DataFrame({"Accuracy":[(y==y_pred).mean()]}).round(3))

    # ➕ Salva nel Results Summary
    if st.button("➕ Aggiungi risultati Logit al Results Summary"):
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

    # Guida interpretativa
    with st.expander("ℹ️ Come leggere i risultati (Logistica)", expanded=False):
        st.markdown(f"""
- **Odds Ratio (OR)**: effetto moltiplicativo sul **rapporto di odds** della classe positiva (**{positive_class}**).  
  OR>1 ↑ probabilità relativa, OR<1 ↓. IC95% che **non** include 1 → effetto significativo.  
- **Pseudo-R² (McFadden)**: bontà di adattamento (0 = nullo; ~0.2–0.4 spesso “buono”).  
- **ROC/AUC**: accuratezza indipendente dalla soglia; più la curva è lontana dalla diagonale, meglio è.  
- **Confusion matrix / Precision / Recall / F1**: dipendono dalla **soglia** scelta.
""")

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

    y, X = _make_design_matrix(df.assign(__y__=y_bin), "__y__", X_sel, dropna=True)
    if X.shape[0] < 10 or X.shape[1] < 2:
        st.error("Dati insufficienti per stimare il modello (pochi casi utili o nessun predittore dopo la codifica).")
        st.stop()

    pipe = make_pipeline(
        StandardScaler(with_mean=False),  # compatibile con sparse/dummies
        LogisticRegression(penalty=penalty, C=C_val, solver=solver, max_iter=int(max_iter))
    )
    pipe.fit(X, y)

    st.subheader("Risultati modello (Logistica regolarizzata)")
    lr = pipe.named_steps["logisticregression"]
    coefs = lr.coef_[0]
    ORs = np.exp(coefs)
    coef_df = pd.DataFrame({"term": X.columns, "coef": coefs, "OR (exp coef)": ORs}).round(4)
    st.dataframe(coef_df, use_container_width=True)
    st.caption("**Come leggere — Coefficienti penalizzati:** la penalizzazione riduce overfitting e collinearità; "
               "i coefficienti piccoli possono essere spinti verso 0 (soprattutto con L1).")

    # Prestazioni — grafici affiancati
    st.subheader("Valutazione predittiva")
    y_pred_prob = pipe.predict_proba(X)[:, 1]
    thresh = st.slider("Soglia di classificazione", 0.05, 0.95, 0.50, step=0.01, key="thresh_reg")
    y_pred = (y_pred_prob >= thresh).astype(int)

    if _HAS_SKLEARN:
        fpr, tpr, _ = roc_curve(y, y_pred_prob)
        auc_val = auc(fpr, tpr)

        col1, col2 = st.columns(2)

        with col1:
            figroc = go.Figure()
            figroc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"ROC (AUC={auc_val:.3f})"))
            figroc.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines", name="No skill", line=dict(dash="dash")))
            figroc.update_layout(title="ROC curve", xaxis_title="FPR", yaxis_title="TPR")
            st.plotly_chart(figroc, use_container_width=True)
            st.caption("**Come leggere — ROC curve:** l’AUC riassume la discriminazione del modello su tutte le soglie.")

        with col2:
            cm = confusion_matrix(y, y_pred, labels=[1,0])
            figcm = go.Figure(data=go.Heatmap(
                z=cm, x=["Pred 1","Pred 0"], y=["True 1","True 0"],
                text=cm, texttemplate="%{text}", colorscale="Blues"))
            figcm.update_layout(title="Confusion matrix", xaxis_title="", yaxis_title="")
            st.plotly_chart(figcm, use_container_width=True)
            st.caption("**Come leggere — Confusion matrix:** regoli la soglia per bilanciare FP e FN; "
                       "osservi l’impatto su Precision, Recall e F1.")

        prec, rec, f1, _ = precision_recall_fscore_support(y, y_pred, average="binary", zero_division=0)
        st.write(pd.DataFrame({"Precision": [prec], "Recall": [rec], "F1": [f1], "Accuracy":[(y==y_pred).mean()]}).round(3))
    else:
        st.info("Per ROC/AUC e confusion matrix installare `scikit-learn`.")
        st.write(pd.DataFrame({"Accuracy":[(y==y_pred).mean()]}).round(3))

    # ➕ Salva nel Results Summary
    if st.button("➕ Aggiungi risultati Logistica regolarizzata al Results Summary"):
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
    y, X = _make_design_matrix(df, target, X_sel, dropna=True)
    if X.shape[0] < 10 or X.shape[1] < 2:
        st.error("Dati insufficienti per stimare il modello di Poisson.")
        st.stop()

    try:
        model = sm.GLM(y, X, family=sm.families.Poisson()).fit()
    except Exception as e:
        st.error(f"Errore nella stima del modello di Poisson: {e}")
        st.stop()

    st.subheader("Risultati modello (GLM Poisson)")

    # Coefficienti → IRR con IC95%
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

    # Info generali
    info = {
        "N": int(model.nobs),
        "Deviance": float(model.deviance),
        "Pearson Chi2": float(model.pearson_chi2),
        "AIC": float(model.aic)
    }
    st.write(pd.DataFrame(info, index=["Valore"]).T)

    # ➕ Salva nel Results Summary
    if st.button("➕ Aggiungi risultati Poisson al Results Summary"):
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

    # Guida interpretativa
    with st.expander("ℹ️ Come leggere i risultati (Poisson)", expanded=False):
        st.markdown("""
- **IRR (Incidence Rate Ratio)**: effetto moltiplicativo sul **tasso atteso** di eventi.  
  - **IRR > 1** → aumento del numero atteso di eventi; **IRR < 1** → diminuzione.  
  - Se l’**IC95%** non include 1, l’effetto è statisticamente significativo.  
- **Deviance** / **Pearson Chi2**: misure di bontà di adattamento; valori molto elevati possono indicare **overdispersione** (valutare quasi-Poisson o Negative Binomial).  
- **AIC**: confronto tra modelli (più basso = migliore compromesso qualità/complessità).
""")
