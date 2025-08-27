# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np

from core.state import init_state

# Stats / Modeling
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.tools.tools import add_constant

# Optional: sklearn for ROC/AUC & confusion matrix
try:
    from sklearn.metrics import roc_curve, auc, confusion_matrix, precision_recall_fscore_support
    _HAS_SKLEARN = True
except Exception:
    _HAS_SKLEARN = False

# Optional: SciPy for Shapiro & QQ theoretical quantiles (fallback se non presente)
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
    if "df_working" in st.session_state and st.session_state.df_working is not None:
        return st.session_state.df_working.copy()
    return st.session_state.df.copy()

def _make_design_matrix(df: pd.DataFrame, y: str, X_cols: list[str], dropna=True):
    X = df[X_cols].copy()
    # one-hot encode categoriche
    X = pd.get_dummies(X, drop_first=True)
    # aggiungi costante
    Xc = add_constant(X, has_constant="add")
    yv = df[y].copy()
    if dropna:
        data = pd.concat([yv, Xc], axis=1).dropna()
        yv = data[y]
        Xc = data.drop(columns=[y])
    return yv, Xc

def _is_binary_series(s: pd.Series) -> bool:
    vals = s.dropna().unique()
    return len(vals) == 2

def _compute_vif(X: pd.DataFrame) -> pd.DataFrame:
    # X deve includere la costante; la escludiamo dal calcolo VIF
    cols = [c for c in X.columns if c != "const"]
    if len(cols) < 2:
        return pd.DataFrame({"feature": cols, "VIF": [np.nan]*len(cols)})
    vif = []
    for i, c in enumerate(cols):
        try:
            v = variance_inflation_factor(X[cols].values, i)
        except Exception:
            v = np.nan
        vif.append({"feature": c, "VIF": v})
    return pd.DataFrame(vif).sort_values("VIF", ascending=False)

def _qq_plot_data(residuals: np.ndarray):
    # restituisce quantili teorici e campionari per Q-Q plot
    if _HAS_SCIPY:
        osm, osr = spstats.probplot(residuals, dist="norm", sparams=())
        theo = np.array(osm[0], dtype=float)
        samp = np.array(osr, dtype=float)
    else:
        # fallback semplice: quantili empirici vs quantili normali
        r = np.sort((residuals - np.mean(residuals)) / (np.std(residuals, ddof=1) or 1.0))
        n = len(r)
        probs = (np.arange(1, n+1) - 0.5) / n
        theo = spstats.norm.ppf(probs) if _HAS_SCIPY else np.sqrt(2) * erfinv(2*probs-1)
        samp = r
    return theo, samp

# -----------------------------
# Init & checks
# -----------------------------
init_state()
st.title("🧮 Step 8 — Regressione lineare e logistica")

if "df" not in st.session_state or st.session_state.df is None:
    st.warning("Nessun dataset disponibile. Carichi i dati in Step 0 — Upload Dataset.")
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
    options=["Lineare (OLS)", "Logistica (binaria)"],
    horizontal=True
)

# vincoli per logistica
if model_type == "Logistica (binaria)":
    if not _is_binary_series(df[target]) and df[target].dropna().nunique() > 2:
        st.error("L'outcome selezionato ha più di due livelli. Selezionare una variabile binaria.")
        st.stop()

# predittori
candidate_preds = [c for c in df.columns if c != target]
X_sel = st.multiselect("Seleziona i predittori (può scegliere più variabili):", candidate_preds)

if not X_sel:
    st.info("Selezioni almeno un predittore per stimare il modello.")
    st.stop()

# -----------------------------
# Regressione LINEARE (OLS)
# -----------------------------
if model_type == "Lineare (OLS)":
    # y deve essere numerica
    if not pd.api.types.is_numeric_dtype(df[target]):
        st.error("Per la regressione lineare l'outcome deve essere **numerico**.")
        st.stop()

    y, X = _make_design_matrix(df, target, X_sel, dropna=True)

    if X.shape[0] < 5 or X.shape[1] < 2:
        st.error("Dati insufficienti per stimare il modello (pochi casi o nessun predittore dopo la codifica).")
        st.stop()

    model = sm.OLS(y, X).fit()

    st.subheader("Risultati modello (OLS)")
    # Sommario principale (sintetico)
    info = {
        "N": int(model.nobs),
        "R²": float(model.rsquared),
        "R² adj.": float(model.rsquared_adj),
        "AIC": float(model.aic),
        "BIC": float(model.bic)
    }
    st.write(pd.DataFrame(info, index=["Valore"]).T)

    # Coefficienti con CI e p-value
    params = model.params
    conf = model.conf_int()
    pvals = model.pvalues
    df_coefs = pd.DataFrame({
        "coef": params,
        "CI 2.5%": conf[0],
        "CI 97.5%": conf[1],
        "p-value": pvals
    }).rename_axis("term").reset_index()
    st.markdown("**Coefficienti (IC95% e p-value)**")
    st.dataframe(df_coefs.round(4), use_container_width=True)

    # Diagnostica
    st.subheader("Diagnostica modello")
    resid = model.resid.values
    fitted = model.fittedvalues.values

    # Residui vs Fitted
    fig1 = px.scatter(x=fitted, y=resid, labels={"x":"Fitted", "y":"Residui"},
                      title="Residui vs Fitted")
    fig1.add_hline(y=0, line_dash="dash")
    st.plotly_chart(fig1, use_container_width=True)

    # Q-Q plot dei residui
    theo, samp = _qq_plot_data(resid)
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=theo, y=samp, mode="markers", name="Residui"))
    # linea 45°
    minv = float(np.nanmin([np.nanmin(theo), np.nanmin(samp)]))
    maxv = float(np.nanmax([np.nanmax(theo), np.nanmax(samp)]))
    fig2.add_trace(go.Scatter(x=[minv, maxv], y=[minv, maxv], mode="lines", name="45°"))
    fig2.update_layout(title="Q-Q plot dei residui", xaxis_title="Quantili teorici", yaxis_title="Quantili residui")
    st.plotly_chart(fig2, use_container_width=True)

    # Shapiro-Wilk sui residui (se disponibile)
    if _HAS_SCIPY and len(resid) >= 3:
        try:
            W, p_sh = spstats.shapiro(resid if len(resid) <= 5000 else resid[:5000])
            st.write(f"Shapiro–Wilk sui residui: W={W:.3f}, p={p_sh:.3g}  "
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
    st.caption("Interpretazione VIF: ≈1 nessuna collinearità; >5 moderata; >10 problematica.")

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

    # Guida interpretativa
    with st.expander("ℹ️ Come leggere i risultati (OLS)", expanded=False):
        st.markdown("""
- **R² / R² adj.**: quota di varianza dell’outcome spiegata dai predittori (R² adj. penalizza i modelli troppo complessi).  
- **Coefficienti**: effetto marginale medio (aggiustato) del predittore sull’outcome; IC95% e p-value valutano l’incertezza.  
- **Residui vs Fitted**: pattern non casuali suggeriscono violazioni (non linearità, eteroscedasticità).  
- **Q-Q plot residui**: deviazioni marcate dalla linea indicano non normalità.  
- **Breusch–Pagan**: p<0.05 → eteroscedasticità (valutare robust standard errors).  
- **VIF**: collinearità alta inflaziona le varianze stimate dei coefficienti.
""")

# -----------------------------
# Regressione LOGISTICA (binaria)
# -----------------------------
else:
    # outcome: binario; se testuale, scegli "classe positiva"
    y_raw = df[target]
    unique_vals = sorted(y_raw.dropna().unique().tolist(), key=lambda x: str(x))
    if len(unique_vals) != 2:
        st.error("Per la regressione logistica l’outcome deve avere **esattamente due** livelli (dopo gestione NA).")
        st.stop()

    positive_class = st.selectbox("Seleziona la **classe positiva** (codificata come 1):", options=unique_vals)
    y_bin = (y_raw == positive_class).astype(int)

    # design matrix
    y, X = _make_design_matrix(df.assign(__y__=y_bin), "__y__", X_sel, dropna=True)

    if X.shape[0] < 10 or X.shape[1] < 2:
        st.error("Dati insufficienti per stimare il modello (pochi casi utili o nessun predittore dopo la codifica).")
        st.stop()

    # stima Logit (statsmodels)
    try:
        logit = sm.Logit(y, X).fit(disp=False)
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
    llf = float(logit.llf)
    llnull = float(logit.llnull) if hasattr(logit, "llnull") else np.nan
    pseudo_r2 = 1 - (llf / llnull) if (llnull is not np.nan and llnull != 0) else np.nan

    info = {
        "N": int(logit.nobs),
        "LogLik (modello)": llf,
        "LogLik (null)": llnull,
        "Pseudo-R² (McFadden)": float(pseudo_r2) if np.isfinite(pseudo_r2) else np.nan,
        "AIC": float(logit.aic) if hasattr(logit, "aic") else np.nan,
        "BIC": float(logit.bic) if hasattr(logit, "bic") else np.nan,
    }
    st.write(pd.DataFrame(info, index=["Valore"]).T)

    # Valutazione predittiva (se sklearn disponibile)
    st.subheader("Valutazione predittiva")
    y_pred_prob = logit.predict(X)
    thresh = st.slider("Soglia di classificazione", 0.05, 0.95, 0.50, step=0.01)
    y_pred = (y_pred_prob >= thresh).astype(int)

    if _HAS_SKLEARN:
        # ROC curve
        fpr, tpr, _ = roc_curve(y, y_pred_prob)
        auc_val = auc(fpr, tpr)
        figroc = go.Figure()
        figroc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"ROC (AUC={auc_val:.3f})"))
        figroc.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines", name="No skill", line=dict(dash="dash")))
        figroc.update_layout(title="ROC curve", xaxis_title="FPR", yaxis_title="TPR")
        st.plotly_chart(figroc, use_container_width=True)

        # Confusion matrix
        cm = confusion_matrix(y, y_pred, labels=[1,0])
        figcm = go.Figure(data=go.Heatmap(
            z=cm, x=["Pred 1","Pred 0"], y=["True 1","True 0"],
            text=cm, texttemplate="%{text}", colorscale="Blues"))
        figcm.update_layout(title="Confusion matrix", xaxis_title="", yaxis_title="")
        st.plotly_chart(figcm, use_container_width=True)

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
                "nobs": int(logit.nobs),
                "loglik": llf,
                "loglik_null": llnull,
                "pseudo_r2_mcfadden": float(pseudo_r2) if np.isfinite(pseudo_r2) else None,
                "aic": float(logit.aic) if hasattr(logit, "aic") else None,
                "bic": float(logit.bic) if hasattr(logit, "bic") else None,
                "odds_ratios": df_or.round(6).to_dict(orient="records")
            }
        })
        st.success("Modello logit aggiunto al Results Summary.")

    # Guida interpretativa
    with st.expander("ℹ️ Come leggere i risultati (Logistica)", expanded=False):
        st.markdown(f"""
- **Odds Ratio (OR)**: effetto moltiplicativo sul **rapporto di odds** della classe positiva (**{positive_class}**).
  - OR > 1 aumenta la probabilità relativa della classe positiva; OR < 1 la riduce.
  - IC95% che **non** include 1 → effetto statisticamente significativo.
- **Pseudo-R² (McFadden)**: misura di bontà di adattamento (0 = modello nullo; valori più alti = meglio, tipicamente 0.2–0.4 “buono”).
- **ROC/AUC**: capacità discriminante indipendente dalla soglia (AUC=0.5 casuale; 0.7–0.8 accettabile; 0.8–0.9 buono; >0.9 eccellente).
- **Confusion matrix / Precision / Recall / F1**: valutazione dipendente dalla **soglia** scelta.
""")
