# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np

from core.state import init_state

# Opzionali
try:
    from sklearn.metrics import (
        roc_curve, auc, confusion_matrix,
        precision_recall_curve, average_precision_score
    )
    _HAS_SKLEARN = True
except Exception:
    _HAS_SKLEARN = False

try:
    from scipy import stats as spstats
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

import plotly.graph_objects as go

# ===========================================================
# Utility comuni
# ===========================================================
def _use_active_df() -> pd.DataFrame:
    """Usa il dataset working se esiste, altrimenti l'originale df."""
    if "df_working" in st.session_state and st.session_state.df_working is not None:
        return st.session_state.df_working.copy()
    return st.session_state.df.copy()

def _is_binary(s: pd.Series) -> bool:
    vals = s.dropna().unique()
    return len(vals) == 2

def _confusion_from_binary(y_true: np.ndarray, y_pred: np.ndarray):
    """Restituisce TP, FP, TN, FN (positiva=1)."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    TP = int(((y_true == 1) & (y_pred == 1)).sum())
    TN = int(((y_true == 0) & (y_pred == 0)).sum())
    FP = int(((y_true == 0) & (y_pred == 1)).sum())
    FN = int(((y_true == 1) & (y_pred == 0)).sum())
    return TP, FP, TN, FN

def _safe_div(a, b):
    return float(a) / float(b) if b not in (0, 0.0) else np.nan

def _binom_ci(x, n, alpha=0.05):
    """
    IC 95% per proporzioni. Preferisce Clopper-Pearson (esatto, se SciPy),
    altrimenti Wilson approx; se nulla disponibile, ritorna NaN.
    """
    if n is None or n == 0:
        return (np.nan, np.nan)
    p = x / n
    if _HAS_SCIPY:
        # Clopper–Pearson (beta)
        lo = spstats.beta.ppf(alpha/2, x, n - x + 1) if x > 0 else 0.0
        hi = spstats.beta.ppf(1 - alpha/2, x + 1, n - x) if x < n else 1.0
        return (float(lo), float(hi))
    # Wilson score interval
    z = 1.959963984540054  # ~N(0,1) 97.5%
    denom = 1 + z**2 / n
    center = (p + z**2/(2*n)) / denom
    halfw = (z * np.sqrt((p*(1-p) + z**2/(4*n))/n)) / denom
    return (float(center - halfw), float(center + halfw))

def _metric_table(TP, FP, TN, FN, alpha=0.05):
    sens = _safe_div(TP, TP + FN)
    spec = _safe_div(TN, TN + FP)
    ppv  = _safe_div(TP, TP + FP)
    npv  = _safe_div(TN, TN + FN)
    acc  = _safe_div(TP + TN, TP + FP + TN + FN)
    lrp  = _safe_div(sens, 1 - spec) if spec is not np.nan else np.nan
    lrm  = _safe_div(1 - sens, spec) if spec is not np.nan else np.nan
    youden = sens + spec - 1

    sens_lo, sens_hi = _binom_ci(TP, TP + FN, alpha)
    spec_lo, spec_hi = _binom_ci(TN, TN + FP, alpha)
    ppv_lo,  ppv_hi  = _binom_ci(TP, TP + FP, alpha) if (TP + FP) > 0 else (np.nan, np.nan)
    npv_lo,  npv_hi  = _binom_ci(TN, TN + FN, alpha) if (TN + FN) > 0 else (np.nan, np.nan)
    acc_lo,  acc_hi  = _binom_ci(TP + TN, TP + FP + TN + FN, alpha)

    df = pd.DataFrame([
        {"Metrica": "Sensibilità", "Valore": sens, "CI 2.5%": sens_lo, "CI 97.5%": sens_hi},
        {"Metrica": "Specificità", "Valore": spec, "CI 2.5%": spec_lo, "CI 97.5%": spec_hi},
        {"Metrica": "PPV",         "Valore": ppv,  "CI 2.5%": ppv_lo,  "CI 97.5%": ppv_hi},
        {"Metrica": "NPV",         "Valore": npv,  "CI 2.5%": npv_lo,  "CI 97.5%": npv_hi},
        {"Metrica": "Accuratezza", "Valore": acc,  "CI 2.5%": acc_lo,  "CI 97.5%": acc_hi},
        {"Metrica": "LR+",         "Valore": lrp,  "CI 2.5%": np.nan,  "CI 97.5%": np.nan},
        {"Metrica": "LR−",         "Valore": lrm,  "CI 2.5%": np.nan,  "CI 97.5%": np.nan},
        {"Metrica": "Youden J",    "Valore": youden, "CI 2.5%": np.nan, "CI 97.5%": np.nan},
    ])
    return df.round(4)

def _roc_plot(y_true, y_score):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc_val = auc(fpr, tpr)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"ROC (AUC={auc_val:.3f})"))
    fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines", name="No skill", line=dict(dash="dash")))
    fig.update_layout(title="ROC curve", xaxis_title="FPR (1 - Specificità)", yaxis_title="TPR (Sensibilità)")
    return fig, auc_val

def _pr_plot(y_true, y_score):
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=recall, y=precision, mode="lines", name=f"PR (AP={ap:.3f})"))
    fig.update_layout(title="Precision–Recall curve", xaxis_title="Recall (Sensibilità)", yaxis_title="Precision (PPV)")
    return fig, ap

def _optimal_threshold(y_true, y_score):
    """Soglia che massimizza Youden J sulla ROC."""
    fpr, tpr, thr = roc_curve(y_true, y_score)
    j = tpr - fpr
    k = int(np.argmax(j))
    return float(thr[k]), float(tpr[k]), float(1 - fpr[k])

def _bayes_post_test(pre_prob, lr):
    """Dato pre_prob (0-1) e un LR restituisce la post_prob."""
    if pre_prob is None or not np.isfinite(pre_prob):
        return np.nan
    pre_prob = float(np.clip(pre_prob, 1e-12, 1-1e-12))
    pre_odds = pre_prob / (1 - pre_prob)
    post_odds = pre_odds * float(lr) if np.isfinite(lr) else np.nan
    return float(post_odds / (1 + post_odds)) if np.isfinite(post_odds) else np.nan

# ===========================================================
# Pagina
# ===========================================================
init_state()
st.title("🔬 Analisi Test Diagnostici")

# Check dataset
if "df" not in st.session_state or st.session_state.df is None:
    st.warning("Nessun dataset disponibile. Carichi i dati in **Step 0 — Upload Dataset**.")
    st.page_link("pages/0_📂_Upload_Dataset.py", label="➡️ Vai a Upload Dataset", icon="📂")
    st.stop()

df = _use_active_df()
if df is None or df.empty:
    st.error("Il dataset attivo è vuoto.")
    st.stop()

st.subheader("Selezione variabili")
with st.container():
    outcome = st.selectbox("Outcome (gold standard, binario: 0/1 o due categorie):", options=list(df.columns))
    test_var = st.selectbox("Variabile di test (predittore diagnostico):", options=[c for c in df.columns if c != outcome])

# Preparazione outcome binario
y_raw = df[outcome]
non_na_mask = ~y_raw.isna()
if not _is_binary(y_raw):
    st.error("L’outcome selezionato deve avere **esattamente due** livelli (dopo gestione dei missing).")
    st.stop()

levels = sorted(y_raw.dropna().unique().tolist(), key=lambda x: str(x))
pos_class = st.selectbox("Classe positiva dell’outcome (codificata come 1):", options=levels)
y = (y_raw == pos_class).astype(int)

# Test var e strategie
x = df[test_var]
valid_mask = non_na_mask & (~x.isna())
y_valid = y[valid_mask]
x_valid = x[valid_mask]

st.markdown("### Impostazioni del test")
is_binary_test = _is_binary(x_valid)

if is_binary_test:
    # Categoriale/binaria: scegliere quale livello considerare "test positivo"
    t_levels = sorted(x_valid.dropna().unique().tolist(), key=lambda z: str(z))
    test_pos = st.selectbox("Livello del TEST considerato positivo:", options=t_levels)
    with st.spinner("Calcolo la classificazione del test…"):
        y_pred = (x_valid == test_pos).astype(int).values
else:
    # Continuo: soglia + direzione
    st.info("La variabile di test è continua (o con più di 2 livelli). Selezioni una soglia e la direzione.")
    direction = st.radio("Direzione di positività del test:", ["≥ soglia = positivo", "≤ soglia = positivo"], horizontal=True)
    # Soglia iniziale: mediana
    x_min, x_max = float(np.nanmin(x_valid)), float(np.nanmax(x_valid))
    default_thr = float(np.nanmedian(x_valid))
    thr = st.slider("Soglia", min_value=x_min, max_value=x_max, value=default_thr, step=(x_max-x_min)/100 if x_max>x_min else 1.0)

    # Soglia ottimale (Youden J) se disponibile sklearn
    if _HAS_SKLEARN:
        if st.checkbox("Trova soglia ottimale (massimo Youden J)"):
            with st.spinner("Ottimizzo la soglia sul massimo Youden J…"):
                score = x_valid.astype(float).values
                best_thr, best_sens, best_spec = _optimal_threshold(y_valid.values, score)
                thr = best_thr
                st.success(f"Soglia ottimale = {thr:.6g} (Sens={best_sens:.3f}, Spec={best_spec:.3f})")

    with st.spinner("Applico la soglia per classificare il test…"):
        if direction == "≥ soglia = positivo":
            y_pred = (x_valid.astype(float).values >= thr).astype(int)
        else:
            y_pred = (x_valid.astype(float).values <= thr).astype(int)

# Confusion matrix
with st.spinner("Calcolo matrice di confusione e metriche…"):
    TP, FP, TN, FN = _confusion_from_binary(y_valid.values, y_pred)
    total = TP + FP + TN + FN
    prev  = (TP + FN) / total if total > 0 else np.nan

    cm_df = pd.DataFrame(
        [[TP, FP],
         [FN, TN]],
        index=["Vero 1 (Positivo)", "Vero 0 (Negativo)"],
        columns=["Pred 1 (Test +)", "Pred 0 (Test -)"]
    )

st.markdown("### Matrice di confusione")
st.dataframe(cm_df, use_container_width=True)

st.markdown("**Prevalenza osservata**: {:.3f}".format(prev) if np.isfinite(prev) else "**Prevalenza osservata**: n.d.")

# Metriche con IC
with st.spinner("Stimo sensibilità, specificità, PPV, NPV, LR e intervalli di confidenza…"):
    metrics_df = _metric_table(TP, FP, TN, FN, alpha=0.05)
st.dataframe(metrics_df, use_container_width=True)

# ROC/AUC (se possibile)
if _HAS_SKLEARN:
    st.markdown("### Analisi ROC")
    with st.spinner("Calcolo ROC/AUC…"):
        if is_binary_test:
            score = (x_valid == test_pos).astype(int).values
        else:
            if 'direction' in locals() and direction == "≤ soglia = positivo":
                score = (-x_valid.astype(float)).values
            else:
                score = x_valid.astype(float).values
        fig_roc, auc_val = _roc_plot(y_valid.values, score)
    st.plotly_chart(fig_roc, use_container_width=True)
    st.caption(f"**AUC**: {auc_val:.3f} — maggiore è l’AUC, migliore è la capacità discriminativa.")
else:
    st.info("Per la curva ROC e l’AUC è necessario `scikit-learn`.")
    auc_val = None

# Precision–Recall (AP) se possibile
ap_val = None
if _HAS_SKLEARN:
    st.markdown("### Curva Precision–Recall (PPV vs Sensibilità)")
    with st.spinner("Calcolo Precision–Recall e Average Precision…"):
        # score coerente con quanto usato per ROC
        if is_binary_test:
            score_pr = (x_valid == test_pos).astype(int).values
        else:
            score_pr = score  # già definito sopra
        fig_pr, ap_val = _pr_plot(y_valid.values, score_pr)
    st.plotly_chart(fig_pr, use_container_width=True)
    st.caption(f"**Average Precision (AP)**: {ap_val:.3f} — utile in presenza di **classi sbilanciate**; "
               "AP più alto indica migliore precisione media lungo i livelli di recall.")

# =========================
#   Analisi per sottogruppi
# =========================
st.markdown("## Analisi per sottogruppi (opzionale)")
with st.expander("Imposta stratificazione per sottogruppi", expanded=False):
    stratify = st.selectbox("Variabile di stratificazione (opzionale):", options=["— Nessuna —"] + [c for c in df.columns if c not in [outcome, test_var]])
    subgroup_table = None

    if stratify and stratify != "— Nessuna —":
        s = df.loc[valid_mask, stratify]
        # Se numerica con molti livelli, proponi binning
        is_numeric = pd.api.types.is_numeric_dtype(s)
        if is_numeric and s.dropna().nunique() > 10:
            method = st.radio("Binning per variabile numerica:", ["Quantili", "Nessuno (usa come continua)"], horizontal=True)
            if method == "Quantili":
                qn = st.slider("Numero di quantili (bin)", 2, 10, 4, step=1)
                with st.spinner("Creo sottogruppi (quantili)…"):
                    try:
                        s_binned = pd.qcut(s, q=qn, duplicates="drop")
                        s_used = s_binned.astype(str)
                    except Exception:
                        s_used = pd.cut(s, bins=qn).astype(str)
            else:
                # Se “continua”: per coerenza con analisi diagnostica binaria serve categorizzarla; usiamo 4 quantili di default
                with st.spinner("Variabile continua: applico quantili di default (4) per creare gruppi…"):
                    s_used = pd.qcut(s, q=4, duplicates="drop").astype(str)
        else:
            # Categoriale o numerica con pochi livelli
            with st.spinner("Preparo livelli di stratificazione…"):
                s_used = s.astype(str)

        levels_s = [lv for lv in pd.Series(s_used).dropna().unique().tolist()]
        levels_s = sorted(levels_s, key=lambda v: v)

        rows = []
        with st.spinner("Calcolo metriche per ciascun sottogruppo…"):
            for lv in levels_s:
                mask_g = (pd.Series(s_used).astype(str) == str(lv)).values
                yg = y_valid.values[mask_g]
                # Predizione del test: ricalcola in base alle stesse regole globali
                if is_binary_test:
                    ypg = (x_valid[mask_g] == test_pos).astype(int).values
                else:
                    xv = x_valid[mask_g].astype(float).values
                    if 'direction' in locals() and direction == "≤ soglia = positivo":
                        ypg = (xv <= thr).astype(int)
                    else:
                        ypg = (xv >= thr).astype(int)

                TPg, FPg, TNg, FNg = _confusion_from_binary(yg, ypg)
                mdf = _metric_table(TPg, FPg, TNg, FNg, alpha=0.05)
                sens = float(mdf.loc[mdf["Metrica"]=="Sensibilità","Valore"].values[0])
                spec = float(mdf.loc[mdf["Metrica"]=="Specificità","Valore"].values[0])
                ppv  = float(mdf.loc[mdf["Metrica"]=="PPV","Valore"].values[0])
                npv  = float(mdf.loc[mdf["Metrica"]=="NPV","Valore"].values[0])
                acc  = float(mdf.loc[mdf["Metrica"]=="Accuratezza","Valore"].values[0])
                lrp  = float(mdf.loc[mdf["Metrica"]=="LR+","Valore"].values[0])
                lrm  = float(mdf.loc[mdf["Metrica"]=="LR−","Valore"].values[0])

                rows.append({
                    "Sottogruppo": str(lv),
                    "N": int(len(yg)),
                    "Sensibilità": sens,
                    "Specificità": spec,
                    "PPV": ppv,
                    "NPV": npv,
                    "Accuratezza": acc,
                    "LR+": lrp,
                    "LR−": lrm
                })

        subgroup_table = pd.DataFrame(rows)
        st.markdown("### Risultati per sottogruppi")
        st.dataframe(subgroup_table.round(4), use_container_width=True)
        st.caption("Le metriche sono calcolate usando la medesima definizione di test positivo impostata sopra.")

# ================================
#   Probabilità post-test (Bayes)
# ================================
st.markdown("## Probabilità post-test (Bayes)")
with st.expander("Imposta la probabilità pre-test", expanded=False):
    pre_test = st.slider("Probabilità pre-test (stima clinica o prevalenza attesa)", 0.0, 1.0, float(prev) if np.isfinite(prev) else 0.2, step=0.01)

# estrai LR da metrics_df
try:
    lr_plus  = float(metrics_df.loc[metrics_df["Metrica"]=="LR+","Valore"].values[0])
    lr_minus = float(metrics_df.loc[metrics_df["Metrica"]=="LR−","Valore"].values[0])
except Exception:
    lr_plus, lr_minus = np.nan, np.nan

with st.spinner("Calcolo probabilità post-test…"):
    post_pos = _bayes_post_test(pre_test, lr_plus)
    post_neg = _bayes_post_test(pre_test, lr_minus)

post_tbl = pd.DataFrame([
    {"Scenario": "Test positivo", "LR": lr_plus,  "Probabilità post-test": post_pos},
    {"Scenario": "Test negativo", "LR": lr_minus, "Probabilità post-test": post_neg},
]).round(4)

st.dataframe(post_tbl, use_container_width=True)
st.caption("Aggiorno la probabilità pre-test con **LR+** per un test positivo e con **LR−** per un test negativo: "
           "[math] \\text{odds}_{post} = \\text{odds}_{pre} \\times LR [\\/math], poi "
           "[math] p_{post} = \\dfrac{\\text{odds}_{post}}{1+\\text{odds}_{post}} [\\/math].")

# Esportazione opzionale nel Results Summary
st.markdown("### Esporta nel Results Summary")
if st.button("➕ Aggiungi analisi diagnostica al Results Summary"):
    with st.spinner("Salvo i risultati nel Results Summary…"):
        if "report_items" not in st.session_state:
            st.session_state.report_items = []
        item = {
            "type": "diagnostic_test",
            "title": f"Analisi diagnostica — Outcome: {outcome} (positivo: {pos_class}), Test: {test_var}",
            "content": {
                "outcome_positive": str(pos_class),
                "test_definition": (
                    f"Test positivo: {str(test_pos)}" if is_binary_test
                    else f"Soglia: {float(thr) if 'thr' in locals() else None}, "
                         f"Direzione: {direction if 'direction' in locals() else None}"
                ),
                "confusion_matrix": {
                    "TP": TP, "FP": FP, "TN": TN, "FN": FN, "Total": TP+FP+TN+FN, "Prevalence": float(prev) if np.isfinite(prev) else None
                },
                "metrics": metrics_df.to_dict(orient="records"),
                "auc": float(auc_val) if auc_val is not None and np.isfinite(auc_val) else None,
                "average_precision": float(ap_val) if ap_val is not None and np.isfinite(ap_val) else None,
                "post_test": post_tbl.to_dict(orient="records")
            }
        }
        if 'subgroup_table' in locals() and isinstance(subgroup_table, pd.DataFrame) and not subgroup_table.empty:
            item["content"]["subgroups"] = subgroup_table.round(6).to_dict(orient="records")
        st.session_state.report_items.append(item)
    st.success("Analisi diagnostica aggiunta al Results Summary.")

with st.expander("ℹ️ Note metodologiche"):
    st.markdown("""
- **Sensibilità** = TP / (TP + FN) — probabilità che il test risulti positivo tra i malati.  
- **Specificità** = TN / (TN + FP) — probabilità che il test risulti negativo tra i sani.  
- **PPV** e **NPV** dipendono dalla **prevalenza**.  
- **LR+** = Sens / (1−Spec), **LR−** = (1−Sens) / Spec.  
- **Youden J** = Sens + Spec − 1; una soglia che massimizza J bilancia sensibilità e specificità.  
- **Curva Precision–Recall**: più informativa della ROC con **classi sbilanciate**; **AP** riassume l’area sotto tale curva.  
- **Aggiornamento bayesiano**: consente di stimare la **probabilità post-test** partendo da una **probabilità pre-test** clinica (o prevalenza attesa).  
""")
