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
    """IC 95% per proporzioni: Clopper–Pearson (SciPy) o Wilson."""
    if n is None or n == 0:
        return (np.nan, np.nan)
    p = x / n
    if _HAS_SCIPY:
        lo = spstats.beta.ppf(alpha/2, x, n - x + 1) if x > 0 else 0.0
        hi = spstats.beta.ppf(1 - alpha/2, x + 1, n - x) if x < n else 1.0
        return (float(lo), float(hi))
    z = 1.959963984540054
    denom = 1 + z**2 / n
    center = (p + z**2/(2*n)) / denom
    halfw = (z * np.sqrt((p*(1-p) + z**2/(4*n))/n)) / denom
    return (float(center - halfw), float(center + halfw))

def _lr_ci(tp, fp, tn, fn, alpha=0.05):
    """
    IC 95% per LR+ e LR− con metodo log-trasformato (Simel et al.).
    log(LR+) ± z * SE_logLR+,  SE_logLR+ = sqrt(1/TP - 1/(TP+FN) + 1/FP - 1/(FP+TN))
    log(LR−) ± z * SE_logLR−,  SE_logLR− = sqrt(1/FN - 1/(TP+FN) + 1/TN - 1/(FP+TN))
    Gestisce zero mediante NaN.
    """
    z = 1.959963984540054
    # Sensibilità e specificità
    sens = _safe_div(tp, tp + fn)
    spec = _safe_div(tn, tn + fp)
    lr_plus = _safe_div(sens, 1 - spec) if np.isfinite(sens) and np.isfinite(spec) and spec != 1 else np.nan
    lr_minus = _safe_div(1 - sens, spec) if np.isfinite(sens) and np.isfinite(spec) and spec != 0 else np.nan

    # SE per log LR
    def _safe_sum_terms(terms):
        if any(t is None for t in terms):
            return np.nan
        return np.sum(terms)

    try:
        se_log_lr_plus = np.sqrt(
            _safe_sum_terms([
                _safe_div(1, tp) if tp>0 else np.nan,
                -_safe_div(1, (tp + fn)) if (tp+fn)>0 else np.nan,
                _safe_div(1, fp) if fp>0 else np.nan,
                -_safe_div(1, (fp + tn)) if (fp+tn)>0 else np.nan
            ])
        )
    except Exception:
        se_log_lr_plus = np.nan

    try:
        se_log_lr_minus = np.sqrt(
            _safe_sum_terms([
                _safe_div(1, fn) if fn>0 else np.nan,
                -_safe_div(1, (tp + fn)) if (tp+fn)>0 else np.nan,
                _safe_div(1, tn) if tn>0 else np.nan,
                -_safe_div(1, (fp + tn)) if (fp+tn)>0 else np.nan
            ])
        )
    except Exception:
        se_log_lr_minus = np.nan

    def _ci_from_log(lr, se):
        if not (np.isfinite(lr) and np.isfinite(se)):
            return (np.nan, np.nan, lr)
        lo = np.exp(np.log(lr) - z * se)
        hi = np.exp(np.log(lr) + z * se)
        return (float(lo), float(hi), float(lr))

    lo_p, hi_p, lr_p = _ci_from_log(lr_plus, se_log_lr_plus)
    lo_m, hi_m, lr_m = _ci_from_log(lr_minus, se_log_lr_minus)
    return (lr_p, lo_p, hi_p, lr_m, lo_m, hi_m)

def _metric_table(TP, FP, TN, FN, alpha=0.05):
    """Tabella metrica con IC per proporzioni + IC per LR (log-transform)."""
    sens = _safe_div(TP, TP + FN)
    spec = _safe_div(TN, TN + FP)
    ppv  = _safe_div(TP, TP + FP)
    npv  = _safe_div(TN, TN + FN)
    acc  = _safe_div(TP + TN, TP + FP + TN + FN)
    youden = sens + spec - 1

    sens_lo, sens_hi = _binom_ci(TP, TP + FN, alpha)
    spec_lo, spec_hi = _binom_ci(TN, TN + FP, alpha)
    ppv_lo,  ppv_hi  = _binom_ci(TP, TP + FP, alpha) if (TP + FP) > 0 else (np.nan, np.nan)
    npv_lo,  npv_hi  = _binom_ci(TN, TN + FN, alpha) if (TN + FN) > 0 else (np.nan, np.nan)
    acc_lo,  acc_hi  = _binom_ci(TP + TN, TP + FP + TN + FN, alpha)

    lr_p, lr_p_lo, lr_p_hi, lr_m, lr_m_lo, lr_m_hi = _lr_ci(TP, FP, TN, FN, alpha)

    df = pd.DataFrame([
        {"Metrica": "Sensibilità", "Valore": sens, "CI 2.5%": sens_lo, "CI 97.5%": sens_hi},
        {"Metrica": "Specificità", "Valore": spec, "CI 2.5%": spec_lo, "CI 97.5%": spec_hi},
        {"Metrica": "PPV",         "Valore": ppv,  "CI 2.5%": ppv_lo,  "CI 97.5%": ppv_hi},
        {"Metrica": "NPV",         "Valore": npv,  "CI 2.5%": npv_lo,  "CI 97.5%": npv_hi},
        {"Metrica": "Accuratezza", "Valore": acc,  "CI 2.5%": acc_lo,  "CI 97.5%": acc_hi},
        {"Metrica": "LR+",         "Valore": lr_p, "CI 2.5%": lr_p_lo, "CI 97.5%": lr_p_hi},
        {"Metrica": "LR−",         "Valore": lr_m, "CI 2.5%": lr_m_lo, "CI 97.5%": lr_m_hi},
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

def _dca_compute(y_true, scores, thresholds):
    """
    Decision Curve Analysis per un set di soglie p_t (0-1).
    Net Benefit (NB) del Test: NB = TP/N - FP/N * (p_t / (1 - p_t)).
    'Tratta tutti': NB_all = prevalenza - (1 - prevalenza) * (p_t / (1 - p_t)).
    'Tratta nessuno': NB_none = 0.
    scores: punteggio continuo (maggiore = più 'positivo').
    """
    y_true = np.asarray(y_true).astype(int)
    N = len(y_true)
    prev = y_true.mean() if N>0 else np.nan
    rows = []
    for pt in thresholds:
        if not (0 < pt < 1):  # evita 0 e 1
            continue
        thr = np.quantile(scores, 1-pt) if np.isfinite(np.nanmean(scores)) else None
        # Classifica in base alla soglia sullo score (percentile 1-pt per coerenza)
        y_pred = (scores >= thr).astype(int) if thr is not None else np.zeros_like(y_true)
        TP, FP, TN, FN = _confusion_from_binary(y_true, y_pred)
        TPn = TP / N if N>0 else np.nan
        FPn = FP / N if N>0 else np.nan
        w = pt / (1 - pt)
        nb_test = TPn - FPn * w
        nb_all  = prev - (1 - prev) * w if np.isfinite(prev) else np.nan
        nb_none = 0.0
        rows.append({"threshold": float(pt), "NB_test": float(nb_test), "NB_all": float(nb_all), "NB_none": float(nb_none)})
    return pd.DataFrame(rows)

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

# Outcome binario
y_raw = df[outcome]
non_na_mask = ~y_raw.isna()
if not _is_binary(y_raw):
    st.error("L’outcome selezionato deve avere **esattamente due** livelli (dopo gestione dei missing).")
    st.stop()

levels = sorted(y_raw.dropna().unique().tolist(), key=lambda x: str(x))
pos_class = st.selectbox("Classe positiva dell’outcome (codificata come 1):", options=levels)
y = (y_raw == pos_class).astype(int)

# Test var
x = df[test_var]
valid_mask = non_na_mask & (~x.isna())
y_valid = y[valid_mask]
x_valid = x[valid_mask]

st.markdown("### Impostazioni del test")
is_binary_test = _is_binary(x_valid)

if is_binary_test:
    t_levels = sorted(x_valid.dropna().unique().tolist(), key=lambda z: str(z))
    test_pos = st.selectbox("Livello del TEST considerato positivo:", options=t_levels)
    with st.spinner("Calcolo la classificazione del test…"):
        y_pred = (x_valid == test_pos).astype(int).values
else:
    st.info("La variabile di test è continua (o con più di 2 livelli). Selezioni una soglia e la direzione.")
    direction = st.radio("Direzione di positività del test:", ["≥ soglia = positivo", "≤ soglia = positivo"], horizontal=True)
    x_min, x_max = float(np.nanmin(x_valid)), float(np.nanmax(x_valid))
    default_thr = float(np.nanmedian(x_valid))
    thr = st.slider("Soglia", min_value=x_min, max_value=x_max, value=default_thr, step=(x_max-x_min)/100 if x_max>x_min else 1.0)

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

# Confusion matrix e metriche
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
st.caption(
    "Come leggere: **TP** (veri positivi) e **FN** (falsi negativi) appartengono ai casi realmente positivi; "
    "**TN** (veri negativi) e **FP** (falsi positivi) ai realmente negativi. La riga indica lo **stato reale**, la colonna l’**esito del test**."
)

st.markdown("**Prevalenza osservata**: {:.3f}".format(prev) if np.isfinite(prev) else "**Prevalenza osservata**: n.d.")

with st.spinner("Stimo sensibilità, specificità, PPV, NPV, LR (con IC) e altre metriche…"):
    metrics_df = _metric_table(TP, FP, TN, FN, alpha=0.05)
st.dataframe(metrics_df, use_container_width=True)
st.caption(
    "Interpretazione rapida: **Sensibilità** = quota di malati che risultano positivi; **Specificità** = quota di sani che risultano negativi. "
    "**PPV/NPV** dipendono dalla **prevalenza**. **LR+** (con IC) indica quanto un **test positivo** aumenti l’evidenza di malattia; "
    "**LR−** (con IC) indica quanto un **test negativo** la riduca."
)

# =========================
#   Grafici affiancati
# =========================
left, right = st.columns(2)

auc_val = None
ap_val = None
score_for_plots = None

if _HAS_SKLEARN:
    # score continuo coerente (maggiore => più 'positivo')
    if is_binary_test:
        score_for_plots = (x_valid == test_pos).astype(int).values
    else:
        score_for_plots = x_valid.astype(float).values
        if 'direction' in locals() and direction == "≤ soglia = positivo":
            score_for_plots = -score_for_plots

    with left:
        with st.spinner("Calcolo e traccio ROC/AUC…"):
            fig_roc, auc_val = _roc_plot(y_valid.values, score_for_plots)
            st.plotly_chart(fig_roc, use_container_width=True)
        st.caption(
            "Curva **ROC**: mostra il compromesso tra **Sensibilità (TPR)** e **1−Specificità (FPR)** variando la soglia. "
            "Un’**AUC** più alta indica migliore capacità discriminativa (0.5 = casuale; >0.8 buona)."
        )

    with right:
        with st.spinner("Calcolo e traccio Precision–Recall (AP)…"):
            fig_pr, ap_val = _pr_plot(y_valid.values, score_for_plots)
            st.plotly_chart(fig_pr, use_container_width=True)
        st.caption(
            "Curva **Precision–Recall**: **Precision (PPV)** vs **Recall (Sensibilità)**. "
            "È spesso più informativa della ROC quando la classe positiva è rara. **AP** riassume l’area sotto questa curva."
        )
else:
    st.info("Per le curve ROC e Precision–Recall è necessario `scikit-learn`.")

# ================================
#   Probabilità post-test (Bayes)
# ================================
st.markdown("## Probabilità post-test (Bayes)")
with st.expander("Imposta la probabilità pre-test", expanded=False):
    pre_default = float(prev) if np.isfinite(prev) else 0.2
    pre_test = st.slider("Probabilità pre-test (stima clinica o prevalenza attesa)", 0.0, 1.0, pre_default, step=0.01)

# LR da tabella
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

st.latex(r"\text{odds}_{post} = \text{odds}_{pre} \times LR")
st.latex(r"p_{post} = \frac{\text{odds}_{post}}{1+\text{odds}_{post}}")
st.caption(
    "Guida: imposti una **probabilità pre-test**. Con un **test positivo**, usa **LR+** per aggiornare la probabilità; "
    "con un **test negativo**, usa **LR−**. Esempio: pre-test 0.30 ⇒ odds=0.43; con LR+=5 ⇒ odds_post=2.15 ⇒ p_post≈0.68."
)

# =========================
#   Decision Curve Analysis
# =========================
st.markdown("## Decision Curve Analysis (DCA)")
if _HAS_SKLEARN and score_for_plots is not None and len(score_for_plots) == len(y_valid):
    with st.spinner("Calcolo Net Benefit su un range di soglie…"):
        # Soglie uniformi in (0,1), escludendo estremi
        thresholds = np.linspace(0.01, 0.99, 99)
        dca_df = _dca_compute(y_valid.values, score_for_plots, thresholds)

    with st.spinner("Genero il grafico DCA…"):
        fig_dca = go.Figure()
        fig_dca.add_trace(go.Scatter(x=dca_df["threshold"], y=dca_df["NB_test"], mode="lines", name="Test"))
        fig_dca.add_trace(go.Scatter(x=dca_df["threshold"], y=dca_df["NB_all"],  mode="lines", name="Tratta tutti"))
        fig_dca.add_trace(go.Scatter(x=dca_df["threshold"], y=dca_df["NB_none"], mode="lines", name="Tratta nessuno"))
        fig_dca.update_layout(
            title="Decision Curve Analysis — Net Benefit",
            xaxis_title="Soglia di decisione (p_t)",
            yaxis_title="Net Benefit"
        )
        st.plotly_chart(fig_dca, use_container_width=True)

    st.caption(
        "L’asse **x** è la **soglia di decisione** \(p_t\), cioè la probabilità a partire dalla quale si decide di trattare. "
        "Il **Net Benefit (NB)** confronta il test con due strategie estreme: **Tratta tutti** e **Tratta nessuno**. "
        "Dove la curva del **Test** sta **sopra** le altre, l’uso del test **porta beneficio clinico** rispetto alle alternative."
    )
else:
    st.info("Per la DCA è necessario uno **score continuo** (o prob.) e `scikit-learn`.")

# =========================
#   Analisi per sottogruppi
# =========================
st.markdown("## Analisi per sottogruppi (opzionale)")
with st.expander("Imposta stratificazione per sottogruppi", expanded=False):
    stratify = st.selectbox("Variabile di stratificazione (opzionale):", options=["— Nessuna —"] + [c for c in df.columns if c not in [outcome, test_var]])
    subgroup_table = None

    if stratify and stratify != "— Nessuna —":
        s = df.loc[valid_mask, stratify]
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
                with st.spinner("Variabile continua: applico quantili di default (4) per creare gruppi…"):
                    s_used = pd.qcut(s, q=4, duplicates="drop").astype(str)
        else:
            with st.spinner("Preparo livelli di stratificazione…"):
                s_used = s.astype(str)

        levels_s = [lv for lv in pd.Series(s_used).dropna().unique().tolist()]
        levels_s = sorted(levels_s, key=lambda v: v)

        rows = []
        with st.spinner("Calcolo metriche per ciascun sottogruppo…"):
            for lv in levels_s:
                mask_g = (pd.Series(s_used).astype(str) == str(lv)).values
                yg = y_valid.values[mask_g]
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
                rows.append({
                    "Sottogruppo": str(lv),
                    "N": int(len(yg)),
                    "Sensibilità": float(mdf.loc[mdf["Metrica"]=="Sensibilità","Valore"].values[0]),
                    "Specificità": float(mdf.loc[mdf["Metrica"]=="Specificità","Valore"].values[0]),
                    "PPV": float(mdf.loc[mdf["Metrica"]=="PPV","Valore"].values[0]),
                    "NPV": float(mdf.loc[mdf["Metrica"]=="NPV","Valore"].values[0]),
                    "Accuratezza": float(mdf.loc[mdf["Metrica"]=="Accuratezza","Valore"].values[0]),
                    "LR+": float(mdf.loc[mdf["Metrica"]=="LR+","Valore"].values[0]),
                    "LR−": float(mdf.loc[mdf["Metrica"]=="LR−","Valore"].values[0])
                })

        subgroup_table = pd.DataFrame(rows)
        st.markdown("### Risultati per sottogruppi")
        st.dataframe(subgroup_table.round(4), use_container_width=True)
        st.caption(
            "Attenzione: differenze tra sottogruppi possono dipendere da **campioni piccoli**, **soglie non ottimali** o "
            "**prevalenza diversa**. Usare cautela nell’interpretazione e, se possibile, validare in un set indipendente."
        )

# =========================
#   Esporta risultati
# =========================
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
            }
        }
        if _HAS_SKLEARN and score_for_plots is not None and len(score_for_plots)==len(y_valid):
            item["content"]["dca"] = _dca_compute(y_valid.values, score_for_plots, np.linspace(0.01,0.99,99)).round(6).to_dict(orient="records")
        if 'subgroup_table' in locals() and isinstance(subgroup_table, pd.DataFrame) and not subgroup_table.empty:
            item["content"]["subgroups"] = subgroup_table.round(6).to_dict(orient="records")
        st.session_state.report_items.append(item)
    st.success("Analisi diagnostica aggiunta al Results Summary.")

# =========================
#  Sezione didattica finale
# =========================
with st.expander("ℹ️ Spiegazione completa (utente non esperto)", expanded=False):
    st.markdown("""
**Che cosa misuro?**  
- **Sensibilità**: tra i malati, quanti risultano positivi.  
- **Specificità**: tra i sani, quanti risultano negativi.  
- **PPV/NPV**: probabilità che un positivo/negativo al test sia davvero malato/sano; dipendono dalla **prevalenza**.  
- **LR+ / LR− (con IC 95%)**: quanto cambia l’evidenza a favore/contro la malattia; si basano su sensibilità e specificità.  
- **AUC**: capacità del test (o score) di separare malati da sani su tutte le soglie.  
- **AP (Average Precision)**: media della precisione lungo i valori di recall, utile con classi sbilanciate.  
- **DCA (Decision Curve Analysis)**: valuta l’**utilità clinica** del test confrontando il **Net Benefit** del test con strategie
  di riferimento (**Tratta tutti** e **Tratta nessuno**) su varie soglie di decisione \(p_t\).

**Scelta della soglia (test continui)**  
- Spesso si massimizza **Youden J = Sens + Spec − 1**, ma in screening si tende a privilegiare la **sensibilità**,
  mentre nella conferma diagnostica la **specificità**.

**Da pre-test a post-test (Bayes)**  
""")
    st.latex(r"\text{odds}_{post} = \text{odds}_{pre} \times LR")
    st.latex(r"p_{post} = \frac{\text{odds}_{post}}{1+\text{odds}_{post}}")
    st.markdown("""
**Esempio**: pre-test 30% ⇒ odds=0.30/0.70≈0.43. Con **LR+=5** ⇒ odds_post=2.15 ⇒ p_post≈0.68.

**Avvertenze**  
- Con **campioni piccoli** gli IC sono ampi; per LR gli IC sono calcolati su scala logaritmica.  
- **PPV/NPV** cambiano con la prevalenza della popolazione in cui applico il test.  
- Nella **DCA**, il beneficio dipende dalle preferenze cliniche espresse da \(p_t\): scegliere il range di soglie coerente col contesto.
""")
