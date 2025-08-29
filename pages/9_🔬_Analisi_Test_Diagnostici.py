# -*- coding: utf-8 -*-
# pages/9_🔬_Analisi_Test_Diagnostici.py
from __future__ import annotations

import math
import streamlit as st
import pandas as pd
import numpy as np

# Plot (opzionali)
try:
    import plotly.express as px
    import plotly.graph_objects as go
except Exception:
    px = None
    go = None

# ──────────────────────────────────────────────────────────────────────────────
# Data store centralizzato (+ fallback sicuro)
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
# Config pagina + nav laterale
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="🔬 Analisi Test Diagnostici", layout="wide")
try:
    from nav import sidebar
    sidebar()
except Exception:
    pass

KEY = "diag"
def k(name: str) -> str:
    return f"{KEY}_{name}"

# ──────────────────────────────────────────────────────────────────────────────
# Helper: ROC/PR, metriche e tabelle
# ──────────────────────────────────────────────────────────────────────────────
def auc_mann_whitney(y_true: np.ndarray, score: np.ndarray) -> float:
    """AUC = probabilità che uno score positivo superi uno negativo (U di Mann–Whitney)."""
    try:
        from scipy.stats import rankdata
        y = np.asarray(y_true).astype(int)
        s = np.asarray(score).astype(float)
        n1 = int((y == 1).sum()); n0 = int((y == 0).sum())
        if n1 == 0 or n0 == 0: return float("nan")
        r = rankdata(s)
        u = float(r[y == 1].sum()) - n1 * (n1 + 1) / 2.0
        return float(u / (n1 * n0))
    except Exception:
        return float("nan")

def roc_curve_strict(y_true: np.ndarray, score: np.ndarray, greater_is_positive: bool = True):
    """ROC corretta: soglie ai valori distinti dello score (desc), gestione dei tie, include (0,0) e (1,1)."""
    y = np.asarray(y_true).astype(int)
    s = np.asarray(score).astype(float)
    if not greater_is_positive:
        s = -s
    P = int((y == 1).sum()); N = int((y == 0).sum())
    if P == 0 or N == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0]), np.array([np.nan])
    order = np.argsort(-s, kind="mergesort")
    ys, ss = y[order], s[order]
    fpr = [0.0]; tpr = [0.0]; thr_list = [np.inf]
    tp = 0; fp = 0; i = 0; n = len(ys)
    while i < n:
        thr = ss[i]
        tp_inc = 0; fp_inc = 0
        while i < n and ss[i] == thr:
            if ys[i] == 1: tp_inc += 1
            else: fp_inc += 1
            i += 1
        tp += tp_inc; fp += fp_inc
        tpr.append(tp / P); fpr.append(fp / N); thr_list.append(thr)
    if tpr[-1] != 1.0 or fpr[-1] != 1.0:
        tpr.append(1.0); fpr.append(1.0); thr_list.append(-np.inf)
    return np.array(fpr), np.array(tpr), np.array(thr_list)

def pr_curve(y_true: np.ndarray, score: np.ndarray, greater_is_positive: bool = True):
    """Precision-Recall con soglie ai valori distinti."""
    y = np.asarray(y_true).astype(int)
    s = np.asarray(score).astype(float)
    if not greater_is_positive: s = -s
    order = np.argsort(-s, kind="mergesort")
    ys = y[order]
    tp = 0; fp = 0
    P = int((y == 1).sum())
    recalls = [0.0]; precisions = [1.0]
    i = 0; n = len(ys)
    while i < n:
        thr = s[order][i]
        tp_inc = 0; fp_inc = 0
        while i < n and s[order][i] == thr:
            if ys[i] == 1: tp_inc += 1
            else: fp_inc += 1
            i += 1
        tp += tp_inc; fp += fp_inc
        rec = tp / max(P, 1)
        prec = tp / max(tp + fp, 1)
        recalls.append(rec); precisions.append(prec)
    return np.array(recalls), np.array(precisions)

def metrics_at_threshold(y_true: np.ndarray, score: np.ndarray, thr: float, greater_is_positive: bool = True):
    """Metriche a una soglia data."""
    y = np.asarray(y_true).astype(int)
    s = np.asarray(score).astype(float)
    if greater_is_positive:
        yhat = (s >= thr).astype(int)
    else:
        yhat = (s <= thr).astype(int)

    TP = int(((y == 1) & (yhat == 1)).sum())
    TN = int(((y == 0) & (yhat == 0)).sum())
    FP = int(((y == 0) & (yhat == 1)).sum())
    FN = int(((y == 1) & (yhat == 0)).sum())

    sens = TP / max(TP + FN, 1)
    spec = TN / max(TN + FP, 1)
    ppv  = TP / max(TP + FP, 1)
    npv  = TN / max(TN + FN, 1)
    acc  = (TP + TN) / max(len(y), 1)
    bacc = (sens + spec) / 2.0
    f1   = (2 * ppv * sens) / max(ppv + sens, 1e-12)
    denom = math.sqrt(max((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN), 1))
    mcc  = ((TP * TN - FP * FN) / denom) if denom > 0 else float("nan")
    lr_p = sens / max(1 - spec, 1e-12)
    lr_m = (1 - sens) / max(spec, 1e-12)

    return dict(threshold=thr, TP=TP, TN=TN, FP=FP, FN=FN, Sens=sens, Spec=spec,
                PPV=ppv, NPV=npv, Acc=acc, BAcc=bacc, F1=f1, MCC=mcc,
                LR_plus=lr_p, LR_minus=lr_m, Youden=sens + spec - 1)

def build_threshold_table(y_true: np.ndarray, score: np.ndarray, greater_is_positive: bool):
    """Tabella metriche per tutte le soglie distinte (ordinate per soglia)."""
    s = np.asarray(score).astype(float)
    thr_unique = np.unique(np.sort(s))
    rows = []
    for thr in thr_unique:
        rows.append(metrics_at_threshold(y_true, score, thr, greater_is_positive))
    dfm = pd.DataFrame(rows).sort_values("threshold")
    # suggerimenti soglia
    best_youden = dfm.loc[dfm["Youden"].idxmax(), "threshold"] if not dfm["Youden"].isna().all() else np.nan
    best_f1 = dfm.loc[dfm["F1"].idxmax(), "threshold"] if not dfm["F1"].isna().all() else np.nan
    return dfm, float(best_youden), float(best_f1)

def confusion_table_counts_percent(TN, FP, FN, TP) -> pd.DataFrame:
    """Matrice di confusione con percentuali per riga."""
    row0 = TN + FP; row1 = FN + TP
    def cell(c, r):
        p = (c / r * 100.0) if r > 0 else np.nan
        return f"{int(c)} ({p:.1f}%)" if p == p else f"{int(c)}"
    return pd.DataFrame(
        {"Pred 0": [cell(TN, row0), cell(FN, row1)],
         "Pred 1": [cell(FP, row0), cell(TP, row1)]},
        index=["Vera 0", "Vera 1"]
    )

# ──────────────────────────────────────────────────────────────────────────────
# Header
# ──────────────────────────────────────────────────────────────────────────────
st.title("🔬 Analisi Test Diagnostici")
st.caption("ROC/PR, metriche e scelta soglia per un outcome binario a partire da uno score/valore continuo.")

ensure_initialized()
df = get_active(required=True)
with st.expander("Stato dati", expanded=False):
    stamp_meta()
if df is None or df.empty:
    st.stop()

num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
all_cols = list(df.columns)

# ──────────────────────────────────────────────────────────────────────────────
# STEP 1 — Selezione variabili
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("### 1) Seleziona outcome e score")

c1, c2, c3 = st.columns([2, 2, 1.2])
with c1:
    y_col = st.selectbox("Outcome (binario o categoriale)", options=all_cols, key=k("y"))
with c2:
    score_col = st.selectbox("Score / probabilità (numerico)", options=num_cols, index=0 if num_cols else None, key=k("score"))
with c3:
    direction = st.selectbox("Direzione", ["Più alto = più positivo", "Più basso = più positivo"], key=k("dir"))

y_raw = df[y_col]
# binarizzazione outcome
if pd.api.types.is_numeric_dtype(y_raw) and set(pd.unique(y_raw.dropna())) <= {0, 1}:
    pos_label = 1
    y = y_raw.astype(int)
    st.caption("Outcome interpretato come binario {0,1} con '1' = positivo.")
else:
    levels = sorted(y_raw.dropna().astype(str).unique().tolist())
    pos_label = st.selectbox("Scegli la categoria 'positiva' (outcome = 1)", options=levels, key=k("poslab"))
    y = (y_raw.astype(str) == pos_label).astype(int)

s = pd.to_numeric(df[score_col], errors="coerce")
greater_is_positive = (direction == "Più alto = più positivo")

# controllo classi
if y.nunique(dropna=True) < 2:
    st.error("L'outcome deve contenere entrambe le classi (0 e 1).")
    st.stop()

# ──────────────────────────────────────────────────────────────────────────────
# STEP 2 — Scelta soglia e metriche
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("### 2) Soglia e metriche")

# Tabella completa (per suggerire soglie)
df_thr, thr_youden, thr_f1 = build_threshold_table(y.values, s.values, greater_is_positive)

colA, colB = st.columns([1.6, 1.4])
with colA:
    thr_mode = st.radio("Come scegliere la soglia?",
                        ["Manuale", "Ottimizza Youden (Sens+Spec−1)", "Ottimizza F1"],
                        index=1, key=k("thr_mode"))
with colB:
    if np.nanmin(s) == np.nanmax(s):
        st.warning("Lo score è costante: non è possibile costruire una curva ROC/PR.")
    min_s, max_s = float(np.nanmin(s)), float(np.nanmax(s))
    step = max((max_s - min_s) / 100.0, 1e-6)
    thr_manual = st.slider("Soglia manuale",
                           min_value=min_s, max_value=max_s,
                           value=float(np.percentile(s.dropna(), 50)) if s.notna().any() else 0.5,
                           step=step, key=k("thr_manual"))

if thr_mode.startswith("Ottimizza Youden"):
    thr = thr_youden
elif thr_mode.startswith("Ottimizza F1"):
    thr = thr_f1
else:
    thr = thr_manual

# metriche a soglia selezionata
M = metrics_at_threshold(y.values, s.values, thr, greater_is_positive)
AUC = auc_mann_whitney(y.values, s.values)

# pannello metriche (con brevi spiegazioni)
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Accuracy", f"{M['Acc']:.3f}")
m1.caption("Quota di classificazioni corrette (dipende dal bilanciamento delle classi).")
m2.metric("Sensibilità (TPR)", f"{M['Sens']:.3f}")
m2.caption("Pr(positivo predetto | positivo vero): capacità di cogliere i veri positivi.")
m3.metric("Specificità (TNR)", f"{M['Spec']:.3f}")
m3.caption("Pr(negativo predetto | negativo vero): evita falsi positivi.")
m4.metric("PPV (Precision)", f"{M['PPV']:.3f}")
m4.caption("Pr(positivo vero | positivo predetto): utilità in contesti clinici di conferma.")
m5.metric("NPV", f"{M['NPV']:.3f}")
m5.caption("Pr(negativo vero | negativo predetto): utilità per escludere la malattia.")

m6, m7, m8, m9, m10 = st.columns(5)
m6.metric("Balanced Acc.", f"{M['BAcc']:.3f}")
m6.caption("Media di Sens e Spec: robusta a classi sbilanciate.")
m7.metric("F1", f"{M['F1']:.3f}")
m7.caption("Media armonica tra Precision (PPV) e Recall (Sensibilità).")
m8.metric("MCC", f"{M['MCC']:.3f}")
m8.caption("Correlazione tra vero e predetto (−1…1): robusto allo sbilanciamento.")
m9.metric("LR+", f"{M['LR_plus']:.2f}")
m9.caption("Quanto aumenta l’odds di malattia se il test è positivo (idealmente ≫1).")
m10.metric("LR−", f"{M['LR_minus']:.2f}")
m10.caption("Quanto resta l’odds con test negativo (idealmente ≪1).")

st.markdown(
    f"**Soglia corrente:** `{thr:.6g}`  "
    f"• **Indice di Youden:** `{M['Youden']:.3f}`  "
    f"• **AUC (ROC):** `{AUC:.3f}`"
)
st.caption("AUC misura la discriminazione complessiva (0.5=casuale, >0.8=ottima, 1=perfetta).")

# Matrice di confusione
st.markdown("#### Matrice di confusione")
cm = confusion_table_counts_percent(M["TN"], M["FP"], M["FN"], M["TP"])
st.dataframe(cm, use_container_width=True)

# ──────────────────────────────────────────────────────────────────────────────
# STEP 3 — Grafici (ROC classica + PR + distribuzione p̂)
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("### 3) Grafici")

left, right = st.columns(2)

# ROC (curva verde, area grigia, diagonale, marker soglia)
with left:
    if px is not None and go is not None and s.notna().any():
        fpr, tpr, thr_list = roc_curve_strict(y.values, s.values, greater_is_positive)
        figroc = go.Figure()
        figroc.add_trace(go.Scatter(
            x=fpr, y=tpr, mode="lines", line_shape="hv",
            line=dict(color="#2ecc71", width=3),
            fill="tozeroy", fillcolor="rgba(128,128,128,0.35)",
            name=f"ROC (AUC={AUC:.3f})"
        ))
        figroc.add_shape(type="line", x0=0, y0=0, x1=1, y1=1,
                         line=dict(color="rgba(0,0,0,0.5)", dash="dash"))
        # punto alla soglia corrente
        fpr_thr = 1 - M["Spec"]
        tpr_thr = M["Sens"]
        figroc.add_trace(go.Scatter(
            x=[fpr_thr], y=[tpr_thr], mode="markers",
            marker=dict(symbol="x", size=10, color="#2ecc71"),
            name=f"Soglia {thr:.2f}"
        ))
        figroc.update_layout(
            template="simple_white", title="Curva ROC",
            xaxis_title="FPR (1 − Specificità)", yaxis_title="TPR (Sensibilità)",
            legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="left", x=0),
        )
        figroc.update_xaxes(range=[0, 1], showline=True, linewidth=2, linecolor="black")
        figroc.update_yaxes(range=[0, 1], showline=True, linewidth=2, linecolor="black",
                            scaleanchor="x", scaleratio=1)
        st.plotly_chart(figroc, use_container_width=True)

# PR curve + distribuzione dello score per classe in tab affiancato
with right:
    if px is not None and go is not None and s.notna().any():
        t1, t2 = st.tabs(["📈 Precision–Recall", "📊 Distribuzione score"])
        with t1:
            rec, prec = pr_curve(y.values, s.values, greater_is_positive)
            figpr = go.Figure()
            figpr.add_trace(go.Scatter(x=rec, y=prec, mode="lines", line_shape="hv", name="PR"))
            # baseline = prevalenza
            prev = float((y == 1).mean())
            figpr.add_hline(y=prev, line_dash="dash")
            figpr.update_layout(template="simple_white", title="Curva Precision–Recall",
                                xaxis_title="Recall (Sensibilità)", yaxis_title="Precision (PPV)")
            figpr.update_xaxes(range=[0, 1]); figpr.update_yaxes(range=[0, 1])
            st.plotly_chart(figpr, use_container_width=True)
        with t2:
            df_plot = pd.DataFrame({"score": s, "y": y})
            df_plot["Classe"] = df_plot["y"].map({0: "Classe 0", 1: "Classe 1"})
            figd = px.histogram(df_plot.dropna(), x="score", color="Classe",
                                barmode="overlay", nbins=30, template="simple_white",
                                title="Distribuzione dello score per classe")
            figd.add_vline(x=thr, line_dash="dash")
            figd.update_layout(xaxis_title=score_col, yaxis_title="Frequenza")
            st.plotly_chart(figd, use_container_width=True)

with st.expander("ℹ️ Come leggere i grafici", expanded=False):
    st.markdown(
        "- **ROC**: curva verde con **area grigia** (AUC). La **diagonale** è il caso casuale. "
        "Il marcatore indica la coppia (**FPR**, **TPR**) alla **soglia scelta**.  \n"
        "- **PR**: utile con classi sbilanciate; la linea orizzontale tratteggiata è la **prevalenza** (baseline).  \n"
        "- **Distribuzione score**: una buona separazione tra le due classi indica **buona discriminazione**; "
        "la linea tratteggiata è la **soglia corrente**."
    )

# ──────────────────────────────────────────────────────────────────────────────
# STEP 4 — Tabella completa delle soglie
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("### 4) Tabella completa per tutte le soglie")
df_show = df_thr.copy()
cols_order = ["threshold", "Acc", "Sens", "Spec", "PPV", "NPV", "BAcc", "F1", "MCC", "LR_plus", "LR_minus", "Youden", "TP", "FP", "FN", "TN"]
df_show = df_show[[c for c in cols_order if c in df_show.columns]]
st.dataframe(df_show.round(4), use_container_width=True, height=300)

csv = df_show.to_csv(index=False).encode("utf-8")
st.download_button("Scarica tabella (CSV)", data=csv, file_name="diagnostics_thresholds.csv", mime="text/csv", key=k("dl"))

# ──────────────────────────────────────────────────────────────────────────────
# Navigazione
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("---")
nav1, nav2 = st.columns(2)
with nav1:
    if st.button("⬅️ Torna: Regression", use_container_width=True, key=k("go_prev")):
        st.switch_page("pages/8_🧮_Regression.py")
with nav2:
    if st.button("➡️ Vai: Agreement", use_container_width=True, key=k("go_next")):
        # adatti il nome del file se diverso
        try:
            st.switch_page("pages/10_🧾_Agreement.py")
        except Exception:
            pass
