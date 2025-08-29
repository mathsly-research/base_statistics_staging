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
st.set_page_config(page_title="Analisi Test Diagnostici", layout="wide")
try:
    from nav import sidebar
    sidebar()
except Exception:
    pass

KEY = "diag"
def k(name: str) -> str:
    return f"{KEY}_{name}"

# ──────────────────────────────────────────────────────────────────────────────
# Helper: ROC/PR, metriche, tabelle
# ──────────────────────────────────────────────────────────────────────────────
def auc_mann_whitney(y_true: np.ndarray, score: np.ndarray) -> float:
    """AUC = Pr(score_pos > score_neg) via U di Mann–Whitney."""
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
    """ROC corretta: soglie ai valori distinti (discendente), gestione tie; include (0,0) e (1,1)."""
    y = np.asarray(y_true).astype(int)
    s = np.asarray(score).astype(float)
    if not greater_is_positive: s = -s
    P = int((y == 1).sum()); N = int((y == 0).sum())
    if P == 0 or N == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0]), np.array([np.nan])
    order = np.argsort(-s, kind="mergesort")
    ys, ss = y[order], s[order]
    fpr = [0.0]; tpr = [0.0]; thr_list = [np.inf]
    tp = fp = 0; i = 0; n = len(ys)
    while i < n:
        thr = ss[i]; tp_inc = fp_inc = 0
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
    """Precision–Recall con step sui valori distinti dello score."""
    y = np.asarray(y_true).astype(int)
    s = np.asarray(score).astype(float)
    if not greater_is_positive: s = -s
    order = np.argsort(-s, kind="mergesort")
    ys = y[order]
    tp = fp = 0; P = int((y == 1).sum())
    recalls = [0.0]; precisions = [1.0]
    i = 0; n = len(ys)
    while i < n:
        thr = s[order][i]
        tp_inc = fp_inc = 0
        while i < n and s[order][i] == thr:
            if ys[i] == 1: tp_inc += 1
            else: fp_inc += 1
            i += 1
        tp += tp_inc; fp += fp_inc
        rec = tp / max(P, 1); prec = tp / max(tp + fp, 1)
        recalls.append(rec); precisions.append(prec)
    return np.array(recalls), np.array(precisions)

def metrics_at_threshold(y_true: np.ndarray, score: np.ndarray, thr: float, greater_is_positive: bool = True):
    """Metriche per una soglia data (assoluta sullo score)."""
    y = np.asarray(y_true).astype(int)
    s = np.asarray(score).astype(float)
    if greater_is_positive: yhat = (s >= thr).astype(int)
    else:                   yhat = (s <= thr).astype(int)

    TP = int(((y == 1) & (yhat == 1)).sum())
    TN = int(((y == 0) & (yhat == 0)).sum())
    FP = int(((y == 0) & (yhat == 1)).sum())
    FN = int(((y == 1) & (yhat == 0)).sum())

    sens = TP / max(TP + FN, 1); spec = TN / max(TN + FP, 1)
    ppv  = TP / max(TP + FP, 1); npv  = TN / max(TN + FN, 1)
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
    """Tabella metriche per tutte le soglie distinte (ordinate)."""
    s = np.asarray(score).astype(float)
    thr_unique = np.unique(np.sort(s))
    rows = [metrics_at_threshold(y_true, score, thr, greater_is_positive) for thr in thr_unique]
    dfm = pd.DataFrame(rows).sort_values("threshold")
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

# ── Smussamento ROC ───────────────────────────────────────────────────────────
def _uniq_monotone(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rende FPR unico crescente e TPR non decrescente (max per FPR duplicati)."""
    df = pd.DataFrame({"x": x, "y": y}).groupby("x", as_index=False)["y"].max().sort_values("x")
    xx = df["x"].to_numpy()
    yy = df["y"].to_numpy()
    # assicura non-decrescenza
    yy = np.maximum.accumulate(yy)
    return xx, yy

def smooth_roc(fpr: np.ndarray, tpr: np.ndarray, grid_points: int = 300) -> tuple[np.ndarray, np.ndarray]:
    """
    Smussa la ROC in modo monotòno:
    - usa PCHIP se disponibile; altrimenti interp lineare su griglia densa;
    - forza (0,0) e (1,1) e clamp a [0,1].
    """
    # garantisce punti 0 e 1
    fpr = np.asarray(fpr, float); tpr = np.asarray(tpr, float)
    pts = np.vstack([fpr, tpr]).T
    pts = np.vstack([[0.0, 0.0], pts, [1.0, 1.0]])
    fpr_u, tpr_u = _uniq_monotone(pts[:, 0], pts[:, 1])

    grid = np.linspace(0.0, 1.0, grid_points)
    try:
        from scipy.interpolate import PchipInterpolator  # monotone shape-preserving
        interp = PchipInterpolator(fpr_u, tpr_u, extrapolate=False)
        tpr_s = interp(grid)
    except Exception:
        # fallback: interpolazione lineare + cumulativo per monotonia
        tpr_s = np.interp(grid, fpr_u, tpr_u)
    tpr_s = np.clip(tpr_s, 0.0, 1.0)
    tpr_s = np.maximum.accumulate(tpr_s)
    return grid, tpr_s

def make_roc_figure(fpr, tpr, auc_value, sens_at_thr, spec_at_thr, thr_label: str = "",
                    style: str = "classic", smooth: bool = False):
    """
    ROC:
    - style='classic' → verde, linea piena; style='soft' → rosso, tratteggiata (come l'immagine).
    - smooth=True → curva smussata (PCHIP se disponibile).
    """
    # Prepara punti (eventuale smoothing)
    if smooth:
        x_plot, y_plot = smooth_roc(fpr, tpr)
    else:
        # garantisce (0,0) e (1,1) anche nella visualizzazione
        x_plot, y_plot = _uniq_monotone(np.array(fpr, float), np.array(tpr, float))
        if x_plot[0] > 0:
            x_plot = np.insert(x_plot, 0, 0.0); y_plot = np.insert(y_plot, 0, 0.0)
        if x_plot[-1] < 1:
            x_plot = np.append(x_plot, 1.0);    y_plot = np.append(y_plot, 1.0)

    # Stili
    if style == "soft":
        line_color = "#8e1b1b"
        line_dash = "dot"
        fill_color = "rgba(200,0,0,0.20)"
        marker_color = "#8e1b1b"
    else:
        line_color = "#2ecc71"
        line_dash = None
        fill_color = "rgba(128,128,128,0.35)"
        marker_color = "#2ecc71"

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_plot, y=y_plot,
        mode="lines",
        line=dict(color=line_color, width=3, dash=line_dash),
        fill="tozeroy", fillcolor=fill_color,
        name=f"ROC (AUC={auc_value:.3f})"
    ))
    # Diagonale disegnata SOPRA l'area
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode="lines",
        line=dict(color="rgba(0,0,0,0.75)", dash="dash", width=2),
        name="No-skill"
    ))
    # Punto alla soglia corrente
    fig.add_trace(go.Scatter(
        x=[1 - spec_at_thr], y=[sens_at_thr],
        mode="markers",
        marker=dict(symbol="x", size=11, color=marker_color),
        name=(f"Soglia {thr_label}" if thr_label else "Soglia")
    ))
    fig.update_layout(
        template="simple_white",
        title="Curva ROC",
        xaxis_title="FPR (1 − Specificità)",
        yaxis_title="TPR (Sensibilità)",
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="left", x=0),
        height=420,
    )
    fig.update_xaxes(range=[0, 1], showline=True, linewidth=2, linecolor="black")
    fig.update_yaxes(range=[0, 1], showline=True, linewidth=2, linecolor="black",
                     scaleanchor="x", scaleratio=1)
    return fig

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
c1, c2, c3 = st.columns([2, 2, 1.4])
with c1:
    y_col = st.selectbox("Outcome (binario o categoriale)", options=all_cols, key=k("y"))
with c2:
    score_col = st.selectbox("Score / probabilità (numerico)", options=num_cols, index=0 if num_cols else None, key=k("score"))
with c3:
    direction = st.selectbox("Direzione", ["Più alto = più positivo", "Più basso = più positivo"], key=k("dir"))

y_raw = df[y_col]
# Binarizzazione outcome
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

# Controllo classi
if y.nunique(dropna=True) < 2:
    st.error("L'outcome deve contenere entrambe le classi (0 e 1).")
    st.stop()

# ──────────────────────────────────────────────────────────────────────────────
# STEP 2 — Soglia e metriche
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("### 2) Soglia e metriche")
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
    thr_manual = st.slider("Soglia manuale (assoluta)",
                           min_value=min_s, max_value=max_s,
                           value=float(np.percentile(s.dropna(), 50)) if s.notna().any() else 0.5,
                           step=step, key=k("thr_manual"))

if   thr_mode.startswith("Ottimizza Youden"): thr = thr_youden
elif thr_mode.startswith("Ottimizza F1"):     thr = thr_f1
else:                                         thr = thr_manual

# Metriche alla soglia scelta
M   = metrics_at_threshold(y.values, s.values, thr, greater_is_positive)
AUC = auc_mann_whitney(y.values, s.values)

# Pannello metriche con brevi spiegazioni
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Accuracy", f"{M['Acc']:.3f}");   m1.caption("Quota di classificazioni corrette (dipende dal bilanciamento).")
m2.metric("Sensibilità (TPR)", f"{M['Sens']:.3f}"); m2.caption("Pr(positivo predetto | positivo vero).")
m3.metric("Specificità (TNR)", f"{M['Spec']:.3f}"); m3.caption("Pr(negativo predetto | negativo vero).")
m4.metric("PPV (Precision)", f"{M['PPV']:.3f}");    m4.caption("Pr(positivo vero | positivo predetto).")
m5.metric("NPV", f"{M['NPV']:.3f}");                m5.caption("Pr(negativo vero | negativo predetto).")

m6, m7, m8, m9, m10 = st.columns(5)
m6.metric("Balanced Acc.", f"{M['BAcc']:.3f}"); m6.caption("Media di Sens e Spec: robusta a sbilanciamento.")
m7.metric("F1", f"{M['F1']:.3f}");             m7.caption("Media armonica tra Precision e Recall.")
m8.metric("MCC", f"{M['MCC']:.3f}");           m8.caption("Correlazione vero↔predetto (−1…1).")
m9.metric("LR+", f"{M['LR_plus']:.2f}");       m9.caption("Aumento dell’odds se test positivo (≫1 meglio).")
m10.metric("LR−", f"{M['LR_minus']:.2f}");     m10.caption("Riduzione dell’odds se test negativo (≪1 meglio).")

st.markdown(f"**Soglia corrente:** `{thr:.6g}` • **Indice di Youden:** `{M['Youden']:.3f}` • **AUC (ROC):** `{AUC:.3f}`")
st.caption("AUC misura la discriminazione complessiva (0.5=casuale, >0.8=ottima, 1=perfetta).")

# Matrice di confusione
st.markdown("#### Matrice di confusione")
cm = confusion_table_counts_percent(M["TN"], M["FP"], M["FN"], M["TP"])
st.dataframe(cm, use_container_width=True)

# ──────────────────────────────────────────────────────────────────────────────
# STEP 3 — Grafici (ROC classica/morbida + PR + distribuzione score)
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("### 3) Grafici")
left, right = st.columns(2)

with left:
    if px is not None and go is not None and s.notna().any():
        fpr, tpr, _ = roc_curve_strict(y.values, s.values, greater_is_positive)

        opt1, opt2 = st.columns([1.3, 1.7])
        with opt1:
            roc_smooth = st.checkbox("Andamento morbido (interpolato)", value=True, key=k("roc_smooth"))
        with opt2:
            roc_style = st.selectbox("Stile curva",
                                     ["Classico (verde)", "Morbido (rosso)"],
                                     index=1 if roc_smooth else 0, key=k("roc_style"))

        style = "soft" if roc_style.startswith("Morbido") else "classic"
        figroc = make_roc_figure(
            fpr=fpr, tpr=tpr,
            auc_value=AUC,
            sens_at_thr=M["Sens"],
            spec_at_thr=M["Spec"],
            thr_label=f"{thr:.2f}",
            style=style,
            smooth=roc_smooth
        )
        st.plotly_chart(figroc, use_container_width=True)

with right:
    if px is not None and go is not None and s.notna().any():
        t1, t2 = st.tabs(["📈 Precision–Recall", "📊 Distribuzione score"])

        with t1:
            rec, prec = pr_curve(y.values, s.values, greater_is_positive)
            prevalenza = float((y == 1).mean())

            st.markdown("**Annotazioni sulla curva**")
            r1c1, r1c2 = st.columns([1.2, 1.6])
            with r1c1:
                show_current_thr = st.checkbox("Mostra soglia corrente", value=True, key=k("pr_show_cur"))
            with r1c2:
                add_compare = st.checkbox("Aggiungi soglie di confronto", value=True, key=k("pr_add_cmp"))

            method = st.radio("Metodo per le soglie di confronto", ["Percentili", "Valori assoluti"],
                              horizontal=True, key=k("pr_method"))

            low_thr = high_thr = None
            if add_compare:
                if method == "Percentili":
                    pct_low, pct_high = st.slider("Percentili (basso, alto)",
                                                  min_value=0, max_value=100, value=(10, 90), step=1, key=k("pr_pcts"))
                    if s.notna().any():
                        low_thr  = float(np.percentile(s.dropna(), pct_low))
                        high_thr = float(np.percentile(s.dropna(), pct_high))
                else:
                    min_s, max_s = float(np.nanmin(s)), float(np.nanmax(s))
                    default_low  = min_s + 0.1 * (max_s - min_s)
                    default_high = min_s + 0.9 * (max_s - min_s)
                    low_thr, high_thr = st.slider("Valori assoluti (basso, alto)",
                                                  min_value=min_s, max_value=max_s,
                                                  value=(default_low, default_high),
                                                  step=max((max_s - min_s) / 100.0, 1e-6),
                                                  key=k("pr_abs"))

            figpr = go.Figure()
            figpr.add_trace(go.Scatter(x=rec, y=prec, mode="lines", line_shape="hv", name="PR"))
            figpr.add_hline(y=prevalenza, line_dash="dash",
                            annotation_text="Prevalenza", annotation_position="bottom right")

            def add_point_with_label(fig, thr_value: float, label: str, dy: int = -40):
                try:
                    Mx = metrics_at_threshold(y.values, s.values, float(thr_value), greater_is_positive)
                    r = float(Mx["Sens"]); p = float(Mx["PPV"])
                    fig.add_trace(go.Scatter(x=[r], y=[p], mode="markers",
                                             marker=dict(size=10, color="#f39c12"), showlegend=False))
                    fig.add_annotation(x=r, y=p, text=f"{label}<br>P={p:.2f}, R={r:.2f}",
                                       showarrow=True, arrowhead=2, ax=0, ay=dy,
                                       bgcolor="rgba(255,255,255,0.9)")
                except Exception:
                    pass

            if show_current_thr:
                add_point_with_label(figpr, thr, f"soglia {thr:.3g}", dy=-50)
            if add_compare and (low_thr is not None) and (high_thr is not None):
                add_point_with_label(figpr, low_thr,  f"thr {low_thr:.3g}",  dy=40)
                add_point_with_label(figpr, high_thr, f"thr {high_thr:.3g}", dy=-40)

            figpr.update_layout(
                template="simple_white",
                title="Curva Precision–Recall (con soglie annotate)",
                xaxis_title="Recall (Sensibilità)", yaxis_title="Precision (PPV)",
                height=420, legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="left", x=0),
            )
            figpr.update_xaxes(range=[0, 1], showline=True, linewidth=2, linecolor="black")
            figpr.update_yaxes(range=[0, 1], showline=True, linewidth=2, linecolor="black")
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
        "**Curva ROC**  \n"
        "- **Assi**: orizzontale = **FPR** (1 − Specificità); verticale = **TPR** (Sensibilità).  \n"
        "- **Diagonale tratteggiata**: classificatore casuale (AUC = 0.5).  \n"
        "- **AUC**: area sotto la curva; misura la **discriminazione** (0.5=casuale, >0.8=ottima, 1=perfetta).  \n"
        "- **Soglia**: il marcatore indica Sensibilità/Specificità alla soglia corrente; variando la soglia ci si muove lungo la curva.  \n"
        "- **Curva morbida**: è un’interpolazione monotòna dei punti ROC (estetica/leggibilità), non cambia l’AUC.\n\n"
        "**Curva Precision–Recall (PR)**  \n"
        "- **Assi**: orizzontale = **Recall** (Sensibilità), verticale = **Precision** (PPV).  \n"
        "- **Linea orizzontale**: **prevalenza** (baseline della Precision).  \n"
        "- **Interpretazione**: più la curva è **in alto a destra**, meglio è. Con **classi sbilanciate** la PR è spesso più informativa della ROC.  \n"
        "- **Punti annotati**: Precision/Recall a soglie **assolute** o per **percentile**; scelga la soglia in base ai **costi degli errori**."
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
        try:
            st.switch_page("pages/10_🧾_Agreement.py")
        except Exception:
            pass
