# -*- coding: utf-8 -*-
# pages/10_📏_Agreement.py
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
# Data store centralizzato (+ fallback sicuro come negli altri moduli)
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
st.set_page_config(page_title="Agreement", layout="wide")
try:
    from nav import sidebar
    sidebar()
except Exception:
    pass

KEY = "agr"
def k(name: str) -> str:
    return f"{KEY}_{name}"

# ──────────────────────────────────────────────────────────────────────────────
# UTIL: Agreement per dati categoriali
# ──────────────────────────────────────────────────────────────────────────────
def confusion_from_series(a: pd.Series, b: pd.Series) -> pd.DataFrame:
    cats = sorted(pd.unique(pd.concat([a.dropna().astype(str), b.dropna().astype(str)])))
    cm = pd.crosstab(a.astype(str), b.astype(str), dropna=False)
    cm = cm.reindex(index=cats, columns=cats, fill_value=0)
    cm.index.name = "Rater A"; cm.columns.name = "Rater B"
    return cm

def percent_agreement(cm: pd.DataFrame) -> float:
    n = cm.to_numpy().sum()
    return float(np.trace(cm)) / n if n > 0 else float("nan")

def cohen_kappa(cm: pd.DataFrame, weights: str | None = None) -> float:
    """
    Kappa di Cohen (non pesata / pesi lineari / quadratici).
    weights: None | "linear" | "quadratic"
    """
    M = cm.to_numpy(dtype=float)
    n = M.sum()
    if n == 0: return float("nan")
    r = M.sum(axis=1) / n
    c = M.sum(axis=0) / n
    k = M.shape[0]

    # Matrice dei pesi W_ij
    if weights is None:
        W = np.eye(k)
    else:
        i = np.arange(k).reshape(-1, 1)
        j = np.arange(k).reshape(1, -1)
        d = np.abs(i - j) / (k - 1) if k > 1 else 0.0
        if weights == "linear":
            W = 1.0 - d
        elif weights == "quadratic":
            W = 1.0 - d**2
        else:
            raise ValueError("weights deve essere None, 'linear' o 'quadratic'.")

    P_o = (W * (M / n)).sum()
    P_e = (W * (r.reshape(-1, 1) @ c.reshape(1, -1))).sum()
    if np.isclose(1 - P_e, 0): return float("nan")
    return (P_o - P_e) / (1 - P_e)

def fleiss_kappa_from_raters(df_rat: pd.DataFrame) -> tuple[float, pd.DataFrame]:
    """
    Fleiss' kappa per ≥3 valutatori.
    df_rat: colonne = valutatori, righe = soggetti; valori = categoria (stringa/numero).
    Restituisce (kappa, tabella N×K di conteggi per soggetto e categoria).
    """
    cats = sorted(pd.unique(df_rat.astype(str).stack()))
    cat_index = {c: i for i, c in enumerate(cats)}
    N = len(df_rat); k = len(cats)
    count = np.zeros((N, k), dtype=float)
    for i, (_, row) in enumerate(df_rat.iterrows()):
        for val in row.dropna().astype(str):
            count[i, cat_index[val]] += 1.0

    n_i = count.sum(axis=1)
    if not np.all(n_i == n_i[0]):
        st.warning("Numero di valutatori non costante su tutti i soggetti: Fleiss' κ richiede n fisso. Verranno escluse le righe non conformi.")
        mode_n = pd.Series(n_i).mode().iat[0]
        keep = (n_i == mode_n)
        count = count[keep, :]
        n_i = n_i[keep]
        if count.shape[0] == 0:
            return float("nan"), pd.DataFrame(count, columns=cats)

    n = n_i[0]
    N = count.shape[0]
    P_i = (1.0 / (n * (n - 1))) * (count * (count - 1)).sum(axis=1)
    P_bar = P_i.mean()
    p_j = count.sum(axis=0) / (N * n)
    P_e = (p_j**2).sum()
    if np.isclose(1 - P_e, 0): return float("nan"), pd.DataFrame(count, columns=cats)
    kappa = (P_bar - P_e) / (1 - P_e)
    tab = pd.DataFrame(count, columns=cats)
    tab.index.name = "Soggetto"
    return float(kappa), tab

# ──────────────────────────────────────────────────────────────────────────────
# UTIL: Agreement per dati continui (2 metodi/valutatori)
# ──────────────────────────────────────────────────────────────────────────────
def icc_two_way(data: pd.DataFrame, kind: str = "agreement") -> float:
    """
    ICC a due vie, unità singola.
    kind = 'agreement' → ICC(2,1)   (two-way random, absolute agreement)
    kind = 'consistency' → ICC(3,1) (two-way mixed, consistency)
    data: DataFrame n×k con colonne = valutatori/metodi, righe = soggetti.
    """
    X = data.to_numpy(dtype=float)
    if np.isnan(X).any():
        X = X[~np.isnan(X).any(axis=1), :]
    n, k = X.shape
    if n < 2 or k < 2:
        return float("nan")
    grand = X.mean()
    mean_r = X.mean(axis=1, keepdims=True)
    mean_c = X.mean(axis=0, keepdims=True)

    SSR = k * ((mean_r - grand)**2).sum()   # tra soggetti
    SSC = n * ((mean_c - grand)**2).sum()   # tra valutatori
    SST = ((X - grand)**2).sum()
    SSE = SST - SSR - SSC

    MSR = SSR / (n - 1)
    MSC = SSC / (k - 1)
    MSE = SSE / ((n - 1) * (k - 1))

    if kind == "agreement":  # ICC(2,1)
        denom = MSR + (k - 1) * MSE + (k * (MSC - MSE) / n)
        return float((MSR - MSE) / denom) if denom != 0 else float("nan")
    else:  # "consistency" → ICC(3,1)
        denom = MSR + (k - 1) * MSE
        return float((MSR - MSE) / denom) if denom != 0 else float("nan")

def lins_ccc(a: np.ndarray, b: np.ndarray) -> float:
    """Concordance Correlation Coefficient (Lin)."""
    a = pd.to_numeric(pd.Series(a), errors="coerce")
    b = pd.to_numeric(pd.Series(b), errors="coerce")
    df = pd.concat([a, b], axis=1).dropna()
    if df.empty: return float("nan")
    x = df.iloc[:, 0].to_numpy(dtype=float)
    y = df.iloc[:, 1].to_numpy(dtype=float)
    mx, my = x.mean(), y.mean()
    vx, vy = x.var(ddof=1), y.var(ddof=1)
    sxy = np.cov(x, y, ddof=1)[0, 1]
    return float((2 * sxy) / (vx + vy + (mx - my)**2)) if (vx + vy + (mx - my)**2) != 0 else float("nan")

def bland_altman_figure(a: pd.Series, b: pd.Series) -> tuple[go.Figure | None, dict]:
    """Bland–Altman con linee orizzontali alle LoA e banda ombreggiata tra le LoA."""
    if go is None:
        return None, {}

    x = pd.to_numeric(a, errors="coerce")
    y = pd.to_numeric(b, errors="coerce")
    dfp = pd.DataFrame({"m": (x + y) / 2.0, "d": x - y}).dropna()
    if dfp.empty:
        return None, {}

    bias = float(dfp["d"].mean())
    sd   = float(dfp["d"].std(ddof=1))
    loa_low  = bias - 1.96 * sd
    loa_high = bias + 1.96 * sd

    fig = go.Figure()

    # Punti
    fig.add_trace(go.Scatter(
        x=dfp["m"], y=dfp["d"],
        mode="markers",
        marker=dict(size=7, opacity=0.75),
        name="Differenze"
    ))

    # Banda tra le LoA (sotto ai punti)
    fig.add_shape(
        type="rect",
        x0=float(dfp["m"].min()), x1=float(dfp["m"].max()),
        y0=loa_low, y1=loa_high,
        line=dict(width=0),
        fillcolor="rgba(231, 76, 60, 0.10)", layer="below"
    )

    # Linee orizzontali: Bias e LoA ±1.96 SD
    fig.add_hline(
        y=bias, line_dash="solid", line_width=2, line_color="#2c3e50",
        annotation_text=f"Bias = {bias:.3f}", annotation_position="top left"
    )
    fig.add_hline(
        y=loa_low, line_dash="dash", line_width=2, line_color="#e74c3c",
        annotation_text=f"LoA− (−1.96·SD) = {loa_low:.3f}", annotation_position="bottom left"
    )
    fig.add_hline(
        y=loa_high, line_dash="dash", line_width=2, line_color="#e74c3c",
        annotation_text=f"LoA+ (+1.96·SD) = {loa_high:.3f}", annotation_position="bottom left"
    )

    # Layout
    fig.update_layout(
        template="simple_white",
        title="Bland–Altman",
        xaxis_title="Media dei due metodi",
        yaxis_title="Differenza (A − B)",
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="left", x=0),
        height=420,
    )
    fig.update_xaxes(showline=True, linewidth=2, linecolor="black")
    fig.update_yaxes(showline=True, linewidth=2, linecolor="black")

    stats = {"bias": bias, "sd": sd, "loa_low": loa_low, "loa_high": loa_high}
    return fig, stats

# ──────────────────────────────────────────────────────────────────────────────
# Header
# ──────────────────────────────────────────────────────────────────────────────
st.title("📏 Agreement")
st.caption("Misure di accordo tra valutatori/metodi per dati **categoriali** e **continui**. Interfaccia guidata e spiegazioni puntuali.")

ensure_initialized()
df = get_active(required=True)
with st.expander("Stato dati", expanded=False):
    stamp_meta()
if df is None or df.empty:
    st.stop()

all_cols = list(df.columns)
num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

# ──────────────────────────────────────────────────────────────────────────────
# Modalità di analisi
# ──────────────────────────────────────────────────────────────────────────────
mode = st.radio("Scegli il tipo di dati:", ["Categoriali (2 valutatori)", "Categoriali (≥3 valutatori)", "Continui (2 metodi)"],
                horizontal=True, key=k("mode"))

# ========================= CATEGORIALI: due valutatori ========================
if mode.startswith("Categoriali (2"):
    st.markdown("### 1) Selezione colonne (due valutatori)")
    c1, c2 = st.columns(2)
    with c1:
        a_col = st.selectbox("Valutatore A", options=[c for c in all_cols], key=k("a"))
    with c2:
        b_col = st.selectbox("Valutatore B", options=[c for c in all_cols if c != a_col], key=k("b"))

    a = df[a_col].astype(str)
    b = df[b_col].astype(str)

    # Guida rapida
    with st.expander("ℹ️ Guida rapida (categoriali, 2 valutatori)", expanded=False):
        st.markdown(
            "- **Matrice di confusione**: accordo sulla **diagonale**; fuori-diagonale = disaccordi.  \n"
            "- **% Accordo**: quota di casi in accordo; **non** corregge l’accordo atteso per caso.  \n"
            "- **Cohen’s κ**: corregge per il caso; usare **pesi lineari/quadratici** con **scale ordinali**.  \n"
            "- **Soglie orientative per κ** *(Landis & Koch)*: 0–0.20 scarso, 0.21–0.40 discreto, 0.41–0.60 moderato, 0.61–0.80 buono, 0.81–1.00 eccellente *(interpretare con prudenza e contesto)*.  \n"
            "- **Attenzione**: κ può essere **sensibile alla prevalenza** delle categorie (paradosso di κ)."
        )

    cm = confusion_from_series(a, b)

    st.markdown("### 2) Matrice di confusione")
    st.dataframe(cm, use_container_width=True)

    st.markdown("### 3) Kappa e accordo")
    colw1, colw2, colw3 = st.columns([1, 1, 2])
    with colw1:
        wtype = st.selectbox("Pesi", ["Nessuno", "Lineari", "Quadratici"], key=k("w"))
    with colw2:
        show_pct = st.checkbox("Mostra % accordo", value=True, key=k("pa"))
    with colw3:
        st.caption("Per categorie **ordinali**, preferire pesi **lineari** o **quadratici**.")

    wt = None if wtype == "Nessuno" else ("linear" if wtype == "Lineari" else "quadratic")
    kap = cohen_kappa(cm, weights=wt)
    pa = percent_agreement(cm)

    m1, m2 = st.columns(2)
    m1.metric("Cohen's κ", f"{kap:.3f}" if kap == kap else "—")
    m1.caption("0: accordo al caso; →1: accordo perfetto. Valutare con la distribuzione delle categorie.")
    if show_pct:
        m2.metric("% Accordo", f"{pa*100:.1f}%")
        m2.caption("Quota di osservazioni coincidenti. **Non** corregge per l’accordo atteso per caso.")

    with st.expander("📘 Come interpretare gli **indici** (2 valutatori)"):
        st.markdown(
            "**Cohen’s κ**  \n"
            "- Misura l’accordo corretto per il caso. κ=1 perfetto, κ=0 pari al caso, κ<0 peggio del caso.  \n"
            "- **Pesi**: con scale ordinali gli errori ‘vicini’ pesano meno (lineari) o molto meno (quadratici).  \n"
            "- **Prevalenza e bias** possono ridurre κ anche con % accordo elevata; usare % accordo come informazione complementare.\n\n"
            "**% Accordo**  \n"
            "- Semplice e intuitiva ma può **sovrastimare** l’accordo in presenza di categorie molto sbilanciate.  \n"
            "- Usarla insieme a κ e alla matrice per capire **dove** avvengono i disaccordi."
        )

# ========================= CATEGORIALI: ≥3 valutatori =========================
elif mode.startswith("Categoriali (≥3"):
    st.markdown("### 1) Seleziona le colonne dei valutatori (almeno 3)")
    raters = st.multiselect("Valutatori", options=all_cols, key=k("raters"))
    if len(raters) < 3:
        st.info("Selezionare almeno **3** colonne di valutatori.")
        st.stop()

    df_rat = df[raters].copy()
    with st.expander("ℹ️ Guida rapida (≥3 valutatori)", expanded=False):
        st.markdown(
            "- **Fleiss’ κ** generalizza κ di Cohen a **più valutatori**.  \n"
            "- Richiede lo **stesso numero di valutazioni** per soggetto; in caso contrario vengono escluse le righe non conformi.  \n"
            "- Interpretazione simile a Cohen’s κ; considerare anche la **distribuzione delle categorie**."
        )

    kappa_f, tab = fleiss_kappa_from_raters(df_rat)

    st.markdown("### 2) Tabella conteggi per soggetto e categoria (input Fleiss)")
    st.dataframe(tab, use_container_width=True, height=260)

    st.markdown("### 3) Fleiss' κ (≥3 valutatori)")
    st.metric("Fleiss' κ", f"{kappa_f:.3f}" if kappa_f == kappa_f else "—")
    st.caption("0: accordo al caso; →1: accordo perfetto. Valutare insieme a distribuzione categorie e compito di rating.")

    with st.expander("📘 Come interpretare **Fleiss’ κ**"):
        st.markdown(
            "- κ aumenta quando i valutatori **convergono** sulle stesse categorie.  \n"
            "- Con categorie **rare** o **sbilanciate**, la stima può essere attenuata (paradosso).  \n"
            "- Soglie orientative (come per Cohen’s κ) vanno **contestualizzate** al dominio e al rischio degli errori."
        )

# ================================ CONTINUI ====================================
else:
    st.markdown("### 1) Seleziona le colonne (due metodi/valutatori)")
    c1, c2 = st.columns(2)
    with c1:
        x_col = st.selectbox("Metodo/Valutatore A (continuo)", options=num_cols, key=k("x"))
    with c2:
        y_col = st.selectbox("Metodo/Valutatore B (continuo)", options=[c for c in num_cols if c != x_col], key=k("y"))

    x = pd.to_numeric(df[x_col], errors="coerce")
    y = pd.to_numeric(df[y_col], errors="coerce")
    df_xy = pd.concat([x, y], axis=1).dropna()
    if df_xy.empty:
        st.error("Servono almeno due colonne numeriche con valori non mancanti.")
        st.stop()

    with st.expander("ℹ️ Guida rapida (continui, 2 metodi)", expanded=False):
        st.markdown(
            "- **ICC(2,1)**: accordo **assoluto** (two-way *random*). Valido per generalizzare a valutatori simili.  \n"
            "- **ICC(3,1)**: **consistency** (two-way *mixed*): ignora differenze sistematiche di livello tra valutatori.  \n"
            "- **Lin’s CCC**: combina correlazione e accuratezza (penalizza bias di livello e scala).  \n"
            "- **Scatter + OLS**: identifica bias di **scala** (pendenza ≠1).  \n"
            "- **Bland–Altman**: evidenzia **bias medio** e **limiti di accordo (LoA)**; cercare trend della differenza vs media (bias proporzionale)."
        )

    st.markdown("### 2) Indici di accordo")
    icc_ag = icc_two_way(df_xy.rename(columns={x_col: "A", y_col: "B"}), kind="agreement")
    icc_con = icc_two_way(df_xy.rename(columns={x_col: "A", y_col: "B"}), kind="consistency")
    ccc = lins_ccc(df_xy.iloc[:, 0], df_xy.iloc[:, 1])
    r = np.corrcoef(df_xy.iloc[:, 0], df_xy.iloc[:, 1])[0, 1]

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("ICC(2,1) — Agreement", f"{icc_ag:.3f}" if icc_ag == icc_ag else "—")
    m1.caption("Due vie, effetti casuali, **accordo assoluto**. Soglie orientative: <0.50 scarso, 0.50–0.75 moderato, 0.75–0.90 buono, >0.90 eccellente.")
    m2.metric("ICC(3,1) — Consistency", f"{icc_con:.3f}" if icc_con == icc_con else "—")
    m2.caption("Due vie, valutatori fissi, **coerenza** (ignora bias di livello). Stesse soglie orientative dell’ICC(2,1).")
    m3.metric("Lin's CCC", f"{ccc:.3f}" if ccc == ccc else "—")
    m3.caption("Concordanza = correlazione × accuratezza; 1 = identità perfetta. Valori >0.90 tipicamente ottimi.")
    m4.metric("r di Pearson", f"{r:.3f}" if r == r else "—")
    m4.caption("Misura **associazione** lineare, **non** accordo. Un r alto non garantisce concordanza.")

    st.markdown("### 3) Grafici")
    g1, g2 = st.columns(2)
    with g1:
        if px is not None:
            fig = px.scatter(df_xy, x=x_col, y=y_col, trendline="ols", template="simple_white",
                             title="Scatter con retta OLS")
            fig.update_layout(xaxis_title=x_col, yaxis_title=y_col)
            st.plotly_chart(fig, use_container_width=True)
            st.caption(
                "Interpretazione: punti vicini alla **bisettrice** (y≈x) indicano accordo. "
                "Una **pendenza** OLS diversa da 1 suggerisce **bias di scala**; "
                "un **intercetta** distante da 0 suggerisce **bias di livello**."
            )
    with g2:
        fig_ba, stats_ba = bland_altman_figure(df_xy.iloc[:, 0], df_xy.iloc[:, 1])
        if fig_ba is not None:
            st.plotly_chart(fig_ba, use_container_width=True)
            st.caption(
                f"Bland–Altman: **Bias**={stats_ba['bias']:.3f}, **LoA**±1.96·SD=({stats_ba['loa_low']:.3f}, {stats_ba['loa_high']:.3f}).  "
                "L’**area** tra le LoA evidenzia l’intervallo atteso per ~95% delle differenze. "
                "Verificare eventuale **trend** delle differenze con la media (bias proporzionale): in tal caso considerare **trasformazioni** "
                "o modelli che tengano conto dell’eteroschedasticità."
            )

    with st.expander("📘 Come interpretare gli **indici** (continui)"):
        st.markdown(
            "**ICC(2,1) — Agreement**  \n"
            "- Valuta quanto i metodi forniscano **misure identiche** (accordo assoluto). Utile se i valutatori sono un **campione** da una popolazione più ampia.  \n"
            "- Soglie (orientative): <0.50 **scarso**, 0.50–0.75 **moderato**, 0.75–0.90 **buono**, >0.90 **eccellente**.\n\n"
            "**ICC(3,1) — Consistency**  \n"
            "- Valuta la **coerenza** ignorando le differenze sistematiche di livello tra valutatori (es. uno misura sempre +2).  \n"
            "- Indicata quando i valutatori sono **fissi** e interessa la coerenza interna.\n\n"
            "**Lin’s CCC**  \n"
            "- Integra correlazione e **accuratezza**: penalizza shift e cambi di scala.  \n"
            "- 1 = identità perfetta; valori elevati indicano alta **concordanza** (spesso atteso >0.90 in contesti clinici).\n\n"
            "**r di Pearson**  \n"
            "- Misura **associazione** lineare, non l’accordo; due metodi possono essere fortemente correlati ma con **bias** rilevante.\n\n"
            "**Bland–Altman**  \n"
            "- **Bias**: media delle differenze (A−B). **LoA**: bias ±1.96·SD, atteso coprire ~95% delle differenze.  \n"
            "- Valutare se le LoA sono **clinicamente accettabili**; controllare **trend** (bias proporzionale) e **varianza non costante**."
        )

# ──────────────────────────────────────────────────────────────────────────────
# Navigazione
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("---")
nav1, nav2 = st.columns(2)
with nav1:
    if st.button("⬅️ Torna: Test diagnostici", use_container_width=True, key=k("go_prev")):
        try:
            st.switch_page("pages/9_🔬_Analisi_Test_Diagnostici.py")
        except Exception:
            pass
with nav2:
    if st.button("➡️ Vai: Sopravvivenza", use_container_width=True, key=k("go_next")):
        for target in ["pages/11_🧭_Analisi_di_Sopravvivenza.py", "pages/11_🧭_Analisi_di_Sopravvivenza....py"]:
            try:
                st.switch_page(target)
                break
            except Exception:
                continue
