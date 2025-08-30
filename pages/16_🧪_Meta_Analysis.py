# -*- coding: utf-8 -*-
# pages/17_🧪_Meta_Analysis.py
from __future__ import annotations

import os
import re
import math
import numpy as np
import pandas as pd
import streamlit as st

# Plotly per grafici (forest/funnel)
try:
    import plotly.graph_objects as go
except Exception:
    go = None

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG E STATO
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="🧪 Meta-analisi", layout="wide")

KEY = "meta"
def k(x: str) -> str:
    return f"{KEY}_{x}"

# Data store coerente con gli altri moduli
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

ensure_initialized()
DF = get_active(required=True)

# ──────────────────────────────────────────────────────────────────────────────
# UTILITY
# ──────────────────────────────────────────────────────────────────────────────
def _list_pages():
    try:
        return sorted([f for f in os.listdir("pages") if f.endswith(".py")])
    except FileNotFoundError:
        return []

def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())

def safe_switch_by_tokens(primary_candidates: list[str], fallback_tokens: list[str]):
    files = _list_pages()
    for cand in primary_candidates or []:
        if cand in files:
            st.switch_page(os.path.join("pages", cand)); return
    toks = [_norm(t) for t in fallback_tokens]
    for f in files:
        nf = _norm(f)
        if all(t in nf for t in toks):
            st.switch_page(os.path.join("pages", f)); return
    st.error("Pagina richiesta non trovata nei file di /pages. Verificare i nomi reali.")

def ci_from(theta, se, level=0.95):
    z = {0.90:1.6449, 0.95:1.95996, 0.99:2.5758}.get(level, 1.95996)
    return theta - z*se, theta + z*se

def _safe_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan

# Effect size da dati grezzi (due bracci)
def hedges_g(m1, sd1, n1, m0, sd0, n0):
    """Hedges g (SMD) e varianza (Hedges 1981)."""
    s_p = np.sqrt(((n1-1)*sd1**2 + (n0-1)*sd0**2) / (n1 + n0 - 2))
    d = (m1 - m0) / s_p
    # correzione per small-sample (J)
    J = 1 - (3/(4*(n1+n0)-9))
    g = J * d
    # varianza di g (Hedges & Olkin)
    v = ( (n1+n0)/(n1*n0) + (g**2)/(2*(n1+n0-2)) ) * J**2
    return g, v

def log_rr(a1, b1, a0, b0, cc=0.5):
    """
    Log Risk Ratio con correzione di continuità opzionale.
    a1=eventi tratt; b1=non eventi tratt; a0=eventi controllo; b0=non eventi controllo
    """
    # Applico correzione solo se necessario
    if min(a1, b1, a0, b0) == 0:
        a1+=cc; b1+=cc; a0+=cc; b0+=cc
    rr = (a1/(a1+b1)) / (a0/(a0+b0))
    yi = np.log(rr)
    vi = 1/a1 - 1/(a1+b1) + 1/a0 - 1/(a0+b0)
    return yi, vi

def log_or(a1, b1, a0, b0, cc=0.5):
    """Log Odds Ratio con correzione di continuità opzionale."""
    if min(a1, b1, a0, b0) == 0:
        a1+=cc; b1+=cc; a0+=cc; b0+=cc
    orr = (a1*b0)/(a0*b1)
    yi = np.log(orr)
    vi = 1/a1 + 1/b1 + 1/a0 + 1/b0
    return yi, vi

def fixed_effect(yi, vi):
    wi = 1/vi
    mu = np.sum(wi*yi)/np.sum(wi)
    se = np.sqrt(1/np.sum(wi))
    Q = np.sum(wi*(yi - mu)**2)
    k = yi.size
    df = max(k-1, 1)
    I2 = max(0.0, (Q - df)/Q) if Q>0 else 0.0
    H2 = 1/(1-I2) if I2<1 else np.inf
    return dict(model="FE", mu=mu, se=se, Q=Q, df=df, I2=I2*100, H2=H2, tau2=0.0)

def dersimonian_laird(yi, vi):
    wi = 1/vi
    mu_FE = np.sum(wi*yi)/np.sum(wi)
    Q = np.sum(wi*(yi - mu_FE)**2)
    k = yi.size
    df = max(k-1, 1)
    c = np.sum(wi) - (np.sum(wi**2)/np.sum(wi))
    tau2 = max(0.0, (Q - df)/c) if c>0 else 0.0
    wi_star = 1/(vi + tau2)
    mu = np.sum(wi_star*yi)/np.sum(wi_star)
    se = np.sqrt(1/np.sum(wi_star))
    I2 = max(0.0, (Q - df)/Q) if Q>0 else 0.0
    H2 = 1/(1-I2) if I2<1 else np.inf
    return dict(model="RE (DL)", mu=mu, se=se, Q=Q, df=df, I2=I2*100, H2=H2, tau2=tau2)

def wls_meta_reg(yi, vi, X, tau2=0.0):
    """
    Meta-regressione WLS: y = X b + e, weights = 1 / (vi + tau2).
    Restituisce coef, se, t, pval, cov, fitted.
    """
    import numpy.linalg as LA
    w = 1.0 / (vi + tau2)
    W = np.diag(w)
    XtW = X.T @ W
    XtWX = XtW @ X
    XtWy = XtW @ yi.reshape(-1, 1)
    try:
        beta = LA.solve(XtWX, XtWy)  # (p x 1)
    except LA.LinAlgError:
        return None
    beta = beta.flatten()
    # Var(beta) = (X'WX)^(-1)
    covb = LA.inv(XtWX)
    se = np.sqrt(np.diag(covb))
    tvals = beta / se
    # approssimazione normale (k grande)
    from scipy.stats import t as tdist
    df = max(len(yi) - X.shape[1], 1)
    pvals = 2*(1 - tdist.cdf(np.abs(tvals), df=df))
    fitted = X @ beta
    return dict(beta=beta, se=se, t=tvals, p=pvals, cov=covb, fitted=fitted, df=df)

def egger_test(yi, vi):
    """
    Egger's regression test per funnel plot asymmetry.
    Regressione: zi = b0 + b1 * precisione, con zi = yi/sei, precisione = 1/sei.
    Test sull'intercetta b0 != 0.
    """
    sei = np.sqrt(vi)
    zi = yi / sei
    prec = 1 / sei
    X = np.column_stack([np.ones_like(prec), prec])
    # OLS
    XtX = X.T @ X
    XtY = X.T @ zi
    try:
        beta = np.linalg.solve(XtX, XtY)
    except np.linalg.LinAlgError:
        return None
    resid = zi - X @ beta
    s2 = np.sum(resid**2) / max(len(zi) - 2, 1)
    covb = s2 * np.linalg.inv(XtX)
    se_b0 = np.sqrt(covb[0,0])
    t0 = beta[0] / se_b0 if se_b0>0 else np.nan
    from scipy.stats import t as tdist
    df = max(len(zi) - 2, 1)
    p = 2 * (1 - tdist.cdf(abs(t0), df=df))
    return dict(intercept=beta[0], se=se_b0, t=t0, df=df, p=p)

def pretty_df(df: pd.DataFrame, title: str):
    st.markdown(f"**{title}**")
    st.dataframe(df, width="stretch")

# ──────────────────────────────────────────────────────────────────────────────
# HEADER E GUIDA
# ──────────────────────────────────────────────────────────────────────────────
st.title("🧪 Meta-analisi")
st.caption("Calcolo dell'effetto sintetico con modelli fixed/random, eterogeneità, forest & funnel, Egger, leave-one-out, subgroup e meta-regressione.")

with st.expander("Stato dati", expanded=False):
    stamp_meta()

if DF is None or DF.empty:
    st.stop()

all_cols = list(DF.columns)
num_cols = [c for c in all_cols if pd.api.types.is_numeric_dtype(DF[c])]
int_cols = [c for c in all_cols if pd.api.types.is_integer_dtype(DF[c])]

with st.expander("📌 Come impostare correttamente la meta-analisi (leggere prima)", expanded=True):
    st.markdown(
        "- **Scegliere la fonte dell'effect size**:\n"
        "   1) **Pre-calcolato** (colonne con *yi* e *SE* o *Var*),\n"
        "   2) **Continuo** (due bracci: medie, SD, n → **Hedges g**),\n"
        "   3) **Binario** (due bracci: eventi/non-eventi → **log RR** o **log OR**).\n"
        "- **Modello**: *Fixed effect* (eterogeneità trascurabile) o *Random effects* (**DerSimonian–Laird** default) quando attesa variabilità tra studi.\n"
        "- **Interpretazione**:\n"
        "   - **Q** alto e **I²** elevato ⇒ forte eterogeneità; **τ²** è la varianza tra-studi.\n"
        "   - **Forest plot**: ciascuna riga è uno studio (IC al 95%), rombo = stima combinata.\n"
        "   - **Funnel plot**: simmetria attesa; asimmetria (test di **Egger**) può indicare bias di pubblicazione.\n"
        "- **Complete-case**: le righe con campi mancanti essenziali vengono escluse automaticamente."
    )

# ──────────────────────────────────────────────────────────────────────────────
# 1) SELEZIONE TIPO DATI / MAPPATURA COLONNE
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("### 1) Sorgente dell'effect size e mappatura colonne")

# Reset sicuro delle form key (se richiesto)
if st.session_state.get(k("reset_forms"), False):
    for fld in [k("study"), k("es_mode"), k("yi_col"), k("se_col"), k("var_col"),
                k("m1"), k("sd1"), k("n1"), k("m0"), k("sd0"), k("n0"),
                k("e1"), k("ne1"), k("e0"), k("ne0"), k("bin_es")]:
        st.session_state.pop(fld, None)
    st.session_state[k("reset_forms")] = False

c1, c2 = st.columns([1.5, 1.5])
with c1:
    study_col = st.selectbox("Colonna identificatore studio", options=all_cols, key=k("study"))
with c2:
    es_mode = st.radio(
        "Origine dell'effect size",
        ["Pre-calcolato", "Continuo (due bracci)", "Binario (due bracci)"],
        key=k("es_mode")
    )

raw_config = {}
if es_mode == "Pre-calcolato":
    c1, c2, c3 = st.columns([1.2, 1.2, 1.2])
    with c1:
        yi_col = st.selectbox("Colonna effetto (yi)", options=num_cols, key=k("yi_col"))
    with c2:
        se_col = st.selectbox("Colonna SE (se disponibile)", options=["—"] + num_cols, key=k("se_col"))
    with c3:
        var_col = st.selectbox("Colonna Var (se non c'è SE)", options=["—"] + num_cols, key=k("var_col"))
elif es_mode == "Continuo (due bracci)":
    c1, c2, c3 = st.columns([1.2, 1.2, 1.2])
    with c1:
        m1 = st.selectbox("Media tratt.", options=num_cols, key=k("m1"))
        m0 = st.selectbox("Media controllo", options=num_cols, key=k("m0"))
    with c2:
        sd1 = st.selectbox("SD tratt.", options=num_cols, key=k("sd1"))
        sd0 = st.selectbox("SD controllo", options=num_cols, key=k("sd0"))
    with c3:
        n1 = st.selectbox("N tratt.", options=num_cols + int_cols, key=k("n1"))
        n0 = st.selectbox("N controllo", options=num_cols + int_cols, key=k("n0"))
    st.caption("L'effect size calcolato è **Hedges g** (SMD) con varianza corretta per small-sample.")
else:  # Binario
    c1, c2, c3 = st.columns([1.2, 1.2, 1.2])
    with c1:
        e1 = st.selectbox("Eventi tratt.", options=num_cols + int_cols, key=k("e1"))
        ne1 = st.selectbox("Non-eventi tratt.", options=num_cols + int_cols, key=k("ne1"))
    with c2:
        e0 = st.selectbox("Eventi controllo", options=num_cols + int_cols, key=k("e0"))
        ne0 = st.selectbox("Non-eventi controllo", options=num_cols + int_cols, key=k("ne0"))
    with c3:
        bin_es = st.radio("Scelta effetto", ["log RR", "log OR"], key=k("bin_es"))
    st.caption("È applicata automaticamente una **correzione di continuità** (0.5) se compaiono zeri.")

# Moderatori opzionali
st.markdown("### 2) Moderatori (opzionali)")
c1, c2 = st.columns([1.2, 1.2])
with c1:
    subgroup_col = st.selectbox("Moderatore **categorico** per subgroup (opzionale)", options=["—"] + all_cols, key=k("subg"))
with c2:
    meta_reg_col = st.selectbox("Moderatore **numerico** per meta-regressione (opzionale)", options=["—"] + num_cols, key=k("metareg"))

# ──────────────────────────────────────────────────────────────────────────────
# 3) COSTRUZIONE DEL DATASET DI EFFECT SIZE
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("### 3) Costruzione effect size per studio")

df = DF.copy()
df = df.dropna(subset=[study_col])  # serve un ID studio
rows = []
errors = []
for idx, row in df.iterrows():
    sid = str(row[study_col])
    try:
        if es_mode == "Pre-calcolato":
            yi = _safe_float(row[st.session_state[k("yi_col")]])
            if st.session_state[k("se_col")] != "—":
                se = _safe_float(row[st.session_state[k("se_col")]])
                vi = se**2
            elif st.session_state[k("var_col")] != "—":
                vi = _safe_float(row[st.session_state[k("var_col")]])
                se = np.sqrt(vi)
            else:
                raise ValueError("Fornire SE o Var.")
        elif es_mode == "Continuo (due bracci)":
            m1v = _safe_float(row[st.session_state[k("m1")]])
            sd1v = _safe_float(row[st.session_state[k("sd1")]])
            n1v = _safe_float(row[st.session_state[k("n1")]])
            m0v = _safe_float(row[st.session_state[k("m0")]])
            sd0v = _safe_float(row[st.session_state[k("sd0")]])
            n0v = _safe_float(row[st.session_state[k("n0")]])
            yi, vi = hedges_g(m1v, sd1v, n1v, m0v, sd0v, n0v)
            se = np.sqrt(vi)
        else:  # Binario
            a1 = _safe_float(row[st.session_state[k("e1")]])
            b1 = _safe_float(row[st.session_state[k("ne1")]])
            a0 = _safe_float(row[st.session_state[k("e0")]])
            b0 = _safe_float(row[st.session_state[k("ne0")]])
            if st.session_state[k("bin_es")] == "log RR":
                yi, vi = log_rr(a1, b1, a0, b0)
            else:
                yi, vi = log_or(a1, b1, a0, b0)
            se = np.sqrt(vi)

        # Moderatori
        subg = None if st.session_state[k("subg")] == "—" else row[st.session_state[k("subg")]]
        metareg = np.nan if st.session_state[k("metareg")] == "—" else _safe_float(row[st.session_state[k("metareg")]])

        if np.isfinite(yi) and np.isfinite(vi) and vi > 0:
            rows.append(dict(study=sid, yi=yi, vi=vi, se=se, subgroup=subg, moderator=metareg))
        else:
            errors.append((sid, "Effect size o varianza non finiti/validi."))
    except Exception as e:
        errors.append((sid, str(e)))

ES = pd.DataFrame(rows)
if ES.empty:
    st.error("Nessuno studio valido dopo il controllo dei campi richiesti. Verificare le colonne selezionate.")
    if errors:
        st.warning("Esempi di errori:\n- " + "\n- ".join([f"{sid}: {msg}" for sid, msg in errors[:5]]))
    st.stop()

pretty_df(ES, "Effect size per studio (yi e varianza)")

if errors:
    with st.expander("Righe escluse e motivazione"):
        bad = pd.DataFrame(errors, columns=["study", "motivo"])
        st.dataframe(bad, width="stretch")

# ──────────────────────────────────────────────────────────────────────────────
# 4) MODELLO DI META-ANALISI
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("### 4) Modello di sintesi")

c1, c2 = st.columns([1.0, 1.0])
with c1:
    model_type = st.radio("Modello", ["Fixed effect", "Random effects (DL)"], key=k("model"))
with c2:
    conf = st.selectbox("Intervallo di confidenza", options=[0.95, 0.90, 0.99], index=0, key=k("ci"))

yi = ES["yi"].to_numpy()
vi = ES["vi"].to_numpy()
k_studies = yi.size

if model_type == "Fixed effect":
    res = fixed_effect(yi, vi)
else:
    res = dersimonian_laird(yi, vi)

mu_hat = res["mu"]
se_hat = res["se"]
lo, hi = ci_from(mu_hat, se_hat, level=conf)
summary_table = pd.DataFrame({
    "Modello": [res["model"]],
    "k": [k_studies],
    "Effetto combinato (μ̂)": [mu_hat],
    f"CI {int(conf*100)}% inf": [lo],
    f"CI {int(conf*100)}% sup": [hi],
    "τ²": [res["tau2"]],
    "Q": [res["Q"]],
    "df": [res["df"]],
    "I² (%)": [res["I2"]],
    "H²": [res["H2"]],
})
pretty_df(summary_table.round(4), "Sintesi")

with st.expander("📝 Come leggere questi risultati", expanded=True):
    st.markdown(
        "- **μ̂**: stima dell’effetto complessivo (in unità dell’effetto scelto: g, log RR/OR, ecc.).\n"
        "- **CI**: intervallo di confidenza; se non include 0 ⇒ effetto statisticamente diverso da 0.\n"
        "- **Q** (Cochran): grande ⇒ eterogeneità; confrontato con χ²(df=k−1).\n"
        "- **I²**: % di variabilità dovuta a eterogeneità (≈25% bassa, 50% moderata, 75% alta).\n"
        "- **τ²**: varianza tra-studi; usata nel modello random per pesi 1/(vᵢ+τ²)."
    )

# ──────────────────────────────────────────────────────────────────────────────
# 5) FOREST PLOT
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("### 5) Forest plot")
if go is None:
    st.info("Plotly non disponibile: impossibile disegnare il forest plot.")
else:
    # Ordine: per default ordino per precisione decrescente
    ES_plot = ES.copy()
    ES_plot["w"] = 1/(ES_plot["vi"] + (res["tau2"] if model_type.startswith("Random") else 0))
    ES_plot = ES_plot.sort_values("w", ascending=False).reset_index(drop=True)
    ES_plot["ypos"] = np.arange(len(ES_plot))[::-1]  # dall'alto verso il basso

    # CI per studio
    z = {0.90:1.6449, 0.95:1.95996, 0.99:2.5758}.get(conf, 1.95996)
    ES_plot["lo"] = ES_plot["yi"] - z*ES_plot["se"]
    ES_plot["hi"] = ES_plot["yi"] + z*ES_plot["se"]

    fig = go.Figure()

    # Barre CI + punti per ogni studio
    for _, r in ES_plot.iterrows():
        fig.add_shape(type="line", x0=r["lo"], x1=r["hi"], y0=r["ypos"], y1=r["ypos"])
        fig.add_trace(go.Scatter(
            x=[r["yi"]], y=[r["ypos"]],
            mode="markers",
            marker=dict(size=8),
            name=str(r["study"]),
            hovertemplate=f"Studio: {r['study']}<br>yi: {r['yi']:.3f}<br>SE: {r['se']:.3f}<br>CI: [{r['lo']:.3f}, {r['hi']:.3f}]<extra></extra>"
        ))

    # Rombo della sintesi
    y_sum = -1.5
    fig.add_trace(go.Scatter(
        x=[mu_hat, lo, mu_hat, hi],
        y=[y_sum, y_sum-0.5, y_sum-1.0, y_sum-0.5],
        mode="lines+markers",
        line=dict(shape="linear"),
        marker=dict(size=0),
        showlegend=False,
        hoverinfo="skip"
    ))
    fig.add_annotation(x=mu_hat, y=y_sum-0.5, text="Sintesi", showarrow=False, yshift=-6)

    # Asse Y come etichette studio
    yticks = list(ES_plot["ypos"]) + [y_sum-0.5]
    ytext = list(ES_plot["study"]) + ["Sintesi"]
    fig.update_yaxes(tickvals=yticks, ticktext=ytext, autorange="reversed")

    fig.update_layout(
        title=f"Forest plot ({res['model']})",
        xaxis_title="Effect size (yi)",
        height=400 + 20*len(ES_plot),
        margin=dict(l=10, r=10, t=60, b=10)
    )
    st.plotly_chart(fig, width="stretch")

with st.expander("📝 Come leggere il forest plot", expanded=False):
    st.markdown(
        "Ogni riga è uno studio con il proprio **effect size** (punto) e **IC** (segmento). "
        "Il **rombo** in basso rappresenta la stima combinata e il suo IC. "
        "Se molti IC non si sovrappongono o l’eterogeneità (I²) è alta, preferire il modello **random**."
    )

# ──────────────────────────────────────────────────────────────────────────────
# 6) FUNNEL PLOT + TEST DI EGGER
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("### 6) Funnel plot e test di Egger")
if go is None:
    st.info("Plotly non disponibile: impossibile disegnare il funnel plot.")
else:
    sei = np.sqrt(ES["vi"].to_numpy())
    x_center = mu_hat
    # Limiti del funnel ±z*SE
    x_left = x_center - z*sei
    x_right = x_center + z*sei

    fig2 = go.Figure()
    # Triangolo (area attesa a 95%)
    order = np.argsort(sei)[::-1]  # dal SE più grande (in basso)
    fig2.add_trace(go.Scatter(
        x=np.concatenate([x_left[order], x_right[order][::-1]]),
        y=np.concatenate([sei[order], sei[order][::-1]]),
        fill="toself", mode="lines", line=dict(width=0.5), name=f"95% atteso"
    ))
    # Punti studi
    fig2.add_trace(go.Scatter(
        x=ES["yi"], y=sei,
        mode="markers", name="Studi",
        hovertemplate="yi: %{x:.3f}<br>SE: %{y:.3f}<extra></extra>"
    ))
    fig2.update_yaxes(autorange="reversed")
    fig2.update_layout(
        title="Funnel plot",
        xaxis_title="Effect size (yi)",
        yaxis_title="Errore standard (SE)",
        height=500, margin=dict(l=10, r=10, t=60, b=10),
        showlegend=True
    )
    st.plotly_chart(fig2, width="stretch")

    # Egger
    eg = egger_test(yi, vi)
    if eg is None:
        st.info("Impossibile eseguire il test di Egger (matrice singolare).")
    else:
        trow = pd.DataFrame({
            "Intercetta (bias)": [eg["intercept"]],
            "SE": [eg["se"]],
            "t": [eg["t"]],
            "df": [eg["df"]],
            "p-value": [eg["p"]],
        })
        pretty_df(trow.round(4), "Test di Egger (asimmetria)")

with st.expander("📝 Come leggere il funnel plot", expanded=False):
    st.markdown(
        "In assenza di bias di pubblicazione gli studi (punti) sono **simmetrici** attorno all’effetto combinato: "
        "con **SE** bassi (in alto) i punti sono più concentrati; con SE alti (in basso) più dispersi. "
        "Un **test di Egger** con *p* piccola suggerisce **asimmetria** (possibile bias di pubblicazione)."
    )

# ──────────────────────────────────────────────────────────────────────────────
# 7) SUBGROUP E META-REGRESSIONE
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("### 7) Subgroup e Meta-regressione")

# Subgroup (categorico)
if subgroup_col != "—":
    st.markdown("**Subgroup analysis**")
    subrows = []
    for g, gdf in ES.groupby("subgroup", dropna=False):
        y = gdf["yi"].to_numpy()
        v = gdf["vi"].to_numpy()
        if y.size < 2:
            continue
        resg = dersimonian_laird(y, v) if model_type.startswith("Random") else fixed_effect(y, v)
        mu_g, se_g = resg["mu"], resg["se"]
        lo_g, hi_g = ci_from(mu_g, se_g, level=conf)
        subrows.append(dict(
            Gruppo=str(g),
            k=y.size,
            Effetto=mu_g, **{f"CI {int(conf*100)}% inf": lo_g, f"CI {int(conf*100)}% sup": hi_g},
            Q=resg["Q"], df=resg["df"], I2=resg["I2"], tau2=resg["tau2"]
        ))
    if subrows:
        pretty_df(pd.DataFrame(subrows).round(4), "Risultati per sottogruppo")
    else:
        st.info("Sottogruppi con meno di 2 studi non vengono sintetizzati.")

# Meta-regressione (numerico)
if meta_reg_col != "—":
    st.markdown("**Meta-regressione (WLS)**")
    MR = ES.dropna(subset=["moderator"]).copy()
    if MR.shape[0] >= 3:
        X = np.column_stack([np.ones(MR.shape[0]), MR["moderator"].to_numpy()])
        y = MR["yi"].to_numpy()
        v = MR["vi"].to_numpy()
        tau2_use = res["tau2"] if model_type.startswith("Random") else 0.0
        out = wls_meta_reg(y, v, X, tau2=tau2_use)
        if out is None:
            st.info("Meta-regressione non stimabile (matrice quasi singolare).")
        else:
            tab = pd.DataFrame({
                "Termine": ["Intercept", f"Moderatore ({meta_reg_col})"],
                "β": out["beta"], "SE": out["se"], "t": out["t"], "p-value": out["p"]
            })
            pretty_df(tab.round(4), "Risultati WLS")
            with st.expander("📝 Interpretazione meta-regressione", expanded=False):
                st.markdown(
                    "La meta-regressione valuta se l’effect size varia **linearmente** con il moderatore. "
                    "Il coefficiente del moderatore indica l’incremento atteso di **yi** per unità di moderatore "
                    "(p-value piccolo ⇒ evidenza di associazione)."
                )
    else:
        st.info("Servono almeno 3 studi con moderatore non mancante per stimare la meta-regressione.")

# Leave-one-out
with st.expander("Analisi leave-one-out (influenza di singoli studi)", expanded=False):
    loo = []
    for i in range(k_studies):
        mask = np.ones(k_studies, dtype=bool); mask[i] = False
        yi_ = yi[mask]; vi_ = vi[mask]
        if yi_.size < 2:
            continue
        res_ = dersimonian_laird(yi_, vi_) if model_type.startswith("Random") else fixed_effect(yi_, vi_)
        loo.append(dict(studio_rimosso=ES["study"].iloc[i], mu=res_["mu"], tau2=res_["tau2"], I2=res_["I2"]))
    if loo:
        pretty_df(pd.DataFrame(loo).round(4), "Stima combinata rimuovendo ogni studio a turno")
    else:
        st.info("Non calcolabile (troppo pochi studi).")

# ──────────────────────────────────────────────────────────────────────────────
# 8) NOTE DI INTERPRETAZIONE (RIEPILOGO)
# ──────────────────────────────────────────────────────────────────────────────
with st.expander("📝 Riepilogo interpretativo", expanded=False):
    st.markdown(
        "- **Scelta modello**: usare **random** quando c’è eterogeneità sostanziale (I² elevato) o variabilità attesa tra studi.\n"
        "- **Sottogruppi**: differenze tra sottogruppi suggeriscono fonti d’eterogeneità (attenzione a test multipli).\n"
        "- **Bias di pubblicazione**: funnel asimmetrico / Egger significativo ⇒ possibile bias; interpretare con cautela.\n"
        "- **Misure**: per **log RR/OR** l’interpretazione è sull’esponenziale (RR/OR); per **g** (SMD), soglie orientative: 0.2 piccolo, 0.5 medio, 0.8 grande."
    )

# ──────────────────────────────────────────────────────────────────────────────
# NAVIGAZIONE
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("---")
nav1, nav2, nav3 = st.columns(3)
with nav1:
    if st.button("⬅️ Torna: SEM", key=k("go_prev")):
        safe_switch_by_tokens(
            primary_candidates=["15_🧩_SEM_Structural_Equation_Modeling.py"],
            fallback_tokens=["sem", "structural", "equation"]
        )
with nav2:
    if st.button("↔️ Vai: Serie Temporali", key=k("go_ts")):
        safe_switch_by_tokens(
            primary_candidates=[
                "14_⏱️_Analisi_Serie_Temporali.py", "14_⏱️_Analisi_Serie_Temporali.py"
            ],
            fallback_tokens=["analisi", "serie", "temporali"]
        )
with nav3:
    if st.button("➡️ Vai: Report / Export", key=k("go_report")):
        safe_switch_by_tokens(
            primary_candidates=[
                "17_🧾_Glossario.py",
            ],
            fallback_tokens=["report", "export"]
        )
