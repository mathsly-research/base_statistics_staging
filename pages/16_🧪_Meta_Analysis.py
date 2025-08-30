# -*- coding: utf-8 -*-
# pages/17_🧪_Meta_Analysis.py
from __future__ import annotations

import os, re, math
import numpy as np
import pandas as pd
import streamlit as st

# Plotly per forest/funnel (richiede plotly>=5)
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# SciPy per alcuni p-value (fallback se assente)
try:
    from scipy.stats import t as tdist
except Exception:  # fallback normale
    tdist = None

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG E STATO
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="🧪 Meta-analisi — Guida passo-passo", layout="wide")

KEY = "meta"
def k(x: str) -> str: return f"{KEY}_{x}"

# Data store coerente con gli altri moduli (fallback se non presente)
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
            st.error("Nessun dataset attivo. Importi i dati nella pagina di Upload e riprovi.")
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
# HELPER
# ──────────────────────────────────────────────────────────────────────────────
def _list_pages():
    try: return sorted([f for f in os.listdir("pages") if f.endswith(".py")])
    except FileNotFoundError: return []
def _norm(s: str) -> str: return re.sub(r"[^a-z0-9]+", "", str(s).lower())
def safe_switch_by_tokens(primary: list[str], tokens: list[str]):
    files = _list_pages()
    for p in primary or []:
        if p in files:
            st.switch_page(os.path.join("pages", p)); return
    toks = [_norm(t) for t in tokens]
    for f in files:
        if all(t in _norm(f) for t in toks):
            st.switch_page(os.path.join("pages", f)); return
    st.error("Pagina di destinazione non trovata in /pages.")

def z_for(level: float) -> float:
    return {0.90:1.644853, 0.95:1.959964, 0.99:2.575829}.get(float(level), 1.959964)

def ci_from(theta, se, level=0.95):
    z = z_for(level); return theta - z*se, theta + z*se

def _safe_float(x):
    try: return float(x)
    except Exception: return np.nan

# Effect size (due bracci continui)
def hedges_g(m1, sd1, n1, m0, sd0, n0):
    s_p = np.sqrt(((n1-1)*sd1**2 + (n0-1)*sd0**2) / max(n1 + n0 - 2, 1))
    d = (m1 - m0) / s_p if s_p>0 else np.nan
    J = 1 - (3/(4*(n1+n0)-9)) if (n1+n0)>2 else 1.0
    g = J * d
    v = ( (n1+n0)/(n1*n0) + (g**2)/(2*max(n1+n0-2,1)) ) * (J**2)
    return g, v

# Effect size (due bracci binari)
def log_rr(a1, b1, a0, b0, cc=0.5):
    if min(a1,b1,a0,b0) == 0: a1+=cc; b1+=cc; a0+=cc; b0+=cc
    rr = (a1/(a1+b1)) / (a0/(a0+b0)); yi = np.log(rr)
    vi = 1/a1 - 1/(a1+b1) + 1/a0 - 1/(a0+b0)
    return yi, vi
def log_or(a1, b1, a0, b0, cc=0.5):
    if min(a1,b1,a0,b0) == 0: a1+=cc; b1+=cc; a0+=cc; b0+=cc
    orr = (a1*b0)/(a0*b1); yi = np.log(orr)
    vi = 1/a1 + 1/b1 + 1/a0 + 1/b0
    return yi, vi

# Fixed e Random (DerSimonian–Laird)
def fixed_effect(yi, vi):
    wi = 1/vi; mu = np.sum(wi*yi)/np.sum(wi)
    se = np.sqrt(1/np.sum(wi)); Q = np.sum(wi*(yi - mu)**2)
    k = yi.size; df = max(k-1, 1)
    I2 = max(0.0, (Q - df)/Q) if Q>0 else 0.0; H2 = 1/(1-I2) if I2<1 else np.inf
    return dict(model="FE", mu=mu, se=se, Q=Q, df=df, I2=I2*100, H2=H2, tau2=0.0)
def dersimonian_laird(yi, vi):
    wi = 1/vi; mu_FE = np.sum(wi*yi)/np.sum(wi)
    Q = np.sum(wi*(yi - mu_FE)**2); k = yi.size; df = max(k-1, 1)
    c = np.sum(wi) - (np.sum(wi**2)/np.sum(wi))
    tau2 = max(0.0, (Q - df)/c) if c>0 else 0.0
    wi_star = 1/(vi + tau2)
    mu = np.sum(wi_star*yi)/np.sum(wi_star); se = np.sqrt(1/np.sum(wi_star))
    I2 = max(0.0, (Q - df)/Q) if Q>0 else 0.0; H2 = 1/(1-I2) if I2<1 else np.inf
    return dict(model="RE (DL)", mu=mu, se=se, Q=Q, df=df, I2=I2*100, H2=H2, tau2=tau2)

def egger_test(yi, vi):
    sei = np.sqrt(vi); zi = yi / sei; prec = 1 / sei
    X = np.column_stack([np.ones_like(prec), prec])
    XtX = X.T @ X; XtY = X.T @ zi
    try: beta = np.linalg.solve(XtX, XtY)
    except np.linalg.LinAlgError: return None
    resid = zi - X @ beta; df = max(len(zi)-2, 1)
    s2 = float(np.sum(resid**2) / df)
    covb = s2 * np.linalg.inv(XtX)
    se_b0 = float(np.sqrt(covb[0,0])); t0 = float(beta[0]/se_b0) if se_b0>0 else np.nan
    if tdist is None:
        # approx normale
        p = float(2*(1 - 0.5*(1+math.erf(abs(t0)/np.sqrt(2)))))
    else:
        p = float(2*(1 - tdist.cdf(abs(t0), df=df)))
    return dict(intercept=float(beta[0]), se=se_b0, t=t0, df=df, p=p)

def pretty_df(df: pd.DataFrame, title: str):
    st.markdown(f"**{title}**")
    st.dataframe(df, width="stretch")

# ──────────────────────────────────────────────────────────────────────────────
# HEADER + GUIDA
# ──────────────────────────────────────────────────────────────────────────────
st.title("🧪 Meta-analisi")
with st.expander("Stato dati", expanded=False): stamp_meta()

with st.expander("📋 Guida rapida (leggimi prima)", expanded=True):
    st.markdown(
        "1. **Scegli il tipo di dati**: effect size già calcolati **oppure** dati grezzi a **due bracci** (continuo/binario).  \n"
        "2. **Mappa le colonne** richieste: il pannello mostra solo i campi necessari al caso scelto.  \n"
        "3. Seleziona **modello** (Fixed o Random-DL) e **confidenza**.  \n"
        "4. Leggi **sintesi** (μ̂, IC, Q, τ², I²) e verifica **forest plot** (diamante verde) e **funnel plot** (triangolo al 95%).  \n"
        "5. Usa **Subgroup** (categorico) e **Meta-regressione** (numerico) per esplorare l’eterogeneità.  \n\n"
        "💡 *Regola pratica*: con **I² alto** o studi eterogenei, preferire **Random effects**."
    )

if DF is None or DF.empty: st.stop()
all_cols = list(DF.columns)
num_cols = [c for c in all_cols if pd.api.types.is_numeric_dtype(DF[c])]
int_cols = [c for c in all_cols if pd.api.types.is_integer_dtype(DF[c])]

# ──────────────────────────────────────────────────────────────────────────────
# STEP 1 — SCELTA ORIGINE EFFECT SIZE E MAPPATURA
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("### 1) Scegli dati e mappa le colonne")

c0, c1 = st.columns([1.2, 1.2])
with c0:
    study_col = st.selectbox("Colonna identificatore **Studio**", options=all_cols, key=k("study"))
with c1:
    es_mode = st.radio(
        "Origine dell’effect size",
        ["Pre-calcolato (yi + SE/Var)",
         "Continuo (due bracci: medie/SD/n)",
         "Binario (due bracci: eventi/non-eventi)"],
        key=k("es_mode")
    )

if es_mode.startswith("Pre-calcolato"):
    c1, c2, c3 = st.columns([1.2, 1.2, 1.2])
    with c1: yi_col = st.selectbox("Colonna **yi**", options=num_cols, key=k("yi"))
    with c2: se_col = st.selectbox("Colonna **SE** (se presente)", options=["—"]+num_cols, key=k("se"))
    with c3: var_col = st.selectbox("Colonna **Var** (se non c’è SE)", options=["—"]+num_cols, key=k("var"))
elif es_mode.startswith("Continuo"):
    c1, c2, c3 = st.columns([1.2, 1.2, 1.2])
    with c1:
        m1 = st.selectbox("Media **tratt.**", options=num_cols, key=k("m1"))
        m0 = st.selectbox("Media **controllo**", options=num_cols, key=k("m0"))
    with c2:
        sd1 = st.selectbox("SD **tratt.**", options=num_cols, key=k("sd1"))
        sd0 = st.selectbox("SD **controllo**", options=num_cols, key=k("sd0"))
    with c3:
        n1 = st.selectbox("N **tratt.**", options=num_cols+int_cols, key=k("n1"))
        n0 = st.selectbox("N **controllo**", options=num_cols+int_cols, key=k("n0"))
    st.caption("Verrà calcolato **Hedges g** (SMD) con correzione small-sample.")
else:
    c1, c2, c3 = st.columns([1.2, 1.2, 1.2])
    with c1:
        e1 = st.selectbox("**Eventi** tratt.", options=num_cols+int_cols, key=k("e1"))
        ne1 = st.selectbox("**Non-eventi** tratt.", options=num_cols+int_cols, key=k("ne1"))
    with c2:
        e0 = st.selectbox("**Eventi** controllo", options=num_cols+int_cols, key=k("e0"))
        ne0 = st.selectbox("**Non-eventi** controllo", options=num_cols+int_cols, key=k("ne0"))
    with c3:
        bin_es = st.radio("Tipo effetto", ["log RR", "log OR"], horizontal=True, key=k("bin_es"))
    st.caption("Se compaiono zeri viene applicata **correzione di continuità** (0.5).")

# Moderatori (facoltativi)
st.markdown("### 2) Moderatori (facoltativi)")
c1, c2 = st.columns([1.2, 1.2])
with c1: subgroup_col = st.selectbox("Moderatore **categorico** (Subgroup)", options=["—"]+all_cols, key=k("subg"))
with c2: metareg_col = st.selectbox("Moderatore **numerico** (Meta-regressione)", options=["—"]+num_cols, key=k("metareg"))

# ──────────────────────────────────────────────────────────────────────────────
# COSTRUZIONE EFFECT SIZE
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("### 3) Costruzione effect size per studio")

df0 = DF.dropna(subset=[study_col]).copy()
rows, errors = [], []
for idx, r in df0.iterrows():
    sid = str(r[study_col])
    try:
        if es_mode.startswith("Pre-calcolato"):
            yi = _safe_float(r[st.session_state[k("yi")]])
            if st.session_state[k("se")] != "—":
                se = _safe_float(r[st.session_state[k("se")]]); vi = se**2
            elif st.session_state[k("var")] != "—":
                vi = _safe_float(r[st.session_state[k("var")]]); se = np.sqrt(vi)
            else:
                raise ValueError("Fornire **SE** o **Var**.")
            meta_type = "pre"
            extra = dict()
        elif es_mode.startswith("Continuo"):
            m1v=_safe_float(r[st.session_state[k("m1")]]); sd1v=_safe_float(r[st.session_state[k("sd1")]])
            n1v=_safe_float(r[st.session_state[k("n1")]]); m0v=_safe_float(r[st.session_state[k("m0")]])
            sd0v=_safe_float(r[st.session_state[k("sd0")]]); n0v=_safe_float(r[st.session_state[k("n0")]])
            yi, vi = hedges_g(m1v, sd1v, n1v, m0v, sd0v, n0v); se = np.sqrt(vi)
            meta_type = "continuous"
            extra = dict(m1=m1v, sd1=sd1v, n1=n1v, m0=m0v, sd0=sd0v, n0=n0v)
        else:
            a1=_safe_float(r[st.session_state[k("e1")]]); b1=_safe_float(r[st.session_state[k("ne1")]])
            a0=_safe_float(r[st.session_state[k("e0")]]); b0=_safe_float(r[st.session_state[k("ne0")]])
            yi, vi = (log_rr(a1,b1,a0,b0) if st.session_state[k("bin_es")]=="log RR" else log_or(a1,b1,a0,b0))
            se = np.sqrt(vi); meta_type = "binary"
            extra = dict(e1=a1, ne1=b1, e0=a0, ne0=b0)

        if not np.isfinite(yi) or not np.isfinite(vi) or vi<=0:
            raise ValueError("yi/Var non validi.")
        subgroup = None if st.session_state[k("subg")]=="—" else r[st.session_state[k("subg")]]
        moderator = np.nan if st.session_state[k("metareg")]=="—" else _safe_float(r[st.session_state[k("metareg")]])
        rows.append(dict(study=sid, yi=yi, vi=vi, se=se, type=meta_type,
                         subgroup=subgroup, moderator=moderator, **extra))
    except Exception as e:
        errors.append((sid, str(e)))

ES = pd.DataFrame(rows)
if ES.empty:
    st.error("Nessuno studio valido dopo la costruzione degli effect size. Verificare la mappatura.")
    if errors: st.warning("Esempi di errori:\n- " + "\n- ".join([f"{s}: {m}" for s,m in errors[:6]]))
    st.stop()

st.dataframe(ES.head(15), width="stretch")
if errors:
    with st.expander("Righe escluse e motivazione"):
        st.dataframe(pd.DataFrame(errors, columns=["study","motivo"]), width="stretch")

# ──────────────────────────────────────────────────────────────────────────────
# MODELLO E SINTESI
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("### 4) Modello e sintesi complessiva")

colA, colB, colC = st.columns([1.0, 1.0, 1.0])
with colA:
    model_type = st.radio("Modello di combinazione", ["Fixed effect", "Random effects (DL)"], key=k("model"))
with colB:
    conf = st.selectbox("Intervallo di confidenza", [0.95, 0.90, 0.99], index=0, key=k("ci"))
with colC:
    show_exp = st.checkbox("Per effetti log (RR/OR) mostra anche l’esponenziale", value=True, key=k("exp"))

yi = ES["yi"].to_numpy(); vi = ES["vi"].to_numpy()
res = fixed_effect(yi, vi) if model_type.startswith("Fixed") else dersimonian_laird(yi, vi)
mu_hat, se_hat = res["mu"], res["se"]
lo, hi = ci_from(mu_hat, se_hat, level=conf)
summary = pd.DataFrame({
    "Modello":[res["model"]], "k":[len(yi)], "μ̂":[mu_hat],
    f"CI {int(conf*100)}% inf":[lo], f"CI {int(conf*100)}% sup":[hi],
    "τ²":[res["tau2"]], "Q":[res["Q"]], "df":[res["df"]],
    "I² (%)":[res["I2"]], "H²":[res["H2"]]
})
pretty_df(summary.round(4), "Sintesi (effetto complessivo e eterogeneità)")

with st.expander("📝 Come leggere questi numeri", expanded=True):
    st.markdown(
        "- **μ̂**: stima combinata; l’IC che **non include 0** indica effetto ≠ 0.  \n"
        "- **Q/df, I², τ²**: più sono alti ⇒ maggiore **eterogeneità** tra studi.  \n"
        "- **Fixed** da usare con eterogeneità bassa; **Random-DL** quando la variabilità tra studi è attesa/rilevante."
    )

# ──────────────────────────────────────────────────────────────────────────────
# 5) Forest plot (compatibile Plotly: niente add_vline, uso add_shape con xref/yref)
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("### 5) Forest plot")

tau2_use = res["tau2"] if model_type.startswith("Random") else 0.0
ES_plot = ES.copy()
ES_plot["weight"] = 1/(ES_plot["vi"] + tau2_use)
ES_plot["w_pct"] = 100*ES_plot["weight"]/ES_plot["weight"].sum()
ES_plot = ES_plot.sort_values("weight", ascending=False).reset_index(drop=True)

z = z_for(conf)
ES_plot["lo"] = ES_plot["yi"] - z*ES_plot["se"]
ES_plot["hi"] = ES_plot["yi"] + z*ES_plot["se"]
ES_plot["ypos"] = np.arange(len(ES_plot))[::-1]  # dall’alto verso il basso

def _fmt(v, d=2):
    return "—" if (v is None or (isinstance(v, float) and not np.isfinite(v))) else f"{v:.{d}f}"
def _as_str_cont(r):
    if r["type"]!="continuous": return "—","—","—","—","—","—"
    return int(r["n1"]), _fmt(r["m1"]), _fmt(r["sd1"]), int(r["n0"]), _fmt(r["m0"]), _fmt(r["sd0"])
def _as_str_bin(r):
    if r["type"]!="binary": return "—","—","—","—"
    return int(r["e1"]), int(r["ne1"]), int(r["e0"]), int(r["ne0"])

tab_rows = []
for _, r in ES_plot.iterrows():
    n1, m1v, sd1v, n0, m0v, sd0v = _as_str_cont(r)
    e1v, ne1v, e0v, ne0v = _as_str_bin(r)
    ci_txt = f"{_fmt(r['yi'],3)} [{_fmt(r['lo'],3)}, {_fmt(r['hi'],3)}]"
    tab_rows.append([
        str(r["study"]),
        n1, m1v, sd1v,
        n0, m0v, sd0v,
        e1v, ne1v, e0v, ne0v,
        ci_txt, f"{r['w_pct']:.2f}%"
    ])

table_header = [
    "Study", "Tr N","Tr Mean","Tr SD", "Ctrl N","Ctrl Mean","Ctrl SD",
    "Tr E","Tr NE","Ctrl E","Ctrl NE",
    "Effect (95% CI)","Weight (%)"
]

from plotly.subplots import make_subplots
import plotly.graph_objects as go

fig = make_subplots(
    rows=1, cols=2,
    column_widths=[0.58, 0.42],
    specs=[[{"type":"table"}, {"type":"xy"}]],
    horizontal_spacing=0.02
)

# Tabella sinistra
fig.add_trace(go.Table(
    header=dict(values=table_header, align="left"),
    cells=dict(values=list(map(list, zip(*tab_rows))), align="left"),
), row=1, col=1)

# Pannello forest (asse x2/y2)
sizes = 6 + 24*(ES_plot["w_pct"]/ES_plot["w_pct"].max())
fig.add_trace(go.Scatter(
    x=ES_plot["yi"], y=ES_plot["ypos"],
    mode="markers",
    marker=dict(symbol="square", size=sizes),
    hovertemplate="yi: %{x:.3f}<br>SE: %{customdata:.3f}<extra></extra>",
    customdata=ES_plot["se"],
    showlegend=False
), row=1, col=2)

# Barre CI
for _, r in ES_plot.iterrows():
    fig.add_shape(
        type="line",
        x0=r["lo"], x1=r["hi"], y0=r["ypos"], y1=r["ypos"],
        line=dict(width=2),
        xref="x2", yref="y2"
    )

# Linea verticale dell'effetto complessivo (usa shape con xref/yref)
y_min = float(min(ES_plot["ypos"].min()-2.0, -2.5))
y_max = float(ES_plot["ypos"].max()+1.0)
fig.add_shape(
    type="line",
    x0=mu_hat, x1=mu_hat, y0=y_min, y1=y_max,
    line=dict(color="black", width=1.5),
    xref="x2", yref="y2"
)

# Diamante verde
y_d = -1.5
diamond_x = [lo, mu_hat, hi, mu_hat, lo]
diamond_y = [y_d, y_d-0.6, y_d, y_d+0.6, y_d]
fig.add_trace(go.Scatter(
    x=diamond_x, y=diamond_y, mode="lines",
    fill="toself", line=dict(color="green"),
    fillcolor="rgba(0,128,0,0.25)", showlegend=False
), row=1, col=2)
fig.add_annotation(x=mu_hat, y=y_d-0.9, text="Overall", showarrow=False, row=1, col=2)

# Assi e layout
pad = max( (ES_plot["hi"].max()-ES_plot["lo"].min())*0.05, 0.1 )
x_min = float(ES_plot["lo"].min()-pad); x_max = float(ES_plot["hi"].max()+pad)
fig.update_xaxes(title_text="Effect size (yi)", range=[x_min, x_max], row=1, col=2)
fig.update_yaxes(visible=False, range=[y_min, y_max], row=1, col=2)

fig.update_layout(
    title=f"Forest plot — {res['model']}  •  μ̂={mu_hat:.3f}  [{lo:.3f}, {hi:.3f}]",
    margin=dict(l=10, r=10, t=60, b=10),
    height=420 + 22*len(ES_plot)
)
st.plotly_chart(fig, width="stretch")

# ──────────────────────────────────────────────────────────────────────────────
# FUNNEL PLOT (stile con triangolo tratteggiato e linea μ̂)
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("### 6) Funnel plot e test di Egger")

sei = np.sqrt(ES["vi"].to_numpy())
order = np.argsort(sei)  # crescente (alto = più preciso)
x_left = mu_hat - z*sei
x_right = mu_hat + z*sei

fig2 = go.Figure()

# Bordo triangolo 95% (tratteggiato)
fig2.add_trace(go.Scatter(
    x=np.concatenate([x_left[order], x_right[order][::-1], [x_left[order][0]]]),
    y=np.concatenate([sei[order], sei[order][::-1], [sei[order][0]]]),
    mode="lines",
    line=dict(dash="dash", width=1.5),
    fill=None,
    showlegend=False,
    hoverinfo="skip"
))
# Punti studio
fig2.add_trace(go.Scatter(
    x=ES["yi"], y=sei, mode="markers",
    marker=dict(size=8),
    name="Studi",
    hovertemplate="yi: %{x:.3f}<br>SE: %{y:.3f}<extra></extra>"
))
# Linea verticale dell'effetto complessivo
fig2.add_vline(x=mu_hat, line=dict(color="black", width=1.5))

fig2.update_yaxes(autorange="reversed", title="Errore standard (SE)")
fig2.update_xaxes(title="Effect size (yi)")
fig2.update_layout(
    title="Funnel plot (triangolo 95% e linea μ̂)",
    height=520, margin=dict(l=10, r=10, t=60, b=10)
)
st.plotly_chart(fig2, width="stretch")

eg = egger_test(yi, vi)
if eg is None:
    st.info("Test di Egger non calcolabile (matrice quasi singolare).")
else:
    pretty_df(pd.DataFrame({
        "Intercetta (bias)":[eg["intercept"]],"SE":[eg["se"]],"t":[eg["t"]],
        "df":[eg["df"]],"p-value":[eg["p"]]
    }).round(4), "Egger: regressione dell'asimmetria")

with st.expander("📝 Come leggere il funnel plot", expanded=False):
    st.markdown(
        "In assenza di bias, i punti sono **simmetrici** attorno alla linea di μ̂; il triangolo tratteggiato delimita l’area attesa al **95%**. "
        "Un **Egger p-value** piccolo indica **asimmetria** (possibile bias di pubblicazione)."
    )

# ──────────────────────────────────────────────────────────────────────────────
# SUBGROUP & META-REG
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("### 7) Subgroup e Meta-regressione")

# Subgroup
if subgroup_col != "—":
    st.markdown("**Analisi per sottogruppo**")
    rows = []
    for g, gdf in ES.groupby("subgroup", dropna=False):
        if gdf.shape[0] < 2: continue
        yy, vv = gdf["yi"].to_numpy(), gdf["vi"].to_numpy()
        resg = fixed_effect(yy, vv) if model_type.startswith("Fixed") else dersimonian_laird(yy, vv)
        mu_g, se_g = resg["mu"], resg["se"]; lo_g, hi_g = ci_from(mu_g, se_g, level=conf)
        rows.append(dict(Gruppo=str(g), k=len(yy), Effetto=mu_g,
                         **{f"CI {int(conf*100)}% inf":lo_g, f"CI {int(conf*100)}% sup":hi_g},
                         Q=resg["Q"], df=resg["df"], I2=resg["I2"], tau2=resg["tau2"]))
    if rows: pretty_df(pd.DataFrame(rows).round(4), "Risultati per sottogruppo")
    else: st.info("Servono ≥2 studi per ciascun sottogruppo.")

# Meta-regressione WLS (lineare, un moderatore)
if metareg_col != "—":
    st.markdown("**Meta-regressione (WLS)**")
    MR = ES.dropna(subset=["moderator"]).copy()
    if MR.shape[0] >= 3:
        X = np.column_stack([np.ones(MR.shape[0]), MR["moderator"].to_numpy()])
        y = MR["yi"].to_numpy(); v = MR["vi"].to_numpy()
        tau2 = res["tau2"] if model_type.startswith("Random") else 0.0
        w = 1.0/(v+tau2)
        # (X'WX)^(-1) X'Wy
        XtW = X.T * w
        XtWX = XtW @ X
        XtWy = XtW @ y
        try:
            beta = np.linalg.solve(XtWX, XtWy)
            covb = np.linalg.inv(XtWX)
            se = np.sqrt(np.diag(covb))
            tvals = beta/se
            df = max(MR.shape[0]-X.shape[1], 1)
            if tdist is None:
                pvals = 2*(1 - 0.5*(1+np.vectorize(lambda z: math.erf(abs(z)/np.sqrt(2)))(tvals)))
            else:
                pvals = 2*(1 - tdist.cdf(np.abs(tvals), df=df))
            tab = pd.DataFrame({"Termine":["Intercept", f"Moderatore ({metareg_col})"],
                                "β":beta, "SE":se, "t":tvals, "p-value":pvals})
            pretty_df(tab.round(4), "Coefficiente meta-regressione")
            with st.expander("📝 Come leggere la meta-regressione", expanded=False):
                st.markdown(
                    "Il coefficiente del **moderatore** indica la variazione attesa dell’effect size per **unità** di moderatore. "
                    "Un p-value piccolo → evidenza di associazione; verificare sempre linearità e possibili influenze di singoli studi."
                )
        except np.linalg.LinAlgError:
            st.info("Meta-regressione non stimabile (matrice quasi singolare).")
    else:
        st.info("Servono almeno 3 studi con moderatore non mancante.")

# Leave-one-out
with st.expander("Analisi **leave-one-out** (influenza dei singoli studi)", expanded=False):
    loo = []
    for i in range(len(ES)):
        mask = np.ones(len(ES), dtype=bool); mask[i]=False
        if mask.sum()<2: continue
        y_, v_ = yi[mask], vi[mask]
        rs = fixed_effect(y_, v_) if model_type.startswith("Fixed") else dersimonian_laird(y_, v_)
        loo.append(dict(studio_rimosso=ES.iloc[i]["study"], mu=rs["mu"], tau2=rs["tau2"], I2=rs["I2"]))
    if loo: pretty_df(pd.DataFrame(loo).round(4), "Stima combinata rimuovendo ogni studio a turno")
    else: st.info("Non calcolabile (troppo pochi studi).")

# ──────────────────────────────────────────────────────────────────────────────
# ESPONENZIAZIONE (solo per log-RR/log-OR)
# ──────────────────────────────────────────────────────────────────────────────
if show_exp and ES["type"].eq("binary").any():
    st.markdown("### 8) Interpretazione su scala esponenziale (per effetti log)")
    exp_tab = pd.DataFrame({
        "Misura":["Exp(μ̂)","Exp(CI inf)","Exp(CI sup)"],
        "Valore":[np.exp(mu_hat), np.exp(lo), np.exp(hi)]
    })
    pretty_df(exp_tab.round(4), "Scala naturale (RR/OR)")

# ──────────────────────────────────────────────────────────────────────────────
# NAVIGAZIONE
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("---")
n1, n2, n3 = st.columns(3)
with n1:
    if st.button("⬅️ Torna: SEM", key=k("go_prev")):
        safe_switch_by_tokens(["16_🧩_SEM_Structural_Equation_Modeling.py"], ["sem","equation"])
with n2:
    if st.button("↔️ Vai: Serie Temporali", key=k("go_ts")):
        safe_switch_by_tokens(
            ["14_⏱️_Analisi_Serie_Temporali.py", "15_⏱️_Analisi_Serie_Temporali.py"],
            ["serie","temporali"]
        )
with n3:
    if st.button("➡️ Vai: Report/Export", key=k("go_report")):
        safe_switch_by_tokens(
            ["18_🧾_Report_Automatico.py", "17_📤_Export_Risultati.py"],
            ["report","export"]
        )
