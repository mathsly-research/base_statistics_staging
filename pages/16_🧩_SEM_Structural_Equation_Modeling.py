# -*- coding: utf-8 -*-
# pages/16_🧩_SEM_Structural_Equation_Modeling.py
from __future__ import annotations

import os
import re
import numpy as np
import pandas as pd
import streamlit as st

# Plotly (per tabelle/grafici e fallback del diagramma)
try:
    import plotly.graph_objects as go
except Exception:
    go = None

# Librerie SEM opzionali
try:
    from semopy import Model, calc_stats
    try:
        from semopy import semplot  # diagramma (richiede binario graphviz/dot)
        _has_semplot = True
    except Exception:
        _has_semplot = False
    _has_semopy = True
except Exception:
    _has_semopy = False
    _has_semplot = False

# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="🧩 SEM — Structural Equation Modeling", layout="wide")

KEY = "sem"
def k(x: str) -> str:
    return f"{KEY}_{x}"

# ──────────────────────────────────────────────────────────────────────────────
# Data store (coerente con altri moduli)
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

ensure_initialized()
DF = get_active(required=True)

# ──────────────────────────────────────────────────────────────────────────────
# Helper UI e navigazione sicura (evita PageNotFound su file rinominati)
# ──────────────────────────────────────────────────────────────────────────────
def _list_pages():
    try:
        return sorted([f for f in os.listdir("pages") if f.endswith(".py")])
    except FileNotFoundError:
        return []

def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", s.lower())

def safe_switch_by_tokens(primary_candidates: list[str], fallback_tokens: list[str]):
    files = _list_pages()
    # match esatto
    for cand in primary_candidates or []:
        if cand in files:
            st.switch_page(os.path.join("pages", cand)); return
    # match fuzzy
    toks = [_norm(t) for t in fallback_tokens]
    for f in files:
        nf = _norm(f)
        if all(t in nf for t in toks):
            st.switch_page(os.path.join("pages", f)); return
    st.error("Pagina richiesta non trovata nei file di /pages. Verificare i nomi reali.")

# ──────────────────────────────────────────────────────────────────────────────
# Utility SEM: validazioni, sanitizzazione nomi, affidabilità
# ──────────────────────────────────────────────────────────────────────────────
def fmt_p(p: float | None) -> str:
    if p is None or p != p: return "—"
    if p < 1e-4: return "< 1e-4"
    return f"{p:.4f}"

def cronbach_alpha(df: pd.DataFrame) -> float | None:
    kcols = df.shape[1]
    if kcols < 2: return None
    variances = df.var(ddof=1)
    total_var = df.sum(axis=1).var(ddof=1)
    if total_var <= 0: return None
    return float((kcols/(kcols-1.0)) * (1.0 - variances.sum()/total_var))

SAFE_RE = re.compile(r"[^A-Za-z0-9_]")
def safe_token(name: str) -> str:
    """Converte un nome (latente/variabile) in token sicuro per la sintassi."""
    if name is None: name = ""
    tok = SAFE_RE.sub("_", str(name)).strip("_")
    if tok == "": tok = "v"
    return tok

def unique_tokens(tokens: list[str]) -> list[str]:
    """Rende univoci i token aggiungendo suffissi _1, _2 in caso di collisioni."""
    seen = {}
    out = []
    for t in tokens:
        base = t
        i = seen.get(base, 0)
        if i > 0:
            t = f"{base}_{i}"
        out.append(t)
        seen[base] = i + 1
    return out

def build_rename_map(columns: list[str]) -> dict[str, str]:
    """Crea una mappa {originale -> token_sicuro_univoco} per le colonne osservate."""
    base = [safe_token(c) for c in columns]
    uniq = unique_tokens(base)
    return {orig: tok for orig, tok in zip(columns, uniq)}

def lavaan_syntax_from_builder(
    latents: list[dict], regressions: list[dict], covs: list[tuple[str, str]],
    id_mode: str, rename_map: dict[str, str]
) -> tuple[str, dict]:
    """Genera sintassi e mappa latenti {lat_originale->lat_token}."""
    lat_orig = [comp["name"] for comp in latents]
    lat_tok = unique_tokens([safe_token(n) for n in lat_orig])
    lat_map = dict(zip(lat_orig, lat_tok))

    lines = []
    # Misura
    for comp in latents:
        name_o = comp["name"]
        name_t = lat_map[name_o]
        inds_o = comp["indicators"]
        if not inds_o:
            continue
        inds_t = [rename_map.get(i, safe_token(i)) for i in inds_o]
        lines.append(f"{name_t} =~ " + " + ".join(inds_t))
        if id_mode == "Varianza latente = 1":
            lines.append(f"{name_t} ~~ 1*{name_t}")

    # Strutturale
    dep_to_preds: dict[str, list[str]] = {}
    for r in regressions or []:
        y_o = r.get("y"); xs_o = r.get("X") or []
        if not y_o or not xs_o:
            continue
        y_t = lat_map.get(y_o, rename_map.get(y_o, safe_token(y_o)))
        xs_t = [lat_map.get(x, rename_map.get(x, safe_token(x))) for x in xs_o]
        dep_to_preds.setdefault(y_t, []).extend(xs_t)
    for y_t, xs_t in dep_to_preds.items():
        seen = set(); xs_u = [x for x in xs_t if not (x in seen or seen.add(x))]
        lines.append(f"{y_t} ~ " + " + ".join(xs_u))

    # Covarianze
    for a_o, b_o in covs or []:
        a_t = lat_map.get(a_o, rename_map.get(a_o, safe_token(a_o)))
        b_t = lat_map.get(b_o, rename_map.get(b_o, safe_token(b_o)))
        if a_t and b_t and a_t != b_t:
            lines.append(f"{a_t} ~~ {b_t}")

    return "\n".join(lines), lat_map

def pretty_table(df: pd.DataFrame, title: str):
    st.markdown(f"**{title}**")
    st.dataframe(df, width="stretch")

# ──────────────────────────────────────────────────────────────────────────────
# Fallback Plotly per il diagramma SEM (senza Graphviz)
# ──────────────────────────────────────────────────────────────────────────────
def draw_sem_plotly(latents_config: list[dict], regressions: list[dict] | None = None):
    """Disegna un diagramma SEM basilare con Plotly (niente binario 'dot' richiesto)."""
    if go is None or not latents_config:
        return None

    lat_names = [c["name"] for c in latents_config]
    n_lat = len(lat_names)
    if n_lat == 0:
        return None

    # Coordinate normalizzate (0..1)
    xs, ys = {}, {}
    # Latenti su riga alta
    for i, L in enumerate(lat_names):
        xs[L] = (i + 1) / (n_lat + 1)
        ys[L] = 0.8

    # Indicatori su riga bassa, sotto ogni latente
    obs_nodes = []
    for comp in latents_config:
        L = comp["name"]
        inds = comp.get("indicators", [])
        k = len(inds)
        if k == 0:
            continue
        base_x = xs[L]
        if k == 1:
            positions = [base_x]
        else:
            span = 0.20  # larghezza del gruppo indicatori
            positions = np.linspace(base_x - span / 2, base_x + span / 2, k)
        for j, ind in enumerate(inds):
            xs[ind] = float(positions[j])
            ys[ind] = 0.35
            obs_nodes.append(ind)

    # Edges: loadings
    edge_x, edge_y = [], []
    for comp in latents_config:
        L = comp["name"]
        for ind in comp.get("indicators", []):
            if L in xs and ind in xs:
                edge_x += [xs[L], xs[ind], None]
                edge_y += [ys[L], ys[ind], None]

    fig = go.Figure()
    # Linee dei loadings
    fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode="lines", hoverinfo="none", name="Loadings"))

    # Nodi latenti
    fig.add_trace(go.Scatter(
        x=[xs[L] for L in lat_names], y=[ys[L] for L in lat_names],
        mode="markers+text", marker=dict(size=26, symbol="circle-open"),
        text=lat_names, textposition="top center", name="Latenti"
    ))

    # Nodi indicatori
    if obs_nodes:
        fig.add_trace(go.Scatter(
            x=[xs[o] for o in obs_nodes], y=[ys[o] for o in obs_nodes],
            mode="markers+text", marker=dict(size=16),
            text=obs_nodes, textposition="bottom center", name="Indicatori"
        ))

    # Relazioni strutturali (frecce)
    if regressions:
        for r in regressions:
            y_dep = r.get("y")
            for x_src in (r.get("X") or []):
                if x_src in xs and y_dep in xs:
                    fig.add_annotation(
                        x=xs[y_dep], y=ys[y_dep], ax=xs[x_src], ay=ys[x_src],
                        xref="x", yref="y", axref="x", ayref="y",
                        showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=1
                    )

    fig.update_xaxes(visible=False, range=[0, 1])
    fig.update_yaxes(visible=False, range=[0, 1])
    fig.update_layout(
        title="Diagramma SEM (fallback Plotly)",
        height=520, margin=dict(l=10, r=10, t=60, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
    )
    return fig

# ──────────────────────────────────────────────────────────────────────────────
# Header
# ──────────────────────────────────────────────────────────────────────────────
st.title("🧩 Structural Equation Modeling (SEM)")
st.caption("CFA/SEM con validazioni robuste all’inserimento dei costrutti, sanitizzazione automatica dei nomi, indici di fit e diagramma con fallback.")

with st.expander("Stato dati", expanded=False):
    stamp_meta()

if DF is None or DF.empty:
    st.stop()

all_cols = list(DF.columns)
num_cols = [c for c in all_cols if pd.api.types.is_numeric_dtype(DF[c])]

# ──────────────────────────────────────────────────────────────────────────────
# Guida in alto
# ──────────────────────────────────────────────────────────────────────────────
with st.expander("📌 Come impostare correttamente la SEM (leggere prima)", expanded=True):
    st.markdown(
        "- **CFA**: definisce il **modello di misura** (indicatori numerici → costrutti latenti).  \n"
        "- **SEM**: aggiunge il **modello strutturale** (relazioni fra latenti/osservate).  \n"
        "- **Identificazione**: *Marker* (loading del primo=1) oppure *Varianza=1*.  \n"
        "- **Dati**: indicatori **numerici**; qui si usa **complete-case** (rimuove righe con NA)."
    )

# ──────────────────────────────────────────────────────────────────────────────
# 1) Impostazioni generali
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("### 1) Impostazioni generali")
c1, c2, c3 = st.columns([1.2, 1.2, 1.2])
with c1:
    analysis_type = st.radio("Tipo analisi", ["CFA (solo misura)", "SEM (misura + struttura)"], key=k("atype"))
with c2:
    id_mode = st.selectbox("Identificazione dei costrutti", ["Marker (loading del primo = 1)", "Varianza latente = 1"], index=0, key=k("idmode"))
with c3:
    standardize = st.checkbox("Standardizza variabili (z-score) prima della stima", value=False, key=k("z"))

# Sorgente dati (eventuale z-score)
data_source = DF.copy()
if standardize:
    for c in num_cols:
        s = pd.to_numeric(data_source[c], errors="coerce")
        mu, sd = float(s.mean()), float(s.std(ddof=1))
        if sd and sd > 0:
            data_source[c] = (s - mu) / sd

# ──────────────────────────────────────────────────────────────────────────────
# 2) Modello di misura — Reset sicuro + Aggiunta costrutti con validazioni
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("### 2) Modello di **misura** (CFA)")
st.session_state.setdefault(k("latents"), [])

# Reset sicuro del form (prima di creare i widget)
if st.session_state.get(k("reset_lat_form"), False):
    for field in [k("lat_name"), k("lat_inds")]:
        st.session_state.pop(field, None)
    st.session_state[k("reset_lat_form")] = False

with st.container():
    cL1, cL2, cL3 = st.columns([1.0, 2.0, 0.7])
    with cL1:
        lat_name = st.text_input("Nome costrutto latente", value="", placeholder="es. Soddisfazione", key=k("lat_name"))
    with cL2:
        lat_inds = st.multiselect("Indicatori osservati (numerici)", options=num_cols, key=k("lat_inds"))
    with cL3:
        add_lat = st.button("➕ Aggiungi costrutto", key=k("add_lat"))

    if add_lat:
        name = (lat_name or "").strip()
        if not name:
            st.error("Specificare un **nome** per il costrutto.")
        elif any(name.lower() == l["name"].lower() for l in st.session_state[k("latents")]):
            st.error(f"Il costrutto **{name}** esiste già. Usi un nome univoco.")
        elif not lat_inds or len(lat_inds) < 2:
            st.error("Selezionare **almeno due indicatori** per il costrutto.")
        else:
            bad = []
            for v in lat_inds:
                s = pd.to_numeric(data_source[v], errors="coerce")
                if s.isna().all():
                    bad.append(v)
            if bad:
                st.error("I seguenti indicatori **non sono numerici** o non sono convertibili: " + ", ".join(bad))
            else:
                st.session_state[k("latents")].append({"name": name, "indicators": lat_inds})
                st.success(f"Aggiunto costrutto **{name}**: {', '.join(lat_inds)}")
                st.session_state[k("reset_lat_form")] = True
                st.rerun()

# Elenco costrutti
if st.session_state[k("latents")]:
    st.markdown("**Costrutti definiti**")
    for i, comp in enumerate(st.session_state[k("latents")], start=1):
        st.write(f"{i}. **{comp['name']}** ← {', '.join(comp['indicators'])}")
    cA, cB = st.columns([1, 1])
    with cA:
        if st.button("🗑️ Rimuovi ultimo costrutto", key=k("pop_lat")):
            if st.session_state[k("latents")]:
                st.session_state[k("latents")].pop()
                st.rerun()
    with cB:
        if st.button("♻️ Svuota tutti", key=k("clr_lat")):
            st.session_state[k("latents")] = []
            st.rerun()

# ──────────────────────────────────────────────────────────────────────────────
# 3) Modello strutturale (opzionale per SEM)
# ──────────────────────────────────────────────────────────────────────────────
regressions: list[dict] = st.session_state.setdefault(k("regs"), [])
covs: list[tuple[str, str]] = st.session_state.setdefault(k("covs"), [])

# Reset sicuro form relazione
if st.session_state.get(k("reset_reg_form"), False):
    for field in [k("dep"), k("preds")]:
        st.session_state.pop(field, None)
    st.session_state[k("reset_reg_form")] = False

# Reset sicuro form covarianza
if st.session_state.get(k("reset_cov_form"), False):
    for field in [k("cov_a"), k("cov_b")]:
        st.session_state.pop(field, None)
    st.session_state[k("reset_cov_form")] = False

if analysis_type.startswith("SEM"):
    st.markdown("### 3) Modello **strutturale**")
    latent_names = [c["name"] for c in st.session_state[k("latents")]]
    used_inds = set(sum([c["indicators"] for c in st.session_state[k("latents")]], []))
    observed_candidates = [c for c in num_cols if c not in used_inds]
    pool_vars = latent_names + observed_candidates

    with st.container():
        r1, r2, r3 = st.columns([1.0, 2.0, 0.7])
        with r1:
            dep = st.selectbox("Variabile dipendente (endogena)", options=(pool_vars or ["—"]), key=k("dep"))
        with r2:
            preds = st.multiselect("Predittori (uno o più)", options=[v for v in pool_vars if v != dep], key=k("preds"))
        with r3:
            add_reg = st.button("➕ Aggiungi relazione", key=k("add_reg"))
        if add_reg:
            if not dep or not preds:
                st.error("Selezionare una **dipendente** e almeno **un predittore**.")
            else:
                regressions.append({"y": dep, "X": preds})
                st.success(f"Aggiunta relazione **{dep} ~ {' + '.join(preds)}**")
                st.session_state[k("reset_reg_form")] = True
                st.rerun()

    if regressions:
        st.markdown("**Relazioni strutturali:**")
        for i, r in enumerate(regressions, start=1):
            st.write(f"{i}. {r['y']} ~ {', '.join(r['X'])}")
        c1, c2 = st.columns([1, 1])
        with c1:
            if st.button("🗑️ Rimuovi ultima relazione", key=k("pop_reg")):
                if st.session_state[k("regs")]:
                    st.session_state[k("regs")].pop()
                    st.rerun()
        with c2:
            if st.button("♻️ Svuota relazioni", key=k("clr_reg")):
                st.session_state[k("regs")] = []
                st.rerun()

    with st.expander("Covarianze (opzionali)"):
        c1, c2, c3 = st.columns([1.2, 1.2, 0.8])
        with c1:
            a = st.selectbox("Variabile A", options=(pool_vars or ["—"]), key=k("cov_a"))
        with c2:
            b = st.selectbox("Variabile B", options=[v for v in pool_vars if v != a], key=k("cov_b"))
        with c3:
            add_cov = st.button("➕ Aggiungi covarianza", key=k("add_cov"))
        if add_cov:
            if not a or not b:
                st.error("Selezionare entrambe le variabili per la covarianza.")
            else:
                covs.append((a, b))
                st.success(f"Aggiunta covarianza **{a} ~~ {b}**")
                st.session_state[k("reset_cov_form")] = True
                st.rerun()
        if covs:
            st.caption("Covarianze definite: " + "; ".join([f"{x} ~~ {y}" for x, y in covs]))
            if st.button("♻️ Svuota covarianze", key=k("clr_cov")):
                st.session_state[k("covs")] = []
                st.rerun()

# ──────────────────────────────────────────────────────────────────────────────
# 4) Sintassi del modello (sanitizzata) e dati
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("### 4) Sintassi del modello e dati")

if not st.session_state[k("latents")]:
    st.info("Definire almeno **un costrutto** con i suoi indicatori per procedere.")
    st.stop()

# Colonne osservate effettivamente usate
cols_used = set()
for comp in st.session_state[k("latents")]:
    cols_used.update(comp["indicators"])
if analysis_type.startswith("SEM"):
    for r in st.session_state[k("regs")]:
        for v in r["X"] + [r["y"]]:
            if v in data_source.columns: cols_used.add(v)
if st.session_state[k("covs")]:
    for a, b in st.session_state[k("covs")]:
        if a in data_source.columns: cols_used.add(a)
        if b in data_source.columns: cols_used.add(b)

cols_used = list(cols_used)

# Mappa osservate → token sicuri
rename_map = build_rename_map(cols_used)

# Sintassi lavaan-like con latenti/variabili sanitizzate
syntax, lat_map = lavaan_syntax_from_builder(
    latents=st.session_state[k("latents")],
    regressions=(st.session_state[k("regs")] if analysis_type.startswith("SEM") else []),
    covs=st.session_state[k("covs")],
    id_mode=id_mode,
    rename_map=rename_map
)

st.code(syntax or "# (sintassi vuota)")

# Dati per la stima (complete-case), poi rinominati coi token
work = data_source[cols_used].copy()
for c in work.columns:
    work[c] = pd.to_numeric(work[c], errors="coerce")
work = work.dropna(axis=0, how="any")
work_ren = work.rename(columns=rename_map)

st.markdown("**Anteprima dati usati (complete-case)**")
st.dataframe(work.head(10), width="stretch")

# ──────────────────────────────────────────────────────────────────────────────
# 5) Affidabilità (Cronbach α)
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("### 5) Affidabilità (Cronbach α)")
rel_rows = []
for comp in st.session_state[k("latents")]:
    inds = [c for c in comp["indicators"] if c in work.columns]
    alpha = cronbach_alpha(work[inds]) if len(inds) >= 2 else None
    rel_rows.append({"Costrutto": comp["name"], "k": len(inds), "Cronbach α": (None if alpha is None else round(alpha, 3))})
pretty_table(pd.DataFrame(rel_rows), "Affidabilità di base")

# ──────────────────────────────────────────────────────────────────────────────
# 6) Stima SEM con semopy (con fallback del diagramma)
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("### 6) Stima del modello e risultati")

if not _has_semopy:
    st.error("`semopy` non è installato. Per la stima SEM: `pip install semopy graphviz`.\n"
             "Il modulo continua a fornire α e controlli preliminari.")
    st.stop()

if work_ren.empty:
    st.error("Nessuna riga completa disponibile dopo la rimozione dei missing. Controllare i dati/indicatori selezionati.")
    st.stop()

try:
    model = Model(syntax)
    model.fit(work_ren)

    # Ispezione (standardizzata se possibile)
    try:
        est_df = model.inspect(std_est=True).copy()
        std_available = True
    except Exception:
        est_df = model.inspect().copy()
        std_available = False
    if not isinstance(est_df, pd.DataFrame):
        est_df = pd.DataFrame(est_df)

    # Indici di fit
    try:
        stats = calc_stats(model, work_ren)
        if isinstance(stats, pd.DataFrame):
            fit = stats.copy()
        else:
            keys = ["chi2", "df", "p-value", "CFI", "TLI", "RMSEA", "SRMR", "AIC", "BIC"]
            values = [getattr(stats, k, np.nan) for k in keys]
            fit = pd.DataFrame({"metric": keys, "value": values})
    except Exception:
        fit = pd.DataFrame({"metric": [], "value": []})

    # Mappa inversa per visualizzare nomi originali
    inv_map = {v: k for k, v in rename_map.items()}
    inv_lat = {v: k for k, v in lat_map.items()}
    def backname(x: str) -> str:
        return inv_lat.get(x, inv_map.get(x, x))

    if "op" in est_df.columns:
        loadings = est_df[est_df["op"] == "=~"].copy()
        regress  = est_df[est_df["op"] == "~"].copy()
        covars   = est_df[est_df["op"] == "~~"].copy()

        val_col = "Est" if "Est" in est_df.columns else ("Estimate" if "Estimate" in est_df.columns else None)
        se_col  = "SE" if "SE" in est_df.columns else None
        p_col   = "p-value" if "p-value" in est_df.columns else ("pval" if "pval" in est_df.columns else None)

        if not loadings.empty:
            L = loadings.rename(columns={"lval":"Latente","rval":"Indicatore"})
            L["Latente"] = L["Latente"].map(backname)
            L["Indicatore"] = L["Indicatore"].map(backname)
            cols = ["Latente","Indicatore"] + ([val_col] if val_col else []) + ([se_col] if se_col else []) + ([p_col] if p_col else [])
            pretty_table(L[cols].round(4), "Loadings (modello di misura)")

        if not regress.empty:
            R = regress.rename(columns={"lval":"Dipendente","rval":"Predittore"})
            R["Dipendente"] = R["Dipendente"].map(backname)
            R["Predittore"] = R["Predittore"].map(backname)
            cols = ["Dipendente","Predittore"] + ([val_col] if val_col else []) + ([se_col] if se_col else []) + ([p_col] if p_col else [])
            pretty_table(R[cols].round(4), "Relazioni strutturali (β)")

        C = covars[(covars["lval"] != covars["rval"])].copy()
        if not C.empty and val_col:
            C["Var A"] = C["lval"].map(backname)
            C["Var B"] = C["rval"].map(backname)
            pretty_table(C[["Var A","Var B", val_col]].rename(columns={val_col:"Cov"}).round(4), "Covarianze stimate")
    else:
        pretty_table(est_df.round(4), "Stime dei parametri")

    st.markdown("**Indici di bontà d’adattamento**")
    if isinstance(fit, pd.DataFrame) and ("metric" in fit.columns and "value" in fit.columns):
        st.dataframe(fit, width="stretch")
        def get_fit(m):
            try:
                row = fit.loc[fit["metric"].str.upper()==m.upper(), "value"]
                return float(row.iloc[0]) if not row.empty else np.nan
            except Exception:
                return np.nan
        chi2 = get_fit("chi2"); dfv = get_fit("df"); pv = get_fit("p-value"); cfi = get_fit("CFI"); tli = get_fit("TLI"); rmsea = get_fit("RMSEA")
    else:
        chi2=dfv=pv=cfi=tli=rmsea=np.nan

    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("χ²/df", f"{(chi2/dfv):.2f}" if chi2==chi2 and dfv and dfv>0 else "—")
    with c2: st.metric("CFI", f"{cfi:.3f}" if cfi==cfi else "—")
    with c3: st.metric("TLI", f"{tli:.3f}" if tli==tli else "—")
    with c4: st.metric("RMSEA", f"{rmsea:.3f}" if rmsea==rmsea else "—")
    st.caption("Regole pratiche: CFI/TLI ≥ 0.90–0.95; RMSEA ≤ 0.06–0.08; SRMR ≤ 0.08.")

    # ── Diagramma del modello: Graphviz se disponibile, altrimenti fallback Plotly
    with st.expander("Diagramma del modello"):
        used_latents = st.session_state.get(k("latents"), [])
        tried_graphviz = False
        if _has_semplot:
            try:
                out_path = os.path.join("/tmp", "sem_diagram.png")
                semplot(model, out_path)   # richiede binario 'dot'
                st.image(out_path, caption="Schema SEM (Graphviz)", width="stretch")
                tried_graphviz = True
            except Exception as e:
                st.info(f"Impossibile generare con Graphviz: {e}")

        if (not _has_semplot) or (tried_graphviz is False):
            fig = draw_sem_plotly(latents_config=used_latents, regressions=st.session_state.get(k("regs"), []))
            if fig is not None:
                st.plotly_chart(fig, width="stretch")
                st.caption("Rendering alternativo senza Graphviz. Per lo schema classico installare il binario `graphviz` (dot).")
            else:
                st.info("Definire almeno un costrutto con indicatori per visualizzare il diagramma.")

except Exception as e:
    st.error(f"Errore nella stima del modello: {e}")
    st.stop()

# ──────────────────────────────────────────────────────────────────────────────
# Interpretazione (promemoria)
# ──────────────────────────────────────────────────────────────────────────────
with st.expander("📝 Come leggere i risultati", expanded=True):
    st.markdown(
        "- **Loadings (λ)**: ≥0.5–0.7 indicano indicatori forti; p piccoli ⇒ loading ≠ 0.  \n"
        "- **Affidabilità**: **α ≥ 0.70**; con soluzione standardizzata si possono calcolare **CR** e **AVE**.  \n"
        "- **Regressioni (β)**: effetto diretto sul costrutto/variabile endogena; la versione **standardizzata** è più leggibile.  \n"
        "- **Fit globale**: CFI/TLI alti, RMSEA/SRMR bassi; giudizio sempre contestualizzato a teoria e dati."
    )

# ──────────────────────────────────────────────────────────────────────────────
# Navigazione sicura
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("---")
nav1, nav2, nav3 = st.columns(3)
with nav1:
    if st.button("⬅️ Torna: Upload Dataset", key=k("go_upload")):
        safe_switch_by_tokens(
            primary_candidates=[
                "0_📂_Upload_Dataset.py", "0_📂_Upload_Dataset (1).py",
                "1_📂_Upload_Dataset.py", "1_📂_Upload_Dataset (1).py",
            ],
            fallback_tokens=["upload", "dataset"]
        )
with nav2:
    if st.button("↔️ Vai: Serie Temporali", key=k("go_ts")):
        safe_switch_by_tokens(
            primary_candidates=[
                "14_⏱️_Analisi_Serie_Temporali.py",
                "15_⏱️_Analisi_Serie_Temporali.py",
            ],
            fallback_tokens=["analisi", "serie", "temporali"]
        )
with nav3:
    if st.button("➡️ Vai: Report / Export", key=k("go_report")):
        safe_switch_by_tokens(
            primary_candidates=[
                "17_🧾_Report_Automatico.py", "17_📤_Export_Risultati.py",
                "15_🧾_Report_Automatico.py", "15_📤_Export_Risultati.py",
                "14_🧾_Report_Automatico.py", "14_📤_Export_Risultati.py",
            ],
            fallback_tokens=["report"]
        )
