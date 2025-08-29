
# -*- coding: utf-8 -*-
# pages/16_🧩_SEM_Structural_Equation_Modeling.py
from __future__ import annotations

import os
import re
import numpy as np
import pandas as pd
import streamlit as st

# Plotly (per tabelle/grafici semplici)
try:
    import plotly.graph_objects as go
except Exception:
    go = None

# Librerie SEM opzionali
try:
    from semopy import Model, calc_stats
    try:
        from semopy import semplot  # per il diagramma
        _has_semplot = True
    except Exception:
        _has_semplot = False
    _has_semopy = True
except Exception:
    _has_semopy = False
    _has_semplot = False

try:
    from scipy import stats as sps
    _has_scipy = True
except Exception:
    _has_scipy = False

# ──────────────────────────────────────────────────────────────────────────────
# NAVIGATION SAFE HELPERS (evitano PageNotFound su file rinominati)
# ──────────────────────────────────────────────────────────────────────────────
def _list_pages():
    try:
        return sorted([f for f in os.listdir("pages") if f.endswith(".py")])
    except FileNotFoundError:
        return []

def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", s.lower())

def safe_switch_by_tokens(primary_candidates: list[str], fallback_tokens: list[str]):
    """
    Prova prima i candidati precisi, poi cerca fuzzy per token (es. 'upload','dataset').
    """
    files = _list_pages()
    # match esatto
    for cand in primary_candidates or []:
        if cand in files:
            st.switch_page(os.path.join("pages", cand))
            return
    # match fuzzy
    toks = [_norm(t) for t in fallback_tokens]
    for f in files:
        nf = _norm(f)
        if all(t in nf for t in toks):
            st.switch_page(os.path.join("pages", f))
            return
    st.error("Pagina richiesta non trovata nei file di /pages. Verificare i nomi reali.")

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

# ──────────────────────────────────────────────────────────────────────────────
# Config (non richiamiamo sidebar esterna per evitare link rigidi errati)
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="🧩 SEM — Structural Equation Modeling", layout="wide")

KEY = "sem"
def k(x: str) -> str:
    return f"{KEY}_{x}"

# ──────────────────────────────────────────────────────────────────────────────
# Utility
# ──────────────────────────────────────────────────────────────────────────────
def fmt_p(p: float | None) -> str:
    if p is None or p != p:
        return "—"
    if p < 1e-4:
        return "< 1e-4"
    return f"{p:.4f}"

def cronbach_alpha(df: pd.DataFrame) -> float | None:
    k = df.shape[1]
    if k < 2: return None
    variances = df.var(ddof=1)
    total_var = df.sum(axis=1).var(ddof=1)
    if total_var <= 0: return None
    alpha = (k / (k - 1.0)) * (1.0 - variances.sum() / total_var)
    return float(alpha)

def compute_cr_ave(std_loadings: pd.Series) -> tuple[float | None, float | None]:
    lam2 = (std_loadings**2).dropna()
    if lam2.empty: return (None, None)
    theta = 1.0 - lam2
    sum_lam = float(std_loadings.dropna().sum())
    sum_lam2 = float(lam2.sum())
    sum_theta = float(theta.sum())
    cr = (sum_lam**2) / ((sum_lam**2) + sum_theta) if (sum_lam**2 + sum_theta) > 0 else None
    ave = (sum_lam2) / (sum_lam2 + sum_theta) if (sum_lam2 + sum_theta) > 0 else None
    return (cr, ave)

def lavaan_syntax_from_builder(latents: list[dict], regressions: list[dict], covs: list[tuple[str, str]], id_mode: str) -> str:
    lines = []
    for comp in latents:
        name = comp["name"]
        inds = comp["indicators"]
        if not inds: continue
        lines.append(f"{name} =~ " + " + ".join(inds))
        if id_mode == "Varianza latente = 1":
            lines.append(f"{name} ~~ 1*{name}")
    dep_to_preds = {}
    for r in regressions:
        y = r["y"]; xs = r["X"]
        if not y or not xs: continue
        dep_to_preds.setdefault(y, []).extend(xs)
    for y, xs in dep_to_preds.items():
        seen = set(); xs_unique = [x for x in xs if not (x in seen or seen.add(x))]
        lines.append(f"{y} ~ " + " + ".join(xs_unique))
    for a, b in covs:
        if a and b and a != b:
            lines.append(f"{a} ~~ {b}")
    return "\n".join(lines)

def pretty_table(df: pd.DataFrame, title: str):
    st.markdown(f"**{title}**")
    st.dataframe(df, width="stretch")

# ──────────────────────────────────────────────────────────────────────────────
# Header
# ──────────────────────────────────────────────────────────────────────────────
st.title("🧩 Structural Equation Modeling (SEM)")
st.caption("Builder guidato per CFA/SEM, stima ML con indici di fit (χ²/df, CFI, TLI, RMSEA, SRMR, AIC, BIC), affidabilità (α, CR, AVE), soluzione standardizzata e diagramma del modello.")

ensure_initialized()
DF = get_active(required=True)
with st.expander("Stato dati", expanded=False):
    stamp_meta()
if DF is None or DF.empty:
    st.stop()

all_cols = list(DF.columns)
num_cols = [c for c in all_cols if pd.api.types.is_numeric_dtype(DF[c])]

# ──────────────────────────────────────────────────────────────────────────────
# Guida all'impostazione (in alto)
# ──────────────────────────────────────────────────────────────────────────────
with st.expander("📌 Come impostare correttamente la SEM (leggere prima)", expanded=True):
    st.markdown(
        "- **CFA**: definisce il **modello di misura** (indicatori → costrutti).  \n"
        "- **SEM**: aggiunge il **modello strutturale** (relazioni tra variabili latenti/osservate).  \n"
        "- **Identificazione**: *Marker* (loading del primo=1) oppure *Varianza=1*.  \n"
        "- **Dati**: indicatori **numerici**, gestione missing a monte (qui complete-case).  \n"
        "- **Fit**: CFI/TLI ≥ 0.90–0.95; RMSEA ≤ 0.06–0.08; SRMR ≤ 0.08 (linee guida)."
    )

# ──────────────────────────────────────────────────────────────────────────────
# 1) Impostazioni generali
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("### 1) Impostazioni generali")
c1, c2, c3 = st.columns([1.2, 1.2, 1.2])
with c1:
    analysis_type = st.radio("Tipo analisi", ["CFA (solo misura)", "SEM (misura + struttura)"], horizontal=False, key=k("atype"))
with c2:
    id_mode = st.selectbox("Identificazione dei costrutti", ["Marker (loading del primo = 1)", "Varianza latente = 1"], index=0, key=k("idmode"))
with c3:
    standardize = st.checkbox("Standardizza variabili (z-score) prima della stima", value=False, key=k("z"))

data_source = DF.copy()
if standardize:
    for c in num_cols:
        s = pd.to_numeric(data_source[c], errors="coerce")
        mu, sd = float(s.mean()), float(s.std(ddof=1))
        if sd and sd > 0:
            data_source[c] = (s - mu) / sd

# ──────────────────────────────────────────────────────────────────────────────
# 2) Modello di misura (CFA)
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("### 2) Modello di **misura** (CFA)")
st.session_state.setdefault(k("latents"), [])

with st.container(border=True):
    cL1, cL2, cL3 = st.columns([1.0, 2.0, 0.6])
    with cL1:
        lat_name = st.text_input("Nome costrutto latente", value="", placeholder="es. Soddisfazione", key=k("lat_name"))
    with cL2:
        lat_inds = st.multiselect("Indicatori osservati", options=num_cols, key=k("lat_inds"))
    with cL3:
        add_lat = st.button("➕ Aggiungi costrutto", key=k("add_lat"))
    if add_lat and lat_name and lat_inds:
        st.session_state[k("latents")].append({"name": lat_name.strip(), "indicators": lat_inds})
        st.success(f"Aggiunto costrutto **{lat_name}**: {', '.join(lat_inds)}")

if st.session_state[k("latents")]:
    st.markdown("**Costrutti definiti**")
    for i, comp in enumerate(st.session_state[k("latents")], start=1):
        st.write(f"{i}. **{comp['name']}** ← {', '.join(comp['indicators'])}")
    if st.button("♻️ Svuota costrutti", key=k("clr_lat")):
        st.session_state[k("latents")] = []
        st.experimental_rerun()

# ──────────────────────────────────────────────────────────────────────────────
# 3) Modello strutturale (opzionale)
# ──────────────────────────────────────────────────────────────────────────────
regressions: list[dict] = st.session_state.setdefault(k("regs"), [])
covs: list[tuple[str, str]] = st.session_state.setdefault(k("covs"), [])

if analysis_type.startswith("SEM"):
    st.markdown("### 3) Modello **strutturale**")
    latent_names = [c["name"] for c in st.session_state[k("latents")]]
    used_inds = set(sum([c["indicators"] for c in st.session_state[k("latents")]], []))
    observed_candidates = [c for c in num_cols if c not in used_inds]
    pool_vars = latent_names + observed_candidates

    with st.container(border=True):
        r1, r2, r3 = st.columns([1.0, 2.0, 0.6])
        with r1:
            dep = st.selectbox("Variabile dipendente (endogena)", options=(pool_vars or ["—"]), key=k("dep"))
        with r2:
            preds = st.multiselect("Predittori (selezionare uno o più)", options=[v for v in pool_vars if v != dep], key=k("preds"))
        with r3:
            add_reg = st.button("➕ Aggiungi relazione", key=k("add_reg"))
        if add_reg and dep and preds:
            regressions.append({"y": dep, "X": preds})
            st.success(f"Aggiunta relazione **{dep} ~ {' + '.join(preds)}**")

    if regressions:
        st.markdown("**Relazioni strutturali:**")
        for i, r in enumerate(regressions, start=1):
            st.write(f"{i}. {r['y']} ~ {', '.join(r['X'])}")
        if st.button("♻️ Svuota relazioni", key=k("clr_reg")):
            st.session_state[k("regs")] = []
            st.experimental_rerun()

    with st.expander("Covarianze (opzionali)"):
        c1, c2, c3 = st.columns([1.2, 1.2, 0.8])
        with c1:
            a = st.selectbox("Variabile A", options=(pool_vars or ["—"]), key=k("cov_a"))
        with c2:
            b = st.selectbox("Variabile B", options=[v for v in pool_vars if v != a], key=k("cov_b"))
        with c3:
            add_cov = st.button("➕ Aggiungi covarianza", key=k("add_cov"))
        if add_cov and a and b:
            covs.append((a, b))
            st.success(f"Aggiunta covarianza **{a} ~~ {b}**")
        if covs:
            st.caption("Covarianze definite: " + "; ".join([f"{x} ~~ {y}" for x, y in covs]))
            if st.button("♻️ Svuota covarianze", key=k("clr_cov")):
                st.session_state[k("covs")] = []
                st.experimental_rerun()

# ──────────────────────────────────────────────────────────────────────────────
# 4) Sintassi del modello e stima
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("### 4) Sintassi del modello e stima")

if not st.session_state[k("latents")]:
    st.info("Definire almeno **un costrutto** con i suoi indicatori per procedere.")
    st.stop()

syntax = lavaan_syntax_from_builder(
    latents=st.session_state[k("latents")],
    regressions=(st.session_state[k("regs")] if analysis_type.startswith("SEM") else []),
    covs=st.session_state[k("covs")],
    id_mode=id_mode
)
st.code(syntax or "# (sintassi vuota)")

# Colonne realmente usate
cols_used = set()
for comp in st.session_state[k("latents")]:
    cols_used.update(comp["indicators"])
if analysis_type.startswith("SEM"):
    for r in st.session_state[k("regs")]:
        for v in r["X"] + [r["y"]]:
            if v in data_source.columns:
                cols_used.add(v)
if st.session_state[k("covs")]:
    for a, b in st.session_state[k("covs")]:
        if a in data_source.columns: cols_used.add(a)
        if b in data_source.columns: cols_used.add(b)

work = data_source[list(cols_used)].copy()
work = work.apply(pd.to_numeric, errors="coerce")
work = work.dropna(axis=0, how="any")
st.markdown("**Anteprima dati usati (complete-case)**")
st.dataframe(work.head(10), width="stretch")

# ──────────────────────────────────────────────────────────────────────────────
# 5) Affidabilità di base (α, CR, AVE)
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("### 5) Affidabilità di base (α, CR, AVE)")
reliab_rows = []
for comp in st.session_state[k("latents")]:
    inds = [c for c in comp["indicators"] if c in work.columns]
    if len(inds) >= 2:
        alpha = cronbach_alpha(work[inds])
    else:
        alpha = None
    reliab_rows.append({"Costrutto": comp["name"], "k": len(inds), "Cronbach α": (None if alpha is None else round(alpha, 3))})
reliab_tab = pd.DataFrame(reliab_rows)
pretty_table(reliab_tab, "Affidabilità (Cronbach α)")

# ──────────────────────────────────────────────────────────────────────────────
# 6) Stima SEM con semopy
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("### 6) Stima del modello e risultati")

if not _has_semopy:
    st.error("`semopy` non è installato. Per abilitare la stima SEM:\n\n`pip install semopy graphviz`\n\n"
             "Nel frattempo può usare le misure di affidabilità sopra come verifica preliminare.")
    st.stop()

if work.empty:
    st.error("Nessuna riga completa disponibile dopo la rimozione dei missing. Controllare i dati/indicatori selezionati.")
    st.stop()

try:
    model = Model(syntax)
    model.fit(work)

    try:
        est = model.inspect(std_est=True)
        std_available = True
    except Exception:
        est = model.inspect()
        std_available = False
    est_df = est.copy() if isinstance(est, pd.DataFrame) else pd.DataFrame(est)

    try:
        stats = calc_stats(model, work)
        if isinstance(stats, pd.DataFrame):
            fit = stats.copy()
        else:
            keys = ["chi2", "df", "p-value", "CFI", "TLI", "RMSEA", "SRMR", "AIC", "BIC"]
            values = [getattr(stats, k, np.nan) for k in keys]
            fit = pd.DataFrame({"metric": keys, "value": values})
    except Exception:
        fit = pd.DataFrame({"metric": [], "value": []})

    if "op" in est_df.columns:
        loadings = est_df[est_df["op"] == "=~"].copy()
        regress = est_df[est_df["op"] == "~"].copy()
        covars = est_df[est_df["op"] == "~~"].copy()
        val_col = "Est" if "Est" in est_df.columns else ("Estimate" if "Estimate" in est_df.columns else None)
        se_col = "SE" if "SE" in est_df.columns else None
        p_col = "p-value" if "p-value" in est_df.columns else ("pval" if "pval" in est_df.columns else None)

        if not loadings.empty:
            cols = ["lval", "rval"] + ([val_col] if val_col else []) + ([se_col] if se_col else []) + ([p_col] if p_col else [])
            pretty_table(loadings[cols].rename(columns={"lval":"Latente","rval":"Indicatore", val_col:"Stima", se_col:"SE", p_col:"p"}).round(4), "Loadings (modello di misura)")

        if not regress.empty:
            cols = ["lval", "rval"] + ([val_col] if val_col else []) + ([se_col] if se_col else []) + ([p_col] if p_col else [])
            pretty_table(regress[cols].rename(columns={"lval":"Dipendente","rval":"Predittore", val_col:"β", se_col:"SE", p_col:"p"}).round(4), "Relazioni strutturali (β)")

        cov_show = covars.copy()
        cov_show = cov_show[cov_show["lval"] != cov_show["rval"]]
        if not cov_show.empty and val_col:
            pretty_table(cov_show[["lval","rval", val_col]].rename(columns={"lval":"Var A","rval":"Var B", val_col:"Cov"}).round(4), "Covarianze stimate")

        if std_available and val_col:
            rel_rows = []
            for comp in st.session_state[k("latents")]:
                L = loadings[(loadings["lval"] == comp["name"])]
                if L.empty:
                    rel_rows.append({"Costrutto": comp["name"], "CR": None, "AVE": None})
                    continue
                lam = L.set_index("rval")[val_col]
                cr, ave = compute_cr_ave(lam)
                rel_rows.append({"Costrutto": comp["name"], "CR": (None if cr is None else round(cr, 3)),
                                 "AVE": (None if ave is None else round(ave, 3))})
            rel_tab = pd.DataFrame(rel_rows)
            pretty_table(rel_tab, "Affidabilità composita (CR) e AVE (da soluzione standardizzata)")
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
    st.caption("Regole pratiche: CFI/TLI ≥ 0.90–0.95; RMSEA ≤ 0.06–0.08; SRMR ≤ 0.08 (da leggere con cautela).")

    with st.expander("Diagramma del modello"):
        if _has_semplot:
            try:
                out_path = os.path.join("/tmp", "sem_diagram.png")
                semplot(model, out_path)
                st.image(out_path, caption="Schema SEM generato", width="stretch")
            except Exception as e:
                st.info(f"Impossibile generare il diagramma: {e}")
        else:
            st.info("Per il diagramma serve `semopy` con `semplot` e `graphviz`.")

except Exception as e:
    st.error(f"Errore nella stima del modello: {e}")
    st.stop()

# ──────────────────────────────────────────────────────────────────────────────
# Interpretazione
# ──────────────────────────────────────────────────────────────────────────────
with st.expander("📝 Come leggere i risultati", expanded=True):
    st.markdown(
        "- **Loadings (λ)**: valori ≥0.5–0.7 indicano indicatori forti; p piccoli ⇒ loading ≠ 0.  \n"
        "- **CR/AVE**: **CR ≥ 0.70** (affidabilità interna); **AVE ≥ 0.50** (validità convergente).  \n"
        "- **Regressioni (β)**: effetto diretto sul costrutto/variabile endogena (preferire la soluzione **standardizzata** per interpretazione).  \n"
        "- **Fit globale**: CFI/TLI misurano miglioramento vs modello nullo; **RMSEA** penalizza complessità; **SRMR** è discrepanza media.  \n"
        "- **Respecification**: se fit insufficiente, riconsiderare indicatori, correlazioni errori (solo con giustificazione teorica) o la struttura causale."
    )

# ──────────────────────────────────────────────────────────────────────────────
# Navigazione LOCALE robusta (evita riferimenti rigidi a 1_📂_Upload_Dataset.py)
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
