# -*- coding: utf-8 -*-
# pages/16_🧩_SEM_Structural_Equation_Modeling.py
from __future__ import annotations

import os
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
        # semopy >= 2.x
        from semopy import semplot  # per il diagramma
        _has_semplot = True
    except Exception:
        _has_semplot = False
    _has_semopy = True
except Exception:
    _has_semopy = False
    _has_semplot = False

# Per Cronbach α, ecc.
try:
    from scipy import stats as sps
    _has_scipy = True
except Exception:
    _has_scipy = False

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
# Config & NAV
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="🧩 SEM — Structural Equation Modeling", layout="wide")
try:
    from nav import sidebar
    sidebar()
except Exception:
    pass

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
    # α = (k/(k-1)) * (1 - sum(Var_i)/Var_tot)
    k = df.shape[1]
    if k < 2: return None
    variances = df.var(ddof=1)
    total_var = df.sum(axis=1).var(ddof=1)
    if total_var <= 0: return None
    alpha = (k / (k - 1.0)) * (1.0 - variances.sum() / total_var)
    return float(alpha)

def compute_cr_ave(std_loadings: pd.Series) -> tuple[float | None, float | None]:
    # Dalla soluzione standardizzata: θ_i = 1 - λ_i^2
    lam2 = (std_loadings**2).dropna()
    if lam2.empty: return (None, None)
    theta = 1.0 - lam2
    sum_lam = float(std_loadings.dropna().sum())
    sum_lam2 = float(lam2.sum())
    sum_theta = float(theta.sum())
    # CR = (Σλ)^2 / ((Σλ)^2 + Σθ)
    cr = (sum_lam**2) / ((sum_lam**2) + sum_theta) if (sum_theta is not None and (sum_lam**2 + sum_theta) > 0) else None
    # AVE = Σλ^2 / (Σλ^2 + Σθ)
    ave = (sum_lam2) / (sum_lam2 + sum_theta) if (sum_lam2 + sum_theta) > 0 else None
    return (cr, ave)

def lavaan_syntax_from_builder(latents: list[dict], regressions: list[dict], covs: list[tuple[str, str]], id_mode: str) -> str:
    lines = []
    # Misura
    for comp in latents:
        name = comp["name"]
        inds = comp["indicators"]
        if not inds: continue
        if id_mode == "Marker (loading del primo = 1)":
            # Formato standard semopy/lavaan: il marker è implicito (primo indicatore)
            lines.append(f"{name} =~ " + " + ".join(inds))
        else:
            # Varianza = 1: imposto varianza latente a 1
            lines.append(f"{name} =~ " + " + ".join(inds))
            lines.append(f"{name} ~~ 1*{name}")
    # Strutturale: raggruppo per dipendente
    dep_to_preds = {}
    for r in regressions:
        y = r["y"]; xs = r["X"]
        if not y or not xs: continue
        dep_to_preds.setdefault(y, []).extend(xs)
    for y, xs in dep_to_preds.items():
        # rimuovo duplicati mantenendo ordine
        seen = set(); xs_unique = [x for x in xs if not (x in seen or seen.add(x))]
        lines.append(f"{y} ~ " + " + ".join(xs_unique))
    # Covarianze
    for a, b in covs:
        if a and b and a != b:
            lines.append(f"{a} ~~ {b}")
    return "\n".join(lines)

def pretty_table(df: pd.DataFrame, title: str):
    st.markdown(f"**{title}**")
    st.dataframe(df, use_container_width=True)

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
# Guida all'impostazione
# ──────────────────────────────────────────────────────────────────────────────
with st.expander("📌 Come impostare correttamente la SEM (leggere prima)", expanded=True):
    st.markdown(
        "- **CFA**: definisce il **modello di misura** (quali indicatori osservati caricano su ciascun costrutto latente).  \n"
        "- **SEM**: aggiunge il **modello strutturale** (relazioni causali tra latenti/variabili).  \n"
        "- **Identificazione**:  \n"
        "  • *Marker*: si fissa il **primo loading** di ciascun costrutto a 1 (scelta comune).  \n"
        "  • *Varianza=1*: si fissa la **varianza latente** a 1.  \n"
        "- **Dati**: usare indicatori **numerici**; standardizzare se necessario. Rimuovere/outlier e gestire missing (qui viene fatto il **complete-case**).  \n"
        "- **Fit**: CFI/TLI ≥ 0.90–0.95; RMSEA ≤ 0.06–0.08; SRMR ≤ 0.08 (regole pratiche, non dogmi)."
    )

# ──────────────────────────────────────────────────────────────────────────────
# STEP 1 — Impostazioni generali
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("### 1) Impostazioni generali")
c1, c2, c3 = st.columns([1.2, 1.2, 1.2])
with c1:
    analysis_type = st.radio("Tipo analisi", ["CFA (solo misura)", "SEM (misura + struttura)"], horizontal=False, key=k("atype"))
with c2:
    id_mode = st.selectbox("Identificazione dei costrutti", ["Marker (loading del primo = 1)", "Varianza latente = 1"], index=0, key=k("idmode"))
with c3:
    standardize = st.checkbox("Standardizza variabili (z-score) prima della stima", value=False, key=k("z"))

if standardize:
    Z = DF.copy()
    for c in num_cols:
        s = pd.to_numeric(Z[c], errors="coerce")
        mu, sd = float(s.mean()), float(s.std(ddof=1))
        if sd and sd > 0:
            Z[c] = (s - mu) / sd
    data_source = Z
else:
    data_source = DF

# ──────────────────────────────────────────────────────────────────────────────
# STEP 2 — Builder del modello di misura (latenti)
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

# Lista costrutti
if st.session_state[k("latents")]:
    st.markdown("**Costrutti definiti**")
    for i, comp in enumerate(st.session_state[k("latents")], start=1):
        st.write(f"{i}. **{comp['name']}** ← {', '.join(comp['indicators'])}")
    if st.button("♻️ Svuota costrutti", key=k("clr_lat")):
        st.session_state[k("latents")] = []
        st.experimental_rerun()

# ──────────────────────────────────────────────────────────────────────────────
# STEP 3 — Modello **strutturale** (facoltativo per SEM)
# ──────────────────────────────────────────────────────────────────────────────
regressions: list[dict] = st.session_state.setdefault(k("regs"), [])
covs: list[tuple[str, str]] = st.session_state.setdefault(k("covs"), [])

if analysis_type.startswith("SEM"):
    st.markdown("### 3) Modello **strutturale**")
    # Variabili disponibili: latenti + osservate numeriche non già usate come indicatori (opzionali)
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
# STEP 4 — Sintassi del modello e stima
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("### 4) Sintassi del modello e stima")

if not st.session_state[k("latents")]:
    st.info("Definire almeno **un costrutto** con i suoi indicatori per procedere.")
    st.stop()

# Costruisco la sintassi lavaan-like per semopy
syntax = lavaan_syntax_from_builder(
    latents=st.session_state[k("latents")],
    regressions=(st.session_state[k("regs")] if analysis_type.startswith("SEM") else []),
    covs=st.session_state[k("covs")],
    id_mode=id_mode
)
st.code(syntax or "# (sintassi vuota)")

# Dati per la stima: mantengo solo le colonne effettivamente usate
cols_used = set()
for comp in st.session_state[k("latents")]:
    cols_used.update(comp["indicators"])
if analysis_type.startswith("SEM"):
    # includo eventuali osservate nei predittori/dipendenti
    for r in st.session_state[k("regs")]:
        for v in r["X"] + [r["y"]]:
            # se è osservata (presente tra le colonne del DF) la includo
            if v in data_source.columns:
                cols_used.add(v)
if st.session_state[k("covs")]:
    for a, b in st.session_state[k("covs")]:
        if a in data_source.columns: cols_used.add(a)
        if b in data_source.columns: cols_used.add(b)

work = data_source[list(cols_used)].copy()
work = work.apply(pd.to_numeric, errors="coerce")
work = work.dropna(axis=0, how="any")  # complete-case per semplicità/robustezza
st.markdown("**Anteprima dati usati (complete-case)**")
st.dataframe(work.head(10), use_container_width=True)

# ──────────────────────────────────────────────────────────────────────────────
# Affidabilità (sempre calcolata, utile anche senza semopy)
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
# Stima SEM con semopy (se disponibile)
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("### 6) Stima del modello e risultati")

if not _has_semopy:
    st.error("`semopy` non è installato. Per abilitare la stima SEM:\n\n`pip install semopy graphviz`\n\n"
             "Nel frattempo può usare le misure di affidabilità sopra come verifica preliminare.")
    st.stop()

if work.empty:
    st.error("Nessuna riga completa disponibile dopo la rimozione dei missing. Controllare i dati/indicatori selezionati.")
    st.stop()

# Fit
try:
    model = Model(syntax)
    model.fit(work)
    # Stime (proviamo ad ottenere la soluzione standardizzata se supportata)
    try:
        est = model.inspect(std_est=True)
        std_available = True
    except Exception:
        est = model.inspect()
        std_available = False
    if isinstance(est, pd.DataFrame):
        est_df = est.copy()
    else:
        # Fallback: semopy versioni diverse
        est_df = pd.DataFrame(est)

    # Indici di fit
    try:
        stats = calc_stats(model, work)
        # alcune versioni restituiscono DataFrame, altre oggetto con attributi
        if isinstance(stats, pd.DataFrame):
            fit = stats.copy()
        else:
            # Provo ad estrarre i principali in un dict
            keys = ["chi2", "df", "p-value", "CFI", "TLI", "RMSEA", "SRMR", "AIC", "BIC"]
            values = []
            for kkey in keys:
                values.append(getattr(stats, kkey, np.nan))
            fit = pd.DataFrame({"metric": keys, "value": values})
    except Exception as e:
        fit = pd.DataFrame({"metric": [], "value": []})

    # Presentazione: separo per tipo di parametro
    if "op" in est_df.columns:
        # lavaan-like: =~, ~, ~~ (loadings, regressioni, cov/var)
        loadings = est_df[est_df["op"] == "=~"].copy()
        regress = est_df[est_df["op"] == "~"].copy()
        covars = est_df[est_df["op"] == "~~"].copy()
        # colonna stima
        val_col = "Est" if "Est" in est_df.columns else ("Estimate" if "Estimate" in est_df.columns else None)
        se_col = "SE" if "SE" in est_df.columns else None
        p_col = "p-value" if "p-value" in est_df.columns else ("pval" if "pval" in est_df.columns else None)
        # Loadings
        if not loadings.empty:
            cols = ["lval", "rval"] + ([val_col] if val_col else []) + ([se_col] if se_col else []) + ([p_col] if p_col else [])
            pretty_table(loadings[cols].rename(columns={"lval":"Latente","rval":"Indicatore", val_col:"Stima", se_col:"SE", p_col:"p"}).round(4), "Loadings (modello di misura)")
        # Regressioni
        if not regress.empty:
            cols = ["lval", "rval"] + ([val_col] if val_col else []) + ([se_col] if se_col else []) + ([p_col] if p_col else [])
            pretty_table(regress[cols].rename(columns={"lval":"Dipendente","rval":"Predittore", val_col:"β", se_col:"SE", p_col:"p"}).round(4), "Relazioni strutturali (β)")
        # Varianze/covarianze
        if not covars.empty:
            cov_show = covars.copy()
            cov_show = cov_show[cov_show["lval"] != cov_show["rval"]]  # solo covarianze (no var)
            if not cov_show.empty and val_col:
                pretty_table(cov_show[["lval","rval", val_col]].rename(columns={"lval":"Var A","rval":"Var B", val_col:"Cov"}).round(4), "Covarianze stimate")

        # Standardizzati: se disponibili, calcolo CR/AVE per ogni latente dalle λ standardizzate
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
        # Formato non atteso: mostro tutto
        pretty_table(est_df.round(4), "Stime dei parametri")

    # Fit indices
    st.markdown("**Indici di bontà d’adattamento**")
    if isinstance(fit, pd.DataFrame) and ("metric" in fit.columns and "value" in fit.columns):
        st.dataframe(fit, use_container_width=True)
        # estraggo eventuali metriche per le metric cards
        def get_fit(m):
            try:
                row = fit.loc[fit["metric"].str.upper()==m.upper(), "value"]
                return float(row.iloc[0]) if not row.empty else np.nan
            except Exception:
                return np.nan
        chi2 = get_fit("chi2"); df = get_fit("df"); pv = get_fit("p-value"); cfi = get_fit("CFI"); tli = get_fit("TLI"); rmsea = get_fit("RMSEA"); srmr = get_fit("SRMR")
    else:
        # provo alternative (DataFrame key->value)
        try:
            chi2 = float(fit.loc["chi2"].values[0]); df = float(fit.loc["df"].values[0]); pv = float(fit.loc["p-value"].values[0])
            cfi = float(fit.loc["CFI"].values[0]); tli = float(fit.loc["TLI"].values[0]); rmsea = float(fit.loc["RMSEA"].values[0]); srmr = float(fit.loc["SRMR"].values[0])
        except Exception:
            chi2=df=pv=cfi=tli=rmsea=srmr=np.nan

    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("χ²/df", f"{(chi2/df):.2f}" if chi2==chi2 and df and df>0 else "—")
    with c2: st.metric("CFI", f"{cfi:.3f}" if cfi==cfi else "—")
    with c3: st.metric("TLI", f"{tli:.3f}" if tli==tli else "—")
    with c4: st.metric("RMSEA", f"{rmsea:.3f}" if rmsea==rmsea else "—")
    st.caption("Regole pratiche: CFI/TLI ≥ 0.90–0.95; RMSEA ≤ 0.06–0.08; SRMR ≤ 0.08 (da leggere con cautela rispetto al contesto).")

    # Diagramma del modello (se possibile)
    with st.expander("Diagramma del modello"):
        if _has_semplot:
            try:
                out_path = os.path.join("/tmp", "sem_diagram.png")
                semplot(model, out_path)
                st.image(out_path, caption="Schema SEM generato", use_column_width=True)
            except Exception as e:
                st.info(f"Impossibile generare il diagramma: {e}")
        else:
            st.info("Per il diagramma è necessario `semopy` con `semplot` e `graphviz` installati.")

except Exception as e:
    st.error(f"Errore nella stima del modello: {e}")
    st.stop()

# ──────────────────────────────────────────────────────────────────────────────
# Interpretazione
# ──────────────────────────────────────────────────────────────────────────────
with st.expander("📝 Come leggere i risultati", expanded=True):
    st.markdown(
        "- **Loadings (λ)**: valori alti (≥0.5–0.7) indicano forte associazione indicatore→latente. p piccoli ⇒ loading ≠ 0.  \n"
        "- **CR/AVE**: **CR ≥ 0.70** (affidabilità interna); **AVE ≥ 0.50** (validità convergente).  \n"
        "- **Regressioni (β)**: effetto diretto di un predittore su una variabile endogena; segno e grandezza in soluzione **standardizzata** sono più leggibili.  \n"
        "- **Fit globale**: CFI/TLI misurano miglioramento vs modello nullo; **RMSEA** penalizza la complessità; **SRMR** è la discrepanza media sugli RSS.  \n"
        "- **Respecification**: se il fit è insufficiente, riconsideri indicatori, correlazioni tra errori con giustificazione teorica, o la struttura causale (evitando overfitting)."
    )

# ──────────────────────────────────────────────────────────────────────────────
# Navigazione
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("---")
nav1, nav2 = st.columns(2)
with nav1:
    if st.button("⬅️ Torna: Serie Temporali", use_container_width=True, key=k("go_prev")):
        for target in [
            "pages/14_⏱️_Analisi_Serie_Temporali.py",
            "pages/15_⏱️_Analisi_Serie_Temporali.py",
        ]:
            try:
                st.switch_page(target)
                break
            except Exception:
                continue
with nav2:
    if st.button("➡️ Vai: Report / Export", use_container_width=True, key=k("go_next")):
        for target in [
            "pages/17_🧾_Report_Automatico.py",
            "pages/17_📤_Export_Risultati.py",
            "pages/15_🧾_Report_Automatico.py",
            "pages/15_📤_Export_Risultati.py",
            "pages/14_🧾_Report_Automatico.py",
            "pages/14_📤_Export_Risultati.py",
        ]:
            try:
                st.switch_page(target)
                break
            except Exception:
                continue
