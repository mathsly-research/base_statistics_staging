# -*- coding: utf-8 -*-
# pages/11_🧭_Analisi_di_Sopravvivenza.py
from __future__ import annotations

import math
import itertools
import streamlit as st
import numpy as np
import pandas as pd

# Plot
try:
    import plotly.graph_objects as go
    import plotly.express as px
except Exception:
    go = None
    px = None

# Lifelines per KM, log-rank, Cox (opzionale: il modulo funziona in fallback)
try:
    from lifelines import KaplanMeierFitter, CoxPHFitter
    from lifelines.statistics import logrank_test, multivariate_logrank_test, proportional_hazard_test
    _has_lifelines = True
except Exception:
    _has_lifelines = False

# ──────────────────────────────────────────────────────────────────────────────
# Data store centralizzato (come negli altri moduli)
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
# Config pagina + nav
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="🧭 Analisi di Sopravvivenza", layout="wide")
try:
    from nav import sidebar
    sidebar()
except Exception:
    pass

KEY = "surv"
def k(name: str) -> str:
    return f"{KEY}_{name}"

# ──────────────────────────────────────────────────────────────────────────────
# Utility numeriche e di sintesi
# ──────────────────────────────────────────────────────────────────────────────
def fmt_p(p: float | None) -> str:
    if p is None or p != p:
        return "—"
    if p < 1e-4:
        return "< 1e-4"
    return f"{p:.4f}"

def interpret_p(p: float | None, alpha: float = 0.05) -> str:
    if p is None or p != p:
        return "non conclusivo"
    return "statisticamente significativo" if p < alpha else "non significativo"

# ──────────────────────────────────────────────────────────────────────────────
# Utility: KM fallback (se lifelines assente), log-rank 2 gruppi, risk table
# ──────────────────────────────────────────────────────────────────────────────
def _km_curve_fallback(time: np.ndarray, event: np.ndarray):
    """Restituisce (t, S(t)) step-wise senza CI (fallback semplice)."""
    df = pd.DataFrame({"t": time, "e": event}).sort_values("t")
    df = df[df["t"].notna()]
    t_unique = np.sort(df.loc[df["e"] == 1, "t"].unique())
    if t_unique.size == 0:
        tmax = float(df["t"].max() if df["t"].notna().any() else 1.0)
        return np.array([0.0, tmax]), np.array([1.0, 1.0])
    s = 1.0
    t_list = [0.0]; S_list = [1.0]
    for t in t_unique:
        at_risk = ((df["t"] >= t)).sum()
        d = ((df["t"] == t) & (df["e"] == 1)).sum()
        if at_risk > 0:
            s *= (1.0 - d / at_risk)
        t_list.append(float(t)); S_list.append(float(s))
    t_list.append(float(df["t"].max()))
    S_list.append(float(S_list[-1]))
    return np.array(t_list, dtype=float), np.array(S_list, dtype=float)

def _median_survival_from_km(t: np.ndarray, S: np.ndarray) -> float | None:
    if len(t) == 0: return None
    below = np.where(S <= 0.5)[0]
    if len(below) == 0:
        return None
    i = below[0]
    return float(t[i])

def _logrank_two_groups_fallback(t1, e1, t0, e0):
    """Log-rank per 2 gruppi (statistica ~ χ²(1))."""
    times = np.sort(np.unique(np.concatenate([t1[e1==1], t0[e0==1]])))
    Z = 0.0; V = 0.0
    for t in times:
        n1 = ((t1 >= t)).sum(); n0 = ((t0 >= t)).sum(); n = n1 + n0
        d1 = ((t1 == t) & (e1 == 1)).sum()
        d0 = ((t0 == t) & (e0 == 1)).sum()
        d = d1 + d0
        if n > 1 and n1 > 0 and n0 > 0:
            exp1 = d * (n1 / n)
            Z += (d1 - exp1)
            V += (n1 * n0 * d * (n - d)) / (n**2 * (n - 1)) if (n - 1) > 0 else 0.0
    chi2 = (Z**2) / V if V > 0 else np.nan
    try:
        from scipy.stats import chi2 as chi2dist
        p = float(chi2dist.sf(chi2, df=1))
    except Exception:
        p = math.exp(-chi2 / 2.0) if chi2 == chi2 else np.nan
    return float(chi2), float(p)

def _numbers_at_risk(df_: pd.DataFrame, time_col: str, event_col: str, group_col: str | None, cutpoints: list[float]):
    """
    Tabella 'numeri a rischio' ai cutpoints richiesti.
    ATTENZIONE: df_ deve contenere la colonna group_col se group_col è non-None.
    """
    out = []
    if group_col is None:
        groups = [("Tutti", df_)]
    else:
        # group_col deve ESISTERE in df_ → in questo modulo vale "group"
        groups = [(str(g), gdf) for g, gdf in df_.groupby(group_col, dropna=False)]
    for gname, gdf in groups:
        t = pd.to_numeric(gdf[time_col], errors="coerce")
        at_risk = [(t >= cp).sum() for cp in cutpoints]
        out.append(pd.Series(at_risk, index=cutpoints, name=gname))
    return pd.DataFrame(out).set_index(pd.Index([g for g, _ in groups]))

# ──────────────────────────────────────────────────────────────────────────────
# Header
# ──────────────────────────────────────────────────────────────────────────────
st.title("🧭 Analisi di Sopravvivenza")
st.caption("Curve di Kaplan–Meier, test di log-rank e modello di Cox PH con diagnostica. Interfaccia coerente e guidata.")

ensure_initialized()
df = get_active(required=True)
with st.expander("Stato dati", expanded=False):
    stamp_meta()
if df is None or df.empty:
    st.stop()

all_cols = list(df.columns)
num_cols = [c for c in all_cols if pd.api.types.is_numeric_dtype(df[c])]

# ──────────────────────────────────────────────────────────────────────────────
# STEP 1 — Selezione variabili
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("### 1) Seleziona le variabili")
c1, c2, c3 = st.columns([1.4, 1.2, 1.4])
with c1:
    time_col = st.selectbox("Tempo di follow-up", options=num_cols, key=k("time"))
with c2:
    event_col = st.selectbox("Evento (0/1 o categoriale)", options=all_cols, key=k("event"))
with c3:
    group_col = st.selectbox("Gruppo (opzionale)", options=["— nessuno —"] + all_cols, key=k("group"))
    group_col = None if group_col == "— nessuno —" else group_col

# Preparazione evento binario
y_raw = df[event_col]
if pd.api.types.is_numeric_dtype(y_raw) and set(pd.unique(y_raw.dropna())) <= {0, 1}:
    event_label = 1
    event = (y_raw == 1).astype(int)
    st.caption("Evento interpretato come binario {0,1} con '1' = **evento**.")
else:
    levels = sorted(y_raw.dropna().astype(str).unique().tolist())
    event_label = st.selectbox("Qual è il valore che rappresenta **evento** (vs censura)?",
                               options=levels, key=k("event_label"))
    event = (y_raw.astype(str) == event_label).astype(int)

time = pd.to_numeric(df[time_col], errors="coerce")
if (time < 0).any():
    st.warning("Sono presenti tempi negativi: verranno ignorati nelle analisi.")
    mask_nonneg = time >= 0
else:
    mask_nonneg = np.ones(len(time), dtype=bool)

work = pd.DataFrame({"time": time, "event": event})
if group_col is not None:
    work["group"] = df[group_col].astype(str)
work = work[mask_nonneg].dropna(subset=["time", "event"])

if work.empty:
    st.error("Dopo la pulizia (NA/tempi negativi) non restano osservazioni utilizzabili.")
    st.stop()

# ──────────────────────────────────────────────────────────────────────────────
# STEP 2 — Curve di Kaplan–Meier e test di log-rank
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("### 2) Curve di Kaplan–Meier e test di log-rank")
left, right = st.columns([1.6, 1.4])

alpha = st.select_slider("Soglia di significatività (α)", options=[0.01, 0.05, 0.10], value=0.05, key=k("alpha"))

with left:
    show_ci = st.checkbox("Mostra intervalli di confidenza 95%", value=True, key=k("km_ci"))
    show_censors = st.checkbox("Mostra tick di censura", value=False, key=k("km_ticks"))
    group_palette = px.colors.qualitative.D3 if px is not None else None

    if go is not None:
        fig = go.Figure()
        groups_iter = [("Tutti", work)] if group_col is None else list(work.groupby("group", dropna=False))
        for idx, (gname, gdf) in enumerate(groups_iter):
            t = gdf["time"].to_numpy(float)
            e = gdf["event"].to_numpy(int)
            color = (group_palette[idx % len(group_palette)] if group_palette else None)

            if _has_lifelines:
                km = KaplanMeierFitter()
                km.fit(t, e, label=str(gname))
                x = km.survival_function_.index.values.astype(float)
                y = km.survival_function_[str(gname)].values.astype(float)
                fig.add_trace(go.Scatter(x=x, y=y, mode="lines", line_shape="hv",
                                         line=dict(width=3, color=color),
                                         name=str(gname)))
                if show_ci:
                    ci = km.confidence_interval_
                    low = ci.iloc[:, 0].values.astype(float); high = ci.iloc[:, 1].values.astype(float)
                    fig.add_trace(go.Scatter(x=x, y=low, line=dict(width=0), showlegend=False,
                                             hoverinfo="skip", name=None))
                    fig.add_trace(go.Scatter(x=x, y=high, fill="tonexty",
                                             fillcolor="rgba(127,127,127,0.15)",
                                             line=dict(width=0), showlegend=False, hoverinfo="skip", name=None))
                if show_censors:
                    cens_t = gdf.loc[gdf["event"] == 0, "time"].values
                    fig.add_trace(go.Scatter(x=cens_t, y=km.predict(cens_t), mode="markers",
                                             marker=dict(symbol="line-ns", size=8, color=color),
                                             name=f"Censura {gname}", showlegend=False, hoverinfo="skip"))
            else:
                x, y = _km_curve_fallback(t, e)
                fig.add_trace(go.Scatter(x=x, y=y, mode="lines", line_shape="hv",
                                         line=dict(width=3, color=color),
                                         name=str(gname)))
        fig.update_layout(template="simple_white", height=460,
                          title="Curve di sopravvivenza (Kaplan–Meier)",
                          xaxis_title=f"Tempo ({time_col})", yaxis_title="S(t)")
        fig.update_yaxes(range=[0, 1], showline=True, linewidth=2, linecolor="black")
        fig.update_xaxes(showline=True, linewidth=2, linecolor="black")
        st.plotly_chart(fig, use_container_width=True)

with right:
    # Mediane e log-rank
    if _has_lifelines:
        rows = []
        if group_col is None:
            km = KaplanMeierFitter().fit(work["time"], work["event"], label="Tutti")
            med = km.median_survival_time_
            low, high = (np.nan, np.nan)
            try:
                low, high = km.confidence_interval_.median().values
            except Exception:
                pass
            rows.append(["Tutti", med, low, high, int(work["event"].sum()), int(len(work))])
        else:
            for gname, gdf in work.groupby("group", dropna=False):
                km = KaplanMeierFitter().fit(gdf["time"], gdf["event"], label=str(gname))
                med = km.median_survival_time_
                low = high = np.nan
                rows.append([str(gname), med, low, high, int(gdf["event"].sum()), int(len(gdf))])
        tbl = pd.DataFrame(rows, columns=["Gruppo", "Mediana", "CI 2.5%", "CI 97.5%", "Eventi", "N"])
    else:
        rows = []
        if group_col is None:
            x, y = _km_curve_fallback(work["time"].to_numpy(float), work["event"].to_numpy(int))
            rows.append(["Tutti", _median_survival_from_km(x, y), np.nan, np.nan, int(work["event"].sum()), int(len(work))])
        else:
            for gname, gdf in work.groupby("group", dropna=False):
                x, y = _km_curve_fallback(gdf["time"].to_numpy(float), gdf["event"].to_numpy(int))
                rows.append([str(gname), _median_survival_from_km(x, y), np.nan, np.nan, int(gdf["event"].sum()), int(len(gdf))])
        tbl = pd.DataFrame(rows, columns=["Gruppo", "Mediana", "CI 2.5%", "CI 97.5%", "Eventi", "N"])

    st.markdown("**Statistiche riassuntive**")
    st.dataframe(tbl, use_container_width=True)

    # Log-rank (e memorizzo p-value per interpretazione)
    p_logrank = None
    if group_col is not None:
        levels = list(work["group"].astype(str).unique())
        if len(levels) == 2:
            g0 = work[work["group"] == levels[0]]
            g1 = work[work["group"] == levels[1]]
            if _has_lifelines:
                res = logrank_test(g0["time"], g1["time"], g0["event"], g1["event"])
                p_logrank = float(res.p_value)
                st.metric("Log-rank (2 gruppi) — p-value", fmt_p(p_logrank))
            else:
                _, p_logrank = _logrank_two_groups_fallback(g1["time"].values, g1["event"].values,
                                                            g0["time"].values, g0["event"].values)
                st.metric("Log-rank (2 gruppi) — p-value", fmt_p(p_logrank))
        elif len(levels) > 2 and _has_lifelines:
            res = multivariate_logrank_test(work["time"], work["group"], work["event"])
            p_logrank = float(res.p_value)
            st.metric("Log-rank (k gruppi) — p-value", fmt_p(p_logrank))
            with st.expander("Confronti **pairwise** (Holm-Bonferroni)", expanded=False):
                pairs = list(itertools.combinations(levels, 2))
                rows = []
                for a, b in pairs:
                    A = work[work["group"] == a]; B = work[work["group"] == b]
                    p = logrank_test(A["time"], B["time"], A["event"], B["event"]).p_value
                    rows.append([a, b, p])
                dfp = pd.DataFrame(rows, columns=["A", "B", "p"])
                dfp = dfp.sort_values("p").reset_index(drop=True)
                m = len(dfp)
                dfp["p_Holm"] = [min((m - i) * p, 1.0) for i, p in enumerate(dfp["p"])]
                st.dataframe(dfp, use_container_width=True)

# Numeri a rischio (tabella) — versione robusta e senza KeyError
with st.expander("📊 Numeri a rischio (seleziona tempi)", expanded=False):
    tmin = float(work["time"].min()); tmax = float(work["time"].max())
    def r2(x): return float(np.round(x, 2))
    linpts = [r2(x) for x in np.linspace(tmin, tmax, 10)]
    qpts   = [r2(x) for x in np.quantile(work["time"], [0, 0.25, 0.5, 0.75, 1.0])]
    options = sorted(set(linpts + qpts))

    def snap_to_options(vals: list[float], opts: list[float]) -> list[float]:
        if not opts: return []
        snapped = []
        for v in vals:
            if v in opts:
                snapped.append(v)
            else:
                nearest = min(opts, key=lambda o: abs(o - v))
                if nearest not in snapped:
                    snapped.append(nearest)
        return snapped or [opts[0]]

    default_opts = snap_to_options(qpts, options)

    cutpoints = st.multiselect("Tempi in cui mostrare N a rischio",
                               options=options,
                               default=default_opts, key=k("risk_pts"))
    if cutpoints:
        # ⚠️ qui passiamo 'group' se esiste, altrimenti None → evita KeyError
        group_key = "group" if ("group" in work.columns) else None
        risk = _numbers_at_risk(work, "time", "event", group_key, cutpoints)
        st.dataframe(risk.astype(int), use_container_width=True)
        st.caption("Valori mostrati: **numero di soggetti ancora a rischio** a ciascun tempo selezionato (censurati ed eventi pregressi esclusi).")

with st.expander("ℹ️ Come leggere KM, log-rank e N a rischio", expanded=False):
    st.markdown(
        "- **S(t)**: probabilità di **non** aver avuto l’evento entro t (linea a gradini). Intervalli di confidenza (se mostrati) quantificano l’incertezza.  \n"
        "- **Mediana di sopravvivenza**: tempo per cui S(t)=0.5; se la curva resta >0.5 non è stimabile.  \n"
        "- **Tick di censura**: istanti di uscita dall’osservazione; non rappresentano eventi.  \n"
        "- **Log-rank**: p-value piccolo ⇒ differenza globale tra le curve (assume **hazard proporzionali**).  \n"
        "- **Numeri a rischio**: aiutano a valutare l’**affidabilità** della coda delle curve; valori molto bassi implicano stime **instabili**."
    )

# Sintesi automatica risultati KM/log-rank
with st.expander("📝 Interpretazione automatica — KM & log-rank", expanded=True):
    bullets = []
    if 'tbl' in locals() and isinstance(tbl, pd.DataFrame) and not tbl.empty:
        try:
            order = tbl.sort_values("Mediana").reset_index(drop=True)
            med_txt = " • ".join([f"{row['Gruppo']}: mediana={row['Mediana']:.2f}" if pd.notnull(row['Mediana']) else f"{row['Gruppo']}: mediana n/d"
                                  for _, row in order.iterrows()])
            bullets.append(f"**Mediane** (se stimabili): {med_txt}.")
        except Exception:
            pass
        try:
            evt_txt = " • ".join([f"{row['Gruppo']}: eventi={int(row['Eventi'])}/N={int(row['N'])}" for _, row in tbl.iterrows()])
            bullets.append(f"**Eventi/N** per gruppo: {evt_txt}.")
        except Exception:
            pass
    if group_col is not None:
        bullets.append(f"**Log-rank**: p = {fmt_p(p_logrank)} ⇒ {interpret_p(p_logrank, alpha)} (α = {alpha:.2f}).")
    if 'risk' in locals() and isinstance(risk, pd.DataFrame):
        low_n = (risk.min(axis=1) < 10).any()
        if low_n:
            bullets.append("Nella coda temporale alcuni gruppi hanno **N a rischio < 10**: interpretare con cautela le differenze oltre tali tempi.")
    if not bullets:
        bullets.append("Nessuna sintesi disponibile (fornire gruppi o risultati adeguati).")
    st.markdown("\n\n".join([f"- {b}" for b in bullets]))

# ──────────────────────────────────────────────────────────────────────────────
# STEP 3 — Modello di Cox PH
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("### 3) Modello di Cox (hazard proporzionali)")
if not _has_lifelines:
    st.info("Il modello di Cox richiede **lifelines**. Installare `lifelines` per abilitare la sezione seguente.")
else:
    covar_opts = [c for c in df.columns if c not in {time_col, event_col}]
    colA, colB = st.columns([1.6, 1.4])
    with colA:
        covars = st.multiselect("Covariate (selezionare una o più)", options=covar_opts, key=k("covars"))
    with colB:
        zscore = st.checkbox("Standardizza variabili numeriche (z-score)", value=False, key=k("z"))
        st.caption("Ties gestiti automaticamente con **Efron** (impostazione di default in lifelines).")

    if covars:
        X = df[[time_col, event_col] + covars].copy()
        X[event_col] = event
        if zscore:
            for c in covars:
                if pd.api.types.is_numeric_dtype(X[c]):
                    s = pd.to_numeric(X[c], errors="coerce")
                    mu, sd = float(s.mean()), float(s.std(ddof=1))
                    if sd and sd > 0:
                        X.loc[:, c] = (s - mu) / sd
        X = pd.get_dummies(X, columns=[c for c in covars if not pd.api.types.is_numeric_dtype(df[c])],
                           drop_first=True)
        X = X.dropna(subset=[time_col, event_col])
        X = X[X[time_col] >= 0]

        if X.empty or X[event_col].nunique() < 2:
            st.error("Dati insufficienti per la stima del modello di Cox.")
        else:
            cph = CoxPHFitter()
            try:
                cph.fit(X, duration_col=time_col, event_col=event_col, show_progress=False)
                st.markdown("**Tabella dei coefficienti (scala HR)**")
                summ = cph.summary.copy()
                cols = {
                    "exp(coef)": "HR",
                    "exp(coef) lower 95%": "HR CI 2.5%",
                    "exp(coef) upper 95%": "HR CI 97.5%",
                    "p": "p"
                }
                view = summ.rename(columns=cols)[["HR", "HR CI 2.5%", "HR CI 97.5%", "p"]].copy()
                view = view.round(3)
                st.dataframe(view, use_container_width=True)

                m1, m2, m3 = st.columns(3)
                m1.metric("Log-likelihood", f"{cph.log_likelihood_: .3f}")
                try:
                    aic_like = -2.0 * cph.log_likelihood_ + 2 * len(cph.params_)
                    m2.metric("AIC (parz.)", f"{aic_like:.1f}")
                except Exception:
                    pass
                try:
                    concord = float(cph.concordance_index_)
                    m3.metric("Concordance index", f"{concord:.3f}")
                except Exception:
                    pass

                with st.expander("ℹ️ Come interpretare il **modello di Cox**", expanded=False):
                    st.markdown(
                        "- **HR (Hazard Ratio)**: fattore moltiplicativo sul **rischio istantaneo**. HR>1 aumenta il rischio, HR<1 lo riduce.  \n"
                        "- **Intervallo di confidenza**: se il CI di HR **non** include 1, l’effetto è statisticamente significativo.  \n"
                        "- **Concordance index**: probabilità che l’ordine dei tempi osservati sia coerente con l’ordine dei rischi stimati (discriminazione).  \n"
                        "- **Ties**: gestiti con il metodo di **Efron** (default)."
                    )

                # Diagnostica PH (Schoenfeld)
                st.markdown("#### Verifica dell’assunzione di **hazard proporzionali**")
                try:
                    zph = proportional_hazard_test(cph, X, time_transform="rank")
                    ztab = zph.summary.copy()
                    ztab = ztab.rename(columns={"test_statistic": "χ²", "p": "p"})
                    ztab = ztab[["χ²", "p"]].round(4)
                    st.dataframe(ztab, use_container_width=True)
                    st.caption("p-value piccoli indicano **violazione** dell’assunzione PH per la covariata corrispondente (test su residui di Schoenfeld).")
                except Exception as e:
                    st.caption(f"Test di PH non disponibile: {e}")

                # Sopravvivenza predetta (due profili)
                with st.expander("📈 Sopravvivenza predetta per profili di covariate (facoltativo)"):
                    base = X.drop(columns=[time_col, event_col]).median(numeric_only=True)
                    prof1 = base.copy(); prof2 = base.copy()
                    for c in base.index:
                        if c in X.columns and pd.api.types.is_numeric_dtype(X[c]):
                            prof2[c] = base[c] + X[c].std(ddof=1)
                    try:
                        sf1 = cph.predict_survival_function(prof1.to_frame().T)
                        sf2 = cph.predict_survival_function(prof2.to_frame().T)
                        figp = go.Figure()
                        figp.add_trace(go.Scatter(x=sf1.index.values, y=sf1.values.flatten(),
                                                  mode="lines", name="Profilo 1 (baseline)",
                                                  line=dict(width=3)))
                        figp.add_trace(go.Scatter(x=sf2.index.values, y=sf2.values.flatten(),
                                                  mode="lines", name="Profilo 2 (+1 SD numeriche)",
                                                  line=dict(width=3, dash="dash")))
                        figp.update_layout(template="simple_white", height=420,
                                           title="Sopravvivenza predetta (Cox)",
                                           xaxis_title=f"Tempo ({time_col})", yaxis_title="S(t)")
                        st.plotly_chart(figp, use_container_width=True)
                        st.caption("Confrontare i profili per quantificare l’effetto combinato delle covariate sulla sopravvivenza prevista.")
                    except Exception:
                        st.info("Impossibile calcolare le curve predette con il profilo scelto.")

                # ───────── Sintesi automatica risultati Cox
                with st.expander("📝 Interpretazione automatica — Cox PH", expanded=True):
                    lines = []
                    # covariate significative
                    try:
                        sig = view[view["p"] < alpha].sort_values("p")
                        if not sig.empty:
                            covs = " • ".join([f"{ix}: HR={row['HR']:.2f} (p={fmt_p(row['p'])})" for ix, row in sig.iterrows()])
                            lines.append(f"**Covariate significative** (α={alpha:.2f}): {covs}.")
                        else:
                            lines.append(f"Nessuna covariata significativa al livello α={alpha:.2f}.")
                    except Exception:
                        pass
                    # concordance index
                    try:
                        c_index = float(cph.concordance_index_)
                        qualit = "ottima" if c_index >= 0.75 else ("discreta" if c_index >= 0.65 else "limitata")
                        lines.append(f"**Concordance index** = {c_index:.3f} ⇒ capacità discriminante **{qualit}**.")
                    except Exception:
                        pass
                    if not lines:
                        lines.append("Nessuna sintesi disponibile (esiti modello non calcolabili).")
                    st.markdown("\n\n".join([f"- {t}" for t in lines]))

            except Exception as e:
                st.error(f"Errore nella stima del modello di Cox: {e}")
    else:
        st.info("Selezionare almeno **una** covariata per stimare il modello di Cox.")

# ──────────────────────────────────────────────────────────────────────────────
# Esportazione tabelle principali
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("### 4) Esporta risultati")
if 'tbl' in locals() and isinstance(tbl, pd.DataFrame):
    csv_km = tbl.to_csv(index=False).encode("utf-8")
    st.download_button("Scarica statistiche KM (CSV)", data=csv_km,
                       file_name="km_summary.csv", mime="text/csv", key=k("dl_km"))

# ──────────────────────────────────────────────────────────────────────────────
# Navigazione
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("---")
nav1, nav2 = st.columns(2)
with nav1:
    if st.button("⬅️ Torna: Agreement", use_container_width=True, key=k("go_prev")):
        try:
            st.switch_page("pages/10_📏_Agreement.py")
        except Exception:
            pass
with nav2:
    if st.button("➡️ Vai: Longitudinale — Misure ripetute", use_container_width=True, key=k("go_next")):
        for target in [
            "pages/12_📈_Longitudinale_Misure_Ripetute.py",
            "pages/12_📈_Longitudinale_Misure_Ripetute (1).py",
            "pages/12_📈_Longitudinale_Misure_Ripetute....py"
        ]:
            try:
                st.switch_page(target)
                break
            except Exception:
                continue
