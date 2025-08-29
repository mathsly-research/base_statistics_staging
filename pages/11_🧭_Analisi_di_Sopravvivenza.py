# -*- coding: utf-8 -*-
# pages/12_📈_Longitudinale_Misure_Ripetute.py
from __future__ import annotations

import itertools
import math
import numpy as np
import pandas as pd
import streamlit as st

# Plotly
try:
    import plotly.express as px
    import plotly.graph_objects as go
except Exception:
    px = None
    go = None

# Stats: statsmodels, scipy, pingouin (opzionali)
try:
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    from statsmodels.stats.anova import AnovaRM
    _has_sm = True
except Exception:
    _has_sm = False

try:
    from scipy import stats as sps
    _has_scipy = True
except Exception:
    _has_scipy = False

try:
    import pingouin as pg  # sfericità, pairwise, epsilon GG/HF, ecc.
    _has_pg = True
except Exception:
    _has_pg = False

# ──────────────────────────────────────────────────────────────────────────────
# DATA STORE (uniforme agli altri moduli)
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
# CONFIG & NAV
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="📈 Longitudinale — Misure ripetute", layout="wide")
try:
    from nav import sidebar
    sidebar()
except Exception:
    pass

KEY = "rm"
def k(name: str) -> str:
    return f"{KEY}_{name}"

# ──────────────────────────────────────────────────────────────────────────────
# UTILS
# ──────────────────────────────────────────────────────────────────────────────
def fmt_p(p: float | None) -> str:
    if p is None or p != p: return "—"
    if p < 1e-4: return "< 1e-4"
    return f"{p:.4f}"

def holm_correction(pvals: list[float]) -> list[float]:
    # Holm–Bonferroni
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    for rank, idx in enumerate(order):
        adj[idx] = min((m - rank) * pvals[idx], 1.0)
    return adj.tolist()

def infer_time_order(levels: list[str]) -> list[str]:
    # prova ad ordinare estraendo eventuali numeri, altrimenti alfabetico
    def keyf(s):
        import re
        nums = re.findall(r"[-+]?\d*\.?\d+", str(s))
        return float(nums[0]) if nums else None
    nums = [keyf(x) for x in levels]
    if all(v is not None for v in nums):
        return [x for _, x in sorted(zip(nums, levels))]
    return sorted(levels)

def to_long_from_wide(df: pd.DataFrame, id_col: str, time_cols: list[str],
                      value_name: str = "y", time_name: str = "time",
                      between_col: str | None = None) -> pd.DataFrame:
    id_vars = [id_col] + ([between_col] if between_col else [])
    long_df = df.melt(id_vars=id_vars, value_vars=time_cols,
                      var_name=time_name, value_name=value_name)
    # ordina livelli del tempo
    long_df[time_name] = long_df[time_name].astype(str)
    order = infer_time_order(sorted(long_df[time_name].unique()))
    long_df[time_name] = pd.Categorical(long_df[time_name], categories=order, ordered=True)
    return long_df

# ──────────────────────────────────────────────────────────────────────────────
# HEADER
# ──────────────────────────────────────────────────────────────────────────────
st.title("📈 Longitudinale — Misure ripetute")
st.caption("ANOVA per misure ripetute, alternativa non parametrica (Friedman) e modello lineare a effetti misti (LMM). Strumenti per passare da **wide** a **long** e guida all’interpretazione.")

ensure_initialized()
df = get_active(required=True)
with st.expander("Stato dati", expanded=False):
    stamp_meta()
if df is None or df.empty:
    st.stop()

all_cols = list(df.columns)
num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
cat_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]

# ──────────────────────────────────────────────────────────────────────────────
# STEP 1 — FORMATO DATI (Long vs Wide) + TRASFORMAZIONE
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("### 1) Formato dei dati")
c1, c2 = st.columns([1.2, 1.8])
with c1:
    data_format = st.radio("I dati sono già in **long** o sono in **wide**?", ["Long", "Wide"], horizontal=True, key=k("format"))
with c2:
    st.caption(
        "- **Long**: una riga per soggetto × tempo (colonne: ID, Tempo, Misura).  \n"
        "- **Wide**: una riga per soggetto con **più colonne** (Misura_T1, Misura_T2, …).  \n"
        "Se è in wide, usi il pannello seguente per convertirlo in long."
    )

with st.expander("🛠️ Converti da **wide** a **long** (se necessario)", expanded=(data_format == "Wide")):
    cA, cB = st.columns([1.2, 1.8])
    with cA:
        id_col = st.selectbox("ID soggetto", options=all_cols, key=k("id_w"))
        between_col = st.selectbox("Fattore **between** (opzionale)", options=["— nessuno —"] + all_cols, key=k("bet_w"))
        between_col = None if between_col == "— nessuno —" else between_col
        time_cols = st.multiselect("Colonne delle **misure nel tempo** (wide)", options=[c for c in all_cols if c != id_col and c != between_col], key=k("time_w"))
    with cB:
        value_name = st.text_input("Nome colonna misura (es. y)", value="y", key=k("valname"))
        time_name = st.text_input("Nome colonna tempo (es. time)", value="time", key=k("timename"))
        if time_cols:
            st.caption("Anteprima dei livelli tempo dedotti e del loro ordine.")
            preview = infer_time_order([str(c) for c in time_cols])
            st.code(", ".join(preview))
    long_df = None
    if st.button("↳ Crea dataset **long**", key=k("do_long")):
        if not time_cols:
            st.warning("Selezionare almeno una colonna di misura.")
        else:
            long_df = to_long_from_wide(df, id_col=id_col, time_cols=time_cols,
                                        value_name=value_name, time_name=time_name, between_col=between_col)
            st.session_state[k("long_df")] = long_df
            st.success(f"Creato dataset long con colonne: **{id_col}**, **{time_name}**, **{value_name}**" + (f", **{between_col}**" if between_col else ""))

# Costruzione dataset di lavoro in long
if data_format == "Long":
    c1, c2, c3 = st.columns(3)
    with c1:
        id_long = st.selectbox("Colonna **ID**", options=all_cols, key=k("id_l"))
    with c2:
        time_long = st.selectbox("Colonna **Tempo**", options=[c for c in all_cols if c != id_long], key=k("time_l"))
    with c3:
        y_long = st.selectbox("Colonna **Misura** (continua)", options=[c for c in all_cols if c not in {id_long, time_long}], key=k("y_l"))
    between_long = st.selectbox("Fattore **between** (opzionale)", options=["— nessuno —"] + [c for c in all_cols if c not in {id_long, time_long, y_long}], key=k("bet_l"))
    between_long = None if between_long == "— nessuno —" else between_long
    work = df[[id_long, time_long, y_long] + ([between_long] if between_long else [])].copy()
    work.columns = ["id", "time", "y"] + (["between"] if between_long else [])
else:
    # prendi dall'expander se creato, altrimenti attesa
    work = st.session_state.get(k("long_df"))
    if work is None:
        st.info("Creare il dataset **long** con il pannello sopra, poi procedere.")
        st.stop()
    # già con nomi id/time/y/(between)
    if "id" not in work.columns or "time" not in work.columns or "y" not in work.columns:
        st.error("Il dataset long deve contenere le colonne: id, time, y (e facoltativamente between).")
        st.stop()

# Pulizia base
work = work.dropna(subset=["id", "time"])
work["y"] = pd.to_numeric(work["y"], errors="coerce")
work = work.dropna(subset=["y"])
work["id"] = work["id"].astype(str)
work["time"] = work["time"].astype(str)
# Ordine dei livelli tempo (intelligente + personalizzabile)
auto_order = infer_time_order(sorted(work["time"].unique()))
order_choice = st.multiselect("Ordine dei livelli **tempo** (trascina/riordina)", options=auto_order, default=auto_order, key=k("ord_t"))
if order_choice:
    work["time"] = pd.Categorical(work["time"], categories=order_choice, ordered=True)
else:
    work["time"] = pd.Categorical(work["time"], categories=auto_order, ordered=True)

st.markdown("#### Anteprima dati (long)")
st.dataframe(work.head(10), use_container_width=True)

# ──────────────────────────────────────────────────────────────────────────────
# STEP 2 — CHECK ASSUNZIONI
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("### 2) Verifica delle assunzioni")
c1, c2, c3 = st.columns([1.2, 1.2, 1.6])
with c1:
    alpha = st.select_slider("α (significatività)", options=[0.01, 0.05, 0.10], value=0.05, key=k("alpha"))
with c2:
    check_norm = st.checkbox("Controlla **normalità** entro livello tempo (Shapiro)", value=True, key=k("chk_norm"))
with c3:
    check_spher = st.checkbox("Controlla **sfericità** (Mauchly, se disponibile)", value=True, key=k("chk_sph"))

if check_norm and _has_scipy:
    rows = []
    for lev, g in work.groupby("time"):
        yv = pd.to_numeric(g["y"], errors="coerce").dropna()
        if len(yv) >= 3:
            try:
                # limiti Shapiro consigliati: n<=5000
                samp = yv.sample(min(len(yv), 5000), random_state=123)
                W, p = sps.shapiro(samp)
            except Exception:
                W, p = (np.nan, np.nan)
        else:
            W, p = (np.nan, np.nan)
        rows.append([str(lev), len(yv), W, p])
    tab_norm = pd.DataFrame(rows, columns=["Tempo", "N", "Shapiro W", "p"])
    st.markdown("**Normalità (per livello tempo)**")
    st.dataframe(tab_norm.round(4), use_container_width=True)
    st.caption("p piccoli indicano **deviazione dalla normalità**. In presenza di forti deviazioni considerare **Friedman** o **LMM**.")
elif check_norm and not _has_scipy:
    st.info("SciPy non disponibile: salto il test di Shapiro.")

sphericity_p = None
epsilon_gg = epsilon_hf = None
if check_spher:
    if _has_pg:
        try:
            # pingouin richiede dati wide per sfericità: pivot per soggetto × tempo
            wide = work.pivot_table(index="id", columns="time", values="y", aggfunc="mean")
            spher, sphericity_p, W = pg.sphericity(wide.dropna(axis=0, how="any"))
            # epsilon GG/HF
            eps = pg.epsilon(wide.dropna(axis=0, how="any"))
            epsilon_gg = float(eps.loc["GG", "epsilon"]) if "GG" in eps.index else None
            epsilon_hf = float(eps.loc["HF", "epsilon"]) if "HF" in eps.index else None
            st.metric("Test di Mauchly (sfericità) — p", fmt_p(float(sphericity_p)) if sphericity_p is not None else "—")
            if epsilon_gg:
                st.caption(f"Epsilon **Greenhouse–Geisser** ≈ {epsilon_gg:.3f} • Epsilon **Huynh–Feldt** ≈ {epsilon_hf:.3f}" if epsilon_hf else
                           f"Epsilon **Greenhouse–Geisser** ≈ {epsilon_gg:.3f}")
        except Exception as e:
            st.info(f"Sfericità non valutabile: {e}")
    else:
        st.info("Pingouin non disponibile: salto il test di sfericità. In alternativa usare **LMM** o applicare correzioni conservative.")

# ──────────────────────────────────────────────────────────────────────────────
# STEP 3 — SCELTA MODELLO E ANALISI
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("### 3) Analisi")
model_choice = st.radio("Scegli l’analisi principale", [
    "ANOVA per misure ripetute (entro-soggetto)",
    "Friedman (non parametrico)",
    "Modello lineare a effetti misti (LMM)"
], index=0, key=k("model"))

# ———————————————————— ANOVA RM ————————————————————
if model_choice.startswith("ANOVA"):
    if not _has_sm:
        st.error("Statsmodels non disponibile: impossibile eseguire AnovaRM. Usi **Friedman** o **LMM**.")
        st.stop()

    # AnovaRM richiede dati completi (complete-case per soggetto)
    comp = work.dropna(subset=["y"])
    counts = comp.groupby("id")["time"].nunique()
    full_ids = counts[counts == work["time"].nunique()].index
    anova_df = comp[comp["id"].isin(full_ids)].copy()

    if anova_df["id"].nunique() < 2 or anova_df["time"].nunique() < 2:
        st.error("Dati insufficienti/bilanciamento insufficiente per AnovaRM. Usi **LMM** o **Friedman**.")
    else:
        try:
            res = AnovaRM(anova_df, depvar="y", subject="id", within=["time"]).fit()
            st.markdown("**ANOVA per misure ripetute (entro-soggetto)**")
            st.dataframe(res.anova_table.round(4), use_container_width=True)

            # Effetto del tempo: calcolo eta^2 parziale se disponibile
            try:
                row = res.anova_table.loc["time"]
                F = float(row["F Value"]); df1 = float(row["Num DF"]); df2 = float(row["Den DF"])
                eta_p2 = (F * df1) / (F * df1 + df2) if (F == F and df1 > 0 and df2 > 0) else np.nan
            except Exception:
                eta_p2 = np.nan

            c1, c2, c3 = st.columns(3)
            with c1: st.metric("Livelli tempo", f"{work['time'].nunique()}")
            with c2: st.metric("Soggetti completi", f"{len(full_ids)}")
            with c3: st.metric("η² parziale (tempo)", f"{eta_p2:.3f}" if eta_p2 == eta_p2 else "—")
            st.caption("η² parziale ≈ proporzione di varianza dell’effetto di **tempo** sul totale residuo+effetto. 0.01 piccolo • 0.06 medio • 0.14 grande (indicativo).")

            # Correzioni per sfericità (se disponibili)
            if _has_pg and sphericity_p is not None and epsilon_gg is not None:
                st.markdown("**Correzioni per sfericità**")
                try:
                    # Applico GG/HF ai gradi di libertà e ricalcolo p da F
                    row = res.anova_table.loc["time"]
                    F = float(row["F Value"]); df1 = float(row["Num DF"]); df2 = float(row["Den DF"])
                    for label, eps in [("GG", epsilon_gg), ("HF", epsilon_hf)]:
                        if eps and eps == eps:
                            from scipy.stats import f as fdist if _has_scipy else (None)
                            df1_c = eps * df1
                            df2_c = eps * df2
                            if _has_scipy:
                                p_corr = float(fdist.sf(F, df1_c, df2_c))
                                st.write(f"- **{label}**: F({df1_c:.2f}, {df2_c:.2f}) = {F:.3f}, p = {fmt_p(p_corr)}")
                            else:
                                st.write(f"- **{label}**: F({df1_c:.2f}, {df2_c:.2f}) = {F:.3f} (p non calcolabile senza SciPy)")
                except Exception:
                    pass
            elif sphericity_p is not None:
                st.caption("Sfericità violata: valutare **correzioni** o preferire **LMM**.")

            # Confronti post-hoc (pairwise entro-soggetto)
            with st.expander("Confronti **post-hoc** (pairwise entro-soggetto)"):
                if _has_pg:
                    pc = pg.pairwise_ttests(dv="y", within="time", subject="id", data=anova_df,
                                            padjust="holm", effsize="hedges")
                    st.dataframe(pc.round(4), use_container_width=True)
                elif _has_scipy:
                    # pairwise t-test per coppie di livelli
                    levels = list(anova_df["time"].cat.categories if isinstance(anova_df["time"].dtype, pd.CategoricalDtype) else sorted(anova_df["time"].unique()))
                    rows = []
                    for a, b in itertools.combinations(levels, 2):
                        da = anova_df[anova_df["time"] == a].set_index("id")["y"]
                        db = anova_df[anova_df["time"] == b].set_index("id")["y"]
                        common = da.index.intersection(db.index)
                        if len(common) >= 3:
                            t, p = sps.ttest_rel(da.loc[common], db.loc[common], nan_policy="omit")
                            rows.append([a, b, float(t), float(p)])
                    if rows:
                        dfp = pd.DataFrame(rows, columns=["A", "B", "t", "p"])
                        dfp["p_Holm"] = holm_correction(dfp["p"].tolist())
                        st.dataframe(dfp.round(4), use_container_width=True)
                    else:
                        st.info("Confronti non disponibili (campioni troppo piccoli).")
                else:
                    st.info("Né Pingouin né SciPy disponibili: impossibile calcolare i confronti post-hoc.")

# ———————————————————— FRIEDMAN ————————————————————
elif model_choice.startswith("Friedman"):
    if not _has_scipy:
        st.error("SciPy non disponibile: impossibile eseguire il test di Friedman.")
    else:
        wide = work.pivot_table(index="id", columns="time", values="y", aggfunc="mean")
        wide = wide.dropna(axis=0, how="any")
        levels = list(wide.columns)
        if wide.shape[0] < 2 or len(levels) < 3:
            st.error("Friedman richiede ≥2 soggetti completi e ≥3 tempi.")
        else:
            # Friedman
            samples = [wide[c].values for c in levels]
            Q, p = sps.friedmanchisquare(*samples)
            st.markdown("**Test di Friedman (non parametrico)**")
            st.write(f"Q = {Q:.3f}, p = {fmt_p(p)}")
            st.caption("Friedman testa differenze **mediane** tra tempi (entro-soggetto) senza assumere normalità.")

            # Post-hoc Wilcoxon con Holm
            with st.expander("Confronti post-hoc (Wilcoxon con Holm)"):
                rows = []
                for a, b in itertools.combinations(levels, 2):
                    w, p = sps.wilcoxon(wide[a], wide[b], zero_method="wilcox", alternative="two-sided", correction=False)
                    rows.append([a, b, float(w), float(p)])
                dfp = pd.DataFrame(rows, columns=["A", "B", "W", "p"])
                dfp["p_Holm"] = holm_correction(dfp["p"].tolist())
                st.dataframe(dfp.round(4), use_container_width=True)

# ———————————————————— LMM ————————————————————
else:
    if not _has_sm:
        st.error("Statsmodels non disponibile: impossibile eseguire LMM.")
    else:
        treat_time_as = st.radio("Trattamento del **tempo**", ["Categorico (C(time))", "Lineare (tempo numerico)"], horizontal=True, key=k("time_as"))
        # Costruzione formula
        if treat_time_as.startswith("Categorico"):
            formula = "y ~ C(time)"
        else:
            # prova a estrarre componente numerica dal livello
            def to_num(s):
                import re
                m = re.findall(r"[-+]?\d*\.?\d+", str(s))
                return float(m[0]) if m else np.nan
            work["_time_num"] = work["time"].map(to_num)
            if work["_time_num"].isna().all():
                st.warning("Impossibile interpretare il tempo come numerico dai livelli. Uso C(time).")
                formula = "y ~ C(time)"
            else:
                formula = "y ~ _time_num"

        if "between" in work.columns:
            formula = formula + " + C(between)"

        st.code(f"MixedLM: {formula} + (1 | id)")

        # Modello con random intercept per soggetto
        try:
            md = smf.mixedlm(formula, data=work, groups=work["id"])
            mdf = md.fit(method="lbfgs", reml=True, maxiter=200)
            st.markdown("**Modello lineare a effetti misti (random intercept per soggetto)**")
            st.text(mdf.summary().as_text())

            # Omnibus per l'effetto del tempo (LRT vs modello nullo)
            try:
                md0 = smf.mixedlm("y ~ 1" + (" + C(between)" if "between" in work.columns else ""),
                                  data=work, groups=work["id"])
                m0 = md0.fit(method="lbfgs", reml=True, maxiter=200)
                ll1, ll0 = float(mdf.llf), float(m0.llf)
                chi2 = 2 * (ll1 - ll0)
                df_diff = mdf.df_modelwc - m0.df_modelwc
                from scipy.stats import chi2 as chi2dist if _has_scipy else (None)
                if _has_scipy and df_diff > 0:
                    p_lrt = float(chi2dist.sf(chi2, df=int(df_diff)))
                    st.metric("LRT (tempo vs nullo) — p", fmt_p(p_lrt))
                else:
                    st.caption(f"LRT non disponibile (df={df_diff}).")
            except Exception:
                pass

            # Confronti marginal means (solo se Pingouin)
            with st.expander("Confronti marginali stimati (EMMs) tra tempi"):
                if _has_pg:
                    try:
                        emms = pg.marginal_means(data=work, dv="y", within="time", subject="id")
                        st.dataframe(emms.round(4), use_container_width=True)
                        if _has_scipy:
                            # pairwise sulle emmeans con Holm
                            levs = list(work["time"].cat.categories if isinstance(work["time"].dtype, pd.CategoricalDtype) else sorted(work["time"].unique()))
                            rows = []
                            for a, b in itertools.combinations(levs, 2):
                                ya = work.loc[work["time"] == a, "y"].groupby(work["id"]).mean()
                                yb = work.loc[work["time"] == b, "y"].groupby(work["id"]).mean()
                                common = ya.index.intersection(yb.index)
                                if len(common) >= 3:
                                    t, p = sps.ttest_rel(ya.loc[common], yb.loc[common], nan_policy="omit")
                                    rows.append([a, b, float(t), float(p)])
                            if rows:
                                dfp = pd.DataFrame(rows, columns=["A", "B", "t", "p"])
                                dfp["p_Holm"] = holm_correction(dfp["p"].tolist())
                                st.dataframe(dfp.round(4), use_container_width=True)
                    except Exception as e:
                        st.info(f"EMMs non disponibili: {e}")
                else:
                    st.info("Pingouin non disponibile: stima EMMs non eseguita.")

        except Exception as e:
            st.error(f"Errore nella stima LMM: {e}")
            st.stop()

# ──────────────────────────────────────────────────────────────────────────────
# STEP 4 — GRAFICI
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("### 4) Visualizzazioni")
g1, g2 = st.columns(2)

with g1:
    show_spaghetti = st.checkbox("Mostra **spaghetti plot** (traiettorie individuali)", value=False, key=k("spag"))
    if px is not None:
        if show_spaghetti:
            # campionamento se troppi soggetti
            ids = work["id"].unique().tolist()
            max_lines = st.slider("Numero massimo soggetti nel grafico", min_value=10, max_value=300, value=min(100, len(ids)), key=k("maxlines"))
            sel_ids = ids[:max_lines]
            fig = px.line(work[work["id"].isin(sel_ids)].sort_values(["id","time"]),
                          x="time", y="y", color="id", line_group="id",
                          template="simple_white", markers=True,
                          title="Traiettorie individuali (spaghetti)")
            fig.update_layout(showlegend=False, height=420, xaxis_title="Tempo", yaxis_title="Misura")
            st.plotly_chart(fig, use_container_width=True)
        else:
            agg = work.groupby("time")["y"].agg(["mean", "std", "count"]).reset_index()
            agg["se"] = agg["std"] / np.sqrt(agg["count"])
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=agg["time"], y=agg["mean"], mode="lines+markers", name="Media",
                                     line=dict(width=3)))
            fig.add_trace(go.Scatter(x=agg["time"], y=agg["mean"] + 1.96*agg["se"], mode="lines",
                                     line=dict(width=0), showlegend=False))
            fig.add_trace(go.Scatter(x=agg["time"], y=agg["mean"] - 1.96*agg["se"], mode="lines",
                                     fill="tonexty", fillcolor="rgba(127,127,127,0.15)",
                                     line=dict(width=0), showlegend=False))
            fig.update_layout(template="simple_white", height=420, title="Media ± IC 95% nel tempo",
                              xaxis_title="Tempo", yaxis_title="Misura")
            st.plotly_chart(fig, use_container_width=True)

with g2:
    if px is not None:
        plot_kind = st.radio("Distribuzione per tempo", ["Box", "Violin"], horizontal=True, key=k("dist_kind"))
        if plot_kind == "Box":
            fig = px.box(work, x="time", y="y", color="time", points="outliers", template="simple_white",
                         title="Distribuzione per tempo (boxplot)")
        else:
            fig = px.violin(work, x="time", y="y", color="time", box=True, points=False, template="simple_white",
                            title="Distribuzione per tempo (violin + box)")
        fig.update_layout(showlegend=False, height=420, xaxis_title="Tempo", yaxis_title="Misura")
        st.plotly_chart(fig, use_container_width=True)

# Se fattore between, media per gruppo
if "between" in work.columns and px is not None:
    st.markdown("#### Andamento medio per **gruppo (between)**")
    agg2 = work.groupby(["between","time"])["y"].agg(["mean", "std", "count"]).reset_index()
    agg2["se"] = agg2["std"] / np.sqrt(agg2["count"])
    fig = go.Figure()
    for gname, gdf in agg2.groupby("between"):
        fig.add_trace(go.Scatter(x=gdf["time"], y=gdf["mean"], mode="lines+markers", name=str(gname), line=dict(width=3)))
        fig.add_trace(go.Scatter(x=gdf["time"], y=gdf["mean"] + 1.96*gdf["se"], mode="lines",
                                 line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=gdf["time"], y=gdf["mean"] - 1.96*gdf["se"], mode="lines",
                                 fill="tonexty", fillcolor="rgba(127,127,127,0.12)",
                                 line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.update_layout(template="simple_white", height=440,
                      xaxis_title="Tempo", yaxis_title="Misura", title="Media ± IC 95% per gruppo")
    st.plotly_chart(fig, use_container_width=True)

# ──────────────────────────────────────────────────────────────────────────────
# STEP 5 — Come leggere i risultati (guide rapide)
# ──────────────────────────────────────────────────────────────────────────────
with st.expander("ℹ️ Come leggere i **risultati**"):
    st.markdown(
        "- **ANOVA RM**: l’effetto di **tempo** è significativo se p < α. In caso di **sfericità violata**, usare correzioni **GG/HF** (se disponibili) o preferire **LMM**.  \n"
        "- **Friedman**: alternativa **non parametrica**; se p < α, eseguire **post-hoc** (Wilcoxon con Holm) per capire **quali** tempi differiscono.  \n"
        "- **LMM**: robusto a **missing** e sbilanciamento; l’**LRT** tempo vs nullo indica se il pattern nel tempo è significativo. Coefficienti del modello quantificano direzione e grandezza degli effetti.  \n"
        "- **Grafici**: la **media ± IC 95%** mostra l’andamento centrale e l’incertezza; lo **spaghetti plot** evidenzia l’eterogeneità individuale; i **box/violin** mostrano la dispersione per tempo."
    )

# ──────────────────────────────────────────────────────────────────────────────
# NAVIGAZIONE
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("---")
nav1, nav2 = st.columns(2)
with nav1:
    if st.button("⬅️ Torna: Sopravvivenza", use_container_width=True, key=k("go_prev")):
        try:
            st.switch_page("pages/11_🧭_Analisi_di_Sopravvivenza.py")
        except Exception:
            pass
with nav2:
    if st.button("➡️ Vai: Modulo successivo", use_container_width=True, key=k("go_next")):
        # Prova alcune denominazioni possibili del modulo successivo
        for target in [
            "pages/13_📤_Export_Risultati.py",
            "pages/13_🧾_Report_Automatico.py",
            "pages/13_📦_Power_Analysis.py",
        ]:
            try:
                st.switch_page(target)
                break
            except Exception:
                continue
