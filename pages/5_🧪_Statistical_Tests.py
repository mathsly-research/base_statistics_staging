# pages/5_🧪_Statistical_Tests.py
from __future__ import annotations
import math
import streamlit as st
import pandas as pd
import numpy as np

# Librerie opzionali
try:
    import plotly.express as px
    import plotly.graph_objects as go
except Exception:
    px = None

try:
    from scipy import stats
except Exception:
    stats = None

try:
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    from statsmodels.stats.proportion import proportions_ztest, proportion_confint
    from statsmodels.stats.multicomp import pairwise_tukeyhsd
except Exception:
    sm = None
    smf = None
    proportions_ztest = None
    proportion_confint = None
    pairwise_tukeyhsd = None

# ──────────────────────────────────────────────────────────────────────────────
# Data store centralizzato (+ fallback)
# ──────────────────────────────────────────────────────────────────────────────
try:
    from data_store import ensure_initialized, get_active, stamp_meta
except Exception:
    def ensure_initialized():
        st.session_state.setdefault("ds_active_df", None)
        st.session_state.setdefault("ds_meta", {"version": 0, "updated_at": None, "source": None, "note": ""})
    def get_active(required: bool = True):
        ensure_initialized()
        df = st.session_state.get("ds_active_df")
        if required and (df is None or df.empty):
            st.error("Nessun dataset attivo. Importi i dati e completi la pulizia.")
            st.stop()
        return df
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
st.set_page_config(page_title="Test statistici", layout="wide")
try:
    from nav import sidebar
    sidebar()
except Exception:
    pass

KEY = "stt"  # statistical tests
def k(name: str) -> str: return f"{KEY}_{name}"

# ──────────────────────────────────────────────────────────────────────────────
# Header
# ──────────────────────────────────────────────────────────────────────────────
st.title("🧪 Test statistici")
st.caption("Esecuzione guidata dei test più comuni (medie, proporzioni, associazioni, correlazioni, ANOVA) con risultati interpretati.")

ensure_initialized()
df = get_active(required=True)

with st.expander("Stato dati", expanded=False):
    stamp_meta()

# Guida generale all'interpretazione
with st.expander("📘 Guida rapida all’interpretazione", expanded=False):
    st.markdown("""
**p-value**  
- Indica quanto sarebbero **estremi** (o più) i dati se H₀ fosse vera. **p < 0.05**: evidenza contro H₀.  
- Non è la probabilità che H₀ sia vera.

**Intervallo di confidenza (CI95%)**  
- Gamma di valori compatibili con i dati. Se la CI95% **non** include 0 (o 1 per OR/proporzioni), l’effetto è statisticamente diverso da 0/1.  
- Più è **stretta**, più la stima è precisa.

**Dimensione dell’effetto (effect size)**  
- Misura la **rilevanza** pratica:  
  - **Cohen’s d**: ≈ 0.2 piccolo · 0.5 medio · 0.8 grande  
  - **r / ρ / τ**: ≈ 0.1 piccolo · 0.3 medio · 0.5 grande  
  - **φ / V di Cramér**: ≈ 0.1 piccolo · 0.3 medio · 0.5 grande *(soglie indicative)*  
  - **η² / ω²**: quota di varianza spiegata (0→1)

**Assunzioni & robustezza**  
- T-test/ANOVA richiedono **indipendenza**, normalità (residui) e **omoscedasticità**; **Welch** è più robusto con varianze diseguali.  
- Non parametrici (MW/Wilcoxon/Kruskal) testano differenze di **distribuzione/mediane**.

**Campione e potenza**  
- p non significativo **non** prova l’assenza di effetto; guardi **effect size** e **CI**.

**Confronti multipli**  
- Con molti test aumenta il rischio di falsi positivi: consideri correzioni (es. **Holm**, **BH/FDR**).
""")

# Liste variabili
num_vars = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
cat_vars = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]

if px is None:
    st.info("Plotly non è disponibile nell'ambiente. I grafici interattivi potrebbero non comparire.")

# ──────────────────────────────────────────────────────────────────────────────
# Helper statistici
# ──────────────────────────────────────────────────────────────────────────────
def _ci_t(diff: float, se: float, dfree: int, alpha: float = 0.05):
    if stats is None or se is None or se == 0 or dfree is None or dfree < 1:
        return (np.nan, np.nan)
    tcrit = stats.t.ppf(1 - alpha/2, dfree)
    return diff - tcrit*se, diff + tcrit*se

def _welch_df(s1, n1, s2, n2):
    num = (s1/n1 + s2/n2)**2
    den = ((s1/n1)**2 / (n1-1)) + ((s2/n2)**2 / (n2-1))
    return num/den

def _cohen_d_ind(mean1, mean2, sd1, sd2, n1, n2):
    if n1 < 2 or n2 < 2: return np.nan
    sp2 = ((n1-1)*sd1**2 + (n2-1)*sd2**2) / (n1+n2-2)
    sp = math.sqrt(sp2) if sp2 > 0 else np.nan
    return (mean1 - mean2) / sp if sp and not np.isnan(sp) else np.nan

def _hedges_g(d, n1, n2):
    if np.isnan(d): return np.nan
    J = 1 - (3 / (4*(n1+n2) - 9))
    return d * J

def _cohen_d_paired(diff_mean, diff_sd):
    return diff_mean / diff_sd if diff_sd and diff_sd > 0 else np.nan

def _cliffs_delta(x, y):
    x = np.asarray(x); y = np.asarray(y)
    n_x, n_y = len(x), len(y)
    if n_x == 0 or n_y == 0: return np.nan
    total = 0
    for xi in x:
        total += np.sum(xi > y) - np.sum(xi < y)
    return total / (n_x * n_y)

def _ranks_biserial_from_u(u, n1, n2):
    return 1 - (2 * u) / (n1 * n2)

def _pearson_ci(r, n, alpha=0.05):
    if n < 4 or r is None or np.isnan(r):
        return (np.nan, np.nan)
    z = 0.5 * np.log((1+r)/(1-r))
    se = 1 / math.sqrt(n - 3)
    zcrit = stats.norm.ppf(1 - alpha/2) if stats else 1.96
    lo = z - zcrit*se
    hi = z + zcrit*se
    rlo = (np.exp(2*lo) - 1) / (np.exp(2*lo) + 1)
    rhi = (np.exp(2*hi) - 1) / (np.exp(2*hi) + 1)
    return rlo, rhi

def _safe_numeric(s):
    return pd.to_numeric(s, errors="coerce").dropna()

# Quoting sicuro per formule Patsy (nomi con spazi/simboli/emoji)
def fq(s: str) -> str:
    # escape di backslash e apici singoli
    return s.replace("\\", "\\\\").replace("'", "\\'")

# Legende centralizzate
def how_to_read(topic: str):
    if topic == "t_one":
        st.markdown("""
**Scopo**: testare se la media differisce da un valore atteso (μ₀).  
**Assunzioni**: indipendenza; normalità (oppure n grande → CLT).  
**Cosa guardare**: **p-value**, **CI95% della differenza** (media−μ₀), **Cohen’s d**.  
**Nota**: outlier/asimmetrie possono influenzare; valutare trasformazioni o bootstrap.
""")
    elif topic == "t_ind":
        st.markdown("""
**Scopo**: confronto tra **due medie indipendenti**.  
**Assunzioni**: indipendenza; normalità entro gruppo; varianze uguali (Student) o disuguali (**Welch** consigliato).  
**Cosa guardare**: **p-value**, **CI95% (A−B)**, **Cohen’s d / Hedges’ g**.  
""")
    elif topic == "t_paired":
        st.markdown("""
**Scopo**: confronto **prima/dopo** sugli **stessi** soggetti.  
**Assunzioni**: normalità delle **differenze**.  
**Cosa guardare**: **p-value**, **CI95% di Δ̄**, **Cohen’s dₚ** (sui differenziali).  
""")
    elif topic == "mw":
        st.markdown("""
**Scopo**: alternativa **non parametrica** al t a 2 campioni.  
**Cosa testa**: differenze di **distribuzione/mediana** tra gruppi.  
**Cosa guardare**: **U**, **p-value**, **rank-biserial r / Cliff’s δ**.  
""")
    elif topic == "wilcoxon":
        st.markdown("""
**Scopo**: alternativa **non parametrica** al t appaiato.  
**Assunzioni**: differenze appaiate con distribuzione **simmetrica**.  
**Cosa guardare**: **W**, **p-value**, **rank-biserial r**.  
""")
    elif topic == "prop_one":
        st.markdown("""
**Scopo**: verificare se una **proporzione** differisce da **p₀**.  
**Cosa guardare**: **p̂**, **p-value z**, **CI95% (Wilson)**.  
""")
    elif topic == "prop_two":
        st.markdown("""
**Scopo**: confronto tra **due proporzioni** indipendenti.  
**Cosa guardare**: **p̂₁, p̂₂**, **p-value z**, **CI95%** per ciascuna e per **(p̂₁−p̂₂)**.  
""")
    elif topic == "chi_fisher":
        st.markdown("""
**Scopo**: testare l’**associazione** tra due categoriali.  
**Cosa guardare**: **χ²** (o **Fisher** in 2×2 con basse attese), **p-value**, **φ / V di Cramér** (intensità).  
""")
    elif topic == "corr":
        st.markdown("""
**Scopo**: forza/direzione relazione tra due **numeriche**.  
**Pearson** (lineare), **Spearman** (monotona), **Kendall** (concordanza).  
**Cosa guardare**: coefficiente, **p-value**, **CI95%** (Pearson via Fisher z).  
""")
    elif topic == "anova":
        st.markdown("""
**Scopo**: confronto tra ≥3 medie (un fattore).  
**Cosa guardare**: **p** (F-test), **η² / ω²** (varianza spiegata).  
**Alternative**: **Welch** (varianze disuguali), **Kruskal–Wallis** (non parametrico).  
""")
    elif topic == "tukey":
        st.markdown("""
**Scopo**: confronti multipli tra coppie dopo ANOVA.  
**Cosa guardare**: per ogni coppia → **diff. media**, **CI95%**, **p-adj**; `reject=True` indica differenza significativa.  
""")

# ──────────────────────────────────────────────────────────────────────────────
# Tabs principali
# ──────────────────────────────────────────────────────────────────────────────
tab_means, tab_props, tab_cat, tab_corr, tab_anova = st.tabs([
    "📏 Medie (t/Wilcoxon/MW)", "🧮 Proporzioni", "🔗 Associazione categoriali", "📉 Correlazioni", "📊 ANOVA / Kruskal"
])

# =============================================================================
# 1) TEST SU MEDIE
# =============================================================================
with tab_means:
    st.subheader("📏 Test su medie")
    if not num_vars:
        st.info("Nessuna variabile numerica rilevata.")
    else:
        test_type = st.radio("Selezioni il test", ["t-test a 1 campione", "t-test a 2 campioni (indipendenti)", "t-test appaiato"], horizontal=True, key=k("mean_type"))

        # ── 1 campione
        if test_type == "t-test a 1 campione":
            x = st.selectbox("Variabile numerica", options=num_vars, key=k("t1_x"))
            mu0 = st.number_input("Media attesa (H₀: μ = …)", value=0.0, step=0.1, key=k("t1_mu0"))
            alt = st.selectbox("Alternativa", ["two-sided", "less", "greater"], key=k("t1_alt"))

            s = _safe_numeric(df[x])
            n = len(s)
            if stats is None or n < 2:
                st.info("Campione insufficiente o SciPy non disponibile.")
            else:
                tstat, p = stats.ttest_1samp(s, popmean=mu0, alternative=alt)
                mean, sd = float(s.mean()), float(s.std(ddof=1))
                se = sd / math.sqrt(n) if n > 0 else np.nan
                diff = mean - mu0
                ci_lo, ci_hi = _ci_t(diff, se, n-1)

                d = _cohen_d_paired(diff, sd)  # 1-campione: (mean - mu0)/sd

                st.markdown(f"**n = {n}**, **media = {mean:.3f}**, **sd = {sd:.3f}**")
                st.markdown(f"**t = {tstat:.3f}**, **p = {p:.4f}**, **diff = {diff:.3f}** (CI95%: {ci_lo:.3f} … {ci_hi:.3f}), **d = {d:.3f}**")

                with st.expander("ℹ️ Come leggere"):
                    how_to_read("t_one")

                if px is not None:
                    fig = px.box(pd.DataFrame({x: s}), y=x, points="all", template="simple_white", title=f"{x}: distribuzione campionaria")
                    fig.add_hline(y=mu0, line_dash="dash", annotation_text=f"μ₀={mu0}", annotation_position="top left")
                    st.plotly_chart(fig, use_container_width=True)

        # ── 2 campioni indipendenti
        elif test_type == "t-test a 2 campioni (indipendenti)":
            y = st.selectbox("Variabile numerica", options=num_vars, key=k("t2_y"))
            g = st.selectbox("Fattore (categoriale)", options=cat_vars, key=k("t2_g"))
            levels = sorted(df[g].dropna().astype(str).unique().tolist())
            use_lvls = st.multiselect("Selezioni 2 livelli da confrontare", options=levels, default=levels[:2], key=k("t2_lvls"))
            welch = st.checkbox("Usa Welch (varianze disuguali) – consigliato", value=True, key=k("t2_welch"))
            nonpar = st.checkbox("Alternativa non parametrica (Mann–Whitney)", value=False, key=k("t2_mw"))

            if len(use_lvls) != 2:
                st.info("Selezioni esattamente due livelli.")
            else:
                a = _safe_numeric(df.loc[df[g].astype(str)==use_lvls[0], y])
                b = _safe_numeric(df.loc[df[g].astype(str)==use_lvls[1], y])
                n1, n2 = len(a), len(b)
                if n1 < 2 or n2 < 2 or stats is None:
                    st.info("Dati insufficienti o SciPy non disponibile.")
                else:
                    if nonpar:
                        res = stats.mannwhitneyu(a, b, alternative="two-sided")
                        u = res.statistic; p = res.pvalue
                        r_rb = _ranks_biserial_from_u(u, n1, n2)
                        cd = _cliffs_delta(a, b)
                        st.markdown(f"**Mann–Whitney U = {u:.1f}**, **p = {p:.4f}**, **rank-biserial r = {r_rb:.3f}**, **Cliff’s δ = {cd:.3f}**")
                        with st.expander("ℹ️ Come leggere"):
                            how_to_read("mw")
                    else:
                        res = stats.ttest_ind(a, b, equal_var=not welch, alternative="two-sided")
                        tstat, p = res.statistic, res.pvalue
                        m1, m2 = a.mean(), b.mean()
                        s1, s2 = a.std(ddof=1), b.std(ddof=1)
                        if welch:
                            se = math.sqrt(s1**2/n1 + s2**2/n2)
                            dfree = _welch_df(s1**2, n1, s2**2, n2)
                        else:
                            sp2 = ((n1-1)*s1**2 + (n2-1)*s2**2)/(n1+n2-2)
                            se = math.sqrt(sp2*(1/n1 + 1/n2))
                            dfree = n1 + n2 - 2
                        diff = m1 - m2
                        lo, hi = _ci_t(diff, se, int(round(dree if (dree:=dfree) else dfree)))
                        # ↑ piccola salvaguardia per evitare UnboundLocalError in edge case (py3.13)
                        d = _cohen_d_ind(m1, m2, s1, s2, n1, n2)
                        g_ = _hedges_g(d, n1, n2)
                        st.markdown(
                            f"**t = {tstat:.3f}**, **p = {p:.4f}**, **diff = {diff:.3f}** "
                            f"(CI95%: {lo:.3f} … {hi:.3f}); **d = {d:.3f}**, **Hedges’ g = {g_:.3f}**"
                        )
                        with st.expander("ℹ️ Come leggere"):
                            how_to_read("t_ind")

                    if px is not None:
                        plot_df = df[df[g].astype(str).isin(use_lvls)]
                        fig = px.violin(plot_df, x=g, y=y, box=True, points="all", color=g, template="simple_white",
                                        title=f"{y} per livelli di {g}")
                        st.plotly_chart(fig, use_container_width=True)

        # ── appaiato
        else:
            a = st.selectbox("Misura A (baseline)", options=num_vars, key=k("tp_a"))
            b = st.selectbox("Misura B (follow-up)", options=[c for c in num_vars if c != a], key=k("tp_b"))
            nonpar = st.checkbox("Alternativa non parametrica (Wilcoxon)", value=False, key=k("tp_wx"))

            aligned = pd.DataFrame({a: df[a], b: df[b]}).dropna()
            if aligned.shape[0] < 2 or stats is None:
                st.info("Dati insufficienti o SciPy non disponibile.")
            else:
                da = aligned[a].astype(float).values
                db = aligned[b].astype(float).values
                if nonpar:
                    res = stats.wilcoxon(da, db, alternative="two-sided", zero_method="wilcox")
                    W, p = res.statistic, res.pvalue
                    n_pairs = len(da)
                    denom = n_pairs*(n_pairs+1)/2
                    r_rb = 1 - (2*W)/denom if denom > 0 else np.nan
                    st.markdown(f"**Wilcoxon W = {W:.1f}**, **p = {p:.4f}**, **rank-biserial r = {r_rb:.3f}**")
                    with st.expander("ℹ️ Come leggere"):
                        how_to_read("wilcoxon")
                else:
                    res = stats.ttest_rel(da, db, alternative="two-sided")
                    tstat, p = res.statistic, res.pvalue
                    diff = da - db
                    dbar, sd = diff.mean(), diff.std(ddof=1)
                    se = sd / math.sqrt(len(diff))
                    lo, hi = _ci_t(dbar, se, len(diff)-1)
                    dz = _cohen_d_paired(dbar, sd)
                    st.markdown(f"**t = {tstat:.3f}**, **p = {p:.4f}**, **Δ̄ = {dbar:.3f}** (CI95%: {lo:.3f} … {hi:.3f}), **dₚ = {dz:.3f}**")
                    with st.expander("ℹ️ Come leggere"):
                        how_to_read("t_paired")

                if px is not None:
                    plot_df = aligned.melt(value_vars=[a, b], var_name="Misura", value_name="Valore")
                    fig = px.box(plot_df, x="Misura", y="Valore", points="all", template="simple_white",
                                 title=f"{a} vs {b} (appaiato)")
                    st.plotly_chart(fig, use_container_width=True)

# =============================================================================
# 2) TEST SU PROPORZIONI
# =============================================================================
with tab_props:
    st.subheader("🧮 Proporzioni")
    if not cat_vars and not num_vars:
        st.info("Servono variabili binarie o categoriali.")
    else:
        mode = st.radio("Selezioni il test", ["1 proporzione", "2 proporzioni (tra gruppi)"], horizontal=True, key=k("prop_mode"))

        def _counts_from_var(series: pd.Series, success_label: str | None):
            s = series
            if pd.api.types.is_numeric_dtype(s) and set(pd.unique(s.dropna())) <= {0, 1}:
                x = int((s == 1).sum()); n = int(s.notna().sum()); lab = "1"
            else:
                s = s.astype(str)
                if success_label is None:
                    return None, None, sorted(s.dropna().unique().tolist())
                x = int((s == success_label).sum()); n = int(s.notna().sum()); lab = success_label
            return x, n, lab

        # ── 1 proporzione
        if mode == "1 proporzione":
            y = st.selectbox("Variabile binaria/categoriale", options=cat_vars + num_vars, key=k("p1_y"))
            s = df[y]
            succ_options = None
            if not (pd.api.types.is_numeric_dtype(s) and set(pd.unique(s.dropna())) <= {0, 1}):
                succ_options = sorted(s.astype(str).dropna().unique().tolist())
                success = st.selectbox("Categoria considerata 'successo'", options=succ_options, key=k("p1_succ"))
            else:
                success = None
            p0 = st.number_input("Proporzione attesa H₀ (p₀)", min_value=0.0, max_value=1.0, value=0.5, step=0.01, key=k("p1_p0"))
            alt = st.selectbox("Alternativa", ["two-sided", "smaller", "larger"], key=k("p1_alt"))

            x, n, label = _counts_from_var(s, success)
            if x is None:
                st.info("Selezioni la modalità 'successo'.")
            else:
                phat = x/n if n > 0 else np.nan
                if proportions_ztest is not None:
                    stat, p = proportions_ztest(count=x, nobs=n, value=p0, alternative=alt)
                    if proportion_confint is not None:
                        lo, hi = proportion_confint(count=x, nobs=n, alpha=0.05, method="wilson")
                    else:
                        se = math.sqrt(phat*(1-phat)/n) if n>0 else np.nan
                        lo, hi = phat - 1.96*se, phat + 1.96*se
                    st.markdown(f"**x = {x}**, **n = {n}**, **p̂ = {phat:.3f}**, **z = {stat:.3f}**, **p = {p:.4f}**, **CI95% p̂ = ({lo:.3f}, {hi:.3f})**")
                else:
                    st.info("`statsmodels` non disponibile: uso approssimazione normale.")
                    se = math.sqrt(p0*(1-p0)/n) if n>0 else np.nan
                    z = (phat - p0)/se if se>0 else np.nan
                    st.markdown(f"**x = {x}**, **n = {n}**, **p̂ = {phat:.3f}**, **z ≈ {z:.3f}** (p non calcolata senza statsmodels)")

                if px is not None and not np.isnan(phat):
                    fig = go.Figure()
                    fig.add_trace(go.Bar(x=[str(label)], y=[phat], name="p̂"))
                    fig.add_shape(type="line", x0=-0.5, x1=0.5, y0=p0, y1=p0, line=dict(dash="dash"))
                    fig.update_layout(template="simple_white", title=f"Proporzione di '{label}'", yaxis=dict(range=[0,1]))
                    st.plotly_chart(fig, use_container_width=True)

                with st.expander("ℹ️ Come leggere"):
                    how_to_read("prop_one")

        # ── 2 proporzioni
        else:
            y = st.selectbox("Outcome (binario/categoriale)", options=cat_vars + num_vars, key=k("p2_y"))
            g = st.selectbox("Gruppo (categoriale)", options=cat_vars, key=k("p2_g"))
            s = df[y]
            if not (pd.api.types.is_numeric_dtype(s) and set(pd.unique(s.dropna())) <= {0, 1}):
                succ_options = sorted(s.astype(str).dropna().unique().tolist())
                success = st.selectbox("Categoria considerata 'successo'", options=succ_options, key=k("p2_succ"))
            else:
                success = None
            levels = sorted(df[g].dropna().astype(str).unique().tolist())
            use_lvls = st.multiselect("Selezioni 2 gruppi", options=levels, default=levels[:2], key=k("p2_lvls"))
            if len(use_lvls) != 2:
                st.info("Selezioni esattamente due gruppi.")
            else:
                dfg = df[df[g].astype(str).isin(use_lvls)].copy()
                s1 = dfg[dfg[g].astype(str)==use_lvls[0]][y]
                s2 = dfg[dfg[g].astype(str)==use_lvls[1]][y]

                def _count(series: pd.Series, succ_label: str | None):
                    if pd.api.types.is_numeric_dtype(series) and set(pd.unique(series.dropna())) <= {0, 1}:
                        return int((series == 1).sum()), int(series.notna().sum())
                    else:
                        return int((series.astype(str) == succ_label).sum()), int(series.notna().sum())

                x_a, n_a = _count(s1, success)
                x_b, n_b = _count(s2, success)

                if proportions_ztest is not None:
                    stat, p = proportions_ztest(count=[x_a, x_b], nobs=[n_a, n_b], alternative="two-sided")
                    p1, p2 = (x_a/n_a if n_a else np.nan), (x_b/n_b if n_b else np.nan)
                    if proportion_confint is not None:
                        lo1, hi1 = proportion_confint(x_a, n_a, method="wilson")
                        lo2, hi2 = proportion_confint(x_b, n_b, method="wilson")
                    else:
                        se1 = math.sqrt(p1*(1-p1)/n_a); se2 = math.sqrt(p2*(1-p2)/n_b)
                        lo1, hi1 = p1-1.96*se1, p1+1.96*se1
                        lo2, hi2 = p2-1.96*se2, p2+1.96*se2
                    diff = p1 - p2
                    se_diff = math.sqrt(p1*(1-p1)/n_a + p2*(1-p2)/n_b)
                    lo_d, hi_d = diff - 1.96*se_diff, diff + 1.96*se_diff
                    st.markdown(
                        f"**z = {stat:.3f}**, **p = {p:.4f}**  \n"
                        f"{use_lvls[0]}: **p̂ = {p1:.3f}** (CI95%: {lo1:.3f}…{hi1:.3f})  \n"
                        f"{use_lvls[1]}: **p̂ = {p2:.3f}** (CI95%: {lo2:.3f}…{hi2:.3f})  \n"
                        f"**Differenza p̂₁−p̂₂ = {diff:.3f}** (CI95%: {lo_d:.3f}…{hi_d:.3f})"
                    )
                else:
                    st.info("`statsmodels` non disponibile: impossibile eseguire 2-proporzioni.")

                if px is not None:
                    plot_df = pd.DataFrame({
                        g: [use_lvls[0], use_lvls[1]],
                        "p_hat": [x_a/n_a if n_a else np.nan, x_b/n_b if n_b else np.nan]
                    })
                    fig = px.bar(plot_df, x=g, y="p_hat", template="simple_white", title="Proporzioni per gruppo")
                    fig.update_yaxes(range=[0,1])
                    st.plotly_chart(fig, use_container_width=True)

                with st.expander("ℹ️ Come leggere"):
                    how_to_read("prop_two")

# =============================================================================
# 3) ASSOCIAZIONE TRA CATEGORIALI
# =============================================================================
with tab_cat:
    st.subheader("🔗 Associazione tra variabili categoriali")
    if len(cat_vars) < 2:
        st.info("Servono almeno due variabili categoriali.")
    else:
        a = st.selectbox("Variabile A (righe)", options=cat_vars, key=k("cat_a"))
        b = st.selectbox("Variabile B (colonne)", options=[c for c in cat_vars if c != a], key=k("cat_b"))

        ct = pd.crosstab(df[a].astype(str), df[b].astype(str), dropna=False)
        st.markdown("**Tabella di contingenza**")
        st.dataframe(ct, use_container_width=True)

        if stats is None:
            st.info("SciPy non disponibile: impossibile calcolare chi-quadrato/Fisher.")
        else:
            chi2, p, dof, expected = stats.chi2_contingency(ct)
            n = ct.to_numpy().sum()
            r, c = ct.shape
            if r == 2 and c == 2:
                phi = math.sqrt(chi2 / n) if n>0 else np.nan
                st.markdown(f"**χ² = {chi2:.3f}** (df={dof}), **p = {p:.4f}**, **φ = {phi:.3f}**")
                try:
                    OR, p_f = stats.fisher_exact(ct.values)
                    st.markdown(f"**Fisher exact**: OR = {OR:.3f}, p = {p_f:.4f}")
                except Exception:
                    pass
            else:
                k_min = min(r-1, c-1)  # non sovrascrive k(...)
                cramer_v = math.sqrt(chi2 / (n * k_min)) if n>0 and k_min>0 else np.nan
                st.markdown(f"**χ² = {chi2:.3f}** (df={dof}), **p = {p:.4f}**, **V di Cramér = {cramer_v:.3f}**")

        if px is not None:
            heat = px.imshow(ct, text_auto=True, aspect="auto", color_continuous_scale="Blues",
                             title=f"Heatmap contingenza: {a} × {b}", template="simple_white")
            st.plotly_chart(heat, use_container_width=True)

        with st.expander("ℹ️ Come leggere"):
            how_to_read("chi_fisher")

# =============================================================================
# 4) CORRELAZIONI
# =============================================================================
with tab_corr:
    st.subheader("📉 Correlazioni")
    if len(num_vars) < 2 or stats is None:
        st.info("Servono almeno due variabili numeriche e SciPy.")
    else:
        x = st.selectbox("Variabile X", options=num_vars, key=k("corr_x"))
        y = st.selectbox("Variabile Y", options=[c for c in num_vars if c != x], key=k("corr_y"))
        method = st.selectbox("Metodo", ["Pearson", "Spearman", "Kendall"], key=k("corr_meth"))

        xs = _safe_numeric(df[x]); ys = _safe_numeric(df[y])
        n = min(len(xs), len(ys))
        if n < 3:
            st.info("Dati insufficienti per la correlazione.")
        else:
            if method == "Pearson":
                r, p = stats.pearsonr(xs, ys)
                # CI via Fisher z
                z = 0.5 * np.log((1+r)/(1-r))
                se = 1 / math.sqrt(n - 3)
                zcrit = stats.norm.ppf(1 - 0.05/2) if stats else 1.96
                lo = z - zcrit*se; hi = z + zcrit*se
                rlo = (np.exp(2*lo)-1)/(np.exp(2*lo)+1); rhi = (np.exp(2*hi)-1)/(np.exp(2*hi)+1)
                st.markdown(f"**r = {r:.3f}**, **p = {p:.4f}**, **CI95% r = ({rlo:.3f}, {rhi:.3f})**, **n = {n}**")
            elif method == "Spearman":
                r, p = stats.spearmanr(xs, ys)
                st.markdown(f"**ρ = {r:.3f}**, **p = {p:.4f}**, **n = {n}**")
            else:
                r, p = stats.kendalltau(xs, ys)
                st.markdown(f"**τ = {r:.3f}**, **p = {p:.4f}**, **n = {n}**")

            if px is not None:
                fig = px.scatter(df, x=x, y=y, trendline=("ols" if method=="Pearson" and sm is not None else None),
                                 template="simple_white", title=f"{y} vs {x}")
                st.plotly_chart(fig, use_container_width=True)

            with st.expander("ℹ️ Come leggere"):
                how_to_read("corr")

# =============================================================================
# 5) ANOVA / KRUSKAL
# =============================================================================
with tab_anova:
    st.subheader("📊 ANOVA a 1 via / Kruskal–Wallis")
    if not num_vars or not cat_vars:
        st.info("Servono una variabile numerica e una categoriale.")
    else:
        y = st.selectbox("Variabile risposta (numerica)", options=num_vars, key=k("a_y"))
        g = st.selectbox("Fattore (gruppi)", options=cat_vars, key=k("a_g"))
        method = st.selectbox("Metodo", ["ANOVA (var. uguali)", "Welch ANOVA (var. diverse)", "Kruskal–Wallis"], key=k("a_method"))

        # Gruppi (per Kruskal)
        groups = [_safe_numeric(sub[y]) for _, sub in df.groupby(g)]
        labels = [str(lv) for lv, _ in df.groupby(g)]

        if method == "Kruskal–Wallis":
            if stats is None or any(len(gr) < 2 for gr in groups):
                st.info("Dati insufficienti o SciPy non disponibile.")
            else:
                H, p = stats.kruskal(*groups)
                n_tot = sum(len(gr) for gr in groups)
                eps2 = (H - len(groups) + 1) / (n_tot - len(groups)) if n_tot > len(groups) else np.nan
                st.markdown(f"**H = {H:.3f}**, **p = {p:.4f}**, **ε² ≈ {eps2:.3f}**")
                with st.expander("ℹ️ Come leggere"):
                    how_to_read("anova")
        else:
            if smf is None:
                st.info("`statsmodels` non disponibile: per ANOVA si consiglia StatsModels.")
            else:
                # >>> FIX: formula con quote sicuro per nomi non standard
                formula = f"Q('{fq(y)}') ~ C(Q('{fq(g)}'))"
                model = smf.ols(formula=formula, data=df).fit()
                anova = sm.stats.anova_lm(model, typ=2)  # Type II
                st.dataframe(anova, use_container_width=True)

                try:
                    ss_between = float(anova.loc[f"C(Q('{fq(g)}'))", "sum_sq"])
                    ss_resid  = float(anova.loc["Residual", "sum_sq"])
                    df_between = int(anova.loc[f"C(Q('{fq(g)}'))", "df"])
                    df_resid   = int(anova.loc["Residual", "df"])
                    sst = ss_between + ss_resid
                    eta2 = ss_between / sst if sst>0 else np.nan
                    msw = ss_resid/df_resid if df_resid>0 else np.nan
                    omega2 = ((ss_between - df_between*msw) / (sst + msw)) if (sst>0 and not np.isnan(msw)) else np.nan
                    st.markdown(f"**η² = {eta2:.3f}**, **ω² = {omega2:.3f}**  \n_ANOVA Type II mostrata come riferimento._")
                except Exception:
                    pass

                with st.expander("ℹ️ Come leggere"):
                    how_to_read("anova")

        if px is not None:
            fig = px.box(df, x=g, y=y, points="outliers", template="simple_white", title=f"{y} per {g}")
            st.plotly_chart(fig, use_container_width=True)

        # Post-hoc (Tukey) se disponibile e pertinente (ANOVA classica)
        if pairwise_tukeyhsd is not None and method != "Kruskal–Wallis":
            st.markdown("**Confronti multipli (Tukey HSD)**")
            try:
                res = pairwise_tukeyhsd(endog=df[y].astype(float), groups=df[g].astype(str), alpha=0.05)
                st.text(res.summary().as_text())
                with st.expander("ℹ️ Come leggere Tukey HSD"):
                    how_to_read("tukey")
            except Exception as e:
                st.caption(f"Tukey non disponibile: {e}")
        else:
            st.caption("Per confronti post-hoc (Tukey) è consigliato `statsmodels`.")

# ──────────────────────────────────────────────────────────────────────────────
# Navigazione
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("---")
nav1, nav2 = st.columns(2)
with nav1:
    if st.button("⬅️ Torna: Assumption Checks", use_container_width=True, key=k("go_prev")):
        st.switch_page("pages/4_🔎_Assumption_Checks.py")
with nav2:
    if st.button("➡️ Vai: Correlation Analysis", use_container_width=True, key=k("go_next")):
        # Adegui questo percorso al nome reale del file successivo nel suo progetto
        st.switch_page("pages/6__Correlation_Analysis.py")
