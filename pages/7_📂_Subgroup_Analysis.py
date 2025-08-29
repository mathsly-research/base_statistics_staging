# pages/7_📂_Subgroup_Analysis.py
from __future__ import annotations
import math
import streamlit as st
import pandas as pd
import numpy as np

# Plotting (opzionale)
try:
    import plotly.express as px
    import plotly.graph_objects as go
except Exception:
    px = None
    go = None

# Statistiche (opzionale)
try:
    from scipy import stats
except Exception:
    stats = None

try:
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
except Exception:
    sm = None
    smf = None

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
st.set_page_config(page_title="Subgroup Analysis", layout="wide")
try:
    from nav import sidebar
    sidebar()
except Exception:
    pass

KEY = "sga"
def k(name: str) -> str: return f"{KEY}_{name}"

# ──────────────────────────────────────────────────────────────────────────────
# Header
# ──────────────────────────────────────────────────────────────────────────────
st.title("📂 Subgroup Analysis")
st.caption("Confronti per sottogruppi e verifica di eterogeneità (interazioni), con forest plot e guida alla lettura.")

ensure_initialized()
df = get_active(required=True)

with st.expander("Stato dati", expanded=False):
    stamp_meta()

# Guida generale
with st.expander("📘 Guida rapida all’interpretazione", expanded=False):
    st.markdown("""
**Perché analizzare i sottogruppi?**  
- Per verificare se l’**effetto** di un’esposizione/trattamento **cambia** tra livelli di una variabile *stratificatrice* (es. sesso, fascia d’età).

**Cosa guardare**  
- **Effetto per sottogruppo** (differenza di medie, differenza di rischio, OR, r, …) con **CI95%** (forest plot).  
- **Test di interazione**: verifica **formale** di eterogeneità tra sottogruppi.  
- **P-value** e **dimensione dell’effetto** (rilevanza pratica oltre alla significatività).

**Attenzioni**  
- Molti sottogruppi ⇒ rischio di **falsi positivi** (valuti correzioni o p interpretati con cautela).  
- Campioni piccoli ⇒ stime **instabili** (CI larghe).  
- Rischio **paradosso di Simpson**: controlli sempre la coerenza con l’analisi complessiva.
""")

# Variabili disponibili
num_vars = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
cat_vars = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]

if px is None:
    st.info("Plotly non è disponibile nell'ambiente. Le visualizzazioni interattive potrebbero non comparire.")

# ──────────────────────────────────────────────────────────────────────────────
# Helper
# ──────────────────────────────────────────────────────────────────────────────
def fq(s: str) -> str:
    """Escape sicuro per formule Patsy (nomi con spazi/simboli/emoji)."""
    return s.replace("\\", "\\\\").replace("'", "\\'")

def _safe_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

def _mean_diff_ci(a: np.ndarray, b: np.ndarray):
    """Differenza di medie (A−B) con CI95% (Welch)."""
    a = _safe_num(pd.Series(a)).dropna().values
    b = _safe_num(pd.Series(b)).dropna().values
    if a.size < 2 or b.size < 2 or stats is None:
        return np.nan, np.nan, np.nan, np.nan, np.nan
    m1, m2 = a.mean(), b.mean()
    s1, s2 = a.std(ddof=1), b.std(ddof=1)
    n1, n2 = a.size, b.size
    diff = m1 - m2
    se = math.sqrt((s1**2)/n1 + (s2**2)/n2)
    # Welch df
    num = ((s1**2)/n1 + (s2**2)/n2)**2
    den = ((s1**2/n1)**2)/(n1-1) + ((s2**2/n2)**2)/(n2-1)
    dfree = num/den if den>0 else (n1+n2-2)
    tcrit = stats.t.ppf(0.975, dfree) if stats else 1.96
    lo, hi = diff - tcrit*se, diff + tcrit*se
    return diff, lo, hi, n1, n2

def _risk_diff_ci(x1, n1, x2, n2):
    """Differenza di rischio (p1−p2) con CI95% (Wald)."""
    p1 = x1 / n1 if n1>0 else np.nan
    p2 = x2 / n2 if n2>0 else np.nan
    diff = p1 - p2
    se = math.sqrt(p1*(1-p1)/n1 + p2*(1-p2)/n2) if (n1>0 and n2>0) else np.nan
    lo, hi = (diff - 1.96*se, diff + 1.96*se) if (se==se and se>0) else (np.nan, np.nan)
    return diff, lo, hi, p1, p2

def _or_ci(x1, n1, x2, n2):
    """Odds ratio con CI95% (log-OR, correzione 0.5 se celle 0)."""
    a = x1; b = n1 - x1; c = x2; d = n2 - x2
    if min(a,b,c,d) == 0:
        a += 0.5; b += 0.5; c += 0.5; d += 0.5
    or_ = (a*d) / (b*c) if (b*c)>0 else np.nan
    se = math.sqrt(1/a + 1/b + 1/c + 1/d) if min(a,b,c,d)>0 else np.nan
    llog = math.log(or_) if or_>0 else np.nan
    lo = math.exp(llog - 1.96*se) if (llog==llog and se==se) else np.nan
    hi = math.exp(llog + 1.96*se) if (llog==llog and se==se) else np.nan
    return or_, lo, hi

def _two_level_options(series: pd.Series):
    lvls = sorted(series.dropna().astype(str).unique().tolist())
    return lvls, (lvls[:2] if len(lvls) >= 2 else lvls)

def _forest_plot(df_effect: pd.DataFrame, effect_col: str, lo_col: str, hi_col: str,
                 label_col: str, title: str, xaxis_title: str, ref=0.0):
    if px is None and go is None:
        return None
    dfp = df_effect.copy()
    dfp = dfp.replace([np.inf, -np.inf], np.nan).dropna(subset=[effect_col, lo_col, hi_col, label_col])
    dfp = dfp.sort_values(effect_col)
    y = list(range(len(dfp)))
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dfp[effect_col], y=y, mode="markers",
        marker=dict(symbol="square", size=10),
        text=dfp[label_col], hovertemplate="%{text}<br>stima=%{x:.3f}<extra></extra>"
    ))
    fig.add_traces([
        go.Scatter(
            x=dfp[lo_col], y=y, mode="lines", showlegend=False,
            hoverinfo="skip"
        ),
        go.Scatter(
            x=dfp[hi_col], y=y, mode="lines", showlegend=False,
            hoverinfo="skip"
        )
    ])
    # Segmenti orizzontali (CI)
    for yy, lo, hi in zip(y, dfp[lo_col], dfp[hi_col]):
        fig.add_shape(type="line", x0=lo, x1=hi, y0=yy, y1=yy)
    # Linea di riferimento
    fig.add_shape(type="line", x0=ref, x1=ref, y0=-1, y1=len(y), line=dict(dash="dash"))
    fig.update_yaxes(tickvals=y, ticktext=dfp[label_col], autorange="reversed")
    fig.update_layout(template="simple_white", title=title, xaxis_title=xaxis_title, height=max(300, 60*len(y)))
    return fig

def _fishers_z(r, n):
    """Trasforma r→z Fisher; var = 1/(n-3)."""
    if n <= 3 or r is None or np.isnan(r) or abs(r) >= 1:
        return np.nan, np.nan
    z = 0.5 * np.log((1+r)/(1-r))
    var = 1 / (n - 3)
    return z, var

# ──────────────────────────────────────────────────────────────────────────────
# Tabs principali
# ──────────────────────────────────────────────────────────────────────────────
tab_comp, tab_summary, tab_corr = st.tabs([
    "🆚 Confronto tra gruppi (per sottogruppo)",
    "📊 Statistiche per sottogruppo",
    "🔗 Correlazioni per sottogruppo"
])

# =============================================================================
# TAB 1 · CONFRONTO TRA GRUPPI (FOREST + INTERAZIONE)
# =============================================================================
with tab_comp:
    st.subheader("🆚 Confronto tra gruppi all’interno dei sottogruppi")
    st.markdown("Selezioni **outcome**, **variabile di gruppo** (2 livelli) e **stratificatore** (sottogruppi).")

    c1, c2 = st.columns([2,2])
    with c1:
        outcome = st.selectbox("Outcome", options=df.columns, key=k("cmp_out"))
        is_numeric_outcome = pd.api.types.is_numeric_dtype(df[outcome])
    with c2:
        strat = st.selectbox("Stratificatore (sottogruppi)", options=cat_vars if cat_vars else df.columns, key=k("cmp_strat"))

    # Selezione gruppo a 2 livelli
    grp_candidates = cat_vars if cat_vars else df.columns
    group_var = st.selectbox("Variabile di gruppo (2 livelli)", options=[c for c in grp_candidates if c != strat], key=k("cmp_group"))
    lvls, default_lvls = _two_level_options(df[group_var].astype(str))
    sel_lvls = st.multiselect("Selezioni 2 livelli del gruppo", options=lvls, default=default_lvls, key=k("cmp_lvls"))
    if len(sel_lvls) != 2:
        st.info("Selezioni **esattamente due** livelli del gruppo.")
        st.stop()

    dfw = df[[outcome, strat, group_var]].copy()
    dfw[group_var] = dfw[group_var].astype(str)
    dfw = dfw[dfw[group_var].isin(sel_lvls)]
    dfw[strat] = dfw[strat].astype(str)

    if is_numeric_outcome:
        st.markdown("**Esito**: variabile numerica → effetto = **differenza di medie (A−B)** per sottogruppo.")
        rows = []
        for s, sub in dfw.groupby(strat):
            a = sub.loc[sub[group_var]==sel_lvls[0], outcome]
            b = sub.loc[sub[group_var]==sel_lvls[1], outcome]
            diff, lo, hi, n1, n2 = _mean_diff_ci(a, b)
            rows.append({"Sottogruppo": s, "Stima": diff, "CI_lo": lo, "CI_hi": hi, "n_A": n1, "n_B": n2})
        eff_df = pd.DataFrame(rows)

        col_l, col_r = st.columns([3,2])
        with col_l:
            st.dataframe(eff_df, use_container_width=True)
        with col_r:
            if go is not None and eff_df.dropna(subset=["Stima","CI_lo","CI_hi"]).shape[0] > 0:
                fig = _forest_plot(eff_df, "Stima", "CI_lo", "CI_hi", "Sottogruppo",
                                   title="Forest plot: differenze di medie (A−B)",
                                   xaxis_title=f"Differenza {sel_lvls[0]}−{sel_lvls[1]}", ref=0.0)
                st.plotly_chart(fig, use_container_width=True)

        # Test di interazione (OLS con termine di interazione)
        if smf is not None:
            try:
                form = f"Q('{fq(outcome)}') ~ C(Q('{fq(group_var)}')) * C(Q('{fq(strat)}'))"
                model_int = smf.ols(formula=form, data=dfw).fit()
                an = sm.stats.anova_lm(model_int, typ=2)
                # Riga dell'interazione: C(group):C(strat)
                # Cerco una riga che contenga entrambe le stringhe
                inter_rows = [idx for idx in an.index if "C(" in idx and ":" in idx and fq(group_var) in idx and fq(strat) in idx]
                p_inter = float(an.loc[inter_rows[0], "PR(>F)"]) if inter_rows else np.nan
                st.markdown(f"**Test di interazione (OLS)**: p = {p_inter:.4f}" if p_inter==p_inter else "Test di interazione non disponibile.")
            except Exception as e:
                st.caption(f"Interazione non calcolabile: {e}")
        else:
            st.caption("`statsmodels` non disponibile: test di interazione non eseguibile.")

        with st.expander("ℹ️ Come leggere"):
            st.markdown(f"""
- Ogni riga mostra l’**effetto** (differenza di medie **{sel_lvls[0]}−{sel_lvls[1]}**) **nel sottogruppo** con **CI95%**.  
- La **linea verticale a 0** indica “nessun effetto”.  
- Il **test di interazione** verifica se l’effetto **cambia** tra sottogruppi (p<0.05 → eterogeneità credibile).  
""")

    else:
        st.markdown("**Esito**: variabile binaria/categoriale → effetto = **differenza di rischio** (e **OR**) per sottogruppo.")
        # Mappo outcome a 0/1: se già numerica 0/1 ok, altrimenti chiedo livello del "successo"
        if pd.api.types.is_numeric_dtype(df[outcome]) and set(pd.unique(df[outcome].dropna())) <= {0,1}:
            success_label = 1
            dfw["_y"] = dfw[outcome].astype(int)
        else:
            levels_y = sorted(df[outcome].dropna().astype(str).unique().tolist())
            success_label = st.selectbox("Categoria considerata 'successo' dell'outcome", options=levels_y, key=k("succ_y"))
            dfw["_y"] = (dfw[outcome].astype(str) == success_label).astype(int)

        rows = []
        for s, sub in dfw.groupby(strat):
            A = sub[sub[group_var]==sel_lvls[0]]["_y"].astype(int)
            B = sub[sub[group_var]==sel_lvls[1]]["_y"].astype(int)
            x1, n1 = int(A.sum()), int(A.count())
            x2, n2 = int(B.sum()), int(B.count())
            diff, lo, hi, p1, p2 = _risk_diff_ci(x1, n1, x2, n2)
            or_, or_lo, or_hi = _or_ci(x1, n1, x2, n2)
            rows.append({"Sottogruppo": s, "Diff_rischio": diff, "CI_lo": lo, "CI_hi": hi,
                         "p1": p1, "p2": p2, "OR": or_, "OR_lo": or_lo, "OR_hi": or_hi,
                         "n_A": n1, "n_B": n2})
        eff_df = pd.DataFrame(rows)

        col_l, col_r = st.columns([3,2])
        with col_l:
            st.dataframe(eff_df, use_container_width=True)
        with col_r:
            if go is not None and eff_df.dropna(subset=["Diff_rischio","CI_lo","CI_hi"]).shape[0] > 0:
                fig = _forest_plot(eff_df, "Diff_rischio", "CI_lo", "CI_hi", "Sottogruppo",
                                   title=f"Forest plot: differenza di rischio ({success_label})",
                                   xaxis_title=f"Δ p̂ ({sel_lvls[0]}−{sel_lvls[1]})", ref=0.0)
                st.plotly_chart(fig, use_container_width=True)

        # Test di interazione (logistica con LRT)
        if smf is not None and sm is not None:
            try:
                form_red = f"_y ~ C(Q('{fq(group_var)}')) + C(Q('{fq(strat)}'))"
                form_full = f"_y ~ C(Q('{fq(group_var)}')) * C(Q('{fq(strat)}'))"
                fit_red = smf.glm(formula=form_red, data=dfw, family=sm.families.Binomial()).fit()
                fit_full = smf.glm(formula=form_full, data=dfw, family=sm.families.Binomial()).fit()
                LR = 2 * (fit_full.llf - fit_red.llf)
                df_diff = fit_full.df_model - fit_red.df_model
                p_inter = stats.chi2.sf(LR, df_diff) if stats else np.nan
                st.markdown(f"**Test di interazione (Logistica, LRT)**: χ² = {LR:.3f} (df={int(df_diff)}), p = {p_inter:.4f}")
            except Exception as e:
                st.caption(f"Interazione (logistica) non calcolabile: {e}")
        else:
            st.caption("`statsmodels` non disponibile: test di interazione logistica non eseguibile.")

        with st.expander("ℹ️ Come leggere"):
            st.markdown(f"""
- Per ogni sottogruppo: **differenza di rischio** (Δ p̂ = {sel_lvls[0]}−{sel_lvls[1]}) con **CI95%**; opzionalmente **OR** con CI.  
- La **linea a 0** indica nessuna differenza di rischio (la linea a 1 varrebbe per OR).  
- Il **test di interazione** (LRT) verifica variazioni dell’effetto tra sottogruppi.
""")

# =============================================================================
# TAB 2 · STATISTICHE DESCRITTIVE PER SOTTOGRUPPO
# =============================================================================
with tab_summary:
    st.subheader("📊 Statistiche per sottogruppo")
    st.markdown("Riepiloghi e grafici facili da leggere per variabili numeriche o categoriali, stratificati.")

    vtype = st.radio("Tipo di variabile da riassumere", ["Numerica", "Categoriale"], horizontal=True, key=k("sum_type"))
    strat2 = st.selectbox("Stratificatore (sottogruppi)", options=cat_vars if cat_vars else df.columns, key=k("sum_strat"))

    if vtype == "Numerica":
        if not num_vars:
            st.info("Nessuna variabile numerica rilevata.")
        else:
            v = st.selectbox("Variabile numerica", options=num_vars, key=k("sum_num"))
            summ = df.groupby(strat2)[v].agg(n="count", media="mean", sd="std", q25=lambda s: s.quantile(0.25),
                                             mediana="median", q75=lambda s: s.quantile(0.75)).reset_index().rename(columns={strat2: "Sottogruppo"})
            st.dataframe(summ, use_container_width=True)
            if px is not None:
                c1, c2 = st.columns(2)
                with c1:
                    fig1 = px.box(df, x=strat2, y=v, points="outliers", template="simple_white",
                                  title=f"{v} per {strat2}")
                    st.plotly_chart(fig1, use_container_width=True)
                with c2:
                    fig2 = px.violin(df, x=strat2, y=v, box=True, points=False, template="simple_white",
                                     title=f"Distribuzione di {v} per {strat2}")
                    st.plotly_chart(fig2, use_container_width=True)
            with st.expander("ℹ️ Come leggere"):
                st.markdown("""
- Confronti di **mediana/IQR** e **media/SD** tra sottogruppi; le **code** o **outlier** sono evidenziati dal box-plot.
""")

    else:
        if not cat_vars:
            st.info("Nessuna variabile categoriale rilevata.")
        else:
            v = st.selectbox("Variabile categoriale", options=cat_vars, key=k("sum_cat"))
            ct = pd.crosstab(df[strat2].astype(str), df[v].astype(str), normalize="index")*100
            st.dataframe(ct.round(1), use_container_width=True)
            if px is not None:
                fig = px.bar(ct.reset_index().melt(id_vars=strat2, var_name=v, value_name="Perc"),
                             x=strat2, y="Perc", color=v, barmode="stack", template="simple_white",
                             title=f"Distribuzione percentuale di {v} per {strat2}")
                st.plotly_chart(fig, use_container_width=True)
            with st.expander("ℹ️ Come leggere"):
                st.markdown("""
- Ogni barra mostra la **composizione percentuale** della variabile categoriale **entro** il sottogruppo.
""")

# =============================================================================
# TAB 3 · CORRELAZIONI PER SOTTOGRUPPO (+ TEST DI ETEROGENEITÀ)
# =============================================================================
with tab_corr:
    st.subheader("🔗 Correlazioni per sottogruppo")
    if len(num_vars) < 2:
        st.info("Servono almeno due variabili numeriche.")
    else:
        c1, c2, c3 = st.columns([2,2,2])
        with c1:
            x = st.selectbox("Variabile X (numerica)", options=num_vars, key=k("cx"))
        with c2:
            y = st.selectbox("Variabile Y (numerica)", options=[c for c in num_vars if c != x], key=k("cy"))
        with c3:
            strat3 = st.selectbox("Stratificatore", options=cat_vars if cat_vars else df.columns, key=k("cstrat"))
        method = st.selectbox("Metodo", ["Pearson", "Spearman", "Kendall"], key=k("cmeth"))

        rows = []
        for s, sub in df.groupby(strat3):
            xs = pd.to_numeric(sub[x], errors="coerce")
            ys = pd.to_numeric(sub[y], errors="coerce")
            tmp = pd.DataFrame({"x": xs, "y": ys}).dropna()
            n = tmp.shape[0]
            if stats is None or n < 3:
                r = p = np.nan
                lo = hi = np.nan
            else:
                if method == "Pearson":
                    r, p = stats.pearsonr(tmp["x"], tmp["y"])
                    if n > 3:
                        z = 0.5*np.log((1+r)/(1-r))
                        se = 1/math.sqrt(n-3)
                        zcrit = stats.norm.ppf(1-0.05/2)
                        loz, hiz = z - zcrit*se, z + zcrit*se
                        lo = (np.exp(2*loz)-1)/(np.exp(2*loz)+1)
                        hi = (np.exp(2*hiz)-1)/(np.exp(2*hiz)+1)
                    else:
                        lo = hi = np.nan
                elif method == "Spearman":
                    r, p = stats.spearmanr(tmp["x"], tmp["y"])
                    lo = hi = np.nan
                else:
                    r, p = stats.kendalltau(tmp["x"], tmp["y"])
                    lo = hi = np.nan
            rows.append({"Sottogruppo": str(s), "r": r, "p": p, "n": n, "CI_lo": lo, "CI_hi": hi})
        out = pd.DataFrame(rows)
        st.dataframe(out, use_container_width=True)

        # Test di eterogeneità delle correlazioni (solo Pearson via Fisher z)
        if method == "Pearson" and stats is not None:
            zs, vars_ = [], []
            for _, row in out.iterrows():
                z, var = _fishers_z(row["r"], int(row["n"]))
                if not np.isnan(z) and not np.isnan(var):
                    zs.append(z); vars_.append(var)
            if len(zs) >= 2:
                zs = np.array(zs); vars_ = np.array(vars_)
                w = 1/vars_
                z_bar = np.sum(w*zs)/np.sum(w)
                Q = np.sum(w*(zs - z_bar)**2)
                df_q = len(zs) - 1
                phet = stats.chi2.sf(Q, df_q)
                st.markdown(f"**Eterogeneità (Fisher z)**: Q = {Q:.3f} (df={df_q}), p = {phet:.4f}")
            else:
                st.caption("Eterogeneità non valutabile (sottogruppi insufficienti o n troppo piccolo).")
        else:
            st.caption("Eterogeneità formale calcolata solo per Pearson (via Fisher z).")

        # Grafici
        if px is not None:
            fig = px.scatter(df, x=x, y=y, color=strat3, template="simple_white",
                             title=f"{y} vs {x} per {strat3}")
            st.plotly_chart(fig, use_container_width=True)
        with st.expander("ℹ️ Come leggere"):
            st.markdown("""
- **Correlazione per sottogruppo**: confronto della forza/direzione tra livelli del fattore.  
- **Eterogeneità** (solo Pearson): p<0.05 → differenze credibili tra sottogruppi.  
- Ispezioni sempre i grafici: outlier o non linearità possono guidare apparenti differenze.
""")

# ──────────────────────────────────────────────────────────────────────────────
# Navigazione
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("---")
nav1, nav2 = st.columns(2)
with nav1:
    if st.button("⬅️ Torna: Correlation Analysis", use_container_width=True, key=k("go_prev")):
        st.switch_page("pages/6_🔗_Correlation_Analysis.py")
with nav2:
    if st.button("➡️ Vai: Regression", use_container_width=True, key=k("go_next")):
        # Adegui questo percorso al nome reale del file successivo nel suo progetto
        st.switch_page("pages/8_📐_Regression.py")
