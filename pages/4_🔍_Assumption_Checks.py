# pages/4_🔎_Assumption_Checks.py
from __future__ import annotations
import streamlit as st
import pandas as pd
import numpy as np

# Librerie opzionali per test e modelli
try:
    import plotly.express as px
    import plotly.graph_objects as go
except Exception:
    px = None

try:
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    from statsmodels.stats.diagnostic import het_breuschpagan
except Exception:
    sm = None
    smf = None
    het_breuschpagan = None

try:
    from scipy import stats
except Exception:
    stats = None

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
st.set_page_config(page_title="Assumption Checks", layout="wide")
try:
    from nav import sidebar
    sidebar()
except Exception:
    pass

KEY = "ac"  # assumption checks
def k(name: str) -> str: return f"{KEY}_{name}"

# ──────────────────────────────────────────────────────────────────────────────
# Header
# ──────────────────────────────────────────────────────────────────────────────
st.title("🔎 Assumption Checks")
st.caption("Verifichi in modo guidato le **assunzioni** principali (normalità, omoscedasticità, linearità, indipendenza).")

ensure_initialized()
df = get_active(required=True)

with st.expander("Stato dati", expanded=False):
    stamp_meta()

# Liste variabili
num_vars = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
cat_vars = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]

if px is None:
    st.info("Plotly non è disponibile nell'ambiente. Aggiorni l'ambiente per visualizzazioni interattive.")
    st.stop()

# ──────────────────────────────────────────────────────────────────────────────
# Tabs principali
# ──────────────────────────────────────────────────────────────────────────────
tab_norm, tab_homo, tab_reg = st.tabs([
    "📦 Normalità (univariata)", "⚖️ Omoscedasticità tra gruppi", "📈 Assunzioni regressione"
])

# =============================================================================
# TAB 1 · NORMALITÀ UNIVARIATA
# =============================================================================
with tab_norm:
    st.subheader("📦 Normalità (univariata)")
    if not num_vars:
        st.info("Nessuna variabile numerica rilevata.")
    else:
        top, _ = st.columns([3, 2])
        with top:
            x_var = st.selectbox("Variabile numerica", options=num_vars, key=k("norm_x"))
        add_group = st.checkbox("Valuta per sottogruppi (facoltativo)", value=False, key=k("norm_grp_on"))
        if add_group and cat_vars:
            g_var = st.selectbox("Sottogruppo (categoriale)", options=cat_vars, key=k("norm_g"))
        else:
            g_var = None

        # Grafici: Istogramma + QQ-plot affiancati
        c1, c2 = st.columns(2, vertical_alignment="top")

        with c1:
            nbins = st.slider("Bin istogramma", 10, 100, 30, 5, key=k("norm_bins"))
            fig_h = px.histogram(df, x=x_var, color=g_var if add_group else None,
                                 nbins=nbins, template="simple_white", barmode="overlay")
            fig_h.update_layout(title=f"Istogramma di {x_var}", height=420, font=dict(size=14))
            st.plotly_chart(fig_h, use_container_width=True)

        with c2:
            # QQ-plot manuale
            def qq_data(series: pd.Series) -> pd.DataFrame:
                s = pd.to_numeric(series, errors="coerce").dropna().sort_values()
                n = len(s)
                if n < 3:
                    return pd.DataFrame(columns=["theoretical", "sample"])
                # quantili teorici da N(0,1), poi ridimensionamento usando media/sd campionaria
                theo = stats.norm.ppf((np.arange(1, n+1) - 0.5) / n) if stats else np.linspace(-3, 3, n)
                # Standardizzazione e rescaling
                m, sd = s.mean(), s.std(ddof=1) if s.std(ddof=1) > 0 else 1.0
                sample_std = (s - m) / sd
                # reintroduco la scala campionaria per mostrare la linea y=x corretta
                sample = s.values
                # per linea teorica, mappo i quantili teorici a scala campionaria
                theoretical = theo * sd + m
                return pd.DataFrame({"theoretical": theoretical, "sample": sample})

            if g_var:
                fig = go.Figure()
                for lv, sub in df.groupby(g_var):
                    dqq = qq_data(sub[x_var])
                    fig.add_trace(go.Scatter(
                        x=dqq["theoretical"], y=dqq["sample"], mode="markers",
                        name=str(lv), marker=dict(size=6)
                    ))
                # linea 45°
                allqq = qq_data(df[x_var])
                if not allqq.empty:
                    lo, hi = np.nanmin(allqq.values), np.nanmax(allqq.values)
                    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="y=x", line=dict(dash="dash")))
                fig.update_layout(template="simple_white", title=f"QQ-plot di {x_var}", height=420, font=dict(size=14))
                st.plotly_chart(fig, use_container_width=True)
            else:
                dqq = qq_data(df[x_var])
                fig = go.Figure()
                if not dqq.empty:
                    fig.add_trace(go.Scatter(x=dqq["theoretical"], y=dqq["sample"], mode="markers",
                                             name=x_var, marker=dict(size=6)))
                    lo, hi = np.nanmin(dqq.values), np.nanmax(dqq.values)
                    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="y=x", line=dict(dash="dash")))
                fig.update_layout(template="simple_white", title=f"QQ-plot di {x_var}", height=420, font=dict(size=14))
                st.plotly_chart(fig, use_container_width=True)

        # Test di normalità
        st.markdown("**Test di normalità**")
        if stats is None:
            st.info("`scipy` non disponibile: test di normalità non eseguibili.")
        else:
            def shapiro_safe(s: pd.Series):
                s = pd.to_numeric(s, errors="coerce").dropna()
                if len(s) < 3:
                    return None
                # Shapiro è consigliato n≤5000; oltre, lo eseguo comunque ma segnalo
                W, p = stats.shapiro(s)
                return {"n": len(s), "W": W, "p": p, "note": "Campione > 5000: interpretare con cautela." if len(s) > 5000 else ""}

            if g_var:
                rows = []
                for lv, sub in df.groupby(g_var):
                    res = shapiro_safe(sub[x_var])
                    rows.append([lv, res["n"] if res else "-", f'{res["W"]:.3f}' if res else "-", f'{res["p"]:.4f}' if res else "-", res["note"] if res else ""])
                out = pd.DataFrame(rows, columns=[g_var, "n", "W (Shapiro)", "p-value", "Note"])
                st.dataframe(out, use_container_width=True)
            else:
                res = shapiro_safe(df[x_var])
                if res:
                    st.write(f"n = {res['n']}, **W = {res['W']:.3f}**, **p = {res['p']:.4f}**. {res['note']}")
                else:
                    st.info("Campione troppo piccolo per eseguire il test.")

        with st.expander("ℹ️ Come leggere"):
            st.markdown("""
            - **Istogramma & QQ-plot**: punti del QQ-plot vicini alla linea 45° → distribuzione compatibile con la Normalità.  
            - **Shapiro–Wilk**: p-value < 0.05 → **evidenza contro** la normalità. p-value ≥ 0.05 → normalità plausibile.  
            - Valuti anche **asimmetria** e **code** guardando i grafici; i test sono sensibili per n grandi.  
            """)

# =============================================================================
# TAB 2 · OMOSCEDASTICITÀ tra gruppi
# =============================================================================
with tab_homo:
    st.subheader("⚖️ Omoscedasticità (varianze uguali tra gruppi)")
    if not num_vars or not cat_vars:
        st.info("Servono una variabile numerica e una categoriale.")
    else:
        y = st.selectbox("Variabile numerica (risposta)", options=num_vars, key=k("homo_y"))
        g = st.selectbox("Fattore (gruppi)", options=cat_vars, key=k("homo_g"))
        method = st.selectbox("Test", options=["Levene", "Brown–Forsythe (mediana)"], key=k("homo_meth"))

        # Grafico box a gruppi
        fig_b = px.box(df, x=g, y=y, points="outliers", template="simple_white")
        fig_b.update_layout(title=f"{y} per livelli di {g}", height=460, font=dict(size=14))
        st.plotly_chart(fig_b, use_container_width=True)

        # Test
        if stats is None:
            st.info("`scipy` non disponibile: test di omoscedasticità non eseguibile.")
        else:
            groups = [pd.to_numeric(sub[y], errors="coerce").dropna().values for _, sub in df.groupby(g)]
            groups = [garr for garr in groups if len(garr) > 1]
            if len(groups) < 2:
                st.info("Numero di gruppi insufficiente o con pochi dati.")
            else:
                if method.startswith("Levene"):
                    stat, p = stats.levene(*groups, center="mean")
                else:
                    stat, p = stats.levene(*groups, center="median")
                st.write(f"**Statistic = {stat:.3f}**, **p-value = {p:.4f}**")

        with st.expander("ℹ️ Come leggere"):
            st.markdown("""
            - **Boxplot per gruppo**: altezze e spread simili → varianze comparabili.  
            - **Levene/Brown–Forsythe**: p-value < 0.05 → varianze **diverse** (violazione omoscedasticità).  
            - In caso di violazione, consideri test **robusti** o trasformazioni (es. log).  
            """)

# =============================================================================
# TAB 3 · ASSUNZIONI per REGRESSIONE LINEARE
# =============================================================================
with tab_reg:
    st.subheader("📈 Assunzioni della regressione lineare (OLS)")
    if sm is None or smf is None:
        st.info("`statsmodels` non disponibile: impossibile stimare il modello OLS.")
    elif not num_vars:
        st.info("Servono variabili numeriche per stimare l'OLS.")
    else:
        # Scelte modello
        left, right = st.columns([2, 3])
        with left:
            y = st.selectbox("Risposta (numerica)", options=num_vars, key=k("reg_y"))
        with right:
            X_candidates = [c for c in num_vars if c != y]
            X = st.multiselect("Predittori (numerici)", options=X_candidates, default=X_candidates[:1], key=k("reg_xs"))

        if not X:
            st.warning("Selezioni almeno un predittore.")
        else:
            formula = f"`{y}` ~ " + " + ".join([f"`{xi}`" for xi in X])
            st.code(f"Formula OLS: {formula}", language="r")
            # Stima
            try:
                model = smf.ols(formula=formula, data=df).fit()
                st.text(model.summary().as_text())
            except Exception as e:
                st.error(f"Errore stima modello: {e}")
                model = None

            if model is not None:
                fitted = model.fittedvalues
                resid = model.resid
                df_diag = pd.DataFrame({ "fitted": fitted, "resid": resid, "abs_sqrt_resid": np.sqrt(np.abs(resid)) })

                # Grafici diagnostici affiancati
                g1, g2 = st.columns(2, vertical_alignment="top")
                with g1:
                    fig_r = px.scatter(df_diag, x="fitted", y="resid", template="simple_white")
                    fig_r.add_hline(y=0, line_dash="dash")
                    fig_r.update_layout(title="Residui vs Valori Fittati", height=420, font=dict(size=14),
                                        xaxis_title="Fitted", yaxis_title="Residui")
                    st.plotly_chart(fig_r, use_container_width=True)
                with g2:
                    # Scale-Location (spread ~ fitted)
                    fig_s = px.scatter(df_diag, x="fitted", y="abs_sqrt_resid", template="simple_white")
                    fig_s.update_layout(title="Scale-Location (√|residui| vs Fitted)", height=420, font=dict(size=14),
                                        xaxis_title="Fitted", yaxis_title="√|residui|")
                    st.plotly_chart(fig_s, use_container_width=True)

                # QQ-plot residui
                with st.expander("QQ-plot dei residui"):
                    def qq_resid(e: pd.Series):
                        s = pd.to_numeric(e, errors="coerce").dropna().sort_values()
                        n = len(s)
                        if stats is None or n < 3:
                            return pd.DataFrame(columns=["theoretical", "sample"])
                        theo = stats.norm.ppf((np.arange(1, n+1) - 0.5) / n)
                        m, sd = s.mean(), s.std(ddof=1) if s.std(ddof=1) > 0 else 1.0
                        theoretical = theo * sd + m
                        return pd.DataFrame({"theoretical": theoretical, "sample": s.values})

                    dqq = qq_resid(resid)
                    fig = go.Figure()
                    if not dqq.empty:
                        fig.add_trace(go.Scatter(x=dqq["theoretical"], y=dqq["sample"], mode="markers",
                                                 name="Residui", marker=dict(size=6)))
                        lo, hi = np.nanmin(dqq.values), np.nanmax(dqq.values)
                        fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="y=x", line=dict(dash="dash")))
                    fig.update_layout(template="simple_white", title="QQ-plot residui", height=420, font=dict(size=14))
                    st.plotly_chart(fig, use_container_width=True)

                # Test diagnostici
                st.markdown("**Test diagnostici sui residui**")
                cols_test = st.columns(3)
                # Normalità residui (Shapiro)
                with cols_test[0]:
                    if stats is not None:
                        try:
                            W, p = stats.shapiro(resid)
                            st.write(f"Shapiro: **W = {W:.3f}**, **p = {p:.4f}**")
                        except Exception:
                            st.write("Shapiro non calcolabile (campione troppo grande o errore).")
                    else:
                        st.write("Shapiro non disponibile.")
                # Omoscedasticità (Breusch-Pagan)
                with cols_test[1]:
                    if het_breuschpagan is not None:
                        try:
                            # BP richiede matrice regressori inclusa costante
                            exog = model.model.exog
                            lm, lmp, f, fp = het_breuschpagan(resid, exog)
                            st.write(f"Breusch–Pagan: **LM p = {lmp:.4f}**, **F p = {fp:.4f}**")
                        except Exception:
                            st.write("Breusch–Pagan non calcolabile.")
                    else:
                        st.write("Breusch–Pagan non disponibile.")
                # Indipendenza (Durbin–Watson)
                with cols_test[2]:
                    if sm is not None:
                        try:
                            dw = sm.stats.stattools.durbin_watson(resid)
                            st.write(f"Durbin–Watson: **{dw:.3f}** (≈2 indica indipendenza)")
                        except Exception:
                            st.write("Durbin–Watson non calcolabile.")
                    else:
                        st.write("Durbin–Watson non disponibile.")

                with st.expander("ℹ️ Come leggere"):
                    st.markdown("""
                    - **Residui vs Fitted**: assenza di pattern → **linearità** e **varianza costante** plausibili.  
                    - **Scale-Location**: spread circa uniforme lungo i fitted → **omoscedasticità**.  
                    - **QQ-plot residui**: punti vicino a 45° → **normalità** dei residui.  
                    - **Shapiro p<0.05** → residui non normali; **Breusch–Pagan p<0.05** → eteroschedasticità;  
                      **Durbin–Watson ≈ 2** → indipendenza; valori <<2 o >>2 indicano autocorrelazione.  
                    - In caso di violazioni, consideri **trasformazioni** (es. log), termini **non lineari** o modelli **robusti**.  
                    """)

# ──────────────────────────────────────────────────────────────────────────────
# Navigazione
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("---")
nav1, nav2 = st.columns(2)
with nav1:
    if st.button("⬅️ Torna: Explore Distributions", use_container_width=True, key=k("go_prev")):
        st.switch_page("pages/3_📊_Explore_Distributions.py")
with nav2:
    if st.button("➡️ Vai: Test statistici", use_container_width=True, key=k("go_next")):
        st.switch_page("pages/5_✏️_Statistical_Tests.py")
