# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np

from core.state import init_state

# Stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.genmod.families import Gaussian, Binomial, Poisson
from statsmodels.genmod.cov_struct import Independence, Exchangeable, Autoregressive

# Opzionali
try:
    from scipy import stats as spstats
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

import plotly.graph_objects as go

# ===========================================================
# Utility
# ===========================================================
def _use_active_df() -> pd.DataFrame:
    if "df_working" in st.session_state and st.session_state.df_working is not None:
        return st.session_state.df_working.copy()
    return st.session_state.df.copy()

def _is_numeric(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s)

def _mode_cat(s: pd.Series):
    try:
        return s.mode(dropna=True).iloc[0]
    except Exception:
        return s.dropna().iloc[0] if s.dropna().size else None

def _to_numeric_time(s: pd.Series):
    """Converte tempo: se datetime -> giorni dalla minima data; altrimenti coercizione a float."""
    if np.issubdtype(s.dtype, np.datetime64):
        base = pd.to_datetime(s).min()
        return (pd.to_datetime(s) - base).dt.total_seconds() / (3600*24.0)
    return pd.to_numeric(s, errors="coerce")

def _center(x: pd.Series | np.ndarray, ref: float | None = None):
    x = pd.to_numeric(x, errors="coerce")
    if ref is None:
        ref = np.nanmin(x)
    return x - ref, ref

def _icc_from_lmm_res(res, time_ref=0.0):
    """
    ICC per modello con random intercept; se c'è random slope restituiamo ICC "a tempo medio".
    Var_tot(t) ≈ Var_u0 + 2*t*Cov(u0,u1) + t^2*Var_u1 + Var_eps
    """
    try:
        cov_re = res.cov_re  # cov random effects
        var_eps = float(res.scale)  # resid
        if cov_re.shape == (1,1):  # solo RI
            var_u0 = float(cov_re.iloc[0,0])
            return var_u0 / (var_u0 + var_eps)
        elif cov_re.shape == (2,2):  # RI + slope
            var_u0 = float(cov_re.iloc[0,0]); cov_u0u1 = float(cov_re.iloc[0,1]); var_u1 = float(cov_re.iloc[1,1])
            var_t = var_u0 + 2* time_ref * cov_u0u1 + (time_ref**2) * var_u1 + var_eps
            return (var_u0 + 2*time_ref*cov_u0u1 + (time_ref**2)*var_u1) / var_t
    except Exception:
        pass
    return np.nan

def _pseudo_r2_marg_cond(y, yhat_fe, yhat_re=None):
    """
    Pseudo-R² marginale (solo fixed) e condizionale (fixed + random) per LMM.
    Marginale = Var(Xβ) / Var(Xβ + ε); Condizionale = Var(ŷ_cond) / Var(ŷ_cond + ε).
    Qui approssimiamo con varianza spiegata su y.
    """
    y = np.asarray(y, float)
    def _r2(y, yhat):
        if yhat is None: return np.nan
        ss_res = np.nansum((y - yhat)**2)
        ss_tot = np.nansum((y - np.nanmean(y))**2)
        return 1 - ss_res/ss_tot if ss_tot>0 else np.nan
    return _r2(y, yhat_fe), _r2(y, yhat_re) if yhat_re is not None else np.nan

def _build_prediction_grid(df_long, time_col, covars):
    """Crea griglia (tempo) + covariate a valori di riferimento (media per numeriche, moda per categoriche)."""
    tmin, tmax = np.nanmin(df_long[time_col]), np.nanmax(df_long[time_col])
    grid = np.linspace(tmin, tmax, 100)
    base = {time_col: grid}
    for c in covars:
        s = df_long[c]
        if _is_numeric(s):
            base[c] = np.repeat(np.nanmean(pd.to_numeric(s, errors="coerce")), len(grid))
        else:
            base[c] = np.repeat(_mode_cat(s), len(grid))
    return pd.DataFrame(base)

def _spaghetti_plot(df_long, id_col, time_col, y_col, max_ids=60):
    """Linee per soggetto (campionamento se molti soggetti)."""
    fig = go.Figure()
    ids = df_long[id_col].dropna().unique().tolist()
    if len(ids) > max_ids:
        rng = np.random.default_rng(42)
        ids = list(rng.choice(ids, size=max_ids, replace=False))
        note = f"(campionati {max_ids} soggetti su {df_long[id_col].nunique()})"
    else:
        note = ""
    for sid in ids:
        sub = df_long[df_long[id_col]==sid].sort_values(time_col)
        fig.add_trace(go.Scatter(x=sub[time_col], y=sub[y_col], mode="lines+markers", showlegend=False))
    fig.update_layout(title=f"Spaghetti plot per soggetto {note}", xaxis_title=time_col, yaxis_title=y_col)
    return fig

# ===========================================================
# Pagina
# ===========================================================
init_state()
st.title("📈 Analisi Longitudinale (misure ripetute)")

# Presenza dataset
if "df" not in st.session_state or st.session_state.df is None:
    st.warning("Nessun dataset disponibile. Carichi i dati in **Step 0 — Upload Dataset**.")
    st.page_link("pages/0_📂_Upload_Dataset.py", label="➡️ Vai a Upload Dataset", icon="📂")
    st.stop()

df = _use_active_df()
if df is None or df.empty:
    st.error("Il dataset attivo è vuoto.")
    st.stop()

# -----------------------------------------------------------
# Guida all'immissione dati
# -----------------------------------------------------------
with st.expander("ℹ️ Come immettere i dati (FORMATO LONG consigliato)", expanded=False):
    st.markdown("""
**Formato LONG (consigliato)**: una riga per **soggetto × timepoint**.  
Colonne tipiche:
- **ID_soggetto** (identificativo),  
- **tempo** (numerico: p.es. giorni/settimane/mesi; sono ammessi timepoint **irregolari**),  
- **outcome** (misura ripetuta),  
- **covariate** (es. sesso, età, gruppo, tratt., variabili tempo-varianti se presenti).

**Esempio**  
| id | tempo | outcome | gruppo | età_basale |
|----|------:|--------:|:------:|-----------:|
|  1 |   0.0 |   12.1  |   A    |     63     |
|  1 |  30.0 |   11.3  |   A    |     63     |
|  2 |   0.0 |    9.8  |   B    |     55     |
|  2 |  45.0 |   10.2  |   B    |     55     |

**Formato WIDE (opzionale)**: colonne multiple per i timepoint (es. `y_t0`, `y_t30`, `y_t90`).  
Il convertitore sottostante aiuta a passare a **LONG**.
""")

with st.expander("🧰 Convertitore opzionale: da WIDE a LONG", expanded=False):
    id_for_wide = st.selectbox("ID soggetto", options=list(df.columns))
    wide_cols = st.multiselect("Colonne (outcome ai vari timepoint)", options=[c for c in df.columns if c!=id_for_wide])
    static_covs = st.multiselect("Covariate statiche da mantenere", options=[c for c in df.columns if c not in ([id_for_wide] + wide_cols)])
    tp_str = st.text_input("Valori di tempo (nello stesso ordine delle colonne selezionate)", value=", ".join([str(i) for i in range(0, 30*len(wide_cols), 30)]) if wide_cols else "")
    if st.button("➜ Converti in LONG"):
        with st.spinner("Converto il formato…"):
            try:
                times = [float(x.strip()) for x in tp_str.split(",") if x.strip()!=""]
                if len(times) != len(wide_cols):
                    st.error("Il numero di timepoint non coincide con il numero di colonne selezionate.")
                else:
                    stack = []
                    for col, t in zip(wide_cols, times):
                        tmp = df[[id_for_wide, col] + static_covs].copy()
                        tmp["tempo"] = float(t)
                        tmp.rename(columns={col: "outcome"}, inplace=True)
                        stack.append(tmp)
                    df_long = pd.concat(stack, axis=0, ignore_index=True)
                    # Riordino
                    df_long = df_long[[id_for_wide, "tempo", "outcome"] + static_covs].sort_values([id_for_wide, "tempo"])
                    st.dataframe(df_long.head(20))
                    if st.button("Usa questo LONG come df_working"):
                        st.session_state.df_working = df_long
                        st.success("Impostato df_working al dataframe LONG convertito.")
            except Exception as e:
                st.error(f"Errore nella conversione: {e}")

# -----------------------------------------------------------
# Selezione variabili (LONG)
# -----------------------------------------------------------
st.subheader("Selezione variabili (formato LONG)")
id_col = st.selectbox("ID soggetto", options=list(df.columns))
time_col = st.selectbox("Colonna tempo", options=[c for c in df.columns if c != id_col])
y_col = st.selectbox("Outcome (misura ripetuta)", options=[c for c in df.columns if c not in [id_col, time_col]])

# Prepara LONG pulito
with st.spinner("Preparo dati LONG (gestione tempo irregolare, centering)…"):
    dat = df[[id_col, time_col, y_col]].copy()
    dat[time_col] = _to_numeric_time(dat[time_col])
    dat[y_col] = pd.to_numeric(dat[y_col], errors="coerce")
    # covariate extra (opzionali)
    cov_candidates = [c for c in df.columns if c not in [id_col, time_col, y_col]]
    covars = st.multiselect("Covariate (fisse o tempo-varianti)", options=cov_candidates)
    if covars:
        dat = pd.concat([dat, df[covars]], axis=1)

    dat = dat.dropna(subset=[id_col, time_col, y_col])  # righe con tripla info mancante
    dat = dat.sort_values([id_col, time_col]).reset_index(drop=True)
    # center time (opzione)
    do_center = st.checkbox("Centra il tempo (t0 = minimo globale)", value=True)
    if do_center:
        dat[time_col], t0 = _center(dat[time_col], ref=None)
        st.caption(f"Tempo centrato: t0 = {t0:.3g} (stesso per tutti i soggetti).")

# Spaghetti + trend medio
left, right = st.columns(2)
with left:
    with st.spinner("Genero lo spaghetti plot…"):
        fig_spa = _spaghetti_plot(dat, id_col, time_col, y_col, max_ids=60)
        st.plotly_chart(fig_spa, use_container_width=True)
    st.caption("Ogni linea è un soggetto. Le **distanze irregolari** tra punti sono ammesse; l’assenza di punti indica **missing**.")

with right:
    with st.spinner("Calcolo media per finestre temporali…"):
        # Media mobile semplice su bin temporali
        q = np.linspace(dat[time_col].min(), dat[time_col].max(), 21)
        bins = pd.cut(dat[time_col], q, include_lowest=True, duplicates="drop")
        g = dat.groupby(bins, observed=True)[[time_col, y_col]].agg({time_col:"mean", y_col:"mean"}).dropna()
        fig_mean = go.Figure()
        fig_mean.add_trace(go.Scatter(x=g[time_col], y=g[y_col], mode="lines+markers", name="Media per bin"))
        fig_mean.update_layout(title="Andamento medio (descrittivo)", xaxis_title=time_col, yaxis_title=y_col)
        st.plotly_chart(fig_mean, use_container_width=True)
    st.caption("Linea di **tendenza media** (descrittiva) per orientarsi prima del modello.")

# -----------------------------------------------------------
# Scelta modello
# -----------------------------------------------------------
st.subheader("Modellazione")
model_type = st.radio("Tipo di modello:", ["LMM (effetti misti, esito continuo)", "GEE (generalized estimating equations)"], horizontal=True)

# ===========================================================
# LMM — Linear Mixed-Effects (Gaussian)
# ===========================================================
if model_type == "LMM (effetti misti, esito continuo)":
    if not _is_numeric(dat[y_col]):
        st.error("Per LMM l'outcome deve essere **numerico continuo**.")
        st.stop()

    # Struttura effetti fissi e random
    fixed_terms = [time_col]
    fixed_terms += st.multiselect("Aggiungi effetti fissi:", options=covars)
    add_interactions = st.multiselect("Interazioni (con il tempo):", options=fixed_terms)
    for v in add_interactions:
        fixed_terms.append(f"{v}:{time_col}" if v != time_col else time_col)

    re_structure = st.radio("Effetti casuali:", ["Intercept casuale", "Intercept + slope casuale (tempo)"], horizontal=True)
    re_formula = "1" if re_structure == "Intercept casuale" else f"1 + {time_col}"

    formula = f"{y_col} ~ " + " + ".join(sorted(set([time_col] + [t for t in fixed_terms if t!=time_col])))
    st.code(f"Formula LMM: {formula}\nGruppi: {id_col} | Random: {re_formula}")

    with st.spinner("Stimo il modello LMM…"):
        try:
            md = smf.mixedlm(formula, data=dat, groups=dat[id_col], re_formula=re_formula)
            res = md.fit(method="lbfgs", disp=False)
        except Exception:
            # fallback più tollerante
            res = md.fit(method="powell", disp=False)

    # Tabella parametri
    st.markdown("### Stime dei parametri (effetti fissi)")
    summ = res.summary().tables[1]  # coeff table
    tab = pd.DataFrame(summ.data[1:], columns=summ.data[0])
    tab = tab.rename(columns={"Coef.":"Coef", "[0.025":"CI 2.5%", "0.975]":"CI 97.5%"}).set_index("")
    # Converte numeri
    for c in ["Coef","Std.Err.","z","P>|z|","CI 2.5%","CI 97.5%"]:
        if c in tab.columns:
            tab[c] = pd.to_numeric(tab[c], errors="coerce")
    st.dataframe(tab.round(4), use_container_width=True)
    st.caption("**Interpretazione**: il coefficiente del **tempo** quantifica la variazione media dell’esito per unità di tempo, a parità delle altre covariate.")

    # Varianze random & ICC
    var_eps = float(res.scale)
    try:
        cov_re = res.cov_re.copy()
    except Exception:
        cov_re = None
    icc_val = _icc_from_lmm_res(res, time_ref=float(np.nanmean(dat[time_col])))
    st.markdown("### Componenti di varianza e ICC")
    info_rows = [{"Parametro":"Var(errore residuo)", "Valore": var_eps}]
    if cov_re is not None:
        for i, r in enumerate(cov_re.index):
            for j, c in enumerate(cov_re.columns):
                if j>=i:
                    info_rows.append({"Parametro": f"Cov_RE[{r},{c}]", "Valore": float(cov_re.iloc[i,j])})
    info_rows.append({"Parametro":"ICC (a tempo medio)", "Valore": icc_val})
    st.dataframe(pd.DataFrame(info_rows).round(5), use_container_width=True)
    st.caption("**ICC** ≈ quota di varianza spiegata dalle **differenze tra soggetti**. Con random slope, l’ICC dipende dal tempo: qui è valutato al tempo medio.")

    # Pseudo-R² marginale/condizionale (approssimazione)
    with st.spinner("Calcolo pseudo-R² marginale/condizionale…"):
        fe_pred = res.fittedvalues  # per MixedLM: fitted condizionali; ricaviamo approx marginale con solo fixed
        try:
            # Predizioni marginali (solo fixed): usa predict senza effetti casuali
            grid = dat.copy()
            fe_only = res.predict(exog=grid, exog_re=np.zeros((len(grid), res.k_re)))
        except Exception:
            fe_only = None
        r2_marg, r2_cond = _pseudo_r2_marg_cond(dat[y_col].values, fe_only, res.fittedvalues)
    st.write(pd.DataFrame({"R² marginale":[r2_marg], "R² condizionale":[r2_cond]}).round(3))
    st.caption("**R² marginale**: solo effetti fissi; **R² condizionale**: fissi + casuali (capacità predittiva complessiva).")

    # Grafici affiancati: curva media prevista + diagnostica
    g1, g2 = st.columns(2)
    with g1:
        with st.spinner("Genero andamento previsto (marginale)…"):
            grid = _build_prediction_grid(dat, time_col, covars)
            try:
                yhat = res.predict(exog=grid)  # per MixedLM: per default predizioni marginali (RE=0)
            except Exception:
                # fallback: retta del tempo + intercetta media
                beta0 = res.fe_params.get("Intercept", np.nan)
                betat = res.fe_params.get(time_col, np.nan)
                yhat = beta0 + betat*grid[time_col]
            fig_pred = go.Figure()
            fig_pred.add_trace(go.Scatter(x=grid[time_col], y=yhat, mode="lines", name="Fisso (media popolazione)"))
            # Aggiunge media per bin come contesto
            fig_pred.add_trace(go.Scatter(x=g[time_col], y=g[y_col], mode="markers", name="Media per bin"))
            fig_pred.update_layout(title="Andamento previsto (marginale)", xaxis_title=time_col, yaxis_title=y_col)
            st.plotly_chart(fig_pred, use_container_width=True)
        st.caption("Linea **prevista** dal modello (solo effetti fissi: media di popolazione). I punti mostrano la **media empirica** per bin temporali.")

    with g2:
        with st.spinner("Diagnostica residui…"):
            resid = res.resid
            fitted = res.fittedvalues
            fig_res = go.Figure()
            fig_res.add_trace(go.Scatter(x=fitted, y=resid, mode="markers", name="Residui vs Fitted"))
            fig_res.add_hline(y=0, line=dict(dash="dash"))
            fig_res.update_layout(title="Residui vs valori previsti", xaxis_title="Fitted", yaxis_title="Residui")
            st.plotly_chart(fig_res, use_container_width=True)
            fig_hist = go.Figure()
            fig_hist.add_trace(go.Histogram(x=resid, nbinsx=30))
            fig_hist.update_layout(title="Distribuzione residui", xaxis_title="Residuo", yaxis_title="Frequenza")
            st.plotly_chart(fig_hist, use_container_width=True)
        st.caption("Residui **senza pattern** e distribuzione circa **simmetrica** supportano le assunzioni del modello lineare (omoscedasticità/normalità).")

    # Export
    if st.button("➕ Aggiungi risultati LMM al Results Summary"):
        with st.spinner("Salvo i risultati…"):
            if "report_items" not in st.session_state:
                st.session_state.report_items = []
            st.session_state.report_items.append({
                "type":"longitudinal_lmm",
                "title": f"LMM — {y_col} ~ {time_col} (+covariate)",
                "content":{
                    "formula": formula, "groups": id_col, "random": re_formula,
                    "fixed_effects": tab.round(6).reset_index().rename(columns={"": "term"}).to_dict(orient="records"),
                    "var_components": pd.DataFrame(info_rows).round(6).to_dict(orient="records"),
                    "r2_marginal": float(r2_marg) if np.isfinite(r2_marg) else None,
                    "r2_conditional": float(r2_cond) if np.isfinite(r2_cond) else None
                }
            })
        st.success("Risultati LMM aggiunti al Results Summary.")

# ===========================================================
# GEE — Generalized Estimating Equations
# ===========================================================
else:
    # Famiglia
    fam = st.selectbox("Famiglia (link predefinito)", options=["Gaussian (identity)", "Binomial (logit)", "Poisson (log)"])
    if fam.startswith("Gaussian"):
        family = Gaussian()
    elif fam.startswith("Binomial"):
        family = Binomial()
    else:
        family = Poisson()

    # Struttura di correlazione
    corr = st.selectbox("Struttura di correlazione intra-soggetto", options=["Independence", "Exchangeable", "AR(1)"])
    cov_struct = {"Independence": Independence(), "Exchangeable": Exchangeable(), "AR(1)": Autoregressive() }[corr]

    # Effetti fissi (GEE non ha RE)
    fixed_terms = [time_col] + st.multiselect("Effetti fissi (aggiuntivi):", options=covars)
    inters = st.multiselect("Interazioni (con il tempo):", options=fixed_terms)
    for v in inters:
        if v != time_col:
            fixed_terms.append(f"{v}:{time_col}")
    formula = f"{y_col} ~ " + " + ".join(sorted(set(fixed_terms)))
    st.code(f"Formula GEE: {formula}\nGruppi: {id_col} | Corr: {corr}")

    with st.spinner("Stimo il modello GEE…"):
        try:
            # GEE formula interface
            gee_mod = sm.GEE.from_formula(formula, groups=dat[id_col], data=dat, family=family, cov_struct=cov_struct, time=dat[time_col])
            gee_res = gee_mod.fit()
        except Exception as e:
            st.error(f"Impossibile stimare GEE: {e}")
            st.stop()

    # Tabella coefficienti (robust SE)
    st.markdown("### Stime dei parametri (robust SE)")
    geet = gee_res.summary().tables[1]
    tab = pd.DataFrame(geet.data[1:], columns=geet.data[0]).set_index("")
    # Rinominazione colonne comuni
    rename_map = {"coef":"Coef", "std err":"Std.Err.", "P>|z|":"P>|z|", "[0.025":"CI 2.5%", "0.975]":"CI 97.5%"}
    tab = tab.rename(columns={c:rename_map.get(c,c) for c in tab.columns})
    for c in ["Coef","Std.Err.","z","P>|z|","CI 2.5%","CI 97.5%"]:
        if c in tab.columns:
            tab[c] = pd.to_numeric(tab[c], errors="coerce")
    st.dataframe(tab.round(4), use_container_width=True)
    st.caption(
        "**Interpretazione**: il coefficiente del **tempo** indica variazione media attesa per unità di tempo (link di famiglia). "
        "Per **Binomiale (logit)** l’esponenziale del coefficiente è un **odds ratio**; per **Poisson (log)** è un **rate ratio**."
    )

    # Predizione media (marginale)
    g1, g2 = st.columns(2)
    with g1:
        with st.spinner("Genero andamento previsto (marginale)…"):
            grid = _build_prediction_grid(dat, time_col, covars)
            yhat = gee_res.predict(exog=grid)
            fig_pred = go.Figure()
            fig_pred.add_trace(go.Scatter(x=grid[time_col], y=yhat, mode="lines", name="Previsto"))
            fig_pred.add_trace(go.Scatter(x=g[time_col], y=g[y_col], mode="markers", name="Media per bin"))
            fig_pred.update_layout(title="Andamento previsto (GEE)", xaxis_title=time_col, yaxis_title=y_col)
            st.plotly_chart(fig_pred, use_container_width=True)
        st.caption("Curva **marginale** prevista dal GEE; robusta rispetto alla specificazione della varianza (SE sandwich).")

    with g2:
        with st.spinner("Diagnostica residui (Pearson)…"):
            resid = gee_res.resid_pearson
            fitted = gee_res.fittedvalues
            fig_res = go.Figure()
            fig_res.add_trace(go.Scatter(x=fitted, y=resid, mode="markers"))
            fig_res.add_hline(y=0, line=dict(dash="dash"))
            fig_res.update_layout(title="Residui di Pearson vs fitted", xaxis_title="Fitted", yaxis_title="Residui Pearson")
            st.plotly_chart(fig_res, use_container_width=True)
            fig_hist = go.Figure()
            fig_hist.add_trace(go.Histogram(x=resid, nbinsx=30))
            fig_hist.update_layout(title="Distribuzione residui (Pearson)", xaxis_title="Residuo", yaxis_title="Frequenza")
            st.plotly_chart(fig_hist, use_container_width=True)
        st.caption("Residui vicino a 0 senza pattern marcati sono desiderabili; attenzione a eteroschedasticità o outlier.")

    # Export
    if st.button("➕ Aggiungi risultati GEE al Results Summary"):
        with st.spinner("Salvo i risultati…"):
            if "report_items" not in st.session_state:
                st.session_state.report_items = []
            st.session_state.report_items.append({
                "type":"longitudinal_gee",
                "title": f"GEE — {y_col} ~ {time_col} (+covariate)",
                "content":{
                    "formula": formula, "groups": id_col, "cov_struct": corr,
                    "coefficients": tab.round(6).reset_index().rename(columns={"": "term"}).to_dict(orient="records")
                }
            })
        st.success("Risultati GEE aggiunti al Results Summary.")

# -----------------------------------------------------------
# Sezione didattica
# -----------------------------------------------------------
with st.expander("ℹ️ Spiegazione completa (cosa si calcola e come leggere)", expanded=False):
    st.markdown("""
**Perché i modelli longitudinali?**  
Le misure ripetute sullo stesso soggetto sono **correlate**. I modelli semplici (OLS) ignorano tale dipendenza e sottostimano gli errori.

**LMM (Linear Mixed-Effects)**  
- Adatti per esiti **continui** (Gaussiani).  
- Separano l’effetto **medio di popolazione** (effetti fissi) dalla **variabilità tra soggetti** (effetti casuali).  
- **Random intercept**: consente a ciascun soggetto un proprio livello di partenza.  
- **Random slope sul tempo**: consente a ciascun soggetto un proprio **trend**.  
- **ICC**: quota di varianza attribuibile alle differenze tra soggetti.  
- **R² marginale**: quota spiegata dai soli effetti fissi; **R² condizionale**: fissi + casuali.  
- **Interpretazione**: il coefficiente del **tempo** è la variazione media per unità di tempo, a parità delle altre covariate.

**GEE (Generalized Estimating Equations)**  
- Adatti a esiti **non Gaussiani** (Binario/Poisson) o anche continui con specifica robusta.  
- Forniscono stime **marginali** (di popolazione) con **SE robusti**.  
- Richiedono una struttura di correlazione intra-soggetto (**Independence**, **Exchangeable**, **AR(1)**); le stime dei coefficienti sono robuste, i SE migliorano se la struttura è vicina al vero.  
- **Interpretazione**: con link logit/log, l’esponenziale dei coefficienti è un **odds ratio**/**rate ratio**.

**Irregolarità dei timepoint**  
- Entrambi gli approcci gestiscono timepoint **non equispaziati**. Nel LMM il tempo può entrare come **variabile continua** e gli effetti casuali (slope) modellano differenze individuali di pendenza.

**Dati mancanti**  
- LMM/GEE sono validi in genere sotto **MAR (Missing At Random)**. Se i drop-out dipendono dall’esito non osservato, valutare modelli più complessi (pattern-mixture, joint models).

**Suggerimenti pratici**  
- Centrando il **tempo** si facilita l’interpretazione dell’intercetta (valore medio a t=0).  
- Valutare interazioni **covariata × tempo** per effetti che cambiano nel tempo.  
- Controllare i **residui** e la plausibilità della struttura casuale/correlativa.  
""")
