# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np

from core.state import init_state

# Opzionali (solo per comodità dei test statistici/percentili precisi; il modulo funziona anche senza)
try:
    from scipy import stats as spstats
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

# Plot
import plotly.graph_objects as go

# ===========================================================
# Utility di base
# ===========================================================
def _use_active_df() -> pd.DataFrame:
    """Usa il dataset working se esiste, altrimenti l'originale df."""
    if "df_working" in st.session_state and st.session_state.df_working is not None:
        return st.session_state.df_working.copy()
    return st.session_state.df.copy()

def _clean_pair(a: pd.Series, b: pd.Series):
    """Allinea due colonne e rimuove i missing."""
    df = pd.DataFrame({"a": a, "b": b}).dropna()
    return df["a"].astype(float).values, df["b"].astype(float).values

def _percentile(x, q):
    if _HAS_SCIPY:
        return float(spstats.scoreatpercentile(x, q))
    return float(np.percentile(x, q))

# ===========================================================
# Bland–Altman
# ===========================================================
def bland_altman(a, b, mode="assoluto", use_t_ci=True):
    """
    mode ∈ {"assoluto", "percentuale", "log"}.
    Restituisce: dict con bias, sd_diff, loa_low/high, IC per bias e LoA, diff, mean (o base per ratio).
    """
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    if mode == "assoluto":
        diff = a - b
        base = (a + b) / 2.0
    elif mode == "percentuale":
        base = (a + b) / 2.0
        diff = 100.0 * (a - b) / base
    elif mode == "log":
        # log-ratio: su base naturale; interpretazione su scala % ~ 100*(e^bias - 1)
        a_pos = np.where(a <= 0, np.nan, a)
        b_pos = np.where(b <= 0, np.nan, b)
        mask = ~np.isnan(a_pos) & ~np.isnan(b_pos)
        a = a_pos[mask]; b = b_pos[mask]
        diff = np.log(a) - np.log(b)
        base = (np.log(a) + np.log(b)) / 2.0
    else:
        raise ValueError("mode non riconosciuto")

    n = len(diff)
    bias = float(np.mean(diff))
    sd = float(np.std(diff, ddof=1)) if n > 1 else np.nan
    z = 1.96
    loa_low = bias - z * sd
    loa_high = bias + z * sd

    # IC per bias (t) e per LoA (Bland & Altman 1986 – stima SE delle LoA)
    tcrit = float(spstats.t.ppf(0.975, n-1)) if _HAS_SCIPY and n > 1 else 1.96
    se_bias = sd / np.sqrt(n) if (n > 0 and np.isfinite(sd)) else np.nan
    ci_bias = (bias - tcrit * se_bias, bias + tcrit * se_bias) if np.isfinite(se_bias) else (np.nan, np.nan)

    # SE delle LoA: sd * sqrt(1/n + z^2/(2*(n-1)))
    if n > 1 and np.isfinite(sd):
        se_loa = sd * np.sqrt(1.0/n + z**2 / (2.0*(n-1)))
        ci_loa_low = (loa_low - tcrit*se_loa, loa_low + tcrit*se_loa)
        ci_loa_high = (loa_high - tcrit*se_loa, loa_high + tcrit*se_loa)
    else:
        ci_loa_low = (np.nan, np.nan); ci_loa_high = (np.nan, np.nan)

    # Proportional bias: regressione diff ~ base
    slope, intercept = np.nan, np.nan
    if n >= 3 and np.all(np.isfinite([diff.mean(), base.mean()])):
        # OLS semplice
        x = base - np.mean(base)
        y = diff - np.mean(diff)
        denom = np.sum(x**2)
        if denom > 0:
            slope = float(np.sum(x*y) / denom)
            intercept = float(np.mean(diff) - slope * np.mean(base))

    # Misure riassuntive utili
    repeatability = z * sd  # 1.96*SD delle differenze (half-range delle LoA)
    perc_error = 100.0 * sd / np.mean(base) if mode != "log" and np.mean(base) != 0 else np.nan

    return {
        "mode": mode, "n": n, "diff": diff, "base": base,
        "bias": bias, "sd": sd, "loa_low": loa_low, "loa_high": loa_high,
        "ci_bias": ci_bias, "ci_loa_low": ci_loa_low, "ci_loa_high": ci_loa_high,
        "prop_bias_slope": slope, "prop_bias_intercept": intercept,
        "repeatability": repeatability, "perc_error": perc_error
    }

def bland_altman_plot(res):
    """Restituisce due figure affiancate: BA scatter e istogramma delle differenze."""
    diff, base = res["diff"], res["base"]
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=base, y=diff, mode="markers", name="Differenze"))
    fig1.add_hline(y=res["bias"], line=dict(dash="dash"), annotation_text=f"Bias={res['bias']:.3g}")
    fig1.add_hline(y=res["loa_low"], line=dict(dash="dot"), annotation_text=f"LoA−={res['loa_low']:.3g}")
    fig1.add_hline(y=res["loa_high"], line=dict(dash="dot"), annotation_text=f"LoA+={res['loa_high']:.3g}")
    if np.isfinite(res.get("prop_bias_slope", np.nan)):
        # linea di proportional bias: diff = intercept + slope*base
        xgrid = np.linspace(np.min(base), np.max(base), 100)
        yhat = res["prop_bias_intercept"] + res["prop_bias_slope"]*xgrid
        fig1.add_trace(go.Scatter(x=xgrid, y=yhat, mode="lines", name="Proportional bias (fit)"))
    fig1.update_layout(
        title="Bland–Altman plot",
        xaxis_title=("Media dei metodi" if res["mode"]!="log" else "Media dei log"),
        yaxis_title=("Differenza (A−B)" if res["mode"]=="assoluto" else ("Diff % su media" if res["mode"]=="percentuale" else "log(A)−log(B)"))
    )

    fig2 = go.Figure()
    fig2.add_trace(go.Histogram(x=diff, nbinsx=30, name="Differenze"))
    fig2.update_layout(title="Distribuzione delle differenze", xaxis_title="Differenza", yaxis_title="Frequenza")

    return fig1, fig2

# ===========================================================
# Lin’s CCC con bootstrap
# ===========================================================
def ccc(a, b):
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    mu_a, mu_b = np.mean(a), np.mean(b)
    va, vb = np.var(a, ddof=1), np.var(b, ddof=1)
    cov = np.cov(a, b, ddof=1)[0,1]
    return float((2*cov) / (va + vb + (mu_a - mu_b)**2)) if np.isfinite(va+vb) else np.nan

def bootstrap_ci_ccc(a, b, n_boot=1000, alpha=0.05, seed=123):
    rng = np.random.default_rng(seed)
    n = len(a); boots = []
    idx = np.arange(n)
    for _ in range(n_boot):
        ii = rng.integers(0, n, size=n)
        boots.append(ccc(a[idx[ii]], b[idx[ii]]))
    lo, hi = np.nanpercentile(boots, [100*alpha/2, 100*(1-alpha/2)])
    return float(lo), float(hi)

# ===========================================================
# Deming regression (errors-in-variables)
# ===========================================================
def deming(a, b, lamb=1.0):
    """
    Stima chiusa di Deming: y = alpha + beta*x con errore su entrambe le misure.
    lamb = Var(error_x) / Var(error_y). Se incerto, usare 1.0.
    """
    x = np.asarray(a, dtype=float); y = np.asarray(b, dtype=float)
    xbar, ybar = np.mean(x), np.mean(y)
    s_xx = np.var(x, ddof=1); s_yy = np.var(y, ddof=1); s_xy = np.cov(x, y, ddof=1)[0,1]
    if s_xy == 0:
        beta = np.nan; alpha = np.nan
    else:
        delta = (s_yy - lamb*s_xx)
        beta = (delta + np.sqrt(delta**2 + 4*lamb*s_xy**2)) / (2*s_xy)
        alpha = ybar - beta*xbar
    return float(alpha), float(beta)

def bootstrap_ci_deming(a, b, lamb=1.0, n_boot=1000, alpha=0.05, seed=123):
    rng = np.random.default_rng(seed)
    n = len(a); idx = np.arange(n)
    betas, alphas = [], []
    for _ in range(n_boot):
        ii = rng.integers(0, n, size=n)
        aa, bb = np.asarray(a)[ii], np.asarray(b)[ii]
        al, bt = deming(aa, bb, lamb=lamb)
        alphas.append(al); betas.append(bt)
    lo_b, hi_b = np.nanpercentile(betas, [100*alpha/2, 100*(1-alpha/2)])
    lo_a, hi_a = np.nanpercentile(alphas, [100*alpha/2, 100*(1-alpha/2)])
    return (float(lo_a), float(hi_a)), (float(lo_b), float(hi_b))

def deming_plot(a, b, alpha, beta):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=a, y=b, mode="markers", name="Dati"))
    # retta identità
    minx, maxx = float(np.nanmin(a)), float(np.nanmax(a))
    fig.add_trace(go.Scatter(x=[minx, maxx], y=[minx, maxx], mode="lines", name="Identità (y=x)", line=dict(dash="dash")))
    # retta Deming
    xs = np.linspace(minx, maxx, 200)
    ys = alpha + beta*xs
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name="Deming fit"))
    fig.update_layout(title="Deming regression (errors-in-variables)", xaxis_title="Metodo A (x)", yaxis_title="Metodo B (y)")
    return fig

# ===========================================================
# ICC (two-way random, absolute agreement) — ICC(2,1) e ICC(2,k)
# ===========================================================
def icc_two_way_random_absolute(data_matrix):
    """
    data_matrix: array shape (n_soggetti, n_valutatori) con possibili NaN (righe con NaN scartate).
    Ritorna: ICC(2,1), ICC(2,k), componenti ANOVA.
    """
    X = np.asarray(data_matrix, dtype=float)
    # drop righe con NaN
    X = X[~np.isnan(X).any(axis=1)]
    n, k = X.shape
    if n < 2 or k < 2:
        return np.nan, np.nan, {}
    grand = np.mean(X)
    mean_i = np.mean(X, axis=1, keepdims=True)  # per soggetto
    mean_j = np.mean(X, axis=0, keepdims=True)  # per rater
    # SS
    ss_subject = k * np.sum((mean_i - grand)**2)
    ss_rater   = n * np.sum((mean_j - grand)**2)
    ss_total   = np.sum((X - grand)**2)
    ss_error   = ss_total - ss_subject - ss_rater
    df_subject = n - 1
    df_rater   = k - 1
    df_error   = (n - 1) * (k - 1)
    ms_subject = ss_subject / df_subject if df_subject>0 else np.nan
    ms_rater   = ss_rater / df_rater if df_rater>0 else np.nan
    ms_error   = ss_error / df_error if df_error>0 else np.nan

    icc21 = (ms_subject - ms_error) / (ms_subject + (k - 1)*ms_error + (k*(ms_rater - ms_error) / n))
    icc2k = (ms_subject - ms_error) / (ms_subject + (ms_rater - ms_error) / n)

    comps = {
        "n": n, "k": k,
        "MS_subject": float(ms_subject), "MS_rater": float(ms_rater), "MS_error": float(ms_error),
        "DF_subject": int(df_subject), "DF_rater": int(df_rater), "DF_error": int(df_error)
    }
    return float(icc21), float(icc2k), comps

# ===========================================================
# Kappa (non pesata e pesata quadratica)
# ===========================================================
def kappa_categ(r1, r2, weighted=False):
    s = pd.DataFrame({"r1": r1, "r2": r2}).dropna()
    cats = sorted(pd.unique(s[["r1","r2"]].values.ravel()), key=lambda x: str(x))
    m = len(cats)
    mapcat = {c:i for i,c in enumerate(cats)}
    i1 = s["r1"].map(mapcat).values
    i2 = s["r2"].map(mapcat).values
    # Matrice di confusione m x m
    C = np.zeros((m,m), dtype=float)
    for a,b in zip(i1,i2):
        C[a,b] += 1.0
    N = np.sum(C)
    Po = np.trace(C)/N if N>0 else np.nan
    p1 = np.sum(C, axis=1)/N
    p2 = np.sum(C, axis=0)/N
    if weighted and m>1:
        W = np.zeros((m,m))
        for i in range(m):
            for j in range(m):
                W[i,j] = 1 - ((i - j)**2)/((m - 1)**2)  # quadratic weights (Cicchetti-Allison)
        # kappa pesata: (sum w_ij * p_ij - sum w_ij * p_i. p_.j) / (1 - sum w_ij * p_i. p_.j)
        Pij = C/N
        Pe_w = np.sum(W * np.outer(p1,p2))
        k_w = (np.sum(W * Pij) - Pe_w) / (1 - Pe_w) if (1 - Pe_w)!=0 else np.nan
        return float(k_w), C, cats, True
    else:
        Pe = np.sum(p1*p2)
        k = (Po - Pe) / (1 - Pe) if (1 - Pe)!=0 else np.nan
        return float(k), C, cats, False

# ===========================================================
# Pagina Streamlit
# ===========================================================
init_state()
st.title("📏 Analisi di Agreement (Bland–Altman, CCC, Deming, ICC, Kappa)")

# Check dataset
if "df" not in st.session_state or st.session_state.df is None:
    st.warning("Nessun dataset disponibile. Carichi i dati in **Step 0 — Upload Dataset**.")
    st.page_link("pages/0_📂_Upload_Dataset.py", label="➡️ Vai a Upload Dataset", icon="📂")
    st.stop()

df = _use_active_df()
if df is None or df.empty:
    st.error("Il dataset attivo è vuoto.")
    st.stop()

st.subheader("Selezione modalità")
mode = st.radio(
    "Tipo di agreement:",
    ["Continua (BA, CCC, Deming)", "ICC (più valutatori)", "Categoriale (Kappa)"],
    horizontal=True
)

# ===========================================================
# 1) CONTINUA: Bland–Altman, CCC, Deming
# ===========================================================
if mode == "Continua (BA, CCC, Deming)":
    cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if len(cols) < 2:
        st.error("Servono almeno due colonne numeriche.")
        st.stop()

    c1, c2 = st.columns(2)
    with c1:
        a_col = st.selectbox("Metodo A (colonna numerica):", options=cols, index=0)
    with c2:
        b_col = st.selectbox("Metodo B (colonna numerica):", options=[c for c in cols if c != a_col], index=0)

    a, b = _clean_pair(df[a_col], df[b_col])
    if len(a) < 3:
        st.error("Servono almeno 3 coppie senza missing.")
        st.stop()

    st.markdown("### Bland–Altman")
    ba_type = st.radio("Scala delle differenze:", ["assoluto", "percentuale", "log"], horizontal=True)
    with st.spinner("Calcolo Bland–Altman…"):
        ba = bland_altman(a, b, mode=ba_type)
    g1, g2 = st.columns(2)
    with g1:
        fig_ba, fig_hist = bland_altman_plot(ba)
        st.plotly_chart(fig_ba, use_container_width=True)
        st.caption(
            "Bland–Altman: punti = differenza tra metodi vs loro media. "
            "La linea centrale è il **bias**; le linee tratteggiate sono le **Limits of Agreement (±1.96·SD)**. "
            "La retta obliqua (se presente) indica **proportional bias** (differenza che cambia con l’ampiezza)."
        )
    with g2:
        st.plotly_chart(fig_hist, use_container_width=True)
        st.caption(
            "Distribuzione delle differenze: utile per verificare approssimazione alla normalità (ipotesi usata per le LoA). "
            "Valutare code pesanti/asimmetrie. Con dati positivi e proporzioni, preferire scala **log** o **%**."
        )

    # Tabella riassuntiva BA
    st.markdown("**Statistiche Bland–Altman**")
    ci_bias = ba["ci_bias"]; ci_l = ba["ci_loa_low"]; ci_h = ba["ci_loa_high"]
    ba_tbl = pd.DataFrame({
        "Parametro": ["n", "Bias", "SD diff", "LoA bassa", "LoA alta", "CI Bias (low)", "CI Bias (high)", "CI LoA− (low)", "CI LoA− (high)", "CI LoA+ (low)", "CI LoA+ (high)", "Repeatability (1.96·SD)", "% error (rel. media)"],
        "Valore": [ba["n"], ba["bias"], ba["sd"], ba["loa_low"], ba["loa_high"], ci_bias[0], ci_bias[1], ci_l[0], ci_l[1], ci_h[0], ci_h[1], ba["repeatability"], ba["perc_error"]]
    }).round(4)
    st.dataframe(ba_tbl, use_container_width=True)

    # CCC con bootstrap
    st.markdown("### Concordance Correlation Coefficient (Lin)")
    nboot_ccc = st.slider("Bootstrap per IC del CCC (repliche)", 200, 5000, 1000, step=100)
    with st.spinner("Calcolo CCC e intervalli bootstrap…"):
        ccc_val = ccc(a,b)
        c_lo, c_hi = bootstrap_ci_ccc(a,b, n_boot=nboot_ccc, alpha=0.05)
    st.write(pd.DataFrame({"CCC":[round(ccc_val,4)], "CI 2.5%":[round(c_lo,4)], "CI 97.5%":[round(c_hi,4)]}))
    st.caption(
        "Il **CCC** combina **precisione** (correlazione) e **accuratezza** (scostamento dalla linea identità). "
        "Valori vicini a 1 indicano forte concordanza. La correlazione da sola **non** misura l’accordo."
    )

    # Deming regression
    st.markdown("### Deming regression (errors-in-variables)")
    lamb = st.number_input("Rapporto λ = Var(errore_x) / Var(errore_y)", min_value=0.0001, value=1.0, step=0.1, format="%.4f")
    nboot_dem = st.slider("Bootstrap per IC della retta di Deming (repliche)", 200, 5000, 1000, step=100)
    with st.spinner("Stimo retta di Deming e IC bootstrap…"):
        alpha_hat, beta_hat = deming(a, b, lamb=lamb)
        (a_lo, a_hi), (b_lo, b_hi) = bootstrap_ci_deming(a, b, lamb=lamb, n_boot=nboot_dem)
        fig_dem = deming_plot(a, b, alpha_hat, beta_hat)
    g3, g4 = st.columns(2)
    with g3:
        st.plotly_chart(fig_dem, use_container_width=True)
        st.caption(
            "Scatter con **retta identità** (y=x) e **retta di Deming**. "
            "Deming corregge l’attenuazione dovuta all’errore su **entrambi** gli assi; "
            "impostare λ≈1 quando le varianze d’errore sono simili."
        )
    with g4:
        st.write(pd.DataFrame({
            "Parametro":["Intercetta (α)","Pendenza (β)","CI α low","CI α high","CI β low","CI β high"],
            "Valore":[alpha_hat, beta_hat, a_lo, a_hi, b_lo, b_hi]
        }).round(4))
        st.caption("IC tramite **bootstrap** a campione con rimpiazzo; utili quando le ipotesi parametriche non sono pienamente garantite.")

    # Export
    if st.button("➕ Aggiungi risultati (BA/CCC/Deming) al Results Summary"):
        with st.spinner("Salvo nel Results Summary…"):
            if "report_items" not in st.session_state:
                st.session_state.report_items = []
            st.session_state.report_items.append({
                "type":"agreement_continuous",
                "title": f"Agreement continuo — {a_col} vs {b_col}",
                "content":{
                    "bland_altman": ba_tbl.to_dict(orient="records"),
                    "ccc": {"value": float(ccc_val), "ci":[float(c_lo), float(c_hi)], "n_boot": int(nboot_ccc)},
                    "deming": {
                        "lambda": float(lamb),
                        "alpha": float(alpha_hat), "beta": float(beta_hat),
                        "alpha_ci":[float(a_lo), float(a_hi)],
                        "beta_ci":[float(b_lo), float(b_hi)],
                        "n_boot": int(nboot_dem)
                    }
                }
            })
        st.success("Risultati aggiunti al Results Summary.")

    with st.expander("ℹ️ Spiegazione (utente non esperto)"):
        st.markdown("""
**Bland–Altman**: misura l’accordo guardando le **differenze** tra metodi.  
- **Bias** = differenza media; **LoA** = intervallo in cui ci si aspetta cadranno ~95% delle differenze.  
- Se le differenze **crescono** con i valori medi → **proportional bias** (valutare scala % o **log**).  
**CCC (Lin)**: concordanza totale rispetto alla **linea di identità** (non solo correlazione).  
**Deming**: regressione che considera errore in **entrambi** i metodi; utile per calibrazione e conversione.
""")

# ===========================================================
# 2) ICC — più valutatori
# ===========================================================
elif mode == "ICC (più valutatori)":
    st.info("Formato atteso: un’**ID del soggetto** e **≥2 colonne** con le misure dei diversi valutatori/metodi.")
    id_col = st.selectbox("Colonna ID soggetto:", options=list(df.columns))
    meas_cols = st.multiselect("Selezioni le colonne dei valutatori (≥2):", options=[c for c in df.columns if c != id_col])
    if len(meas_cols) < 2:
        st.stop()
    # Costruisce matrice n x k
    mat = df[meas_cols].copy()
    with st.spinner("Calcolo ICC(2,1) e ICC(2,k)…"):
        icc21, icc2k, comps = icc_two_way_random_absolute(mat.values)
    st.write(pd.DataFrame({
        "ICC(2,1)":[round(icc21,4)], "ICC(2,k)":[round(icc2k,4)],
        "MS_subject":[round(comps.get("MS_subject", np.nan),4)],
        "MS_rater":[round(comps.get("MS_rater", np.nan),4)],
        "MS_error":[round(comps.get("MS_error", np.nan),4)]
    }))
    st.caption(
        "**ICC(2,1)**: accordo assoluto per **una singola** valutazione. **ICC(2,k)**: media di k valutazioni. "
        "Valori ~0.5 moderati, ~0.75 buoni, ≥0.9 eccellenti (regole pratiche)."
    )

    # Grafici affiancati: heatmap correlazioni e boxplot per valutatore
    g1, g2 = st.columns(2)
    with g1:
        with st.spinner("Genero heatmap di correlazione tra valutatori…"):
            C = np.corrcoef(mat.dropna().values, rowvar=False)
            fig_hm = go.Figure(data=go.Heatmap(z=C, x=meas_cols, y=meas_cols, colorscale="RdBu", zmid=0))
            fig_hm.update_layout(title="Correlazioni tra valutatori")
            st.plotly_chart(fig_hm, use_container_width=True)
        st.caption("Correlazioni alte non implicano necessariamente **accordo**: l’ICC tiene conto di **bias** tra valutatori.")
    with g2:
        with st.spinner("Genero boxplot per valutatore…"):
            fig_bx = go.Figure()
            for c in meas_cols:
                fig_bx.add_trace(go.Box(y=df[c], name=str(c), boxmean=True))
            fig_bx.update_layout(title="Distribuzioni per valutatore", yaxis_title="Valore")
            st.plotly_chart(fig_bx, use_container_width=True)
        st.caption("Boxplot per verificare differenze di **livello** e **variabilità** tra valutatori (possibile fonte di disaccordo).")

    if st.button("➕ Aggiungi risultati ICC al Results Summary"):
        with st.spinner("Salvo nel Results Summary…"):
            if "report_items" not in st.session_state:
                st.session_state.report_items = []
            st.session_state.report_items.append({
                "type":"agreement_icc",
                "title": f"ICC — valutatori: {', '.join(meas_cols)}",
                "content":{"icc21": float(icc21), "icc2k": float(icc2k), "anova_components": comps}
            })
        st.success("Risultati ICC aggiunti al Results Summary.")

    with st.expander("ℹ️ Spiegazione (utente non esperto)"):
        st.markdown("""
**ICC** valuta quanto le differenze tra soggetti superino le differenze tra **valutatori** e **errore**.  
- **ICC(2,1)**: two-way random effects, **absolute agreement**, singola misura.  
- **ICC(2,k)**: come sopra, ma per la **media** di k misure (accordo migliora).  
""")

# ===========================================================
# 3) CATEGORIALE — Cohen’s kappa
# ===========================================================
else:
    cols = [c for c in df.columns]
    r1 = st.selectbox("Valutatore 1 (colonna categoriale):", options=cols, index=0)
    r2 = st.selectbox("Valutatore 2 (colonna categoriale):", options=[c for c in cols if c != r1], index=0)
    weighted = st.checkbox("Usa kappa pesata (pesi quadratici)", value=False)
    s = pd.DataFrame({"r1": df[r1], "r2": df[r2]}).dropna()
    if s.empty:
        st.error("Nessuna coppia disponibile dopo rimozione NA.")
        st.stop()

    with st.spinner("Calcolo kappa…"):
        k_val, C, cats, is_w = kappa_categ(s["r1"], s["r2"], weighted=weighted)

    g1, g2 = st.columns(2)
    with g1:
        fig_cm = go.Figure(data=go.Heatmap(z=C, x=cats, y=cats, colorscale="Blues", text=C, texttemplate="%{text}"))
        fig_cm.update_layout(title="Matrice di confusione", xaxis_title="Valutatore 2", yaxis_title="Valutatore 1")
        st.plotly_chart(fig_cm, use_container_width=True)
        st.caption("Matrice di conteggi tra le categorie dei due valutatori.")
    with g2:
        st.write(pd.DataFrame({"Kappa":[round(k_val,4)], "Pesi quadratici":[bool(is_w)]}))
        st.caption(
            "**Kappa** corregge l’accordo **atteso per caso**. La versione pesata considera la **distanza** tra categorie "
            "(qui pesi **quadratici**, più tolleranti per disaccordi di una categoria)."
        )

    if st.button("➕ Aggiungi risultati Kappa al Results Summary"):
        with st.spinner("Salvo nel Results Summary…"):
            if "report_items" not in st.session_state:
                st.session_state.report_items = []
            st.session_state.report_items.append({
                "type":"agreement_kappa",
                "title": f"Kappa — {r1} vs {r2}",
                "content":{"kappa": float(k_val), "weighted": bool(is_w), "categories": list(map(str,cats)), "confusion": C.astype(int).tolist()}
            })
        st.success("Risultati Kappa aggiunti al Results Summary.")

    with st.expander("ℹ️ Spiegazione (utente non esperto)"):
        st.markdown("""
**Cohen’s kappa** misura l’accordo tra due valutatori per dati **categoriali**, correggendo l’accordo atteso per caso.  
- **Kappa pesata** (quadratica) penalizza meno i disaccordi **vicini** e di più quelli **lontani**.
""")
