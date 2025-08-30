# -*- coding: utf-8 -*-
# pages/15_⏱️_Analisi_Serie_Temporali.py
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

# Plotly
try:
    import plotly.graph_objects as go
    import plotly.express as px
except Exception:
    go = None
    px = None

# Statsmodels / time series
try:
    import statsmodels.api as sm
    from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from statsmodels.stats.diagnostic import acorr_ljungbox
    _has_sm = True
except Exception:
    _has_sm = False

# SciPy (per distribuzioni e diagnostica)
try:
    from scipy import stats as sps
    _has_scipy = True
except Exception:
    _has_scipy = False

# Auto-ARIMA (opzionale)
try:
    import pmdarima as pm
    _has_pmdarima = True
except Exception:
    _has_pmdarima = False

# ──────────────────────────────────────────────────────────────────────────────
# Data store condiviso
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
            st.error("Nessun dataset attivo. Importi i dati e completi la pulizia nella pagina di Upload/Cleaning.")
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
st.set_page_config(page_title="Analisi Serie Temporali", layout="wide")
try:
    from nav import sidebar
    sidebar()
except Exception:
    pass

KEY = "ts"
def k(name: str) -> str:
    return f"{KEY}_{name}"

# ──────────────────────────────────────────────────────────────────────────────
# Utility
# ──────────────────────────────────────────────────────────────────────────────
def fmt_p(p: float | None) -> str:
    if p is None or p != p:
        return "—"
    if p < 1e-4:
        return "< 1e-4"
    return f"{p:.4f}"

def infer_freq_from_index(idx: pd.DatetimeIndex) -> str | None:
    try:
        f = pd.infer_freq(idx)
        return f
    except Exception:
        return None

def resample_series(s: pd.Series, freq: str, how: str) -> pd.Series:
    if how == "Somma":
        return s.resample(freq).sum()
    elif how == "Media":
        return s.resample(freq).mean()
    elif how == "Ultimo valore":
        return s.resample(freq).last()
    elif how == "Primo valore":
        return s.resample(freq).first()
    else:
        return s.resample(freq).mean()

def interpolate_series(s: pd.Series, method: str) -> pd.Series:
    if method == "Forward-fill":
        return s.ffill()
    elif method == "Backward-fill":
        return s.bfill()
    elif method == "Lineare":
        return s.interpolate(method="linear")
    elif method == "Spline (ordine 2)":
        try:
            return s.interpolate(method="spline", order=2)
        except Exception:
            return s.interpolate(method="linear")
    else:
        return s

def add_transformations(y: pd.Series, do_log: bool, do_diff: bool, d_seasonal: int, m: int) -> tuple[pd.Series, dict]:
    info = {}
    y_trans = y.copy()
    if do_log:
        if (y_trans <= 0).any():
            info["log_warning"] = "Trasformazione log non applicata: la serie contiene valori ≤ 0."
        else:
            y_trans = np.log(y_trans)
            info["log_applied"] = True
    if do_diff:
        y_trans = y_trans.diff().dropna()
        info["diff_applied"] = True
    if d_seasonal > 0 and m > 1:
        y_trans = y_trans.diff(m * d_seasonal).dropna()
        info["sdiff_applied"] = True
    return y_trans, info

def train_test_split_ts(y: pd.Series, test_size: int) -> tuple[pd.Series, pd.Series]:
    if test_size <= 0 or test_size >= len(y):
        return y, None
    return y.iloc[:-test_size], y.iloc[-test_size:]

def metrics_forecast(y_true: pd.Series, y_pred: pd.Series) -> dict:
    y_true, y_pred = pd.Series(y_true).align(pd.Series(y_pred), join="inner")
    e = y_true - y_pred
    mae = float(np.mean(np.abs(e)))
    rmse = float(np.sqrt(np.mean(e**2)))
    mape = float(np.mean(np.abs(e) / np.maximum(np.abs(y_true), 1e-12))) * 100.0
    smape = float(np.mean(2.0 * np.abs(e) / np.maximum(np.abs(y_true) + np.abs(y_pred), 1e-12))) * 100.0
    return {"MAE": mae, "RMSE": rmse, "MAPE%": mape, "sMAPE%": smape}

def acf_plot_values(x: pd.Series, nlags: int = 40, alpha: float = 0.05) -> tuple[np.ndarray, np.ndarray, float]:
    acfv = acf(x, nlags=nlags, fft=True, missing="drop")
    z = sps.norm.ppf(1 - alpha/2) if _has_scipy else 1.96
    conf = z / np.sqrt(len(x.dropna()))
    lags = np.arange(len(acfv))
    return lags, acfv, conf

def pacf_plot_values(x: pd.Series, nlags: int = 40, alpha: float = 0.05) -> tuple[np.ndarray, np.ndarray, float]:
    pacfv = pacf(x, nlags=nlags, method="ywmle")
    z = sps.norm.ppf(1 - alpha/2) if _has_scipy else 1.96
    conf = z / np.sqrt(len(x.dropna()))
    lags = np.arange(len(pacfv))
    return lags, pacfv, conf

# ──────────────────────────────────────────────────────────────────────────────
# Header & dati
# ──────────────────────────────────────────────────────────────────────────────
st.title("⏱️ Analisi di Serie Temporali")
st.caption("EDA, test di stazionarietà (ADF/KPSS), decomposizione stagionale, ACF/PACF, ARIMA/SARIMA/SARIMAX, diagnostica residui e forecast con IC. Layout guidato e coerente con gli altri moduli.")

ensure_initialized()
DF = get_active(required=True)
with st.expander("Stato dati", expanded=False):
    stamp_meta()
if DF is None or DF.empty:
    st.stop()

all_cols = list(DF.columns)
num_cols = [c for c in all_cols if pd.api.types.is_numeric_dtype(DF[c])]
dt_candidates = [c for c in all_cols if pd.api.types.is_datetime64_any_dtype(DF[c]) or "date" in c.lower() or "time" in c.lower()]

# ──────────────────────────────────────────────────────────────────────────────
# 1) Impostazione della serie
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("### 1) Impostazione della serie")

c1, c2, c3 = st.columns([1.2, 1.2, 1.2])
with c1:
    time_col = st.selectbox("Colonna **tempo** (datetime o convertibile)", options=all_cols, index=(all_cols.index(dt_candidates[0]) if dt_candidates else 0), key=k("time"))
with c2:
    y_col = st.selectbox("Variabile **target** (numerica)", options=[c for c in num_cols if c != time_col], key=k("y"))
with c3:
    agg = st.selectbox("In caso di più righe per periodo, aggrega con", options=["Media", "Somma", "Ultimo valore", "Primo valore"], index=0, key=k("agg"))

# Preparazione serie
work = DF[[time_col, y_col]].copy()
# parsing datetime
try:
    work[time_col] = pd.to_datetime(work[time_col], errors="coerce")
except Exception:
    pass
work = work.dropna(subset=[time_col, y_col]).sort_values(time_col)
work = work.set_index(time_col)
# aggregazione per periodo esatto (se duplicati)
if work.index.has_duplicates:
    how = agg
    s = resample_series(work[y_col], freq="D", how=how)  # placeholder giornaliero per consolidare? Meglio: raggruppo per indice esatto
    # in realtà se duplicati nello stesso timestamp, faccio groupby sull'indice
    work = work.groupby(level=0).agg({y_col: {"Media": "mean", "Somma": "sum", "Ultimo valore": "last", "Primo valore": "first"}[agg]})
    work.columns = [y_col]

# Frequenza
infreq = infer_freq_from_index(work.index)
freq_map = {"Giornaliera": "D", "Settimanale": "W", "Mensile": "M", "Trimestrale": "Q", "Annuale": "A"}
c4, c5, c6 = st.columns([1.2, 1.2, 1.2])
with c4:
    freq_ui = st.selectbox("Frequenza della serie", options=list(freq_map.keys()) + ["(Tieni frequenza originale)"], index=(list(freq_map.values()).index(infreq) if infreq in freq_map.values() else len(freq_map)), key=k("freq"))
with c5:
    impute = st.selectbox("Gestione **missing**", options=["Nessuna", "Forward-fill", "Backward-fill", "Lineare", "Spline (ordine 2)"], index=1, key=k("impute"))
with c6:
    resample_how = st.selectbox("Aggregazione nel **resampling**", options=["Media", "Somma", "Ultimo valore", "Primo valore"], index=0, key=k("how_res"))

# Eventuale resampling
y = work[y_col].copy()
if freq_ui != "(Tieni frequenza originale)":
    freq_code = freq_map[freq_ui]
    y = resample_series(y, freq=freq_code, how=resample_how)

# Interpolazione/riempimento missing
if impute != "Nessuna":
    y = interpolate_series(y, impute)

# Mostra info serie
st.markdown("#### Anteprima e info serie")
cA, cB, cC, cD = st.columns(4)
with cA: st.metric("Osservazioni", f"{len(y.dropna())}")
with cB: st.metric("Periodo iniziale", f"{y.index.min()}" if len(y) else "—")
with cC: st.metric("Periodo finale", f"{y.index.max()}" if len(y) else "—")
with cD: st.metric("Frequenza (inferita)", infreq or "—")
st.line_chart(y, height=300, use_container_width=True)

# ──────────────────────────────────────────────────────────────────────────────
# 2) EDA: decomposizione e ACF/PACF
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("### 2) Esplorazione: decomposizione e autocorrelazioni")
c1, c2, c3 = st.columns([1.1, 1.1, 1.1])
with c1:
    seasonality_guess = st.selectbox("Periodo stagionale **m**", options=[0, 4, 7, 12, 24, 52], index=3, help="0=nessuna stagionalità; tipici: 12 mensile, 52 settimanale (dati settimanali in un anno).", key=k("m"))
with c2:
    decomp_model = st.selectbox("Modello decomposizione", options=["additive", "multiplicative"], index=0, key=k("decomp"))
with c3:
    nlags = st.slider("Lag per ACF/PACF", min_value=12, max_value=200, value=48, step=4, key=k("nlags"))

# Decomposizione
if _has_sm and len(y.dropna()) >= max(3*max(1, seasonality_guess), 24):
    try:
        decomp = seasonal_decompose(y.dropna(), model=decomp_model, period=(seasonality_guess if seasonality_guess > 0 else None), extrapolate_trend="freq")
        fig = go.Figure()
        # trend
        fig.add_trace(go.Scatter(x=y.index, y=y.values, mode="lines", name="Serie", line=dict(width=2)))
        if decomp.trend is not None:
            fig.add_trace(go.Scatter(x=decomp.trend.index, y=decomp.trend.values, mode="lines", name="Trend", line=dict(width=2)))
        if decomp.seasonal is not None:
            fig.add_trace(go.Scatter(x=decomp.seasonal.index, y=decomp.seasonal.values, mode="lines", name="Stagionale", line=dict(width=1)))
        if decomp.resid is not None:
            fig.add_trace(go.Scatter(x=decomp.resid.index, y=decomp.resid.values, mode="lines", name="Residuo", line=dict(width=1)))
        fig.update_layout(template="simple_white", height=420, title="Decomposizione (serie, trend, stagionale, residuo)")
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.info(f"Decomposizione non disponibile: {e}")
else:
    st.caption("Per la decomposizione servono dati sufficienti (≥24 osservazioni e multipli del periodo).")

# ACF/PACF
if _has_sm and len(y.dropna()) >= 10:
    try:
        lags_a, acfv, conf_a = acf_plot_values(y.dropna(), nlags=nlags)
        lags_p, pacfv, conf_p = pacf_plot_values(y.dropna(), nlags=nlags)
        f1 = go.Figure()
        f1.add_trace(go.Bar(x=lags_a, y=acfv, name="ACF"))
        f1.add_hline(y=conf_a, line_width=1)
        f1.add_hline(y=-conf_a, line_width=1)
        f1.update_layout(template="simple_white", height=300, title="ACF (con bande di conf.)", xaxis_title="Lag", yaxis_title="ACF")
        f2 = go.Figure()
        f2.add_trace(go.Bar(x=lags_p, y=pacfv, name="PACF"))
        f2.add_hline(y=conf_p, line_width=1)
        f2.add_hline(y=-conf_p, line_width=1)
        f2.update_layout(template="simple_white", height=300, title="PACF (con bande di conf.)", xaxis_title="Lag", yaxis_title="PACF")
        c1, c2 = st.columns(2)
        c1.plotly_chart(f1, use_container_width=True)
        c2.plotly_chart(f2, use_container_width=True)
    except Exception as e:
        st.info(f"ACF/PACF non disponibili: {e}")

with st.expander("ℹ️ Come leggere ACF/PACF e decomposizione"):
    st.markdown(
        "- **Decomposizione**: separa **trend**, **stagionalità** e **residuo**; utile per capire trasformazioni necessarie.  \n"
        "- **ACF**: picchi persistenti ⇒ dipendenza seriale/possibile stagionalità; decadimento lento ⇒ **non stazionarietà** (serve differenziazione).  \n"
        "- **PACF**: aiuta a scegliere **p** (AR): cut-off netto a lag p. ACF aiuta su **q** (MA): cut-off a lag q. Con stagionalità, guardare multipli di **m**."
    )

# ──────────────────────────────────────────────────────────────────────────────
# 3) Test di stazionarietà
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("### 3) Test di stazionarietà")
c1, c2, c3 = st.columns([1.1, 1.1, 1.3])
with c1:
    do_log = st.checkbox("Trasformazione **log(y)**", value=False, key=k("log"))
with c2:
    do_diff = st.checkbox("Differenza **d=1**", value=False, key=k("d1"))
with c3:
    d_seasonal = st.selectbox("Differenza **stagionale D**", options=[0,1,2], index=0, help="Applicata come differenza a lag m.", key=k("D"))

m = int(seasonality_guess) if seasonality_guess else 0
y_t, tinfo = add_transformations(y, do_log, do_diff, d_seasonal, m)
st.line_chart(y_t, height=220, use_container_width=True)

if _has_sm and len(y_t.dropna()) >= 12:
    try:
        adf = adfuller(y_t.dropna(), autolag="AIC")
        st.metric("ADF — p-value", fmt_p(float(adf[1])))
        if 'stationary' in st.session_state: pass
        if adf[1] < 0.05:
            st.caption("ADF: p < 0.05 ⇒ **rifiuto** unit root ⇒ serie (approssimativamente) **stazionaria**.")
        else:
            st.caption("ADF: p ≥ 0.05 ⇒ **unit root** non rifiutata ⇒ probabile **non stazionarietà** (considerare differenze/trasformazioni).")
    except Exception as e:
        st.info(f"ADF non disponibile: {e}")
    try:
        kps = kpss(y_t.dropna(), regression="c", nlags="auto")
        st.metric("KPSS — p-value", fmt_p(float(kps[1])))
        st.caption("KPSS: p piccolo ⇒ **rifiuto** stazionarietà (al contrario di ADF). Usare ADF+KPSS in combinazione.")
    except Exception as e:
        st.info(f"KPSS non disponibile: {e}")
else:
    st.caption("I test richiedono un numero sufficiente di osservazioni (≥12).")

# ──────────────────────────────────────────────────────────────────────────────
# 4) Modellazione: ARIMA/SARIMA/SARIMAX
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("### 4) Modellazione (ARIMA/SARIMA/SARIMAX)")

c1, c2 = st.columns([1.2, 1.8])
with c1:
    mode = st.radio("Selezione ordini", ["Automatica (auto-ARIMA)", "Manuale"], index=(0 if _has_pmdarima else 1), key=k("mode"))
    test_size = st.number_input("Osservazioni nel **test set**", min_value=0, max_value=max(0, len(y_t)-5), value=min(12, max(0, len(y_t)//5)), step=1, key=k("testsz"))
    fh = st.number_input("Orizzonte **forecast** (passi avanti)", min_value=1, max_value=1000, value=12, step=1, key=k("fh"))
with c2:
    exog_cols = st.multiselect("Variabili **esogene** (opzionale)", options=[c for c in DF.columns if c not in {y_col, time_col}], help="Devono essere note nel futuro per il forecast.", key=k("exog"))

# Allinei eventuali esogene alla serie trasformata (stesse operazioni di resampling/imputazione)
X = None
if exog_cols:
    X = DF[[time_col] + exog_cols].copy()
    X[time_col] = pd.to_datetime(X[time_col], errors="coerce")
    X = X.dropna(subset=[time_col]).set_index(time_col).sort_index()
    if freq_ui != "(Tieni frequenza originale)":
        freq_code = freq_map[freq_ui]
        for c in exog_cols:
            X[c] = resample_series(X[c], freq=freq_code, how="Mean" if X[c].dtype.kind in "fc" else "Last")
    if impute != "Nessuna":
        for c in exog_cols:
            X[c] = interpolate_series(X[c], impute)
    # se trasformazioni (diff) sono applicate a y, NON si applicano a X (in SARIMAX la differenza è interna al modello)

# Train/Test
y_train, y_test = train_test_split_ts(y_t.dropna(), test_size=test_size)
X_train = X.loc[y_train.index] if X is not None else None
X_test = (X.loc[y_test.index] if (X is not None and y_test is not None) else None)

# Stima
model_summary_text = None
fitted = None
pred_test = None
pred_future = None
order = seasonal_order = None

if _has_sm and len(y_train) >= 10:
    try:
        if mode.startswith("Automatica") and _has_pmdarima:
            seasonal = (m is not None and m > 1)
            auto_model = pm.auto_arima(
                y_train, X=X_train, seasonal=seasonal, m=(m if seasonal else 1),
                start_p=0, start_q=0, start_P=0, start_Q=0,
                max_p=5, max_q=5, max_P=2, max_Q=2, max_d=2, max_D=1,
                information_criterion="aic",
                stepwise=True, suppress_warnings=True, error_action="ignore"
            )
            order = auto_model.order
            seasonal_order = auto_model.seasonal_order if seasonal else (0,0,0,0)
            st.caption(f"Ordini selezionati automaticamente: ARIMA{order}  SARIMA{seasonal_order}")
            model = SARIMAX(y_train, exog=X_train, order=order, seasonal_order=seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
            res = model.fit(disp=False)
        else:
            cA, cB, cC, cD = st.columns(4)
            with cA: p = st.number_input("p (AR)", 0, 10, 1, 1, key=k("p"))
            with cB: d = st.number_input("d (diff.)", 0, 2, (1 if do_diff else 0), 1, key=k("d"))
            with cC: q = st.number_input("q (MA)", 0, 10, 1, 1, key=k("q"))
            with cD: pass
            if m and m > 1:
                cE, cF, cG, cH = st.columns(4)
                with cE: P = st.number_input("P (stag. AR)", 0, 3, 0, 1, key=k("P"))
                with cF: D = st.number_input("D (stag. diff.)", 0, 2, d_seasonal, 1, key=k("D"))
                with cG: Q = st.number_input("Q (stag. MA)", 0, 3, 0, 1, key=k("Q"))
                with cH: m_ui = st.number_input("m (stag. periodi)", 1, 366, m or 1, 1, key=k("m_ui"))
            else:
                P=Q=D=0; m_ui=0
            order = (int(p), int(d), int(q))
            seasonal_order = (int(P), int(D), int(Q), int(m_ui))
            model = SARIMAX(y_train, exog=X_train, order=order, seasonal_order=seasonal_order if m_ui>1 else (0,0,0,0),
                            enforce_stationarity=False, enforce_invertibility=False)
            res = model.fit(disp=False)

        fitted = res
        model_summary_text = res.summary().as_text()

        # Previsioni su test
        if y_test is not None and len(y_test) > 0:
            pred_test = res.get_prediction(start=y_test.index[0], end=y_test.index[-1], exog=X_test)
            df_test_pred = pred_test.predicted_mean
            ci_test = pred_test.conf_int(alpha=0.05)
            met = metrics_forecast(y_test, df_test_pred)
            t1, t2, t3, t4 = st.columns(4)
            t1.metric("MAE (test)", f"{met['MAE']:.4f}")
            t2.metric("RMSE (test)", f"{met['RMSE']:.4f}")
            t3.metric("MAPE (test) %", f"{met['MAPE%']:.2f}")
            t4.metric("sMAPE (test) %", f"{met['sMAPE%']:.2f}")

            # Grafico test
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=y_train.index, y=y_train, mode="lines", name="Train"))
            fig.add_trace(go.Scatter(x=y_test.index, y=y_test, mode="lines", name="Test"))
            fig.add_trace(go.Scatter(x=df_test_pred.index, y=df_test_pred, mode="lines", name="Pred (test)", line=dict(width=2)))
            if isinstance(ci_test, pd.DataFrame) and ci_test.shape[1] >= 2:
                fig.add_trace(go.Scatter(x=ci_test.index, y=ci_test.iloc[:,1], mode="lines", line=dict(width=0), showlegend=False))
                fig.add_trace(go.Scatter(x=ci_test.index, y=ci_test.iloc[:,0], mode="lines", fill="tonexty", fillcolor="rgba(127,127,127,0.2)", line=dict(width=0), showlegend=False))
            fig.update_layout(template="simple_white", height=420, title="Predizione su test con IC 95%")
            st.plotly_chart(fig, use_container_width=True)

        # Forecast futuro
        future_index = None
        if fh > 0:
            last = y.index.max()
            # genera future index coerente con frequenza attuale
            freq_for_future = y.index.freqstr if y.index.freqstr else (infer_freq_from_index(y.index) or ("M" if len(y) >= 12 else "D"))
            future_index = pd.date_range(start=last, periods=fh+1, freq=freq_for_future, inclusive="right")
            X_future = None
            if X is not None:
                # Nota: per forecast con exog servono valori futuri; se non disponibili, usiamo ultimo valore (hold)
                X_future = X.iloc[[-1]].reindex(future_index, method="ffill")
            pr = res.get_forecast(steps=fh, exog=X_future)
            y_f = pr.predicted_mean
            ci = pr.conf_int(alpha=0.05)

            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=y.index, y=y, mode="lines", name="Serie (usata nel modello)"))
            fig2.add_trace(go.Scatter(x=y_f.index, y=y_f, mode="lines", name="Forecast", line=dict(width=3)))
            if isinstance(ci, pd.DataFrame) and ci.shape[1] >= 2:
                fig2.add_trace(go.Scatter(x=ci.index, y=ci.iloc[:,1], mode="lines", line=dict(width=0), showlegend=False))
                fig2.add_trace(go.Scatter(x=ci.index, y=ci.iloc[:,0], mode="lines", fill="tonexty", fillcolor="rgba(127,127,127,0.2)", line=dict(width=0), showlegend=False))
            fig2.update_layout(template="simple_white", height=420, title=f"Forecast {fh} passi con IC 95%")
            st.plotly_chart(fig2, use_container_width=True)

        # Diagnostica residui
        with st.expander("Diagnostica residui"):
            resid = res.resid.dropna()
            c1, c2, c3 = st.columns(3)
            with c1:
                fig_r = go.Figure()
                fig_r.add_trace(go.Scatter(x=resid.index, y=resid, mode="lines", name="Residui"))
                fig_r.add_hline(y=0, line_width=1)
                fig_r.update_layout(template="simple_white", height=260, title="Residui nel tempo")
                st.plotly_chart(fig_r, use_container_width=True)
            with c2:
                lags_r, acfr, conf_r = acf_plot_values(resid, nlags=min(40, max(10, len(resid)//5)))
                fig_acf = go.Figure()
                fig_acf.add_trace(go.Bar(x=lags_r, y=acfr, name="ACF residui"))
                fig_acf.add_hline(y=conf_r, line_width=1)
                fig_acf.add_hline(y=-conf_r, line_width=1)
                fig_acf.update_layout(template="simple_white", height=260, title="ACF residui", xaxis_title="Lag")
                st.plotly_chart(fig_acf, use_container_width=True)
            with c3:
                try:
                    lb = acorr_ljungbox(resid, lags=[min(10, len(resid)//3)], return_df=True)
                    st.metric("Ljung–Box p", fmt_p(float(lb["lb_pvalue"].iloc[0])))
                    st.caption("p grande ⇒ residui ≈ **rumore bianco** (assenza di autocorrelazione residua).")
                except Exception:
                    st.caption("Ljung–Box non disponibile.")
        with st.expander("Dettagli modello"):
            st.text(model_summary_text)

    except Exception as e:
        st.error(f"Errore nella stima del modello: {e}")
else:
    st.info("Dati insufficienti per stimare un modello (servono almeno ~10–20 osservazioni nel train).")

# ──────────────────────────────────────────────────────────────────────────────
# 5) Come impostare e interpretare
# ──────────────────────────────────────────────────────────────────────────────
with st.expander("📌 Guida rapida: come impostare e interpretare", expanded=True):
    st.markdown(
        "- **Frequenza**: uniformi la serie con un’unica frequenza (D/W/M/…). Colmi i **missing** con metodi semplici (FFill/Lineare) solo se appropriato.  \n"
        "- **Trasformazioni**:  \n"
        "  • **log(y)** per ridurre varianza crescente (solo se y>0);  \n"
        "  • **differenze** (d, D) per ottenere **stazionarietà** (ADF/KPSS).  \n"
        "- **Ordini**: use ACF/PACF per orientare **p/q** e **P/Q**; **m** è il periodo stagionale (es. 12 per mensile).  \n"
        "- **Valutazione**: usi un **test set** finale e confronti MAE/RMSE/MAPE.  \n"
        "- **Diagnostica**: residui senza pattern e ACF residui entro bande ⇒ modello adeguato; p(Ljung–Box) alta ⇒ residui ~ rumore.  \n"
        "- **Forecast con esogene**: con **SARIMAX** le covariate future devono essere **note o previste**; in mancanza, l’app usa l’ultimo valore disponibile (scelta cauta ma limitata)."
    )

# ──────────────────────────────────────────────────────────────────────────────
# Navigazione
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("---")
nav1, nav2 = st.columns(2)
with nav1:
    if st.button("⬅️ Torna: Panel Analysis", use_container_width=True, key=k("go_prev")):
        for target in [
            "pages/13_🏛️_Panel_Analysis.py",
            "pages/13_🏛️_Panel_Analysis (1).py",
        ]:
            try:
                st.switch_page(target)
                break
            except Exception:
                continue
with nav2:
    if st.button("➡️ Vai: Report / Export", use_container_width=True, key=k("go_next")):
        for target in [
            "pages/14_🧾_Report_Automatico.py",
            "pages/14_📤_Export_Risultati.py",
            "pages/13_🧾_Report_Automatico.py",
            "pages/13_📤_Export_Risultati.py",
        ]:
            try:
                st.switch_page(target)
                break
            except Exception:
                continue
