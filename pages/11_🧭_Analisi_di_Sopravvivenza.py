# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np

from core.state import init_state

# Opzionali
try:
    import lifelines
    from lifelines import KaplanMeierFitter, NelsonAalenFitter
    from lifelines import CoxPHFitter, WeibullAFTFitter
    from lifelines.statistics import logrank_test, multivariate_logrank_test, proportional_hazard_test
    _HAS_LIFELINES = True
except Exception:
    _HAS_LIFELINES = False

try:
    from scipy import stats as spstats
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

import plotly.graph_objects as go

# ===========================================================
# Utility comuni
# ===========================================================
def _use_active_df() -> pd.DataFrame:
    if "df_working" in st.session_state and st.session_state.df_working is not None:
        return st.session_state.df_working.copy()
    return st.session_state.df.copy()

def _is_binary_like(s: pd.Series) -> bool:
    vals = pd.Series(s).dropna().unique().tolist()
    return len(vals) == 2

def _as_event(series: pd.Series, event_value):
    """Restituisce indicatori 0/1 dove 1 = evento (== event_value)."""
    s = series.copy()
    return (s == event_value).astype(int)

def _numeric(series: pd.Series):
    return pd.to_numeric(series, errors="coerce")

def _group_levels(s: pd.Series):
    return sorted(pd.Series(s).dropna().unique().tolist(), key=lambda x: str(x))

def _km_plotly_from_kmf(kmf: KaplanMeierFitter, name="KM", show_ci=False):
    """Crea una traccia Plotly stepwise dalla stima KM di lifelines."""
    t = kmf.survival_function_.index.values
    s = kmf.survival_function_.iloc[:, 0].values
    trace = go.Scatter(x=t, y=s, mode="lines", name=name, line_shape="hv")
    traces = [trace]
    if show_ci and hasattr(kmf, "confidence_interval_"):
        ci = kmf.confidence_interval_
        lo = ci.iloc[:, 0].values
        hi = ci.iloc[:, 1].values
        traces.append(go.Scatter(x=t, y=hi, mode="lines", line=dict(width=0), showlegend=False))
        traces.append(go.Scatter(x=t, y=lo, mode="lines", fill="tonexty", name=f"{name} (IC95%)",
                                 line=dict(width=0)))
    return traces

def _risk_table_from_kmf(kmf: KaplanMeierFitter, at_times: list[float]):
    """Restituisce n_at_risk a tempi specificati (interpolando allo step precedente)."""
    ev = kmf.event_table  # columns include 'at_risk'
    # per ogni t*, prendi at_risk all'ultimo tempo <= t*
    times = ev.index.values
    at_risk = ev['at_risk'].values
    out = []
    for tt in at_times:
        idx = np.searchsorted(times, tt, side="right") - 1
        idx = np.clip(idx, 0, len(times)-1)
        out.append(int(at_risk[idx]))
    return out

def _parse_timepoints(text: str):
    try:
        vals = [float(x.strip()) for x in text.split(",") if x.strip() != ""]
        return [v for v in vals if np.isfinite(v)]
    except Exception:
        return []

def _design_matrix(df: pd.DataFrame, covars: list[str]):
    """Dummies drop_first per le categoriche; float per le numeriche."""
    X = df[covars].copy()
    X = pd.get_dummies(X, drop_first=True)
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    return X

# ===========================================================
# Pagina
# ===========================================================
init_state()
st.title("🧭 Analisi di Sopravvivenza")

# Check dataset
if "df" not in st.session_state or st.session_state.df is None:
    st.warning("Nessun dataset disponibile. Carichi i dati in **Step 0 — Upload Dataset**.")
    st.page_link("pages/0_📂_Upload_Dataset.py", label="➡️ Vai a Upload Dataset", icon="📂")
    st.stop()

df = _use_active_df()
if df is None or df.empty:
    st.error("Il dataset attivo è vuoto.")
    st.stop()

# -----------------------------------------------------------
# Selettori base
# -----------------------------------------------------------
st.subheader("Selezione variabili")
num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
time_col = st.selectbox("Tempo di follow-up (numerico, stessa unità per tutti):", options=num_cols)
event_col = st.selectbox("Indicatore di evento/censura (binario o due categorie):", options=[c for c in df.columns if c != time_col])

ev_vals = _group_levels(df[event_col])
if len(ev_vals) != 2:
    st.error("La variabile evento deve avere esattamente due livelli (es. 0/1).")
    st.stop()

event_value = st.selectbox("Valore che indica **evento** (l'altro sarà censura):", options=ev_vals, index=1 if 1 in ev_vals else 0)

# Stratificazione facoltativa (per KM/log-rank)
group_var = st.selectbox("Stratificazione (opzionale, per curve KM e log-rank):", options=["— Nessuna —"] + [c for c in df.columns if c not in [time_col, event_col]])

# Covariate per Cox/AFT (opzionali)
covariates = st.multiselect("Covariate per il modello di Cox (opzionale):", options=[c for c in df.columns if c not in [time_col, event_col]])

# Tempi per S(t) e tabella a rischio
suggest = np.quantile(pd.to_numeric(df[time_col], errors="coerce").dropna(), [0.25,0.5,0.75])
tp_text = st.text_input("Tempi specifici (separati da virgola) per riportare S(t) e n a rischio (opzionale)", value=", ".join([f"{x:.2f}" for x in suggest]))

# Prepara dati
T = _numeric(df[time_col])
E = _as_event(df[event_col], event_value)
mask = ~(T.isna() | E.isna())
if group_var != "— Nessuna —":
    G = df[group_var]
    mask = mask & (~G.isna())
else:
    G = None

T = T[mask]; E = E[mask]
if group_var != "— Nessuna —":
    G = G[mask]

n_all = int(mask.sum())
n_events = int(E.sum())
n_cens = int(n_all - n_events)
st.markdown(f"**Campione analizzabile**: N={n_all} (Eventi={n_events}, Censure={n_cens})")

# =========================
#   Kaplan–Meier & Nelson–Aalen
# =========================
left, right = st.columns(2)

if not _HAS_LIFELINES:
    st.error("Per l’analisi di sopravvivenza completa è necessario installare `lifelines` (es.: `pip install lifelines`).")
else:
    with left:
        with st.spinner("Stimo le curve di sopravvivenza (Kaplan–Meier)…"):
            fig_km = go.Figure()
            if group_var == "— Nessuna —":
                kmf = KaplanMeierFitter()
                kmf.fit(T, event_observed=E, label="Totale")
                for tr in _km_plotly_from_kmf(kmf, name="Totale", show_ci=True):
                    fig_km.add_trace(tr)
            else:
                for g in _group_levels(G):
                    sel = (G == g)
                    if sel.sum() == 0: 
                        continue
                    kmf = KaplanMeierFitter()
                    kmf.fit(T[sel], event_observed=E[sel], label=str(g))
                    for tr in _km_plotly_from_kmf(kmf, name=str(g), show_ci=False):
                        fig_km.add_trace(tr)

            fig_km.update_layout(title="Curva di sopravvivenza (Kaplan–Meier)",
                                 xaxis_title=f"Tempo ({time_col})",
                                 yaxis_title="S(t)", yaxis=dict(range=[0,1]))
            st.plotly_chart(fig_km, use_container_width=True)

        st.caption(
            "La **curva KM** stima la probabilità di **sopravvivere oltre t** (S(t)). "
            "I **salti** avvengono agli eventi; le **censure** non causano salti ma riducono il numero a rischio. "
            "L’IC95% è mostrato quando si stima una sola curva."
        )

    with right:
        with st.spinner("Calcolo hazard cumulativa (Nelson–Aalen)…"):
            fig_na = go.Figure()
            if group_var == "— Nessuna —":
                naf = NelsonAalenFitter()
                naf.fit(T, event_observed=E, label="Totale")
                fig_na.add_trace(go.Scatter(x=naf.cumulative_hazard_.index.values,
                                            y=naf.cumulative_hazard_.iloc[:,0].values,
                                            mode="lines", name="Totale", line_shape="hv"))
            else:
                for g in _group_levels(G):
                    sel = (G == g)
                    if sel.sum() == 0:
                        continue
                    naf = NelsonAalenFitter()
                    naf.fit(T[sel], event_observed=E[sel], label=str(g))
                    fig_na.add_trace(go.Scatter(x=naf.cumulative_hazard_.index.values,
                                                y=naf.cumulative_hazard_.iloc[:,0].values,
                                                mode="lines", name=str(g), line_shape="hv"))
            fig_na.update_layout(title="Hazard cumulativa (Nelson–Aalen)",
                                 xaxis_title=f"Tempo ({time_col})",
                                 yaxis_title="H(t)")
            st.plotly_chart(fig_na, use_container_width=True)

        st.caption(
            "L’**hazard cumulativa** \(H(t)\) cresce con gli eventi; pendenze più ripide indicano **rischio istantaneo** maggiore. "
            "Relazione con la sopravvivenza: \(S(t)=e^{-H(t)}\)."
        )

    # Mediana di sopravvivenza e S(t) a tempi scelti
    st.markdown("### Riassunti numerici KM")
    with st.spinner("Calcolo mediana di sopravvivenza e S(t) a tempi specifici…"):
        rows = []
        timepoints = _parse_timepoints(tp_text)
        if group_var == "— Nessuna —":
            kmf = KaplanMeierFitter().fit(T, E)
            med = kmf.median_survival_time_
            ci_med = kmf.confidence_interval_ if hasattr(kmf, "confidence_interval_") else None
            med_lo = float(ci_med.iloc[0,0]) if ci_med is not None else np.nan
            med_hi = float(ci_med.iloc[0,1]) if ci_med is not None else np.nan
            row = {"Gruppo":"Totale", "Mediana": float(med), "CI 2.5%": med_lo, "CI 97.5%": med_hi}
            # S(t)
            for tt in timepoints:
                row[f"S({tt})"] = float(kmf.predict(tt))
            rows.append(row)
        else:
            for g in _group_levels(G):
                sel = (G == g)
                if sel.sum()==0: 
                    continue
                kmf = KaplanMeierFitter().fit(T[sel], E[sel])
                med = kmf.median_survival_time_
                row = {"Gruppo": str(g), "Mediana": float(med) if np.isfinite(med) else np.nan, "CI 2.5%": np.nan, "CI 97.5%": np.nan}
                for tt in timepoints:
                    row[f"S({tt})"] = float(kmf.predict(tt))
                rows.append(row)
        km_tbl = pd.DataFrame(rows).round(4)
        st.dataframe(km_tbl, use_container_width=True)
        st.caption("**Mediana di sopravvivenza**: tempo al quale S(t)=0.5. Se non raggiunta, la mediana risulta **NaN** o non definita.")

    # Numero a rischio a tempi specifici
    if len(timepoints) > 0:
        st.markdown("### Numero a rischio a tempi specifici")
        with st.spinner("Calcolo numero a rischio…"):
            risk_rows = []
            if group_var == "— Nessuna —":
                kmf = KaplanMeierFitter().fit(T, E)
                risk_rows.append(["Totale"] + _risk_table_from_kmf(kmf, timepoints))
            else:
                for g in _group_levels(G):
                    sel = (G == g)
                    if sel.sum()==0: continue
                    kmf = KaplanMeierFitter().fit(T[sel], E[sel])
                    risk_rows.append([str(g)] + _risk_table_from_kmf(kmf, timepoints))
            risk_tbl = pd.DataFrame(risk_rows, columns=["Gruppo"]+[f"t={tt}" for tt in timepoints])
            st.dataframe(risk_tbl, use_container_width=True)
            st.caption("**Numero a rischio**: soggetti ancora in osservazione immediatamente prima del tempo indicato.")

    # Log-rank test
    if group_var != "— Nessuna —":
        glv = _group_levels(G)
        if len(glv) >= 2:
            st.markdown("### Confronto tra curve: Log-rank test")
            with st.spinner("Eseguo il log-rank test…"):
                if len(glv) == 2:
                    g1, g2 = glv[0], glv[1]
                    res = logrank_test(T[G==g1], T[G==g2], event_observed_A=E[G==g1], event_observed_B=E[G==g2])
                    st.write(pd.DataFrame({"Statistic":[round(res.test_statistic,4)], "p-value":[res.p_value]}))
                else:
                    res = multivariate_logrank_test(T, G, E)
                    st.write(pd.DataFrame({"Statistic":[round(res.test_statistic,4)], "p-value":[res.p_value]}))
            st.caption(
                "Il **log-rank test** verifica l’ipotesi nulla di **curve di sopravvivenza uguali** tra gruppi. "
                "p-value piccolo → evidenza di differenza tra le curve."
            )

    # =========================
    #   Cox PH
    # =========================
    st.markdown("## Modello di Cox (Proportional Hazards)")
    if len(covariates) == 0 and group_var == "— Nessuna —":
        st.info("Selezioni almeno una covariata o una stratificazione per stimare un modello di Cox.")
    else:
        with st.spinner("Preparo il dataset per Cox…"):
            df_cox = pd.DataFrame({"T": T, "E": E})
            covs = list(covariates)
            if group_var != "— Nessuna —":
                covs = [group_var] + covs
            if len(covs) > 0:
                X = _design_matrix(df.loc[mask], covs)
                df_cox = pd.concat([df_cox, X], axis=1)
        robust = st.checkbox("Errori standard robusti (sandwich)", value=True)
        with st.spinner("Stimo il modello di Cox…"):
            try:
                cph = CoxPHFitter()
                cph.fit(df_cox, duration_col="T", event_col="E", robust=robust, show_progress=False)
                cph_sum = cph.summary.copy()
                # Tabella HR
                hr_tbl = cph_sum[["exp(coef)", "exp(coef) lower 95%", "exp(coef) upper 95%", "p"]].rename(
                    columns={"exp(coef)":"HR", "exp(coef) lower 95%":"CI 2.5%", "exp(coef) upper 95%":"CI 97.5%", "p":"p-value"}
                ).round(4)
                st.dataframe(hr_tbl, use_container_width=True)
                st.caption(
                    "**HR > 1**: rischio maggiore all’aumentare della covariata (o rispetto alla categoria di riferimento). "
                    "**HR < 1**: rischio minore. IC95% che **non** include 1 → effetto statisticamente significativo."
                )
                # c-index
                try:
                    cindex = float(cph.concordance_index_)
                    st.write(f"**Concordance index (c-index)**: {cindex:.3f}")
                    st.caption("Il **c-index** misura la capacità discriminativa del modello (0.5 casuale; 1 perfetto).")
                except Exception:
                    pass

                # Forest plot HR
                with st.spinner("Genero forest plot degli HR…"):
                    fig_forest = go.Figure()
                    ylabels = hr_tbl.index.tolist()[::-1]
                    HR = hr_tbl["HR"].values[::-1]
                    lo = hr_tbl["CI 2.5%"].values[::-1]
                    hi = hr_tbl["CI 97.5%"].values[::-1]
                    fig_forest.add_trace(go.Scatter(
                        x=HR, y=ylabels, mode="markers", name="HR", marker=dict(size=10)
                    ))
                    # barre di errore
                    fig_forest.add_trace(go.Scatter(
                        x=np.ravel(np.column_stack([lo, hi])), 
                        y=np.repeat(ylabels, 2),
                        mode="lines", showlegend=False
                    ))
                    fig_forest.add_vline(x=1.0, line=dict(dash="dash"))
                    fig_forest.update_layout(title="Forest plot — Hazard Ratio (IC95%)",
                                             xaxis_title="HR", yaxis_title="")
                    st.plotly_chart(fig_forest, use_container_width=True)
                st.caption("**Forest plot**: visualizza HR e IC95% per ogni covariata; la linea verticale a 1 indica **nessun effetto**.")

                # Test di PH (Schoenfeld)
                st.markdown("### Verifica assunzione di proporzionalità degli hazard")
                with st.spinner("Eseguo test sui residui di Schoenfeld…"):
                    ph_test = proportional_hazard_test(cph, df_cox, time_transform="rank")
                    ph_tbl = ph_test.summary.copy()
                    ph_tbl = ph_tbl[["test_statistic","p"]].rename(columns={"p":"p-value"}).round(4)
                    st.dataframe(ph_tbl, use_container_width=True)
                st.caption(
                    "L’assunzione di **hazard proporzionali** richiede che l’effetto delle covariate sia **costante nel tempo**. "
                    "p-value **piccolo** suggerisce violazione per la covariata corrispondente (considerare termini time-varying, stratificazione o AFT)."
                )
            except Exception as e:
                st.error(f"Impossibile stimare Cox: {e}")

    # =========================
    #   AFT (Weibull) opzionale
    # =========================
    st.markdown("## Modello AFT (Weibull) — Opzionale")
    aft_on = st.checkbox("Stima AFT (Weibull) con le stesse covariate", value=False)
    if aft_on:
        if len(covariates)==0 and group_var=="— Nessuna —":
            st.info("Selezioni almeno una covariata o una stratificazione per stimare l’AFT.")
        else:
            with st.spinner("Stimo Weibull AFT…"):
                try:
                    df_aft = pd.DataFrame({"T": T, "E": E})
                    covs = list(covariates)
                    if group_var != "— Nessuna —":
                        covs = [group_var] + covs
                    if len(covs)>0:
                        X = _design_matrix(df.loc[mask], covs)
                        df_aft = pd.concat([df_aft, X], axis=1)
                    aft = WeibullAFTFitter()
                    aft.fit(df_aft, duration_col="T", event_col="E")
                    aft_sum = aft.summary.copy()
                    # Time Ratio = exp(+coef) nella parametrizzazione di lifelines (sui log-tempi)
                    tr_tbl = aft_sum[["coef", "ci_lower_", "ci_upper_", "p"]].rename(
                        columns={"coef":"log(TR)", "ci_lower_":"CI 2.5% (log)", "ci_upper_":"CI 97.5% (log)", "p":"p-value"}
                    ).round(4)
                    tr_tbl["TR"] = np.exp(tr_tbl["log(TR)"])
                    tr_tbl["CI 2.5%"] = np.exp(tr_tbl["CI 2.5% (log)"])
                    tr_tbl["CI 97.5%"] = np.exp(tr_tbl["CI 97.5% (log)"])
                    st.dataframe(tr_tbl[["TR","CI 2.5%","CI 97.5%","p-value"]], use_container_width=True)
                    st.caption(
                        "**AFT (Accelerated Failure Time)**: **TR > 1** ⇒ tempi di sopravvivenza **più lunghi** (evento ritardato); "
                        "**TR < 1** ⇒ tempi **più brevi**. Modello utile quando l’assunzione PH non è soddisfatta."
                    )
                except Exception as e:
                    st.error(f"Impossibile stimare AFT: {e}")

# =========================
#   Export nel Results Summary
# =========================
st.markdown("### Esporta nel Results Summary")
if st.button("➕ Aggiungi analisi di sopravvivenza al Results Summary"):
    with st.spinner("Salvo i risultati nel Results Summary…"):
        if "report_items" not in st.session_state:
            st.session_state.report_items = []
        item = {
            "type":"survival_analysis",
            "title": f"Sopravvivenza — tempo: {time_col}, evento: {event_col} (= {event_value})",
            "content": {
                "N": int(n_all), "events": int(n_events), "censored": int(n_cens),
                "km_summary": km_tbl.to_dict(orient="records") if 'km_tbl' in locals() else None,
                "risk_table": risk_tbl.to_dict(orient="records") if 'risk_tbl' in locals() else None
            }
        }
        if _HAS_LIFELINES:
            # Aggiungi Cox/AFT se presenti
            if 'hr_tbl' in locals():
                item["content"]["cox_hr"] = hr_tbl.reset_index().rename(columns={"index":"term"}).to_dict(orient="records")
            if 'ph_tbl' in locals():
                item["content"]["ph_test"] = ph_tbl.reset_index().rename(columns={"index":"term"}).to_dict(orient="records")
            if 'tr_tbl' in locals():
                item["content"]["aft_tr"] = tr_tbl[["TR","CI 2.5%","CI 97.5%","p-value"]].reset_index().rename(columns={"index":"term"}).to_dict(orient="records")
        st.session_state.report_items.append(item)
    st.success("Analisi di sopravvivenza aggiunta al Results Summary.")

# =========================
#   Spiegazione didattica
# =========================
with st.expander("ℹ️ Spiegazione completa (lettura risultati)", expanded=False):
    st.markdown("""
**Kaplan–Meier (KM)**  
- Stima la **probabilità di sopravvivenza** oltre il tempo *t*, tenendo conto delle **censure** (soggetti che escono senza evento).  
- La **mediana di sopravvivenza** è il tempo in cui S(t)=0.5 (se raggiunta).  
- **Numero a rischio**: quanti sono ancora osservati appena prima di *t*.  
- **Confronto tra curve**: il **log-rank test** verifica se le curve sono uguali (ipotesi nulla). p-value piccolo → differenza tra gruppi.

**Hazard cumulativa (Nelson–Aalen)**  
- L’**hazard** è il rischio istantaneo di evento. L’integrale nel tempo produce l’**hazard cumulativa** \(H(t)\), con relazione \(S(t)=e^{-H(t)}\).  
- Curve con pendenza maggiore indicano rischio più elevato.

**Cox Proportional Hazards (PH)**  
- Modello semiparametrico: \\(h(t\\mid X)=h_0(t)\\exp(\\beta^\\top X)\\).  
- **Hazard Ratio (HR)** = \\(e^{\\beta}\\): HR>1 ⇒ rischio maggiore; HR<1 ⇒ rischio minore.  
- **c-index**: capacità discriminativa (0.5 casuale; 1 perfetta).  
- **Assunzione PH**: l’effetto delle covariate è costante nel tempo. Si verifica con test sui **residui di Schoenfeld**: p-value piccolo → possibile violazione (considerare stratificazione, termini time-varying o passare ad **AFT**).

**AFT (Weibull)**  
- Modello sui **tempi**: \\(\\log T = \\alpha + \\gamma^\\top X + \\sigma W\\).  
- **Time Ratio (TR)=e^{\\gamma}**: TR>1 ⇒ tempi più **lunghi** (evento ritardato); TR<1 ⇒ tempi **più brevi**.  
- Utile quando l’ipotesi PH non è adeguata.

**Buone pratiche**  
- Controlli sempre il **numero a rischio**: stime con pochissimi soggetti sono instabili.  
- Valuti PH: in caso di violazioni, consideri **stratificazione** o **covariate time-varying**.  
- Riporti **mediana** e **S(t)** a tempi clinicamente rilevanti con **IC** quando opportuno.
""")
