# 12_📈_Longitudinale_Misure_Ripetute.py
from __future__ import annotations

import streamlit as st
import pandas as pd
import numpy as np

# Opzionali: attivi funzioni avanzate (grafici/LMEM)
try:
    import plotly.express as px
except Exception:
    px = None

try:
    import statsmodels.formula.api as smf
except Exception:
    smf = None

# -----------------------------------------
# Config pagina
# -----------------------------------------
st.set_page_config(page_title="Longitudinale · Misure ripetute", layout="wide")

# -----------------------------------------
# Utility per chiavi univoche
# -----------------------------------------
KEY_PREFIX = "lr"  # lr = longitudinal repeated

def k(name: str) -> str:
    """Genera una chiave univoca con prefisso di pagina."""
    return f"{KEY_PREFIX}_{name}"

def k_sec(section: str, name: str) -> str:
    """Chiave univoca per widget 'simili' ripetuti in sezioni/tab diverse."""
    return f"{KEY_PREFIX}_{section}_{name}"

# -----------------------------------------
# Guard di inizializzazione stato
# -----------------------------------------
if k("initialized") not in st.session_state:
    st.session_state[k("initialized")] = True
    st.session_state[k("id_col")] = None
    st.session_state[k("time_col")] = None
    st.session_state[k("value_col")] = None
    st.session_state[k("is_long_format")] = True  # assunzione iniziale
    st.session_state[k("filter_enabled")] = False

# -----------------------------------------
# Header
# -----------------------------------------
st.title("📈 Analisi Longitudinale / Misure ripetute")

st.markdown(
    """
Questa pagina consente di configurare un dataset longitudinale (ID soggetto, tempo/misura, variabile di esito), 
visualizzare grafici descrittivi e avviare modelli statistici di base.  
Le chiavi dei widget sono **tutte esplicite e univoche** per prevenire `StreamlitDuplicateElementId`.
"""
)

# -----------------------------------------
# Caricamento dati
# -----------------------------------------
with st.expander("1) Carica dati", expanded=True):
    file = st.file_uploader(
        "Carica un file CSV o Excel",
        type=["csv", "xlsx", "xls"],
        key=k("uploader"),
        help="Il file dovrebbe contenere almeno colonne per ID soggetto, tempo e variabile di esito."
    )

    df: pd.DataFrame | None = None
    if file is not None:
        try:
            if file.name.lower().endswith(".csv"):
                df = pd.read_csv(file)
            else:
                df = pd.read_excel(file)
        except Exception as e:
            st.error(f"Errore nel caricamento: {e}")

    if df is not None:
        st.success(f"Dati caricati: {df.shape[0]} righe × {df.shape[1]} colonne")
        st.dataframe(df.head(20), use_container_width=True)
    else:
        st.info("Carichi i dati per procedere.")

# -----------------------------------------
# Configurazione colonne (ID, tempo, outcome)
# -----------------------------------------
if df is not None and not df.empty:
    with st.expander("2) Configurazione struttura dati", expanded=True):

        st.radio(
            "Formato del dataset",
            options=["Lungo (long)", "Largo (wide)"],
            index=0 if st.session_state[k("is_long_format")] else 1,
            key=k("format_radio"),
            help="In formato long: colonne = ID, Tempo, Valore. In formato wide: una colonna ID e più colonne di tempo/misura.",
        )
        st.session_state[k("is_long_format")] = (st.session_state[k("format_radio")] == "Lungo (long)")

        # -- Selettori principali, unici nella pagina (non replicati in altri tab) --
        if st.session_state[k("is_long_format")]:
            cols = st.columns(3)
            with cols[0]:
                st.session_state[k("id_col")] = st.selectbox(
                    "ID soggetto",
                    options=list(df.columns),
                    index=0,
                    key=k("id_col_select"),
                )
            with cols[1]:
                st.session_state[k("time_col")] = st.selectbox(
                    "Tempo / Visita",
                    options=list(df.columns),
                    index=min(1, len(df.columns)-1),
                    key=k("time_col_select"),
                )
            with cols[2]:
                st.session_state[k("value_col")] = st.selectbox(
                    "Variabile di esito",
                    options=list(df.columns),
                    index=min(2, len(df.columns)-1),
                    key=k("value_col_select"),
                )
        else:
            # Formato largo: si seleziona ID e le colonne di misure ripetute
            cols = st.columns([1, 2])
            with cols[0]:
                st.session_state[k("id_col")] = st.selectbox(
                    "ID soggetto",
                    options=list(df.columns),
                    key=k("id_col_wide"),
                )
            with cols[1]:
                time_like = [c for c in df.columns if c != st.session_state[k("id_col")]]
                time_cols_sel = st.multiselect(
                    "Colonne delle misure (tempi/visite)",
                    options=time_like,
                    default=time_like[:min(3, len(time_like))],
                    key=k("time_cols_wide"),
                )
            # Melt in long per analisi successive
            if st.session_state.get(k("id_col")) and time_cols_sel:
                try:
                    df = df.melt(
                        id_vars=[st.session_state[k("id_col")]],
                        value_vars=time_cols_sel,
                        var_name="__tempo__",
                        value_name="__valore__",
                    )
                    st.session_state[k("time_col")] = "__tempo__"
                    st.session_state[k("value_col")] = "__valore__"
                    st.success("Il dataset è stato convertito in formato long per le analisi successive.")
                    st.dataframe(df.head(20), use_container_width=True)
                except Exception as e:
                    st.error(f"Errore nel reshape (wide→long): {e}")

        # Validazione
        id_ok = st.session_state.get(k("id_col")) in (df.columns if df is not None else [])
        time_ok = st.session_state.get(k("time_col")) in (df.columns if df is not None else [])
        val_ok = st.session_state.get(k("value_col")) in (df.columns if df is not None else [])
        if not (id_ok and time_ok and val_ok):
            st.warning("Selezioni non ancora complete o non valide.")
        else:
            st.success(f"Configurazione OK → ID = `{st.session_state[k('id_col')]}`, "
                       f"Tempo = `{st.session_state[k('time_col')]}`, "
                       f"Esito = `{st.session_state[k('value_col')]}`")

# -----------------------------------------
# Tabs principali
# -----------------------------------------
if df is not None and not df.empty and all(
    st.session_state.get(k(x)) in df.columns for x in ["id_col", "time_col", "value_col"]
):
    tab_desc, tab_plot, tab_model = st.tabs(["Impostazioni & Filtri", "Grafici", "Modelli"])

    # ---------------------------
    # TAB 1: Impostazioni & Filtri
    # ---------------------------
    with tab_desc:
        st.subheader("Impostazioni & Filtri")

        with st.form(key=k_sec("tab1", "filters_form")):
            st.checkbox(
                "Attiva filtro per sottogruppi",
                value=st.session_state.get(k("filter_enabled"), False),
                key=k_sec("tab1", "filter_enabled"),
            )

            # Esempio: l'utente può scegliere una colonna categoriale per filtrare
            cat_cols = [c for c in df.columns if df[c].dtype == "object" or str(df[c].dtype).startswith("category")]
            if cat_cols and st.session_state[k_sec("tab1", "filter_enabled")]:
                filt_col = st.selectbox(
                    "Colonna categoriale per il filtro",
                    options=cat_cols,
                    key=k_sec("tab1", "filt_col"),
                )
                # valori possibili
                vals = sorted(df[filt_col].dropna().unique().tolist())
                sel_vals = st.multiselect(
                    "Valori inclusi",
                    options=vals,
                    default=vals[:1],
                    key=k_sec("tab1", "filt_vals"),
                )
            else:
                filt_col, sel_vals = None, None

            submitted = st.form_submit_button("Applica impostazioni", use_container_width=True)

        # Applica filtro se richiesto
        if submitted:
            st.session_state[k("filter_enabled")] = st.session_state[k_sec("tab1", "filter_enabled")]
            if st.session_state[k("filter_enabled")] and filt_col and sel_vals:
                st.session_state[k("active_filter")] = (filt_col, sel_vals)
            else:
                st.session_state[k("active_filter")] = None

        if st.session_state.get(k("active_filter")):
            filt_col, sel_vals = st.session_state[k("active_filter")]
            st.info(f"Filtro attivo: **{filt_col}** in {sel_vals}")
        else:
            st.info("Nessun filtro attivo.")

    # ---------------------------
    # Funzione helper per ottenere df analizzabile (con filtro)
    # ---------------------------
    def get_work_df() -> pd.DataFrame:
        work = df.copy()
        if st.session_state.get(k("active_filter")):
            col, vals = st.session_state[k("active_filter")]
            work = work[work[col].isin(vals)]
        return work

    # ---------------------------
    # TAB 2: Grafici
    # ---------------------------
    with tab_plot:
        st.subheader("Grafici descrittivi")
        work = get_work_df()

        # Parametri grafico (chiavi specifiche della tab)
        with st.expander("Opzioni grafico", expanded=True):
            agg_fun = st.selectbox(
                "Aggregazione per tempo (linea media ± errore)",
                options=["media", "mediana"],
                key=k_sec("tab2", "agg_fun"),
            )
            show_ci = st.checkbox("Mostra intervallo (bootstrap semplice)", value=True, key=k_sec("tab2", "show_ci"))
            max_ids = st.number_input("Mostra al massimo N soggetti (per linee individuali)", min_value=0, value=0, step=1, key=k_sec("tab2", "max_ids"))

        id_col = st.session_state[k("id_col")]
        t_col = st.session_state[k("time_col")]
        y_col = st.session_state[k("value_col")]

        # Prova conversione del tempo a ordinabile
        t_vals = work[t_col]
        if not np.issubdtype(t_vals.dtype, np.number):
            # Tentativo di parsing numerico o categoriale ordinato
            try:
                work["__t__"] = pd.to_numeric(work[t_col], errors="coerce")
                if work["__t__"].isna().all():
                    work["__t__"] = pd.Categorical(work[t_col], ordered=True).codes
            except Exception:
                work["__t__"] = pd.Categorical(work[t_col], ordered=True).codes
            t_plot = "__t__"
        else:
            t_plot = t_col

        # Linee individuali (facoltative)
        if max_ids and max_ids > 0:
            ids = work[id_col].dropna().unique().tolist()[:max_ids]
            w_ind = work[work[id_col].isin(ids)]
            if px is not None:
                fig_ind = px.line(
                    w_ind.sort_values(by=[id_col, t_plot]),
                    x=t_plot,
                    y=y_col,
                    color=id_col,
                    markers=True,
                )
                fig_ind.update_layout(height=450)
                st.plotly_chart(fig_ind, use_container_width=True)
            else:
                st.warning("Plotly non disponibile: impossibile mostrare le linee individuali.")

        # Aggregato per tempo
        if agg_fun == "media":
            g = work.groupby(t_plot)[y_col].agg(["mean", "count", "std"]).reset_index()
            g.rename(columns={"mean": "media", "count": "n", "std": "sd"}, inplace=True)
            if show_ci:
                se = g["sd"] / np.sqrt(np.maximum(g["n"], 1))
                g["low"] = g["media"] - 1.96 * se
                g["high"] = g["media"] + 1.96 * se
        else:
            g = work.groupby(t_plot)[y_col].median().reset_index().rename(columns={y_col: "mediana"})

        if px is not None:
            if agg_fun == "media":
                fig = px.line(g, x=t_plot, y="media", markers=True)
                if show_ci and "low" in g.columns:
                    fig.add_scatter(x=g[t_plot], y=g["low"], mode="lines", name="low", showlegend=False)
                    fig.add_scatter(x=g[t_plot], y=g["high"], mode="lines", name="high", showlegend=False)
            else:
                fig = px.line(g, x=t_plot, y="mediana", markers=True)
            fig.update_layout(height=480)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Plotly non disponibile: impossibile mostrare il grafico aggregato.")

    # ---------------------------
    # TAB 3: Modelli
    # ---------------------------
    with tab_model:
        st.subheader("Modelli statistici (dimostrativi)")

        if smf is None:
            st.warning("`statsmodels` non è disponibile nell'ambiente: saltiamo l'analisi dei modelli.")
        else:
            work = get_work_df()

            # Opzioni modello (chiavi specifiche della tab)
            with st.form(key=k_sec("tab3", "model_form")):
                model_type = st.selectbox(
                    "Tipo di modello",
                    options=["ANOVA per misure ripetute (semplificata)", "LM (random intercept ~ soggetto)"],
                    key=k_sec("tab3", "model_type"),
                )
                submit_model = st.form_submit_button("Esegui modello", use_container_width=True)

            id_col = st.session_state[k("id_col")]
            t_col = st.session_state[k("time_col")]
            y_col = st.session_state[k("value_col")]

            if submit_model:
                # Preparazione dati
                data_m = work[[id_col, t_col, y_col]].dropna().copy()

                # Prova a trattare il tempo come categoriale
                data_m[t_col] = data_m[t_col].astype("category")

                try:
                    if model_type == "ANOVA per misure ripetute (semplificata)":
                        # Approccio semplice: OLS con tempo categoriale + cluster-robust SE per soggetto
                        formula = f"`{y_col}` ~ C(`{t_col}`)"
                        model = smf.ols(formula=formula, data=data_m).fit(
                            cov_type="cluster",
                            cov_kwds={"groups": data_m[id_col]}
                        )
                        st.write("**Formula:**", formula)
                        st.write(model.summary().as_text())
                    else:
                        # LM con random intercept via mixedlm
                        import statsmodels.api as sm
                        formula = f"`{y_col}` ~ C(`{t_col}`)"
                        md = sm.MixedLM.from_formula(
                            formula, data=data_m, groups=data_m[id_col]
                        )
                        mdf = md.fit(method="lbfgs", reml=True)
                        st.write("**Formula:**", formula + " + (1|ID)")
                        st.write(mdf.summary().as_text())
                except Exception as e:
                    st.error(f"Errore nel fitting del modello: {e}")

# -----------------------------------------
# Footer tecnico
# -----------------------------------------
with st.expander("Dettagli tecnici", expanded=False):
    st.caption(
        "Tutti i widget usano chiavi esplicite con prefisso `lr_` (o `lr_<tab>_...`) per prevenire "
        "`StreamlitDuplicateElementId`. Le scelte primarie (ID, Tempo, Esito) sono dichiarate una sola volta "
        "e memorizzate in `st.session_state`, poi riutilizzate nelle tab per evitare la ricreazione dello stesso widget."
    )
