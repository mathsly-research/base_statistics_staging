# 12_📈_Longitudinale_Misure_Ripetute.py
from __future__ import annotations

import streamlit as st
import pandas as pd
import numpy as np

# Opzionali (grafici e modelli)
try:
    import plotly.express as px
except Exception:
    px = None
try:
    import statsmodels.formula.api as smf
    import statsmodels.api as sm
except Exception:
    smf, sm = None, None

# ------------------------------
# Configurazione pagina
# ------------------------------
st.set_page_config(page_title="Misure ripetute (longitudinale)", layout="wide")

# ------------------------------
# Utility chiavi univoche e stato sicuro
# ------------------------------
KEY_PREFIX = "lr"  # longitudinal repeated

def k(name: str) -> str:
    return f"{KEY_PREFIX}_{name}"

def k_sec(section: str, name: str) -> str:
    return f"{KEY_PREFIX}_{section}_{name}"

def ss_get(name: str, default=None):
    return st.session_state.get(name, default)

def ss_set_default(name: str, value):
    if name not in st.session_state:
        st.session_state[name] = value

# Inizializzazioni sicure
ss_set_default(k("initialized"), True)
ss_set_default(k("id_col"), None)
ss_set_default(k("time_col"), None)
ss_set_default(k("value_col"), None)
ss_set_default(k("work_df"), None)
ss_set_default(k("format_ok"), False)
ss_set_default(k("active_filter"), None)

# ------------------------------
# Header pulito
# ------------------------------
st.title("📈 Analisi Longitudinale – Misure ripetute")
st.markdown(
    """
**Obiettivo:** portare i dati nel **formato long** (una riga per *Soggetto × Tempo*) con tre colonne:
1) **ID soggetto** · 2) **Tempo/Visita** · 3) **Valore (esito)**.
"""
)

# =============================================================================
# PASSO 1 · DATI
# =============================================================================
with st.container():
    st.subheader("Passo 1 · Dati")
    # Recupero silenzioso da session_state (senza testo sulla pagina di upload)
    df, found_key = None, None
    for key in ["uploaded_df", "df_upload", "main_df", "dataset", "df"]:
        if key in st.session_state and isinstance(st.session_state[key], pd.DataFrame):
            df, found_key = st.session_state[key], key
            break

    cols_head = st.columns([3, 2])
    with cols_head[0]:
        if df is not None and not df.empty:
            st.success(f"Dataset trovato: {df.shape[0]} righe × {df.shape[1]} colonne.")
            st.dataframe(df.head(20), use_container_width=True)
        else:
            st.warning("Nessun dataset trovato. Carichi un file qui sotto (solo per questa pagina).")
    with cols_head[1]:
        temp_file = st.file_uploader(
            "Carica CSV/XLSX (opzionale)",
            type=["csv", "xlsx", "xls"],
            key=k("temp_uploader"),
            help="Se non ha caricato i dati nella pagina iniziale, può caricarli qui."
        )
        if temp_file is not None:
            try:
                if temp_file.name.lower().endswith(".csv"):
                    df = pd.read_csv(temp_file)
                else:
                    df = pd.read_excel(temp_file)
                st.success(f"File caricato: {df.shape[0]} × {df.shape[1]}")
                st.dataframe(df.head(20), use_container_width=True)
            except Exception as e:
                st.error(f"Errore caricamento: {e}")

if df is None or df.empty:
    st.stop()

# =============================================================================
# PASSO 2 · PREPARAZIONE (portare al formato long)
# =============================================================================
st.subheader("Passo 2 · Preparazione")
st.markdown(
    "Porti i dati nel **formato long**. Se già long, selezioni le colonne corrette; se wide, converta in long."
)

with st.expander("Verifica rapida e selezione colonne (se già LONG)", expanded=True):
    cols = list(df.columns)

    # Tentativi di riconoscimento
    guess_id = guess_time = guess_val = None
    for c in cols:
        cl = c.lower()
        if guess_id is None and any(t in cl for t in ["id", "soggetto", "subject", "patient", "pt", "case"]):
            guess_id = c
        if guess_time is None and any(t in cl for t in ["time", "tempo", "visit", "visita", "giorno", "day", "t_"]):
            guess_time = c
        if guess_val is None and any(t in cl for t in ["val", "value", "esito", "outcome", "y", "score", "measure"]):
            guess_val = c

    c1, c2, c3 = st.columns(3)
    with c1:
        sel_id = st.selectbox(
            "Colonna ID soggetto",
            options=cols,
            index=(cols.index(guess_id) if guess_id in cols else 0),
            key=k("sel_id_try"),
        )
    with c2:
        sel_time = st.selectbox(
            "Colonna Tempo/Visita",
            options=cols,
            index=(cols.index(guess_time) if guess_time in cols else min(1, len(cols)-1)),
            key=k("sel_time_try"),
        )
    with c3:
        sel_val = st.selectbox(
            "Colonna Valore (esito)",
            options=cols,
            index=(cols.index(guess_val) if guess_val in cols else min(2, len(cols)-1)),
            key=k("sel_val_try"),
        )

    # Long valido se 3 colonne distinte e una sola riga per (ID,Tempo)
    long_candidate = False
    if len({sel_id, sel_time, sel_val}) == 3:
        try:
            grp = df[[sel_id, sel_time]].value_counts()
            long_candidate = (grp.max() == 1)
        except Exception:
            long_candidate = False

    info_col = st.columns([1, 1, 1, 1])
    with info_col[0]:
        cast_time = st.checkbox("Tempo numerico (se possibile)", value=False, key=k("A_cast_time"))
    with info_col[1]:
        drop_na = st.checkbox("Elimina righe NA", value=True, key=k("A_drop_na"))
    with info_col[2]:
        dedup = st.selectbox("Duplicati (ID,Tempo)", ["Mantieni primo", "Media"], key=k("A_dedup"))
    with info_col[3]:
        run_A = st.button("Usa queste colonne (LONG)", key=k("A_run"), use_container_width=True)

    if run_A:
        try:
            work = df[[sel_id, sel_time, sel_val]].copy()
            work.columns = ["__ID__", "__Tempo__", "__Valore__"]

            if cast_time:
                as_num = pd.to_numeric(work["__Tempo__"], errors="coerce")
                if not as_num.isna().all():
                    work["__Tempo__"] = as_num

            if drop_na:
                work = work.dropna(subset=["__ID__", "__Tempo__", "__Valore__"])

            if dedup == "Media":
                work = (work.groupby(["__ID__", "__Tempo__"], as_index=False)
                              .agg(__Valore__=("__Valore__", "mean")))
            else:
                work = work.drop_duplicates(subset=["__ID__", "__Tempo__"], keep="first")

            # Salvataggio stato con i nomi scelti dall'utente
            st.session_state[k("work_df")] = work.rename(
                columns={"__ID__": sel_id, "__Tempo__": sel_time, "__Valore__": sel_val}
            )
            st.session_state[k("id_col")] = sel_id
            st.session_state[k("time_col")] = sel_time
            st.session_state[k("value_col")] = sel_val
            st.session_state[k("format_ok")] = True

            st.success("Dataset LONG impostato correttamente.")
            st.dataframe(st.session_state[k("work_df")].head(20), use_container_width=True)
        except Exception as e:
            st.error(f"Errore nella preparazione LONG: {e}")

with st.expander("Conversione da WIDE a LONG", expanded=not ss_get(k("format_ok"), False)):
    cols = list(df.columns)
    c1, c2 = st.columns([1, 2])
    with c1:
        id_col_b = st.selectbox("Colonna ID", options=cols, key=k("B_id"))
    with c2:
        time_like = [c for c in cols if c != id_col_b]
        time_cols_b = st.multiselect(
            "Colonne di misura (tempi/visite)",
            options=time_like,
            default=time_like[: min(5, len(time_like))],
            key=k("B_time_cols"),
        )

    t1, t2 = st.columns(2)
    with t1:
        strip_prefix = st.text_input("Rimuovi prefisso da Tempo (opz.)", value="", key=k("B_strip_prefix"))
    with t2:
        to_numeric = st.checkbox("Tempo numerico (se possibile)", value=True, key=k("B_to_numeric"))

    d1, d2 = st.columns(2)
    with d1:
        drop_na_b = st.checkbox("Elimina righe NA", value=True, key=k("B_drop_na"))
    with d2:
        dedup_b = st.selectbox("Duplicati (ID,Tempo)", ["Mantieni primo", "Media"], key=k("B_dedup"))

    if st.button("Converte WIDE → LONG", key=k("B_run"), use_container_width=True):
        if not time_cols_b:
            st.error("Selezioni almeno una colonna di misura.")
        else:
            try:
                work = df.melt(
                    id_vars=[id_col_b],
                    value_vars=time_cols_b,
                    var_name="__Tempo__",
                    value_name="__Valore__",
                )
                if strip_prefix:
                    work["__Tempo__"] = work["__Tempo__"].astype(str).str.replace(f"^{strip_prefix}", "", regex=True)
                if to_numeric:
                    as_num = pd.to_numeric(work["__Tempo__"], errors="coerce")
                    if not as_num.isna().all():
                        work["__Tempo__"] = as_num
                if drop_na_b:
                    work = work.dropna(subset=[id_col_b, "__Tempo__", "__Valore__"])

                if dedup_b == "Media":
                    work = (work.groupby([id_col_b, "__Tempo__"], as_index=False)
                                  .agg(__Valore__=("__Valore__", "mean")))
                else:
                    work = work.drop_duplicates(subset=[id_col_b, "__Tempo__"], keep="first")

                # Standard: usiamo nomi neutri interni
                work = work.rename(columns={id_col_b: "__ID__"})
                st.session_state[k("work_df")] = work.rename(
                    columns={"__ID__": "__ID__", "__Tempo__": "__Tempo__", "__Valore__": "__Valore__"}
                )
                st.session_state[k("id_col")] = "__ID__"
                st.session_state[k("time_col")] = "__Tempo__"
                st.session_state[k("value_col")] = "__Valore__"
                st.session_state[k("format_ok")] = True

                st.success("Conversione WIDE → LONG completata.")
                st.dataframe(st.session_state[k("work_df")].head(20), use_container_width=True)
            except Exception as e:
                st.error(f"Errore nella conversione: {e}")

# Controlli qualità sintetici
if ss_get(k("format_ok"), False) and ss_get(k("work_df")) is not None:
    st.subheader("Passo 3 · Analisi")
    st.markdown("Opzionalmente, applichi piccoli controlli prima dei grafici e dei modelli.")

    with st.expander("Controlli rapidi (facoltativi)", expanded=False):
        work = st.session_state[k("work_df")].copy()
        id_col = st.session_state[k("id_col")]
        time_col = st.session_state[k("time_col")]
        val_col = st.session_state[k("value_col")]

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            handle_missing = st.selectbox("Missing su Valore", ["Nessuna azione", "Drop righe NA"], key=k("QC_missing"))
        with c2:
            time_type = st.selectbox("Tipo di Tempo", ["Automatico", "Numerico", "Categoriale ordinato"], key=k("QC_ttype"))
        with c3:
            sort_by_time = st.checkbox("Ordina per ID e Tempo", value=True, key=k("QC_sort"))
        with c4:
            check_dups = st.checkbox("Aggrega duplicati (media)", value=False, key=k("QC_dedup"))

        if st.button("Applica", key=k("QC_apply"), use_container_width=True):
            try:
                if handle_missing == "Drop righe NA":
                    work = work.dropna(subset=[id_col, time_col, val_col])

                if time_type == "Numerico":
                    as_num = pd.to_numeric(work[time_col], errors="coerce")
                    if not as_num.isna().all():
                        work[time_col] = as_num
                elif time_type == "Categoriale ordinato":
                    work[time_col] = pd.Categorical(work[time_col], ordered=True)

                if check_dups:
                    work = (work.groupby([id_col, time_col], as_index=False)
                                  .agg(**{val_col: (val_col, "mean")}))

                if sort_by_time:
                    try:
                        work = work.sort_values(by=[id_col, time_col])
                    except Exception:
                        work = work.sort_values(by=[id_col])

                st.session_state[k("work_df")] = work
                st.success("Controlli applicati.")
                st.dataframe(work.head(30), use_container_width=True)
            except Exception as e:
                st.error(f"Errore durante i controlli: {e}")

    # ---------------------------
    # Tabs: Filtri · Grafici · Modelli
    # ---------------------------
    tab_desc, tab_plot, tab_model = st.tabs(["Filtri", "Grafici", "Modelli"])

    # FILTRI
    with tab_desc:
        st.subheader("Filtri (facoltativi)")
        work = st.session_state[k("work_df")].copy()
        # Colonne categoriali (escludiamo ID e Tempo)
        cat_cols = [
            c for c in work.columns
            if (work[c].dtype == "object" or str(work[c].dtype).startswith("category"))
            and c not in [st.session_state[k("id_col")], st.session_state[k("time_col")]]
        ]

        with st.form(key=k_sec("tab1", "filters_form")):
            use_filter = st.checkbox("Attiva filtro su una colonna categoriale", value=False, key=k_sec("tab1", "use_filter"))
            if use_filter and cat_cols:
                fcol = st.selectbox("Colonna", options=cat_cols, key=k_sec("tab1", "fcol"))
                vals = sorted(work[fcol].dropna().unique().tolist())
                fvals = st.multiselect("Valori da includere", options=vals, default=vals[:1], key=k_sec("tab1", "fvals"))
            else:
                fcol, fvals = None, None
            submitted = st.form_submit_button("Applica filtro", use_container_width=True)

        if submitted:
            if use_filter and fcol and fvals:
                st.session_state[k("active_filter")] = (fcol, fvals)
            else:
                st.session_state[k("active_filter")] = None

        active_filter = ss_get(k("active_filter"))
        if active_filter:
            fcol, fvals = active_filter
            st.info(f"Filtro attivo: **{fcol}** in {fvals}")
        else:
            st.info("Nessun filtro attivo.")

    # Helper per applicare il filtro (sicuro)
    def filtered_df(df_in: pd.DataFrame) -> pd.DataFrame:
        af = ss_get(k("active_filter"))
        if af:
            col, vals = af
            if col in df_in.columns:
                return df_in[df_in[col].isin(vals)]
        return df_in

    # GRAFICI
    with tab_plot:
        st.subheader("Grafici descrittivi")
        work = filtered_df(st.session_state[k("work_df")].copy())
        id_col = st.session_state[k("id_col")]
        t_col = st.session_state[k("time_col")]
        y_col = st.session_state[k("value_col")]

        with st.expander("Opzioni grafico", expanded=True):
            agg_fun = st.selectbox("Aggregazione per tempo", ["media", "mediana"], key=k_sec("tab2", "agg"))
            show_ci = st.checkbox("Intervallo ±1.96·SE (solo media)", value=True, key=k_sec("tab2", "ci"))
            max_ids = st.number_input("Linee individuali: massimo N soggetti (0 = nessuna)", min_value=0, value=0, step=1, key=k_sec("tab2", "nids"))

        # Linee individuali
        if max_ids and max_ids > 0 and px is not None:
            ids = work[id_col].dropna().unique().tolist()[:max_ids]
            w_ind = work[work[id_col].isin(ids)].copy()
            try:
                w_ind = w_ind.sort_values(by=[id_col, t_col])
            except Exception:
                w_ind = w_ind.sort_values(by=[id_col])
            fig_ind = px.line(w_ind, x=t_col, y=y_col, color=id_col, markers=True)
            fig_ind.update_layout(height=420)
            st.plotly_chart(fig_ind, use_container_width=True)

        # Aggregazione
        if agg_fun == "media":
            g = work.groupby(t_col)[y_col].agg(["mean", "count", "std"]).reset_index()
            g.rename(columns={"mean": "media", "count": "n", "std": "sd"}, inplace=True)
            if show_ci:
                se = g["sd"] / np.sqrt(np.maximum(g["n"], 1))
                g["low"] = g["media"] - 1.96 * se
                g["high"] = g["media"] + 1.96 * se
        else:
            g = work.groupby(t_col)[y_col].median().reset_index().rename(columns={y_col: "mediana"})

        if px is not None:
            if agg_fun == "media":
                fig = px.line(g, x=t_col, y="media", markers=True)
                if show_ci and "low" in g.columns:
                    fig.add_scatter(x=g[t_col], y=g["low"], mode="lines", name="low", showlegend=False)
                    fig.add_scatter(x=g[t_col], y=g["high"], mode="lines", name="high", showlegend=False)
            else:
                fig = px.line(g, x=t_col, y="mediana", markers=True)
            fig.update_layout(height=460)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Plotly non disponibile nell'ambiente.")

    # MODELLI
    with tab_model:
        st.subheader("Modelli (dimostrativi)")
        if smf is None or sm is None:
            st.info("`statsmodels` non disponibile: saltiamo i modelli.")
        else:
            work = filtered_df(st.session_state[k("work_df")])[[st.session_state[k("id_col")], st.session_state[k("time_col")], st.session_state[k("value_col")]]].dropna().copy()
            id_col = st.session_state[k("id_col")]
            t_col = st.session_state[k("time_col")]
            y_col = st.session_state[k("value_col")]

            with st.form(key=k_sec("tab3", "form")):
                model_type = st.selectbox(
                    "Selezioni il modello",
                    ["ANOVA per misure ripetute (semplificata)", "Linear Mixed Model (random intercept)"],
                    key=k_sec("tab3", "mtype")
                )
                submit = st.form_submit_button("Esegui", use_container_width=True)

            if submit:
                try:
                    work[t_col] = work[t_col].astype("category")
                    if model_type.startswith("ANOVA"):
                        formula = f"`{y_col}` ~ C(`{t_col}`)"
                        fit = smf.ols(formula=formula, data=work).fit(
                            cov_type="cluster", cov_kwds={"groups": work[id_col]}
                        )
                        st.markdown(f"**Formula:** `{formula}`")
                        st.text(fit.summary().as_text())
                    else:
                        formula = f"`{y_col}` ~ C(`{t_col}`)"
                        md = sm.MixedLM.from_formula(formula, data=work, groups=work[id_col])
                        mdf = md.fit(method="lbfgs", reml=True)
                        st.markdown(f"**Formula:** `{formula} + (1|ID)`")
                        st.text(mdf.summary().as_text())
                except Exception as e:
                    st.error(f"Errore nel fitting del modello: {e}")
