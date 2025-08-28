# 12_📈_Longitudinale_Misure_Ripetute.py
from __future__ import annotations

import streamlit as st
import pandas as pd
import numpy as np

# Opzionali
try:
    import plotly.express as px
except Exception:
    px = None
try:
    import statsmodels.formula.api as smf
    import statsmodels.api as sm
except Exception:
    smf = None
    sm = None

st.set_page_config(page_title="Longitudinale · Misure ripetute", layout="wide")

# ------------------------------
# Utility chiavi univoche
# ------------------------------
KEY_PREFIX = "lr"

def k(name: str) -> str:
    return f"{KEY_PREFIX}_{name}"

def k_sec(section: str, name: str) -> str:
    return f"{KEY_PREFIX}_{section}_{name}"

# ------------------------------
# Stato iniziale
# ------------------------------
if k("initialized") not in st.session_state:
    st.session_state[k("initialized")] = True
    st.session_state[k("id_col")] = None
    st.session_state[k("time_col")] = None
    st.session_state[k("value_col")] = None
    st.session_state[k("work_df")] = None
    st.session_state[k("format_ok")] = False
    st.session_state[k("active_filter")] = None

st.title("📈 Analisi Longitudinale / Misure ripetute")

st.markdown(
    """
**Formato richiesto (long):** il dataset deve contenere **una riga per soggetto e tempo**, con tre campi:
- **ID soggetto** (identificativo univoco del soggetto),
- **Tempo/Visita** (ordine o etichetta della misura ripetuta, numerica o categoriale ordinata),
- **Variabile di esito** (valore misurato).

Se i dati non sono in questo formato (ad es. **wide**, con più colonne una per ciascun tempo), può usare gli **strumenti di preparazione** sottostanti per convertirli.
"""
)

# ------------------------------------------------
# 1) Recupero dataset dalla pagina di upload (0_📂)
# ------------------------------------------------
with st.expander("1) Origine dati", expanded=True):
    st.caption("La pagina prova a recuperare automaticamente i dati caricati nella pagina **0_📂 Upload Dataset**.")
    possible_keys = ["uploaded_df", "df_upload", "main_df", "dataset", "df"]
    df = None
    found_key = None
    for key in possible_keys:
        if key in st.session_state and isinstance(st.session_state[key], pd.DataFrame):
            df = st.session_state[key]
            found_key = key
            break

    if df is not None:
        st.success(f"Dati trovati in `st.session_state['{found_key}']`: {df.shape[0]} righe × {df.shape[1]} colonne.")
        st.dataframe(df.head(20), use_container_width=True)
    else:
        st.warning("Nessun dataset trovato nelle chiavi attese di `st.session_state`.")
        st.info("Come alternativa, carichi qui temporaneamente un file per test (CSV/Excel).")
        temp_file = st.file_uploader("Carica CSV/XLSX (opzionale, per test)", type=["csv", "xlsx", "xls"], key=k("temp_uploader"))
        if temp_file is not None:
            try:
                if temp_file.name.lower().endswith(".csv"):
                    df = pd.read_csv(temp_file)
                else:
                    df = pd.read_excel(temp_file)
                st.success(f"Caricato file di test: {df.shape[0]} × {df.shape[1]}")
                st.dataframe(df.head(20), use_container_width=True)
            except Exception as e:
                st.error(f"Errore caricamento file di test: {e}")

if df is None or df.empty:
    st.stop()

# ------------------------------
# 2) Verifica formato
# ------------------------------
with st.expander("2) Verifica formato richiesto (long)", expanded=True):
    cols = list(df.columns)
    st.write("Colonne disponibili:", cols)

    # Proviamo ad indovinare colonne candidate
    guess_id = None
    guess_time = None
    guess_val = None

    # Heuristiche semplici
    for c in cols:
        cl = c.lower()
        if guess_id is None and any(tok in cl for tok in ["id", "soggetto", "subject", "patient", "pt", "case"]):
            guess_id = c
        if guess_time is None and any(tok in cl for tok in ["time", "tempo", "visit", "visita", "giorno", "day", "t"]):
            guess_time = c
        if guess_val is None and any(tok in cl for tok in ["val", "value", "esito", "outcome", "y", "score", "measure"]):
            guess_val = c

    c1, c2, c3 = st.columns(3)
    with c1:
        sel_id = st.selectbox("Seleziona colonna ID soggetto", options=cols, index=(cols.index(guess_id) if guess_id in cols else 0), key=k("sel_id_try"))
    with c2:
        sel_time = st.selectbox("Seleziona colonna Tempo/Visita", options=cols, index=(cols.index(guess_time) if guess_time in cols else min(1, len(cols)-1)), key=k("sel_time_try"))
    with c3:
        sel_val = st.selectbox("Seleziona colonna Variabile di esito", options=cols, index=(cols.index(guess_val) if guess_val in cols else min(2, len(cols)-1)), key=k("sel_val_try"))

    # Criterio: se le tre colonne selezionate sono distinte e la distribuzione per (ID,Tempo) è 1 riga
    long_candidate = False
    if len({sel_id, sel_time, sel_val}) == 3:
        grp = df[[sel_id, sel_time]].value_counts()
        long_candidate = (grp.max() == 1)

    if long_candidate:
        st.success("Il dataset **sembra già in formato long** adatto alle analisi.")
        st.session_state[k("id_col")] = sel_id
        st.session_state[k("time_col")] = sel_time
        st.session_state[k("value_col")] = sel_val
        st.session_state[k("work_df")] = df[[sel_id, sel_time, sel_val]].copy()
        st.session_state[k("format_ok")] = True
    else:
        st.warning("Il dataset **non è in formato long** oppure presenta più righe per la stessa coppia (ID, Tempo). Usi gli strumenti di preparazione di seguito.")

# ---------------------------------------------------------
# 3) Strumenti per mettere nel formato corretto (prep tool)
# ---------------------------------------------------------
with st.expander("3) Strumenti di preparazione dati (wide → long / rinomina colonne)", expanded=not st.session_state[k("format_ok")]):
    st.markdown("**Scegli un percorso di preparazione:**")
    prep_mode = st.radio(
        "Modalità",
        options=[
            "A) Il mio dataset è LONG ma con nomi colonne diversi → seleziona/ri-nomina",
            "B) Il mio dataset è WIDE (una colonna per ciascun tempo) → converti in LONG",
        ],
        key=k("prep_mode"),
        index=(0 if st.session_state[k("format_ok")] else 1),
    )

    if prep_mode.startswith("A)"):
        st.info("Selezioni le colonne corrispondenti a ID, Tempo, Valore. Il sistema creerà un dataframe standardizzato.")
        c1, c2, c3 = st.columns(3)
        with c1:
            id_col_a = st.selectbox("Colonna ID", options=cols, key=k("A_id"))
        with c2:
            time_col_a = st.selectbox("Colonna Tempo/Visita", options=cols, key=k("A_time"))
        with c3:
            value_col_a = st.selectbox("Colonna Valore", options=cols, key=k("A_val"))

        cast_time = st.checkbox("Converti Tempo in numerico se possibile", value=False, key=k("A_cast_time"))
        drop_na = st.checkbox("Elimina righe con NA in ID/Tempo/Valore", value=True, key=k("A_drop_na"))
        dedup = st.selectbox("Se esistono duplicati (ID,Tempo) come gestirli?", options=["Mantieni prima occorrenza", "Media per duplicati"], key=k("A_dedup"))

        if st.button("Crea dataset LONG standardizzato", key=k("A_run"), use_container_width=True):
            try:
                work = df[[id_col_a, time_col_a, value_col_a]].copy()
                work.columns = ["__ID__", "__Tempo__", "__Valore__"]

                if cast_time:
                    as_num = pd.to_numeric(work["__Tempo__"], errors="coerce")
                    if not as_num.isna().all():
                        work["__Tempo__"] = as_num

                if drop_na:
                    work = work.dropna(subset=["__ID__", "__Tempo__", "__Valore__"])

                # gestisci duplicati su (ID,Tempo)
                if dedup == "Media per duplicati":
                    work = (work
                            .groupby(["__ID__", "__Tempo__"], as_index=False)
                            .agg(__Valore__=("__Valore__", "mean")))
                else:
                    work = work.drop_duplicates(subset=["__ID__", "__Tempo__"], keep="first")

                st.success("Creato dataset LONG standardizzato.")
                st.dataframe(work.head(20), use_container_width=True)

                # salva stato
                st.session_state[k("work_df")] = work.rename(columns={"__ID__": id_col_a, "__Tempo__": time_col_a, "__Valore__": value_col_a})
                st.session_state[k("id_col")] = id_col_a
                st.session_state[k("time_col")] = time_col_a
                st.session_state[k("value_col")] = value_col_a
                st.session_state[k("format_ok")] = True
            except Exception as e:
                st.error(f"Errore nella standardizzazione LONG: {e}")

    else:  # B) WIDE -> LONG
        st.info(
            "Se il dataset è in **wide** (una colonna per ciascun tempo/visita), selezioni la colonna ID e le colonne delle misure. "
            "Il sistema eseguirà un `melt()` per ottenere il formato long."
        )
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

        st.markdown("**Estrazione etichette Tempo dalle intestazioni delle colonne wide**")
        t1, t2 = st.columns(2)
        with t1:
            strip_prefix = st.text_input("Rimuovi prefisso (opzionale, es. 'T')", value="", key=k("B_strip_prefix"))
        with t2:
            to_numeric = st.checkbox("Converte etichetta Tempo in numerico se possibile", value=True, key=k("B_to_numeric"))

        drop_na_b = st.checkbox("Elimina righe con NA in ID/Tempo/Valore", value=True, key=k("B_drop_na"))
        dedup_b = st.selectbox("Se esistono duplicati (ID,Tempo)", options=["Mantieni prima occorrenza", "Media per duplicati"], key=k("B_dedup"))

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
                    # Rimozione prefisso
                    if strip_prefix:
                        work["__Tempo__"] = work["__Tempo__"].astype(str).str.replace(f"^{strip_prefix}", "", regex=True)

                    # Numerico se richiesto
                    if to_numeric:
                        as_num = pd.to_numeric(work["__Tempo__"], errors="coerce")
                        if not as_num.isna().all():
                            work["__Tempo__"] = as_num

                    if drop_na_b:
                        work = work.dropna(subset=[id_col_b, "__Tempo__", "__Valore__"])

                    # Dedup
                    if dedup_b == "Media per duplicati":
                        work = (work
                                .groupby([id_col_b, "__Tempo__"], as_index=False)
                                .agg(__Valore__=("__Valore__", "mean")))
                    else:
                        work = work.drop_duplicates(subset=[id_col_b, "__Tempo__"], keep="first")

                    # Salva stato con nomi standard scelti
                    st.session_state[k("work_df")] = work.rename(columns={id_col_b: "__ID__", "__Tempo__": "__Tempo__", "__Valore__": "__Valore__"})
                    st.session_state[k("id_col")] = "__ID__"
                    st.session_state[k("time_col")] = "__Tempo__"
                    st.session_state[k("value_col")] = "__Valore__"
                    st.session_state[k("format_ok")] = True

                    st.success("Conversione **WIDE → LONG** completata.")
                    st.dataframe(st.session_state[k("work_df")].head(20), use_container_width=True)
                except Exception as e:
                    st.error(f"Errore nella conversione wide→long: {e}")

# ------------------------------------------------
# 4) Controlli qualità e tipizzazione del Tempo
# ------------------------------------------------
if st.session_state[k("format_ok")] and st.session_state[k("work_df")] is not None:
    with st.expander("4) Controlli qualità (duplicati, missing) e tipizzazione del Tempo", expanded=True):
        work = st.session_state[k("work_df")].copy()
        id_col = st.session_state[k("id_col")]
        time_col = st.session_state[k("time_col")]
        val_col = st.session_state[k("value_col")]

        st.write(f"Colonne attive → **ID:** `{id_col}` · **Tempo:** `{time_col}` · **Valore:** `{val_col}`")
        st.write("Dimensioni:", work.shape)
        st.dataframe(work.head(20), use_container_width=True)

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            handle_missing = st.selectbox("Missing in Valore", ["Nessuna azione", "Drop righe NA"], key=k("QC_missing"))
        with c2:
            time_type = st.selectbox("Tipo Tempo", ["Auto", "Numerico", "Categoriale ordinato"], key=k("QC_ttype"))
        with c3:
            sort_by_time = st.checkbox("Ordina per Tempo", value=True, key=k("QC_sort"))
        with c4:
            check_dups = st.checkbox("Raggruppa duplicati (ID,Tempo) con media", value=False, key=k("QC_dedup"))

        if st.button("Applica controlli/trasformazioni", key=k("QC_apply"), use_container_width=True):
            try:
                if handle_missing == "Drop righe NA":
                    work = work.dropna(subset=[id_col, time_col, val_col])

                if time_type == "Numerico":
                    as_num = pd.to_numeric(work[time_col], errors="coerce")
                    if not as_num.isna().all():
                        work[time_col] = as_num
                    else:
                        st.warning("Impossibile convertire completamente Tempo in numerico; lasciato invariato.")
                elif time_type == "Categoriale ordinato":
                    work[time_col] = pd.Categorical(work[time_col], ordered=True)

                if check_dups:
                    work = (work
                            .groupby([id_col, time_col], as_index=False)
                            .agg(**{val_col: (val_col, "mean")}))

                if sort_by_time:
                    try:
                        work = work.sort_values(by=[id_col, time_col])
                    except Exception:
                        work = work.sort_values(by=[id_col])

                st.session_state[k("work_df")] = work
                st.success("Controlli/trasformazioni applicati.")
                st.dataframe(work.head(30), use_container_width=True)
            except Exception as e:
                st.error(f"Errore durante i controlli qualità: {e}")

# ------------------------------------------------
# 5) Analisi: Grafici e Modelli (solo se formato OK)
# ------------------------------------------------
def get_work_df() -> pd.DataFrame | None:
    return st.session_state.get(k("work_df"))

if st.session_state[k("format_ok")] and get_work_df() is not None:
    tab_desc, tab_plot, tab_model = st.tabs(["Impostazioni & Filtri", "Grafici", "Modelli"])

    # ---------------------------
    # TAB 1: Impostazioni & Filtri
    # ---------------------------
    with tab_desc:
        st.subheader("Impostazioni & Filtri")
        work = get_work_df()
        # Filtro opzionale su colonna categoriale
        with st.form(key=k_sec("tab1", "filters_form")):
            cat_cols = [c for c in work.columns if (work[c].dtype == "object" or str(work[c].dtype).startswith("category")) and c not in [st.session_state[k("id_col")], st.session_state[k("time_col")]]]
            use_filter = st.checkbox("Attiva filtro su colonna categoriale", value=False, key=k_sec("tab1", "use_filter"))
            if use_filter and cat_cols:
                fcol = st.selectbox("Colonna categoriale", options=cat_cols, key=k_sec("tab1", "fcol"))
                vals = sorted(work[fcol].dropna().unique().tolist())
                fvals = st.multiselect("Valori inclusi", options=vals, default=vals[:1], key=k_sec("tab1", "fvals"))
            else:
                fcol, fvals = None, None
            submitted = st.form_submit_button("Applica")
        if submitted and use_filter and fcol and fvals:
            st.session_state[k("active_filter")] = (fcol, fvals)
        elif submitted:
            st.session_state[k("active_filter")] = None

        if st.session_state[k("active_filter")]:
            fcol, fvals = st.session_state[k("active_filter")]
            st.info(f"Filtro attivo: **{fcol}** in {fvals}")
        else:
            st.info("Nessun filtro attivo.")

    # helper filtro
    def filtered_df(df_in: pd.DataFrame) -> pd.DataFrame:
        if st.session_state[k("active_filter")]:
            fcol, fvals = st.session_state[k("active_filter")]
            return df_in[df_in[fcol].isin(fvals)]
        return df_in

    # ---------------------------
    # TAB 2: Grafici
    # ---------------------------
    with tab_plot:
        st.subheader("Grafici descrittivi")
        work = filtered_df(get_work_df())
        id_col = st.session_state[k("id_col")]
        t_col = st.session_state[k("time_col")]
        y_col = st.session_state[k("value_col")]

        with st.expander("Opzioni grafico", expanded=True):
            agg_fun = st.selectbox("Aggregazione per tempo", ["media", "mediana"], key=k_sec("tab2", "agg"))
            show_ci = st.checkbox("Mostra intervallo ±1.96·SE (media)", value=True, key=k_sec("tab2", "ci"))
            max_ids = st.number_input("Linee individuali: max N soggetti (0 = nessuna)", min_value=0, value=0, step=1, key=k_sec("tab2", "nids"))

        # Linee individuali
        if max_ids and max_ids > 0 and px is not None:
            ids = work[id_col].dropna().unique().tolist()[:max_ids]
            w_ind = work[work[id_col].isin(ids)].copy()
            # Forziamo ordinamento tempo
            try:
                w_ind = w_ind.sort_values(by=[id_col, t_col])
            except Exception:
                w_ind = w_ind.sort_values(by=[id_col])
            fig_ind = px.line(w_ind, x=t_col, y=y_col, color=id_col, markers=True)
            fig_ind.update_layout(height=420)
            st.plotly_chart(fig_ind, use_container_width=True)

        # Aggregato per tempo
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
            st.warning("Plotly non disponibile nell'ambiente.")

    # ---------------------------
    # TAB 3: Modelli (dimostrativi)
    # ---------------------------
    with tab_model:
        st.subheader("Modelli statistici (dimostrativi)")
        if smf is None or sm is None:
            st.warning("`statsmodels` non è disponibile: impossibile eseguire i modelli.")
        else:
            work = filtered_df(get_work_df())[[st.session_state[k("id_col")], st.session_state[k("time_col")], st.session_state[k("value_col")]]].dropna().copy()
            id_col = st.session_state[k("id_col")]
            t_col = st.session_state[k("time_col")]
            y_col = st.session_state[k("value_col")]

            with st.form(key=k_sec("tab3", "form")):
                model_type = st.selectbox(
                    "Tipo di modello",
                    ["ANOVA per misure ripetute (semplificata)", "LM con random intercept (MixedLM)"],
                    key=k_sec("tab3", "mtype")
                )
                submit = st.form_submit_button("Esegui modello", use_container_width=True)

            if submit:
                try:
                    work[t_col] = work[t_col].astype("category")
                    if model_type.startswith("ANOVA"):
                        formula = f"`{y_col}` ~ C(`{t_col}`)"
                        fit = smf.ols(formula=formula, data=work).fit(cov_type="cluster", cov_kwds={"groups": work[id_col]})
                        st.write("**Formula:**", formula)
                        st.text(fit.summary().as_text())
                    else:
                        formula = f"`{y_col}` ~ C(`{t_col}`)"
                        md = sm.MixedLM.from_formula(formula, data=work, groups=work[id_col])
                        mdf = md.fit(method="lbfgs", reml=True)
                        st.write("**Formula:**", formula + " + (1|ID)")
                        st.text(mdf.summary().as_text())
                except Exception as e:
                    st.error(f"Errore nel fitting del modello: {e}")

# ------------------------------
# Footer tecnico
# ------------------------------
with st.expander("Dettagli tecnici", expanded=False):
    st.caption(
        "• Le chiavi di tutti i widget sono esplicite con prefisso `lr_` per prevenire `StreamlitDuplicateElementId`.\n"
        "• I dati vengono recuperati da più chiavi possibili in `st.session_state` impostate dalla pagina di upload.\n"
        "• Se il dataset è in wide, lo strumento esegue un `melt()` con opzioni per ripulire etichette del tempo, tipizzare e gestire duplicati.\n"
        "• Se il dataset è già long ma con nomi diversi, è possibile selezionare/ri-nominare le colonne e standardizzare."
    )
