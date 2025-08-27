# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
from core.state import init_state
from core.stats import summarize_continuous, summarize_categorical_full
from core.validate import dataset_diagnostics
from core.ui import quality_alert, show_missing_table

# -------------------------
# Inizializzazione stato
# -------------------------
init_state()

st.title("📈 Step 2 — Statistiche descrittive")

# Controllo dataset
if st.session_state.df is None:
    st.warning("Nessun dataset in memoria. Carichi prima i dati nella pagina **Upload Dataset**.")
    st.page_link("pages/1_📂_Upload_Dataset.py", label="➡️ Vai a Upload Dataset", icon="📂")
    st.stop()

df = st.session_state.df

# Qualità dataset (breve)
with st.expander("🔎 Stato dataset e qualità (rapido)", expanded=False):
    diag = st.session_state.diagnostics or dataset_diagnostics(df)
    st.session_state.diagnostics = diag
    quality_alert(diag)
    with st.expander("Dettaglio missing per colonna", expanded=False):
        show_missing_table(diag)

# -------------------------
# Selezione variabili
# -------------------------
st.subheader("Selezione delle variabili")
num_cols_all = list(df.select_dtypes(include="number").columns)
cat_cols_all = list(df.select_dtypes(include=["object", "category", "bool"]).columns)

c1, c2 = st.columns(2)
with c1:
    num_cols = st.multiselect(
        "Variabili **continue**",
        options=num_cols_all,
        default=num_cols_all,
        help="Selezioni le variabili numeriche per le statistiche (media, mediana, ecc.)."
    )
with c2:
    cat_cols = st.multiselect(
        "Variabili **categoriche**",
        options=cat_cols_all,
        default=cat_cols_all,
        help="Selezioni le variabili testuali/booleane (distribuzioni di frequenza)."
    )

# -------------------------
# Opzioni (continue)
# -------------------------
st.subheader("Opzioni per le variabili continue")

c3, c4, c5 = st.columns([1.2, 1, 1])
with c3:
    exclude_outliers = st.checkbox(
        "Escludi outlier (regola IQR)",
        value=False,
        help="Esclude i valori fuori da [Q1−1.5·IQR, Q3+1.5·IQR]."
    )
with c4:
    transform = st.radio(
        "Trasformazione (facoltativa)",
        options=["Nessuna", "Log10", "Box-Cox"],
        index=0,
        help="Le trasformazioni possono rendere più 'simmetriche' le distribuzioni."
    )
with c5:
    show_raw = st.checkbox(
        "Mostra anche versione **senza trasformazioni**",
        value=False,
        help="Confronta i risultati con e senza trasformazione."
    )

with st.expander("ℹ️ Cosa sono le trasformazioni (spiegazione semplice)", expanded=False):
    st.markdown("""
**Perché trasformare?**  
Quando una variabile è molto asimmetrica, una trasformazione può migliorare leggibilità e stabilità delle misure.

- **Log10**: riduce l’impatto dei valori molto grandi (si applica uno shift automatico se presenti valori ≤ 0).
- **Box-Cox**: famiglia di trasformazioni che sceglie un parametro λ (richiede dati positivi; applichiamo uno shift se serve).

Per comunicazione non tecnica, può preferire i valori **non trasformati**.
""")

# -------------------------
# Calcolo (caching) — versione V2 per forzare refresh
# -------------------------
@st.cache_data(show_spinner=False)
def compute_summaries_v2(df_in: pd.DataFrame, num_cols, cat_cols, exclude_outliers: bool, transform_label: str,
                         cat_denominator: str):
    # mappa etichetta → parametro
    _map = {"Nessuna": None, "Log10": "log10", "Box-Cox": "box-cox"}
    transform_param = _map.get(transform_label, None)
    cont = summarize_continuous(
        df_in,
        cols=list(num_cols),
        exclude_outliers=exclude_outliers,
        transform=transform_param
    )
    cat_full = summarize_categorical_full(
        df_in,
        cols=list(cat_cols),
        denominator=cat_denominator  # "total" o "valid"
    )
    return cont, cat_full

# -------------------------
# Presentazione (formati)
# -------------------------
st.subheader("Presentazione delle tabelle")
c_fmt1, c_fmt2, c_fmt3, c_fmt4 = st.columns([1.2, 1, 1, 1.2])
with c_fmt1:
    simple_view = st.toggle(
        "Vista semplice (continue) — consigliata",
        value=True,
        help="Mostra le colonne essenziali per le continue."
    )
with c_fmt2:
    dec = st.number_input("Decimali (continue)", min_value=0, max_value=6, value=2, step=1)
with c_fmt3:
    perc_dec = st.number_input("Decimali (%)", min_value=0, max_value=3, value=1, step=1)
with c_fmt4:
    cat_denominator = st.radio(
        "Percentuali categoriche calcolate su",
        options=["total", "valid"],
        index=0,
        help="• total = sul totale delle osservazioni (inclusi missing);\n• valid = solo sui valori non mancanti."
    )

cont_summary, cat_summary = compute_summaries_v2(
    df, tuple(num_cols), tuple(cat_cols), exclude_outliers, transform, cat_denominator
)

# -------------------------
# Continue (vista semplice / dettagliata)
# -------------------------
st.subheader("Risultati — Variabili continue")

def format_continuous_simple(df_in: pd.DataFrame) -> pd.DataFrame:
    cols = {
        "variable": "Variabile",
        "n": "N",
        "missing_pct": "% Missing",
        "mean": "Media",
        "sd": "Dev. std",
        "median": "Mediana",
        "p25": "Q1",
        "p75": "Q3",
        "min": "Min",
        "max": "Max",
    }
    out = df_in[list(cols.keys())].rename(columns=cols)
    num_cols_fmt = ["Media", "Dev. std", "Mediana", "Q1", "Q3", "Min", "Max"]
    out[num_cols_fmt] = out[num_cols_fmt].astype(float).round(dec)
    out["% Missing"] = out["% Missing"].astype(float).round(perc_dec)
    return out

def style_df(df_in: pd.DataFrame):
    sty = (
        df_in.style
        .format(precision=dec, na_rep="—")
        .set_properties(**{"text-align": "right"})
        .set_table_styles([{"selector": "th", "props": "text-align: left;"}])
    )
    highlight_cols = [c for c in df_in.columns if c in {"Media", "Mediana"}]
    if highlight_cols:
        sty = sty.set_properties(subset=highlight_cols, **{"font-weight": "600"})
    return sty

if simple_view:
    cont_simple = format_continuous_simple(cont_summary)
    st.dataframe(style_df(cont_simple), use_container_width=True)
    st.download_button(
        "Scarica CSV (continue — vista semplice)",
        cont_simple.to_csv(index=False).encode("utf-8"),
        file_name="descriptive_continuous_simple.csv"
    )
else:
    st.dataframe(cont_summary, use_container_width=True)
    st.download_button(
        "Scarica CSV (continue — dettagliata)",
        cont_summary.to_csv(index=False).encode("utf-8"),
        file_name="descriptive_continuous_detailed.csv"
    )

with st.expander("Come leggere la tabella continua (breve guida)", expanded=False):
    st.markdown(f"""
- **N**: numero di valori non mancanti. **% Missing**: percentuale di mancanti.
- **Media** e **Dev. std**: centratura e dispersione (arrotondate a {dec} decimali).
- **Mediana** e **Q1/Q3**: misure robuste.
- **Min/Max**: estremi osservati.
""")

with st.expander("Dettagli tecnici (outlier/trasformazioni)", expanded=False):
    st.dataframe(
        cont_summary[["variable", "n_used", "n_outliers", "transform", "shift", "boxcox_lambda"]]
        .rename(columns={
            "variable": "Variabile",
            "n_used": "N (usati)",
            "n_outliers": "Outlier esclusi",
            "transform": "Trasformazione",
            "shift": "Shift (se applicato)",
            "boxcox_lambda": "λ Box-Cox"
        }),
        use_container_width=True
    )

if show_raw and transform != "Nessuna":
    st.markdown("#### Confronto (senza trasformazioni, senza esclusione outlier)")
    cont_raw, _ = compute_summaries_v2(df, tuple(num_cols), tuple(cat_cols), False, "Nessuna", cat_denominator)
    cont_raw_simple = format_continuous_simple(cont_raw)
    st.dataframe(style_df(cont_raw_simple), use_container_width=True)

# -------------------------
# Categoriche (distribuzioni complete)
# -------------------------
st.subheader("Risultati — Variabili categoriche")

if cat_summary.empty:
    st.info("Nessuna variabile categorica selezionata.")
else:
    st.dataframe(cat_summary, use_container_width=True)
    st.download_button(
        "Scarica CSV (categoriche — distribuzioni complete)",
        cat_summary.to_csv(index=False).encode("utf-8"),
        file_name="descriptive_categorical_full.csv"
    )

with st.expander("Come leggere la tabella categorica (breve guida)", expanded=False):
    st.markdown("""
- Ogni variabile è riportata con **tutte le sue categorie** (e una riga **(Missing)** se presenti valori mancanti).
- **Frequenza**: conteggio delle osservazioni in ciascuna categoria.
- **%**: percentuale calcolata sul **totale** della variabile (opzione *total*) oppure solo sui **valori non mancanti** (opzione *valid*).
""")

# -------------------------
# Aggiunta al Results Summary
# -------------------------
st.divider()
if st.button("➕ Aggiungi queste tabelle al Results Summary"):
    # continua: vista semplice per report
    cont_for_report = format_continuous_simple(cont_summary).to_dict(orient="records")
    # categoriche: distribuzioni complete
    cat_for_report = cat_summary.to_dict(orient="records")

    st.session_state.report_items.append({
        "type": "table",
        "title": "Descrittive — Continue (Vista semplice)",
        "data": cont_for_report
    })
    st.session_state.report_items.append({
        "type": "table",
        "title": "Descrittive — Categoriche (Distribuzioni complete)",
        "data": cat_for_report
    })
    st.success("Tabelle aggiunte al Results Summary.")

st.info("Suggerimento: prosegua con **Explore Distributions** per i grafici e le verifiche di normalità.")
