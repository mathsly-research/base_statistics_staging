# 1_🧹_Data_Cleaning.py
from __future__ import annotations

import streamlit as st
import pandas as pd
import numpy as np
import re
import unicodedata

# ------------------------------
# Configurazione di pagina
# ------------------------------
st.set_page_config(page_title="🧹 Pulizia Dati", layout="wide")

# ------------------------------
# Utility chiavi univoche e stato
# ------------------------------
KEY_PREFIX = "dc"  # data cleaning

def k(name: str) -> str:
    return f"{KEY_PREFIX}_{name}"

def k_sec(section: str, name: str) -> str:
    return f"{KEY_PREFIX}_{section}_{name}"

def ss_get(name: str, default=None):
    return st.session_state.get(name, default)

def ss_set_default(name: str, value):
    if name not in st.session_state:
        st.session_state[name] = value

# Inizializzazione sicura
ss_set_default(k("work_df"), None)            # dataframe su cui operare
ss_set_default(k("source_key"), None)         # chiave da cui è arrivato il df
ss_set_default(k("step_done"), {              # tracking passi guidati
    "dati": False, "colonne": False, "tipi": False,
    "missing": False, "duplicati": False, "outlier": False, "testo": False
})

# ------------------------------
# Header
# ------------------------------
st.title("🧹 Pulizia Dati")
st.markdown(
    """
Percorso guidato per pulire e standardizzare i dati **senza competenze tecniche**.
Segua i passi nell’ordine suggerito. Ogni passo offre scelte predefinite sicure e una spiegazione concisa.
"""
)

# ==============================
# PASSO 0 · CARICAMENTO DATI
# ==============================
st.subheader("Passo 0 · Dati di partenza")
left, right = st.columns([3, 2])

with left:
    # Recupero silenzioso da session_state (prova chiavi comuni)
    df, found_key = None, None
    for key in ["cleaned_df", "work_df", "uploaded_df", "df_upload", "main_df", "dataset", "df"]:
        if key in st.session_state and isinstance(st.session_state[key], pd.DataFrame):
            df, found_key = st.session_state[key], key
            break

    if df is not None and not df.empty:
        st.success(f"Dataset pronto ({df.shape[0]} righe × {df.shape[1]} colonne).")
        st.dataframe(df.head(15), use_container_width=True, height=300)
        st.caption(f"Origine: `st.session_state['{found_key}']`")
        st.session_state[k("work_df")] = df.copy()
        st.session_state[k("source_key")] = found_key
        st.session_state[k("step_done")]["dati"] = True
    else:
        st.warning("Nessun dataset trovato nelle pagine precedenti.")
        st.info("Può caricare un file qui sotto (solo per questa pagina).")

with right:
    temp_file = st.file_uploader(
        "Carica CSV/XLSX (opzionale)", type=["csv", "xlsx", "xls"], key=k("temp_uploader"),
        help="Se non ha caricato i dati in precedenza, può caricarli qui."
    )
    if temp_file is not None:
        try:
            if temp_file.name.lower().endswith(".csv"):
                df = pd.read_csv(temp_file)
            else:
                df = pd.read_excel(temp_file)
            st.success(f"File caricato: {df.shape[0]} × {df.shape[1]}")
            st.dataframe(df.head(10), use_container_width=True, height=220)
            st.session_state[k("work_df")] = df.copy()
            st.session_state[k("source_key")] = None
            st.session_state[k("step_done")]["dati"] = True
        except Exception as e:
            st.error(f"Errore caricamento: {e}")

work_df = ss_get(k("work_df"))
if work_df is None or work_df.empty:
    st.stop()

# ==============================
# Barra stato (passi completati)
# ==============================
done = st.session_state[k("step_done")]
n_done = sum(done.values())
st.progress(n_done / len(done), text=f"Avanzamento: {n_done}/{len(done)} passi completati")

# ==============================
# PASSO 1 · COLONNE (selezione/rinomina)
# ==============================
st.subheader("Passo 1 · Colonne")
st.markdown(
    "Selezioni le colonne utili e (se serve) **rinomini** con etichette chiare. "
    "Suggerimento: rimuovere colonne identiche, costanti o chiaramente irrilevanti."
)

c_cols = work_df.columns.tolist()
c1, c2 = st.columns([2, 1])
with c1:
    keep_cols = st.multiselect(
        "Selezioni le colonne da mantenere",
        options=c_cols,
        default=c_cols,
        key=k_sec("colonne", "keep_cols"),
        help="Per impostazione predefinita si mantengono tutte."
    )
with c2:
    drop_constant = st.checkbox(
        "Rimuovi colonne costanti", value=True, key=k_sec("colonne", "drop_constant"),
        help="Rimuove le colonne con un solo valore distinto."
    )

with st.expander("Rinomina colonne (facoltativo)", expanded=False):
    # Mappatura interactiva: seleziona una colonna e indica il nuovo nome
    col_rn = st.selectbox("Colonna da rinominare", options=keep_cols, key=k_sec("colonne", "rn_col"))
    new_name = st.text_input("Nuovo nome", value=col_rn, key=k_sec("colonne", "rn_name"))
    add_map = st.button("Aggiungi alla mappa di rinomina", key=k_sec("colonne", "add_map"))
    if add_map:
        rn_map = ss_get(k("rename_map"), {})
        rn_map[col_rn] = new_name
        st.session_state[k("rename_map")] = rn_map

    rn_map = ss_get(k("rename_map"), {})
    if rn_map:
        st.write("Mappa di rinomina corrente:")
        st.dataframe(pd.DataFrame({"Colonna": list(rn_map.keys()), "Nuovo nome": list(rn_map.values())}))

apply_cols = st.button("Applica selezione/rinomina colonne", key=k_sec("colonne", "apply"), use_container_width=True)
if apply_cols:
    try:
        tmp = work_df[keep_cols].copy()
        if ss_get(k("rename_map")):
            # Evita collisioni di nomi
            map_clean = {}
            for old, new in ss_get(k("rename_map")).items():
                if new.strip() == "":
                    continue
                map_clean[old] = new.strip()
            # Se collisioni, aggiungi suffisso
            collisions = [n for n in map_clean.values() if list(map_clean.values()).count(n) > 1]
            if collisions:
                for x in collisions:
                    i = 1
                    for old, new in list(map_clean.items()):
                        if new == x:
                            map_clean[old] = f"{new}_{i}"
                            i += 1
            tmp = tmp.rename(columns=map_clean)

        if ss_get(k_sec("colonne", "drop_constant")):
            nunique = tmp.nunique(dropna=False)
            const_cols = nunique[nunique <= 1].index.tolist()
            tmp = tmp.drop(columns=const_cols) if const_cols else tmp
            if const_cols:
                st.info(f"Rimosse colonne costanti: {const_cols}")

        st.session_state[k("work_df")] = tmp
        work_df = tmp
        st.success("Colonne aggiornate.")
        st.dataframe(work_df.head(15), use_container_width=True)
        st.session_state[k("step_done")]["colonne"] = True
    except Exception as e:
        st.error(f"Errore nella gestione colonne: {e}")

# ==============================
# PASSO 2 · TIPI DI DATO
# ==============================
st.subheader("Passo 2 · Tipi di dato")
st.markdown("Imposti i tipi corretti: numerico, data/ora, categoriale. Conversione **tollerante agli errori**.")

cols_now = work_df.columns.tolist()
num_guess = [c for c in cols_now if pd.api.types.is_numeric_dtype(work_df[c])]
obj_guess = [c for c in cols_now if work_df[c].dtype == "object"]

c1, c2, c3 = st.columns(3)
with c1:
    to_numeric_cols = st.multiselect("→ Converti in numerico", options=cols_now, default=[], key=k_sec("tipi", "to_num"))
with c2:
    to_date_cols = st.multiselect("→ Converti in data/ora", options=cols_now, default=[], key=k_sec("tipi", "to_dt"))
with c3:
    to_cat_cols = st.multiselect("→ Converti in categoriale", options=cols_now, default=[], key=k_sec("tipi", "to_cat"))

numeric_opts1, numeric_opts2 = st.columns(2)
with numeric_opts1:
    decimal_comma = st.checkbox("Riconosci virgola decimale", value=True, key=k_sec("tipi", "dec_comma"),
                                help="Converte '1,23' in 1.23 prima della trasformazione.")
with numeric_opts2:
    coerce_errors = st.checkbox("Errori → NA (coerce)", value=True, key=k_sec("tipi", "coerce"))

apply_types = st.button("Applica conversioni di tipo", key=k_sec("tipi", "apply"), use_container_width=True)
if apply_types:
    try:
        tmp = work_df.copy()

        # Numerico (con virgola decimale)
        for c in to_numeric_cols:
            s = tmp[c].astype(str) if tmp[c].dtype == "object" else tmp[c]
            if decimal_comma:
                s = s.str.replace(r"\.", "", regex=True).str.replace(",", ".", regex=False) if s.dtype == "object" else s
            tmp[c] = pd.to_numeric(s, errors=("coerce" if coerce_errors else "raise"))

        # Data/ora
        for c in to_date_cols:
            tmp[c] = pd.to_datetime(tmp[c], errors=("coerce" if coerce_errors else "raise"))

        # Categoriale
        for c in to_cat_cols:
            tmp[c] = tmp[c].astype("category")

        st.session_state[k("work_df")] = tmp
        work_df = tmp
        st.success("Tipi aggiornati.")
        st.dataframe(work_df.head(12), use_container_width=True)
        st.session_state[k("step_done")]["tipi"] = True
    except Exception as e:
        st.error(f"Errore conversione tipi: {e}")

# ==============================
# PASSO 3 · MISSING VALUES
# ==============================
st.subheader("Passo 3 · Valori mancanti")
st.markdown("Gestione semplice di valori mancanti: rimozione righe, imputazione numerica/categoriale.")

miss = work_df.isna().sum().sort_values(ascending=False)
with st.expander("Panoramica NA per colonna", expanded=False):
    st.dataframe(miss.to_frame("NA").T if len(miss) <= 20 else miss, use_container_width=True)

c1, c2, c3 = st.columns(3)
with c1:
    drop_any = st.checkbox("Elimina righe con **almeno un NA**", value=False, key=k_sec("missing", "drop_any"))
with c2:
    drop_all = st.checkbox("Elimina righe con **tutti NA**", value=True, key=k_sec("missing", "drop_all"))
with c3:
    impute_numeric = st.selectbox("Imputazione numerica", ["Nessuna", "Media", "Mediana"], key=k_sec("missing", "imp_num"))

c4, c5 = st.columns(2)
with c4:
    impute_categorical = st.selectbox("Imputazione categoriale", ["Nessuna", "Moda"], key=k_sec("missing", "imp_cat"))
with c5:
    cols_sel_impute = st.multiselect("Colonne specifiche (vuoto = tutte pertinenti)", options=work_df.columns.tolist(),
                                     key=k_sec("missing", "cols_sel"))

apply_missing = st.button("Applica regole sui missing", key=k_sec("missing", "apply"), use_container_width=True)
if apply_missing:
    try:
        tmp = work_df.copy()

        # Drop
        if drop_all:
            tmp = tmp.dropna(how="all")
        if drop_any:
            tmp = tmp.dropna(how="any")

        # Selezione colonne pertinenti
        cols_target = cols_sel_impute or tmp.columns.tolist()

        # Imputazione numerica
        if impute_numeric != "Nessuna":
            num_cols = [c for c in cols_target if pd.api.types.is_numeric_dtype(tmp[c])]
            for c in num_cols:
                if impute_numeric == "Media":
                    val = tmp[c].mean()
                else:
                    val = tmp[c].median()
                tmp[c] = tmp[c].fillna(val)

        # Imputazione categoriale
        if impute_categorical == "Moda":
            cat_cols = [c for c in cols_target if tmp[c].dtype == "object" or str(tmp[c].dtype).startswith("category")]
            for c in cat_cols:
                if tmp[c].dropna().empty:
                    continue
                mode_val = tmp[c].mode(dropna=True).iloc[0]
                tmp[c] = tmp[c].fillna(mode_val)

        st.session_state[k("work_df")] = tmp
        work_df = tmp
        st.success("Missing gestiti.")
        st.dataframe(work_df.head(12), use_container_width=True)
        st.session_state[k("step_done")]["missing"] = True
    except Exception as e:
        st.error(f"Errore gestione missing: {e}")

# ==============================
# PASSO 4 · DUPLICATI
# ==============================
st.subheader("Passo 4 · Duplicati")
st.markdown("Rimuova righe duplicate in base a un sottoinsieme di colonne (default: tutte).")

dup_cols = st.multiselect(
    "Colonne chiave per identificare duplicati",
    options=work_df.columns.tolist(),
    default=work_df.columns.tolist(),
    key=k_sec("dup", "cols")
)
dup_keep = st.selectbox("Conserva", options=["first", "last"], index=0, key=k_sec("dup", "keep"))

apply_dup = st.button("Rimuovi duplicati", key=k_sec("dup", "apply"), use_container_width=True)
if apply_dup:
    try:
        before = work_df.shape[0]
        tmp = work_df.drop_duplicates(subset=dup_cols or None, keep=dup_keep)
        after = tmp.shape[0]
        st.info(f"Righe rimosse: {before - after}")
        st.session_state[k("work_df")] = tmp
        work_df = tmp
        st.success("Duplicati rimossi.")
        st.dataframe(work_df.head(12), use_container_width=True)
        st.session_state[k("step_done")]["duplicati"] = True
    except Exception as e:
        st.error(f"Errore rimozione duplicati: {e}")

# ==============================
# PASSO 5 · OUTLIER
# ==============================
st.subheader("Passo 5 · Outlier")
st.markdown("Individua e gestisci outlier numerici con regole semplici (IQR o Deviazione Standard).")

num_cols_available = [c for c in work_df.columns if pd.api.types.is_numeric_dtype(work_df[c])]
c1, c2, c3 = st.columns(3)
with c1:
    out_cols = st.multiselect("Colonne numeriche", options=num_cols_available, default=num_cols_available[:1],
                              key=k_sec("out", "cols"))
with c2:
    method = st.selectbox("Metodo", ["IQR (1.5×)", "SD (3×)"], key=k_sec("out", "method"))
with c3:
    action = st.selectbox("Azione", ["Segna (nessuna modifica)", "Winsorize", "Rimuovi righe"], key=k_sec("out", "action"))

apply_out = st.button("Applica regole outlier", key=k_sec("out", "apply"), use_container_width=True)
if apply_out:
    try:
        tmp = work_df.copy()
        report = []
        for c in out_cols:
            s = tmp[c]
            if method.startswith("IQR"):
                q1, q3 = s.quantile(0.25), s.quantile(0.75)
                iqr = q3 - q1
                low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            else:
                mu, sd = s.mean(), s.std(ddof=0)
                low, high = mu - 3 * sd, mu + 3 * sd
            mask = (s < low) | (s > high)
            n_out = int(mask.sum())
            report.append(f"{c}: {n_out} outlier")

            if action == "Winsorize":
                tmp.loc[s < low, c] = low
                tmp.loc[s > high, c] = high
            elif action == "Rimuovi righe":
                tmp = tmp[~mask]

        if report:
            st.info(" | ".join(report))
        st.session_state[k("work_df")] = tmp
        work_df = tmp
        if action == "Segna (nessuna modifica)":
            st.success("Outlier individuati (nessuna modifica applicata).")
        else:
            st.success("Regole outlier applicate.")
        st.dataframe(work_df.head(12), use_container_width=True)
        st.session_state[k("step_done")]["outlier"] = True
    except Exception as e:
        st.error(f"Errore gestione outlier: {e}")

# ==============================
# PASSO 6 · TESTO / CATEGORIE
# ==============================
st.subheader("Passo 6 · Testo e categorie")
st.markdown("Normalizzi testo e categorie: spazi, maiuscole/minuscole, accenti, valori 'sì/no'.")

obj_cols = [c for c in work_df.columns if work_df[c].dtype == "object" or str(work_df[c].dtype).startswith("category")]

if obj_cols:
    c1, c2, c3 = st.columns(3)
    with c1:
        trim_spaces = st.checkbox("Rimuovi spazi iniziali/finali", value=True, key=k_sec("txt", "trim"))
    with c2:
        to_lower = st.checkbox("Minuscole", value=False, key=k_sec("txt", "lower"))
    with c3:
        strip_accents = st.checkbox("Rimuovi accenti (ASCII)", value=False, key=k_sec("txt", "acc"))

    c4, c5 = st.columns(2)
    with c4:
        standardize_bool = st.checkbox("Standardizza sì/no (True/False)", value=True, key=k_sec("txt", "bool"))
    with c5:
        collapse_spaces = st.checkbox("Riduci spazi multipli", value=True, key=k_sec("txt", "collapse"))

    apply_text = st.button("Applica normalizzazione testo", key=k_sec("txt", "apply"), use_container_width=True)

    if apply_text:
        try:
            tmp = work_df.copy()

            def _strip_acc(s):
                nfkd = unicodedata.normalize('NFKD', s)
                return "".join([c for c in nfkd if not unicodedata.combining(c)])

            for c in obj_cols:
                s = tmp[c].astype(str)

                if trim_spaces:
                    s = s.str.strip()
                if collapse_spaces:
                    s = s.str.replace(r"\s+", " ", regex=True)
                if to_lower:
                    s = s.str.lower()
                if strip_accents:
                    s = s.apply(_strip_acc)

                if standardize_bool:
                    # rimappa varianti comuni di sì/no/true/false
                    mapping = {
                        "si": "sì", "yes": "sì", "y": "sì", "1": "sì", "vero": "sì", "true": "sì",
                        "no": "no", "n": "no", "0": "no", "falso": "no", "false": "no"
                    }
                    s_norm = s.str.lower().map(mapping).fillna(s)
                    s = s_norm

                tmp[c] = s

            st.session_state[k("work_df")] = tmp
            work_df = tmp
            st.success("Normalizzazione testo applicata.")
            st.dataframe(work_df.head(12), use_container_width=True)
            st.session_state[k("step_done")]["testo"] = True
        except Exception as e:
            st.error(f"Errore normalizzazione testo: {e}")
else:
    st.info("Nessuna colonna testuale/categoriale rilevata.")

# ==============================
# PASSO 7 · SALVA / ESPORTA
# ==============================
st.subheader("Passo 7 · Salva ed esporta")
st.markdown(
    "Salvi il dataset pulito per le pagine successive e, se desidera, lo **scarichi in CSV**."
)

# Salva in sessione con chiavi standard
save_btn = st.button("Salva dataset pulito nelle pagine successive", key=k("save_cleaned"), use_container_width=True)
if save_btn:
    st.session_state["cleaned_df"] = work_df.copy()
    st.session_state["uploaded_df"] = work_df.copy()  # opzionale: aggiorna come dataset principale
    st.success("Dataset pulito salvato: `cleaned_df` (e `uploaded_df`).")

# Download
csv_bytes = work_df.to_csv(index=False).encode("utf-8")
st.download_button(
    label="⬇️ Scarica CSV pulito",
    data=csv_bytes,
    file_name="dataset_pulito.csv",
    mime="text/csv",
    key=k("download_cleaned")
)

# Riepilogo finale
st.markdown("---")
c1, c2, c3 = st.columns(3)
with c1:
    st.metric("Righe", value=work_df.shape[0])
with c2:
    st.metric("Colonne", value=work_df.shape[1])
with c3:
    st.metric("Passi completati", value=f"{sum(st.session_state[k('step_done')].values())}/{len(st.session_state[k('step_done')])}")

with st.expander("Dettagli tecnici (opzionale)"):
    st.caption(
        "• Tutti i widget usano chiavi esplicite con prefisso `dc_` per evitare conflitti.\n"
        "• Le operazioni sono **incrementali** sul `work_df` in `st.session_state`.\n"
        "• Tipi, missing, duplicati, outlier e testo sono gestiti con opzioni conservative di default."
    )
