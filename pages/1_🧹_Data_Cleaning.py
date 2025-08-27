# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np

from core.state import init_state

# ---------------------------------
# Init & check dataset
# ---------------------------------
init_state()
st.title("🧹 Step 1 — Gestione dati mancanti e filtri")

# Serve un dataset caricato in precedenza
if "df" not in st.session_state or st.session_state.df is None:
    st.warning("Nessun dataset in memoria. Carichi i dati in **Upload Dataset** (Step 0).")
    st.page_link("pages/0_📂_Upload_Dataset.py", label="➡️ Vai a Upload Dataset", icon="📂")
    st.stop()

# Inizializzazione sicura delle copie
if "df_original" not in st.session_state or st.session_state.df_original is None:
    st.session_state.df_original = st.session_state.df.copy()
if "df_working" not in st.session_state or st.session_state.df_working is None:
    st.session_state.df_working = st.session_state.df.copy()

# ---------------------------------
# Selettore dataset attivo
# ---------------------------------
choice = st.radio(
    "Seleziona dataset attivo:",
    ["Originale", "Modificato (working)"],
    horizontal=True
)

df = st.session_state.df_original.copy() if choice == "Originale" else st.session_state.df_working.copy()
st.info(f"Dataset attivo: **{choice}** — {df.shape[0]} righe × {df.shape[1]} colonne")

# ---------------------------------
# Diagnostica missing values
# ---------------------------------
st.subheader("🔎 Missing values per variabile")
if len(df) == 0:
    st.warning("Il dataset attivo è vuoto.")
else:
    missing = df.isnull().sum()
    missing_pct = 100 * missing / len(df)
    summary = pd.DataFrame({"Missing": missing, "% Missing": missing_pct.round(1)})
    st.dataframe(summary, use_container_width=True)

# ---------------------------------
# Strategia di gestione missing (con avvisi metodologici)
# ---------------------------------
st.subheader("⚙️ Strategie di gestione")
st.caption("Nota metodologica: l’eliminazione di righe/colonne è **sconsigliata** (potenziale bias o perdita d’informazione). Preferire, quando possibile, imputazioni appropriate.")

options = {
    "Nessuna azione": None,
    "Elimina righe con missing (sconsigliato)": "drop_rows",
    "Elimina variabile (colonna) (sconsigliato)": "drop_col",
    "Sostituisci con media (solo numeriche)": "mean",
    "Sostituisci con mediana (solo numeriche)": "median",
    "Sostituisci con moda": "mode",
}

strategy = {}
for col in df.columns:
    sel = st.selectbox(
        f"{col}:",
        list(options.keys()),
        index=0,
        key=f"strategy_{col}"
    )
    strategy[col] = sel
    if "sconsigliato" in sel:
        st.warning(f"⚠️ {col}: l'eliminazione può introdurre bias. Utilizzare solo se strettamente necessario.")

if st.button("➡️ Applica strategie di gestione sul dataset **Modificato (working)**"):
    working = st.session_state.df_working.copy()
    for col, choice_label in strategy.items():
        action = options[choice_label]
        if action == "drop_rows":
            working = working.dropna(subset=[col])
        elif action == "drop_col":
            if col in working.columns:
                working = working.drop(columns=[col])
        elif action == "mean":
            if col in working.columns and pd.api.types.is_numeric_dtype(working[col]):
                working[col] = working[col].fillna(working[col].mean())
        elif action == "median":
            if col in working.columns and pd.api.types.is_numeric_dtype(working[col]):
                working[col] = working[col].fillna(working[col].median())
        elif action == "mode":
            if col in working.columns and not working[col].dropna().empty:
                mode_val = working[col].mode().iloc[0]
                working[col] = working[col].fillna(mode_val)

    st.session_state.df_working = working
    st.success("Strategie applicate al dataset **Modificato (working)**.")
    st.warning("⚠️ I risultati salvati in precedenza potrebbero non essere coerenti con il dataset attivo: ricalcolarli se necessario.")

# ---------------------------------
# Filtri interattivi
# ---------------------------------
st.subheader("🔎 Filtri sui dati (si applicano al dataset **Modificato**)")

filters = {}
for col in df.columns:
    if pd.api.types.is_numeric_dtype(df[col]):
        # Gestione range degeneri
        col_min, col_max = float(df[col].min()), float(df[col].max())
        if not np.isfinite(col_min) or not np.isfinite(col_max):
            st.info(f"{col}: valori non finiti, filtro disabilitato.")
            continue
        if col_min == col_max:
            st.info(f"{col}: range nullo ({col_min}), filtro disabilitato.")
            continue
        sel_min, sel_max = st.slider(
            f"Filtro per {col}:",
            min_value=col_min,
            max_value=col_max,
            value=(col_min, col_max)
        )
        filters[col] = ("num", (sel_min, sel_max))
    else:
        unique_vals = df[col].dropna().unique().tolist()
        if len(unique_vals) == 0:
            st.info(f"{col}: nessun valore disponibile, filtro disabilitato.")
            continue
        sel_vals = st.multiselect(f"Filtro per {col}:", unique_vals, default=unique_vals)
        filters[col] = ("cat", sel_vals)

if st.button("➡️ Applica filtri sul dataset **Modificato (working)**"):
    working = st.session_state.df_working.copy()
    for col, (ftype, rule) in filters.items():
        if col not in working.columns:
            continue
        if ftype == "num":
            lo, hi = rule
            working = working[(working[col] >= lo) & (working[col] <= hi)]
        else:
            # Se nessun livello selezionato, il filtro rimuove tutte le righe di quella colonna
            working = working[working[col].isin(rule)]
    st.session_state.df_working = working
    st.success(f"Filtri applicati al dataset **Modificato (working)**: {working.shape[0]} righe rimanenti.")
    st.warning("⚠️ I risultati salvati in precedenza potrebbero non essere coerenti con il dataset attivo: ricalcolarli se necessario.")

# ---------------------------------
# Spiegazioni
# ---------------------------------
with st.expander("ℹ️ Guida"):
    st.markdown("""
**Principi metodologici**
- *Elimina righe/colonne*: **sconsigliato** (introduce bias o perdita d’informazione).
- *Imputazione semplice* (media/mediana/moda): mantiene la dimensione del campione ma può attenuare la variabilità; valutare metodi più avanzati (es. imputazione multipla) se l’analisi è sensibile.

**Dataset attivi**
- **Originale**: copia immutata del dataset caricato (riferimento).
- **Modificato (working)**: versione su cui applica imputazioni/filtri; è quella utilizzata dagli step successivi quando selezionata come attiva.

⚠️ Se cambia dataset attivo, verifichi la coerenza dei risultati già salvati nel *Results Summary*.
""")
