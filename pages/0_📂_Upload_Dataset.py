# pages/0_📂_Upload_Dataset.py
from __future__ import annotations

import io
import streamlit as st
import pandas as pd

# ──────────────────────────────────────────────────────────────────────────────
# Configurazione pagina
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="📂 Carica dataset", layout="wide")

# Menu laterale (se presente)
try:
    from nav import sidebar
    sidebar()
except Exception:
    pass  # la pagina funziona anche senza nav.py

# ──────────────────────────────────────────────────────────────────────────────
# Utility
# ──────────────────────────────────────────────────────────────────────────────
KEY = "up"  # prefisso chiavi per evitare collisioni

def k(name: str) -> str:
    return f"{KEY}_{name}"

def ss_get(name: str, default=None):
    return st.session_state.get(name, default)

def ss_set_default(name: str, value):
    if name not in st.session_state:
        st.session_state[name] = value

# Stato iniziale sicuro
ss_set_default(k("df"), None)
ss_set_default(k("source"), None)          # 'upload' | 'sample'
ss_set_default(k("encoding"), "utf-8")
ss_set_default(k("decimal"), ",")          # default europeo
ss_set_default(k("thousands"), ".")        # default europeo
ss_set_default(k("saved_ok"), False)

# ──────────────────────────────────────────────────────────────────────────────
# Header
# ──────────────────────────────────────────────────────────────────────────────
st.title("📂 Importa Dataset")
st.markdown(
    """
**Obiettivo:** caricare un file (CSV/Excel), verificare rapidamente struttura e qualità, quindi
**salvare il dataset** per le pagine successive.

Proceda in tre passi: 1) Sorgente dati → 2) Opzioni di lettura → 3) Anteprima e salvataggio.
"""
)

# Barra di avanzamento semplificata
steps = ["Sorgente", "Opzioni", "Anteprima/Salva"]
progress = 0
if ss_get(k("df")) is not None:
    progress = 2
st.progress((progress+1)/len(steps), text=f"Passo {progress+1} di {len(steps)} · {steps[progress]}")

# ──────────────────────────────────────────────────────────────────────────────
# Passo 1 · Sorgente dati
# ──────────────────────────────────────────────────────────────────────────────
st.subheader("Passo 1 · Sorgente dati")

c1, c2 = st.columns([2, 3], vertical_alignment="top")
with c1:
    source = st.radio(
        "Selezioni la sorgente",
        options=["Carica file", "Usa dataset di esempio"],
        index=0 if ss_get(k("source")) != "sample" else 1,
        key=k("source_radio"),
        help="È possibile usare un piccolo dataset di esempio per provare l’app."
    )

with c2:
    if source == "Carica file":
        st.session_state[k("source")] = "upload"
        uploaded = st.file_uploader(
            "Carichi un file CSV o Excel",
            type=["csv", "xlsx", "xls"],
            key=k("uploader"),
            help="Formati supportati: .csv, .xlsx, .xls"
        )
        if uploaded is not None:
            st.success(f"File selezionato: **{uploaded.name}** ({uploaded.size/1024:.1f} KB)")
            st.session_state[k("raw_file")] = uploaded
            st.session_state[k("saved_ok")] = False
    else:
        st.session_state[k("source")] = "sample"
        if st.button("Carica dataset di esempio (Iris)", key=k("load_sample"), use_container_width=True):
            df_sample = pd.read_csv(
                "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
            )
            st.session_state[k("df")] = df_sample.copy()
            st.session_state[k("saved_ok")] = False
            st.success("Dataset di esempio caricato.")
            st.dataframe(df_sample.head(15), use_container_width=True)

# ──────────────────────────────────────────────────────────────────────────────
# Passo 2 · Opzioni di lettura
# ──────────────────────────────────────────────────────────────────────────────
st.subheader("Passo 2 · Opzioni di lettura")

left, right = st.columns([3, 2])
with left:
    st.markdown("**Opzioni CSV** (ignorate per Excel)")
    c_csv1, c_csv2, c_csv3 = st.columns(3)
    with c_csv1:
        enc = st.selectbox(
            "Encoding",
            options=["utf-8", "latin-1", "cp1252"],
            index=["utf-8", "latin-1", "cp1252"].index(ss_get(k("encoding"))),
            key=k("encoding"),
            help="Se vede caratteri strani, provi un encoding diverso."
        )
    with c_csv2:
        decimal = st.text_input("Separatore decimali", value=ss_get(k("decimal")), key=k("decimal"))
    with c_csv3:
        thousands = st.text_input("Separatore migliaia", value=ss_get(k("thousands")), key=k("thousands"))

    sniff_sep = st.checkbox(
        "Rileva automaticamente il separatore di campo (CSV)",
        value=True,
        key=k("sniff_sep"),
        help="Abilitato: il separatore viene dedotto; Disabilitato: usa il punto e virgola ';' o la virgola ','."
    )
    if not ss_get(k("sniff_sep")):
        sep = st.selectbox("Separatore", options=[",", ";", "\t", "|"], index=1, key=k("sep_fixed"))
    else:
        sep = None

with right:
    st.markdown("**Opzioni Excel**")
    header_row = st.number_input("Riga header (Excel)", min_value=0, value=0, step=1, key=k("header_row"))
    engine_xlsx = st.selectbox("Motore Excel", options=["auto", "openpyxl"], index=0, key=k("engine_xlsx"))

# Pulsante di lettura
read_btn = st.button("📥 Leggi/aggiorna dataset", key=k("read"), use_container_width=True)

# Funzione di lettura robusta
def read_uploaded_file(file) -> pd.DataFrame | None:
    name = file.name.lower()
    if name.endswith(".csv"):
        # Strategie: 1) sep=None (sniffer) con encoding scelto; fallback a separatori comuni
        data = file.read()
        buf = io.BytesIO(data)
        try:
            if ss_get(k("sniff_sep"), True):
                df = pd.read_csv(io.BytesIO(data), encoding=ss_get(k("encoding")), sep=None, engine="python")
            else:
                df = pd.read_csv(io.BytesIO(data), encoding=ss_get(k("encoding")),
                                 sep=ss_get(k("sep_fixed", ",")))
        except Exception:
            # fallback: virgola
            buf.seek(0)
            df = pd.read_csv(buf, encoding=ss_get(k("encoding")), sep=",")
        # Gestione decimali/migliaia opzionale
        dec = ss_get(k("decimal"))
        tho = ss_get(k("thousands"))
        if dec in [",", "."] and tho in [",", ".", " "]:
            # tenta conversione per colonne numeriche lette come stringa
            for col in df.columns:
                if df[col].dtype == "object":
                    s = df[col].str.replace(tho, "", regex=False).str.replace(dec, ".", regex=False)
                    try:
                        num = pd.to_numeric(s, errors="ignore")
                        if pd.api.types.is_numeric_dtype(num):
                            df[col] = num
                    except Exception:
                        pass
        return df

    elif name.endswith((".xlsx", ".xls")):
        # Apertura book e selezione foglio
        try:
            xls = pd.ExcelFile(file, engine=None if engine_xlsx == "auto" else engine_xlsx)
            sheet = st.selectbox("Selezioni il foglio", options=xls.sheet_names, key=k("sheet_name"))
            df = pd.read_excel(xls, sheet_name=sheet, header=ss_get(k("header_row", 0)))
            return df
        except Exception as e:
            st.error(f"Errore lettura Excel: {e}")
            return None
    else:
        st.error("Formato non supportato.")
        return None

if read_btn and ss_get(k("source")) == "upload":
    up = ss_get(k("raw_file"))
    if up is None:
        st.warning("Nessun file selezionato.")
    else:
        df = read_uploaded_file(up)
        if df is not None and not df.empty:
            st.session_state[k("df")] = df.copy()
            st.session_state[k("saved_ok")] = False
            st.success(f"Dataset letto correttamente: {df.shape[0]} righe × {df.shape[1]} colonne.")
        else:
            st.error("Impossibile leggere il dataset o dataset vuoto.")

# ──────────────────────────────────────────────────────────────────────────────
# Passo 3 · Anteprima, controlli rapidi e salvataggio
# ──────────────────────────────────────────────────────────────────────────────
st.subheader("Passo 3 · Anteprima e salvataggio")

df = ss_get(k("df"))
if df is None:
    st.info("Carichi o legga il dataset per proseguire.")
    st.stop()

# Riepilogo rapido
m1, m2, m3, m4 = st.columns(4)
with m1: st.metric("Righe", value=df.shape[0])
with m2: st.metric("Colonne", value=df.shape[1])
with m3: st.metric("Mancanti (tot)", value=int(df.isna().sum().sum()))
with m4: st.metric("Numeriche / Categoriali", value=f"{sum(pd.api.types.is_numeric_dtype(df[c]) for c in df.columns)}/{df.shape[1]}")

# Anteprima
st.dataframe(df.head(25), use_container_width=True)

# Controlli sintetici
with st.expander("🔎 Controlli qualità sintetici", expanded=False):
    cqa1, cqa2 = st.columns(2)
    with cqa1:
        dup_all = int(df.duplicated().sum())
        st.write(f"• **Righe duplicate**: {dup_all}")
        all_na = int((df.isna().all(axis=1)).sum())
        st.write(f"• **Righe con tutti NA**: {all_na}")
    with cqa2:
        na_by_col = df.isna().sum().sort_values(ascending=False)
        st.write("• **NA per colonna (top 10):**")
        st.dataframe(na_by_col.head(10).to_frame("NA"), use_container_width=True, height=240)

# Pulsanti di azione
a1, a2, a3 = st.columns([1.5, 1.2, 1.2])
with a1:
    if st.button("💾 Salva dataset per le pagine successive", key=k("save"), use_container_width=True):
        st.session_state["uploaded_df"] = df.copy()     # chiave principale usata dagli altri moduli
        st.session_state["cleaned_df"] = df.copy()      # opzionale: disponibile anche come 'pulito'
        st.session_state[k("saved_ok")] = True
        st.success("Dataset salvato in `uploaded_df` (e `cleaned_df`).")

with a2:
    if st.button("🧹 Vai a: Pulizia dati", key=k("go_clean"), use_container_width=True):
        if "uploaded_df" not in st.session_state:
            st.session_state["uploaded_df"] = df.copy()
        st.switch_page("pages/1_🧹_Data_Cleaning.py")

with a3:
    if st.button("📈 Vai a: Descrittive", key=k("go_desc"), use_container_width=True):
        if "uploaded_df" not in st.session_state:
            st.session_state["uploaded_df"] = df.copy()
        st.switch_page("pages/2_📈_Descriptive_Statistics.py")

st.markdown("---")

# Strumenti utili
tools1, tools2 = st.columns([1, 1])
with tools1:
    if st.button("♻️ Sostituisci dataset (reset)", key=k("reset"), use_container_width=True):
        for key in [k("df"), k("raw_file"), k("saved_ok")]:
            if key in st.session_state:
                del st.session_state[key]
        st.info("Reset eseguito. Selezioni un nuovo file o carichi il dataset di esempio.")
with tools2:
    # Download rapido del CSV così come caricato
    try:
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Scarica copia CSV",
            data=csv_bytes,
            file_name="dataset_importato.csv",
            mime="text/csv",
            key=k("download")
        )
    except Exception:
        pass

# Suggerimenti
with st.expander("ℹ️ Suggerimenti per file CSV/Excel", expanded=False):
    st.markdown(
        """
- Per CSV europei usare spesso **encoding UTF-8**, **decimale “,”** e **migliaia “.”**.  
- Se i caratteri appaiono corrotti, provare **latin-1** o **cp1252**.  
- Per Excel con più fogli, selezionare il *foglio* corretto nel menù dedicato.  
- Dopo il salvataggio, le altre pagine leggeranno i dati da **`st.session_state['uploaded_df']`**.
        """
    )
