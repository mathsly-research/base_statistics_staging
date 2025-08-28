# pages/0_📂_Upload_Dataset.py
from __future__ import annotations
import io
import streamlit as st
import pandas as pd

# ──────────────────────────────────────────────────────────────────────────────
# Data Store (preferito) + fallback locale
# ──────────────────────────────────────────────────────────────────────────────
try:
    from data_store import ensure_initialized, set_uploaded, stamp_meta
except Exception:
    # Fallback con stesse chiavi/semantica (se manca data_store.py)
    import time
    def ensure_initialized():
        st.session_state.setdefault("ds_active_df", None)
        st.session_state.setdefault("ds_uploaded_df", None)
        st.session_state.setdefault("ds_meta", {"version": 0, "updated_at": None, "source": None, "note": ""})
    def set_uploaded(df: pd.DataFrame, note: str = ""):
        ensure_initialized()
        st.session_state["ds_uploaded_df"] = df.copy()
        # In upload impostiamo anche l'attivo
        meta = st.session_state["ds_meta"]
        st.session_state["ds_active_df"] = df.copy()
        meta["version"] = int(meta.get("version", 0)) + 1
        meta["updated_at"] = int(time.time())
        meta["source"] = "upload"
        meta["note"] = note
        st.session_state["ds_meta"] = meta
    def stamp_meta():
        ensure_initialized()
        meta = st.session_state["ds_meta"]
        ver = meta.get("version", 0)
        src = meta.get("source") or "-"
        ts  = meta.get("updated_at")
        when = "-"
        if ts:
            from datetime import datetime
            when = datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
        c1, c2, c3 = st.columns(3)
        with c1: st.metric("Versione dati", ver)
        with c2: st.metric("Origine", src)
        with c3: st.metric("Ultimo aggiornamento", when)

# ──────────────────────────────────────────────────────────────────────────────
# Config pagina + nav opzionale
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="📂 Carica dataset", layout="wide")
try:
    from nav import sidebar
    sidebar()
except Exception:
    pass

# ──────────────────────────────────────────────────────────────────────────────
# Utility locali
# ──────────────────────────────────────────────────────────────────────────────
KEY = "up"
def k(name: str) -> str: return f"{KEY}_{name}"
def ss_get(name: str, default=None): return st.session_state.get(name, default)
def ss_set_default(name: str, value):
    if name not in st.session_state: st.session_state[name] = value

ss_set_default(k("df"), None)
ss_set_default(k("source"), None)
ss_set_default(k("encoding"), "utf-8")
ss_set_default(k("sniff_sep"), True)
ss_set_default(k("decimal"), ",")
ss_set_default(k("thousands"), ".")
ensure_initialized()

# ──────────────────────────────────────────────────────────────────────────────
# Header
# ──────────────────────────────────────────────────────────────────────────────
st.title("📂 Importa Dataset")
st.markdown("Carichi un file **CSV/Excel**, verifichi l’anteprima e **salvi** per le pagine successive.")

# ──────────────────────────────────────────────────────────────────────────────
# Passo 1 · Sorgente
# ──────────────────────────────────────────────────────────────────────────────
st.subheader("Passo 1 · Sorgente dati")
left, right = st.columns([2, 3], vertical_alignment="top")

with left:
    source = st.radio("Sorgente", ["Carica file", "Usa dataset di esempio"], key=k("source_radio"))
with right:
    if source == "Carica file":
        st.session_state[k("source")] = "upload"
        uploaded = st.file_uploader("CSV/XLSX", type=["csv", "xlsx", "xls"], key=k("uploader"))
        if uploaded is not None:
            st.success(f"Selezionato: {uploaded.name}")
            st.session_state[k("raw_file")] = uploaded
    else:
        st.session_state[k("source")] = "sample"
        if st.button("Carica Iris (esempio)", use_container_width=True, key=k("load_sample")):
            df = pd.read_csv("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv")
            st.session_state[k("df")] = df.copy()
            st.success("Esempio caricato.")
            st.dataframe(df.head(15), use_container_width=True)

# ──────────────────────────────────────────────────────────────────────────────
# Passo 2 · Opzioni lettura
# ──────────────────────────────────────────────────────────────────────────────
st.subheader("Passo 2 · Opzioni di lettura")
c1, c2, c3, c4 = st.columns(4)
with c1:
    enc = st.selectbox("Encoding", ["utf-8", "latin-1", "cp1252"], index=0, key=k("encoding"))
with c2:
    sniff = st.checkbox("Rileva separatore (CSV)", value=True, key=k("sniff_sep"))
with c3:
    dec = st.text_input("Decimali", value=ss_get(k("decimal")), key=k("decimal"))
with c4:
    tho = st.text_input("Migliaia", value=ss_get(k("thousands")), key=k("thousands"))

def read_uploaded_file(file) -> pd.DataFrame | None:
    name = file.name.lower()
    if name.endswith(".csv"):
        data = file.read()
        try:
            if ss_get(k("sniff_sep"), True):
                df = pd.read_csv(io.BytesIO(data), encoding=ss_get(k("encoding")), sep=None, engine="python")
            else:
                df = pd.read_csv(io.BytesIO(data), encoding=ss_get(k("encoding")), sep=",")
        except Exception:
            df = pd.read_csv(io.BytesIO(data), encoding=ss_get(k("encoding")), sep=";")
        # conversione numerica eur/us
        dec, tho = ss_get(k("decimal")), ss_get(k("thousands"))
        if dec in [",", "."] and tho in [",", ".", " "]:
            for c in df.columns:
                if df[c].dtype == "object":
                    s = df[c].str.replace(tho, "", regex=False).str.replace(dec, ".", regex=False)
                    try:
                        num = pd.to_numeric(s, errors="ignore")
                        if pd.api.types.is_numeric_dtype(num): df[c] = num
                    except Exception:
                        pass
        return df
    elif name.endswith((".xlsx", ".xls")):
        try:
            xls = pd.ExcelFile(file)
            sheet = st.selectbox("Foglio Excel", options=xls.sheet_names, key=k("sheet"))
            df = pd.read_excel(xls, sheet_name=sheet, header=0)
            return df
        except Exception as e:
            st.error(f"Errore Excel: {e}")
            return None
    else:
        st.error("Formato non supportato.")
        return None

if st.button("📥 Leggi/aggiorna", use_container_width=True, key=k("read")) and ss_get(k("source")) == "upload":
    up = ss_get(k("raw_file"))
    if up is None:
        st.warning("Selezioni un file.")
    else:
        df = read_uploaded_file(up)
        if df is not None and not df.empty:
            st.session_state[k("df")] = df.copy()
            st.success(f"Letto: {df.shape[0]} righe × {df.shape[1]}")
        else:
            st.error("Lettura fallita o dataset vuoto.")

# ──────────────────────────────────────────────────────────────────────────────
# Passo 3 · Anteprima e salvataggio
# ──────────────────────────────────────────────────────────────────────────────
st.subheader("Passo 3 · Anteprima e salvataggio")
df = st.session_state.get(k("df"))
if df is None:
    st.info("Carichi/legga un dataset per proseguire.")
    st.stop()

m1, m2, m3 = st.columns(3)
with m1: st.metric("Righe", df.shape[0])
with m2: st.metric("Colonne", df.shape[1])
with m3: st.metric("Missing totali", int(df.isna().sum().sum()))
st.dataframe(df.head(25), use_container_width=True)

b1, b2, b3 = st.columns([1.5, 1.2, 1.2])
with b1:
    if st.button("💾 Salva per le altre pagine", use_container_width=True, key=k("save")):
        set_uploaded(df, note="from upload page")
        st.success("Salvato come ‘uploaded’ e impostato come ‘active’.")
with b2:
    if st.button("🧹 Vai a: Pulizia dati", use_container_width=True, key=k("go_clean")):
        set_uploaded(df, note="from upload page")
        st.switch_page("pages/1_🧹_Data_Cleaning.py")
with b3:
    if st.button("📈 Vai a: Descrittive", use_container_width=True, key=k("go_desc")):
        set_uploaded(df, note="from upload page")
        st.switch_page("pages/2_📈_Descriptive_Statistics.py")

with st.expander("Stato dati", expanded=False):
    stamp_meta()
