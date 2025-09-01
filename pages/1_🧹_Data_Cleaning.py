# pages/1_🧹_Data_Cleaning.py
from __future__ import annotations
import streamlit as st
import pandas as pd
import numpy as np
import unicodedata
import os, time

# ──────────────────────────────────────────────────────────────────────────────
# Data Store (preferito) + fallback
# ──────────────────────────────────────────────────────────────────────────────
try:
    from data_store import ensure_initialized, get_uploaded, get_active, set_active, stamp_meta
except Exception:
    def ensure_initialized():
        st.session_state.setdefault("ds_active_df", None)
        st.session_state.setdefault("ds_uploaded_df", None)
        st.session_state.setdefault("ds_meta", {"version": 0, "updated_at": None, "source": None, "note": ""})
    def get_uploaded():
        ensure_initialized(); return st.session_state.get("ds_uploaded_df")
    def get_active(required: bool = True):
        ensure_initialized(); df = st.session_state.get("ds_active_df")
        if required and (df is None or df.empty):
            st.error("Nessun dataset attivo. Carichi i dati nella pagina Upload."); st.stop()
        return df
    def set_active(df: pd.DataFrame, source: str = "cleaning", note: str = ""):
        ensure_initialized()
        st.session_state["ds_active_df"] = df.copy()
        meta = st.session_state["ds_meta"]; meta["version"] = int(meta.get("version", 0)) + 1
        meta["updated_at"] = int(time.time()); meta["source"] = source; meta["note"] = note
        st.session_state["ds_meta"] = meta
    def stamp_meta():
        ensure_initialized()
        meta = st.session_state["ds_meta"]; ver = meta.get("version", 0); src = meta.get("source") or "-"
        ts = meta.get("updated_at"); when = "-"
        if ts:
            from datetime import datetime; when = datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
        c1, c2, c3 = st.columns(3)
        with c1: st.metric("Versione dati", ver)
        with c2: st.metric("Origine", src)
        with c3: st.metric("Ultimo aggiornamento", when)

# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Pulizia Dati", layout="wide")
try:
    from nav import sidebar; sidebar()
except Exception:
    pass

# Utility locali
KEY = "dc"
def k(name: str) -> str: return f"{KEY}_{name}"
def ss_set_default(name: str, value):
    if name not in st.session_state: st.session_state[name] = value

# Stato
ss_set_default(k("step_done"), {"colonne": False, "tipi": False, "missing": False, "duplicati": False, "outlier": False, "testo": False})

# Header
st.title("🧹 Pulizia Dati")
ensure_initialized()

# Recupero dataset iniziale: preferiamo l'ATTIVO; se assente, l'UPLOADED
df_active = st.session_state.get("ds_active_df")
df_uploaded = get_uploaded()
df = df_active if (df_active is not None and not df_active.empty) else df_uploaded
if df is None or df.empty:
    st.error("Nessun dataset disponibile. Torni alla pagina di Upload.")
    if st.button("⬅️ Vai a Upload", use_container_width=True): st.switch_page("pages/0_📂_Upload_Dataset.py")
    st.stop()

with st.expander("Stato dati", expanded=False):
    stamp_meta()

st.success(f"Dataset corrente: {df.shape[0]} righe × {df.shape[1]} colonne.")
st.dataframe(df.head(15), use_container_width=True)

# Barra avanzamento
done = st.session_state[k("step_done")]
st.progress(sum(done.values())/len(done), text=f"Passi completati: {sum(done.values())}/{len(done)}")

# ============ PASSO 1: COLONNE ============
st.subheader("Passo 1 · Colonne")
cols = df.columns.tolist()
c1, c2 = st.columns([2,1])
with c1:
    keep = st.multiselect("Selezioni le colonne da mantenere", options=cols, default=cols, key=k("keep"))
with c2:
    drop_const = st.checkbox("Rimuovi colonne costanti", value=True, key=k("drop_const"))
if st.button("Applica colonne", use_container_width=True, key=k("apply_cols")):
    try:
        tmp = df[keep].copy()
        if drop_const:
            consts = tmp.columns[tmp.nunique(dropna=False) <= 1].tolist()
            if consts: tmp = tmp.drop(columns=consts); st.info(f"Rimosse costanti: {consts}")
        set_active(tmp, source="cleaning", note="columns")
        st.session_state[k("step_done")]["colonne"] = True
        st.success("Colonne aggiornate.")
    except Exception as e:
        st.error(f"Errore: {e}")

df = get_active()  # refresh dopo modifica
st.dataframe(df.head(8), use_container_width=True)

# ============ PASSO 2: TIPI ============
st.subheader("Passo 2 · Tipi di dato")
now_cols = df.columns.tolist()
c1, c2, c3 = st.columns(3)
with c1: to_num = st.multiselect("→ in numerico", options=now_cols, key=k("to_num"))
with c2: to_dt  = st.multiselect("→ in data/ora", options=now_cols, key=k("to_dt"))
with c3: to_cat = st.multiselect("→ in categoriale", options=now_cols, key=k("to_cat"))
opt1, opt2 = st.columns(2)
with opt1: dec_comma = st.checkbox("Riconosci virgola decimale", value=True, key=k("dec"))
with opt2: coerce = st.checkbox("Errori → NA (coerce)", value=True, key=k("coerce"))
if st.button("Applica tipi", use_container_width=True, key=k("apply_types")):
    try:
        tmp = df.copy()
        for c in to_num:
            s = tmp[c].astype(str) if tmp[c].dtype == "object" else tmp[c]
            if dec_comma and s.dtype == "object":
                s = s.str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
            tmp[c] = pd.to_numeric(s, errors=("coerce" if coerce else "raise"))
        for c in to_dt:
            tmp[c] = pd.to_datetime(tmp[c], errors=("coerce" if coerce else "raise"))
        for c in to_cat:
            tmp[c] = tmp[c].astype("category")
        set_active(tmp, source="cleaning", note="dtypes")
        st.session_state[k("step_done")]["tipi"] = True
        st.success("Tipi aggiornati.")
    except Exception as e:
        st.error(f"Errore: {e}")

df = get_active()
st.dataframe(df.head(8), use_container_width=True)

# ============ PASSO 3: MISSING ============
st.subheader("Passo 3 · Missing values")
c1, c2, c3 = st.columns(3)
with c1: drop_any = st.checkbox("Drop righe con ≥1 NA", value=False, key=k("drop_any"))
with c2: drop_all = st.checkbox("Drop righe con tutti NA", value=True, key=k("drop_all"))
with c3: imp_num = st.selectbox("Imputazione numerica", ["Nessuna","Media","Mediana"], key=k("imp_num"))
d1, d2 = st.columns(2)
with d1: imp_cat = st.selectbox("Imputazione categoriale", ["Nessuna","Moda"], key=k("imp_cat"))
with d2: cols_sel = st.multiselect("Colonne (vuoto = tutte)", options=df.columns.tolist(), key=k("imp_cols"))
if st.button("Applica missing", use_container_width=True, key=k("apply_missing")):
    try:
        tmp = df.copy()
        if drop_all: tmp = tmp.dropna(how="all")
        if drop_any: tmp = tmp.dropna(how="any")
        targets = cols_sel or tmp.columns.tolist()
        if imp_num != "Nessuna":
            for c in [x for x in targets if pd.api.types.is_numeric_dtype(tmp[x])]:
                val = tmp[c].mean() if imp_num == "Media" else tmp[c].median()
                tmp[c] = tmp[c].fillna(val)
        if imp_cat == "Moda":
            for c in [x for x in targets if tmp[x].dtype == "object" or str(tmp[x].dtype).startswith("category")]:
                if not tmp[c].dropna().empty:
                    tmp[c] = tmp[c].fillna(tmp[c].mode(dropna=True).iloc[0])
        set_active(tmp, source="cleaning", note="missing")
        st.session_state[k("step_done")]["missing"] = True
        st.success("Missing gestiti.")
    except Exception as e:
        st.error(f"Errore: {e}")

df = get_active()
st.dataframe(df.head(8), use_container_width=True)

# ============ PASSO 4: DUPLICATI ============
st.subheader("Passo 4 · Duplicati")
s1, s2 = st.columns([2,1])
with s1: dup_cols = st.multiselect("Colonne chiave (vuoto = tutte)", options=df.columns.tolist(), key=k("dup_cols"))
with s2: dup_keep = st.selectbox("Conserva", ["first","last"], key=k("dup_keep"))
if st.button("Rimuovi duplicati", use_container_width=True, key=k("apply_dups")):
    try:
        before = df.shape[0]
        tmp = df.drop_duplicates(subset=(dup_cols or None), keep=dup_keep)
        st.info(f"Righe rimosse: {before - tmp.shape[0]}")
        set_active(tmp, source="cleaning", note="duplicates")
        st.session_state[k("step_done")]["duplicati"] = True
        st.success("Duplicati rimossi.")
    except Exception as e:
        st.error(f"Errore: {e}")

df = get_active()
st.dataframe(df.head(8), use_container_width=True)

# ============ PASSO 5: OUTLIER ============
st.subheader("Passo 5 · Outlier")
num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
c1, c2, c3 = st.columns(3)
with c1: out_cols = st.multiselect("Colonne numeriche", options=num_cols, default=num_cols[:1], key=k("out_cols"))
with c2: method = st.selectbox("Metodo", ["IQR (1.5×)","SD (3×)"], key=k("out_method"))
with c3: action = st.selectbox("Azione", ["Segna (nessuna)","Winsorize","Rimuovi righe"], key=k("out_action"))
if st.button("Applica outlier", use_container_width=True, key=k("apply_out")):
    try:
        tmp = df.copy()
        for c in out_cols:
            s = tmp[c]
            if method.startswith("IQR"):
                q1, q3 = s.quantile(0.25), s.quantile(0.75)
                iqr = q3 - q1; low, high = q1 - 1.5*iqr, q3 + 1.5*iqr
            else:
                mu, sd = s.mean(), s.std(ddof=0); low, high = mu - 3*sd, mu + 3*sd
            mask = (s < low) | (s > high)
            if action == "Winsorize":
                tmp.loc[s < low, c] = low; tmp.loc[s > high, c] = high
            elif action == "Rimuovi righe":
                tmp = tmp[~mask]
        set_active(tmp, source="cleaning", note="outliers")
        st.session_state[k("step_done")]["outlier"] = True
        st.success("Outlier gestiti.")
    except Exception as e:
        st.error(f"Errore: {e}")

df = get_active()
st.dataframe(df.head(8), use_container_width=True)

# ============ PASSO 6: TESTO ============
st.subheader("Passo 6 · Testo/Categorie")
obj_cols = [c for c in df.columns if df[c].dtype == "object" or str(df[c].dtype).startswith("category")]
c1, c2, c3 = st.columns(3)
with c1: trim = st.checkbox("Trim spazi", value=True, key=k("trim"))
with c2: lower = st.checkbox("Minuscole", value=False, key=k("lower"))
with c3: deacc = st.checkbox("Rimuovi accenti", value=False, key=k("deacc"))
c4, c5 = st.columns(2)
with c4: collapse = st.checkbox("Collassa spazi multipli", value=True, key=k("collapse"))
with c5: std_bool = st.checkbox("Standardizza sì/no", value=True, key=k("stdbool"))

def _strip_acc(s: str) -> str:
    nfkd = unicodedata.normalize('NFKD', s)
    return "".join([c for c in nfkd if not unicodedata.combining(c)])

if st.button("Applica normalizzazione testo", use_container_width=True, key=k("apply_txt")):
    try:
        tmp = df.copy()
        for c in obj_cols:
            s = tmp[c].astype(str)
            if trim: s = s.str.strip()
            if collapse: s = s.str.replace(r"\s+", " ", regex=True)
            if lower: s = s.str.lower()
            if deacc: s = s.apply(_strip_acc)
            if std_bool:
                mapping = {"si":"sì","yes":"sì","y":"sì","1":"sì","vero":"sì","true":"sì",
                           "no":"no","n":"no","0":"no","falso":"no","false":"no"}
                s = s.str.lower().map(mapping).fillna(s)
            tmp[c] = s
        set_active(tmp, source="cleaning", note="text norm")
        st.session_state[k("step_done")]["testo"] = True
        st.success("Normalizzazione testo applicata.")
    except Exception as e:
        st.error(f"Errore: {e}")

# ============ NUOVA SEZIONE: SALVA / ESPORTA ============
st.markdown("---")
st.subheader("💾 Salva / Esporta dati puliti")

c1, c2, c3 = st.columns(3)

with c1:
    if st.button("💾 Salva versione (in /mnt/data)", use_container_width=True, key=k("save_ver")):
        try:
            ensure_initialized()
            df_current = get_active(required=True)
            meta = st.session_state["ds_meta"]
            ver = int(meta.get("version", 0))
            ts = int(time.time())
            fname = f"cleaned_v{ver}_{ts}.csv"
            path = os.path.join("/mnt/data", fname)
            df_current.to_csv(path, index=False)
            st.session_state.setdefault("ds_saved_versions", [])
            st.session_state["ds_saved_versions"].append(
                {"version": ver, "path": path, "ts": ts, "rows": int(len(df_current))}
            )
            st.session_state[k("last_saved_path")] = path
            st.success(f"Versione salvata: {path}")
        except Exception as e:
            st.error(f"Errore salvataggio: {e}")

with c2:
    try:
        csv_bytes = get_active(required=True).to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Scarica CSV",
            data=csv_bytes,
            file_name="dati_puliti.csv",
            mime="text/csv",
            use_container_width=True,
            key=k("dl_csv")
        )
    except Exception as e:
        st.info("Nessun dataset attivo da scaricare.")

with c3:
    if st.button("📌 Conferma come versione attiva", use_container_width=True, key=k("confirm_active")):
        try:
            # Non modifica i dati, ma aggiorna meta/versione per "blocco" esplicito
            set_active(get_active(required=True), source="cleaning", note="manual confirm")
            st.success("Versione corrente confermata come attiva.")
        except Exception as e:
            st.error(f"Errore: {e}")

# Storico versioni salvate
saved = st.session_state.get("ds_saved_versions", [])
if saved:
    with st.expander("Versioni salvate (storico)", expanded=False):
        hist = pd.DataFrame(saved)
        if not hist.empty:
            # Formattazioni minime
            hist = hist.sort_values("ts", ascending=False)
            pretty = hist.assign(
                when=hist["ts"].apply(lambda t: time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t)))
            )[["version", "rows", "when", "path"]]
            st.dataframe(pretty, use_container_width=True)

# Navigazione
st.markdown("---")
n1, n2 = st.columns(2)
with n1:
    if st.button("📈 Vai a: Descrittive", use_container_width=True, key=k("go_desc")):
        st.switch_page("pages/2_📈_Descriptive_Statistics.py")
with n2:
    if st.button("📄 Vai a: Longitudinale", use_container_width=True, key=k("go_long")):
        st.switch_page("pages/12_📈_Longitudinale_Misure_Ripetute.py")
