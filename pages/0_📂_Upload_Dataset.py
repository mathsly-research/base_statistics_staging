# pages/0_📂_Upload_Dataset.py
from __future__ import annotations
import io, os, re, time
import numpy as np
import pandas as pd
import streamlit as st

# ──────────────────────────────────────────────────────────────────────────────
# Data Store (preferito) + fallback locale
# ──────────────────────────────────────────────────────────────────────────────
try:
    from data_store import ensure_initialized, set_uploaded, stamp_meta
except Exception:
    # Fallback con stesse chiavi/semantica (se manca data_store.py)
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
st.set_page_config(page_title="Carica dataset", layout="wide")
try:
    from nav import sidebar
    sidebar()
except Exception:
    pass

# ──────────────────────────────────────────────────────────────────────────────
# Utility locali (routing, heuristics, ecc.)
# ──────────────────────────────────────────────────────────────────────────────
KEY = "up"
def k(name: str) -> str: return f"{KEY}_{name}"
def ss_get(name: str, default=None): return st.session_state.get(name, default)
def ss_set_default(name: str, value):
    if name not in st.session_state: st.session_state[name] = value

def _list_pages():
    try: return sorted([f for f in os.listdir("pages") if f.endswith(".py")])
    except FileNotFoundError: return []
def _norm(s: str) -> str: return re.sub(r"[^a-z0-9]+", "", str(s).lower())
def safe_switch_by_tokens(primary: list[str], tokens: list[str]):
    files = _list_pages()
    for p in primary or []:
        if p in files:
            st.switch_page(os.path.join("pages", p)); return
    toks = [_norm(t) for t in tokens]
    for f in files:
        if all(t in _norm(f) for t in toks):
            st.switch_page(os.path.join("pages", f)); return
    st.error("Pagina di destinazione non trovata in /pages.")

def is_binary_series(s: pd.Series) -> bool:
    if pd.api.types.is_bool_dtype(s): return True
    vals = pd.unique(s.dropna())
    if len(vals) == 2:
        # Consente {0,1} o due etichette
        if pd.api.types.is_numeric_dtype(s):
            return set(np.sort(vals)) <= {0,1}
        return True
    return False

def guess_id_col(df: pd.DataFrame):
    cand = [c for c in df.columns if _norm(c) in {"id","subject","patient","case","unit"}]
    if cand: return cand[0]
    # alta cardinalità ≈ id
    ratios = {c: df[c].nunique(dropna=True)/len(df) for c in df.columns}
    hi = [c for c,r in ratios.items() if r>0.8 and r<1.01]
    return hi[0] if hi else None

def guess_time_col(df: pd.DataFrame):
    names = ["time","tempo","visit","wave","period","month","year","date","datetime","timestamp","time_months","followup"]
    for n in names:
        for c in df.columns:
            if _norm(n) == _norm(c): return c
    # pattern suffix numerici in wide non hanno colonna tempo; qui proviamo date-like
    for c in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[c]): return c
    return None

def detect_shape(df: pd.DataFrame):
    info = {"shape": "wide", "id": None, "time": None, "notes": []}
    idc = guess_id_col(df); tcol = guess_time_col(df)
    info["id"], info["time"] = idc, tcol
    # LONG/PANEL se c'è id + time e ripetizioni per id
    if idc and tcol and df.duplicated([idc, tcol]).any() is False and df[idc].duplicated().any():
        info["shape"] = "long"
        info["notes"].append("Trovate misure ripetute per lo stesso ID")
    # LONG se id + time e più righe per id
    if idc and tcol and df.groupby(idc).size().max() > 1:
        info["shape"] = "long"
    # TIME SERIES se non c'è id ma c'è una colonna temporale e molte righe
    if (not idc) and tcol and len(df) >= 20:
        info["shape"] = "time-series"
    # WIDE se colonne con suffissi temporali
    pattern = re.compile(r".*(_t\d+|_m\d+|_v\d+|_visit\d+|_time\d+)$", re.IGNORECASE)
    if any(pattern.match(c) for c in df.columns):
        if info["shape"] != "long":
            info["shape"] = "wide"
        info["notes"].append("Rilevati suffissi temporali nelle colonne")
    return info

def detect_topics(df: pd.DataFrame):
    """Riconosce aree d'analisi plausibili."""
    topics = set()
    cols = df.columns
    # Sopravvivenza
    if "time" in cols and "event" in cols and is_binary_series(df["event"]) and pd.api.types.is_numeric_dtype(df["time"]):
        topics.add("survival")
    # Diagnostica (ROC/PR): variabile binaria + score/prob
    bin_cols = [c for c in cols if is_binary_series(df[c])]
    score_like = [c for c in cols if any(t in _norm(c) for t in ["score","prob","p_hat","marker","index","test"])]
    if bin_cols and score_like:
        topics.add("diagnostics")
    # Agreement
    if set(["methoda","methodb"]).issubset({_norm(c) for c in cols}):
        topics.add("agreement")
    # Regressioni (lineare/logistica)
    if any(pd.api.types.is_numeric_dtype(df[c]) for c in cols):
        topics.add("regression")
    if bin_cols:
        topics.add("logistic")
    # Descrittive e test
    topics.add("descriptives")
    topics.add("tests")
    # Longitudinale / Panel
    sh = detect_shape(df)["shape"]
    if sh == "long":
        topics.add("longitudinal")
        topics.add("panel")
    # Serie temporali
    if sh == "time-series":
        topics.add("timeseries")
    # Correlazioni
    if sum(pd.api.types.is_numeric_dtype(df[c]) for c in cols) >= 2:
        topics.add("correlation")
    return topics

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

def make_sample_primary(n: int = 300) -> pd.DataFrame:
    """Mini dataset di studio primario (coerente con i moduli)."""
    rng = np.random.default_rng(7)
    treatment = rng.integers(0, 2, size=n)
    age = np.clip(rng.normal(55, 12, size=n), 18, 90)
    bmi = np.clip(rng.normal(27, 5, size=n), 16, 50)
    diabetes = rng.binomial(1, 0.2, size=n)
    marker = rng.normal(0, 1, size=n) - 0.2*treatment + 0.2*diabetes
    y_cont = 70 + 1.2*treatment - 0.3*age - 0.4*bmi + 1.8*marker + rng.normal(0, 6, size=n)
    lin = -0.5*treatment + 0.02*(age-55) + 0.03*(bmi-27) + 0.6*diabetes - 0.5*marker
    p = 1/(1+np.exp(-lin))
    y_bin = rng.binomial(1, p)
    time = np.clip(rng.weibull(1.3, size=n)*24, 1, 48)
    event = rng.binomial(1, 0.7, size=n)
    methodA = np.clip(1 + 0.02*age + 0.03*bmi + rng.normal(0, 0.2, size=n), 0.2, 6.0)
    methodB = np.clip(methodA + 0.1 + rng.normal(0, 0.18, size=n), 0.2, 6.5)
    df = pd.DataFrame({
        "id": np.arange(1, n+1),
        "treatment": treatment, "age": np.round(age,1), "bmi": np.round(bmi,1),
        "diabetes": diabetes, "marker": np.round(marker,3),
        "y_cont": np.round(y_cont,2), "y_bin": y_bin,
        "time": np.round(time,2), "event": event,
        "methodA": np.round(methodA,3), "methodB": np.round(methodB,3)
    })
    return df

# ──────────────────────────────────────────────────────────────────────────────
# Stato e default UI
# ──────────────────────────────────────────────────────────────────────────────
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
st.markdown("Carichi un file **CSV/Excel**, verifichi l’anteprima e **salvi** per le pagine successive. "
            "Poi le suggeriremo in automatico **cosa può calcolare** in base alla struttura dei dati.")

# ──────────────────────────────────────────────────────────────────────────────
# Passo 1 · Sorgente
# ──────────────────────────────────────────────────────────────────────────────
st.subheader("Passo 1 · Caricamento dataset")
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
        sample_kind = st.selectbox("Selezioni un esempio", ["Studio primario (consigliato)", "Iris (semplice)"], key=k("sample_kind"))
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Carica esempio", use_container_width=True, key=k("load_sample_primary")):
                if sample_kind.startswith("Studio primario"):
                    df = make_sample_primary(280)
                else:
                    # Iris da web potrebbe non essere sempre disponibile: fallback sintetico
                    try:
                        df = pd.read_csv("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv")
                    except Exception:
                        df = make_sample_primary(150)[["age","bmi","marker","y_cont"]].rename(
                            columns={"age":"sepal_length","bmi":"sepal_width","marker":"petal_length","y_cont":"petal_width"}
                        )
                st.session_state[k("df")] = df.copy()
                st.success("Esempio caricato.")
        with c2:
            st.caption("L’esempio *Studio primario* è pensato per testare: descrittive, test, regressioni, "
                       "ROC/PR, agreement, sopravvivenza e (se convertito in long) longitudinale/panel.")

# ──────────────────────────────────────────────────────────────────────────────
# Passo 2 · Opzioni lettura
# ──────────────────────────────────────────────────────────────────────────────
st.subheader("Passo 2 · Conferma dati (opzioni di lettura CSV)")
c1, c2, c3, c4 = st.columns(4)
with c1:
    enc = st.selectbox("Encoding", ["utf-8", "latin-1", "cp1252"], index=0, key=k("encoding"))
with c2:
    sniff = st.checkbox("Rileva separatore (CSV)", value=True, key=k("sniff_sep"))
with c3:
    dec = st.text_input("Decimali", value=ss_get(k("decimal")), key=k("decimal"))
with c4:
    tho = st.text_input("Migliaia", value=ss_get(k("thousands")), key=k("thousands"))

if st.button("📥 Leggi/aggiorna anteprima", use_container_width=True, key=k("read")) and ss_get(k("source")) == "upload":
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
# Passo 3 · Anteprima, riconoscimento struttura e salvataggio
# ──────────────────────────────────────────────────────────────────────────────
st.subheader("Passo 3 · Anteprima, riconoscimento struttura e salvataggio")
df = st.session_state.get(k("df"))
if df is None:
    st.info("Carichi/legga un dataset per proseguire.")
    st.stop()

m1, m2, m3 = st.columns(3)
with m1: st.metric("Righe", df.shape[0])
with m2: st.metric("Colonne", df.shape[1])
with m3: st.metric("Missing totali", int(df.isna().sum().sum()))
st.dataframe(df.head(25), use_container_width=True)

# Riconoscimento struttura
info = detect_shape(df)
topics = detect_topics(df)

badge_map = {
    "wide": "🟦 Wide (una riga per soggetto, misure in colonne)",
    "long": "🟩 Long / Panel (più righe per soggetto nel tempo)",
    "time-series": "🟪 Serie temporali (indice/colonna temporale)"
}
st.markdown("#### 🔎 Riconoscimento automatico")
cA, cB, cC = st.columns([1.4, 1.4, 1.2])
with cA:
    st.info(badge_map.get(info["shape"], "Formato non determinato"))
with cB:
    st.caption(f"ID stimato: **{info['id'] or '—'}**  •  Tempo: **{info['time'] or '—'}**")
with cC:
    if info["notes"]:
        st.caption("Note: " + " · ".join(info["notes"]))

# Consigliati in base ai dati
st.markdown("#### ✅ Consigliati per i tuoi dati")
rec_cards = []

def add_card(label: str, help_txt: str, primary: list[str], tokens: list[str]):
    rec_cards.append((label, help_txt, primary, tokens))

# Base
add_card("📈 Statistiche descrittive", "Tabelle e grafici di base (continue e categoriche).",
         ["2_📈_Descriptive_Statistics.py"], ["descriptive","statistiche"])
add_card("🧪 Test statistici", "t-test, ANOVA, χ² e non parametrici.",
         ["5_🧪_Statistical_Tests.py"], ["test","statistici"])
add_card("🔗 Correlazioni", "Matrice di correlazione, heatmap e p-value.",
         ["6_🔗_Correlation_Analysis.py"], ["correlation","correlazioni"])

# Regressioni
if "regression" in topics:
    add_card("📉 Regressione lineare", "Modello lineare per outcome continuo (es. y_cont).",
             ["8_🧮_Regression.py"], ["regression","lineare"])
if "logistic" in topics:
    add_card("📈 Regressione logistica", "Modello logit per outcome binario (es. y_bin).",
             ["8_🧮_Regression.py"], ["regression","logistica"])

# Diagnostici
if "diagnostics" in topics:
    add_card("🔬 Test diagnostici (ROC/PR)", "Curve ROC/PR, sensibilità/specificità, soglia ottimale.",
             ["9_🔬_Analisi_Test_Diagnostici.py"], ["diagnostici","roc"])

# Agreement
if "agreement" in topics:
    add_card("📏 Agreement (Bland–Altman)", "Bias, LoA e grafici per due metodi.",
             ["10_📏_Agreement.py"], ["agreement","bland"])

# Sopravvivenza
if "survival" in topics:
    add_card("🧭 Sopravvivenza", "Curve di Kaplan–Meier, Cox, numeri a rischio.",
             ["11_🧭_Analisi_di_Sopravvivenza.py"], ["sopravvivenza","survival"])

# Longitudinale / Panel
if "longitudinal" in topics:
    add_card("📈 Longitudinale (misure ripetute)", "Traiettorie e modelli ad effetti misti.",
             ["12_📈_Longitudinale_Misure_Ripetute.py"], ["longitudinale"])
if "panel" in topics:
    add_card("🏷️ Panel (econometria)", "Pooled/FE/RE, Hausman, robust/clustered SE.",
             ["13_📊_Panel_Analysis.py","13_📊_Panel.py"], ["panel"])

# Serie temporali
if "timeseries" in topics:
    add_card("⏱️ Serie temporali", "ARIMA/ETS, decomposizione, previsione.",
             ["14_⏱️_Analisi_Serie_Temporali.py"], ["serie","temporali"])

# SEM e Meta (sempre disponibili, ma messi come “avanzati”)
with st.expander("Opzioni avanzate"):
    adv1, adv2 = st.columns(2)
    with adv1:
        if st.button("🧩 SEM — Modelli di equazioni strutturali", use_container_width=True, key=k("go_sem")):
            set_uploaded(df, note="from upload page"); safe_switch_by_tokens(
                ["16_🧩_SEM_Structural_Equation_Modeling.py"], ["sem","equation"]
            )
    with adv2:
        if st.button("🧪 Meta-analisi", use_container_width=True, key=k("go_meta")):
            set_uploaded(df, note="from upload page"); safe_switch_by_tokens(
                ["17_🧪_Meta_Analysis.py", "16_🧪_Meta_Analysis.py"], ["meta","analysis"]
            )

# Visualizza cards consigliate in griglia
if rec_cards:
    rows = (len(rec_cards)+2)//3
    idx = 0
    for _ in range(rows):
        c1, c2, c3 = st.columns(3)
        for col in (c1, c2, c3):
            if idx >= len(rec_cards): continue
            label, help_txt, prim, toks = rec_cards[idx]
            with col:
                st.caption(help_txt)
                if st.button(label, use_container_width=True, key=k(f"rec_{idx}")):
                    set_uploaded(df, note="from upload page")
                    safe_switch_by_tokens(prim, toks)
            idx += 1

# ──────────────────────────────────────────────────────────────────────────────
# Menù accattivante: “Cosa vuoi calcolare?”
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("❓ Cosa vuoi calcolare?")
st.markdown("Selezioni l’obiettivo: la porteremo **passo passo** al modulo giusto.")

opt = st.selectbox(
    "Scegli un obiettivo",
    [
        "— Seleziona —",
        "Descrivere le variabili",
        "Confrontare gruppi / Test",
        "Stimare una regressione lineare",
        "Stimare una regressione logistica",
        "Valutare un test diagnostico (ROC/PR)",
        "Valutare l’accordo tra due metodi (Agreement)",
        "Analizzare la sopravvivenza",
        "Analizzare dati longitudinali / misure ripetute",
        "Analisi panel (econometria)",
        "Analizzare una serie temporale",
        "SEM — Equazioni strutturali",
        "Meta-analisi"
    ],
    index=0, key=k("goal")
)

go = st.button("➡️ Vai al modulo", use_container_width=True, key=k("go_goal"))
if go and opt != "— Seleziona —":
    set_uploaded(df, note="from upload page")
    route = {
        "Descrivere le variabili": (["2_📈_Descriptive_Statistics.py"], ["descriptive","statistiche"]),
        "Confrontare gruppi / Test": (["5_🧪_Statistical_Tests.py"], ["test"]),
        "Stimare una regressione lineare": (["8_🧮_Regression.py"], ["regression","lineare"]),
        "Stimare una regressione logistica": (["8_🧮_Regression.py"], ["regression","logistica"]),
        "Valutare un test diagnostico (ROC/PR)": (["9_🔬_Analisi_Test_Diagnostici.py"], ["diagnostici","roc"]),
        "Valutare l’accordo tra due metodi (Agreement)": (["10_📏_Agreement.py"], ["agreement","bland"]),
        "Analizzare la sopravvivenza": (["11_🧭_Analisi_di_Sopravvivenza.py"], ["sopravvivenza","survival"]),
        "Analizzare dati longitudinali / misure ripetute": (["12_📈_Longitudinale_Misure_Ripetute.py"], ["longitudinale"]),
        "Analisi panel (econometria)": (["13_📊_Panel_Analysis.py","13_📊_Panel.py"], ["panel"]),
        "Analizzare una serie temporale": (["14_⏱️_Analisi_Serie_Temporali.py"], ["serie","temporali"]),
        "SEM — Equazioni strutturali": (["16_🧩_SEM_Structural_Equation_Modeling.py"], ["sem","equation"]),
        "Meta-analisi": (["17_🧪_Meta_Analysis.py", "16_🧪_Meta_Analysis.py"], ["meta"])
    }
    prim, toks = route[opt]
    safe_switch_by_tokens(prim, toks)

# ──────────────────────────────────────────────────────────────────────────────
# Salvataggio e navigazione rapida classica
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("💾 Salvataggio rapido")
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
