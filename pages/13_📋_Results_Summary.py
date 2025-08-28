# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import uuid, datetime as dt, json

from core.state import init_state
import plotly.graph_objects as go

# Opzionale: Markdown -> HTML (per export)
try:
    import markdown as mdconv
    _HAS_MD = True
except Exception:
    _HAS_MD = False

# ===========================================================
# Setup & stile globale
# ===========================================================
st.set_page_config(page_title="Results Summary", layout="wide")
init_state()
st.title("📋 Results Summary — Dashboard a card")

# Larghezza container e stile card
st.markdown("""
<style>
.block-container {max-width: 1400px; padding-top: 1rem; padding-bottom: 1rem;}
.card {border-radius:16px; padding:16px; margin-bottom:16px; box-shadow:0 2px 12px rgba(0,0,0,.06);}
.card h4{margin:0 0 6px 0}
.meta{color:#616161; font-size:.85em; margin-bottom:6px}
.pill{display:inline-block; padding:4px 10px; border-radius:999px; background:#eef1f4; margin-right:6px; font-size:.85em}
.kpi{font-weight:600}
</style>
""", unsafe_allow_html=True)

# Archivio risultati in sessione
if "report_items" not in st.session_state or not isinstance(st.session_state.report_items, list):
    st.session_state.report_items = []

# Backfill campi minimi e metadati persistenti
for it in st.session_state.report_items:
    it.setdefault("id", str(uuid.uuid4()))
    it.setdefault("created_at", dt.datetime.now().isoformat(timespec="seconds"))
    it.setdefault("title", it.get("type", "Item"))
    it.setdefault("content", {})
    # metadati persistenti per la relazione
    meta = it.setdefault("_rs_meta", {})  # <- QUI si salvano note/selezione/pin/ordine
    meta.setdefault("note", "")
    meta.setdefault("selected", False)
    meta.setdefault("pinned", False)
    # default d’ordine per sezione (vedi mapping sotto)
    # lo impostiamo dopo aver risolto la sezione

items = st.session_state.report_items

# ===========================================================
# Mapping stile/sezione/ordine
# ===========================================================
TYPE_STYLE = {
    "regression_ols":               ("🧮", "#e8f5e9"),
    "regression_logit":             ("🧮", "#e8f5e9"),
    "regression_poisson":           ("🧮", "#e8f5e9"),
    "regression_logit_regularized": ("🧮", "#e8f5e9"),
    "diagnostic_test":              ("🔬", "#fff7e6"),
    "agreement_continuous":         ("📏", "#e6f0ff"),
    "agreement_icc":                ("📏", "#e6f0ff"),
    "agreement_kappa":              ("📏", "#e6f0ff"),
    "survival_analysis":            ("🧭", "#e6fff7"),
    "longitudinal_lmm":             ("📉", "#f3e8ff"),
    "longitudinal_gee":             ("📉", "#f3e8ff"),
}

SECTION_OF = {
    "regression_ols": "Modelli di regressione",
    "regression_logit": "Modelli di regressione",
    "regression_poisson": "Modelli di regressione",
    "regression_logit_regularized": "Modelli di regressione",
    "diagnostic_test": "Test diagnostici",
    "agreement_continuous": "Agreement",
    "agreement_icc": "Agreement",
    "agreement_kappa": "Agreement",
    "survival_analysis": "Analisi di sopravvivenza",
    "longitudinal_lmm": "Dati longitudinali",
    "longitudinal_gee": "Dati longitudinali",
}

DEFAULT_SECTION_ORDER = {
    "Test diagnostici": 1,
    "Modelli di regressione": 2,
    "Agreement": 3,
    "Analisi di sopravvivenza": 4,
    "Dati longitudinali": 5,
    "Altri risultati": 6,
}

def _icon_bg(item_type: str):
    return TYPE_STYLE.get(item_type, ("🗂️", "#f5f5f5"))

# Imposta order di default nei metadati se mancante
for it in items:
    sec = SECTION_OF.get(it.get("type",""), "Altri risultati")
    it["_rs_meta"].setdefault("order", DEFAULT_SECTION_ORDER.get(sec, 99))

# ===========================================================
# Helpers per sintesi e anteprima
# ===========================================================
def _short_line(item: dict) -> str:
    t = (item.get("type") or "").lower()
    c = item.get("content", {})
    try:
        if t == "regression_ols":
            return f"R² {c.get('r2', np.nan):.3f} • AIC {c.get('aic', np.nan):.1f}"
        if t == "regression_logit":
            pr2 = c.get("pseudo_r2_mcfadden", np.nan)
            return f"Pseudo-R² {pr2:.3f} • AIC {c.get('aic', np.nan):.1f}"
        if t == "regression_poisson":
            return f"Dev {c.get('deviance', np.nan):.1f} • AIC {c.get('aic', np.nan):.1f}"
        if t == "regression_logit_regularized":
            return f"{c.get('penalty','pen')}, C={c.get('C','?')}"
        if t == "diagnostic_test":
            mets = pd.DataFrame(c.get("metrics", []))
            def pick(m):
                try: return float(mets.loc[mets["Metrica"]==m, "Valore"].values[0])
                except: return np.nan
            return f"Sens {pick('Sensibilità'):.2f} • Spec {pick('Specificità'):.2f}"
        if t == "agreement_continuous":
            ba = pd.DataFrame(c.get("bland_altman", []))
            if not ba.empty:
                try:
                    bias = float(ba.loc[ba["Parametro"]=="Bias","Valore"].values[0])
                    return f"Bias {bias:.3g}"
                except: pass
            return "Bland–Altman"
        if t == "agreement_icc":
            return f"ICC(2,1) {item['content'].get('icc21', np.nan):.3f}"
        if t == "agreement_kappa":
            return f"Kappa {item['content'].get('kappa', np.nan):.3f}"
        if t == "survival_analysis":
            N = c.get("N"); ev = c.get("events")
            return f"N {N} • Eventi {ev}"
        if t == "longitudinal_lmm":
            rm = c.get("r2_marginal", np.nan); rc = c.get("r2_conditional", np.nan)
            return f"R²m {rm:.2f} • R²c {rc:.2f}"
        if t == "longitudinal_gee":
            return f"Corr {item['content'].get('cov_struct','—')}"
    except Exception:
        pass
    return ""

def _preview_table(item: dict) -> pd.DataFrame:
    t = (item.get("type") or "").lower()
    c = item.get("content", {})
    try:
        if t == "regression_ols":
            return pd.DataFrame(c.get("coefficients", [])).head(8)
        if t == "regression_logit":
            return pd.DataFrame(c.get("odds_ratios", [])).head(8)
        if t == "regression_poisson":
            return pd.DataFrame(c.get("irr", [])).head(8)
        if t == "diagnostic_test":
            return pd.DataFrame(c.get("metrics", [])).head(8)
        if t == "agreement_continuous":
            return pd.DataFrame(c.get("bland_altman", [])).iloc[:8, :]
        if t == "agreement_icc":
            d = item["content"].copy()
            return pd.DataFrame(list(d.items()), columns=["Parametro", "Valore"])
        if t == "agreement_kappa":
            d = {"Kappa": c.get("kappa"), "Weighted": c.get("weighted")}
            return pd.DataFrame(list(d.items()), columns=["Parametro", "Valore"])
        if t == "survival_analysis":
            if "cox_hr" in c:
                return pd.DataFrame(c["cox_hr"]).head(8)
            return pd.DataFrame(c.get("km_summary", []))
        if t == "longitudinal_lmm":
            return pd.DataFrame(c.get("fixed_effects", [])).head(8)
        if t == "longitudinal_gee":
            return pd.DataFrame(c.get("coefficients", [])).head(8)
        return pd.DataFrame(list(c.items()), columns=["Chiave", "Valore"])
    except Exception:
        return pd.DataFrame()

# ===========================================================
# Testata: dataset & conteggi
# ===========================================================
hdrL, hdrR = st.columns([3, 2])
with hdrL:
    if "df_original" in st.session_state and isinstance(st.session_state.df_original, pd.DataFrame):
        df0 = st.session_state.df_original
        c1, c2, c3 = st.columns(3)
        c1.metric("Osservazioni", df0.shape[0])
        c2.metric("Variabili", df0.shape[1])
        c3.metric("Elementi salvati", len(items))
    else:
        st.info("Carichi un dataset nello Step 0 per mostrare i dettagli qui.")

with hdrR:
    if items:
        sec_series = pd.Series([SECTION_OF.get(it.get("type", ""), "Altri risultati") for it in items]).value_counts()
        fig = go.Figure(data=[go.Pie(labels=sec_series.index.tolist(), values=sec_series.values.tolist(), hole=0.55)])
        fig.update_layout(height=220, margin=dict(t=10, b=0, l=0, r=0), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Distribuzione per sezione")

st.divider()

# ===========================================================
# Filtri essenziali
# ===========================================================
sec_options = sorted({SECTION_OF.get(it.get("type", ""), "Altri risultati") for it in items})
colF1, colF2 = st.columns([3, 2])
with colF1:
    sec_sel = st.multiselect("Sezioni da mostrare", options=sec_options, default=sec_options)
with colF2:
    query = st.text_input("Cerca nel titolo", "")

filtered = [it for it in items
            if SECTION_OF.get(it.get("type", ""), "Altri risultati") in sec_sel
            and (query.lower() in it.get("title", "").lower())]

# ===========================================================
# Card grid (2 colonne ampie)
# ===========================================================
st.subheader("Risultati")
if not filtered:
    st.info("Nessun elemento corrisponde ai filtri.")
else:
    cols = st.columns([1, 1])  # due colonne uguali ma con container wide
    for idx, it in enumerate(filtered):
        col = cols[idx % 2]
        with col:
            icon, bg = _icon_bg(it.get("type", ""))
            sec = SECTION_OF.get(it.get("type", ""), "Altri risultati")
            _id = it["id"]
            meta = it["_rs_meta"]  # <- metadati persistenti

            st.markdown(f"<div class='card' style='background:{bg}'>", unsafe_allow_html=True)
            cTop, cBtns = st.columns([5, 2])  # più spazio al testo
            with cTop:
                st.markdown(f"### {icon} {it.get('title','(senza titolo)')}")
                st.markdown(f"<div class='meta'>{sec} • {it.get('created_at','')}</div>", unsafe_allow_html=True)
                line = _short_line(it)
                if line:
                    st.markdown(f"<span class='pill kpi'>{line}</span>", unsafe_allow_html=True)
            with cBtns:
                sel = st.checkbox("Includi", key=f"sel_{_id}", value=meta.get("selected", False), help="Includi nel report")
                pin = st.checkbox("Pin", key=f"pin_{_id}", value=meta.get("pinned", False), help="Evidenzia nel report")
                ordv = st.number_input("Ordine", key=f"ord_{_id}", min_value=1, max_value=99,
                                       value=int(meta.get("order", DEFAULT_SECTION_ORDER.get(sec, 99))), step=1)

            # Anteprima compatta
            with st.expander("Anteprima tabellare", expanded=False):
                tab = _preview_table(it)
                if not tab.empty:
                    st.dataframe(tab.round(4), use_container_width=True)
                else:
                    st.info("Anteprima non disponibile per questo elemento.")

            # Nota interpretativa (entra nel report) — PERSISTENTE
            nota = st.text_area(
                "Nota/interpretazione (includi nel report)",
                value=meta.get("note", ""),
                key=f"note_{_id}",
                height=90
            )
            st.markdown("</div>", unsafe_allow_html=True)

            # ======= PERSISTENZA METADATI NELL'ITEM =======
            meta["selected"] = bool(sel)
            meta["pinned"] = bool(pin)
            meta["order"] = int(ordv)
            meta["note"] = str(nota)

st.divider()

# ===========================================================
# Composer del report
# ===========================================================
st.subheader("Composer del report")
colC1, colC2 = st.columns([3, 2])
with colC1:
    rpt_title = st.text_input("Titolo relazione", "Relazione statistica")
    rpt_author = st.text_input("Autore/i", "")
    rpt_affil = st.text_input("Affiliazione (opzionale)", "")
with colC2:
    rpt_summary = st.text_area("Abstract / Sintesi iniziale (opzionale)", height=120)

# Raccoglie elementi selezionati dai metadati PERSISTENTI
chosen = [it for it in items if it["_rs_meta"].get("selected", False)]

# Ordinamento: pin (prima), ordine sezione, ordine manuale, titolo
def _sort_key(it):
    sec = SECTION_OF.get(it.get("type", ""), "Altri risultati")
    base = DEFAULT_SECTION_ORDER.get(sec, 99)
    pin_bonus = 0 if it["_rs_meta"].get("pinned", False) else 1
    manual = it["_rs_meta"].get("order", base)
    return (pin_bonus, base, manual, it.get("title", ""))

chosen = sorted(chosen, key=_sort_key)

# Funzioni di composizione report
def _md_table(df: pd.DataFrame) -> str:
    try:
        return df.to_markdown(index=False)
    except Exception:
        return "*(tabella non disponibile)*"

def _build_markdown(title, author, affil, abstract, items_sel):
    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    md = [f"# {title}", f"*Data:* {now}  "]
    if author: md.append(f"*Autore/i:* {author}  ")
    if affil: md.append(f"*Affiliazione:* {affil}  ")
    md.append("")
    if abstract:
        md += ["## Sintesi", abstract, ""]

    # Sezione dataset (se disponibile)
    if "df_original" in st.session_state and isinstance(st.session_state.df_original, pd.DataFrame):
        df = st.session_state.df_original
        md += ["## Dataset",
               f"- Osservazioni: **{df.shape[0]}**",
               f"- Variabili: **{df.shape[1]}**", ""]

    # Raggruppa per sezione
    groups = {}
    for it in items_sel:
        sec = SECTION_OF.get(it.get("type", ""), "Altri risultati")
        groups.setdefault(sec, []).append(it)

    # Ordine sezioni
    ordered_secs = sorted(groups.keys(), key=lambda s: DEFAULT_SECTION_ORDER.get(s, 99))

    # Sezioni
    for sec in ordered_secs:
        md += [f"## {sec}", ""]
        for it in groups[sec]:
            md += [f"### {it.get('title', 'Sezione')}",
                   f"*Tipo:* `{it.get('type', '')}`  "]
            line = _short_line(it)
            if line: md.append(line)
            # Tabella chiave
            tab = _preview_table(it)
            if not tab.empty:
                md += ["", _md_table(tab.round(4))]
            # Nota interpretativa (presa dai metadati dell'item)
            note = it["_rs_meta"].get("note", "").strip()
            if note:
                md += ["", "**Nota/interpretazione**", note]
            md.append("")

    # Sezioni conclusione/limiti (bozze modificabili)
    md += ["## Limitazioni (bozza)",
           "- Disegno, assunzioni (normalità, PH, indipendenza) e loro verifica.",
           "- Possibili bias di selezione/informazione; dati mancanti (MAR/MNAR).",
           ""]
    md += ["## Conclusioni (bozza)",
           "Riepilogo dei risultati principali, implicazioni e raccomandazioni operative.",
           ""]
    return "\n".join(md)

md_text = _build_markdown(rpt_title, rpt_author, rpt_affil, rpt_summary, chosen)

# ===========================================================
# Export
# ===========================================================
colD1, colD2, colD3 = st.columns(3)
with colD1:
    st.download_button("⬇️ Scarica Markdown", data=md_text, file_name="relazione_statistica.md", mime="text/markdown")
with colD2:
    if _HAS_MD:
        html_body = mdconv.markdown(md_text, extensions=['tables','fenced_code'])
        html_full = f"""<!doctype html><html><head><meta charset="utf-8">
        <style>
        body{{font-family:-apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif; margin:40px}}
        table{{border-collapse:collapse}} th,td{{border:1px solid #ccc; padding:6px 8px}}
        </style></head><body>{html_body}</body></html>"""
    else:
        html_full = "<pre style='white-space:pre-wrap'>" + md_text + "</pre>"
    st.download_button("⬇️ Scarica HTML", data=html_full, file_name="relazione_statistica.html", mime="text/html")
with colD3:
    # Export JSON include ANCHE i metadati persistenti _rs_meta
    payload = json.dumps(items, ensure_ascii=False, indent=2)
    st.download_button("⬇️ Esporta archivio JSON", data=payload,
                       file_name="results_summary_archive.json", mime="application/json")

# ===========================================================
# Manutenzione
# ===========================================================
with st.expander("Operazioni di manutenzione", expanded=False):
    c1, c2 = st.columns(2)
    with c1:
        if st.button("🧹 Svuota Results Summary"):
            st.session_state.report_items = []
            st.success("Archivio svuotato.")
            st.stop()
    with c2:
        if st.button("🔄 Azzera metadati (note/pin/ordine/selected)"):
            for it in st.session_state.report_items:
                it["_rs_meta"] = {"note":"", "selected":False, "pinned":False,
                                  "order": DEFAULT_SECTION_ORDER.get(SECTION_OF.get(it.get("type",""),"Altri risultati"), 99)}
            st.success("Metadati azzerati.")
