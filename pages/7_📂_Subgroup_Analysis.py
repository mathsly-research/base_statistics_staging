# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from scipy import stats

# Post-hoc: import sicuri
try:
    from statsmodels.stats.multicomp import pairwise_tukeyhsd
    _HAS_STATSMODELS = True
except Exception:
    _HAS_STATSMODELS = False

try:
    import scikit_posthocs as sp
    _HAS_SCIKIT_POSTHOCS = True
except Exception:
    _HAS_SCIKIT_POSTHOCS = False

from core.state import init_state

# ---------------------------------
# Utility: guide di interpretazione
# ---------------------------------
def _guide_global_test(name: str, stat: float | None, pval: float | None, alpha: float = 0.05) -> str:
    """
    Restituisce testo breve su come leggere il risultato del test globale.
    """
    lines = [f"**Come leggere {name}:**"]
    if name.lower().startswith("anova"):
        lines += [
            "- Confronta le **medie** tra più gruppi (assume normalità e varianze simili).",
            "- **F** misura il rapporto tra variabilità **tra** gruppi e **entro** gruppi.",
        ]
    elif "Kruskal" in name:
        lines += [
            "- Confronta la **posizione/ranghi** tra gruppi (robusto a non normalità e outlier).",
            "- **H** cresce quando le distribuzioni dei gruppi differiscono.",
        ]
    else:
        lines += ["- Test globale su differenze tra gruppi."]

    if pval is not None:
        if pval < alpha:
            lines.append(f"- **p = {pval:.3g} < {alpha}** → differenze **statisticamente significative** tra almeno due gruppi.")
            lines.append("- Serve un **post-hoc** per capire **quali** gruppi differiscono.")
        else:
            lines.append(f"- **p = {pval:.3g} ≥ {alpha}** → nessuna evidenza di differenze globali.")
    else:
        lines.append("- p-value non disponibile.")
    return "\n".join(lines)

def _guide_posthoc(name: str) -> str:
    if "Tukey" in name:
        return (
            "**Come leggere Tukey HSD:**\n"
            "- **group1 vs group2**: coppia di gruppi a confronto.\n"
            "- **meandiff**: differenza tra medie (group2 − group1).\n"
            "- **p-adj**: p-value **già corretto** per confronti multipli.\n"
            "- **reject** = True → differenza **significativa** per quella coppia.\n"
        )
    if "Dunn" in name:
        return (
            "**Come leggere Dunn (Bonferroni):**\n"
            "- Matrice di **p-value corretti** per ogni coppia di gruppi.\n"
            "- Celle con p < 0.05 indicano differenze **significative**.\n"
            "- Bonferroni è conservativo: con molti gruppi può ridurre la potenza."
        )
    return "**Come leggere il post-hoc:** p-value per coppie di gruppi con correzione multipla."

# ---------------------------------
# Init & check dataset
# ---------------------------------
init_state()
st.title("📂 Step 7 — Analisi di sottogruppo")

if "df" not in st.session_state or st.session_state.df is None:
    st.warning("Nessun dataset disponibile. Carichi i dati in Step 0 — Upload Dataset.")
    st.page_link("pages/0_📂_Upload_Dataset.py", label="➡️ Vai a Upload Dataset", icon="📂")
    st.stop()

# Dataset attivo: preferisci il working se esiste
df = st.session_state.df_working if "df_working" in st.session_state and st.session_state.df_working is not None else st.session_state.df
if df is None or df.empty:
    st.error("Il dataset attivo è vuoto.")
    st.stop()

# ---------------------------------
# Selezione variabili
# ---------------------------------
cat_vars = df.select_dtypes(exclude="number").columns.tolist()
num_vars = df.select_dtypes(include="number").columns.tolist()

if not cat_vars:
    st.error("Non ci sono variabili categoriche per definire sottogruppi.")
    st.stop()
if not num_vars:
    st.error("Non ci sono variabili numeriche da analizzare.")
    st.stop()

group_var = st.selectbox("Variabile di raggruppamento (sottogruppi):", cat_vars)
target_vars = st.multiselect("Variabili numeriche da analizzare:", num_vars, default=num_vars[:1])

alpha = st.slider("Soglia di significatività (α)", 0.001, 0.10, 0.05, step=0.001)

# ---------------------------------
# Analisi per ogni variabile numerica
# ---------------------------------
for target in target_vars:
    st.subheader(f"🔎 {target} per sottogruppi di {group_var}")

    # Tabella descrittive per sottogruppo
    desc = df.groupby(group_var, dropna=False)[target].agg(
        N="count", Mean="mean", Median="median", SD="std"
    ).round(3)
    st.dataframe(desc, use_container_width=True)

    # Boxplot comparativo
    fig = px.box(df, x=group_var, y=target, points="all", color=group_var,
                 title=f"Distribuzione di {target} per {group_var}")
    st.plotly_chart(fig, use_container_width=True)

    # Test statistici globali
    groups_series = [df.loc[df[group_var] == g, target].dropna() for g in df[group_var].dropna().unique()]
    has_enough = all(len(g) >= 2 for g in groups_series) and len(groups_series) >= 2

    posthoc_results = {}   # raccolta per Results Summary
    any_test_shown = False

    st.markdown("**Test statistici tra sottogruppi**")

    # ANOVA (parametrico)
    if has_enough:
        try:
            F, p_anova = stats.f_oneway(*groups_series)
            any_test_shown = True
            st.write(f"ANOVA: **F = {F:.3f}**, **p = {p_anova:.3g}**")
            st.info(_guide_global_test("ANOVA", F, p_anova, alpha))
            if p_anova < alpha and _HAS_STATSMODELS:
                st.success("ANOVA significativa → eseguo **Tukey HSD** (correzione multipla).")
                tukey = pairwise_tukeyhsd(endog=df[target].dropna(), groups=df.loc[df[target].notna(), group_var], alpha=alpha)
                tukey_df = pd.DataFrame(tukey.summary().data[1:], columns=tukey.summary().data[0])
                # tipicamente: group1, group2, meandiff, p-adj, lower, upper, reject
                st.dataframe(tukey_df, use_container_width=True)
                st.caption(_guide_posthoc("Tukey HSD"))
                posthoc_results["Tukey HSD"] = tukey_df
            elif p_anova < alpha and not _HAS_STATSMODELS:
                st.warning("Post-hoc Tukey non disponibile: installare `statsmodels`.")
        except Exception as e:
            st.info(f"ANOVA non calcolabile: {e}")

    # Kruskal–Wallis (non parametrico)
    if has_enough:
        try:
            H, p_kw = stats.kruskal(*groups_series)
            any_test_shown = True
            st.write(f"Kruskal–Wallis: **H = {H:.3f}**, **p = {p_kw:.3g}**")
            st.info(_guide_global_test("Kruskal–Wallis", H, p_kw, alpha))
            if p_kw < alpha and _HAS_SCIKIT_POSTHOCS:
                st.success("Kruskal–Wallis significativo → eseguo **Dunn test** (Bonferroni).")
                dunn = sp.posthoc_dunn(df[[group_var, target]].dropna(), val_col=target, group_col=group_var, p_adjust='bonferroni')
                st.dataframe(dunn.round(4), use_container_width=True)
                st.caption(_guide_posthoc("Dunn"))
                posthoc_results["Dunn (Bonferroni)"] = dunn.round(4)
            elif p_kw < alpha and not _HAS_SCIKIT_POSTHOCS:
                st.warning("Post-hoc Dunn non disponibile: installare `scikit-posthocs`.")
        except Exception as e:
            st.info(f"Kruskal–Wallis non calcolabile: {e}")

    if not any_test_shown:
        st.info("Gruppi troppo piccoli o dati insufficienti per eseguire i test globali.")

    # Salvataggio nel Results Summary
    if st.button(f"➕ Aggiungi {target} a Results Summary", key=f"add_{group_var}_{target}"):
        item = {
            "type": "subgroup",
            "title": f"Analisi sottogruppi — {target} per {group_var}",
            "content": desc.to_dict()
        }
        if posthoc_results:
            # Converti DataFrame in dict serializzabile
            item["posthoc"] = {k: (v.to_dict() if isinstance(v, pd.DataFrame) else v) for k, v in posthoc_results.items()}
        st.session_state.report_items.append(item)
        st.success(f"Analisi {target} aggiunta al Results Summary.")

# ---------------------------------
# Guida generale (sempre visibile on demand)
# ---------------------------------
with st.expander("ℹ️ Guida — come leggere i risultati dei test", expanded=False):
    st.markdown("""
**Test globali**
- **ANOVA (F, p)**: verifica se **almeno una media** differisce tra i gruppi (assume normalità e varianze simili).  
  - p < α → differenze globali presenti → procedere con **post-hoc**.  
- **Kruskal–Wallis (H, p)**: alternativa **non parametrica**; confronta i **ranghi**/posizioni dei gruppi.  
  - p < α → differenze globali presenti → procedere con **post-hoc**.

**Post-hoc**
- **Tukey HSD** (dopo ANOVA significativa): fornisce, per ogni **coppia** di gruppi, la **differenza di medie**, il **p-value corretto** (*p-adj*) e **reject** (True se significativo).  
- **Dunn (Bonferroni)** (dopo Kruskal–Wallis significativo): matrice di p-value **corretti** per ogni coppia; celle < α indicano coppie **significativamente diverse**.

**Buone pratiche**
- Valutare la **dimensione dell’effetto** oltre al p-value (es. differenze di mediana/di media).  
- Controllare **dimensioni campionarie** e possibili **outlier**.  
- Ricordare che **p < 0.05 ≠ importanza clinica**: interpretare nel contesto.
""")
