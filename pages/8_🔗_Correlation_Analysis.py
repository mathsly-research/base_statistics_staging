import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
import io
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_PARAGRAPH_ALIGNMENT
from PIL import Image

st.set_page_config(page_title="Statistical App", layout="wide")
st.title("Step 7: Correlation Analysis")

if st.session_state.get("df") is not None:
    df = st.session_state.df.select_dtypes(include=np.number).dropna()

    if df.shape[1] < 2:
        st.warning("⚠️ You need at least two numeric variables.")
    else:
        method = st.radio("Select correlation method", ["Pearson", "Spearman"])
        cols = df.columns
        corr_matrix = pd.DataFrame(index=cols, columns=cols, dtype=float)
        p_matrix = pd.DataFrame(index=cols, columns=cols, dtype=float)

        for col1 in cols:
            for col2 in cols:
                if method == "Pearson":
                    corr, pval = stats.pearsonr(df[col1], df[col2])
                else:
                    corr, pval = stats.spearmanr(df[col1], df[col2])
                corr_matrix.loc[col1, col2] = corr
                p_matrix.loc[col1, col2] = pval

        def interpret_effect_size(r):
            if abs(r) < 0.1:
                return "Very weak"
            elif 0.1 <= abs(r) < 0.3:
                return "Weak"
            elif 0.3 <= abs(r) < 0.5:
                return "Moderate"
            else:
                return "Strong"

        def interpret_p_value(p):
            if p < 0.001:
                return "🟢 Very strong evidence (p < 0.001)"
            elif p < 0.01:
                return "✅ Strong evidence (p < 0.01)"
            elif p < 0.05:
                return "🟡 Statistically significant (p < 0.05)"
            else:
                return "⚪ Not significant (p ≥ 0.05)"

        st.subheader(f"{method} Correlation Table with p-values, Effect Size, and Interpretation")
        results = []
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                var1, var2 = cols[i], cols[j]
                r = corr_matrix.loc[var1, var2]
                p = p_matrix.loc[var1, var2]
                effect_size = interpret_effect_size(r)
                interpretation = interpret_p_value(p)
                results.append({
                    "Variable 1": var1,
                    "Variable 2": var2,
                    "Correlation (r)": round(r, 3),
                    "p-value": round(p, 4),
                    "Effect Size": effect_size,
                    "Interpretation": interpretation
                })

        result_df = pd.DataFrame(results)
        st.dataframe(result_df)

        # Save results to session state
        st.session_state.corr_matrix = corr_matrix
        st.session_state.p_matrix = p_matrix
        st.session_state.corr_summary = result_df
        st.session_state.result_text = f"{method} correlation matrix with full interpretation saved."
        st.success("✅ Correlation analysis with effect size and interpretation saved.")
else:
    st.warning("⚠️ Please upload a dataset first in the 'Upload Dataset' section.")
