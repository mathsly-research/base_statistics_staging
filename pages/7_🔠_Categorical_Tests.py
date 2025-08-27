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
import statsmodels.api as sm

st.set_page_config(page_title="Statistical App", layout="wide")
st.title("Step 6: Statistical Tests for Categorical Variable")

if st.session_state.get("df") is not None:
    df = st.session_state.df
    cat_vars = df.select_dtypes(include=["object", "category", "bool"]).columns

    st.session_state.setdefault("test_results", [])

    if len(cat_vars) < 2:
        st.warning("⚠️ You need at least two categorical variables.")
    else:
        var1 = st.selectbox("Select first categorical variable", cat_vars)
        var2 = st.selectbox("Select second categorical variable", [v for v in cat_vars if v != var1])

        st.markdown("""
        ℹ️ This section compares two categorical variables using:
        - **Chi-square** for large samples (expected values ≥ 5).
        - **Fisher's exact test** for 2x2 tables and small samples.
        """)

        table = pd.crosstab(df[var1], df[var2])
        st.write("🔢 Contingency Table:")
        st.dataframe(table)

        test_choice = st.radio("Select test", ["Chi-square", "Fisher (only for 2x2)"])
        if test_choice == "Chi-square":
            with st.expander("❓ What is Yates' continuity correction?"):
                st.write("""
                Yates' correction is applied to the Chi-square test for 2x2 tables to adjust for the overestimation of statistical significance in small samples.
                It makes the test more conservative by reducing the Chi-square value slightly.
                Recommended when sample size is small and table is 2x2.
                """)
            use_correction = st.checkbox("Apply Yates' continuity correction (2x2 only)", value=False)

        def cramers_v(table):
            chi2, _, _, _ = stats.chi2_contingency(table)
            n = table.sum().sum()
            phi2 = chi2 / n
            r, k = table.shape
            return np.sqrt(phi2 / min(k - 1, r - 1))

        result_key = f"{test_choice}_{var1}_{var2}"
        reprocess = st.checkbox("Re-elaborate the test", value=False)

        try:
            if reprocess or result_key not in [r["Key"] for r in st.session_state.test_results]:
                if test_choice == "Fisher (only for 2x2)":
                    if table.shape == (2, 2):
                        stat, p = stats.fisher_exact(table)
                        result = sm.stats.Table2x2(table.values)
                        or_val = result.oddsratio
                        ci_low, ci_high = result.oddsratio_confint()

                        st.success(f"Fisher's exact test: OR = {or_val:.2f} (95% CI: {ci_low:.2f}–{ci_high:.2f}), p = {p:.4f}")
                        st.markdown("✅ Significant" if p < 0.05 else "⚠️ Not significant")

                        st.session_state.test_results.append({
                            "Test": "Fisher's Exact",
                            "Var1": var1,
                            "Var2": var2,
                            "Value": f"OR = {or_val:.2f}",
                            "95% CI": f"{ci_low:.2f}–{ci_high:.2f}",
                            "Effect Size": "Odds Ratio",
                            "p-value": round(p, 4),
                            "Key": result_key
                        })
                    else:
                        st.error("❌ Fisher's exact test can only be applied to 2x2 tables.")
                        st.info("ℹ️ Consider using Chi-square test instead.")
                else:
                    correction = use_correction if table.shape == (2, 2) else False
                    chi2, p, dof, expected = stats.chi2_contingency(table, correction=correction)
                    v = cramers_v(table)
                    st.success(f"Chi-square test: χ² = {chi2:.2f}, dof = {dof}, p = {p:.4f}, Cramér’s V = {v:.2f}")
                    st.markdown("✅ Significant" if p < 0.05 else "⚠️ Not significant")

                    st.session_state.test_results.append({
                        "Test": "Chi-square",
                        "Var1": var1,
                        "Var2": var2,
                        "Value": f"χ² = {chi2:.2f}",
                        "95% CI": "N/A",
                        "Effect Size": f"Cramér’s V = {v:.2f}",
                        "p-value": round(p, 4),
                        "Key": result_key
                    })
        except Exception as e:
            st.error(f"An error occurred while running the test: {e}")
            st.info("❗ Please verify that your selected variables are appropriate for the chosen test.")

        if st.session_state.test_results:
            st.subheader("📋 Summary of Categorical Tests")
            summary_cat = pd.DataFrame([{
                "Test": r["Test"],
                "Variables": f"{r['Var1']} × {r['Var2']}",
                "Statistic": r.get("Value", "-"),
                "p-value": r["p-value"],
                "Effect Size": r.get("Effect Size", "-"),
                "Interpretation": "✅ Significant" if r["p-value"] < 0.05 else "⚠️ Not significant"
            } for r in st.session_state.test_results if r["Test"] in ["Chi-square", "Fisher's Exact"]]).drop_duplicates()

            st.dataframe(summary_cat, use_container_width=True)
else:
    st.warning("⚠️ Please upload a dataset first in the 'Upload Dataset' section.")
