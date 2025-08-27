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
from scipy.stats import ttest_ind, mannwhitneyu, f_oneway, kruskal, ttest_rel, wilcoxon, sem, norm

st.set_page_config(page_title="Statistical App", layout="wide")
st.title("Step 5: Statistical Tests for Continuous Variables")

if st.session_state.get("df") is not None:
    df = st.session_state.df
    num_vars = df.select_dtypes(include=np.number).columns

    st.session_state.setdefault("results_summary", [])
    st.session_state.setdefault("saved_results", [])
    st.session_state.setdefault("saved_figures", [])
    st.session_state.setdefault("test_results", [])

    test_scope = st.radio("What kind of comparison do you want to perform?", ["Between Groups", "Within Groups (Pre/Post)"])

    def ci95(data):
        m = np.mean(data)
        s = sem(data)
        ci = norm.interval(0.95, loc=m, scale=s)
        return m, s, ci

    def get_summary(data, parametric):
        med = np.median(data)
        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)
        return f"Median (IQR) = {med:.2f} ({q1:.2f}–{q3:.2f})"

    def cohens_d(a, b):
        pooled_std = np.sqrt(((np.std(a, ddof=1) ** 2) + (np.std(b, ddof=1) ** 2)) / 2)
        return (np.mean(a) - np.mean(b)) / pooled_std

    def result_already_saved(key):
        return key in [r["Key"] for r in st.session_state.test_results]

    def show_plot(fig):
        st.plotly_chart(fig)

    def show_summary_table():
        if st.session_state.test_results:
            valid_results = [r for r in st.session_state.test_results if isinstance(r, dict) and r.get("Type") == "Continuous"]
            summary_df = pd.DataFrame([{
                "Test": r.get("Test", "N/A"),
                "Variable(s)": r.get("Variable", "N/A"),
                "Group": r.get("Group", "N/A"),
                "Statistic": r.get("Statistic", "N/A"),
                "p-value": r.get("p-value", "N/A"),
                "Effect Size": r.get("Effect Size", "N/A"),
                "Interpretation": "✅ Significant" if r.get("p-value", 1) < 0.05 else "⚠️ Not significant"
            } for r in valid_results]).drop_duplicates()
            st.subheader("🗾 Summary of All Continuous Tests")
            st.dataframe(summary_df, use_container_width=True)

    if test_scope == "Between Groups":
        cat_vars = df.select_dtypes(include=["object", "category", "bool"]).columns
        yvar = st.selectbox("Select numeric variable", num_vars)

        if len(cat_vars) > 0:
            group = st.selectbox("Select grouping variable", cat_vars)

            if group:
                try:
                    groups = df[group].dropna().unique()
                    arrays = [df[df[group] == g][yvar].dropna() for g in groups]
                    stat, p_normal = stats.shapiro(df[yvar].dropna())
                    is_normal = p_normal > 0.05
                    st.markdown(f"**Shapiro-Wilk test p = {p_normal:.4f}** → {'✅ Normal distribution' if is_normal else '⚠️ Not normal'}")

                    test_type = st.radio("Choose test", ["t-test", "Mann-Whitney", "ANOVA", "Kruskal-Wallis"])
                    ci_text = "N/A"
                    effect = "N/A"
                    summary = []
                    result_key = f"{test_type}_{yvar}_{group}"
                    reprocess_test = st.checkbox("Re-elaborate the test", value=False)

                    if not result_already_saved(result_key) or reprocess_test:
                        if test_type == "t-test":
                            stat, p = ttest_ind(*arrays)
                            effect = round(cohens_d(*arrays), 3)
                            m, s, ci = ci95(np.concatenate(arrays))
                            ci_text = f"[{ci[0]:.2f}, {ci[1]:.2f}]"
                            summary = [get_summary(arr, True) for arr in arrays]
                        elif test_type == "Mann-Whitney":
                            stat, p = mannwhitneyu(*arrays)
                            n1, n2 = len(arrays[0]), len(arrays[1])
                            U = stat
                            rb = 1 - (2 * U) / (n1 * n2)
                            effect = round(rb, 3)
                            summary = [get_summary(arr, False) for arr in arrays]
                            st.caption(f"Rank-biserial correlation = {effect} → small: .10, medium: .30, large: .50")
                        elif test_type == "ANOVA":
                            stat, p = f_oneway(*arrays)
                            m, s, ci = ci95(np.concatenate(arrays))
                            ci_text = f"[{ci[0]:.2f}, {ci[1]:.2f}]"
                            grand_mean = np.mean(np.concatenate(arrays))
                            ssb = sum([len(arr) * (np.mean(arr) - grand_mean) ** 2 for arr in arrays])
                            sst = sum([(x - grand_mean) ** 2 for arr in arrays for x in arr])
                            eta_squared = ssb / sst if sst != 0 else 0
                            effect = round(eta_squared, 3)
                            summary = [get_summary(arr, True) for arr in arrays]
                            st.caption(f"Eta squared (η²) = {effect} → small: .01, medium: .06, large: .14")
                        elif test_type == "Kruskal-Wallis":
                            stat, p = kruskal(*arrays)
                            N = sum([len(arr) for arr in arrays])
                            k = len(arrays)
                            epsilon_sq = (stat - k + 1) / (N - k) if N > k else 0
                            effect = round(epsilon_sq, 3)
                            summary = [get_summary(arr, False) for arr in arrays]
                            st.caption(f"Epsilon squared (ε²) = {effect} → small: .01, medium: .08, large: .26")

                        lines = [f"{g} (n={len(arr)}): {s}" for g, arr, s in zip(groups, arrays, summary)]
                        result_msg = f"{test_type} result (between groups on {yvar} by {group}):\nTest Statistic = {stat:.3f}, p = {p:.4f}, 95% CI = {ci_text}, Effect Size = {effect}"
                        st.success(result_msg)
                        st.write("\n".join(lines))
                        st.session_state.result_text = result_msg
                        st.session_state.results_summary.append(result_msg + "\n" + "\n".join(lines))

                        if not result_already_saved(result_key):
                            st.session_state.test_results.append({
                                "Type": "Continuous",
                                "Test": test_type,
                                "Variable": yvar,
                                "Group": group,
                                "Sample Size": str([len(arr) for arr in arrays]),
                                "Statistic": round(stat, 3),
                                "p-value": round(p, 4),
                                "CI 95%": ci_text,
                                "Effect Size": effect,
                                "Key": result_key
                            })

                    box_df = df[[group, yvar]].dropna()
                    fig = px.box(box_df, x=group, y=yvar, points="all", title="Group Comparison Boxplot")
                    show_plot(fig)

                except Exception as e:
                    if test_type in ["ANOVA", "Kruskal-Wallis"] and len(groups) < 3:
                        st.warning("⚠️ ANOVA and Kruskal-Wallis tests require a grouping variable with more than two levels.")
                    elif test_type in ["t-test", "Mann-Whitney"] and len(groups) != 2:
                        st.warning("⚠️ t-test and Mann-Whitney tests require exactly two groups.")
                    else:
                        st.error(f"❌ An unexpected error occurred: {e}")

    elif test_scope == "Within Groups (Pre/Post)":
        col1 = st.selectbox("Select pre-test variable (numeric)", num_vars, key="pre")
        col2 = st.selectbox("Select post-test variable (numeric)", num_vars, key="post")
        paired_data = df[[col1, col2]].dropna()
        diff = paired_data[col1] - paired_data[col2]
        stat, p_normal = stats.shapiro(diff)
        is_normal = p_normal > 0.05
        st.markdown(f"**Shapiro-Wilk test on difference (pre-post): p = {p_normal:.4f}** → {'✅ Normal distribution' if is_normal else '⚠️ Not normal'}")

        test_type = st.radio("Choose paired test", ["Paired t-test", "Wilcoxon"])
        try:
            ci_text = "N/A"
            effect = "N/A"
            result_key = f"{test_type}_{col1}_{col2}"
            reprocess_test = st.checkbox("Re-elaborate the test", value=False)

            if not result_already_saved(result_key) or reprocess_test:
                if test_type == "Paired t-test":
                    stat, p = ttest_rel(paired_data[col1], paired_data[col2])
                    effect = round(cohens_d(paired_data[col1], paired_data[col2]), 3)
                    m, s, ci = ci95(diff)
                    ci_text = f"[{ci[0]:.2f}, {ci[1]:.2f}]"
                    summary = [get_summary(paired_data[col1], True), get_summary(paired_data[col2], True)]
                elif test_type == "Wilcoxon":
                    stat, p = wilcoxon(paired_data[col1], paired_data[col2])
                    n = len(paired_data)
                    rb = 1 - (2 * stat) / (n * (n + 1))
                    effect = round(rb, 3)
                    summary = [get_summary(paired_data[col1], False), get_summary(paired_data[col2], False)]
                    st.caption(f"Rank-biserial correlation = {effect} → small: .10, medium: .30, large: .50")

                result_msg = f"{test_type} result (paired comparison {col1} vs {col2}):\nTest Statistic = {stat:.3f}, p = {p:.4f}, 95% CI = {ci_text}, Effect Size = {effect}"
                st.success(result_msg)
                st.write(f"Pre: {summary[0]}\nPost: {summary[1]}")
                st.session_state.result_text = result_msg
                st.session_state.results_summary.append(result_msg + f"\nPre: {summary[0]}\nPost: {summary[1]}")

                if not result_already_saved(result_key):
                    st.session_state.test_results.append({
                        "Type": "Continuous",
                        "Test": test_type,
                        "Variable": f"{col1} vs {col2}",
                        "Group": "Paired",
                        "Sample Size": len(paired_data),
                        "Statistic": round(stat, 3),
                        "p-value": round(p, 4),
                        "CI 95%": ci_text,
                        "Effect Size": effect,
                        "Key": result_key
                    })

                fig = go.Figure()
                for i in range(len(paired_data)):
                    fig.add_trace(go.Scatter(x=["Pre", "Post"], y=[paired_data[col1].iloc[i], paired_data[col2].iloc[i]],
                                             mode='lines+markers', line=dict(color='gray'), showlegend=False))
                fig.update_layout(title="Pre vs Post Paired Line Plot", yaxis_title="Value")
                show_plot(fig)
            else:
                st.info("⚠️ These test results have already been recorded.")
        except TypeError:
            st.error("⚠️ Error: Please make sure to select two different numeric variables for the pre-post test.")
            st.info("ℹ️ Both variables must contain numeric values and cannot be identical.")
        except Exception as e:
            st.error(f"❌ An unexpected error occurred: {e}")

    show_summary_table()
else:
    st.warning("⚠️ Please upload a dataset first in the 'Upload Dataset' section.")
