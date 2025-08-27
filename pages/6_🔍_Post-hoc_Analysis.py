import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import scikit_posthocs as sp
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multitest import multipletests
import plotly.express as px

st.title("🔍 Post-hoc Analysis")

if "df" not in st.session_state or st.session_state.df is None:
    st.warning("⚠️ Please upload a dataset first.")
else:
    df = st.session_state.df

    num_vars = df.select_dtypes(include=np.number).columns
    cat_vars = df.select_dtypes(include=["object", "category", "bool"]).columns

    if len(num_vars) == 0 or len(cat_vars) == 0:
        st.warning("Dataset must contain both numeric and categorical variables.")
    else:
        yvar = st.selectbox("📈 Select numeric variable", num_vars)
        group = st.selectbox("🧪 Select grouping variable", cat_vars)

        data = df[[yvar, group]].dropna()
        groups = data[group].unique()
        arrays = [data[data[group] == g][yvar] for g in groups]

        if len(groups) < 3:
            st.warning("⚠️ Post-hoc tests are relevant only when there are 3 or more groups.")
        else:
            levene_stat, levene_p = stats.levene(*arrays)
            norm_stat, norm_p = stats.shapiro(data[yvar])

            is_normal = norm_p > 0.05
            use_anova = is_normal and levene_p > 0.05

            st.info(f"✅ Normality p = {norm_p:.4f}, Levene's p = {levene_p:.4f}")
            st.markdown("""
            **About Levene's Test:**
            Levene's test checks whether variances across groups are equal.
            - If **p > 0.05** → assumption of **equal variances** is met (you can use ANOVA + Tukey).
            - If **p < 0.05** → variances are unequal, prefer non-parametric tests or Bonferroni correction.
            """)

            test_type = "ANOVA" if use_anova else "Kruskal-Wallis"

            st.markdown(f"**Global test selected: `{test_type}`**")

            if test_type == "ANOVA":
                stat, p = stats.f_oneway(*arrays)
            else:
                stat, p = stats.kruskal(*arrays)

            st.success(f"{test_type} p = {p:.4f} (uncorrected)")

            # Visualizzazione Boxplot con differenze tra gruppi
            st.markdown("### 📊 Group Comparison Boxplot")
            fig = px.box(data, x=group, y=yvar, points="all", title=f"{yvar} by {group}")

            # Annotazioni per differenze significative (solo se post-hoc ANOVA e Tukey HSD è stato selezionato)
            if 'tukey_result' in locals():
                tukey_df = pd.DataFrame(tukey_result.summary().data[1:], columns=tukey_result.summary().data[0])
                sig_pairs = tukey_df[tukey_df['p-adj'].astype(float) < 0.05][['group1', 'group2']].values
                for pair in sig_pairs:
                    fig.add_annotation(
                        x=(pair[0] + ' vs ' + pair[1]),
                        y=max(data[yvar]),
                        text="*",
                        showarrow=False,
                        yshift=10
                    )
            st.plotly_chart(fig, use_container_width=True)

            # Grafico a barre con intervalli di confidenza
            st.markdown("### 📊 Bar Chart with 95% Confidence Intervals")
            ci_data = data.groupby(group)[yvar].agg(['mean', 'count', 'std'])
            ci_data['sem'] = ci_data['std'] / np.sqrt(ci_data['count'])
            ci_data['ci95'] = 1.96 * ci_data['sem']
            ci_data = ci_data.reset_index()
            fig_ci = px.bar(
                ci_data,
                x=group,
                y='mean',
                error_y='ci95',
                title=f"Mean ± 95% CI for {yvar} by {group}",
                labels={"mean": f"Mean {yvar}"},
                text='mean'
            )
            st.plotly_chart(fig_ci, use_container_width=True)

            if p > 0.05:
                st.warning("The global test is not significant. Post-hoc tests are not recommended.")
            else:
                st.markdown("### Post-hoc Test")

                if test_type == "ANOVA":
                    posthoc_option = st.radio("Select post-hoc test:", ["Tukey HSD (recommended)", "Bonferroni-corrected pairwise t-tests"])

                    if posthoc_option == "Tukey HSD (recommended)":
                        st.markdown("✅ Use when assumptions of normality and equal variances are met.")
                        tukey_result = pairwise_tukeyhsd(endog=data[yvar], groups=data[group], alpha=0.05)
                        st.dataframe(pd.DataFrame(tukey_result.summary().data[1:], columns=tukey_result.summary().data[0]))
                        st.session_state.posthoc_results = tukey_result.summary().as_text()
                    else:
                        st.markdown("⚠️ Use with caution if variances are unequal. Applies Bonferroni correction to all pairwise t-tests.")
                        from itertools import combinations
                        results = []
                        for g1, g2 in combinations(groups, 2):
                            t_stat, p_val = stats.ttest_ind(data[data[group] == g1][yvar], data[data[group] == g2][yvar])
                            results.append((f"{g1} vs {g2}", p_val))
                        labels, raw_pvals = zip(*results)
                        corrected = multipletests(raw_pvals, method='bonferroni')
                        df_res = pd.DataFrame({"Comparison": labels, "Raw p-value": raw_pvals, "Corrected p-value": corrected[1]})
                        st.dataframe(df_res)
                        st.session_state.posthoc_results = df_res.to_string()

                else:
                    st.markdown("✅ Dunn test with Bonferroni correction (non-parametric post-hoc).")
                    result = sp.posthoc_dunn(data, val_col=yvar, group_col=group, p_adjust='bonferroni')
                    st.dataframe(result)

                    # Annotazioni per Dunn test
                    sig_pairs_dunn = result.columns[(result < 0.05).any()].tolist()
                    for col in sig_pairs_dunn:
                        for idx in result.index:
                            if result.loc[idx, col] < 0.05:
                                fig.add_annotation(
                                    x=f"{col} vs {idx}",
                                    y=max(data[yvar]),
                                    text="*",
                                    showarrow=False,
                                    yshift=10
                                )
                    st.markdown("ℹ️ A star (*) indicates a pairwise comparison that is statistically significant (p < 0.05).")
                    st.session_state.posthoc_results = result.to_string()

                if st.button("📎 Save post-hoc results"):
                    st.session_state.saved_results.append("📌 Post-hoc Analysis:\n" + str(st.session_state.posthoc_results))
                    st.toast("Post-hoc results saved!", icon="📎")
