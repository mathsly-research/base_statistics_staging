# 📋 Results Summary - Ottimizzato
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import io

st.title("Results Summary")

if st.session_state.get("df") is not None:
    df = st.session_state.df

    st.markdown("""
    <div style='font-weight:600; color:green;'>✅ Welcome to the Results Summary Panel!</div>
    <p>This is your dashboard to review the results of your analysis. Each box below contains key information, simplified for clarity.</p>
    """, unsafe_allow_html=True)

    with st.expander("Legend"):
        st.markdown("""
        <style>
            .legend-box {
                padding: 12px;
                border-radius: 6px;
                background-color: #f0f4ff;
                border-left: 4px solid #1a73e8;
                margin-bottom: 10px;
            }
            .legend-box .legend-title {
                font-weight: bold;
                color: #1a53ff;
                margin-bottom: 6px;
            }
            .legend-box ul {
                margin: 0; padding-left: 20px;
            }
        </style>
        <div class='legend-box'>
            <div class='legend-title'>Legend</div>
            <ul>
                <li>✔️ Data included</li>
                <li>❗ Result not yet available</li>
                <li>📊 Statistical tests show differences or associations</li>
                <li>🔗 Correlations show relationships between variables</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    # --- Continuous Summary --- #
    st.subheader("📊 Continuous Variables Summary")
    cont_df = df.select_dtypes(include=["number"])
    if not cont_df.empty:
        summary = pd.DataFrame({
            "Mean ± SD": cont_df.mean().round(2).astype(str) + " ± " + cont_df.std().round(2).astype(str),
            "Median (25°–75°)": cont_df.median().round(2).astype(str) +
                " (" + cont_df.quantile(0.25).round(2).astype(str) + "–" + cont_df.quantile(0.75).round(2).astype(str) + ")",
            "Type": "Continuous"
        })
        st.session_state.summary_cont = summary.drop(columns="Type")
        st.dataframe(summary.drop(columns="Type"), use_container_width=True)
    else:
        st.info("❗ No numeric variables found.")

    # --- Categorical Summary --- #
    st.subheader("🧮 Categorical Variables Overview")
    cat_vars = df.select_dtypes(include=["object", "category", "bool"]).columns
    cat_summary = {}
    cat_combined = []

    for col in cat_vars:
        freq = df[col].value_counts(dropna=False)
        total = freq.sum()
        freq_df = freq.to_frame("Count")
        freq_df["Percent"] = round(100 * freq_df["Count"] / total, 1)
        freq_df.index.name = col
        freq_df.reset_index(inplace=True)
        freq_df.set_index(col, inplace=True)

        df_filtered = freq_df.loc[freq_df.index != "Total"]
        df_filtered = freq_df.loc[freq_df.index != "Total"]
        row_texts = [f"{int(c)} ({p}%)" for c, p in zip(df_filtered["Count"], df_filtered["Percent"])]
        cat_combined.append({
            "Variable": col,
            "Mean ± SD": "",
            "Median (25°–75°)": "; ".join(row_texts),
            "Type": "Categorical"
        })

        cat_summary[col] = freq_df
        with st.expander(f"View: {col}"):
            st.dataframe(freq_df, use_container_width=True)

    st.session_state.summary_cat = cat_summary

    # --- Combined Summary Table --- #
    st.subheader("🧾 Combined Descriptive Table")
    combined_summary = summary.reset_index().rename(columns={"index": "Variable"})
    combined_summary = pd.concat([combined_summary, pd.DataFrame(cat_combined)], ignore_index=True)
    combined_summary.drop(columns="Type", inplace=True)
    st.dataframe(combined_summary, use_container_width=True)

    # --- Statistical Tests --- #
    st.subheader("🧪 Statistical Test Results")
    if "test_results" in st.session_state:
        test_df = pd.DataFrame(st.session_state.test_results)

        def color_p(val):
            try:
                val = float(val)
                if val < 0.05:
                    return 'background-color: #c8e6c9'  # green
                elif val < 0.1:
                    return 'background-color: #fff3cd'  # yellow
                else:
                    return 'background-color: #ffcccb'  # red
            except:
                return ''

        if any(col in test_df.columns for col in ['p-value', 'p', 'p_value']):
            styled = test_df.style.applymap(color_p, subset=['p-value'])
            st.dataframe(styled, use_container_width=True)
        else:
            st.dataframe(test_df, use_container_width=True)
    else:
        st.info("❗ No statistical tests have been saved yet.")

    # --- Correlation Matrix --- #
    st.subheader("🔗 Correlation Matrix")
    if "corr_matrix" in st.session_state and "p_matrix" in st.session_state:
        corr = st.session_state.corr_matrix.copy()
        pval = st.session_state.p_matrix.copy()

        annotated = pd.DataFrame(index=corr.index, columns=corr.columns)
        for i in range(len(corr.columns)):
            for j in range(i + 1):
                r = corr.iloc[i, j]
                p = pval.iloc[i, j]
                annotated.iloc[i, j] = f"{r:.2f} (p={p:.3f})" if not pd.isna(r) else ""

        def color_corr(val):
            try:
                r = float(val.split(" ("))[0]
                if abs(r) < 0.3:
                    return 'background-color: #fff3cd'
                elif abs(r) < 0.7:
                    return 'background-color: #ffe0b2'
                else:
                    return 'background-color: #c8e6c9'
            except:
                return ''

        st.dataframe(annotated.style.applymap(color_corr), use_container_width=True)
    else:
        st.info("❗ Correlation analysis not available. Please compute it in the proper section.")
else:
    st.warning("⚠️ Please upload a dataset first.")

# --- Reset Button --- #
st.markdown("---")
st.subheader("🔄 Reset Session")
with st.expander("⚠️ Reset all data and start over"):
    confirm = st.checkbox("Yes, I want to reset everything and upload a new dataset.")
    if confirm:
        reset_button = st.button("🔁 Confirm and Reset", help="Click to clear all session data and restart")
if 'reset_button' in locals() and reset_button:
    st.toast("✔️ Session reset successfully!", icon="🔄")
    st.balloons()
    for key in list(st.session_state.keys()):
        del st.session_state[key]
        st.session_state["section"] = "Upload Dataset"
    st.rerun()
