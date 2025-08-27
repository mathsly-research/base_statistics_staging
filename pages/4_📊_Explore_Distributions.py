import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy import stats

st.set_page_config(page_title="Statistical App", layout="wide")
st.title("Step 4: Explore Variable Distributions")

if st.session_state.get("df") is not None:
    df = st.session_state.df
    num_vars = df.select_dtypes(include=np.number).columns

    if len(num_vars) == 0:
        st.warning("⚠️ No numeric variables found in the dataset.")
    else:
        col = st.selectbox("Select a numeric variable", num_vars)
        data = df[col].dropna()

        st.subheader(f"📊 Distribution of `{col}`")
        st.markdown("""
        ℹ️ This plot compares the **observed distribution** (histogram + KDE)  
        with the **theoretical normal distribution** (dashed red line).
        """)

        # Calculate KDE and theoretical normal curve
        kde = stats.gaussian_kde(data)
        x_range = np.linspace(min(data), max(data), 100)
        kde_values = kde(x_range)
        mean = np.mean(data)
        std = np.std(data)
        normal_curve = stats.norm.pdf(x_range, loc=mean, scale=std)

        # Plot
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=data, histnorm='probability density',
            name='Observed Histogram',
            opacity=0.6,
            marker=dict(color='skyblue', line=dict(color='black', width=1))
        ))
        fig.add_trace(go.Scatter(
            x=x_range, y=kde_values,
            mode='lines',
            name='Observed KDE',
            line=dict(color='blue', width=2)
        ))
        fig.add_trace(go.Scatter(
            x=x_range, y=normal_curve,
            mode='lines',
            name='Theoretical Normal',
            line=dict(color='red', dash='dash', width=2)
        ))
        fig.update_layout(
            title=dict(text=f"Observed vs Theoretical Distribution for '{col}'", x=0.5),
            xaxis_title=col,
            yaxis_title="Density",
            legend_title="Legend",
            bargap=0.05,
            template='simple_white'
        )
        st.plotly_chart(fig)

        # Shapiro-Wilk test + automatic save
        stat, p = stats.shapiro(data)
        normal_text = "✅ Variable appears normally distributed." if p > 0.05 else "⚠️ Variable is not normally distributed."
        st.info(f"Shapiro-Wilk test p = {p:.4f} → {normal_text}")

        result_text = f"📈 Distribution and normality for '{col}':\nShapiro-Wilk p = {p:.4f} → {normal_text}"
        if "saved_results" in st.session_state:
            st.session_state.saved_results.append(result_text)

        st.info("➡️ Proceed to **'Statistical Tests'** to compare variables between groups.")
else:
    st.warning("⚠️ Please upload a dataset first in the 'Upload Dataset' section.")
