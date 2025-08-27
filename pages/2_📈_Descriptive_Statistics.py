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
st.title("Step 2: Descriptive Statistics")

if "df" not in st.session_state:
    st.session_state.df = None

if st.session_state.df is not None:
    df = st.session_state.df

    st.markdown("""
    ℹ️ **This section helps you understand the structure and quality of your dataset.**  
    You will see missing values, variable types, and descriptive summaries.
    """)

    st.subheader("🧪 Data Types")
    if st.checkbox("Show detected variable types"):
        types_df = pd.DataFrame(df.dtypes, columns=["Type"])
        st.dataframe(types_df)

    st.subheader("❓ Missing Values")
    na_df = df.isnull().sum().to_frame("Missing Values")
    na_df["Percent"] = (na_df["Missing Values"] / len(df) * 100).round(1)
    na_filtered = na_df[na_df["Missing Values"] > 0].sort_values(by="Missing Values", ascending=False)
    st.dataframe(na_filtered)

    st.subheader("📏 Continuous Variables")
    cont = df.select_dtypes(include=np.number)
    if cont.shape[1] > 0:
        summary = cont.describe().T
        st.markdown("Basic statistics for each numeric variable:")
        st.dataframe(summary)
    else:
        st.info("No numeric variables detected.")

    st.subheader("🌤 Categorical Variables")
    cat_vars = df.select_dtypes(include=["object", "category", "bool"]).columns
    if len(cat_vars) > 0:
        for col in cat_vars:
            st.markdown(f"**Variable: `{col}`**")
            freq = df[col].value_counts(dropna=False).to_frame("Count")
            freq["Percent"] = round(100 * freq["Count"] / len(df), 1)
            st.dataframe(freq)
    else:
        st.info("No categorical variables detected.")

    st.success("✅ You're now ready to explore your variables' distributions in the next section.")

else:
    st.warning("⚠️ Please upload a dataset first in the 'Upload Dataset' section.")
