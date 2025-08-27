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
st.title("Browse & Edit Dataset")

if st.session_state.get("df") is not None:
    df = st.session_state.df.copy()

    st.markdown("🔍 **Filter your data**")
    with st.expander("➕ Add Filters"):
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                min_val, max_val = float(df[col].min()), float(df[col].max())
                selected_range = st.slider(
                    f"Filter `{col}`", min_val, max_val, (min_val, max_val)
                )
                df = df[df[col].between(*selected_range)]
            elif pd.api.types.is_categorical_dtype(df[col]) or df[col].dtype == object:
                unique_vals = df[col].dropna().unique().tolist()
                selected_vals = st.multiselect(f"Filter `{col}`", unique_vals, default=unique_vals)
                df = df[df[col].isin(selected_vals)]

    st.markdown("✏️ **Edit your filtered dataset**")
    edited_df = st.data_editor(df, num_rows="dynamic", use_container_width=True)

    if st.button("✅ Save changes to session"):
        st.session_state.df = edited_df.copy()
        st.success("Changes saved successfully!")
        st.info(f"🧾 Rows after filtering: {edited_df.shape[0]} | Columns: {edited_df.shape[1]}")
else:
    st.warning("⚠️ Please upload a dataset first in the 'Upload Dataset' section.")
