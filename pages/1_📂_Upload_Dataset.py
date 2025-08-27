import streamlit as st
import pandas as pd
import numpy as np
import os

# --- Upload Dataset --- #
st.title("Step 1: Upload Your Dataset")

st.markdown("""
### File Requirements:
- Each **row** should represent one observation (e.g., a patient).
- Each **column** should be a variable (e.g., Age, Sex, BMI).
- Supported formats: `.csv`, `.xlsx` (Excel files).
- Maximum file size: **200 MB**.

**Example of expected headers:**
```csv
ID,Age,Sex,BMI,Diagnosis
1,34,M,23.5,Control
2,28,F,20.1,Anorexia
```
""")

example_df = pd.DataFrame({
    "ID": [1, 2, 3],
    "Age": [34, 28, 45],
    "Sex": ["M", "F", "F"],
    "BMI": [23.5, 20.1, 27.3],
    "Diagnosis": ["Control", "Anorexia", "Bulimia"]
})
st.markdown("#### Example Dataset Preview")
st.dataframe(example_df, use_container_width=True)

file_type = st.radio("Select file type", ["CSV", "Excel (.xlsx)"])
uploaded_file = st.file_uploader("Upload your dataset file (max 200 MB)", type=["csv", "xlsx"])

if uploaded_file:
    file_size_mb = uploaded_file.size / (1024 * 1024)
    st.info(f"File size: {file_size_mb:.2f} MB")
    if uploaded_file.size > 200 * 1024 * 1024:
        st.error("File too large. The maximum allowed size is 200 MB.")
    else:
        try:
            if file_type == "CSV":
                use_header = st.checkbox("Use the first row as column headers", value=True)
                sep = st.selectbox("Choose CSV separator", [",", ";", "\t"], index=0)
                df = pd.read_csv(uploaded_file, sep=sep, header=0 if use_header else None)
            else:
                sheet_name = st.text_input("Excel sheet name (leave blank to use the first sheet)", "")
                df = pd.read_excel(uploaded_file, sheet_name=sheet_name if sheet_name else 0)

            if df.shape[1] < 2:
                st.error("The file must contain at least 2 columns.")
            else:
                st.session_state.df = df
                st.success("Dataset uploaded successfully!")
                st.write("Data preview:")
                st.dataframe(df.head())

                st.info(f"Rows: {df.shape[0]} | Columns: {df.shape[1]}")

                num_vars = df.select_dtypes(include=np.number).shape[1]
                cat_vars = df.select_dtypes(include=["object", "category", "bool"]).shape[1]
                st.info(f"Numeric variables: {num_vars} | Categorical variables: {cat_vars}")

                st.info("Proceed to **'Sample Overview'** to explore your dataset structure.")
        except Exception as e:
            st.error(f"An error occurred while reading the file: {e}")
