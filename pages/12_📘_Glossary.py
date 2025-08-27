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
st.title("Glossary")

st.markdown("ℹ️ This section provides definitions of key statistical terms used in the app. Click on each item to expand and view the definition and resources.")

with st.expander("📌 p-value"):
    st.write("""
    The p-value represents the probability of obtaining test results at least as extreme
    as the results actually observed, under the assumption that the null hypothesis is correct.
    """)
    st.markdown("🔗 [More on p-values](https://www.mathslyresearch.com/p-value/)")

with st.expander("📌 Normal distribution"):
    st.write("""
    A bell-shaped distribution where most of the data points cluster around the mean.
    Many statistical tests assume that the data follow a normal distribution.
    """)
    st.markdown("🔗 [Learn about normal distribution](https://www.mathslyresearch.com/distribuzione-normale-guida-veloce/)")

with st.expander("📌 t-test"):
    st.write("""
    A statistical test used to compare the means of two groups.
    Requires normally distributed data and equal variances.
    """)
    st.markdown("🔗 [t-test explained](https://www.mathslyresearch.com/test-t-di-studente-una-breve-guida/)")

with st.expander("📌 ANOVA"):
    st.write("""
    Analysis of Variance: used to compare means across three or more groups.
    Assumes normal distribution and equal variances.
    """)
    st.markdown("🔗 [About ANOVA](https://www.mathslyresearch.com/anova-guida-breve-allanalisi-della-varianza/)")

with st.expander("📌 Mann-Whitney U test"):
    st.write("""
    A non-parametric test used to compare differences between two independent groups
    when the data is not normally distributed.
    """)
    st.markdown("🔗 [Mann-Whitney U test](https://www.mathslyresearch.com/mann-whitney-u-test/)")

with st.expander("📌 Kruskal-Wallis test"):
    st.write("""
    A non-parametric method for testing whether samples originate from the same distribution.
    Used for comparing more than two groups when the assumption of normality is not met.
    """)
    st.markdown("🔗 [Kruskal-Wallis test](https://www.mathslyresearch.com/kruskal-wallis-guida-breve/)")

with st.expander("📌 Chi-square test"):
    st.write("""
    A test used to assess whether observed frequencies in a contingency table differ
    from expected frequencies.
    Suitable for large sample sizes.
    """)
    st.markdown("🔗 [Chi-square test](https://www.mathslyresearch.com/test-chi-quadro-guida-breve/)")

with st.expander("📌 Fisher’s exact test"):
    st.write("""
    A statistical test used for small sample sizes and 2x2 contingency tables.
    Provides an exact p-value.
    """)
    st.markdown("🔗 [Fisher’s exact test](https://www.mathslyresearch.com/test-esatto-di-fisher-guida-breve/)")

with st.expander("📌 Correlation (Pearson & Spearman)"):
    st.write("""
    Measures the strength and direction of the linear relationship between two continuous variables.
    Values range from -1 to +1.
    """)
    st.markdown("🔗 [Pearson correlation](https://www.mathslyresearch.com/correlazione/)")
