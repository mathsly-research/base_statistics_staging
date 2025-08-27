import streamlit as st

def init_state():
    ss = st.session_state
    ss.setdefault("df", None)                 # pandas.DataFrame corrente
    ss.setdefault("diagnostics", None)        # dizionario con summary del dataset
    ss.setdefault("results", [])              # risultati dei test
    ss.setdefault("report_items", [])         # elementi salvati per il report
    ss.setdefault("accepted_disclaimer", False)
