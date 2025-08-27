
from __future__ import annotations
import pandas as pd
import numpy as np

def dataset_diagnostics(df: pd.DataFrame) -> dict:
    diag = {}
    diag["n_rows"], diag["n_cols"] = df.shape
    diag["memory_mb"] = float(df.memory_usage(deep=True).sum()/1e6)
    # tipi
    diag["n_numeric"] = df.select_dtypes(include="number").shape[1]
    diag["n_categorical"] = df.select_dtypes(include=["object", "category", "bool"]).shape[1]
    # missing
    miss_perc = df.isna().mean().sort_values(ascending=False)
    diag["missing_by_col"] = miss_perc
    diag["missing_overall"] = float(df.isna().mean().mean())
    # duplicati
    diag["n_duplicates"] = int(df.duplicated().sum())
    # costanti
    constant_cols = [c for c in df.columns if df[c].nunique(dropna=False) <= 1]
    diag["constant_cols"] = constant_cols
    # cardinalità alta per categoriche
    cat_cols = df.select_dtypes(include=["object", "category", "bool"]).columns
    high_card = {c:int(df[c].nunique()) for c in cat_cols if df[c].nunique() > max(20, 0.2*len(df))}
    diag["high_cardinality"] = high_card
    # qualità semplice (indice 0-100)
    penalty = 0
    penalty += 100 * min(diag["missing_overall"], 0.5)          # fino al 50% missing
    penalty += 2 * diag["n_duplicates"] / max(len(df),1) * 100  # duplicati pesano
    penalty += 5 * len(constant_cols)                            # colonne inutili
    score = max(0, 100 - penalty)
    diag["quality_score"] = float(round(score, 1))
    return diag
