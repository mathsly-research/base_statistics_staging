
from __future__ import annotations
import pandas as pd
import io
from typing import Optional

def _read_csv(file, delimiter: Optional[str]=None) -> pd.DataFrame:
    return pd.read_csv(file, sep=delimiter if delimiter else None, engine="python")

def _read_excel(file) -> pd.DataFrame:
    return pd.read_excel(file)

def _read_parquet(file) -> pd.DataFrame:
    try:
        return pd.read_parquet(file)
    except Exception as e:
        raise RuntimeError("Lettura Parquet fallita: installare 'pyarrow'") from e

def read_any(file, delimiter: Optional[str]=None) -> pd.DataFrame:
    \"\"\"Legge CSV/XLSX/Parquet in un DataFrame pandas.\"\"\"
    name = getattr(file, "name", "").lower()
    if name.endswith(".csv"):
        return _read_csv(file, delimiter)
    if name.endswith(".xlsx") or name.endswith(".xls"):
        return _read_excel(file)
    if name.endswith(".parquet"):
        return _read_parquet(file)
    raise ValueError("Formato file non supportato. Usa .csv, .xlsx o .parquet")

def load_sample(sample: str = "clinical_demo", n: int = 200) -> pd.DataFrame:
    \"\"\"Genera un dataset di esempio sintetico realistico (clinico).\"\"\"
    import numpy as np
    rng = np.random.default_rng(42)
    sex = rng.choice(["F", "M"], size=n)
    age = rng.normal(55, 15, size=n).clip(18, 90).round(0)
    bmi = rng.normal(26, 5, size=n).clip(15, 55).round(1)
    group = rng.choice(["Control", "Treatment"], size=n, p=[0.45, 0.55])
    smoker = rng.choice([0, 1], size=n, p=[0.7, 0.3])
    score = (0.2*(age-50) + 0.5*(bmi-25) + rng.normal(0, 5, n)).round(2)
    outcome = (score + (group=="Treatment")*2 + rng.normal(0, 3, n) > 10).astype(int)
    df = pd.DataFrame({
        "ID": np.arange(1, n+1),
        "Sex": sex,
        "Age": age.astype(int),
        "BMI": bmi,
        "Group": group,
        "Smoker": smoker,
        "Score": score,
        "Outcome": outcome
    })
    # Introduci alcuni NA e duplicati per testare la validazione
    mask = rng.choice([True, False], size=n, p=[0.1, 0.9])
    df.loc[mask, "BMI"] = pd.NA
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)  # un duplicato
    return df
