
# -*- coding: utf-8 -*-
from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Iterable, Optional, Dict, Any, Tuple
from dataclasses import dataclass

try:
    from scipy import stats as spstats
except Exception:
    spstats = None  # Box-Cox optional

@dataclass
class TransformInfo:
    name: str
    shift: float | None = None
    lam: float | None = None  # for Box-Cox

def _iqr_bounds(s: pd.Series) -> Tuple[float, float]:
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    lo = q1 - 1.5 * iqr
    hi = q3 + 1.5 * iqr
    return lo, hi

def _apply_transform(s: pd.Series, transform: Optional[str]) -> tuple[pd.Series, TransformInfo | None]:
    if transform is None or transform.lower() == "none":
        return s, None
    s_clean = s.dropna()
    if s_clean.empty:
        return s, None
    if transform.lower() == "log10":
        shift = 0.0
        if (s_clean <= 0).any():
            shift = float(1 - s_clean.min() + 1e-6)
        out = np.log10(s + shift)
        return out, TransformInfo(name="log10", shift=shift)
    if transform.lower() == "box-cox":
        if spstats is None:
            # fallback: behave like no transform
            return s, None
        shift = 0.0
        if (s_clean <= 0).any():
            shift = float(1 - s_clean.min() + 1e-6)
        y, lam = spstats.boxcox((s + shift).dropna())
        out = pd.Series(index=s.index, dtype="float64")
        out.loc[(s + shift).notna()] = y
        return out, TransformInfo(name="box-cox", shift=shift, lam=float(lam))
    return s, None

def summarize_continuous(
    df: pd.DataFrame,
    cols: Optional[Iterable[str]] = None,
    exclude_outliers: bool = False,
    transform: Optional[str] = None,
) -> pd.DataFrame:
    """Return a table of descriptive stats for numeric variables."""
    if cols is None:
        cols = list(df.select_dtypes(include="number").columns)
    out_rows = []
    for c in cols:
        s = df[c]
        s_t, tinfo = _apply_transform(s, transform)
        s_work = s_t.dropna()
        n_total = int(s.size - s.isna().sum())
        n_miss = int(s.isna().sum())
        miss_pct = (n_miss / s.size * 100.0) if s.size else 0.0

        n_used = n_total
        n_removed = 0
        if exclude_outliers and s_work.size > 0:
            lo, hi = _iqr_bounds(s_work)
            mask = (s_work >= lo) & (s_work <= hi)
            n_removed = int((~mask).sum())
            s_work = s_work[mask]
            n_used = int(s_work.size)

        if s_work.size == 0:
            row = dict(
                variable=c, n=n_total, n_used=0, n_outliers=n_removed,
                mean=np.nan, sd=np.nan, min=np.nan, p25=np.nan, median=np.nan, p75=np.nan, max=np.nan,
                missing_pct=round(miss_pct,1),
                transform=tinfo.name if tinfo else "none",
                shift=tinfo.shift if tinfo else None,
                boxcox_lambda=tinfo.lam if tinfo else None,
            )
        else:
            q = s_work.quantile([0.25, 0.5, 0.75])
            row = dict(
                variable=c,
                n=n_total,
                n_used=n_used,
                n_outliers=n_removed,
                mean=float(s_work.mean()),
                sd=float(s_work.std(ddof=1)) if n_used>1 else float("nan"),
                min=float(s_work.min()),
                p25=float(q.loc[0.25]),
                median=float(q.loc[0.5]),
                p75=float(q.loc[0.75]),
                max=float(s_work.max()),
                missing_pct=round(miss_pct,1),
                transform=tinfo.name if tinfo else "none",
                shift=tinfo.shift if tinfo else None,
                boxcox_lambda=tinfo.lam if tinfo else None,
            )
        out_rows.append(row)
    df_out = pd.DataFrame(out_rows)
    desired = ["variable","n","n_used","n_outliers","mean","sd","min","p25","median","p75","max","missing_pct","transform","shift","boxcox_lambda"]
    return df_out[desired]

def summarize_categorical(
    df: pd.DataFrame,
    cols: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """Return a table of categorical summaries per variable (mode and cardinality)."""
    if cols is None:
        cols = list(df.select_dtypes(include=["object","category","bool"]).columns)
    rows = []
    for c in cols:
        s = df[c]
        n = int(s.notna().sum())
        nmiss = int(s.isna().sum())
        miss_pct = (nmiss / s.size * 100.0) if s.size else 0.0
        nunique = int(s.nunique(dropna=True))
        top_val = None
        top_freq = 0
        if n > 0:
            vc = s.dropna().value_counts()
            if len(vc) > 0:
                top_val = str(vc.index[0])
                top_freq = int(vc.iloc[0])
        top_pct = (top_freq / n * 100.0) if n else 0.0
        rows.append(dict(
            variable=c,
            n=n,
            missing_pct=round(miss_pct,1),
            n_unique=nunique,
            mode=top_val,
            mode_freq=top_freq,
            mode_pct=round(top_pct,1),
        ))
    out = pd.DataFrame(rows)
    return out[["variable","n","missing_pct","n_unique","mode","mode_freq","mode_pct"]]
