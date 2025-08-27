# -*- coding: utf-8 -*-
from __future__ import annotations
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from typing import Optional, Sequence, Tuple

def hist_kde(
    s: pd.Series,
    by: Optional[pd.Series] = None,
    bins: int = 30,
    show_kde: bool = True,
    title: Optional[str] = None,
) -> go.Figure:
    """Istogramma (con KDE opzionale). Se `by` è fornito, mostra più istogrammi sovrapposti."""
    df = pd.DataFrame({"x": s})
    if by is not None:
        df["by"] = by.astype(str)
    fig = go.Figure()
    if by is None:
        fig.add_trace(go.Histogram(x=df["x"], nbinsx=bins, name="Distribuzione", opacity=0.75))
        if show_kde and df["x"].notna().sum() > 3:
            kde = _kde_line(df["x"].dropna().values)
            fig.add_trace(go.Scatter(x=kde[0], y=kde[1], name="KDE", mode="lines"))
    else:
        for g, sub in df.groupby("by"):
            fig.add_trace(go.Histogram(x=sub["x"], nbinsx=bins, name=str(g), opacity=0.6))
    fig.update_layout(barmode="overlay")
    fig.update_traces(opacity=0.65)  # migliora lettura
    fig.update_layout(
        title=title or "Istogramma",
        xaxis_title="Valori",
        yaxis_title="Frequenza",
        legend_title="Gruppo" if by is not None else None,
        template="plotly_white",
    )
    return fig

def box_violin(
    s: pd.Series,
    by: Optional[pd.Series] = None,
    show_violin: bool = False,
    title: Optional[str] = None,
) -> go.Figure:
    """Boxplot (o violin). Se `by` è presente, un box/violin per gruppo."""
    df = pd.DataFrame({"x": s})
    if by is not None:
        df["by"] = by.astype(str)
    if by is None:
        if show_violin:
            fig = px.violin(df, y="x", box=True, points="outliers", title=title or "Violin plot")
        else:
            fig = px.box(df, y="x", points="outliers", title=title or "Box plot")
    else:
        if show_violin:
            fig = px.violin(df, x="by", y="x", box=True, points="outliers", title=title or "Violin plot per gruppo")
        else:
            fig = px.box(df, x="by", y="x", points="outliers", title=title or "Box plot per gruppo")
    fig.update_layout(template="plotly_white", xaxis_title=None, yaxis_title="Valori")
    return fig

def qq_plot(
    s: pd.Series,
    title: Optional[str] = None,
) -> go.Figure:
    """Q-Q plot contro la normale standard (stima media/dev.std dalla serie)."""
    x = s.dropna().astype(float).values
    n = x.size
    if n < 3:
        return go.Figure(layout=dict(title="Q-Q plot (insufficiente n)"))
    x_sorted = np.sort(x)
    mean, std = np.mean(x_sorted), np.std(x_sorted, ddof=1) if n > 1 else np.std(x_sorted)
    # quantili teorici (normale standard)
    probs = (np.arange(1, n + 1) - 0.5) / n
    from math import sqrt, log, pi
    # approssimazione inversa CDF (Beasley-Springer/Moro sempl.)
    def inv_norm(p: np.ndarray) -> np.ndarray:
        # regioni di coda
        a = [2.50662823884, -18.61500062529, 41.39119773534, -25.44106049637]
        b = [-8.47351093090, 23.08336743743, -21.06224101826, 3.13082909833]
        c = [0.3374754822726147, 0.9761690190917186, 0.1607979714918209,
             0.0276438810333863, 0.0038405729373609, 0.0003951896511919,
             0.0000321767881768, 0.0000002888167364, 0.0000003960315187]
        x = np.copy(p)
        y = x - 0.5
        r = np.empty_like(x)
        mask = np.abs(y) < 0.42
        # regione centrale
        z = y[mask] * y[mask]
        r[mask] = y[mask] * (((a[3]*z + a[2])*z + a[1])*z + a[0]) / ((((b[3]*z + b[2])*z + b[1])*z + b[0])*z + 1.0)
        # code
        z = np.where(mask, 0, np.where(y > 0, 1 - x, x))
        t = np.sqrt(-2.0*np.log(z))
        r[~mask] = np.sign(y[~mask]) * ( ( ( (c[8]*t + c[7])*t + c[6])*t + c[5])*t + c[4])*t + c[3]
        r[~mask] = r[~mask] + c[2] + c[1]*np.sign(y[~mask])
        return r
    z_theoretical = inv_norm(probs)
    # standardizzo dati per confrontarli con teorici standard
    z_empirical = (x_sorted - mean) / (std if std > 0 else 1.0)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=z_theoretical, y=z_empirical, mode="markers", name="Quantili"))
    # retta di riferimento y=x
    min_ax = float(min(z_theoretical.min(), z_empirical.min()))
    max_ax = float(max(z_theoretical.max(), z_empirical.max()))
    fig.add_trace(go.Scatter(x=[min_ax, max_ax], y=[min_ax, max_ax], mode="lines", name="y = x"))
    fig.update_layout(
        title=title or "Q-Q plot vs Normale",
        xaxis_title="Quantili teorici (Normale)",
        yaxis_title="Quantili empirici (standardizzati)",
        template="plotly_white",
        showlegend=False
    )
    return fig

# ---- util ----

def _kde_line(x: np.ndarray, points: int = 200) -> Tuple[np.ndarray, np.ndarray]:
    """KDE 1D (gaussiana) con banda di Silverman; ritorna (xs, densità)."""
    x = x[~np.isnan(x)]
    n = x.size
    if n < 3:
        xs = np.linspace(x.min() if n else 0, x.max() if n else 1, 2)
        return xs, np.zeros_like(xs)
    sd = np.std(x, ddof=1)
    iqr = np.subtract(*np.percentile(x, [75, 25]))
    sigma = min(sd, iqr / 1.349) if (sd > 0 and iqr > 0) else (sd if sd > 0 else 1.0)
    h = 0.9 * sigma * n ** (-1/5) if sigma > 0 else 1.0
    xs = np.linspace(x.min(), x.max(), points)
    # densità gaussiana
    coef = 1 / (np.sqrt(2*np.pi) * h * n)
    ys = np.zeros_like(xs)
    for xi in x:
        ys += np.exp(-0.5 * ((xs - xi) / h) ** 2)
    ys *= coef
    return xs, ys
