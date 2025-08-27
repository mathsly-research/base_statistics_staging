# -*- coding: utf-8 -*-
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Optional, Tuple

import plotly.graph_objects as go
import plotly.express as px

try:
    from scipy import stats as spstats
except Exception:
    spstats = None  # alcuni componenti degradano con grazia

# -------------------------------------------------------------------
# Utilità
# -------------------------------------------------------------------

def _freedman_diaconis_bins(x: np.ndarray, min_bins: int = 8, max_bins: int = 30) -> int:
    """Numero di classi con regola di Freedman–Diaconis (clamp tra min e max)."""
    x = x[~np.isnan(x)]
    n = x.size
    if n < 2:
        return min_bins
    iqr = np.subtract(*np.percentile(x, [75, 25]))
    if iqr == 0:
        return int(np.clip(np.sqrt(n), min_bins, max_bins))
    h = 2 * iqr * (n ** (-1/3))
    if h <= 0:
        return int(np.clip(np.sqrt(n), min_bins, max_bins))
    bins = int(np.ceil((x.max() - x.min()) / h))
    return int(np.clip(bins if bins > 0 else min_bins, min_bins, max_bins))

def _kde_line(x: np.ndarray, points: int = 200) -> Tuple[np.ndarray, np.ndarray]:
    """
    KDE 1D (gaussiana) con banda di Silverman; ritorna (xs, densità).
    Se i dati sono insufficienti, restituisce una linea piatta.
    """
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
    coef = 1 / (np.sqrt(2*np.pi) * h * n)
    ys = np.zeros_like(xs)
    for xi in x:
        ys += np.exp(-0.5 * ((xs - xi) / h) ** 2)
    ys *= coef
    return xs, ys

# -------------------------------------------------------------------
# Grafico richiesto: Istogramma + KDE + Normale teorica
# -------------------------------------------------------------------

def observed_vs_theoretical_normal(
    s: pd.Series,
    bins: Optional[int] = None,
    title: Optional[str] = None,
    x_label: Optional[str] = None,
) -> go.Figure:
    """
    Istogramma (normalizzato a densità) + KDE osservata + curva della normale teorica
    (μ e σ stimati dai dati). Stile conforme allo screenshot richiesto.
    """
    x = s.dropna().astype(float).values
    if x.size == 0:
        fig = go.Figure()
        fig.update_layout(title="No data", template="plotly_white")
        return fig

    if bins is None:
        bins = _freedman_diaconis_bins(x, min_bins=8, max_bins=20)

    # Parametri della normale teorica
    mu = float(np.mean(x))
    sigma = float(np.std(x, ddof=1)) if x.size > 1 else 1.0
    if sigma <= 0:
        sigma = 1.0

    # Istogramma (densità)
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=x,
        nbinsx=bins,
        histnorm="probability density",
        name="Observed Histogram",
        marker=dict(color="rgba(99, 148, 255, 0.5)", line=dict(color="rgba(99,148,255,1)", width=1)),
        opacity=0.8
    ))

    # KDE osservata (blu)
    xs_kde, ys_kde = _kde_line(x)
    fig.add_trace(go.Scatter(
        x=xs_kde, y=ys_kde, mode="lines",
        name="Observed KDE",
        line=dict(color="rgba(0, 62, 255, 1)", width=3)
    ))

    # Curva normale teorica (rossa tratteggiata)
    xs_norm = np.linspace(min(x.min(), xs_kde.min()), max(x.max(), xs_kde.max()), 200)
    ys_norm = (1.0 / (sigma * np.sqrt(2*np.pi))) * np.exp(-0.5 * ((xs_norm - mu) / sigma) ** 2)
    fig.add_trace(go.Scatter(
        x=xs_norm, y=ys_norm, mode="lines",
        name="Theoretical Normal",
        line=dict(color="rgba(222, 45, 38, 1)", width=3, dash="dash")
    ))

    fig.update_layout(
        title=title or "Observed vs Theoretical Distribution",
        xaxis_title=x_label or "Value",
        yaxis_title="Density",
        template="plotly_white",
        legend=dict(title="Legend"),
        bargap=0.05
    )
    return fig

# -------------------------------------------------------------------
# Altri grafici utili (già usati nello Step 3)
# -------------------------------------------------------------------

def hist_kde(
    s: pd.Series,
    by: Optional[pd.Series] = None,
    bins: Optional[int] = None,
    show_kde: bool = True,
    title: Optional[str] = None,
) -> go.Figure:
    """
    Istogramma generico. Con show_kde=True e senza 'by', istogramma normalizzato a densità.
    """
    x = s.dropna().astype(float).values
    if bins is None:
        bins = _freedman_diaconis_bins(x)

    fig = go.Figure()
    if by is None:
        histnorm = "probability density" if show_kde else None
        fig.add_trace(go.Histogram(
            x=x, nbinsx=bins, histnorm=histnorm,
            name="Distribuzione", opacity=0.65
        ))
        if show_kde and x.size >= 3:
            xs, ys = _kde_line(x)
            fig.add_trace(go.Scatter(x=xs, y=ys, name="KDE", mode="lines"))
        y_label = "Densità" if show_kde else "Frequenza"
    else:
        df = pd.DataFrame({"x": s, "by": by}).dropna()
        for g, sub in df.groupby("by"):
            fig.add_trace(go.Histogram(x=sub["x"].astype(float), nbinsx=bins, name=str(g), opacity=0.6))
        fig.update_layout(barmode="overlay")
        y_label = "Frequenza"

    fig.update_layout(
        title=title or "Istogramma",
        xaxis_title="Valori",
        yaxis_title=y_label,
        template="plotly_white",
        legend_title="Gruppo" if by is not None else None,
    )
    return fig

def box_violin(
    s: pd.Series,
    by: Optional[pd.Series] = None,
    show_violin: bool = False,
    title: Optional[str] = None,
) -> go.Figure:
    df = pd.DataFrame({"x": s})
    if by is not None:
        df["by"] = by.astype(str)
    if by is None:
        fig = px.violin(df, y="x", box=True, points="outliers", title=title or "Violin plot") if show_violin \
              else px.box(df, y="x", points="outliers", title=title or "Box plot")
    else:
        fig = px.violin(df, x="by", y="x", box=True, points="outliers", title=title or "Violin plot per gruppo") if show_violin \
              else px.box(df, x="by", y="x", points="outliers", title=title or "Box plot per gruppo")
    fig.update_layout(template="plotly_white", xaxis_title=None, yaxis_title="Valori")
    return fig

def qq_plot(s: pd.Series, title: Optional[str] = None) -> go.Figure:
    """Q-Q plot contro la normale usando scipy.stats.probplot (robusto)."""
    x = s.dropna().astype(float).values
    fig = go.Figure()
    if x.size < 3:
        fig.update_layout(title="Q-Q plot (dati insufficienti)", template="plotly_white")
        return fig
    if spstats is None:
        fig.update_layout(title="Q-Q plot (SciPy non disponibile)", template="plotly_white")
        return fig
    (osm, osr), (slope, intercept, r) = spstats.probplot(x, dist="norm")
    fig.add_trace(go.Scatter(x=osm, y=osr, mode="markers", name="Quantili"))
    x_line = np.array([osm.min(), osm.max()])
    y_line = slope * x_line + intercept
    fig.add_trace(go.Scatter(x=x_line, y=y_line, mode="lines", name="Retta di riferimento"))
    fig.update_layout(
        title=title or "Q-Q plot vs Normale",
        xaxis_title="Quantili teorici (Normale)",
        yaxis_title="Quantili empirici",
        template="plotly_white",
        showlegend=False
    )
    return fig
