"""The five charts, saved as PNG.

Colour choices are not decorative. Portfolio blue and benchmark orange are a
validated colourblind-safe pair (worst-case deuteranopia/protanopia separation
is far above the readability floor), and the monthly heatmap uses a blue-to-red
diverging scale through a neutral grey rather than the conventional green-to-red,
which is the single most common chart that red-green colourblind readers cannot
use at all. Grid and axis lines are deliberately recessive so the data carries
the ink.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display needed; this must precede the pyplot import

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from . import metrics
from .montecarlo import SimulationResult

PORTFOLIO = "#2a78d6"
BENCHMARK = "#eb6834"
BAND = "#cde2fb"
BAND_INNER = "#86b6ef"
SURFACE = "#fcfcfb"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
MUTED = "#898781"
INK = "#0b0b0b"
SECONDARY_INK = "#52514e"

LINE_WIDTH = 2.0
FIGURE_SIZE = (10, 5.5)
DPI = 150

# Rolling Sharpe uses a full year of history so the statistic means something;
# a shorter window is mostly noise.
ROLLING_WINDOW = metrics.TRADING_DAYS

# Blue for gains, red for losses, neutral grey at zero.
RETURN_COLORMAP = LinearSegmentedColormap.from_list(
    "pbt_diverging", ["#a02c2c", "#e34948", "#f0efec", "#6da7ec", "#184f95"]
)


def generate_all(
    result: pd.DataFrame,
    simulation: SimulationResult,
    risk_free_rate: float,
    output_dir: str | Path,
) -> list[Path]:
    """Write all five charts and return their paths, in display order."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    return [
        plot_equity_curve(result, output_dir / "01_equity_curve.png"),
        plot_drawdown(result, output_dir / "02_drawdown.png"),
        plot_rolling_sharpe(result, risk_free_rate, output_dir / "03_rolling_sharpe.png"),
        plot_monthly_heatmap(result, output_dir / "04_monthly_returns.png"),
        plot_monte_carlo_fan(simulation, output_dir / "05_monte_carlo.png"),
    ]


def plot_equity_curve(result: pd.DataFrame, path: Path) -> Path:
    """Portfolio and benchmark value through time, both starting from capital."""
    fig, ax = _figure("Portfolio value vs benchmark", "Growth of the initial capital")

    ax.plot(result.index, result["portfolio_value"], color=PORTFOLIO, lw=LINE_WIDTH,
            label="Portfolio", zorder=3)
    ax.plot(result.index, result["benchmark_value"], color=BENCHMARK, lw=LINE_WIDTH,
            label="Benchmark", zorder=2)

    # Direct labels on the final values -- selectively, not on every point.
    for column, color in (("portfolio_value", PORTFOLIO), ("benchmark_value", BENCHMARK)):
        final = result[column].iloc[-1]
        ax.annotate(f"{final:,.0f}", xy=(result.index[-1], final), xytext=(6, 0),
                    textcoords="offset points", color=color, fontsize=9,
                    va="center", fontweight="bold")

    ax.yaxis.set_major_formatter(lambda value, _: f"{value:,.0f}")
    ax.legend(frameon=False, loc="upper left", fontsize=9)
    return _save(fig, path)


def plot_drawdown(result: pd.DataFrame, path: Path) -> Path:
    """Depth below the running peak, for both series."""
    fig, ax = _figure("Drawdown from peak", "How far below its own high-water mark each series sat")

    portfolio = metrics.drawdown_series(result["portfolio_return"]) * 100
    benchmark = metrics.drawdown_series(result["benchmark_return"]) * 100

    ax.fill_between(result.index, portfolio, 0, color=PORTFOLIO, alpha=0.15, zorder=2)
    ax.plot(result.index, portfolio, color=PORTFOLIO, lw=LINE_WIDTH, label="Portfolio", zorder=3)
    ax.plot(result.index, benchmark, color=BENCHMARK, lw=LINE_WIDTH, label="Benchmark", zorder=2)

    ax.yaxis.set_major_formatter(lambda value, _: f"{value:.0f}%")
    ax.legend(frameon=False, loc="lower left", fontsize=9)
    return _save(fig, path)


def plot_rolling_sharpe(result: pd.DataFrame, risk_free_rate: float, path: Path) -> Path:
    """Rolling 12-month Sharpe ratio, showing how unstable the statistic is."""
    fig, ax = _figure(
        "Rolling 12-month Sharpe ratio",
        f"{ROLLING_WINDOW}-trading-day window; the first year has no value by construction",
    )

    for column, color, label in (
        ("portfolio_return", PORTFOLIO, "Portfolio"),
        ("benchmark_return", BENCHMARK, "Benchmark"),
    ):
        ax.plot(result.index, _rolling_sharpe(result[column], risk_free_rate),
                color=color, lw=LINE_WIDTH, label=label)

    ax.axhline(0, color=AXIS, lw=1, zorder=1)
    ax.legend(frameon=False, loc="upper left", fontsize=9)
    return _save(fig, path)


def plot_monthly_heatmap(result: pd.DataFrame, path: Path) -> Path:
    """Portfolio return for every month, as a year-by-month grid."""
    table = _monthly_return_table(result["portfolio_return"])
    values = table.to_numpy(dtype="float64") * 100

    height = max(3.0, 0.42 * len(table) + 1.8)
    fig, ax = _figure("Monthly portfolio returns", "Percent, by calendar month",
                      size=(10, height))

    # Symmetric limits keep zero at the neutral midpoint of the diverging scale.
    limit = float(np.nanmax(np.abs(values))) if values.size else 1.0
    image = ax.imshow(values, cmap=RETURN_COLORMAP, vmin=-limit, vmax=limit, aspect="auto")

    ax.set_xticks(range(len(table.columns)), table.columns, color=MUTED, fontsize=9)
    ax.set_yticks(range(len(table.index)), table.index, color=MUTED, fontsize=9)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            value = values[row, column]
            if np.isnan(value):
                continue
            # White text only where the cell is dark enough to need it.
            shade = "#ffffff" if abs(value) > 0.6 * limit else INK
            ax.text(column, row, f"{value:.1f}", ha="center", va="center",
                    fontsize=8, color=shade)

    bar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
    bar.ax.tick_params(colors=MUTED, labelsize=8)
    bar.outline.set_visible(False)
    return _save(fig, path)


def plot_monte_carlo_fan(simulation: SimulationResult, path: Path) -> Path:
    """Simulated portfolio value over five years, with the benchmark median."""
    fig, ax = _figure(
        "Monte Carlo simulation of portfolio value",
        f"{simulation.n_paths:,} stationary block bootstrap paths; "
        f"portfolio and benchmark resampled jointly",
    )

    years = simulation.fan_steps / metrics.TRADING_DAYS
    low, median, high = (simulation.fan_percentiles[p] for p in (10, 50, 90))

    ax.fill_between(years, low, high, color=BAND, alpha=0.55, zorder=2,
                    label="Portfolio, 10th-90th percentile")
    ax.plot(years, median, color=PORTFOLIO, lw=LINE_WIDTH, zorder=4, label="Portfolio median")
    ax.plot(years, simulation.benchmark_median_path, color=BENCHMARK, lw=LINE_WIDTH,
            ls="--", zorder=3, label="Benchmark median")
    ax.axhline(simulation.initial_capital, color=MUTED, lw=1, ls=":", zorder=1,
               label="Initial capital")

    ax.set_xlabel("Years ahead", color=SECONDARY_INK, fontsize=9)
    ax.yaxis.set_major_formatter(lambda value, _: f"{value:,.0f}")
    ax.legend(frameon=False, loc="upper left", fontsize=9)
    return _save(fig, path)


def _rolling_sharpe(returns: pd.Series, risk_free_rate: float) -> pd.Series:
    """Annualised Sharpe over a trailing window, aligned to the window's end."""
    excess = returns - metrics.daily_from_annual_rate(risk_free_rate)
    mean = excess.rolling(ROLLING_WINDOW).mean()
    deviation = excess.rolling(ROLLING_WINDOW).std(ddof=1)
    return (mean / deviation) * np.sqrt(metrics.TRADING_DAYS)


def _monthly_return_table(returns: pd.Series) -> pd.DataFrame:
    """Compound daily returns into a year (rows) by month (columns) grid."""
    monthly = returns.resample("ME").apply(lambda window: (1.0 + window).prod() - 1.0)
    frame = pd.DataFrame(
        {"year": monthly.index.year, "month": monthly.index.month, "value": monthly.to_numpy()}
    )
    table = frame.pivot(index="year", columns="month", values="value")
    labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
              "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    table = table.reindex(columns=range(1, 13))
    table.columns = labels
    return table


def _figure(title: str, subtitle: str, size: tuple[float, float] = FIGURE_SIZE):
    """A styled figure and axis: recessive chrome, no top or right spine."""
    fig, ax = plt.subplots(figsize=size, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)

    ax.set_title(title, color=INK, fontsize=13, fontweight="bold", loc="left", pad=18)
    ax.text(0, 1.02, subtitle, transform=ax.transAxes, color=SECONDARY_INK, fontsize=9)

    ax.grid(True, color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=9)
    return fig, ax


def _save(fig, path: Path) -> Path:
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return path
