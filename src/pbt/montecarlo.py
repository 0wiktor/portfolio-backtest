"""Bootstrap simulation of future terminal values.

Method: the stationary bootstrap of Politis and Romano (1994). Blocks of
consecutive historical days are drawn with geometric lengths averaging
BLOCK_LENGTH_DAYS, which preserves short-horizon structure -- volatility
clustering, momentum, mean reversion -- that an IID daily resample destroys.
Geometric rather than fixed block lengths are what make the resampled series
stationary; fixed blocks are the simpler non-stationary variant.

THE CRITICAL DETAIL: portfolio and benchmark are resampled JOINTLY. One set of
date indices is drawn and applied to both series, so every simulated day pairs
the portfolio's actual return with the benchmark's actual return from that same
historical date. Simulating them independently would destroy their correlation
and make "P(portfolio beats benchmark)" meaningless -- it would be comparing two
unrelated random walks rather than two views of the same market.

This resamples the past. It assumes the return distribution and its dependence
structure carry forward unchanged, which is exactly what a regime change breaks.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .metrics import TRADING_DAYS

BLOCK_LENGTH_DAYS = 21          # roughly one trading month
N_PATHS = 10_000
HORIZON_YEARS = (1, 3, 5)
PERCENTILES = (10, 50, 90)

# The fan chart needs a curve, not 1260 separate days. Keeping every fifth step
# holds the stored paths near 10 MB with no visible difference in the plot.
FAN_CHART_STRIDE = 5

# Paths are simulated in chunks: the full 10,000 x 1260 array for two series
# plus its index array would peak around 300 MB, and nothing here needs every
# path in memory at once.
CHUNK_PATHS = 1_000


@dataclass(frozen=True)
class HorizonResult:
    """Simulated outcomes at one horizon, in currency units."""

    years: int
    median: float
    percentile_10: float
    percentile_90: float
    probability_below_initial: float
    probability_beats_benchmark: float
    benchmark_median: float


@dataclass(frozen=True)
class SimulationResult:
    """Everything the report and the fan chart need from one simulation."""

    horizons: tuple[HorizonResult, ...]
    fan_steps: np.ndarray                   # trading days from the start
    fan_percentiles: dict[int, np.ndarray]  # percentile -> portfolio path
    benchmark_median_path: np.ndarray
    initial_capital: float
    n_paths: int

    def to_frame(self) -> pd.DataFrame:
        """Horizon results as a table, for printing."""
        return pd.DataFrame(
            [
                {
                    "Horizon (years)": h.years,
                    "Median": h.median,
                    "10th percentile": h.percentile_10,
                    "90th percentile": h.percentile_90,
                    "P(below initial capital)": h.probability_below_initial,
                    "P(beats benchmark)": h.probability_beats_benchmark,
                    "Benchmark median": h.benchmark_median,
                }
                for h in self.horizons
            ]
        ).set_index("Horizon (years)")


def simulate(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    initial_capital: float,
    n_paths: int = N_PATHS,
    block_length: int = BLOCK_LENGTH_DAYS,
    seed: int = 0,
) -> SimulationResult:
    """Bootstrap `n_paths` joint futures and summarise them at 1, 3 and 5 years.

    All three horizons are read off ONE simulation of the longest horizon, at
    days 252/756/1260, so they are nested views of the same paths rather than
    three unrelated experiments.

    Args:
        portfolio_returns, benchmark_returns: daily simple returns, same index.
        initial_capital: starting value for every simulated path.
        seed: fixed, so a rerun reproduces these numbers exactly.
    """
    portfolio, benchmark = _aligned_arrays(portfolio_returns, benchmark_returns)
    if len(portfolio) < block_length:
        raise ValueError(
            f"Need at least {block_length} aligned daily returns to bootstrap, "
            f"got {len(portfolio)}."
        )

    max_horizon = max(HORIZON_YEARS) * TRADING_DAYS
    stride = slice(None, None, FAN_CHART_STRIDE)
    n_fan_steps = len(range(0, max_horizon, FAN_CHART_STRIDE))

    portfolio_terminals = {years: np.empty(n_paths) for years in HORIZON_YEARS}
    benchmark_terminals = {years: np.empty(n_paths) for years in HORIZON_YEARS}
    fan_paths = np.empty((n_paths, n_fan_steps), dtype="float32")
    benchmark_fan = np.empty((n_paths, n_fan_steps), dtype="float32")

    rng = np.random.default_rng(seed)
    for start in range(0, n_paths, CHUNK_PATHS):
        size = min(CHUNK_PATHS, n_paths - start)
        indices = _stationary_bootstrap_indices(len(portfolio), size, max_horizon, block_length, rng)

        # One index draw applied to both series: this is what preserves the
        # historical portfolio/benchmark correlation.
        portfolio_chunk = initial_capital * np.cumprod(1.0 + portfolio[indices], axis=1)
        benchmark_chunk = initial_capital * np.cumprod(1.0 + benchmark[indices], axis=1)

        for years in HORIZON_YEARS:
            final_step = years * TRADING_DAYS - 1
            portfolio_terminals[years][start : start + size] = portfolio_chunk[:, final_step]
            benchmark_terminals[years][start : start + size] = benchmark_chunk[:, final_step]

        fan_paths[start : start + size] = portfolio_chunk[:, stride]
        benchmark_fan[start : start + size] = benchmark_chunk[:, stride]

    horizons = tuple(
        _summarise_horizon(
            portfolio_terminals[years], benchmark_terminals[years], years, initial_capital
        )
        for years in HORIZON_YEARS
    )
    return SimulationResult(
        horizons=horizons,
        fan_steps=np.arange(1, max_horizon + 1)[stride],
        fan_percentiles={p: np.percentile(fan_paths, p, axis=0) for p in PERCENTILES},
        benchmark_median_path=np.median(benchmark_fan, axis=0),
        initial_capital=initial_capital,
        n_paths=n_paths,
    )


def _summarise_horizon(
    portfolio_terminal: np.ndarray,
    benchmark_terminal: np.ndarray,
    years: int,
    initial_capital: float,
) -> HorizonResult:
    return HorizonResult(
        years=years,
        median=float(np.median(portfolio_terminal)),
        percentile_10=float(np.percentile(portfolio_terminal, 10)),
        percentile_90=float(np.percentile(portfolio_terminal, 90)),
        probability_below_initial=float(np.mean(portfolio_terminal < initial_capital)),
        # Compared path by path, not median against median: each path is one
        # jointly drawn future for both, so this is a genuine head-to-head.
        probability_beats_benchmark=float(np.mean(portfolio_terminal > benchmark_terminal)),
        benchmark_median=float(np.median(benchmark_terminal)),
    )


def _stationary_bootstrap_indices(
    n_observations: int, n_paths: int, n_steps: int, block_length: int, rng: np.random.Generator
) -> np.ndarray:
    """Draw (n_paths, n_steps) positions into the historical return array.

    Each step either continues the previous block -- advancing one day, wrapping
    at the end of history -- or starts a new block on a uniformly random day.
    The restart probability of 1/block_length is what makes block lengths
    geometric with mean `block_length`.
    """
    restart_probability = 1.0 / block_length
    starts_new_block = rng.random((n_paths, n_steps)) < restart_probability
    starts_new_block[:, 0] = True  # every path begins with a fresh block
    random_starts = rng.integers(0, n_observations, size=(n_paths, n_steps), dtype=np.int64)

    indices = np.empty((n_paths, n_steps), dtype=np.int64)
    indices[:, 0] = random_starts[:, 0]
    for step in range(1, n_steps):
        continued = (indices[:, step - 1] + 1) % n_observations
        indices[:, step] = np.where(starts_new_block[:, step], random_starts[:, step], continued)
    return indices


def _aligned_arrays(
    portfolio_returns: pd.Series, benchmark_returns: pd.Series
) -> tuple[np.ndarray, np.ndarray]:
    """Both series on their shared dates, as float arrays of equal length.

    Alignment happens here rather than being assumed, because a single dropped
    date would silently pair each portfolio return with the wrong benchmark day
    and quietly corrupt the joint resampling this module depends on.
    """
    frame = pd.concat(
        [portfolio_returns, benchmark_returns], axis=1, keys=["portfolio", "benchmark"]
    ).dropna()
    return (
        np.asarray(frame["portfolio"], dtype="float64"),
        np.asarray(frame["benchmark"], dtype="float64"),
    )
