"""Target portfolio weights.

Three schemes only: equal, custom, and inverse volatility.

Market-cap weighting is deliberately absent. yfinance exposes only *current*
shares outstanding, so reconstructing historical market caps from it would apply
today's share counts to past prices -- look-ahead bias baked straight into the
weights. A backtest that quietly does this looks better than it should, which is
exactly the failure mode this project is trying not to demonstrate.

Every function here returns weights indexed by ticker and summing to 1.0.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class WeightError(ValueError):
    """Weights cannot be computed from the data or configuration supplied."""


def equal_weights(tickers: tuple[str, ...] | list[str]) -> pd.Series:
    """1/N across all holdings."""
    if len(tickers) == 0:
        raise WeightError("Cannot build equal weights with no tickers.")
    return pd.Series(1.0 / len(tickers), index=list(tickers), dtype="float64")


def custom_weights(
    tickers: tuple[str, ...] | list[str], weights: tuple[float, ...] | list[float]
) -> pd.Series:
    """Author-supplied weights, normalised to sum to 1.0.

    The config validator already rejects weights that do not sum to 1 within
    tolerance; normalising here removes the residual floating-point drift so the
    engine can rely on an exact sum.
    """
    if len(tickers) != len(weights):
        raise WeightError(
            f"custom weights ({len(weights)}) do not align with tickers ({len(tickers)})."
        )
    series = pd.Series(list(weights), index=list(tickers), dtype="float64")
    total = series.sum()
    if total <= 0:
        raise WeightError("custom weights must sum to a positive number.")
    return series / total


def inverse_vol_weights(
    returns: pd.DataFrame, as_of: pd.Timestamp, lookback_days: int
) -> pd.Series:
    """Weights proportional to 1 / trailing realised volatility.

    LOOK-AHEAD SAFETY: the sample is `returns.loc[:as_of].tail(lookback_days)` --
    the window ends at the rebalance date inclusive and never reaches past it.
    A test mutates all data after `as_of` and asserts these weights are unchanged.

    Volatility is the standard deviation of daily returns over the window. It is
    left un-annualised on purpose: annualising scales every ticker by the same
    sqrt(252), which cancels in the normalisation and would only add noise to
    the code.

    Args:
        returns: daily simple returns, one column per ticker.
        as_of: the rebalance date; the trailing window ends here, inclusive.
        lookback_days: number of trailing observations to use.
    """
    window = returns.loc[:as_of].tail(lookback_days)
    if len(window) < 2:
        raise WeightError(
            f"Only {len(window)} return observation(s) available at {as_of.date()}; "
            f"need at least 2 to estimate a volatility."
        )

    vol = window.std(ddof=1)

    # A zero-volatility column would divide by zero. In practice this means a
    # price that never moved over the window (a stale feed), so it is excluded
    # rather than handed an infinite weight.
    usable = vol[(vol > 0) & vol.notna()]
    if usable.empty:
        raise WeightError(
            f"Every ticker has zero or undefined volatility at {as_of.date()}; "
            f"cannot form inverse-volatility weights."
        )

    inverse = 1.0 / usable
    weights = inverse / inverse.sum()
    # Excluded tickers get an explicit zero so the result always covers every
    # column the engine knows about.
    return weights.reindex(returns.columns).fillna(0.0)


def resolve_weights(
    scheme: str,
    tickers: tuple[str, ...] | list[str],
    returns: pd.DataFrame,
    as_of: pd.Timestamp,
    custom: tuple[float, ...] | None = None,
    lookback_days: int | None = None,
) -> pd.Series:
    """Dispatch to the configured scheme and return target weights at `as_of`.

    This is the single entry point the engine calls at every rebalance, so the
    engine never needs to know which scheme is in force.
    """
    if scheme == "equal":
        return equal_weights(tickers)
    if scheme == "custom":
        if custom is None:
            raise WeightError("scheme 'custom' requires custom weights.")
        return custom_weights(tickers, custom)
    if scheme == "inverse_vol":
        if lookback_days is None:
            raise WeightError("scheme 'inverse_vol' requires lookback_days.")
        return inverse_vol_weights(returns, as_of, lookback_days)
    raise WeightError(f"Unknown weighting scheme: {scheme!r}")
