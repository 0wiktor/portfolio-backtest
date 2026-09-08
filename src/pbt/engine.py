"""The backtest loop: drift, rebalancing, and transaction costs.

Conventions that a reader should not have to infer from the arithmetic:

* Trades execute at the closing price of the rebalance date. The day's return is
  realised first on the old holdings, then the book is rebalanced at that close.

* Turnover at a rebalance is 0.5 * sum(|w_target - w_drifted|). The one-half is
  there because every sale funds a purchase, so summing both sides double-counts
  the trade.

* Initial entry is charged at turnover 1.0, not 0.5. The 0.5 formula assumes both
  sides are fully invested; at entry the "before" weights are all zero, so it
  would charge half the cost of actually buying the book.

* The benchmark is a frictionless buy-and-hold: it pays no entry or rebalancing
  costs. That biases every comparison against the portfolio, which is the
  conservative direction to be wrong in, and it is stated in the README.

* Daily portfolio return is net of any cost charged that day, so costs show up
  in the return series on the date they were actually incurred.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Config
from .weights import resolve_weights

# Basis points to a decimal fraction: 5 bps -> 0.0005.
BPS_PER_UNIT = 10_000.0

# pandas period aliases for each rebalancing frequency.
_PERIOD_ALIASES = {"monthly": "M", "quarterly": "Q", "annual": "Y"}


class EngineError(RuntimeError):
    """The backtest cannot be run as configured."""


def rebalance_dates(index: pd.DatetimeIndex, frequency: str) -> pd.DatetimeIndex:
    """The last ACTUAL trading day of each period in `index`.

    Grouping the real trading calendar matters: calendar month-end is frequently
    a weekend or a holiday, so a date generated from the calendar would not exist
    in the price data. Grouping the observed index cannot produce a date the
    market was closed on.

    The first and last dates of the sample are excluded. Entry is handled
    separately, and a rebalance on the final date cannot affect any subsequent
    return -- charging its cost would reduce the result for no modelled benefit.
    """
    if frequency == "none":
        return pd.DatetimeIndex([])
    if frequency not in _PERIOD_ALIASES:
        raise EngineError(f"Unknown rebalance frequency: {frequency!r}")
    if len(index) < 2:
        return pd.DatetimeIndex([])

    periods = index.to_period(_PERIOD_ALIASES[frequency])
    last_of_period = pd.Series(index, index=periods).groupby(level=0).max()
    dates = pd.DatetimeIndex(last_of_period.to_numpy())
    return dates[(dates > index[0]) & (dates < index[-1])]


def run_backtest(prices: pd.DataFrame, benchmark: pd.DataFrame | pd.Series, config: Config) -> pd.DataFrame:
    """Run the backtest and return one row per trading day.

    Args:
        prices: adjusted closes, one column per surviving holding, no NaNs.
        benchmark: adjusted closes for the single benchmark ticker.
        config: validated run configuration.

    Returns:
        A frame indexed by date (index name "date") with columns
        portfolio_value, portfolio_return, benchmark_value, benchmark_return,
        turnover, costs_paid. Calling .reset_index() yields the tidy column
        layout with date as the leading column.

        turnover and costs_paid are zero on non-rebalance days; both are
        recorded on the entry row for the initial purchase.
    """
    benchmark_series = _as_series(benchmark)
    _validate_inputs(prices, benchmark_series, config)

    asset_returns = prices.pct_change()
    entry_position = _entry_position(prices, config)
    dates = prices.index
    entry_date = dates[entry_position]

    scheduled = set(rebalance_dates(dates[entry_position:], config.rebalance))
    cost_rate = config.transaction_cost_bps / BPS_PER_UNIT

    # --- entry -------------------------------------------------------------
    target = _weights_at(config, prices, asset_returns, entry_date)
    entry_cost = config.initial_capital * cost_rate  # turnover 1.0: buying the whole book
    value = config.initial_capital - entry_cost
    shares = (value * target) / prices.loc[entry_date]

    n = len(dates) - entry_position
    portfolio_value = np.empty(n)
    portfolio_return = np.zeros(n)
    turnover_col = np.zeros(n)
    costs_col = np.zeros(n)

    portfolio_value[0] = value
    turnover_col[0] = 1.0
    costs_col[0] = entry_cost

    # --- daily loop --------------------------------------------------------
    for i in range(1, n):
        date = dates[entry_position + i]
        previous_value = value

        # Holdings drift with prices; share counts are unchanged since the last trade.
        value = float((shares * prices.loc[date]).sum())

        if date in scheduled:
            drifted = (shares * prices.loc[date]) / value
            target = _weights_at(config, prices, asset_returns, date)
            turnover = 0.5 * float((target - drifted).abs().sum())
            cost = value * turnover * cost_rate

            value -= cost
            shares = (value * target) / prices.loc[date]

            turnover_col[i] = turnover
            costs_col[i] = cost

        portfolio_value[i] = value
        portfolio_return[i] = value / previous_value - 1.0

    # --- benchmark ---------------------------------------------------------
    benchmark_window = benchmark_series.iloc[entry_position:]
    benchmark_value = config.initial_capital * (benchmark_window / benchmark_window.iloc[0])
    benchmark_return = benchmark_value.pct_change().fillna(0.0)

    result = pd.DataFrame(
        {
            "portfolio_value": portfolio_value,
            "portfolio_return": portfolio_return,
            "benchmark_value": benchmark_value.to_numpy(),
            "benchmark_return": benchmark_return.to_numpy(),
            "turnover": turnover_col,
            "costs_paid": costs_col,
        },
        index=dates[entry_position:],
    )
    result.index.name = "date"
    return result


def _entry_position(prices: pd.DataFrame, config: Config) -> int:
    """Index position of the first invested day.

    Inverse-volatility weighting needs `lookback_days` of returns before it can
    form its first estimate, and that history has to come from inside the
    window -- reaching back before `start` for it would mean the backtest quietly
    used data the config never asked for. So the first lookback_days trading days
    are spent warming up, and the equity curve begins after them. The benchmark
    is truncated to exactly the same start so the comparison stays honest.

    Equal and custom weighting need no history and start on day one.
    """
    if config.weighting != "inverse_vol":
        return 0

    lookback = int(config.lookback_days or 0)
    if len(prices) <= lookback + 1:
        raise EngineError(
            f"Need more than {lookback + 1} trading days to warm up an inverse-volatility "
            f"estimate with lookback_days={lookback}, but only {len(prices)} are available."
        )
    return lookback


def _weights_at(
    config: Config, prices: pd.DataFrame, asset_returns: pd.DataFrame, date: pd.Timestamp
) -> pd.Series:
    """Target weights on `date`, aligned to the price columns."""
    weights = resolve_weights(
        scheme=config.weighting,
        tickers=tuple(prices.columns),
        returns=asset_returns,
        as_of=date,
        custom=config.custom_weights,
        lookback_days=config.lookback_days,
    )
    return weights.reindex(prices.columns).fillna(0.0)


def _validate_inputs(prices: pd.DataFrame, benchmark: pd.Series, config: Config) -> None:
    if prices.empty or prices.shape[1] == 0:
        raise EngineError("No price data to backtest.")
    if prices.isna().any().any():
        raise EngineError("Price panel still contains NaNs; clean it before backtesting.")
    if (prices <= 0).any().any():
        raise EngineError("Price panel contains non-positive prices.")
    if not prices.index.equals(benchmark.index):
        raise EngineError("Portfolio and benchmark prices are not aligned to the same dates.")
    if len(prices) < 2:
        raise EngineError("Need at least two trading days to compute a return.")

    # Custom weights are positional, so a dropped ticker would silently shift
    # every weight onto the wrong holding.
    if config.weighting == "custom" and len(prices.columns) != len(config.tickers):
        dropped = sorted(set(config.tickers) - set(prices.columns))
        raise EngineError(
            f"weighting is 'custom' but {', '.join(dropped)} was dropped for missing data. "
            f"Custom weights are positional, so the run is stopped rather than reallocating "
            f"your weights for you. Remove the ticker from the config or change the window."
        )


def _as_series(benchmark: pd.DataFrame | pd.Series) -> pd.Series:
    if isinstance(benchmark, pd.DataFrame):
        if benchmark.shape[1] != 1:
            raise EngineError(
                f"Benchmark must be a single series, got {benchmark.shape[1]} columns."
            )
        return benchmark.iloc[:, 0]
    return benchmark
