"""Engine tests: rebalance scheduling, turnover arithmetic, and cost accounting.

The toy portfolio is built so every number can be derived on paper. Prices are
synthetic and no test touches the network.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from pbt.config import Config
from pbt.engine import EngineError, rebalance_dates, run_backtest

TEN_BPS = 10.0  # 0.001 of traded value


def make_config(**overrides) -> Config:
    base = dict(
        tickers=("A", "B"),
        benchmark="BM",
        start=dt.date(2020, 1, 1),
        end=dt.date(2020, 2, 28),
        initial_capital=100_000.0,
        weighting="equal",
        rebalance="monthly",
        risk_free_rate=0.0,
        transaction_cost_bps=TEN_BPS,
        custom_weights=None,
        lookback_days=None,
    )
    base.update(overrides)
    return Config(**base)


def toy_prices() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Two assets. A doubles on the January rebalance date; B never moves.

    That makes the drifted weights on 2020-01-31 exactly 2/3 and 1/3.
    """
    dates = pd.bdate_range("2020-01-01", "2020-02-28")
    prices = pd.DataFrame({"A": 100.0, "B": 100.0}, index=dates)
    prices.loc[dates >= pd.Timestamp("2020-01-31"), "A"] = 200.0
    benchmark = pd.DataFrame({"BM": 50.0}, index=dates)
    return prices, benchmark


# --------------------------------------------------------------------------
# Rebalance calendar
# --------------------------------------------------------------------------

def test_monthly_rebalance_lands_on_the_last_trading_day_not_the_calendar_end():
    # 31 May 2020 was a Sunday, so the May rebalance must fall on Friday 29 May.
    index = pd.bdate_range("2019-01-01", "2021-06-30")
    dates = rebalance_dates(index, "monthly")

    assert pd.Timestamp("2020-05-29") in dates
    assert pd.Timestamp("2020-05-31") not in dates
    assert all(date in index for date in dates)


def test_quarterly_and_annual_land_on_real_trading_days():
    index = pd.bdate_range("2019-01-01", "2021-06-30")

    quarterly = rebalance_dates(index, "quarterly")
    # 31 March 2019 was a Sunday -> Friday 29 March.
    assert pd.Timestamp("2019-03-29") in quarterly
    assert pd.Timestamp("2019-06-28") in quarterly  # 30 June 2019 was a Sunday

    annual = rebalance_dates(index, "annual")
    assert list(annual) == [pd.Timestamp("2019-12-31"), pd.Timestamp("2020-12-31")]


def test_first_and_last_dates_are_never_rebalance_dates():
    # The window starts and ends exactly on month ends.
    index = pd.bdate_range("2020-01-31", "2020-03-31")
    dates = rebalance_dates(index, "monthly")

    assert list(dates) == [pd.Timestamp("2020-02-28")]


def test_frequency_none_schedules_nothing():
    index = pd.bdate_range("2020-01-01", "2020-12-31")
    assert len(rebalance_dates(index, "none")) == 0


# --------------------------------------------------------------------------
# Costs and turnover
# --------------------------------------------------------------------------

def test_entry_is_charged_at_full_turnover():
    # Buying the whole book is turnover 1.0, so 10 bps on 100,000 is 100.
    prices, benchmark = toy_prices()
    result = run_backtest(prices, benchmark, make_config(rebalance="none"))

    assert result["turnover"].iloc[0] == pytest.approx(1.0)
    assert result["costs_paid"].iloc[0] == pytest.approx(100.0)
    assert result["portfolio_value"].iloc[0] == pytest.approx(99_900.0)


def test_no_costs_after_entry_when_never_rebalancing():
    prices, benchmark = toy_prices()
    result = run_backtest(prices, benchmark, make_config(rebalance="none"))

    assert result["costs_paid"].sum() == pytest.approx(100.0)
    assert result["turnover"].iloc[1:].sum() == 0.0


def test_turnover_and_cost_on_a_two_asset_rebalance():
    """Hand-derived: entry at 50/50, A doubles, so weights drift to 2/3 and 1/3.

    turnover = 0.5 * (|0.5 - 2/3| + |0.5 - 1/3|) = 0.5 * (1/6 + 1/6) = 1/6
    """
    prices, benchmark = toy_prices()
    result = run_backtest(prices, benchmark, make_config(rebalance="monthly"))
    row = result.loc[pd.Timestamp("2020-01-31")]

    # Value before the trade: 99,900 * 1.5 = 149,850.
    expected_cost = 149_850.0 * (1 / 6) * (TEN_BPS / 10_000.0)

    assert row["turnover"] == pytest.approx(1 / 6, abs=1e-12)
    assert row["costs_paid"] == pytest.approx(expected_cost, abs=1e-9)
    assert row["portfolio_value"] == pytest.approx(149_850.0 - expected_cost, abs=1e-9)
    assert expected_cost == pytest.approx(24.975, abs=1e-9)


def test_daily_return_is_net_of_the_cost_charged_that_day():
    prices, benchmark = toy_prices()
    result = run_backtest(prices, benchmark, make_config(rebalance="monthly"))

    previous = result["portfolio_value"].loc[pd.Timestamp("2020-01-30")]
    row = result.loc[pd.Timestamp("2020-01-31")]
    assert row["portfolio_return"] == pytest.approx(row["portfolio_value"] / previous - 1.0)


def test_only_scheduled_dates_incur_turnover():
    prices, benchmark = toy_prices()
    result = run_backtest(prices, benchmark, make_config(rebalance="monthly"))
    traded = result.index[result["turnover"] > 0]

    assert list(traded) == [result.index[0], pd.Timestamp("2020-01-31")]


# --------------------------------------------------------------------------
# Benchmark and output shape
# --------------------------------------------------------------------------

def test_benchmark_is_frictionless_and_starts_at_full_capital():
    prices, benchmark = toy_prices()
    result = run_backtest(prices, benchmark, make_config())

    # The portfolio pays to get invested; the benchmark does not. That gap is
    # deliberate and documented.
    assert result["benchmark_value"].iloc[0] == pytest.approx(100_000.0)
    assert result["portfolio_value"].iloc[0] == pytest.approx(99_900.0)


def test_result_carries_the_expected_tidy_columns():
    prices, benchmark = toy_prices()
    result = run_backtest(prices, benchmark, make_config())

    assert result.index.name == "date"
    assert list(result.reset_index().columns) == [
        "date",
        "portfolio_value",
        "portfolio_return",
        "benchmark_value",
        "benchmark_return",
        "turnover",
        "costs_paid",
    ]


def test_flat_benchmark_produces_flat_benchmark_value():
    prices, benchmark = toy_prices()
    result = run_backtest(prices, benchmark, make_config())
    assert result["benchmark_value"].nunique() == 1


# --------------------------------------------------------------------------
# Warm-up and guard rails
# --------------------------------------------------------------------------

def test_inverse_vol_consumes_a_lookback_warm_up_before_investing():
    dates = pd.bdate_range("2020-01-01", periods=100)
    rng = np.random.default_rng(1)
    prices = pd.DataFrame(
        {
            "A": 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 100))),
            "B": 100 * np.exp(np.cumsum(rng.normal(0, 0.02, 100))),
        },
        index=dates,
    )
    benchmark = pd.DataFrame({"BM": 50.0}, index=dates)
    config = make_config(weighting="inverse_vol", lookback_days=10, rebalance="none")

    result = run_backtest(prices, benchmark, config)

    # The first 10 trading days estimate the initial volatility, so the curve
    # starts on the 11th and the benchmark is truncated identically.
    assert result.index[0] == dates[10]
    assert len(result) == 90
    assert result["benchmark_value"].iloc[0] == pytest.approx(100_000.0)


def test_custom_weights_refuse_to_run_when_a_ticker_was_dropped():
    prices, benchmark = toy_prices()
    config = make_config(
        weighting="custom", tickers=("A", "B", "C"), custom_weights=(0.4, 0.3, 0.3)
    )
    with pytest.raises(EngineError, match="dropped for missing data"):
        run_backtest(prices, benchmark, config)


def test_misaligned_benchmark_is_rejected():
    prices, benchmark = toy_prices()
    with pytest.raises(EngineError, match="not aligned"):
        run_backtest(prices, benchmark.iloc[:-5], make_config())


def test_nan_prices_are_rejected():
    prices, benchmark = toy_prices()
    prices.iloc[5, 0] = np.nan
    with pytest.raises(EngineError, match="NaNs"):
        run_backtest(prices, benchmark, make_config())
