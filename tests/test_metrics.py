"""Metric tests.

Every expected value here is computed by hand or from an explicitly written-out
formula. Nothing is compared against the library's own output, because a metric
that agrees with itself proves only that it is deterministic.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from pbt import metrics
from pbt.report import drawdown_episodes

TRADING_DAYS = 252


def series(values: list[float], start: str = "2020-01-01") -> pd.Series:
    """A daily return series on consecutive business days."""
    return pd.Series(values, index=pd.bdate_range(start, periods=len(values)), dtype="float64")


# --------------------------------------------------------------------------
# Growth and volatility
# --------------------------------------------------------------------------

def test_cagr_of_constant_return_is_exact():
    # 252 days of exactly +0.05% compounds to precisely (1.0005 ** 252) - 1.
    daily = 0.0005
    returns = series([daily] * TRADING_DAYS)
    expected = (1.0 + daily) ** TRADING_DAYS - 1.0
    assert metrics.cagr(returns) == pytest.approx(expected, abs=1e-12)


def test_constant_return_has_zero_volatility():
    returns = series([0.0005] * TRADING_DAYS)
    assert metrics.annualised_volatility(returns) == 0.0


def test_zero_volatility_makes_sharpe_undefined_not_infinite():
    # A riskless series has no defined risk-adjusted return; NaN is the honest
    # answer, and an infinity here would propagate into the report.
    returns = series([0.0005] * TRADING_DAYS)
    assert math.isnan(metrics.sharpe_ratio(returns, risk_free_rate=0.0))


def test_annualised_volatility_matches_written_out_formula():
    values = [0.01, -0.02, 0.03, -0.01, 0.005]
    returns = series(values)
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)  # sample, ddof=1
    assert metrics.annualised_volatility(returns) == pytest.approx(
        math.sqrt(variance) * math.sqrt(TRADING_DAYS)
    )


def test_daily_risk_free_compounds_rather_than_divides():
    # 2% a year is NOT 0.02/252 a day; the geometric conversion is the convention.
    assert metrics.daily_from_annual_rate(0.02) == pytest.approx(1.02 ** (1 / 252) - 1)


# --------------------------------------------------------------------------
# Drawdown family
# --------------------------------------------------------------------------

def test_max_drawdown_is_exact_peak_to_trough():
    # Curve: 1.10, then 0.88, then 0.924. Deepest fall is 1.10 -> 0.88 = -20%.
    returns = series([0.10, -0.20, 0.05])
    assert metrics.max_drawdown(returns) == pytest.approx(-0.20, abs=1e-12)


def test_calmar_is_cagr_over_absolute_drawdown():
    returns = series([0.10, -0.20, 0.05])
    assert metrics.calmar_ratio(returns) == pytest.approx(metrics.cagr(returns) / 0.20)


def test_drawdown_episodes_finds_peak_trough_and_recovery():
    # Down 10%, down another ~10%, then two gains that carry it back above the peak.
    returns = series([0.0, -0.10, -0.10, 0.15, 0.10])
    episodes = drawdown_episodes(returns)

    assert len(episodes) == 1
    row = episodes.iloc[0]
    assert row["peak"] == returns.index[0]
    assert row["trough"] == returns.index[2]
    assert row["recovery"] == returns.index[4]
    # 0.9 * 0.9 = 0.81 against a peak of 1.0.
    assert row["depth"] == pytest.approx(-0.19, abs=1e-12)


def test_drawdown_episodes_marks_an_unrecovered_fall():
    returns = series([0.0, -0.10, 0.01])
    row = drawdown_episodes(returns).iloc[0]
    assert pd.isna(row["recovery"])
    assert pd.isna(row["days_to_recover"])


# --------------------------------------------------------------------------
# Tail risk
# --------------------------------------------------------------------------

def test_historical_var_and_cvar_on_a_hand_checked_tail():
    # 20 returns from -0.10 to +0.09 in 0.01 steps.
    # numpy's linear interpolation puts the 5th percentile at position
    # 0.05 * (20 - 1) = 0.95, i.e. 95% of the way from -0.10 to -0.09.
    values = [round(-0.10 + 0.01 * i, 10) for i in range(20)]
    returns = series(values)

    assert metrics.historical_var(returns) == pytest.approx(0.0905, abs=1e-10)
    # Only -0.10 sits at or below that threshold, so the expected shortfall is 10%.
    assert metrics.historical_cvar(returns) == pytest.approx(0.10, abs=1e-10)


def test_var_is_reported_as_a_positive_loss():
    returns = series([-0.05, 0.01, 0.02, 0.03, 0.04])
    assert metrics.historical_var(returns) > 0


# --------------------------------------------------------------------------
# Benchmark-relative metrics
# --------------------------------------------------------------------------

def test_beta_alpha_and_r_squared_recover_a_constructed_relationship():
    # Build a portfolio that is exactly alpha + beta * benchmark in EXCESS terms,
    # with no residual. OLS must return those parameters and an R-squared of 1.
    true_beta, true_daily_alpha, annual_rf = 1.30, 0.0002, 0.02
    daily_rf = 1.02 ** (1 / 252) - 1

    rng = np.random.default_rng(7)
    benchmark = series(list(rng.normal(0.0004, 0.011, 300)))
    portfolio = daily_rf + true_daily_alpha + true_beta * (benchmark - daily_rf)

    assert metrics.beta(portfolio, benchmark, annual_rf) == pytest.approx(true_beta, abs=1e-9)
    assert metrics.jensens_alpha(portfolio, benchmark, annual_rf) == pytest.approx(
        true_daily_alpha * TRADING_DAYS, abs=1e-9
    )
    assert metrics.r_squared(portfolio, benchmark, annual_rf) == pytest.approx(1.0, abs=1e-9)


def test_benchmark_against_itself_has_unit_beta_and_zero_alpha():
    rng = np.random.default_rng(11)
    benchmark = series(list(rng.normal(0.0003, 0.01, 200)))

    assert metrics.beta(benchmark, benchmark, 0.02) == pytest.approx(1.0, abs=1e-12)
    assert metrics.jensens_alpha(benchmark, benchmark, 0.02) == pytest.approx(0.0, abs=1e-12)
    assert metrics.tracking_error(benchmark, benchmark) == pytest.approx(0.0, abs=1e-15)
    assert metrics.correlation_to_benchmark(benchmark, benchmark) == pytest.approx(1.0)


def test_tracking_error_and_information_ratio_on_an_alternating_active_return():
    # Active return alternates +/- 0.001, so its mean is exactly zero and the
    # information ratio must be zero, while tracking error is the sample stdev
    # of that alternating series, annualised.
    n = 100
    rng = np.random.default_rng(3)
    benchmark = series(list(rng.normal(0.0005, 0.009, n)))
    active = pd.Series([0.001 if i % 2 == 0 else -0.001 for i in range(n)], index=benchmark.index)
    portfolio = benchmark + active

    expected_te = 0.001 * math.sqrt(n / (n - 1)) * math.sqrt(TRADING_DAYS)
    assert metrics.tracking_error(portfolio, benchmark) == pytest.approx(expected_te, abs=1e-12)
    assert metrics.information_ratio(portfolio, benchmark) == pytest.approx(0.0, abs=1e-12)


def test_sortino_matches_hand_computed_downside_deviation():
    # rf = 0, so excess returns are the returns themselves.
    values = [0.02, -0.01, 0.02, -0.01]
    returns = series(values)

    mean = sum(values) / len(values)                       # 0.005
    downside_sq = sum(min(v, 0.0) ** 2 for v in values)    # 2e-4
    # Denominator is the FULL sample size, not the count of losing days.
    downside_deviation = math.sqrt(downside_sq / len(values))
    expected = mean * TRADING_DAYS / (downside_deviation * math.sqrt(TRADING_DAYS))

    assert metrics.sortino_ratio(returns, risk_free_rate=0.0) == pytest.approx(expected, abs=1e-12)


def test_sortino_is_undefined_when_nothing_ever_falls():
    returns = series([0.01, 0.02, 0.03])
    assert math.isnan(metrics.sortino_ratio(returns, risk_free_rate=0.0))


def test_correlation_of_a_perfect_linear_transform_is_one():
    rng = np.random.default_rng(5)
    benchmark = series(list(rng.normal(0.0, 0.01, 50)))
    portfolio = 2.0 * benchmark + 0.001
    assert metrics.correlation_to_benchmark(portfolio, benchmark) == pytest.approx(1.0, abs=1e-12)
