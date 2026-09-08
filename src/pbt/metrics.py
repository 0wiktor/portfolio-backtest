"""Performance and risk metrics.

Every metric is a pure function of a daily simple return series, so each one can
be checked against a hand-computed answer in the tests.

Conventions used throughout, stated once here:

* TRADING_DAYS = 252. Everything annualises off that, never off calendar days.
* Rates passed in as `risk_free_rate` are ANNUAL decimals (0.02 = 2% a year) and
  are converted to a daily equivalent geometrically, not by dividing by 252.
* Returns that are undefined (zero volatility, no downside observations) come
  back as NaN rather than as a silently misleading zero or infinity.
* max_drawdown is NEGATIVE (a loss). VaR and CVaR are POSITIVE loss magnitudes,
  which is how they are conventionally quoted: "95% VaR of 1.8%".
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import pandas as pd
from scipy import stats

TRADING_DAYS = 252
VAR_CONFIDENCE = 0.95

# A daily standard deviation below this is floating-point noise around a
# constant series, not real variation. Testing `sigma == 0` is not enough: a
# genuinely constant series sums to a stdev of ~1e-18 rather than exactly zero,
# and dividing by that produces a Sharpe ratio of 7e16 that looks like a result.
ZERO_VOL_TOLERANCE = 1e-12


class Regression(NamedTuple):
    """The three OLS outputs beta, alpha and R-squared are all read from."""

    slope: float
    intercept: float
    rvalue: float


def daily_from_annual_rate(annual_rate: float) -> float:
    """Convert an annual rate to its daily equivalent by compounding.

    (1 + annual) ** (1/252) - 1, not annual/252: the two differ by enough at
    realistic rates to move a Sharpe ratio in the second decimal place.
    """
    return (1.0 + annual_rate) ** (1.0 / TRADING_DAYS) - 1.0


def equity_curve(returns: pd.Series) -> pd.Series:
    """Cumulative growth of 1.0 unit invested, indexed like `returns`."""
    return (1.0 + returns.fillna(0.0)).cumprod()


def drawdown_series(returns: pd.Series) -> pd.Series:
    """Drawdown from the running peak on each date, as a negative decimal."""
    curve = equity_curve(returns)
    return curve / curve.cummax() - 1.0


# --------------------------------------------------------------------------
# The 14 headline metrics
# --------------------------------------------------------------------------

def cagr(returns: pd.Series) -> float:
    """Compound annual growth rate.

    The holding period is len(returns)/252 years, i.e. measured in trading days
    rather than calendar time, so it stays consistent with every other metric.
    """
    if len(returns) == 0:
        return float("nan")
    growth = float((1.0 + returns.fillna(0.0)).prod())
    years = len(returns) / TRADING_DAYS
    if years <= 0 or growth <= 0:
        return float("nan")
    return growth ** (1.0 / years) - 1.0


def annualised_volatility(returns: pd.Series) -> float:
    """Standard deviation of daily returns, scaled by sqrt(252). Sample stdev (ddof=1)."""
    clean = returns.dropna()
    if len(clean) < 2:
        return float("nan")
    sigma = float(clean.std(ddof=1))
    return 0.0 if sigma < ZERO_VOL_TOLERANCE else sigma * np.sqrt(TRADING_DAYS)


def sharpe_ratio(returns: pd.Series, risk_free_rate: float) -> float:
    """Annualised excess return divided by annualised volatility of excess return.

    Args:
        risk_free_rate: annual decimal.

    Returns NaN when excess-return volatility is zero -- an infinite Sharpe is an
    artefact, not a result.
    """
    excess = _excess(returns, risk_free_rate)
    if len(excess) < 2:
        return float("nan")
    sigma = float(excess.std(ddof=1))
    if sigma < ZERO_VOL_TOLERANCE:
        return float("nan")
    return float(excess.mean() * TRADING_DAYS / (sigma * np.sqrt(TRADING_DAYS)))


def sortino_ratio(returns: pd.Series, risk_free_rate: float) -> float:
    """Annualised excess return divided by annualised downside deviation.

    Downside deviation divides by the FULL sample size, not by the count of
    losing days. Dividing by the smaller count is a common variant that inflates
    the ratio, and the two are not comparable.
    """
    excess = _excess(returns, risk_free_rate)
    if len(excess) < 2:
        return float("nan")
    downside = np.minimum(excess, 0.0)
    downside_deviation = float(np.sqrt((downside**2).sum() / len(excess)))
    if downside_deviation < ZERO_VOL_TOLERANCE:
        return float("nan")
    return float(excess.mean() * TRADING_DAYS / (downside_deviation * np.sqrt(TRADING_DAYS)))


def max_drawdown(returns: pd.Series) -> float:
    """Deepest peak-to-trough decline, as a NEGATIVE decimal (-0.34 is -34%)."""
    if len(returns) == 0:
        return float("nan")
    return float(drawdown_series(returns).min())


def calmar_ratio(returns: pd.Series) -> float:
    """CAGR divided by the absolute maximum drawdown."""
    drawdown = max_drawdown(returns)
    if not np.isfinite(drawdown) or drawdown == 0:
        return float("nan")
    return float(cagr(returns) / abs(drawdown))


def beta(returns: pd.Series, benchmark_returns: pd.Series, risk_free_rate: float) -> float:
    """OLS slope of portfolio excess return on benchmark excess return."""
    return _regression(returns, benchmark_returns, risk_free_rate).slope


def jensens_alpha(returns: pd.Series, benchmark_returns: pd.Series, risk_free_rate: float) -> float:
    """Annualised Jensen's alpha: the OLS intercept on daily excess returns × 252.

    Multiplying the daily intercept by 252 is the standard linear annualisation.
    It is an approximation -- it does not compound -- and it is the convention
    alpha is normally quoted under.
    """
    result = _regression(returns, benchmark_returns, risk_free_rate)
    return float(result.intercept * TRADING_DAYS) if np.isfinite(result.intercept) else float("nan")


def tracking_error(returns: pd.Series, benchmark_returns: pd.Series) -> float:
    """Annualised standard deviation of the active return (portfolio minus benchmark)."""
    active = _active_returns(returns, benchmark_returns)
    if len(active) < 2:
        return float("nan")
    return float(active.std(ddof=1) * np.sqrt(TRADING_DAYS))


def information_ratio(returns: pd.Series, benchmark_returns: pd.Series) -> float:
    """Annualised mean active return divided by tracking error."""
    active = _active_returns(returns, benchmark_returns)
    if len(active) < 2:
        return float("nan")
    sigma = float(active.std(ddof=1))
    if sigma < ZERO_VOL_TOLERANCE:
        return float("nan")
    return float(active.mean() * TRADING_DAYS / (sigma * np.sqrt(TRADING_DAYS)))


def historical_var(returns: pd.Series, confidence: float = VAR_CONFIDENCE) -> float:
    """Historical one-day Value at Risk, as a POSITIVE loss magnitude.

    The empirical 5th percentile of daily returns at 95% confidence, sign
    flipped. No distributional assumption is made: this is the observed quantile.
    A positive 0.018 means "on the worst 5% of days, the loss was at least 1.8%".
    """
    clean = returns.dropna()
    if clean.empty:
        return float("nan")
    return float(-np.quantile(clean, 1.0 - confidence))


def historical_cvar(returns: pd.Series, confidence: float = VAR_CONFIDENCE) -> float:
    """Historical Conditional VaR (expected shortfall), as a POSITIVE loss magnitude.

    The mean of all returns at or below the VaR quantile -- the average loss on
    the days that breach VaR, which is what VaR itself does not tell you.
    """
    clean = returns.dropna()
    if clean.empty:
        return float("nan")
    threshold = np.quantile(clean, 1.0 - confidence)
    tail = clean[clean <= threshold]
    if tail.empty:
        return float("nan")
    return float(-tail.mean())


def correlation_to_benchmark(returns: pd.Series, benchmark_returns: pd.Series) -> float:
    """Pearson correlation of daily returns (raw, not excess)."""
    portfolio, bench = _aligned(returns, benchmark_returns)
    if len(portfolio) < 2:
        return float("nan")
    return float(portfolio.corr(bench))


def r_squared(returns: pd.Series, benchmark_returns: pd.Series, risk_free_rate: float) -> float:
    """Share of portfolio excess-return variance explained by the benchmark.

    Taken from the same regression as beta and alpha, so the three are always
    mutually consistent.
    """
    result = _regression(returns, benchmark_returns, risk_free_rate)
    return float(result.rvalue**2) if np.isfinite(result.rvalue) else float("nan")


def _excess(returns: pd.Series, risk_free_rate: float) -> pd.Series:
    """Daily returns net of the daily risk-free rate."""
    return returns.dropna() - daily_from_annual_rate(risk_free_rate)


def _active_returns(returns: pd.Series, benchmark_returns: pd.Series) -> pd.Series:
    portfolio, bench = _aligned(returns, benchmark_returns)
    return portfolio - bench


def _aligned(returns: pd.Series, benchmark_returns: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Restrict two series to their shared, non-null dates."""
    frame = pd.concat([returns, benchmark_returns], axis=1, keys=["p", "b"]).dropna()
    return frame["p"], frame["b"]


def _regression(returns: pd.Series, benchmark_returns: pd.Series, risk_free_rate: float) -> Regression:
    """OLS of portfolio excess return on benchmark excess return."""
    daily_rf = daily_from_annual_rate(risk_free_rate)
    portfolio, bench = _aligned(returns, benchmark_returns)
    nan = float("nan")
    if len(portfolio) < 2 or float(bench.std(ddof=1)) < ZERO_VOL_TOLERANCE:
        return Regression(nan, nan, nan)
    fit = stats.linregress(bench - daily_rf, portfolio - daily_rf)
    return Regression(float(fit.slope), float(fit.intercept), float(fit.rvalue))
