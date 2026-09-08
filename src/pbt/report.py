"""Presentation of results: the side-by-side table, drawdown episodes, and the
plain-text summary.

Deliberately absent: any letter grade, star rating, or Excellent/Good/Poor
verdict. The report states what happened and leaves the judgement to the reader,
because a backtest of a hand-picked ticker list does not contain enough evidence
to grade a strategy, and a confident-sounding label would imply that it does.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import metrics
from .config import Config
from .montecarlo import SimulationResult

TOP_DRAWDOWNS = 3
WIDTH = 78

# Which metrics read as percentages rather than as bare ratios.
PERCENT_METRICS = {
    "CAGR",
    "Annualised volatility",
    "Max drawdown",
    "Jensen's alpha (annual)",
    "Tracking error",
    "Historical VaR 95% (daily loss)",
    "Historical CVaR 95% (daily loss)",
}


def drawdown_episodes(returns: pd.Series) -> pd.DataFrame:
    """Every distinct peak-to-trough-to-recovery episode, deepest first.

    Columns: peak, trough, recovery (NaT if unrecovered by the end of the
    sample), depth (negative decimal), days_to_recover (trading days, NaN if
    unrecovered).
    """
    drawdown = metrics.drawdown_series(returns)
    underwater = drawdown < 0
    if not underwater.any():
        return pd.DataFrame(columns=["peak", "trough", "recovery", "depth", "days_to_recover"])

    episode_id = (underwater != underwater.shift()).cumsum()
    rows = []
    for _, segment in drawdown[underwater].groupby(episode_id[underwater]):
        start_position = drawdown.index.get_loc(segment.index[0])
        # The peak is the last date the curve was still at its running high.
        peak = drawdown.index[start_position - 1] if start_position > 0 else segment.index[0]
        end_position = drawdown.index.get_loc(segment.index[-1])
        recovered = end_position + 1 < len(drawdown)
        rows.append(
            {
                "peak": peak,
                "trough": segment.idxmin(),
                "recovery": drawdown.index[end_position + 1] if recovered else pd.NaT,
                "depth": float(segment.min()),
                "days_to_recover": (end_position + 1 - start_position) if recovered else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values("depth").reset_index(drop=True)


def comparison_table(
    portfolio_returns: pd.Series, benchmark_returns: pd.Series, risk_free_rate: float
) -> pd.DataFrame:
    """All 14 metrics for portfolio and benchmark, side by side.

    The benchmark column runs the identical functions with the benchmark as its
    own reference, which is why its beta is 1.0, its alpha and tracking error
    are 0.0, and its information ratio is undefined.
    """

    def column(series: pd.Series) -> dict[str, float]:
        return {
            "CAGR": metrics.cagr(series),
            "Annualised volatility": metrics.annualised_volatility(series),
            "Sharpe ratio": metrics.sharpe_ratio(series, risk_free_rate),
            "Sortino ratio": metrics.sortino_ratio(series, risk_free_rate),
            "Max drawdown": metrics.max_drawdown(series),
            "Calmar ratio": metrics.calmar_ratio(series),
            "Beta vs benchmark": metrics.beta(series, benchmark_returns, risk_free_rate),
            "Jensen's alpha (annual)": metrics.jensens_alpha(
                series, benchmark_returns, risk_free_rate
            ),
            "Tracking error": metrics.tracking_error(series, benchmark_returns),
            "Information ratio": metrics.information_ratio(series, benchmark_returns),
            "Historical VaR 95% (daily loss)": metrics.historical_var(series),
            "Historical CVaR 95% (daily loss)": metrics.historical_cvar(series),
            "Correlation to benchmark": metrics.correlation_to_benchmark(series, benchmark_returns),
            "R-squared vs benchmark": metrics.r_squared(series, benchmark_returns, risk_free_rate),
        }

    return pd.DataFrame(
        {"Portfolio": column(portfolio_returns), "Benchmark": column(benchmark_returns)}
    )


def build_report(
    result: pd.DataFrame,
    config: Config,
    comparison: pd.DataFrame,
    simulation: SimulationResult | None = None,
) -> str:
    """Assemble the plain-text summary."""
    sections = [
        _header(result, config),
        _headline(result, comparison),
        _drawdowns(result),
        _costs(result, config),
    ]
    if simulation is not None:
        sections.append(_simulation(simulation))
    sections.append(_caveat())
    return "\n".join(sections)


def write_report(text: str, path: str | Path) -> Path:
    """Write the summary to disk and return the path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _header(result: pd.DataFrame, config: Config) -> str:
    weighting = config.weighting
    if config.weighting == "inverse_vol":
        weighting += f" ({config.lookback_days}-day lookback)"

    lines = [
        "=" * WIDTH,
        "PORTFOLIO BACKTEST".center(WIDTH),
        "=" * WIDTH,
        f"Holdings          {', '.join(config.tickers)}",
        f"Benchmark         {config.benchmark}",
        f"Configured window {config.start} to {config.end}",
        f"Invested window   {result.index[0].date()} to {result.index[-1].date()} "
        f"({len(result):,} trading days)",
        f"Weighting         {weighting}",
        f"Rebalancing       {config.rebalance}",
        f"Initial capital   {config.initial_capital:,.2f}",
        f"Costs             {config.transaction_cost_bps:g} bps on turnover",
        f"Risk-free rate    {config.risk_free_rate:.2%} a year",
    ]
    if result.index[0].date() > config.start:
        lines.append(
            "Note              the invested window starts after the configured start "
            "because\n                  inverse-volatility weighting spends its lookback "
            "period estimating\n                  the first set of weights."
        )
    return "\n".join(lines)


def _headline(result: pd.DataFrame, comparison: pd.DataFrame) -> str:
    portfolio_total = result["portfolio_value"].iloc[-1] / result["portfolio_value"].iloc[0] - 1
    benchmark_total = result["benchmark_value"].iloc[-1] / result["benchmark_value"].iloc[0] - 1

    beat_return = portfolio_total > benchmark_total
    beat_sharpe = comparison.loc["Sharpe ratio", "Portfolio"] > comparison.loc[
        "Sharpe ratio", "Benchmark"
    ]

    lines = [
        "",
        _rule("HEADLINE"),
        f"{'':34}{'Portfolio':>14}{'Benchmark':>14}",
        f"{'Final value':34}{result['portfolio_value'].iloc[-1]:>14,.0f}"
        f"{result['benchmark_value'].iloc[-1]:>14,.0f}",
        f"{'Total return':34}{portfolio_total:>14.2%}{benchmark_total:>14.2%}",
        "",
        _rule("METRICS"),
    ]
    for name, row in comparison.iterrows():
        lines.append(f"{name:34}{_format(name, row['Portfolio']):>14}"
                     f"{_format(name, row['Benchmark']):>14}")

    lines += [
        "",
        f"Beat the benchmark on total return: {'yes' if beat_return else 'no'}",
        f"Beat the benchmark on Sharpe ratio: {'yes' if beat_sharpe else 'no'}",
        f"Beat the benchmark on BOTH:         {'yes' if beat_return and beat_sharpe else 'no'}",
    ]
    return "\n".join(lines)


def _drawdowns(result: pd.DataFrame) -> str:
    episodes = drawdown_episodes(result["portfolio_return"]).head(TOP_DRAWDOWNS)
    lines = ["", _rule(f"{TOP_DRAWDOWNS} LARGEST PORTFOLIO DRAWDOWNS")]

    if episodes.empty:
        lines.append("The portfolio never fell below a previous peak.")
        return "\n".join(lines)

    lines.append(f"{'Depth':>8}  {'Peak':12}{'Trough':12}{'Recovered':12}{'Trading days':>13}")
    for _, row in episodes.iterrows():
        recovered = "not yet" if pd.isna(row["recovery"]) else str(row["recovery"].date())
        days = "-" if pd.isna(row["days_to_recover"]) else f"{int(row['days_to_recover']):,}"
        lines.append(
            f"{row['depth']:>8.2%}  {str(row['peak'].date()):12}"
            f"{str(row['trough'].date()):12}{recovered:12}{days:>13}"
        )
    return "\n".join(lines)


def _costs(result: pd.DataFrame, config: Config) -> str:
    total = result["costs_paid"].sum()
    rebalances = int((result["turnover"] > 0).sum()) - 1  # the first is the entry trade
    traded = result.loc[result["turnover"] > 0, "turnover"].iloc[1:]

    return "\n".join(
        [
            "",
            _rule("TRANSACTION COSTS"),
            f"Total paid                {total:>14,.2f}",
            f"As share of capital       {total / config.initial_capital:>14.3%}",
            f"Entry trade               {result['costs_paid'].iloc[0]:>14,.2f}",
            f"Rebalances after entry    {rebalances:>14,}",
            f"Mean turnover per rebalance {(traded.mean() if len(traded) else 0.0):>12.2%}",
        ]
    )


def _simulation(simulation: SimulationResult) -> str:
    lines = [
        "",
        _rule("MONTE CARLO (stationary block bootstrap)"),
        f"{simulation.n_paths:,} paths, portfolio and benchmark resampled jointly so their",
        "historical correlation is preserved. This resamples the past; it is not a forecast.",
        "",
        f"{'Horizon':>8}{'10th pct':>13}{'Median':>13}{'90th pct':>13}"
        f"{'P(loss)':>10}{'P(beat bm)':>12}",
    ]
    for horizon in simulation.horizons:
        lines.append(
            f"{horizon.years:>6}y{horizon.percentile_10:>13,.0f}{horizon.median:>13,.0f}"
            f"{horizon.percentile_90:>13,.0f}{horizon.probability_below_initial:>10.1%}"
            f"{horizon.probability_beats_benchmark:>12.1%}"
        )
    return "\n".join(lines)


def _caveat() -> str:
    return "\n".join(
        [
            "",
            _rule("READ THIS BEFORE DRAWING CONCLUSIONS"),
            "The holdings were chosen today, with today's knowledge of which of them",
            "survived and did well. That selection effect inflates these results and no",
            "part of this engine corrects for it. See Known Limitations in the README",
            "for what else is not modelled.",
            "=" * WIDTH,
        ]
    )


def _rule(title: str) -> str:
    return f"--- {title} " + "-" * max(0, WIDTH - len(title) - 5)


def _format(name: str, value: float) -> str:
    """Percentages for return-like metrics, two decimals for ratios."""
    if not np.isfinite(value):
        return "n/a"
    return f"{value:.2%}" if name in PERCENT_METRICS else f"{value:.2f}"
