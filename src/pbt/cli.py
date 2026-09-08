"""Command line entry point: `pbt run --config config/example.yaml`.

This module is the only place that turns exceptions into user-facing text. Every
error the pipeline raises deliberately (ConfigError, DataError, EngineError,
WeightError) is caught here and printed as one clear line with a non-zero exit
code, because a traceback is a bug report, not an error message.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from . import montecarlo, plots
from .config import Config, ConfigError, load_config
from .data import DataError, align_to_common_dates, clean_prices, fetch_prices
from .engine import EngineError, run_backtest
from .report import build_report, comparison_table, write_report
from .weights import WeightError

DEFAULT_OUTPUT_DIR = "output"
DEFAULT_CACHE_DIR = "data_cache"
REPORT_FILENAME = "report.txt"

# Paths and the simulation seed are run-environment settings, not parameters of
# the backtest itself, so they live here rather than in the YAML -- the config
# validator rejects unknown keys precisely so its key list stays the spec.
EXPECTED_ERRORS = (ConfigError, DataError, EngineError, WeightError)


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        return _run(args)
    except EXPECTED_ERRORS as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pbt", description="Backtest an equity portfolio.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="run a backtest from a config file")
    run.add_argument("--config", required=True, help="path to the YAML run configuration")
    run.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR,
                     help=f"where charts and the report are written (default: {DEFAULT_OUTPUT_DIR})")
    run.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR,
                     help=f"parquet price cache (default: {DEFAULT_CACHE_DIR})")
    run.add_argument("--seed", type=int, default=0,
                     help="Monte Carlo seed; fixed so runs reproduce (default: 0)")
    run.add_argument("--paths", type=int, default=montecarlo.N_PATHS,
                     help=f"Monte Carlo paths (default: {montecarlo.N_PATHS:,})")
    return parser


def _run(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    prices, benchmark = load_market_data(config, args.cache_dir)

    result = run_backtest(prices, benchmark, config)
    comparison = comparison_table(
        result["portfolio_return"], result["benchmark_return"], config.risk_free_rate
    )
    simulation = montecarlo.simulate(
        result["portfolio_return"],
        result["benchmark_return"],
        config.initial_capital,
        n_paths=args.paths,
        seed=args.seed,
    )

    output_dir = Path(args.output_dir)
    written = plots.generate_all(result, simulation, config.risk_free_rate, output_dir)

    text = build_report(result, config, comparison, simulation)
    print(text)
    report_path = write_report(text, output_dir / REPORT_FILENAME)

    print(f"\nWrote {report_path}")
    for path in written:
        print(f"Wrote {path}")
    return 0


def load_market_data(config: Config, cache_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch, clean and align the holdings and the benchmark.

    Split out from _run so a notebook can reach the same data path in one call
    without going through argparse.
    """
    prices = clean_prices(
        fetch_prices(config.tickers, config.start, config.end, cache_dir), config.start
    )
    benchmark = clean_prices(
        fetch_prices([config.benchmark], config.start, config.end, cache_dir),
        config.start,
        protected=config.benchmark,
    )
    return align_to_common_dates(prices, benchmark)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
