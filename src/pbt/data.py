"""Price acquisition, on-disk caching, and missing-data handling.

Two design choices worth stating up front:

1. Prices come from yfinance with auto_adjust=True, which folds dividend and
   split adjustments into the Close column. There is deliberately no reference
   anywhere to a separate "Adj Close" column -- with auto_adjust on, Close IS
   the adjusted series, and reading both is a classic double-counting bug.

2. Downloading is injected as a callable rather than called directly. That is
   what lets the test suite run with no network at all: pass a downloader that
   raises, and any accidental network path fails loudly instead of quietly
   working on the developer's machine and failing elsewhere.
"""

from __future__ import annotations

import datetime as dt
import re
from pathlib import Path
from typing import Callable

import pandas as pd


class DataError(RuntimeError):
    """Price data is absent, too sparse, or does not cover the requested window."""


# A gap of a week or less is a holiday or a halt, and carrying the last price
# across it is standard. Longer gaps are a data problem, not something to patch.
FORWARD_FILL_LIMIT = 5

# Above this share of missing observations a ticker is dropped, not patched.
MAX_MISSING_FRACTION = 0.10

# A ticker whose history starts within this many days of the requested start
# still counts as covering it: absorbs a holiday start, not a late listing.
COVERAGE_SLACK_DAYS = 7

# fetch_prices calls this with (ticker, start, end) and expects a date-indexed
# Series of adjusted closing prices covering [start, end] inclusive.
Downloader = Callable[[str, dt.date, dt.date], pd.Series]


def fetch_prices(
    tickers: tuple[str, ...] | list[str],
    start: dt.date,
    end: dt.date,
    cache_dir: str | Path,
    downloader: Downloader | None = None,
) -> pd.DataFrame:
    """Return adjusted closing prices, one column per ticker, indexed by date.

    Each ticker is cached to its own parquet file keyed by ticker and date
    range, so a repeat run with identical parameters makes no network calls at
    all -- the downloader is never invoked on a cache hit.

    Args:
        start, end: inclusive calendar bounds.
        cache_dir: created if absent.
        downloader: injection point for tests; defaults to yfinance.
    """
    if not tickers:
        raise DataError("No tickers requested.")

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    fetch = downloader if downloader is not None else download_from_yfinance

    columns: dict[str, pd.Series] = {}
    for ticker in tickers:
        series = _load_cached(ticker, start, end, cache_dir)
        if series is None:
            series = _normalise_series(fetch(ticker, start, end), ticker)
            if series.empty:
                raise DataError(
                    f"No price data returned for {ticker} between {start} and {end}. "
                    f"Check the symbol is correct and traded over that window."
                )
            _write_cache(series, ticker, start, end, cache_dir)
        columns[ticker] = series

    panel = pd.DataFrame(columns)
    panel = panel.loc[(panel.index.date >= start) & (panel.index.date <= end)]
    panel.index.name = "date"
    return panel.sort_index()


def clean_prices(panel: pd.DataFrame, start: dt.date, protected: str | None = None) -> pd.DataFrame:
    """Apply the missing-data policy and return a gap-free price panel.

    Policy, in order:
      1. A ticker whose history begins after the requested start is an error, not
         something to silently work around by shortening the window.
      2. Gaps of up to FORWARD_FILL_LIMIT trading days are forward-filled.
      3. A ticker still missing more than MAX_MISSING_FRACTION of the window is
         dropped, and the reason is printed.
      4. Any dates still incomplete across the survivors are dropped so the
         engine never sees a NaN price.

    Args:
        protected: a ticker that must not be dropped (the benchmark); if it
            fails the coverage test the run cannot continue at all.
    """
    if panel.empty:
        raise DataError("Price panel is empty; nothing to backtest.")

    _check_coverage(panel, start)

    filled = panel.ffill(limit=FORWARD_FILL_LIMIT)

    missing_fraction = filled.isna().mean()
    doomed = [t for t in filled.columns if missing_fraction[t] > MAX_MISSING_FRACTION]

    if protected is not None and protected in doomed:
        raise DataError(
            f"Benchmark {protected} is missing "
            f"{missing_fraction[protected]:.1%} of the window; cannot proceed."
        )

    for ticker in doomed:
        print(
            f"[data] Dropping {ticker}: {missing_fraction[ticker]:.1%} of observations "
            f"missing after forward-filling gaps of up to {FORWARD_FILL_LIMIT} days "
            f"(limit is {MAX_MISSING_FRACTION:.0%})."
        )
    survivors = filled.drop(columns=doomed)

    if survivors.empty or survivors.shape[1] == 0:
        raise DataError("Every ticker was dropped for missing data; nothing left to backtest.")

    complete = survivors.dropna(how="any")
    dropped_rows = len(survivors) - len(complete)
    if dropped_rows:
        print(
            f"[data] Dropping {dropped_rows} date(s) where at least one surviving "
            f"ticker still had no price after forward-filling."
        )

    if complete.empty:
        raise DataError("No dates remain where every ticker has a price.")
    return complete


def align_to_common_dates(
    portfolio: pd.DataFrame, benchmark: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Restrict both panels to the dates they share.

    The portfolio and the benchmark must advance on the same clock, or every
    relative metric (beta, tracking error, information ratio) silently compares
    returns from different days.
    """
    common = portfolio.index.intersection(benchmark.index)
    if len(common) < 2:
        raise DataError(
            "Portfolio and benchmark share fewer than two common trading days; "
            "check the tickers and the date range."
        )
    return portfolio.loc[common], benchmark.loc[common]


def download_from_yfinance(ticker: str, start: dt.date, end: dt.date) -> pd.Series:
    """Download one ticker's adjusted closes from Yahoo Finance.

    Imported lazily so the test suite never even loads yfinance. Note the end
    date is bumped by a day because yfinance treats `end` as exclusive.
    """
    import yfinance as yf

    raw = yf.download(
        ticker,
        start=start.isoformat(),
        end=(end + dt.timedelta(days=1)).isoformat(),
        auto_adjust=True,
        actions=False,
        progress=False,
        threads=False,
    )
    if raw is None or raw.empty:
        return pd.Series(dtype="float64")
    return _extract_close(raw, ticker)


def _extract_close(raw: pd.DataFrame, ticker: str) -> pd.Series:
    """Pull the Close column out of a yfinance frame, single- or multi-indexed."""
    if isinstance(raw.columns, pd.MultiIndex):
        # yfinance returns (Price, Ticker) columns; take Close for this ticker.
        close = raw.xs("Close", axis=1, level=0)
        close = close[ticker] if ticker in close.columns else close.iloc[:, 0]
    else:
        if "Close" not in raw.columns:
            raise DataError(f"yfinance returned no Close column for {ticker}.")
        close = raw["Close"]
    return close


def _normalise_series(series: pd.Series, ticker: str) -> pd.Series:
    """Coerce a downloaded series into a clean, date-indexed float Series."""
    if isinstance(series, pd.DataFrame):
        series = series.iloc[:, 0]
    series = pd.Series(series).astype("float64")
    series.index = pd.to_datetime(series.index).tz_localize(None).normalize()
    series = series[~series.index.duplicated(keep="last")].sort_index()
    series.name = ticker
    return series


def _check_coverage(panel: pd.DataFrame, start: dt.date) -> None:
    """Fail loudly when a ticker's history begins after the requested start."""
    cutoff = pd.Timestamp(start) + pd.Timedelta(days=COVERAGE_SLACK_DAYS)
    late = {}
    for ticker in panel.columns:
        first_valid = panel[ticker].first_valid_index()
        if first_valid is None:
            late[ticker] = "no data at all"
        elif first_valid > cutoff:
            late[ticker] = f"first price {first_valid.date()}"

    if late:
        detail = "; ".join(f"{t} ({why})" for t, why in sorted(late.items()))
        raise DataError(
            f"These tickers have no data at the configured start of {start}: {detail}. "
            f"Move `start` later or remove the ticker -- the window is not shortened "
            f"silently, because that would quietly change what the backtest measures."
        )


def _cache_path(ticker: str, start: dt.date, end: dt.date, cache_dir: Path) -> Path:
    safe = re.sub(r"[^A-Za-z0-9]+", "_", ticker).strip("_") or "ticker"
    return cache_dir / f"{safe}_{start:%Y%m%d}_{end:%Y%m%d}.parquet"


def _load_cached(ticker: str, start: dt.date, end: dt.date, cache_dir: Path) -> pd.Series | None:
    path = _cache_path(ticker, start, end, cache_dir)
    if not path.is_file():
        return None
    try:
        frame = pd.read_parquet(path)
    except Exception as exc:  # a truncated or unreadable cache should not be fatal
        print(f"[data] Ignoring unreadable cache file {path.name}: {exc}")
        return None
    return _normalise_series(frame.iloc[:, 0], ticker)


def _write_cache(series: pd.Series, ticker: str, start: dt.date, end: dt.date, cache_dir: Path) -> None:
    path = _cache_path(ticker, start, end, cache_dir)
    series.to_frame(name="close").to_parquet(path)
