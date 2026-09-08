"""Load and validate the YAML run configuration.

Every run parameter lives in the config file rather than in code, so a result
can be reproduced from a single artefact. Validation is strict and happens once,
at load time: an unknown or malformed key should produce one clear sentence the
author can act on, never a traceback from three modules deeper in the pipeline.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """The config file is missing, malformed, or internally inconsistent."""


WEIGHTING_SCHEMES = ("equal", "custom", "inverse_vol")
REBALANCE_FREQUENCIES = ("none", "monthly", "quarterly", "annual")

REQUIRED_KEYS = (
    "tickers",
    "start",
    "end",
    "initial_capital",
    "weighting",
    "rebalance",
    "risk_free_rate",
    "transaction_cost_bps",
)
# Optional at the top level, but two of them become mandatory once a particular
# weighting scheme is chosen -- see _validate_scheme_requirements.
OPTIONAL_KEYS = ("benchmark", "custom_weights", "lookback_days")
VALID_KEYS = REQUIRED_KEYS + OPTIONAL_KEYS

DEFAULT_BENCHMARK = "SPY"

# Custom weights are author-typed decimals, so they will rarely sum to exactly
# 1.0 in floating point. This tolerance accepts rounding, not sloppiness.
WEIGHT_SUM_TOLERANCE = 1e-6


@dataclass(frozen=True)
class Config:
    """A validated backtest specification.

    Attributes carry the units that are easy to get wrong:
      initial_capital      -- currency units, same currency as the price data
      risk_free_rate       -- ANNUAL decimal (0.02 is 2% a year), not percent
      transaction_cost_bps -- basis points charged on turnover (5 is 0.05%)
      lookback_days        -- trading days, not calendar days
    """

    tickers: tuple[str, ...]
    benchmark: str
    start: dt.date
    end: dt.date
    initial_capital: float
    weighting: str
    rebalance: str
    risk_free_rate: float
    transaction_cost_bps: float
    custom_weights: tuple[float, ...] | None = None
    lookback_days: int | None = None


def load_config(path: str | Path) -> Config:
    """Read a YAML config file and return a validated Config.

    Raises ConfigError for anything wrong with the file, including its absence.
    """
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"Config file not found: {path}")

    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"Config file is not valid YAML: {path}\n{exc}") from exc

    if not isinstance(raw, dict):
        raise ConfigError(f"Config file must contain a top-level mapping of keys: {path}")

    _validate_keys(raw)
    return _build_config(raw)


def _validate_keys(raw: dict[str, Any]) -> None:
    """Reject unknown keys and report missing required ones, all at once."""
    unknown = sorted(set(raw) - set(VALID_KEYS))
    if unknown:
        raise ConfigError(
            f"Unknown config key(s): {', '.join(unknown)}. "
            f"Valid keys are: {', '.join(sorted(VALID_KEYS))}."
        )

    missing = [key for key in REQUIRED_KEYS if key not in raw]
    if missing:
        raise ConfigError(
            f"Missing required config key(s): {', '.join(missing)}. "
            f"See config/example.yaml for a complete file."
        )


def _build_config(raw: dict[str, Any]) -> Config:
    tickers = _as_ticker_tuple(raw["tickers"])
    start = _as_date(raw["start"], "start")
    end = _as_date(raw["end"], "end")
    if start >= end:
        raise ConfigError(f"start ({start}) must be earlier than end ({end}).")

    weighting = _as_choice(raw["weighting"], "weighting", WEIGHTING_SCHEMES)
    rebalance = _as_choice(raw["rebalance"], "rebalance", REBALANCE_FREQUENCIES)

    custom_weights, lookback_days = _validate_scheme_requirements(raw, weighting, tickers)

    return Config(
        tickers=tickers,
        benchmark=str(raw.get("benchmark") or DEFAULT_BENCHMARK).strip().upper(),
        start=start,
        end=end,
        initial_capital=_as_positive_number(raw["initial_capital"], "initial_capital"),
        weighting=weighting,
        rebalance=rebalance,
        risk_free_rate=_as_number(raw["risk_free_rate"], "risk_free_rate"),
        transaction_cost_bps=_as_non_negative_number(
            raw["transaction_cost_bps"], "transaction_cost_bps"
        ),
        custom_weights=custom_weights,
        lookback_days=lookback_days,
    )


def _validate_scheme_requirements(
    raw: dict[str, Any], weighting: str, tickers: tuple[str, ...]
) -> tuple[tuple[float, ...] | None, int | None]:
    """Enforce the keys that only matter for the chosen weighting scheme.

    Weights are validated as supplied rather than silently normalised: if the
    author intended 1.0 and typed 0.95, that is a mistake worth surfacing.
    """
    custom_weights: tuple[float, ...] | None = None
    lookback_days: int | None = None

    if weighting == "custom":
        if "custom_weights" not in raw:
            raise ConfigError("weighting is 'custom', so custom_weights is required.")
        values = raw["custom_weights"]
        if not isinstance(values, (list, tuple)) or not values:
            raise ConfigError("custom_weights must be a non-empty list of numbers.")
        custom_weights = tuple(
            _as_number(v, f"custom_weights[{i}]") for i, v in enumerate(values)
        )
        if len(custom_weights) != len(tickers):
            raise ConfigError(
                f"custom_weights has {len(custom_weights)} entries but there are "
                f"{len(tickers)} tickers; they must align one-to-one."
            )
        if any(w < 0 for w in custom_weights):
            raise ConfigError("custom_weights must not be negative; shorting is not modelled.")
        total = sum(custom_weights)
        if abs(total - 1.0) > WEIGHT_SUM_TOLERANCE:
            raise ConfigError(f"custom_weights must sum to 1.0, but they sum to {total:.6f}.")

    if weighting == "inverse_vol":
        if "lookback_days" not in raw:
            raise ConfigError("weighting is 'inverse_vol', so lookback_days is required.")
        lookback_days = _as_positive_int(raw["lookback_days"], "lookback_days")
        if lookback_days < 2:
            raise ConfigError("lookback_days must be at least 2 to estimate a volatility.")

    return custom_weights, lookback_days


def _as_ticker_tuple(value: Any) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ConfigError("tickers must be a non-empty list of ticker symbols.")
    tickers = tuple(str(t).strip().upper() for t in value)
    if any(not t for t in tickers):
        raise ConfigError("tickers must not contain blank entries.")
    duplicates = sorted({t for t in tickers if tickers.count(t) > 1})
    if duplicates:
        raise ConfigError(f"tickers contains duplicate symbol(s): {', '.join(duplicates)}.")
    return tickers


def _as_date(value: Any, key: str) -> dt.date:
    """Accept a YAML-native date or an ISO string; reject anything else."""
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    try:
        return dt.date.fromisoformat(str(value).strip())
    except ValueError as exc:
        raise ConfigError(f"{key} must be a date in YYYY-MM-DD form, got: {value!r}") from exc


def _as_choice(value: Any, key: str, allowed: tuple[str, ...]) -> str:
    text = str(value).strip().lower()
    if text not in allowed:
        raise ConfigError(f"{key} must be one of {', '.join(allowed)}, got: {value!r}")
    return text


def _as_number(value: Any, key: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{key} must be a number, got: {value!r}")
    return float(value)


def _as_non_negative_number(value: Any, key: str) -> float:
    number = _as_number(value, key)
    if number < 0:
        raise ConfigError(f"{key} must not be negative, got: {number}")
    return number


def _as_positive_number(value: Any, key: str) -> float:
    number = _as_number(value, key)
    if number <= 0:
        raise ConfigError(f"{key} must be greater than zero, got: {number}")
    return number


def _as_positive_int(value: Any, key: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{key} must be a whole number of trading days, got: {value!r}")
    if value <= 0:
        raise ConfigError(f"{key} must be greater than zero, got: {value}")
    return value
