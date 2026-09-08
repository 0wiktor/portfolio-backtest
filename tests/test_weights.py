"""Weighting tests, including the look-ahead guarantee.

The look-ahead test is the important one: an inverse-volatility scheme that
peeks at future data produces a backtest that cannot be traded, and the failure
is invisible in the equity curve.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pbt.weights import (
    WeightError,
    custom_weights,
    equal_weights,
    inverse_vol_weights,
    resolve_weights,
)


def returns_frame(columns: dict[str, list[float]]) -> pd.DataFrame:
    length = len(next(iter(columns.values())))
    return pd.DataFrame(columns, index=pd.bdate_range("2020-01-01", periods=length))


def test_equal_weights_are_one_over_n():
    weights = equal_weights(["A", "B", "C", "D"])
    assert list(weights) == [0.25] * 4
    assert weights.sum() == pytest.approx(1.0)


def test_custom_weights_are_normalised():
    # Deliberately sums to 0.8; the config validator rejects that, but the
    # function itself must still return a properly scaled set.
    weights = custom_weights(["A", "B"], [0.6, 0.2])
    assert weights["A"] == pytest.approx(0.75)
    assert weights["B"] == pytest.approx(0.25)
    assert weights.sum() == pytest.approx(1.0)


def test_custom_weights_reject_a_length_mismatch():
    with pytest.raises(WeightError, match="do not align"):
        custom_weights(["A", "B", "C"], [0.5, 0.5])


def test_inverse_vol_gives_double_weight_to_half_the_volatility():
    # A alternates +/-1%, B alternates +/-2%, so B's stdev is exactly twice A's.
    # Inverse-vol weights are then 2/3 to A and 1/3 to B.
    n = 40
    frame = returns_frame(
        {
            "A": [0.01 if i % 2 == 0 else -0.01 for i in range(n)],
            "B": [0.02 if i % 2 == 0 else -0.02 for i in range(n)],
        }
    )
    weights = inverse_vol_weights(frame, as_of=frame.index[-1], lookback_days=20)

    assert weights["A"] == pytest.approx(2 / 3, abs=1e-12)
    assert weights["B"] == pytest.approx(1 / 3, abs=1e-12)
    assert weights.sum() == pytest.approx(1.0)


def test_inverse_vol_weights_ignore_everything_after_the_as_of_date():
    """The look-ahead guarantee: future data must not move today's weights."""
    rng = np.random.default_rng(42)
    frame = returns_frame(
        {"A": list(rng.normal(0, 0.01, 200)), "B": list(rng.normal(0, 0.02, 200))}
    )
    as_of = frame.index[100]

    before = inverse_vol_weights(frame, as_of=as_of, lookback_days=60)

    # Corrupt every observation after the rebalance date, violently: flip the
    # volatility ranking of the two assets. If any future data leaked into the
    # estimate, these weights would move.
    tampered = frame.copy()
    tampered.loc[tampered.index > as_of, "A"] *= 50.0
    tampered.loc[tampered.index > as_of, "B"] *= 0.01

    after = inverse_vol_weights(tampered, as_of=as_of, lookback_days=60)

    pd.testing.assert_series_equal(before, after)


def test_inverse_vol_window_ends_on_the_rebalance_date_inclusive():
    # Changing the return ON the rebalance date SHOULD change the weights --
    # that observation is already known when the trade is placed.
    frame = returns_frame({"A": [0.01, -0.01] * 15, "B": [0.02, -0.02] * 15})
    as_of = frame.index[20]

    baseline = inverse_vol_weights(frame, as_of=as_of, lookback_days=10)
    changed = frame.copy()
    changed.loc[as_of, "A"] = 0.30
    updated = inverse_vol_weights(changed, as_of=as_of, lookback_days=10)

    assert updated["A"] < baseline["A"]


def test_inverse_vol_excludes_a_stale_zero_volatility_series():
    frame = returns_frame({"A": [0.01, -0.01] * 10, "FLAT": [0.0] * 20})
    weights = inverse_vol_weights(frame, as_of=frame.index[-1], lookback_days=10)

    assert weights["FLAT"] == 0.0
    assert weights["A"] == pytest.approx(1.0)


def test_inverse_vol_needs_enough_history():
    frame = returns_frame({"A": [0.01, -0.01, 0.02]})
    with pytest.raises(WeightError, match="at least 2"):
        inverse_vol_weights(frame, as_of=frame.index[0], lookback_days=10)


def test_resolve_weights_dispatches_to_each_scheme():
    frame = returns_frame({"A": [0.01, -0.01] * 10, "B": [0.02, -0.02] * 10})
    tickers = ("A", "B")
    as_of = frame.index[-1]

    equal = resolve_weights("equal", tickers, frame, as_of)
    assert equal["A"] == pytest.approx(0.5)

    custom = resolve_weights("custom", tickers, frame, as_of, custom=(0.7, 0.3))
    assert custom["A"] == pytest.approx(0.7)

    inverse = resolve_weights("inverse_vol", tickers, frame, as_of, lookback_days=10)
    assert inverse["A"] == pytest.approx(2 / 3, abs=1e-12)

    with pytest.raises(WeightError, match="Unknown weighting scheme"):
        resolve_weights("market_cap", tickers, frame, as_of)
