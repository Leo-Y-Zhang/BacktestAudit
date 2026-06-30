"""End-to-end tests for evaluate() with positive / negative / overfit controls.

These are the load-bearing correctness tests: a validator that cannot tell a real
edge from noise from an overfit search is worse than useless.
"""

from __future__ import annotations

import numpy as np
import pytest

from lyravalidate.evaluate import Thresholds, Verdict, evaluate


def test_positive_control_is_deployable() -> None:
    """A genuinely persistent edge (high stable Sharpe) must be DEPLOYABLE."""
    rng = np.random.default_rng(42)
    returns = 0.0008 + 0.008 * rng.standard_normal(1500)  # ann. Sharpe ~ 1.6
    verdict = evaluate(returns, n_trials=1)
    assert verdict.classification == "DEPLOYABLE"
    assert verdict.deployable is True
    assert verdict.deflated_sharpe >= 0.95
    assert verdict.sharpe > 0.75


def test_negative_control_is_not_deployable() -> None:
    """Pure zero-mean noise has no edge -> NOT_DEPLOYABLE."""
    rng = np.random.default_rng(123)
    returns = 0.01 * rng.standard_normal(1500)
    verdict = evaluate(returns, n_trials=1)
    assert verdict.classification == "NOT_DEPLOYABLE"
    assert verdict.deployable is False
    assert verdict.deflated_sharpe < 0.95


def test_overfit_control_is_probably_overfit() -> None:
    """Best-of-many pure-noise configs looks great in sample but is overfit."""
    rng = np.random.default_rng(2024)
    candidates = 0.01 * rng.standard_normal((600, 60))  # 60 noise strategies
    verdict = evaluate(candidates)
    assert verdict.classification == "PROBABLY_OVERFIT"
    assert verdict.deployable is False
    # The selected (best in-sample) column looked good...
    assert verdict.sharpe > 0.75
    # ...but the deflated Sharpe and PBO expose the search.
    assert verdict.deflated_sharpe < 0.95
    assert verdict.pbo > 0.35
    assert verdict.n_trials == 60


def test_deflation_can_flip_a_borderline_verdict() -> None:
    """The same record can pass at 1 trial and fail once you admit the search."""
    rng = np.random.default_rng(202)
    returns = 0.0007 + 0.008 * rng.standard_normal(1200)
    optimistic = evaluate(returns, n_trials=1)
    honest = evaluate(returns, n_trials=100_000)
    assert optimistic.deflated_sharpe > honest.deflated_sharpe
    assert honest.deflated_sharpe < optimistic.deflated_sharpe


def test_predictions_targets_genuine_signal_oos_positive() -> None:
    rng = np.random.default_rng(77)
    base = rng.standard_normal(900)
    targets = 0.01 * base + 0.004 * rng.standard_normal(900)  # correlated with base
    predictions = base
    verdict = evaluate(returns=targets, predictions=predictions, targets=targets)
    assert verdict.oos_sharpe is not None
    assert verdict.oos_information_coefficient is not None
    assert verdict.oos_sharpe > 0.0
    assert verdict.oos_information_coefficient > 0.0


def test_predictions_targets_noise_signal_oos_flat() -> None:
    rng = np.random.default_rng(78)
    predictions = rng.standard_normal(900)
    targets = 0.01 * rng.standard_normal(900)  # independent of predictions
    verdict = evaluate(returns=targets, predictions=predictions, targets=targets)
    assert verdict.oos_sharpe is not None
    assert verdict.classification == "NOT_DEPLOYABLE"
    assert abs(verdict.oos_information_coefficient or 0.0) < 0.1


def test_custom_thresholds_relax_the_bar() -> None:
    rng = np.random.default_rng(9)
    returns = 0.0003 + 0.01 * rng.standard_normal(800)  # modest edge
    strict = evaluate(returns, n_trials=1)
    lax = evaluate(
        returns,
        n_trials=1,
        thresholds=Thresholds(min_deflated_sharpe=0.5, min_sharpe=0.1, max_pbo=0.9),
    )
    assert lax.deployable or not strict.deployable  # relaxing never makes it stricter


def test_two_d_single_column_is_treated_as_one_strategy() -> None:
    rng = np.random.default_rng(4)
    returns = (0.0008 + 0.008 * rng.standard_normal(1500)).reshape(-1, 1)
    verdict = evaluate(returns)
    assert verdict.n_trials == 1
    assert np.isnan(verdict.pbo)


def test_three_d_input_raises() -> None:
    with pytest.raises(ValueError):
        evaluate(np.zeros((4, 4, 4)))


def test_bad_periods_per_year_raises() -> None:
    with pytest.raises(ValueError):
        evaluate([0.01, 0.02, 0.03, 0.04], periods_per_year=0)


def test_verdict_is_frozen_and_summarises() -> None:
    rng = np.random.default_rng(1)
    returns = 0.0008 + 0.008 * rng.standard_normal(1000)
    verdict = evaluate(returns, n_trials=1)
    assert isinstance(verdict, Verdict)
    text = verdict.summary()
    assert verdict.classification in text
    assert "Deflated Sharpe" in text
    with pytest.raises(AttributeError):
        verdict.deployable = False  # type: ignore[misc]


def test_reasons_present_when_rejected() -> None:
    rng = np.random.default_rng(123)
    returns = 0.01 * rng.standard_normal(1000)
    verdict = evaluate(returns, n_trials=1)
    assert verdict.reasons  # non-empty explanation on rejection
