"""Unit tests for the published statistics.

The validator must itself be provably correct, so several tests anchor on
hand-computed numbers and exact analytic special cases, not just self-consistency.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.stats import norm

from lyravalidate.stats import (
    EULER_MASCHERONI,
    annualized_sharpe,
    deflated_sharpe_ratio,
    expected_max_sharpe_benchmark,
    hit_rate,
    information_coefficient,
    max_drawdown,
    minimum_backtest_length,
    minimum_track_record_length,
    probabilistic_sharpe_ratio,
    probability_of_backtest_overfitting,
    rank_information_coefficient,
    sharpe_ratio,
    sharpe_standard_error,
)

# ── Sharpe / drawdown helpers ─────────────────────────────────────────────────


def test_sharpe_ratio_hand_value() -> None:
    # returns 0.01..0.04: mean 0.025, std(ddof=1)=0.0129099, SR=1.9364917
    r = np.array([0.01, 0.02, 0.03, 0.04])
    assert sharpe_ratio(r) == pytest.approx(0.025 / 0.012909944487, rel=1e-9)


def test_annualized_sharpe_scales_by_sqrt_periods() -> None:
    r = np.array([0.01, 0.02, 0.03, 0.04])
    assert annualized_sharpe(r, 252) == pytest.approx(sharpe_ratio(r) * math.sqrt(252))


def test_sharpe_degenerate_is_zero() -> None:
    assert sharpe_ratio([1.0]) == 0.0
    assert sharpe_ratio([2.0, 2.0, 2.0]) == 0.0  # zero variance
    assert annualized_sharpe([]) == 0.0


def test_annualized_sharpe_rejects_bad_periods() -> None:
    with pytest.raises(ValueError):
        annualized_sharpe([0.01, 0.02, 0.03], 0)


def test_max_drawdown_known() -> None:
    # +10% then -50% -> equity 1.1 then 0.55; peak 1.1; drawdown = 1 - 0.55/1.1 = 0.5
    assert max_drawdown([0.10, -0.50]) == pytest.approx(0.5)


def test_max_drawdown_monotonic_up_is_zero() -> None:
    assert max_drawdown([0.01, 0.02, 0.03]) == pytest.approx(0.0)


def test_hit_rate() -> None:
    assert hit_rate([1.0, -1.0, 2.0, -3.0]) == pytest.approx(0.5)
    assert hit_rate([]) == 0.0


# ── information coefficients ──────────────────────────────────────────────────


def test_information_coefficient_perfect() -> None:
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert information_coefficient(x, 2.0 * x + 1.0) == pytest.approx(1.0)
    assert information_coefficient(x, -x) == pytest.approx(-1.0)


def test_information_coefficient_constant_is_zero() -> None:
    assert information_coefficient([1.0, 2.0, 3.0], [5.0, 5.0, 5.0]) == 0.0


def test_rank_ic_monotonic_nonlinear() -> None:
    # Spearman is 1 for any strictly increasing (even nonlinear) relationship.
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert rank_information_coefficient(x, x**3) == pytest.approx(1.0)


def test_ic_length_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        information_coefficient([1.0, 2.0], [1.0])


# ── PSR ───────────────────────────────────────────────────────────────────────


def test_psr_symmetric_zero_mean_is_half() -> None:
    # mean exactly 0 -> SR=0 -> PSR(0)=Phi(0)=0.5 exactly.
    r = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0])
    assert probabilistic_sharpe_ratio(r, 0.0) == pytest.approx(0.5, abs=1e-12)


def test_psr_hand_computed_value() -> None:
    # returns 0.01..0.04: SR=1.9364917, skew=0, non-excess kurtosis=1.64,
    # sr_var=(1+0.16*3.75)/3=0.5333333, sigma=0.7302967, PSR=Phi(2.651650)=0.996000
    r = np.array([0.01, 0.02, 0.03, 0.04])
    assert probabilistic_sharpe_ratio(r, 0.0) == pytest.approx(0.996000, abs=1e-4)


def test_psr_decreasing_in_benchmark() -> None:
    r = np.array([0.01, 0.02, 0.03, 0.04])
    assert probabilistic_sharpe_ratio(r, 0.0) > probabilistic_sharpe_ratio(r, 1.0)


def test_psr_fail_closed() -> None:
    assert probabilistic_sharpe_ratio([0.01, 0.02, 0.03]) == 0.0  # T < 4
    assert probabilistic_sharpe_ratio([0.5, 0.5, 0.5, 0.5]) == 0.0  # zero variance


def test_psr_strips_non_finite() -> None:
    finite = np.array([0.01, 0.02, 0.03, 0.04])
    with_nan = np.array([0.01, np.nan, 0.02, np.inf, 0.03, 0.04])
    assert probabilistic_sharpe_ratio(with_nan) == pytest.approx(
        probabilistic_sharpe_ratio(finite)
    )


# ── Sharpe standard error ─────────────────────────────────────────────────────


def test_sharpe_standard_error_hand_value() -> None:
    # returns 0.01..0.04: SR=1.9364917, skew=0, non-excess kurtosis=1.64,
    # sr_var=(1+0.16*3.75)/3=0.5333333 -> sigma=0.7302967 (same numbers as the
    # hand-computed PSR test above).
    r = np.array([0.01, 0.02, 0.03, 0.04])
    assert sharpe_standard_error(r) == pytest.approx(math.sqrt(1.6 / 3.0), rel=1e-9)


def test_sharpe_standard_error_degenerate_is_infinite() -> None:
    # No information -> infinite uncertainty (the fail-closed direction here:
    # 0.0 would claim perfect certainty and make any Sharpe look significant).
    assert math.isinf(sharpe_standard_error([0.01, 0.02, 0.03]))  # T < 4
    assert math.isinf(sharpe_standard_error([0.5, 0.5, 0.5, 0.5]))  # zero variance


def test_sharpe_standard_error_shrinks_with_length() -> None:
    rng = np.random.default_rng(15)
    r = 0.001 + 0.01 * rng.standard_normal(2000)
    assert sharpe_standard_error(r) < sharpe_standard_error(r[:200])


# ── MinTRL ────────────────────────────────────────────────────────────────────


def test_min_trl_hand_computed_value() -> None:
    # returns 0.01..0.04: SR=1.9364917, skew=0, non-excess kurtosis=1.64, so the
    # variance numerator is 1+0.16*3.75=1.6 (as in the PSR test above) and
    # MinTRL = 1 + 1.6*(z_0.95/SR)^2 = 1 + 1.6*(1.6448536/1.9364917)^2 = 2.154365.
    r = np.array([0.01, 0.02, 0.03, 0.04])
    SR = 0.025 / 0.012909944487
    z = float(norm.ppf(0.95))
    expected = 1.0 + 1.6 * (z / SR) ** 2
    assert expected == pytest.approx(2.154365, abs=1e-6)  # anchor the arithmetic itself
    assert minimum_track_record_length(r, 0.0, confidence=0.95) == pytest.approx(
        expected, rel=1e-9
    )


def test_min_trl_hand_computed_value_nonzero_benchmark() -> None:
    # Same moments, benchmark SR*=1.0: MinTRL = 1 + 1.6*(z_0.95/(SR-1))^2 = 5.935903.
    r = np.array([0.01, 0.02, 0.03, 0.04])
    SR = 0.025 / 0.012909944487
    z = float(norm.ppf(0.95))
    expected = 1.0 + 1.6 * (z / (SR - 1.0)) ** 2
    assert expected == pytest.approx(5.935903, abs=1e-6)
    assert minimum_track_record_length(r, 1.0, confidence=0.95) == pytest.approx(
        expected, rel=1e-9
    )


def test_min_trl_inverts_psr_exactly() -> None:
    # Setting the confidence to the *observed* PSR must return the observed T:
    # MinTRL is the exact algebraic inverse of the PSR in the sample length.
    rng = np.random.default_rng(21)
    r = 0.0005 + 0.01 * rng.standard_normal(750)
    conf = probabilistic_sharpe_ratio(r, 0.0)
    assert minimum_track_record_length(r, 0.0, confidence=conf) == pytest.approx(
        750.0, rel=1e-9
    )


def test_min_trl_increases_with_confidence_and_benchmark() -> None:
    rng = np.random.default_rng(33)
    r = 0.001 + 0.01 * rng.standard_normal(500)
    assert minimum_track_record_length(r, 0.0, confidence=0.99) > minimum_track_record_length(
        r, 0.0, confidence=0.95
    )
    assert minimum_track_record_length(r, 0.05, confidence=0.95) > minimum_track_record_length(
        r, 0.0, confidence=0.95
    )


def test_min_trl_unreachable_benchmark_is_infinite() -> None:
    # SR exactly 0 vs benchmark 0: no track record length can ever reach the bar.
    r = np.array([0.01, -0.01, 0.01, -0.01, 0.01, -0.01])
    assert math.isinf(minimum_track_record_length(r, 0.0))
    assert math.isinf(minimum_track_record_length(r, 1.0))


def test_min_trl_degenerate_is_infinite() -> None:
    assert math.isinf(minimum_track_record_length([0.01, 0.02, 0.03]))  # T < 4
    assert math.isinf(minimum_track_record_length([0.5, 0.5, 0.5, 0.5]))  # zero variance


def test_min_trl_rejects_bad_confidence() -> None:
    r = np.array([0.01, 0.02, 0.03, 0.04])
    for bad in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValueError):
            minimum_track_record_length(r, 0.0, confidence=bad)


# ── MinBTL ────────────────────────────────────────────────────────────────────


def test_min_btl_hand_computed_value() -> None:
    # N=10 trials, observed annualised Sharpe 1.0:
    # factor = (1-gamma)*Z^-1(0.9) + gamma*Z^-1(1-1/(10e)) = 1.574598,
    # MinBTL = (factor/1.0)^2 = 2.479360 years.
    n = 10
    z1 = float(norm.ppf(1.0 - 1.0 / n))
    z2 = float(norm.ppf(1.0 - 1.0 / (n * math.e)))
    factor = (1.0 - EULER_MASCHERONI) * z1 + EULER_MASCHERONI * z2
    expected = factor**2
    assert expected == pytest.approx(2.479360, abs=1e-6)
    assert minimum_backtest_length(1.0, n) == pytest.approx(expected, rel=1e-9)


def test_min_btl_scales_inversely_with_sharpe_squared() -> None:
    # Halving the target Sharpe quadruples the years of backtest needed.
    assert minimum_backtest_length(0.5, 10) == pytest.approx(
        4.0 * minimum_backtest_length(1.0, 10), rel=1e-12
    )


def test_min_btl_increases_with_trials() -> None:
    assert (
        minimum_backtest_length(1.0, 2)
        < minimum_backtest_length(1.0, 10)
        < minimum_backtest_length(1.0, 1000)
    )


def test_min_btl_single_trial_is_zero() -> None:
    # No selection took place -> no multiple-testing minimum applies.
    assert minimum_backtest_length(1.0, 1) == 0.0
    assert minimum_backtest_length(1.0, 0) == 0.0


def test_min_btl_non_positive_sharpe_is_infinite() -> None:
    # A non-positive Sharpe can never exceed the expected best of >= 2 noise trials.
    assert math.isinf(minimum_backtest_length(0.0, 10))
    assert math.isinf(minimum_backtest_length(-1.0, 10))


# ── expected-max-Sharpe benchmark ─────────────────────────────────────────────


def test_euler_mascheroni_constant() -> None:
    assert math.isclose(EULER_MASCHERONI, 0.5772156649015329)


def test_expected_max_benchmark_one_trial_is_zero() -> None:
    assert expected_max_sharpe_benchmark(1.0, 1) == 0.0
    assert expected_max_sharpe_benchmark(1.0, 0) == 0.0


def test_expected_max_benchmark_increases_with_trials() -> None:
    b2 = expected_max_sharpe_benchmark(1.0, 2)
    b10 = expected_max_sharpe_benchmark(1.0, 10)
    b100 = expected_max_sharpe_benchmark(1.0, 100)
    assert 0.0 < b2 < b10 < b100


def test_expected_max_benchmark_scales_with_sigma() -> None:
    assert expected_max_sharpe_benchmark(2.0, 50) == pytest.approx(
        2.0 * expected_max_sharpe_benchmark(1.0, 50)
    )


def test_expected_max_benchmark_matches_formula() -> None:
    n = 25
    z1 = norm.ppf(1.0 - 1.0 / n)
    z2 = norm.ppf(1.0 - 1.0 / (n * math.e))
    expected = (1.0 - EULER_MASCHERONI) * z1 + EULER_MASCHERONI * z2
    assert expected_max_sharpe_benchmark(1.0, n) == pytest.approx(expected)


# ── DSR ───────────────────────────────────────────────────────────────────────


def test_dsr_one_trial_equals_psr_zero() -> None:
    rng = np.random.default_rng(7)
    r = 0.001 + 0.01 * rng.standard_normal(500)
    assert deflated_sharpe_ratio(r, n_trials=1) == pytest.approx(
        probabilistic_sharpe_ratio(r, 0.0)
    )


def test_dsr_deflates_with_more_trials() -> None:
    rng = np.random.default_rng(11)
    r = 0.001 + 0.01 * rng.standard_normal(1000)
    assert (
        deflated_sharpe_ratio(r, 1)
        > deflated_sharpe_ratio(r, 50)
        > deflated_sharpe_ratio(r, 5000)
    )


def test_dsr_explicit_benchmark_overrides_trials() -> None:
    rng = np.random.default_rng(3)
    r = 0.001 + 0.01 * rng.standard_normal(400)
    # An explicit SR* must be used verbatim regardless of n_trials.
    a = deflated_sharpe_ratio(r, n_trials=999, sr_benchmark=0.0)
    b = probabilistic_sharpe_ratio(r, 0.0)
    assert a == pytest.approx(b)


def test_dsr_fail_closed() -> None:
    assert deflated_sharpe_ratio([0.01, 0.02, 0.03], 10) == 0.0
    assert deflated_sharpe_ratio([1.0, 1.0, 1.0, 1.0], 10) == 0.0


# ── PBO via CSCV ──────────────────────────────────────────────────────────────


def test_pbo_pure_noise_is_elevated() -> None:
    # Pure noise: the IS-best config has no reason to persist OOS, so PBO is far
    # from zero (here ~0.33). Contrast with the dominant-config case below (~0).
    rng = np.random.default_rng(2024)
    perf = rng.standard_normal((480, 40)) * 0.01
    pbo = probability_of_backtest_overfitting(perf, n_splits=16)
    assert 0.2 < pbo < 0.8


def test_pbo_dominant_config_is_low() -> None:
    # One config dominates every period -> IS-best is always OOS-best -> PBO ~ 0.
    rng = np.random.default_rng(5)
    perf = rng.standard_normal((400, 6)) * 0.01
    perf[:, 0] += 0.05
    assert probability_of_backtest_overfitting(perf, n_splits=12) < 0.05


def test_pbo_identical_columns_is_zero() -> None:
    rng = np.random.default_rng(9)
    col = rng.standard_normal(120)
    perf = np.column_stack([col, col, col, col])
    assert probability_of_backtest_overfitting(perf, n_splits=8) == 0.0


def test_pbo_degenerate_default_optimistic() -> None:
    # Documented hazard: degenerate input returns the optimistic 0.0 by default.
    assert probability_of_backtest_overfitting(np.ones((10, 1))) == 0.0  # N < 2
    assert probability_of_backtest_overfitting(np.ones(10)) == 0.0  # not 2-D
    assert probability_of_backtest_overfitting(np.ones((3, 5))) == 0.0  # T < 4


def test_pbo_degenerate_can_fail_closed() -> None:
    # Gate path: degenerate input is rejected, not waved through.
    assert (
        probability_of_backtest_overfitting(np.ones((10, 1)), degenerate_value=1.0) == 1.0
    )


def test_pbo_in_unit_interval() -> None:
    rng = np.random.default_rng(1)
    perf = rng.standard_normal((300, 10))
    pbo = probability_of_backtest_overfitting(perf)
    assert 0.0 <= pbo <= 1.0


def test_pbo_max_blocks_caps_cost() -> None:
    # A huge n_splits must not explode: max_blocks caps S (and the run completes).
    rng = np.random.default_rng(0)
    perf = rng.standard_normal((200, 4))
    pbo = probability_of_backtest_overfitting(perf, n_splits=1000, max_blocks=10)
    assert 0.0 <= pbo <= 1.0
