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
    cluster_trials,
    cross_trial_sharpe_std,
    deflated_sharpe_ratio,
    deflated_sharpe_ratio_from_trials,
    effective_trials,
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


def test_near_constant_returns_fail_closed_not_nan() -> None:
    # [0.01]*10 is constant, but 0.01 is not exactly representable, so the
    # sample std is a tiny positive number and scipy skew/kurtosis go NaN under
    # catastrophic cancellation. The NaN must not leak through the degenerate
    # guard (NaN compares False against <= 0.0): everything fails closed.
    r = [0.01] * 10
    assert probabilistic_sharpe_ratio(r, 0.0) == 0.0
    assert deflated_sharpe_ratio(r, 10) == 0.0
    assert math.isinf(sharpe_standard_error(r))
    assert math.isinf(minimum_track_record_length(r, 0.0))


def test_min_trl_iff_with_psr_holds_only_above_half_confidence() -> None:
    # The PSR equivalence is exact for confidence > 0.5 only: at SR == SR* the
    # PSR is exactly 0.5 for every T (so it meets any bar <= 0.5), yet MinTRL
    # fails closed to inf because no length ever *exceeds* the benchmark.
    r = np.array([0.01, -0.01, 0.01, -0.01, 0.01, -0.01])  # SR exactly 0
    assert probabilistic_sharpe_ratio(r, 0.0) == pytest.approx(0.5)
    assert math.isinf(minimum_track_record_length(r, 0.0, confidence=0.3))


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


# ── effective trials via correlation clustering (Lopez de Prado & Lewis 2019) ─


def _planted_trials(
    rho: float, groups: int = 3, per_group: int = 4, T: int = 400, seed: int = 17
) -> np.ndarray:
    """``groups`` blocks of ``per_group`` trials sharing a common factor.

    Within a block the pairwise correlation is ~``rho``; across blocks it is ~0,
    so the true number of effectively independent trials is ``groups``.
    """
    rng = np.random.default_rng(seed)
    w = math.sqrt(rho)
    cols = []
    for _ in range(groups):
        base = rng.standard_normal(T)
        for _ in range(per_group):
            cols.append(0.01 * (w * base + math.sqrt(1.0 - rho) * rng.standard_normal(T)))
    return np.column_stack(cols)


def test_cluster_trials_recovers_planted_groups() -> None:
    clusters = cluster_trials(_planted_trials(0.7))
    assert sorted(sorted(c) for c in clusters) == [
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [8, 9, 10, 11],
    ]


def test_effective_trials_iid_noise_is_column_count() -> None:
    # Independent trials show no correlation structure, so every configuration
    # really is its own trial: no clustering may be invented for them.
    rng = np.random.default_rng(2024)
    perf = 0.01 * rng.standard_normal((600, 20))
    assert effective_trials(perf) == 20


def test_effective_trials_planted_groups() -> None:
    assert effective_trials(_planted_trials(0.9)) == 3
    assert effective_trials(_planted_trials(0.5, groups=4, per_group=5, T=600)) == 4


def test_effective_trials_duplicate_columns_are_one_trial() -> None:
    # N copies of one series were one trial, not N: the search never varied.
    rng = np.random.default_rng(5)
    col = 0.001 + 0.01 * rng.standard_normal(300)
    perf = np.column_stack([col] * 6)
    assert effective_trials(perf) == 1


def test_effective_trials_degenerate_is_zero() -> None:
    # 0 is the "not measurable" sentinel (unusable matrix), never a trial count.
    assert effective_trials(np.ones((10, 3))) == 0  # zero-variance columns
    assert effective_trials(np.zeros((3, 5))) == 0  # T < 4
    assert effective_trials(np.ones(10)) == 0  # not 2-D
    assert effective_trials(np.ones((10, 1))) == 0  # N < 2


def test_cluster_trials_non_finite_rows_are_not_evidence() -> None:
    # NaN/inf rows are dropped (complete-case) before correlations are taken;
    # a few injected holes must not change the recovered partition.
    perf = _planted_trials(0.9)
    noisy = perf.copy()
    noisy[5, 0] = np.nan
    noisy[100, 7] = np.inf
    noisy[200, 11] = -np.inf
    assert cluster_trials(noisy) == cluster_trials(perf)


def test_cross_trial_sharpe_std_hand_value() -> None:
    # Two singleton clusters: std(ddof=1) of two Sharpes is |SR1 - SR2|/sqrt(2).
    a = np.array([0.01, 0.02, 0.03, 0.04])
    b = np.array([-0.01, 0.02, 0.05, 0.01])
    perf = np.column_stack([a, b])
    expected = abs(sharpe_ratio(a) - sharpe_ratio(b)) / math.sqrt(2.0)
    assert cross_trial_sharpe_std(perf, clusters=[[0], [1]]) == pytest.approx(
        expected, rel=1e-12
    )


def test_cross_trial_sharpe_std_aggregates_cluster_members() -> None:
    # A cluster's members are summed into one series before its Sharpe is taken
    # (Lopez de Prado & Lewis 2019 aggregate trials within a cluster), so a
    # cluster of two copies of `a` has exactly the Sharpe of `a` (scale-invariant).
    a = np.array([0.01, 0.02, 0.03, 0.04])
    b = np.array([-0.01, 0.02, 0.05, 0.01])
    perf = np.column_stack([a, a, b])
    expected = abs(sharpe_ratio(a + a) - sharpe_ratio(b)) / math.sqrt(2.0)
    assert cross_trial_sharpe_std(perf, clusters=[[0, 1], [2]]) == pytest.approx(
        expected, rel=1e-12
    )
    assert sharpe_ratio(a + a) == pytest.approx(sharpe_ratio(a), rel=1e-12)


def test_cross_trial_sharpe_std_unmeasurable_is_inf() -> None:
    # Fail-closed: no measurable dispersion means infinite uncertainty, exactly
    # like sharpe_standard_error -- 0.0 would erase the deflation benchmark.
    assert math.isinf(cross_trial_sharpe_std(np.ones(10)))  # not 2-D
    assert math.isinf(cross_trial_sharpe_std(np.ones((10, 4))))  # degenerate columns
    perf = _planted_trials(0.9)
    assert math.isinf(cross_trial_sharpe_std(perf, clusters=[[0, 1, 2]]))  # one cluster


def test_dsr_from_trials_duplicates_collapse_to_psr() -> None:
    # The raw-N deflation punishes a duplicated column as if it were 8 trials;
    # the matrix-faithful DSR sees one effective trial and collapses to the PSR.
    rng = np.random.default_rng(7)
    col = 0.001 + 0.01 * rng.standard_normal(500)
    perf = np.column_stack([col] * 8)
    faithful = deflated_sharpe_ratio_from_trials(perf)
    assert faithful == pytest.approx(probabilistic_sharpe_ratio(col, 0.0), rel=1e-12)
    assert faithful > deflated_sharpe_ratio(col, 8)


def test_dsr_from_trials_correlated_search_deflates_less_than_raw_n() -> None:
    # 24 trials in 3 correlated families: deflating by the raw column count
    # overstates the search, so the matrix-faithful DSR sits strictly between
    # the raw-N published approximation and the undeflated PSR.
    perf = _planted_trials(0.9, groups=3, per_group=8, T=500, seed=11)
    best = int(np.argmax([sharpe_ratio(perf[:, j]) for j in range(perf.shape[1])]))
    faithful = deflated_sharpe_ratio_from_trials(perf)
    published_raw_n = deflated_sharpe_ratio(perf[:, best], perf.shape[1])
    psr = probabilistic_sharpe_ratio(perf[:, best], 0.0)
    assert published_raw_n < faithful < psr


def test_dsr_from_trials_fail_closed() -> None:
    assert deflated_sharpe_ratio_from_trials(np.ones((10, 4))) == 0.0
    assert deflated_sharpe_ratio_from_trials(np.ones(10)) == 0.0


def _single_family(rho: float, n: int = 12, T: int = 500, seed: int = 0) -> np.ndarray:
    """One correlated family: every column shares a factor at pairwise ~``rho``."""
    rng = np.random.default_rng(seed)
    w = math.sqrt(rho)
    base = rng.standard_normal(T)
    return np.column_stack(
        [0.01 * (w * base + math.sqrt(1.0 - rho) * rng.standard_normal(T)) for _ in range(n)]
    )


def test_cluster_trials_single_family_is_one_cluster() -> None:
    # The docstring contract: a parameter sweep around ONE idea is ONE trial.
    # k=1 is unreachable by the silhouette search, so this exercises the
    # homogeneity escape (no structure found + high mean pairwise correlation).
    for seed in (0, 1, 2):
        perf = _single_family(0.9, seed=seed)
        assert cluster_trials(perf) == [list(range(12))]
        assert effective_trials(perf) == 1
    assert effective_trials(_single_family(0.8, seed=3)) == 1


def test_cluster_trials_near_duplicates_are_one_trial() -> None:
    # Near-copies (winner + tiny noise, pairwise rho ~0.9999) are copies for
    # trial-counting purposes: sample correlation of exact copies is 1 only up
    # to float rounding, so an exact-equality duplicate test would miss them
    # and the forced k >= 2 split would report 12 trials where there is one.
    rng = np.random.default_rng(1)
    winner = 0.001 + 0.01 * rng.standard_normal(400)
    for scale in (1e-7, 1e-4):
        dupes = np.column_stack(
            [winner] + [winner + scale * rng.standard_normal(400) for _ in range(11)]
        )
        assert effective_trials(dupes) == 1


def test_cluster_trials_weak_single_family_stays_split() -> None:
    # Below the homogeneity bound (mean pairwise rho < 0.7) a structureless
    # matrix splits into singletons -- the conservative direction: more trials
    # mean stronger deflation, never weaker.
    assert effective_trials(_single_family(0.5, seed=4)) == 12


def test_cluster_trials_short_records_are_not_measurable() -> None:
    # Below 100 complete rows the silhouette search invents families on iid
    # noise (undercounting trials weakens the deflation -- fail-open), so
    # short matrices fail closed to "not measurable" and callers fall back to
    # the raw trial count. At the 100-row floor the count is trusted again.
    rng = np.random.default_rng(2024)
    noise = 0.01 * rng.standard_normal((100, 10))
    assert cluster_trials(noise[:60]) == []
    assert effective_trials(noise[:60]) == 0
    assert effective_trials(noise[:99]) == 0
    assert effective_trials(noise) == 10


def test_cluster_trials_needs_more_rows_than_columns() -> None:
    # With complete rows <= columns the sample correlation matrix is rank
    # deficient and "structure" is guaranteed spurious: not measurable.
    rng = np.random.default_rng(0)
    wide = 0.01 * rng.standard_normal((120, 150))
    assert cluster_trials(wide) == []
    assert effective_trials(wide) == 0


def test_cross_trial_sharpe_std_rejects_malformed_clusters() -> None:
    # A supplied assignment must be partition-like: bad indices would silently
    # wrap (negative) or double-count columns (overlap) under numpy indexing.
    a = np.array([0.01, 0.02, 0.03, 0.04])
    b = np.array([-0.01, 0.02, 0.05, 0.01])
    perf = np.column_stack([a, b])
    with pytest.raises(ValueError, match="valid column index"):
        cross_trial_sharpe_std(perf, clusters=[[0], [-1]])
    with pytest.raises(ValueError, match="valid column index"):
        cross_trial_sharpe_std(perf, clusters=[[0], [2]])
    with pytest.raises(ValueError, match="empty cluster"):
        cross_trial_sharpe_std(perf, clusters=[[0], []])
    with pytest.raises(ValueError, match="more than one cluster"):
        cross_trial_sharpe_std(perf, clusters=[[0, 1], [1]])


def test_dsr_from_trials_explicit_selected_series() -> None:
    # An explicit `selected` series is judged against the matrix's benchmark.
    perf = _planted_trials(0.7, seed=23)
    rng = np.random.default_rng(29)
    other = 0.0005 + 0.01 * rng.standard_normal(400)
    clusters = cluster_trials(perf)
    sigma = cross_trial_sharpe_std(perf, clusters=clusters)
    benchmark = expected_max_sharpe_benchmark(sigma, len(clusters))
    assert deflated_sharpe_ratio_from_trials(perf, selected=other) == pytest.approx(
        probabilistic_sharpe_ratio(other, benchmark), rel=1e-12
    )


def test_pbo_ignores_a_period_that_is_not_fully_observed() -> None:
    """One blank CSV cell must not decide which configuration was IS-best.

    Every other statistic here drops non-finite entries, and Verdict documents
    it: "non-finite rows -- e.g. blank CSV cells -- are not evidence and are not
    counted". PBO did not. A NaN makes that column's summed IS performance NaN,
    np.argmax treats NaN as the maximum, and the NaN column is then selected as
    the in-sample BEST in every partition. Measured before the fix on this exact
    matrix: PBO moved from 0.000 to 0.423 and argmax picked column 3 rather than
    the genuine winner in column 0.
    """
    rng = np.random.default_rng(11)
    perf = rng.normal(0.0, 0.01, (200, 6))
    perf[:, 0] += 0.002  # column 0 is the real winner

    holed = perf.copy()
    holed[37, 3] = np.nan  # one blank cell in a mediocre column

    clean = probability_of_backtest_overfitting(perf)
    with_gap = probability_of_backtest_overfitting(holed)
    without_that_period = probability_of_backtest_overfitting(np.delete(perf, 37, axis=0))

    assert with_gap == pytest.approx(without_that_period)
    assert with_gap == pytest.approx(clean, abs=1e-9)


def test_pbo_still_fails_closed_when_a_whole_column_is_blank() -> None:
    """Dropping incomplete periods must not quietly rescue an unrankable input."""
    rng = np.random.default_rng(11)
    perf = rng.normal(0.0, 0.01, (200, 6))
    perf[:, 3] = np.nan  # nothing is fully observed any more

    assert probability_of_backtest_overfitting(perf, degenerate_value=1.0) == 1.0
