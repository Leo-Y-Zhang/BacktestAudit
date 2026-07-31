"""End-to-end tests for evaluate() with positive / negative / overfit controls.

These are the load-bearing correctness tests: a validator that cannot tell a real
edge from noise from an overfit search is worse than useless.
"""

from __future__ import annotations

import math

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


# ── evidence gap (MinTRL / MinBTL) ────────────────────────────────────────────


def test_passing_verdict_confirms_track_record_length() -> None:
    """On a significant record, MinTRL is finite, met, and confirmed in reasons."""
    rng = np.random.default_rng(42)
    returns = 0.0008 + 0.008 * rng.standard_normal(1500)
    verdict = evaluate(returns, n_trials=1)
    assert verdict.deployable is True
    assert np.isfinite(verdict.min_track_record)
    assert verdict.min_track_record <= verdict.n_periods
    assert verdict.min_backtest_years == 0.0  # single trial: no selection minimum
    assert any("MinTRL" in r for r in verdict.reasons)
    assert "MinTRL" in verdict.summary()


def test_failing_verdict_reports_evidence_gap() -> None:
    """A real-but-weak edge fails significance with a finite observation gap."""
    rng = np.random.default_rng(9)
    returns = 0.0003 + 0.01 * rng.standard_normal(800)  # modest edge
    verdict = evaluate(returns, n_trials=1)
    assert verdict.deployable is False
    assert verdict.deflated_sharpe < 0.95
    assert np.isfinite(verdict.min_track_record)
    assert verdict.min_track_record > verdict.n_periods
    assert any("Evidence gap" in r for r in verdict.reasons)
    assert "MinTRL" in verdict.summary()


def test_negative_drift_evidence_gap_is_unreachable() -> None:
    """With a non-positive Sharpe no track record length ever reaches the bar."""
    rng = np.random.default_rng(8)
    returns = -0.001 + 0.01 * rng.standard_normal(500)
    verdict = evaluate(returns, n_trials=1)
    assert np.isinf(verdict.min_track_record)
    assert any("Evidence gap" in r for r in verdict.reasons)


def test_min_trl_agrees_with_the_dsr_gate() -> None:
    """T >= MinTRL if and only if the DSR clears the bar (same formula, inverted)."""
    for seed in (1, 9, 42, 123, 2024):
        rng = np.random.default_rng(seed)
        returns = 0.0004 + 0.009 * rng.standard_normal(900)
        verdict = evaluate(returns, n_trials=1)
        if np.isfinite(verdict.min_track_record):
            assert (verdict.n_periods >= verdict.min_track_record) == (
                verdict.deflated_sharpe >= 0.95
            )
        else:
            assert verdict.deflated_sharpe < 0.95


@pytest.mark.parametrize("n_trials", [1, 25])
def test_non_finite_observations_are_not_evidence(n_trials: int) -> None:
    """NaN/inf entries are dropped by every statistic, so they must not count
    as track record either: a padded series yields the identical verdict --
    counts, shortfall arithmetic and reason strings -- as its finite subset.
    """
    rng = np.random.default_rng(9)
    finite = 0.0003 + 0.01 * rng.standard_normal(800)  # modest edge
    padded = np.concatenate([finite, np.full(300, np.nan), [np.inf, -np.inf]])
    clean = evaluate(finite, n_trials=n_trials)
    noisy = evaluate(padded, n_trials=n_trials)
    assert clean.n_periods == 800
    assert noisy.n_periods == 800
    assert noisy.min_track_record == clean.min_track_record
    assert noisy.min_backtest_years == clean.min_backtest_years
    assert noisy.deflated_sharpe == clean.deflated_sharpe
    assert noisy.reasons == clean.reasons
    assert noisy.summary() == clean.summary()


def test_nan_padding_cannot_fake_a_sufficient_track_record() -> None:
    """Regression: enough NaN rows to push the *raw* length past MinTRL must
    not make the verdict call the record sufficient while the DSR gate fails.
    """
    rng = np.random.default_rng(9)
    finite = 0.0003 + 0.01 * rng.standard_normal(800)
    base = evaluate(finite, n_trials=1)
    assert base.deflated_sharpe < 0.95
    assert np.isfinite(base.min_track_record)
    needed = math.ceil(base.min_track_record)
    padded = np.concatenate([finite, np.full(needed, np.nan)])  # raw size > MinTRL
    verdict = evaluate(padded, n_trials=1)
    assert verdict.n_periods == 800
    # The MinTRL-vs-DSR equivalence must hold for the *surfaced* count too.
    assert (verdict.n_periods >= verdict.min_track_record) == (
        verdict.deflated_sharpe >= 0.95
    )
    assert "(sufficient)" not in verdict.summary()  # the MinTRL status marker
    assert f"short {needed - 800} obs" in verdict.summary()
    gap = next(r for r in verdict.reasons if r.startswith("Evidence gap"))
    assert f"short about {needed - 800} observations" in gap


def test_degenerate_record_evidence_gap_names_degeneracy_not_benchmark() -> None:
    """A zero-variance record has no Sharpe-vs-benchmark comparison at all, so
    the unreachable-MinTRL reason must name the degeneracy, not the benchmark.
    """
    verdict = evaluate([0.01] * 10, n_trials=1)
    assert np.isinf(verdict.min_track_record)
    gap = next(r for r in verdict.reasons if r.startswith("Evidence gap"))
    assert "degenerate" in gap
    assert "benchmark" not in gap
    assert "unreachable" in verdict.summary()


def test_matrix_verdict_reports_min_backtest_length() -> None:
    """A searched candidate matrix gets a positive MinBTL in years."""
    rng = np.random.default_rng(2024)
    candidates = 0.01 * rng.standard_normal((600, 60))
    verdict = evaluate(candidates)
    assert verdict.n_trials == 60
    assert verdict.min_backtest_years > 0.0
    assert "MinBTL" in verdict.summary()
    # 600 daily bars ~ 2.4 years is far short of the MinBTL for a best-of-60
    # noise search at this Sharpe, so the shortfall is called out.
    assert verdict.min_backtest_years > verdict.n_periods / verdict.periods_per_year
    assert any("MinBTL" in r for r in verdict.reasons)


# ── matrix-faithful deflation (Lopez de Prado & Lewis 2019) ───────────────────


def _planted_candidates(
    rho: float, groups: int = 3, per_group: int = 4, T: int = 500, seed: int = 11
) -> np.ndarray:
    """``groups`` families of ``per_group`` correlated trials (see test_stats)."""
    rng = np.random.default_rng(seed)
    w = math.sqrt(rho)
    cols = []
    for _ in range(groups):
        base = rng.standard_normal(T)
        for _ in range(per_group):
            cols.append(0.01 * (w * base + math.sqrt(1.0 - rho) * rng.standard_normal(T)))
    return np.column_stack(cols)


def test_matrix_default_reports_effective_trials() -> None:
    """The default matrix path measures the search from the matrix itself."""
    perf = _planted_candidates(0.9)
    verdict = evaluate(perf)
    assert verdict.n_trials == 12  # configurations tried stays the column count
    assert verdict.effective_trials == 3  # ...but only 3 are effectively distinct
    assert verdict.cross_trial_sharpe_std is not None
    assert verdict.cross_trial_sharpe_std >= 0.0
    assert any("effective trials" in r for r in verdict.reasons)
    assert any("Lopez de Prado" in r for r in verdict.reasons)
    assert "Effective trials" in verdict.summary()


def test_matrix_explicit_n_trials_keeps_published_deflation() -> None:
    """An explicit n_trials asserts the search size, so the matrix is no longer
    the whole search and the published raw-count deflation applies unchanged."""
    from lyravalidate.stats import annualized_sharpe, deflated_sharpe_ratio

    perf = _planted_candidates(0.9)
    verdict = evaluate(perf, n_trials=12)
    assert verdict.effective_trials is None
    assert verdict.cross_trial_sharpe_std is None
    best = int(np.argmax([annualized_sharpe(perf[:, j]) for j in range(perf.shape[1])]))
    assert verdict.deflated_sharpe == pytest.approx(
        deflated_sharpe_ratio(perf[:, best], 12), rel=1e-12
    )
    assert not any("effective trials" in r for r in verdict.reasons)


def test_matrix_iid_noise_keeps_the_overfit_conclusion() -> None:
    """Independent noise trials have no correlation structure: every column is
    its own effective trial, and the overfit-search conclusion is unchanged."""
    rng = np.random.default_rng(2024)
    candidates = 0.01 * rng.standard_normal((600, 60))
    verdict = evaluate(candidates)
    assert verdict.effective_trials == 60
    assert verdict.classification == "PROBABLY_OVERFIT"
    assert verdict.deflated_sharpe < 0.95


def test_duplicated_configurations_are_not_extra_trials() -> None:
    """Six copies of one strategy were ONE trial: the matrix-faithful DSR
    collapses to the PSR instead of deflating for a search that never happened."""
    from lyravalidate.stats import probabilistic_sharpe_ratio

    rng = np.random.default_rng(7)
    col = 0.001 + 0.01 * rng.standard_normal(500)
    perf = np.column_stack([col] * 6)
    verdict = evaluate(perf)
    assert verdict.effective_trials == 1
    assert verdict.cross_trial_sharpe_std is None  # one trial has no dispersion
    assert verdict.deflated_sharpe == pytest.approx(
        probabilistic_sharpe_ratio(col, 0.0), rel=1e-12
    )


def test_matrix_min_trl_agrees_with_the_dsr_gate() -> None:
    """The MinTRL-vs-DSR equivalence must hold against the matrix-faithful
    benchmark too (same SR*, inverted in the sample length)."""
    for seed in (1, 9, 42, 123, 2024):
        rng = np.random.default_rng(seed)
        perf = 0.0004 + 0.009 * rng.standard_normal((900, 10))
        verdict = evaluate(perf)
        if np.isfinite(verdict.min_track_record):
            assert (verdict.n_periods >= verdict.min_track_record) == (
                verdict.deflated_sharpe >= 0.95
            )
        else:
            assert verdict.deflated_sharpe < 0.95


def test_matrix_falls_back_to_published_when_cross_section_unusable() -> None:
    """A matrix without two usable columns cannot be measured cross-sectionally;
    the verdict falls back to the published raw-count deflation (fail-closed)."""
    from lyravalidate.stats import deflated_sharpe_ratio

    rng = np.random.default_rng(4)
    col = 0.0008 + 0.008 * rng.standard_normal(1000)
    perf = np.column_stack([col, np.zeros(1000)])  # second column: zero variance
    verdict = evaluate(perf)
    assert verdict.effective_trials is None
    assert verdict.deflated_sharpe == pytest.approx(
        deflated_sharpe_ratio(col, 2), rel=1e-12
    )


def test_single_family_matrix_collapses_to_one_effective_trial() -> None:
    """A parameter sweep around one idea (one correlated family, rho ~0.9) is
    one trial: no selection deflation applies and the DSR equals the PSR of
    the selected column. Regression for the k=1-unreachable defect, where the
    homogeneous blob shattered into 12 singletons and the diagnostic reported
    12 effective trials where the truth is 1."""
    from lyravalidate.stats import probabilistic_sharpe_ratio, sharpe_ratio

    rng = np.random.default_rng(0)
    w = math.sqrt(0.9)
    base = rng.standard_normal(500)
    perf = np.column_stack(
        [0.01 * (w * base + math.sqrt(0.1) * rng.standard_normal(500)) for _ in range(12)]
    )
    verdict = evaluate(perf)
    assert verdict.effective_trials == 1
    assert verdict.cross_trial_sharpe_std is None
    best = int(np.argmax([sharpe_ratio(perf[:, j]) for j in range(12)]))
    assert verdict.deflated_sharpe == pytest.approx(
        probabilistic_sharpe_ratio(perf[:, best], 0.0), rel=1e-12
    )


def test_matrix_short_record_falls_back_to_published_deflation() -> None:
    """Below 100 complete rows measured clustering is not trustworthy (it
    invents families on iid noise, weakening the deflation), so the verdict
    falls back to the published raw-count path -- fail-closed."""
    from lyravalidate.stats import deflated_sharpe_ratio, sharpe_ratio

    rng = np.random.default_rng(3)
    short = 0.01 * rng.standard_normal((60, 10))
    verdict = evaluate(short)
    assert verdict.effective_trials is None
    assert verdict.cross_trial_sharpe_std is None
    best = int(np.argmax([sharpe_ratio(short[:, j]) for j in range(10)]))
    assert verdict.deflated_sharpe == pytest.approx(
        deflated_sharpe_ratio(short[:, best], 10), rel=1e-12
    )


def test_hidden_search_near_duplicates_pin_the_documented_trust_model() -> None:
    """TRUST-MODEL PIN (documented limitation, not a defect being celebrated).

    Take the best of a hidden 200-trial noise search and submit only that
    winner plus 11 near-copies, with no n_trials. The matrix-faithful default
    treats the matrix as the WHOLE search: the near-copies collapse to one
    effective trial, no deflation applies, and the DSR equals the winner's
    undeflated PSR -- which can clear every gate. That is faithful to Lopez
    de Prado & Lewis (near-copies ARE one trial; the hidden search is simply
    not in evidence), but it means the default CANNOT protect against a
    search hidden outside the matrix. The protections are (a) the reason
    string that says the matrix is trusted as the whole search and (b) the
    explicit n_trials path, which restores the raw-count deflation and flips
    the verdict here. Both are asserted so neither can silently regress.
    """
    from lyravalidate.stats import probabilistic_sharpe_ratio, sharpe_ratio

    rng = np.random.default_rng(1)
    hidden_search = 0.01 * rng.standard_normal((400, 200))
    best = int(np.argmax([sharpe_ratio(hidden_search[:, j]) for j in range(200)]))
    winner = hidden_search[:, best]
    dupes = np.column_stack(
        [winner] + [winner + 1e-4 * rng.standard_normal(400) for _ in range(11)]
    )

    laundered = evaluate(dupes)
    assert laundered.effective_trials == 1
    selected = int(np.argmax([sharpe_ratio(dupes[:, j]) for j in range(12)]))
    assert laundered.deflated_sharpe == pytest.approx(
        probabilistic_sharpe_ratio(dupes[:, selected], 0.0), rel=1e-12
    )
    assert laundered.deflated_sharpe > 0.95  # undeflated: the hidden search is invisible
    assert any("trusted as the WHOLE search" in r for r in laundered.reasons)
    assert any("pass n_trials" in r for r in laundered.reasons)

    honest = evaluate(dupes, n_trials=200)  # the search size, admitted
    assert honest.deflated_sharpe < 0.95
    assert honest.deployable is False


def test_near_zero_measured_dispersion_adds_trust_caveat() -> None:
    """Two families of near-copies whose aggregates share the same Sharpe:
    the measured cross-trial dispersion is far below the Sharpe estimator
    noise, so the deflation is weak and the reasons must say so."""
    rng = np.random.default_rng(5)
    a = 0.001 + 0.01 * rng.standard_normal(500)
    b = np.random.default_rng(6).permutation(a)  # identical Sharpe, uncorrelated
    fam_a = np.column_stack([a + 1e-5 * rng.standard_normal(500) for _ in range(6)])
    fam_b = np.column_stack([b + 1e-5 * rng.standard_normal(500) for _ in range(6)])
    verdict = evaluate(np.hstack([fam_a, fam_b]))
    assert verdict.effective_trials == 2
    assert verdict.cross_trial_sharpe_std is not None
    assert any(r.startswith("Caveat") for r in verdict.reasons)
    assert any("nearly interchangeable" in r for r in verdict.reasons)


def test_matrix_faithful_dsr_change_is_pinned_and_deliberate() -> None:
    """REGRESSION PIN for the deliberate numeric change on the matrix path.

    Before the matrix-faithful DSR (Lopez de Prado & Lewis 2019), the matrix
    path deflated the best column by the raw column count with the column's
    own Sharpe standard error as the per-trial dispersion. Now the benchmark
    is measured from the matrix itself: cross-trial Sharpe dispersion across
    the effective (correlation-clustered) trials. For a search of 12 pure-noise
    configurations that are really 3 correlated families (rho ~ 0.9):

    * old matrix-path number (still available via an explicit n_trials):
      DSR 0.786238, MinTRL 2146 obs -- 12 correlated columns treated as 12
      independent trials over-deflated the search;
    * new default: DSR 0.961623, MinTRL 433 obs, effective trials 3.

    The verdict itself stays not-deployable HERE: the higher DSR is a noisy
    estimate at only 3 effective trials, and the PBO gate (CSCV measures the
    selection directly) still catches this overfit search. That is a fact
    about this case, not a guarantee -- when the submitted matrix hides the
    real search entirely, no gate can see it (see
    test_hidden_search_near_duplicates_pin_the_documented_trust_model).
    Pinned values were computed by running this code (2026-07-31); they move
    only if the algorithm changes, which is exactly what this pin is here to
    detect.
    """
    from lyravalidate.stats import annualized_sharpe, deflated_sharpe_ratio

    perf = _planted_candidates(0.9)
    best = int(np.argmax([annualized_sharpe(perf[:, j]) for j in range(perf.shape[1])]))
    old = deflated_sharpe_ratio(perf[:, best], 12)
    assert old == pytest.approx(0.786238, abs=1e-6)

    verdict = evaluate(perf)
    assert verdict.effective_trials == 3
    assert verdict.deflated_sharpe == pytest.approx(0.961623, abs=1e-6)
    assert verdict.deflated_sharpe != old
    assert verdict.min_track_record == pytest.approx(432.008638, rel=1e-6)

    old_verdict = evaluate(perf, n_trials=12)  # the published path, unchanged
    assert old_verdict.deflated_sharpe == pytest.approx(old, rel=1e-12)
    assert old_verdict.min_track_record == pytest.approx(2145.528044, rel=1e-6)

    # The deliberate change does not weaken the overall verdict here.
    assert verdict.classification == "PROBABLY_OVERFIT"
    assert verdict.deployable is False
    assert verdict.pbo > 0.5


def test_matrix_faithful_dsr_iid_noise_pin_validates_the_null_approximation() -> None:
    """REGRESSION PIN, independent-trials direction: for 60 iid noise columns
    the effective count equals the column count and the measured cross-trial
    dispersion is close to the null approximation the published path assumes,
    so the DSR barely moves (0.415233 published raw-N vs 0.426080 measured;
    computed by running this code, 2026-07-31). The conclusion is unchanged."""
    from lyravalidate.stats import annualized_sharpe, deflated_sharpe_ratio

    rng = np.random.default_rng(2024)
    candidates = 0.01 * rng.standard_normal((600, 60))
    best = int(np.argmax([annualized_sharpe(candidates[:, j]) for j in range(60)]))
    old = deflated_sharpe_ratio(candidates[:, best], 60)
    assert old == pytest.approx(0.415233, abs=1e-6)
    verdict = evaluate(candidates)
    assert verdict.deflated_sharpe == pytest.approx(0.426080, abs=1e-6)
    assert abs(verdict.deflated_sharpe - old) < 0.02
    assert verdict.classification == "PROBABLY_OVERFIT"


def test_no_surviving_fold_refuses_instead_of_gating_in_sample() -> None:
    """Supplying predictions and targets IS the request to be judged out of sample.

    When purging leaves no usable fold, _walk_forward_oos returns None and the
    OOS block is skipped, so `strategy` stays the IN-SAMPLE series and every
    statistic - Sharpe, PSR, the deflated Sharpe, the gate - is computed on it.
    Measured before the guard: a 60-bar record with a 30-bar label horizon (an
    ordinary setting for a 30-bar-ahead prediction) returned DEPLOYABLE on an
    in-sample Sharpe of 12.11, with nothing in `reasons` saying the out-of-sample
    basis was missing. A tool that exists to refuse overfitted strategies must
    not quietly answer using the overfitted basis.
    """
    from lyravalidate.crossval import PurgedWalkForwardSplitter

    rng = np.random.default_rng(4)
    n = 60
    targets = rng.normal(0.0, 0.01, n)
    predictions = targets + rng.normal(0.0, 0.001, n)  # near-perfect in sample
    returns = predictions * targets

    splitter = PurgedWalkForwardSplitter(
        train_size=20, valid_size=5, test_size=5, embargo_size=2, label_horizon=30
    )
    verdict = evaluate(
        returns, predictions=predictions, targets=targets, n_trials=1, splitter=splitter
    )

    assert not verdict.deployable
    assert verdict.classification == "NOT_DEPLOYABLE"
    assert any("no usable fold" in r for r in verdict.reasons), verdict.reasons
    # and it must say the numbers shown are in-sample, not quietly imply otherwise
    assert any("IN-SAMPLE" in r for r in verdict.reasons)


def test_a_surviving_fold_still_gates_out_of_sample() -> None:
    """Liveness half: the guard must not refuse a run that DOES have folds."""
    from lyravalidate.crossval import PurgedWalkForwardSplitter

    rng = np.random.default_rng(4)
    n = 60
    targets = rng.normal(0.0, 0.01, n)
    predictions = targets + rng.normal(0.0, 0.001, n)
    returns = predictions * targets

    splitter = PurgedWalkForwardSplitter(
        train_size=20, valid_size=5, test_size=5, embargo_size=2, label_horizon=1
    )
    verdict = evaluate(
        returns, predictions=predictions, targets=targets, n_trials=1, splitter=splitter
    )
    assert not any("no usable fold" in r for r in verdict.reasons)
