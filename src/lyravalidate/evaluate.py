"""High-level, default-deny verdict on a backtest.

Point :func:`evaluate` at a strategy's realised returns (and, optionally, a
signal's predictions and the targets it was meant to forecast) and it returns a
typed :class:`Verdict`. The classification is deliberately conservative:

* ``DEPLOYABLE`` only when the deflated Sharpe, the annualised Sharpe and -- when
  computable -- the PBO all clear their bars.
* ``PROBABLY_OVERFIT`` when the record *looked* good in sample but does not
  survive deflation, or when CSCV says the selection is overfit (high PBO).
* ``NOT_DEPLOYABLE`` otherwise (e.g. no edge at all -- pure noise).

This is a *measurement* tool. It does not generate signals, size positions, or
promise returns; it only tells you how much of an apparent edge is likely real.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import numpy.typing as npt

from .crossval import PurgedWalkForwardSplitter, default_walk_forward_splitter
from .stats import (
    annualized_sharpe,
    cluster_trials,
    cross_trial_sharpe_std,
    deflated_sharpe_ratio,
    expected_max_sharpe_benchmark,
    information_coefficient,
    minimum_backtest_length,
    minimum_track_record_length,
    probabilistic_sharpe_ratio,
    probability_of_backtest_overfitting,
    sharpe_standard_error,
)

__all__ = ["Thresholds", "Verdict", "evaluate"]

Classification = Literal["DEPLOYABLE", "NOT_DEPLOYABLE", "PROBABLY_OVERFIT"]

# Matrix-measured cross-trial Sharpe dispersion below this fraction of the
# selected series' own Sharpe estimator noise triggers a trust-model caveat in
# the reasons. Rationale: under the null of genuinely independent trials the
# cross-trial dispersion approximately equals the per-trial estimator noise
# (that equality is exactly the approximation the published raw-count DSR
# makes), so a measured dispersion under half that level says the supposedly
# distinct trials are statistically closer than independent re-estimates of
# one strategy -- a search of near-variants, which earns almost no deflation.
_NEAR_ZERO_DISPERSION_FRACTION = 0.5


@dataclass(frozen=True)
class Thresholds:
    """Deployment policy thresholds.

    These are *policy*, not published constants -- expose and tune them per desk.
    The defaults are intentionally demanding (a ``0.95`` deflated-Sharpe bar
    mirrors the conventional 5% significance level).
    """

    min_deflated_sharpe: float = 0.95
    """Minimum Deflated Sharpe probability (Bailey & Lopez de Prado, 2014)."""
    min_sharpe: float = 0.75
    """Minimum annualised, net-of-cost Sharpe ratio."""
    max_pbo: float = 0.5
    """Maximum tolerated Probability of Backtest Overfitting."""


@dataclass(frozen=True)
class Verdict:
    """The outcome of :func:`evaluate`."""

    deployable: bool
    classification: Classification
    deflated_sharpe: float
    pbo: float
    sharpe: float
    n_trials: int
    reasons: list[str] = field(default_factory=list)
    # Supplementary diagnostics (not gates):
    probabilistic_sharpe: float = float("nan")
    # Finite observations actually judged. Every statistic drops NaN/inf
    # entries (see lyravalidate.stats), so non-finite rows -- e.g. blank CSV
    # cells -- are not evidence and are not counted here either.
    n_periods: int = 0
    periods_per_year: int = 252
    # Evidence gap: MinTRL (observations needed for the deflated Sharpe to reach
    # the significance bar at the observed moments; inf = unreachable, NaN = not
    # computed) and MinBTL (years of backtest needed for the observed Sharpe to
    # beat the expected best of n_trials noise trials; NaN = not computed).
    min_track_record: float = float("nan")
    min_backtest_years: float = float("nan")
    # Matrix-faithful deflation (Lopez de Prado & Lewis, 2019): set only when a
    # candidate matrix was judged without an explicit n_trials override, so the
    # matrix itself was the whole search. `effective_trials` is the number of
    # correlation clusters among the configurations; `cross_trial_sharpe_std`
    # is the per-period Sharpe dispersion across the cluster aggregates that
    # the deflation benchmark was built from (None when only one effective
    # trial exists -- a single trial has no dispersion and needs none).
    effective_trials: int | None = None
    cross_trial_sharpe_std: float | None = None
    oos_sharpe: float | None = None
    oos_information_coefficient: float | None = None

    def summary(self) -> str:
        """A short human-readable report."""
        pbo = "n/a" if math.isnan(self.pbo) else f"{self.pbo:.3f}"
        lines = [
            f"Verdict: {self.classification} (deployable={self.deployable})",
            f"  Deflated Sharpe : {self.deflated_sharpe:.3f} "
            f"(probability the edge is real after deflation)",
            f"  Annualised Sharpe: {self.sharpe:.3f}",
            f"  PBO             : {pbo}",
            f"  Trials assumed  : {self.n_trials}",
        ]
        if self.effective_trials is not None:
            lines.append(
                f"  Effective trials: {self.effective_trials} "
                "(correlation clusters among the configurations)"
            )
        if not math.isnan(self.min_track_record):
            if math.isinf(self.min_track_record):
                # inf can mean Sharpe <= benchmark, a degenerate record, or a
                # confidence bar of 1 -- the reasons say which; stay neutral here.
                lines.append(
                    "  MinTRL          : unreachable (fail-closed: no finite track "
                    "record length reaches the significance bar)"
                )
            else:
                needed = math.ceil(self.min_track_record)
                if self.n_periods >= self.min_track_record:
                    status = "sufficient"
                else:
                    short = needed - self.n_periods
                    years = short / self.periods_per_year
                    status = f"short {short} obs ~ {years:.1f} years"
                lines.append(
                    f"  MinTRL          : {needed} obs needed; have {self.n_periods} ({status})"
                )
        if self.n_trials > 1 and not math.isnan(self.min_backtest_years):
            if math.isinf(self.min_backtest_years):
                lines.append(
                    "  MinBTL          : unattainable (observed Sharpe is not positive)"
                )
            else:
                observed_years = self.n_periods / self.periods_per_year
                lines.append(
                    f"  MinBTL          : {self.min_backtest_years:.1f} years needed for "
                    f"{self.n_trials} trials; have {observed_years:.1f}"
                )
        if self.oos_sharpe is not None:
            lines.append(f"  Walk-forward OOS Sharpe: {self.oos_sharpe:.3f}")
        if self.reasons:
            lines.append("  Reasons:")
            lines.extend(f"    - {r}" for r in self.reasons)
        return "\n".join(lines)


@dataclass(frozen=True)
class _OOSResult:
    returns: npt.NDArray[np.float64]
    mean_ic: float
    n_folds: int


@dataclass(frozen=True)
class _MatrixDeflation:
    """Matrix-measured inputs for the DSR benchmark (Lopez de Prado & Lewis, 2019)."""

    effective_trials: int
    cross_trial_std: float | None  # None when one effective trial (no dispersion exists)


def _walk_forward_oos(
    predictions: npt.ArrayLike,
    targets: npt.ArrayLike,
    splitter: PurgedWalkForwardSplitter,
) -> _OOSResult | None:
    """Honest out-of-sample evaluation of a per-period signal.

    On each fold the signal is standardised using *training* statistics only
    (no look-ahead), then applied to the held-out test bars; the per-bar OOS
    "strategy return" is ``z_test * target_test``. OOS information coefficients
    (prediction vs target correlation on the test block) are averaged across
    folds. Returns ``None`` if no usable fold survives purging.
    """
    preds = np.asarray(predictions, dtype=np.float64).ravel()
    tgts = np.asarray(targets, dtype=np.float64).ravel()
    if preds.size != tgts.size:
        raise ValueError("predictions and targets must have the same length")

    oos_returns: list[npt.NDArray[np.floating[Any]]] = []
    ics: list[float] = []
    for train_idx, _valid_idx, test_idx in splitter.split(preds.size):
        train_p = preds[train_idx]
        mu = float(np.mean(train_p))
        sd = float(np.std(train_p, ddof=1))
        if sd <= 0.0:
            continue
        z = (preds[test_idx] - mu) / sd
        oos_returns.append(z * tgts[test_idx])
        ics.append(information_coefficient(preds[test_idx], tgts[test_idx]))

    if not oos_returns:
        return None
    combined: npt.NDArray[np.float64] = np.asarray(
        np.concatenate(oos_returns), dtype=np.float64
    )
    mean_ic = float(np.mean(ics)) if ics else 0.0
    return _OOSResult(returns=combined, mean_ic=mean_ic, n_folds=len(oos_returns))


def _resolve_strategy(
    returns: npt.ArrayLike,
    *,
    n_trials: int | None,
    pbo_splits: int,
    periods_per_year: int,
) -> tuple[npt.NDArray[np.float64], float, int, _MatrixDeflation | None]:
    """Reduce raw ``returns`` to ``(strategy_series, pbo, n_trials, matrix_deflation)``.

    A 2-D input is treated as a candidate matrix (``T`` periods x ``N`` configs):
    PBO is computed across the columns and the column with the highest full-sample
    annualised Sharpe is taken as the *selected* strategy, with ``n_trials``
    defaulting to ``N`` so the deflation reflects the search. A 1-D input is a
    single strategy with no PBO (``NaN``) and ``n_trials`` defaulting to ``1``.

    When the matrix is judged without an explicit ``n_trials`` override, the
    matrix *is* the whole search, and the DSR benchmark inputs are measured
    from it directly (Lopez de Prado & Lewis, 2019): the effective number of
    independent trials is the count of correlation clusters among the columns
    and the per-trial dispersion is the cross-trial Sharpe standard deviation.
    An explicit ``n_trials`` asserts a search larger than (or different from)
    the matrix, so the published raw-count approximation is kept for it. When
    the cross-section is unusable or too short to measure (fewer than 100
    complete rows, or no more rows than columns -- see
    :func:`lyravalidate.stats.cluster_trials`) the published approximation is
    the fallback (fail-closed: the assumed search is never weakened by a
    failed or untrustworthy measurement).
    """
    arr: npt.NDArray[np.float64] = np.asarray(returns, dtype=np.float64)
    if arr.ndim == 1:
        return arr, float("nan"), (n_trials if n_trials is not None else 1), None
    if arr.ndim == 2:
        if arr.shape[1] <= 1:
            return arr.ravel(), float("nan"), (n_trials if n_trials is not None else 1), None
        N = int(arr.shape[1])
        # Fail-closed PBO: an unrankable matrix is rejected, not waved through.
        # Lift max_blocks to honour a user who asks for more CSCV blocks than the
        # default cap of 16 (otherwise --pbo-splits would silently clamp at 16).
        pbo = probability_of_backtest_overfitting(
            arr,
            pbo_splits,
            max_blocks=max(pbo_splits, 16),
            degenerate_value=1.0,
        )
        sharpes = [annualized_sharpe(arr[:, j], periods_per_year) for j in range(N)]
        best = int(np.argmax(sharpes))
        matrix_deflation: _MatrixDeflation | None = None
        if n_trials is None:
            clusters = cluster_trials(arr)
            if len(clusters) == 1:
                matrix_deflation = _MatrixDeflation(effective_trials=1, cross_trial_std=None)
            elif len(clusters) >= 2:
                sigma_cross = cross_trial_sharpe_std(arr, clusters=clusters)
                if math.isfinite(sigma_cross):
                    matrix_deflation = _MatrixDeflation(
                        effective_trials=len(clusters), cross_trial_std=sigma_cross
                    )
        return arr[:, best], pbo, (n_trials if n_trials is not None else N), matrix_deflation
    raise ValueError("returns must be 1-D (one strategy) or 2-D (T x N candidate matrix)")


def evaluate(
    returns: npt.ArrayLike,
    predictions: npt.ArrayLike | None = None,
    targets: npt.ArrayLike | None = None,
    *,
    n_trials: int | None = None,
    periods_per_year: int = 252,
    thresholds: Thresholds | None = None,
    pbo_splits: int = 16,
    splitter: PurgedWalkForwardSplitter | None = None,
) -> Verdict:
    """Return a default-deny :class:`Verdict` on a backtest.

    Parameters
    ----------
    returns:
        Realised per-period returns. Either 1-D (a single strategy) or 2-D
        (``T`` periods x ``N`` candidate configurations -- PBO is then computed
        across columns and the best in-sample column is judged; without an
        explicit ``n_trials`` the DSR benchmark is also *measured* from the
        matrix -- cross-trial Sharpe dispersion across the effective,
        correlation-clustered trials, per Lopez de Prado & Lewis, 2019 --
        instead of approximated from the selected column alone).
    predictions, targets:
        Optional paired per-period signal scores and the forward returns they aim
        to predict. When supplied, an honest purged walk-forward OOS series is
        built and used as the basis for the Sharpe / deflated-Sharpe gates.
    n_trials:
        Number of configurations tried during research (selection-bias count).
        Defaults to ``N`` for a candidate matrix, else ``1``. Supplying it for
        a matrix asserts the search was *not* just the matrix, so the published
        raw-count deflation is applied instead of the matrix-measured one.
    periods_per_year:
        Annualisation factor for the Sharpe ratio (252 trading days by default).
    thresholds:
        Deployment policy (see :class:`Thresholds`). Defaults to the standard bar.
    pbo_splits:
        Number of CSCV blocks for the PBO estimate (forced even, ``>= 2``).
    splitter:
        Walk-forward splitter for the predictions/targets path; auto-sized if
        omitted.

    Returns
    -------
    Verdict
        Typed result with ``deployable``, ``classification`` and ``reasons``.
    """
    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be > 0")
    thr = thresholds or Thresholds()

    strategy, pbo, n_trials_eff, matrix_deflation = _resolve_strategy(
        returns,
        n_trials=n_trials,
        pbo_splits=pbo_splits,
        periods_per_year=periods_per_year,
    )

    oos_sharpe: float | None = None
    oos_ic: float | None = None
    # Supplying predictions and targets IS the request to be judged out of
    # sample. When no fold survives purging, _walk_forward_oos returns None, and
    # this block used to be skipped in silence: `strategy` stayed the IN-SAMPLE
    # series and every statistic below - Sharpe, PSR, the deflated Sharpe, the
    # gate - was computed on it. A caller who asked to be judged out of sample
    # was judged in sample instead, and could be told DEPLOYABLE with nothing in
    # the output saying so. For a tool whose entire purpose is refusing
    # overfitted strategies, silently falling back to the overfitted basis is
    # the one failure it must not have.
    oos_unavailable = False
    if predictions is not None and targets is not None:
        spl = splitter or default_walk_forward_splitter(len(np.asarray(targets).ravel()))
        oos = _walk_forward_oos(predictions, targets, spl)
        oos_unavailable = oos is None
        if oos is not None:
            strategy = oos.returns  # the OOS series is the honest basis for gating
            oos_sharpe = annualized_sharpe(strategy, periods_per_year)
            oos_ic = oos.mean_ic
            # The matrix-measured benchmark describes the in-sample search, in
            # the in-sample series' per-period Sharpe units; it does not apply
            # to a different (walk-forward OOS) series, so the published
            # approximation is used for the OOS gating basis.
            matrix_deflation = None

    sharpe = annualized_sharpe(strategy, periods_per_year)
    psr = probabilistic_sharpe_ratio(strategy, 0.0)

    # The deflation benchmark SR*: measured from the trials matrix when it was
    # the whole search (Lopez de Prado & Lewis, 2019 -- cross-trial Sharpe
    # dispersion across the effective, i.e. cluster-counted, trials), else the
    # published approximation from the judged series' own Sharpe standard
    # error and the assumed trial count. The DSR and MinTRL below share this
    # SR*, which is what keeps T >= MinTRL equivalent to the DSR gate.
    sigma_sr = sharpe_standard_error(strategy)  # inf on a degenerate record
    if matrix_deflation is None:
        deflation_benchmark = expected_max_sharpe_benchmark(sigma_sr, n_trials_eff)
    elif matrix_deflation.cross_trial_std is None:
        deflation_benchmark = 0.0  # one effective trial: no selection inflation
    else:
        deflation_benchmark = expected_max_sharpe_benchmark(
            matrix_deflation.cross_trial_std, matrix_deflation.effective_trials
        )
    dsr = deflated_sharpe_ratio(strategy, sr_benchmark=deflation_benchmark)

    # Evidence gap: MinTRL against the same SR* the deflation used, at the policy
    # confidence -- so T >= MinTRL if and only if the DSR gate clears -- plus the
    # MinBTL for the observed annualised Sharpe and the size of the search.
    # Count only finite observations: every statistic above drops NaN/inf
    # entries, so the surfaced count must be the same T the thresholds were
    # computed from -- otherwise blank rows would be credited as evidence and
    # the shortfall arithmetic (and the MinTRL comparison) would be wrong.
    n_obs = int(np.count_nonzero(np.isfinite(np.asarray(strategy, dtype=np.float64))))
    conf = thr.min_deflated_sharpe
    if conf >= 1.0:
        min_trl = float("inf")  # Phi^-1(1) is infinite: no finite record suffices
    elif conf <= 0.0:
        min_trl = 0.0  # a non-positive bar is met by any record
    else:
        min_trl = minimum_track_record_length(strategy, deflation_benchmark, confidence=conf)
    # MinBTL stays the published bound for the *assumed* number of trials: the
    # pseudo-mathematics formula is defined for the count of trials tried, and
    # the effective-trials refinement above applies to the DSR benchmark only.
    min_btl = minimum_backtest_length(sharpe, n_trials_eff)
    observed_years = n_obs / periods_per_year

    reasons: list[str] = []
    fail = False

    if oos_unavailable:
        # Default-deny: the requested basis could not be produced, so there is no
        # honest verdict to give. Refuse rather than answer a different question.
        fail = True
        reasons.append(
            "Out-of-sample gating was requested (predictions and targets were "
            "supplied) but the purged walk-forward produced no usable fold, so "
            "there is no out-of-sample series to judge. NOT deployable on that "
            "basis alone: every figure below is measured on the IN-SAMPLE series "
            "and is not evidence about held-out performance. Supply more "
            "observations, or a splitter with a shorter embargo or fewer folds."
        )

    if matrix_deflation is not None:
        if matrix_deflation.cross_trial_std is None:
            reasons.append(
                f"Matrix-faithful deflation: the {n_trials_eff} supplied configurations "
                "are effectively one trial (every column is a near-duplicate or "
                "correlated variant of the others), so no selection deflation applies "
                "(Lopez de Prado & Lewis, 2019). The matrix is trusted as the WHOLE "
                "search: if these columns are only the winner plus variants from a "
                "larger search, this measurement cannot see that - pass n_trials with "
                "the true number of configurations tried."
            )
        else:
            reasons.append(
                "Matrix-faithful deflation: the DSR benchmark is measured from the "
                f"trials matrix - cross-trial Sharpe dispersion "
                f"{matrix_deflation.cross_trial_std:.4f} per period across "
                f"{matrix_deflation.effective_trials} effective trials (correlation "
                f"clusters among the {n_trials_eff} configurations) - per Lopez de "
                "Prado & Lewis (2019)."
            )
            # Near-zero measured dispersion: under the null of genuinely
            # independent trials the cross-trial Sharpe dispersion is
            # approximately the Sharpe estimator noise of a single trial, so a
            # measured dispersion well below that level means the surviving
            # clusters are nearly interchangeable (or the few-cluster estimate
            # is noisy) and the measured deflation is weak. Say so: the matrix
            # is trusted as the whole search, and this is where that trust is
            # cheapest to abuse (submit the winner plus variants only).
            if (
                math.isfinite(sigma_sr)
                and matrix_deflation.cross_trial_std
                < _NEAR_ZERO_DISPERSION_FRACTION * sigma_sr
            ):
                reasons.append(
                    "Caveat: the measured cross-trial dispersion "
                    f"({matrix_deflation.cross_trial_std:.3g} per period) is well below "
                    f"the selected series' own Sharpe estimator noise ({sigma_sr:.3g}), "
                    "so the supplied configurations are nearly interchangeable and the "
                    "measured deflation is weak. The matrix is trusted as the WHOLE "
                    "search - if it holds only a subset of what was tried, pass "
                    "n_trials with the true search size."
                )

    if dsr < thr.min_deflated_sharpe:
        fail = True
        reasons.append(
            f"Deflated Sharpe {dsr:.3f} < {thr.min_deflated_sharpe:.2f}: the Sharpe is not "
            "significant after deflating for trials and non-normality."
        )
        if math.isinf(min_trl):
            # Say *why* MinTRL is infinite: a policy bar of 1, a record too
            # degenerate to measure, or a Sharpe below the benchmark are three
            # different situations and only the last involves the benchmark.
            if conf >= 1.0:
                reasons.append(
                    f"Evidence gap: a confidence bar of {conf:.2f} can never be "
                    "reached by any finite track record."
                )
            elif math.isinf(sigma_sr):
                reasons.append(
                    "Evidence gap: the record is too degenerate to measure (fewer "
                    "than four finite observations, zero variance, or extreme "
                    "moments), so no required track record length can be quoted."
                )
            else:
                reasons.append(
                    f"Evidence gap: no track record length reaches the {conf:.2f} bar at "
                    "the observed moments (the observed Sharpe does not exceed the "
                    "deflation benchmark)."
                )
        else:
            needed = math.ceil(min_trl)
            short = max(needed - n_obs, 0)
            reasons.append(
                f"Evidence gap: {needed} observations are needed at the observed moments "
                f"to reach the {conf:.2f} bar; the record has {n_obs} - short about "
                f"{short} observations (~{short / periods_per_year:.1f} years)."
            )
        if n_trials_eff > 1 and math.isfinite(min_btl) and observed_years < min_btl:
            reasons.append(
                f"MinBTL: {min_btl:.1f} years of backtest are needed for the observed Sharpe "
                f"to clear the expected best of {n_trials_eff} noise trials; the record "
                f"spans {observed_years:.1f} years."
            )
    elif math.isfinite(min_trl):
        reasons.append(
            f"MinTRL confirmed: {n_obs} observations against a minimum of "
            f"{math.ceil(min_trl)} for the {conf:.2f} bar at the observed moments."
        )
    if sharpe <= thr.min_sharpe:
        fail = True
        reasons.append(
            f"Annualised Sharpe {sharpe:.3f} <= {thr.min_sharpe:.2f}: insufficient "
            "risk-adjusted return."
        )
    pbo_high = math.isfinite(pbo) and pbo > thr.max_pbo
    if pbo_high:
        fail = True
        reasons.append(
            f"PBO {pbo:.3f} > {thr.max_pbo:.2f}: the in-sample-best configuration tends to "
            "land in the worse out-of-sample half (selection is overfit)."
        )

    deployable = not fail
    if deployable:
        classification: Classification = "DEPLOYABLE"
        reasons.append("Clears the deflated-Sharpe, Sharpe and PBO bars.")
    elif pbo_high or (sharpe > thr.min_sharpe and dsr < thr.min_deflated_sharpe):
        # Looked good in sample but does not survive deflation / CSCV.
        classification = "PROBABLY_OVERFIT"
    else:
        classification = "NOT_DEPLOYABLE"

    return Verdict(
        deployable=deployable,
        classification=classification,
        deflated_sharpe=dsr,
        pbo=pbo,
        sharpe=sharpe,
        n_trials=n_trials_eff,
        reasons=reasons,
        probabilistic_sharpe=psr,
        n_periods=n_obs,
        periods_per_year=periods_per_year,
        min_track_record=min_trl,
        min_backtest_years=min_btl,
        effective_trials=(
            matrix_deflation.effective_trials if matrix_deflation is not None else None
        ),
        cross_trial_sharpe_std=(
            matrix_deflation.cross_trial_std if matrix_deflation is not None else None
        ),
        oos_sharpe=oos_sharpe,
        oos_information_coefficient=oos_ic,
    )
