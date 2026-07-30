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
) -> tuple[npt.NDArray[np.float64], float, int]:
    """Reduce raw ``returns`` to ``(strategy_series, pbo, n_trials)``.

    A 2-D input is treated as a candidate matrix (``T`` periods x ``N`` configs):
    PBO is computed across the columns and the column with the highest full-sample
    annualised Sharpe is taken as the *selected* strategy, with ``n_trials``
    defaulting to ``N`` so the deflation reflects the search. A 1-D input is a
    single strategy with no PBO (``NaN``) and ``n_trials`` defaulting to ``1``.
    """
    arr: npt.NDArray[np.float64] = np.asarray(returns, dtype=np.float64)
    if arr.ndim == 1:
        return arr, float("nan"), (n_trials if n_trials is not None else 1)
    if arr.ndim == 2:
        if arr.shape[1] <= 1:
            return arr.ravel(), float("nan"), (n_trials if n_trials is not None else 1)
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
        return arr[:, best], pbo, (n_trials if n_trials is not None else N)
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
        across columns and the best in-sample column is judged).
    predictions, targets:
        Optional paired per-period signal scores and the forward returns they aim
        to predict. When supplied, an honest purged walk-forward OOS series is
        built and used as the basis for the Sharpe / deflated-Sharpe gates.
    n_trials:
        Number of configurations tried during research (selection-bias count).
        Defaults to ``N`` for a candidate matrix, else ``1``.
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

    strategy, pbo, n_trials_eff = _resolve_strategy(
        returns,
        n_trials=n_trials,
        pbo_splits=pbo_splits,
        periods_per_year=periods_per_year,
    )

    oos_sharpe: float | None = None
    oos_ic: float | None = None
    if predictions is not None and targets is not None:
        spl = splitter or default_walk_forward_splitter(len(np.asarray(targets).ravel()))
        oos = _walk_forward_oos(predictions, targets, spl)
        if oos is not None:
            strategy = oos.returns  # the OOS series is the honest basis for gating
            oos_sharpe = annualized_sharpe(strategy, periods_per_year)
            oos_ic = oos.mean_ic

    sharpe = annualized_sharpe(strategy, periods_per_year)
    dsr = deflated_sharpe_ratio(strategy, n_trials_eff)
    psr = probabilistic_sharpe_ratio(strategy, 0.0)

    # Evidence gap: MinTRL against the same SR* the deflation used, at the policy
    # confidence -- so T >= MinTRL if and only if the DSR gate clears -- plus the
    # MinBTL for the observed annualised Sharpe and the size of the search.
    # Count only finite observations: every statistic above drops NaN/inf
    # entries, so the surfaced count must be the same T the thresholds were
    # computed from -- otherwise blank rows would be credited as evidence and
    # the shortfall arithmetic (and the MinTRL comparison) would be wrong.
    n_obs = int(np.count_nonzero(np.isfinite(np.asarray(strategy, dtype=np.float64))))
    conf = thr.min_deflated_sharpe
    sigma_sr = sharpe_standard_error(strategy)  # inf on a degenerate record
    deflation_benchmark = expected_max_sharpe_benchmark(sigma_sr, n_trials_eff)
    if conf >= 1.0:
        min_trl = float("inf")  # Phi^-1(1) is infinite: no finite record suffices
    elif conf <= 0.0:
        min_trl = 0.0  # a non-positive bar is met by any record
    else:
        min_trl = minimum_track_record_length(strategy, deflation_benchmark, confidence=conf)
    min_btl = minimum_backtest_length(sharpe, n_trials_eff)
    observed_years = n_obs / periods_per_year

    reasons: list[str] = []
    fail = False

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
        oos_sharpe=oos_sharpe,
        oos_information_coefficient=oos_ic,
    )
