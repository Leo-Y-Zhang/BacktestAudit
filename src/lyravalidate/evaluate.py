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
    information_coefficient,
    probabilistic_sharpe_ratio,
    probability_of_backtest_overfitting,
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
    n_periods: int = 0
    periods_per_year: int = 252
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

    reasons: list[str] = []
    fail = False

    if dsr < thr.min_deflated_sharpe:
        fail = True
        reasons.append(
            f"Deflated Sharpe {dsr:.3f} < {thr.min_deflated_sharpe:.2f}: the Sharpe is not "
            "significant after deflating for trials and non-normality."
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
        n_periods=int(np.asarray(strategy).size),
        periods_per_year=periods_per_year,
        oos_sharpe=oos_sharpe,
        oos_information_coefficient=oos_ic,
    )
