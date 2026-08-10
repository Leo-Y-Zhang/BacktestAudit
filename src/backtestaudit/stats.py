"""Published backtest-validation statistics, re-implemented cleanly.

Every function in this module is *measurement* code: it tells you how much to
trust a track record, not how to make money. All of the harder statistics are
public, peer-reviewed mathematics and are cited inline:

* Probabilistic Sharpe Ratio (PSR) -- Bailey & Lopez de Prado (2012),
  "The Sharpe Ratio Efficient Frontier", *Journal of Risk*.
* Deflated Sharpe Ratio (DSR) -- Bailey & Lopez de Prado (2014),
  "The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest
  Overfitting, and Non-Normality", *Journal of Portfolio Management*.
* Probability of Backtest Overfitting (PBO) via Combinatorially-Symmetric
  Cross-Validation (CSCV) -- Bailey, Borwein, Lopez de Prado & Zhu (2017),
  "The Probability of Backtest Overfitting", *Journal of Computational Finance*.
* Minimum Track Record Length (MinTRL) -- Bailey & Lopez de Prado (2012),
  "The Sharpe Ratio Efficient Frontier", *Journal of Risk*.
* Minimum Backtest Length (MinBTL) -- Bailey, Borwein, Lopez de Prado & Zhu
  (2014), "Pseudo-Mathematics and Financial Charlatanism: The Effects of
  Backtest Overfitting on Out-of-Sample Performance", *Notices of the AMS* 61(5).
* Effective trials and cross-trial Sharpe dispersion for the matrix-faithful
  DSR -- Lopez de Prado & Lewis (2019), "Detection of False Investment
  Strategies Using Unsupervised Learning Methods", *Quantitative Finance* 19(9);
  silhouette scores per Rousseeuw (1987), with the Kaufman & Rousseeuw (1990)
  mean-silhouette bound as the no-structure guard.

Numerical conventions are chosen to match those papers exactly: per-period
(non-annualised) Sharpe inside PSR/DSR, *non-excess* kurtosis (normal == 3),
biased Fisher-Pearson skew, the Euler-Mascheroni constant in the expected-maximum
Sharpe benchmark, natural-log logits and ``rank / (N + 1)`` in CSCV. Degenerate
inputs fail closed: probabilities return ``0.0`` and required-evidence /
uncertainty statistics (MinTRL, MinBTL, the Sharpe standard error) return
``inf``, so that "not enough evidence" can never be mistaken for "significant".
"""

from __future__ import annotations

import math
from itertools import combinations

import numpy as np
import numpy.typing as npt
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from scipy.stats import kurtosis, norm, rankdata, skew

__all__ = [
    "EULER_MASCHERONI",
    "annualized_sharpe",
    "cluster_trials",
    "cross_trial_sharpe_std",
    "deflated_sharpe_ratio",
    "deflated_sharpe_ratio_from_trials",
    "effective_trials",
    "expected_max_sharpe_benchmark",
    "hit_rate",
    "information_coefficient",
    "max_drawdown",
    "minimum_backtest_length",
    "minimum_track_record_length",
    "probabilistic_sharpe_ratio",
    "probability_of_backtest_overfitting",
    "rank_information_coefficient",
    "sharpe_ratio",
    "sharpe_standard_error",
]

FloatArray = npt.NDArray[np.float64]

# Euler-Mascheroni constant, used in the expected-maximum-Sharpe benchmark of
# the Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2014).
EULER_MASCHERONI: float = 0.5772156649015329

_MIN_OBS = 4  # PSR/DSR need >= 4 finite observations to be meaningful.


def _clean(returns: npt.ArrayLike) -> FloatArray:
    """Coerce to a finite 1-D float array (drops NaN/inf)."""
    arr: FloatArray = np.asarray(returns, dtype=np.float64).ravel()
    finite: FloatArray = arr[np.isfinite(arr)]
    return finite


def _sharpe_moments(returns: npt.ArrayLike) -> tuple[int, float, float] | None:
    """Return ``(T, SR, sigma_SR)`` or ``None`` for degenerate input.

    ``SR`` is the *per-period* (non-annualised) Sharpe ratio and ``sigma_SR`` is
    the standard error of that estimator under the non-normal correction of
    Bailey & Lopez de Prado (2012), eq. for ``hat sigma(SR)``::

        sigma_SR^2 = (1 - g3*SR + (g4 - 1)/4 * SR^2) / (T - 1)

    where ``g3`` is the (biased) skew and ``g4`` is the *non-excess* kurtosis.
    """
    r = _clean(returns)
    T = int(r.size)
    if T < _MIN_OBS:
        return None
    sd = float(np.std(r, ddof=1))
    if sd <= 0.0:
        return None
    SR = float(np.mean(r) / sd)
    g3 = float(skew(r, bias=True))
    g4 = float(kurtosis(r, fisher=False))  # non-excess: a normal sample == 3
    sr_var = (1.0 - g3 * SR + (g4 - 1.0) / 4.0 * SR * SR) / (T - 1)
    # Extreme skew/kurtosis can drive the estimator variance <= 0, and on
    # near-constant data (catastrophic cancellation) scipy's moments go NaN --
    # a NaN would slip through a bare <= comparison, so demand finite-positive.
    if not math.isfinite(sr_var) or sr_var <= 0.0:
        return None
    return T, SR, math.sqrt(sr_var)


# ── per-period / annualised Sharpe and drawdown helpers ───────────────────────


def sharpe_ratio(returns: npt.ArrayLike) -> float:
    """Per-period Sharpe ratio ``mean / std`` (sample std, ``ddof=1``).

    Returns ``0.0`` for fewer than two finite observations or zero variance.
    """
    r = _clean(returns)
    if r.size < 2:
        return 0.0
    sd = float(np.std(r, ddof=1))
    if sd <= 0.0:
        return 0.0
    return float(np.mean(r) / sd)


def annualized_sharpe(returns: npt.ArrayLike, periods_per_year: int = 252) -> float:
    """Annualised Sharpe ratio = per-period Sharpe * ``sqrt(periods_per_year)``."""
    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be > 0")
    return float(sharpe_ratio(returns) * math.sqrt(periods_per_year))


def max_drawdown(returns: npt.ArrayLike) -> float:
    """Maximum drawdown of the compounded equity curve, as a positive fraction.

    ``0.0`` means no drawdown (or insufficient data). The returns are treated as
    simple per-period returns and compounded as ``cumprod(1 + r)``.
    """
    r = _clean(returns)
    if r.size == 0:
        return 0.0
    equity = np.cumprod(1.0 + r)
    peak = np.maximum.accumulate(equity)
    drawdown = equity / peak - 1.0
    return float(-np.min(drawdown))


def hit_rate(returns: npt.ArrayLike) -> float:
    """Fraction of finite observations that are strictly positive."""
    r = _clean(returns)
    if r.size == 0:
        return 0.0
    return float(np.mean(r > 0.0))


# ── information coefficients ──────────────────────────────────────────────────


def information_coefficient(predictions: npt.ArrayLike, targets: npt.ArrayLike) -> float:
    """Pearson correlation (the "information coefficient") of preds vs targets.

    Returns ``0.0`` when fewer than two paired finite points remain or either
    series is constant.
    """
    a = np.asarray(predictions, dtype=np.float64).ravel()
    b = np.asarray(targets, dtype=np.float64).ravel()
    if a.size != b.size:
        raise ValueError("predictions and targets must have the same length")
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    if a.size < 2 or float(np.std(a)) == 0.0 or float(np.std(b)) == 0.0:
        return 0.0
    corr = float(np.corrcoef(a, b)[0, 1])
    return corr if math.isfinite(corr) else 0.0


def rank_information_coefficient(predictions: npt.ArrayLike, targets: npt.ArrayLike) -> float:
    """Spearman rank correlation of predictions vs targets (tie-aware ranks)."""
    a = np.asarray(predictions, dtype=np.float64).ravel()
    b = np.asarray(targets, dtype=np.float64).ravel()
    if a.size != b.size:
        raise ValueError("predictions and targets must have the same length")
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    if a.size < 2:
        return 0.0
    ranked_a: FloatArray = np.asarray(rankdata(a), dtype=np.float64)
    ranked_b: FloatArray = np.asarray(rankdata(b), dtype=np.float64)
    return information_coefficient(ranked_a, ranked_b)


# ── Probabilistic & Deflated Sharpe Ratios ────────────────────────────────────


def sharpe_standard_error(returns: npt.ArrayLike) -> float:
    """Standard error of the Sharpe estimator (Bailey & Lopez de Prado, 2012).

    The ``hat sigma(SR)`` of "The Sharpe Ratio Efficient Frontier" under
    non-normal returns::

        sigma_SR = sqrt[ (1 - g3*SR + (g4 - 1)/4 * SR^2) / (T - 1) ]

    with ``g3`` the (biased) skew and ``g4`` the *non-excess* kurtosis. This is
    the denominator inside the PSR and the per-trial dispersion used by the DSR
    benchmark (see :func:`expected_max_sharpe_benchmark`).

    Returns
    -------
    float
        The standard error of the per-period Sharpe estimate; ``inf``
        (fail-closed: no information) on degenerate input -- returning ``0.0``
        here would claim perfect certainty and make any Sharpe look significant.
    """
    moments = _sharpe_moments(returns)
    if moments is None:
        return float("inf")
    return moments[2]


def probabilistic_sharpe_ratio(returns: npt.ArrayLike, sr_benchmark: float = 0.0) -> float:
    """Probabilistic Sharpe Ratio (Bailey & Lopez de Prado, 2012).

    The probability, in ``[0, 1]``, that the strategy's *true* (per-period)
    Sharpe ratio exceeds ``sr_benchmark``, given the observed track record and
    its higher moments::

        PSR(SR*) = Phi[ (SR - SR*) * sqrt(T - 1)
                        / sqrt(1 - g3*SR + (g4 - 1)/4 * SR^2) ]

    where ``Phi`` is the standard-normal CDF. This is the inner kernel of the
    Deflated Sharpe Ratio; with ``sr_benchmark == 0`` it is the probability the
    Sharpe is positive.

    Parameters
    ----------
    returns:
        Per-period (NOT annualised) returns. Non-finite entries are dropped.
    sr_benchmark:
        Per-period Sharpe to test against (``SR*``). Defaults to ``0.0``.

    Returns
    -------
    float
        Probability in ``[0, 1]``; ``0.0`` (fail-closed) on degenerate input
        (fewer than four observations, zero variance, or non-positive estimator
        variance from extreme skew/kurtosis).
    """
    moments = _sharpe_moments(returns)
    if moments is None:
        return 0.0
    _T, SR, sigma = moments
    return float(norm.cdf((SR - float(sr_benchmark)) / sigma))


def expected_max_sharpe_benchmark(sigma: float, n_trials: int) -> float:
    """Expected maximum of ``n_trials`` independent null Sharpe estimates.

    This is the ``SR*`` term of the Deflated Sharpe Ratio (Bailey & Lopez de
    Prado, 2014): the Sharpe you would expect to see by chance as the best of
    ``n_trials`` configurations, each with Sharpe-estimator standard error
    ``sigma``::

        SR* = sigma * [ (1 - gamma) * Z^-1(1 - 1/N) + gamma * Z^-1(1 - 1/(N e)) ]

    with ``gamma`` the Euler-Mascheroni constant and ``Z^-1`` the standard-normal
    quantile function. For ``n_trials <= 1`` there is no selection inflation, so
    ``SR* = 0`` and the DSR collapses to the PSR against a zero benchmark.

    Honesty note
    ------------
    The DSR benchmark formally requires the *cross-trial* dispersion of the
    Sharpe estimates across the configurations that were searched. When only a
    single realised returns series is supplied (the common case), that dispersion
    is not observable, so ``sigma`` is taken to be the strategy's own
    Sharpe-estimator standard error -- which, under the null hypothesis of no
    skill, is the standard approximation to the per-trial Sharpe variance (Bailey
    & Lopez de Prado, 2014). Supplying a ``T x N`` candidate matrix to
    :func:`backtestaudit.evaluate.evaluate` lets the search be measured directly
    (via PBO) rather than only approximated here.
    """
    n = max(int(n_trials), 1)
    if n <= 1:
        return 0.0
    z1 = float(norm.ppf(1.0 - 1.0 / n))
    z2 = float(norm.ppf(1.0 - 1.0 / (n * math.e)))
    return float(sigma * ((1.0 - EULER_MASCHERONI) * z1 + EULER_MASCHERONI * z2))


def deflated_sharpe_ratio(
    returns: npt.ArrayLike,
    n_trials: int = 1,
    sr_benchmark: float | None = None,
) -> float:
    """Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2014).

    The probability, in ``[0, 1]``, that the strategy's *true* Sharpe exceeds a
    benchmark that accounts for (a) sample length, (b) non-normality of returns
    (skew/kurtosis) and (c) the number of configurations tried during research
    (multiple-testing / selection bias). It is exactly ``PSR(SR*)`` with ``SR*``
    set to the expected maximum Sharpe under the null (see
    :func:`expected_max_sharpe_benchmark`).

    Parameters
    ----------
    returns:
        Per-period (NOT annualised) returns.
    n_trials:
        Number of strategy *configurations* tried during research (not folds).
        A smaller count is a conservative lower bound. Floored to ``1``.
    sr_benchmark:
        If given, used directly as ``SR*`` (overriding the multiple-testing
        formula). If ``None`` (default), ``SR*`` is derived from ``n_trials``.

    Returns
    -------
    float
        Probability in ``[0, 1]``; ``0.0`` (fail-closed) on degenerate input.
        A common "significant" cutoff is ``DSR >= 0.95``.
    """
    moments = _sharpe_moments(returns)
    if moments is None:
        return 0.0
    _T, _SR, sigma = moments
    benchmark = (
        expected_max_sharpe_benchmark(sigma, n_trials)
        if sr_benchmark is None
        else float(sr_benchmark)
    )
    # Single source of truth for the final probability: route through PSR.
    return probabilistic_sharpe_ratio(returns, benchmark)


# ── Minimum Track Record / Backtest Length ────────────────────────────────────


def minimum_track_record_length(
    returns: npt.ArrayLike,
    sr_benchmark: float = 0.0,
    confidence: float = 0.95,
) -> float:
    """Minimum Track Record Length (Bailey & Lopez de Prado, 2012).

    The closed-form MinTRL of "The Sharpe Ratio Efficient Frontier": the number
    of observations at which the PSR of the observed Sharpe against
    ``sr_benchmark`` reaches ``confidence``, holding the observed per-period
    Sharpe, skew and kurtosis fixed::

        MinTRL = 1 + (1 - g3*SR + (g4 - 1)/4 * SR^2) * (Z_a / (SR - SR*))^2

    where ``Z_a = Phi^-1(confidence)``, ``g3`` is the (biased) skew and ``g4``
    the *non-excess* kurtosis -- the same variance formula the PSR uses, of
    which MinTRL is the exact algebraic inverse in the sample length: for any
    ``confidence > 0.5`` (every realistic significance bar, including the 0.95
    default), ``T >= MinTRL`` if and only if
    ``PSR(sr_benchmark) >= confidence``. For ``confidence <= 0.5`` the
    non-positive normal quantile breaks the equivalence: a record whose Sharpe
    does not exceed ``sr_benchmark`` fails closed to ``inf`` here even though
    its PSR (at most one half in that regime) can still meet such a bar.

    Parameters
    ----------
    returns:
        Per-period (NOT annualised) returns. Non-finite entries are dropped.
    sr_benchmark:
        Per-period Sharpe to test against (``SR*``). Defaults to ``0.0``.
    confidence:
        Required PSR level, strictly inside ``(0, 1)``; ``0.95`` mirrors the
        conventional 5% significance level.

    Returns
    -------
    float
        Required number of observations (may be fractional; take the ceiling
        for a whole-observation requirement). ``inf`` (fail-closed) when the
        observed Sharpe does not exceed ``sr_benchmark`` -- no track record
        length would ever reach the bar at these moments -- or on degenerate
        input.
    """
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be strictly between 0 and 1")
    moments = _sharpe_moments(returns)
    if moments is None:
        return float("inf")
    T, SR, sigma = moments
    excess = SR - float(sr_benchmark)
    if excess <= 0.0:
        return float("inf")
    variance_numerator = sigma * sigma * (T - 1)  # 1 - g3*SR + (g4 - 1)/4 * SR^2
    z = float(norm.ppf(confidence))
    return float(1.0 + variance_numerator * (z / excess) ** 2)


def minimum_backtest_length(sharpe: float, n_trials: int) -> float:
    """Minimum Backtest Length in years (Bailey, Borwein, Lopez de Prado & Zhu, 2014).

    The MinBTL of "Pseudo-Mathematics and Financial Charlatanism: The Effects
    of Backtest Overfitting on Out-of-Sample Performance" (*Notices of the AMS*
    61(5)): the backtest length below which the expected maximum annualised
    Sharpe among ``n_trials`` independent trials of pure noise meets or exceeds
    the observed annualised Sharpe -- i.e. the record is too short for the
    result to be distinguishable from the best of a noise search::

        MinBTL = [ ((1 - gamma) * Z^-1(1 - 1/N) + gamma * Z^-1(1 - 1/(N e)))
                   / SR ]^2

    with ``gamma`` the Euler-Mascheroni constant. The unit is *years* because
    the null annualised-Sharpe estimate over ``y`` years has standard deviation
    approximately ``1 / sqrt(y)``.

    Parameters
    ----------
    sharpe:
        Observed *annualised* Sharpe ratio (note: annualised, unlike the
        per-period convention inside PSR/DSR -- this matches the paper).
    n_trials:
        Number of strategy configurations tried during research. Floored to
        ``1``.

    Returns
    -------
    float
        Minimum backtest length in years. ``0.0`` for ``n_trials <= 1`` (no
        selection took place, so no multiple-testing minimum applies); ``inf``
        (fail-closed) when ``sharpe <= 0`` -- a non-positive Sharpe can never
        exceed the expected best of two or more noise trials.
    """
    n = max(int(n_trials), 1)
    if n <= 1:
        return 0.0
    if not math.isfinite(sharpe) or sharpe <= 0.0:
        return float("inf")
    # (1-gamma)*Z^-1(1-1/N) + gamma*Z^-1(1-1/(N e)) == the benchmark at sigma=1.
    factor = expected_max_sharpe_benchmark(1.0, n)
    return float((factor / float(sharpe)) ** 2)


# ── Effective trials / matrix-faithful DSR (Lopez de Prado & Lewis, 2019) ─────

# Kaufman & Rousseeuw (1990), "Finding Groups in Data": a mean silhouette at or
# below 0.25 means no substantial clustering structure has been found. Applied
# *per cluster*: a multi-member cluster whose own mean silhouette does not
# clear this bound has not demonstrated cohesion and is split back into
# singleton trials -- the conservative direction, since fewer clusters would
# weaken the deflation. Measured margins for this guard (calibration run,
# 2026-07-30, 10 seeds per shape, final algorithm): on iid-noise matrices
# (T in 100..750, N in 5..60) no multi-member cluster of the winning partition
# ever exceeded a cohesion of 0.188, while planted correlated families never
# fell below 0.384 at pairwise rho 0.7 (0.639 at rho 0.9), and the effective
# count was recovered 10/10 for every noise, family (rho >= 0.7) and mixed
# family-plus-independents shape probed.
_NO_STRUCTURE_SILHOUETTE = 0.25

# Correlation distances at or below this are duplicates for clustering
# purposes. The bound is the distance of pairwise correlation 0.999,
# d = sqrt((1 - 0.999) / 2): columns sharing 99.9% of their correlation are
# re-parameterisations of one strategy, not distinct trials -- and on that
# scale the *relative* differences between distances are estimation noise, so
# letting the silhouette search "find structure" in them would manufacture
# trials out of float dust. (Sample correlation of exact copies is 1 only up
# to rounding, so an exact-equality test would miss near-copies entirely.)
_DUPLICATE_TRIAL_DISTANCE = math.sqrt((1.0 - 0.999) / 2.0)

# When the best k >= 2 partition shows no substantial structure (pooled mean
# silhouette at or below the Kaufman & Rousseeuw bound), the matrix is either
# mutually independent trials (split into singletons -- conservative) or one
# homogeneous correlated family (one cluster). The two are distinguished by
# the mean pairwise correlation: at or above this bound the columns are one
# family. 0.7 matches the calibrated region in which planted families are
# reliably recovered against independents (see the calibration note above);
# a homogeneous blob at mean correlation below 0.7 stays split into
# singletons -- the conservative direction, mirroring the "weak structure"
# band documented in :func:`cluster_trials`.
_HOMOGENEOUS_MEAN_CORRELATION = 0.7

# Measured clustering needs enough rows to trust the sample correlation
# matrix. On short records the silhouette search invents families on iid
# noise (measured 2026-07-31 on the unguarded algorithm, 20 seeds per length,
# N=10: T=8 recovered the true count 6/20, T=20 18/20, T=40 19/20, with clean
# recovery from T=60 in these probes), which *under*-counts the effective
# trials and weakens the deflation -- the fail-open direction. The floor is
# set at 100, the shortest length covered by the calibration runs, not at the
# last observed failure; complete rows must also exceed the column count,
# else the sample correlation matrix is rank-deficient and structure is
# guaranteed spurious. Shorter or wider matrices are "not measurable":
# callers fall back to the published raw-count deflation (fail-closed).
_MIN_CLUSTER_OBS = 100


def _prepared_trials(trials: npt.ArrayLike) -> tuple[FloatArray, list[int]] | None:
    """Reduce a trials matrix to complete-case rows over its usable columns.

    Returns ``(matrix, usable)`` where ``matrix`` holds only rows that are
    finite across every usable column (non-finite rows are not evidence, per
    the library-wide contract) and ``usable`` maps its columns back to the
    original column indices. A column is usable when its own finite
    observations yield valid Sharpe moments and it is not constant on the
    complete-case rows. ``None`` when fewer than two usable columns remain,
    or when the complete-case rows are too few to trust a measured
    correlation structure (fewer than ``_MIN_CLUSTER_OBS``, or not more than
    the number of usable columns -- a rank-deficient sample correlation
    matrix guarantees spurious structure).
    """
    M: FloatArray = np.asarray(trials, dtype=np.float64)
    if M.ndim != 2 or M.shape[1] < 2:
        return None
    usable = [j for j in range(M.shape[1]) if _sharpe_moments(M[:, j]) is not None]
    if len(usable) < 2:
        return None
    sub: FloatArray = M[:, usable]
    sub = sub[np.all(np.isfinite(sub), axis=1)]
    if sub.shape[0] < _MIN_CLUSTER_OBS:
        return None
    keep = np.std(sub, axis=0, ddof=1) > 0.0
    if int(np.count_nonzero(keep)) < 2:
        return None
    if not bool(np.all(keep)):
        usable = [j for j, kept in zip(usable, keep, strict=True) if kept]
        sub = sub[:, keep]
    if sub.shape[0] <= len(usable):
        return None
    return np.ascontiguousarray(sub), usable


def _silhouette_scores(D: FloatArray, labels: npt.NDArray[np.int_]) -> FloatArray:
    """Silhouette scores (Rousseeuw, 1987) from a precomputed distance matrix.

    Singleton clusters score 0 by convention, as does a point whose within- and
    between-cluster mean distances are both 0. Vectorised (one ``D @ onehot``
    matrix product instead of a Python loop over points), because the partition
    search evaluates this for every candidate ``k``: the loop form cost 13.9 s
    for one ``effective_trials`` call at ``N = 200`` (T = 500, measured
    2026-07-31 against 0.3 s vectorised) and grew roughly cubically in the
    column count.
    """
    n = int(labels.size)
    _uniq, inverse = np.unique(labels, return_inverse=True)
    k = int(_uniq.size)
    if k < 2:
        return np.zeros(n, dtype=np.float64)  # one cluster: no "between" exists
    onehot: FloatArray = np.zeros((n, k), dtype=np.float64)
    onehot[np.arange(n), inverse] = 1.0
    counts: FloatArray = onehot.sum(axis=0)  # cluster sizes, all >= 1
    # (n, k): summed distance from point i to the members of cluster c.
    sums: FloatArray = np.asarray(D @ onehot, dtype=np.float64)
    own = counts[inverse]  # own-cluster size per point
    own_sums = sums[np.arange(n), inverse]
    # a_i: mean distance to own cluster (D[i, i] == 0 so divide by size - 1).
    a: FloatArray = np.divide(own_sums, np.maximum(own - 1.0, 1.0))
    # b_i: smallest mean distance to any *other* cluster.
    means: FloatArray = np.asarray(sums / counts, dtype=np.float64)
    means[np.arange(n), inverse] = np.inf
    b: FloatArray = means.min(axis=1)
    denom: FloatArray = np.maximum(a, b)
    scores: FloatArray = np.zeros(n, dtype=np.float64)
    valid = (own > 1.0) & (denom > 0.0)
    scores[valid] = (b[valid] - a[valid]) / denom[valid]
    return scores


def _best_partition(D: FloatArray) -> npt.NDArray[np.int_] | None:
    """ONC-style cluster search over a correlation-distance matrix.

    A deterministic variant of the Optimal Number of Clusters scheme of Lopez
    de Prado & Lewis (2019): candidate partitions for ``k = 2 .. N-1`` come
    from average-linkage hierarchical clustering on ``d = sqrt((1 - rho) / 2)``
    (the paper's distance metric; its randomised k-means is replaced so runs
    are reproducible) and the partition with the highest mean silhouette wins
    (the standard silhouette-based model selection of Rousseeuw, 1987). The
    paper's t-statistic score ``E[s] / sqrt(V[s])`` is deliberately not used
    for selection: it rewards low silhouette *variance*, which on searches
    mixing one correlated family with independent trials prefers merging the
    family and stragglers into one diffuse cluster -- undercounting the
    effective trials, the fail-open direction (measured during calibration).
    Whether each cluster of the winning partition *survives* as a trial family
    is decided per cluster by the Kaufman & Rousseeuw cohesion bound in
    :func:`cluster_trials` -- the deterministic counterpart of the paper's
    recursive redo of low-quality clusters. ``None`` only for ``N < 3``, where
    no candidate ``k`` exists.
    """
    N = int(D.shape[0])
    if N < 3:
        return None
    condensed: FloatArray = squareform(D, checks=False)
    Z = linkage(condensed, method="average")
    best_labels: npt.NDArray[np.int_] | None = None
    best_mean = -math.inf
    for k in range(2, N):
        labels: npt.NDArray[np.int_] = fcluster(Z, t=k, criterion="maxclust")
        mean_s = float(np.mean(_silhouette_scores(D, labels)))
        if mean_s > best_mean:
            best_mean = mean_s
            best_labels = labels
    return best_labels


def cluster_trials(trials: npt.ArrayLike) -> list[list[int]]:
    """Group trial columns into correlation clusters (Lopez de Prado & Lewis, 2019).

    ``trials`` is a ``(T x N)`` matrix: ``T`` time periods (rows) by ``N``
    strategy configurations tried during research (columns). Correlated trials
    are not independent evidence of a search: this function recovers the
    families, so that :func:`effective_trials` can count them and
    :func:`cross_trial_sharpe_std` can measure the Sharpe dispersion across
    them. Correlations are taken over complete-case rows (non-finite rows are
    dropped; they are not evidence) among the usable columns.

    Four outcomes, in decreasing order of measured structure:

    * genuine correlation families -> one cluster per family. Every
      multi-member cluster must itself clear the Kaufman & Rousseeuw
      cohesion bound (see below), so a diffuse "everything else" cluster of
      mutually uncorrelated columns is split back into singletons rather
      than counted as one trial;
    * all columns effectively identical (every pairwise correlation at or
      above 0.999) -> a single cluster: the "search" only ever tried one
      distinct thing, and near-copies are copies for trial-counting purposes;
    * one homogeneous family (the best partition shows no substantial
      structure -- pooled mean silhouette at or below the Kaufman & Rousseeuw
      0.25 bound -- yet the mean pairwise correlation is at least 0.7) -> a
      single cluster: a parameter sweep around one idea is one trial, however
      many configurations it produced;
    * no accepted structure and no homogeneity -> every usable column is its
      own singleton cluster: the trials are effectively independent. This is
      the conservative direction -- fewer clusters would weaken the deflation.

    Weakly correlated families sit near the bounds by construction (Kaufman &
    Rousseeuw call 0.26-0.50 "weak" structure), so families around pairwise
    correlation 0.5 may be counted as separate trials -- again the
    conservative direction, never the optimistic one. Measured on planted
    single families (10 seeds per rho, T = 500, N = 12, 2026-07-31): one
    cluster 10/10 at pairwise rho 0.8, 0.9 and 0.95, while rho <= 0.6 split
    into singletons 10/10 (conservative); rho 0.7 sits exactly on the
    homogeneity bound and collapsed 6/10 (split 4/10, the conservative side
    of a sampling coin-flip).

    Returns
    -------
    list[list[int]]
        Clusters as sorted lists of *original* column indices, ordered by
        first member. Empty list when the matrix is unusable or too small to
        measure (not 2-D, fewer than two usable columns, fewer than 100
        complete rows, or no more complete rows than usable columns -- short
        records let the silhouette search invent families on independent
        noise, the fail-open direction, so they fail closed to "not
        measurable" and callers fall back to the raw trial count).
    """
    prepared = _prepared_trials(trials)
    if prepared is None:
        return []
    sub, usable = prepared
    corr: FloatArray = np.asarray(np.corrcoef(sub, rowvar=False), dtype=np.float64)
    if not bool(np.all(np.isfinite(corr))):
        return []
    D: FloatArray = np.sqrt(np.clip(0.5 * (1.0 - np.clip(corr, -1.0, 1.0)), 0.0, 1.0))
    np.fill_diagonal(D, 0.0)
    if float(np.max(D)) <= _DUPLICATE_TRIAL_DISTANCE:
        return [list(usable)]
    labels = _best_partition(D)
    if labels is None:
        return [[j] for j in usable]
    scores = _silhouette_scores(D, labels)
    # k = 1 is unreachable by the partition search (a silhouette needs two
    # clusters), so the all-one-family outcome is decided here: when even the
    # best k >= 2 partition shows no substantial structure (pooled mean
    # silhouette at or below the Kaufman & Rousseeuw bound) the matrix is
    # either mutually independent trials or one homogeneous blob that the
    # forced split sliced arbitrarily. The mean pairwise correlation tells
    # them apart: a homogeneous correlated family collapses to one cluster,
    # anything else falls through and is split into singletons below.
    off_diagonal = corr[~np.eye(corr.shape[0], dtype=bool)]
    if (
        float(np.mean(scores)) <= _NO_STRUCTURE_SILHOUETTE
        and float(np.mean(off_diagonal)) >= _HOMOGENEOUS_MEAN_CORRELATION
    ):
        return [list(usable)]
    # Per-cluster cohesion check -- the deterministic counterpart of the ONC
    # recursive redo of low-quality clusters: a multi-member cluster only
    # counts as one trial family when its own mean silhouette clears the same
    # Kaufman & Rousseeuw bound. Without this, mutually *uncorrelated* columns
    # that merely have no better home get lumped into one diffuse cluster,
    # undercounting the effective trials and weakening the deflation (the
    # fail-open direction). Splitting an uncohesive cluster into singletons
    # raises the trial count instead -- conservative.
    grouped: dict[int, list[int]] = {}
    for position, label in enumerate(labels):
        grouped.setdefault(int(label), []).append(position)
    clusters: list[list[int]] = []
    for positions in grouped.values():
        cohesive = (
            len(positions) < 2
            or float(np.mean(scores[positions])) > _NO_STRUCTURE_SILHOUETTE
        )
        if cohesive:
            clusters.append([usable[p] for p in positions])
        else:
            clusters.extend([usable[p]] for p in positions)
    return sorted(clusters)


def effective_trials(trials: npt.ArrayLike) -> int:
    """Effective number of independent trials in a search (Lopez de Prado & Lewis, 2019).

    The number of correlation clusters among the trial columns -- the ``E[K]``
    that belongs in the Deflated Sharpe Ratio's expected-maximum benchmark.
    Deflating by the raw column count treats every configuration as an
    independent draw, which overstates a search whose trials are correlated
    (near-duplicate parameterisations); counting clusters restores the
    assumption the benchmark formula actually makes.

    Returns
    -------
    int
        Number of clusters, in ``[1, N]``; ``0`` (a "not measurable" sentinel,
        never a trial count) when the matrix is unusable or has too few
        complete rows for a trustworthy correlation structure (see
        :func:`cluster_trials`) -- callers must then fall back to a trial
        count they can defend, e.g. the raw column count.
    """
    return len(cluster_trials(trials))


def cross_trial_sharpe_std(
    trials: npt.ArrayLike, clusters: list[list[int]] | None = None
) -> float:
    """Cross-trial dispersion of Sharpe estimates (Lopez de Prado & Lewis, 2019).

    The DSR's expected-maximum benchmark formally requires the standard
    deviation of the Sharpe estimates *across the trials of the search*. When
    the trials matrix is available, that dispersion is measurable directly:
    members of each correlation cluster are summed into one aggregate series
    per cluster (Sharpe is scale-invariant, so equal-weight summing is the
    same as averaging), and the standard deviation (``ddof=1``) of the
    per-period cluster Sharpes is returned.

    Parameters
    ----------
    trials:
        ``(T x N)`` matrix of per-period returns, one column per configuration
        tried. Rows that are non-finite for a cluster's members drop out of
        that cluster's aggregate (they are not evidence).
    clusters:
        Cluster assignment as lists of column indices (e.g. from
        :func:`cluster_trials`). Computed from ``trials`` when omitted. A
        degenerate aggregate (e.g. members that cancel) contributes a Sharpe
        of ``0.0``, per :func:`sharpe_ratio`. A supplied assignment must be a
        partition-like list: every member a valid column index in
        ``[0, N)``, no empty clusters, no column in two clusters
        (``ValueError`` otherwise -- a negative index would silently wrap to
        the wrong column and a duplicate would double-count it).

    Returns
    -------
    float
        Standard deviation of the per-period cluster Sharpes; ``inf``
        (fail-closed: no information) when fewer than two clusters exist --
        returning ``0.0`` would erase the deflation benchmark entirely.

    Raises
    ------
    ValueError
        If a supplied ``clusters`` assignment is malformed (out-of-range or
        negative index, empty cluster, or overlapping clusters).

    Honesty note
    ------------
    With few clusters this is a dispersion estimated from few points and is
    correspondingly noisy (a ``ddof=1`` standard deviation of ``K`` values).
    That is inherent to the published method, not a defect of the
    implementation; it is why :func:`backtestaudit.evaluate.evaluate` keeps the
    deflated Sharpe as one gate among three (Sharpe, PBO) rather than the sole
    arbiter.
    """
    M: FloatArray = np.asarray(trials, dtype=np.float64)
    if M.ndim != 2 or M.shape[1] < 1:
        return float("inf")
    if clusters is None:
        clusters = cluster_trials(M)
    else:
        seen: set[int] = set()
        for members in clusters:
            if not members:
                raise ValueError("clusters must not contain an empty cluster")
            for j in members:
                if not 0 <= j < M.shape[1]:
                    raise ValueError(
                        f"cluster member {j} is not a valid column index in [0, {M.shape[1]})"
                    )
                if j in seen:
                    raise ValueError(f"column {j} appears in more than one cluster")
                seen.add(j)
    if len(clusters) < 2:
        return float("inf")
    sharpes = [sharpe_ratio(np.sum(M[:, list(members)], axis=1)) for members in clusters]
    return float(np.std(np.asarray(sharpes, dtype=np.float64), ddof=1))


def deflated_sharpe_ratio_from_trials(
    trials: npt.ArrayLike, selected: npt.ArrayLike | None = None
) -> float:
    """Matrix-faithful Deflated Sharpe Ratio (Lopez de Prado & Lewis, 2019).

    The DSR of :func:`deflated_sharpe_ratio` approximates the null benchmark
    ``SR*`` from a single series' own Sharpe standard error and a caller-
    supplied trial count. When the actual trials matrix is available, both
    inputs are measurable instead of assumed: ``SR*`` is built from the
    *cross-trial* Sharpe dispersion (:func:`cross_trial_sharpe_std`) and the
    *effective* number of independent trials (:func:`effective_trials`), and
    the selected strategy's PSR is taken against that benchmark::

        SR* = sqrt(V{SR_k}) * [ (1 - gamma) Z^-1(1 - 1/K) + gamma Z^-1(1 - 1/(K e)) ]

    with ``K`` the number of correlation clusters and ``V{SR_k}`` the variance
    of the cluster-aggregate Sharpes. With one effective trial (all columns
    effectively identical) no selection took place and ``SR* = 0``: the DSR
    collapses to the PSR, exactly as ``n_trials=1`` does in the published
    approximation.

    Parameters
    ----------
    trials:
        ``(T x N)`` matrix of per-period returns for the configurations tried.
    selected:
        The series to judge. Defaults to the column of ``trials`` with the
        highest full-sample Sharpe (the same selection rule
        :func:`backtestaudit.evaluate.evaluate` applies to a candidate matrix).

    Returns
    -------
    float
        Probability in ``[0, 1]``; ``0.0`` (fail-closed) when the matrix is
        unusable or the cross-trial dispersion cannot be measured. Because the
        matrix-derived ``SR*`` is never negative, this can never exceed the
        undeflated PSR of the selected series.
    """
    M: FloatArray = np.asarray(trials, dtype=np.float64)
    clusters = cluster_trials(M)
    if not clusters:
        return 0.0
    if selected is None:
        best = int(np.argmax([sharpe_ratio(M[:, j]) for j in range(M.shape[1])]))
        selected = M[:, best]
    if len(clusters) == 1:
        benchmark = 0.0
    else:
        sigma = cross_trial_sharpe_std(M, clusters=clusters)
        if not math.isfinite(sigma):
            return 0.0
        benchmark = expected_max_sharpe_benchmark(sigma, len(clusters))
    return probabilistic_sharpe_ratio(selected, benchmark)


# ── Probability of Backtest Overfitting (CSCV) ────────────────────────────────


def probability_of_backtest_overfitting(
    performance: npt.ArrayLike,
    n_splits: int = 16,
    *,
    max_blocks: int = 16,
    degenerate_value: float = 0.0,
) -> float:
    """Probability of Backtest Overfitting via CSCV (Bailey et al., 2017).

    ``performance`` is a ``(T x N)`` matrix: ``T`` time periods (rows) by ``N``
    candidate configurations (columns), each cell a per-period performance
    figure (e.g. a return). CSCV splits the timeline into ``S`` equal contiguous
    blocks and, for every symmetric way of assigning ``S/2`` blocks to in-sample
    (IS) and the rest to out-of-sample (OOS):

    1. rank the configurations by summed IS performance and take the best,
    2. find that config's *relative* OOS rank ``w = rank / (N + 1)``,
    3. record the logit ``ln(w / (1 - w))``.

    PBO is the fraction of partitions whose logit ``<= 0`` -- i.e. the estimated
    probability that the IS-best configuration lands in the *worse* OOS half. A
    value near or above ``0.5`` means the selection process is overfitting.

    Parameters
    ----------
    performance:
        ``(T x N)`` performance matrix, ``N >= 2`` candidate configurations.
    n_splits:
        Target number ``S`` of contiguous time blocks; forced even and ``>= 2``.
    max_blocks:
        Upper bound on ``S`` to cap the combinatorial cost ``C(S, S/2)``
        (``C(16, 8) == 12870``). Forced even.
    degenerate_value:
        Value returned for degenerate input (non-2D, ``N < 2``, ``T < 4`` or no
        usable partitions). Defaults to ``0.0`` to match the published statistic.

        .. warning::
           ``0.0`` reads as "no overfitting", which is *optimistic*, not
           fail-closed. When PBO is used as a deployment gate, pass
           ``degenerate_value=1.0`` so that an unrankable input is rejected
           rather than waved through. :func:`backtestaudit.evaluate.evaluate`
           does exactly this.

    Returns
    -------
    float
        Estimated PBO in ``[0, 1]`` (or ``degenerate_value``).
    """
    M: FloatArray = np.asarray(performance, dtype=np.float64)
    if M.ndim != 2 or M.shape[1] < 2:
        return float(degenerate_value)
    # Drop periods that are not fully observed. That is what every other
    # statistic in this module does, and what Verdict documents in so many words:
    # "non-finite rows -- e.g. blank CSV cells -- are not evidence and are not
    # counted". This function did not, and one blank cell was enough to break it.
    # That column's summed IS performance becomes NaN, np.argmax treats NaN as
    # the maximum, so the NaN column is selected as the in-sample BEST in every
    # partition. Measured on a 200x6 matrix whose column 0 is the genuine winner,
    # a single blank cell in a mediocre column moved PBO from 0.000 to 0.423 and
    # made argmax pick that column instead of the real one. The gate does not
    # become more permissive -- a NaN pushes PBO up, never down -- but the number
    # stops describing the search it claims to describe, and a user with one gap
    # in a CSV is handed a measurement of nothing.
    finite_rows = np.isfinite(M).all(axis=1)
    if not bool(finite_rows.all()):
        M = M[finite_rows, :]
    if M.shape[0] < _MIN_OBS:
        return float(degenerate_value)
    T, N = int(M.shape[0]), int(M.shape[1])

    cap = max(2, int(max_blocks) - (int(max_blocks) % 2))
    S = max(2, int(n_splits) - (int(n_splits) % 2))  # even, >= 2
    S = min(S, cap)
    while S > 2 and T // S < 1:  # guarantee >= 1 row per block
        S -= 2

    blocks: list[FloatArray] = [M[idx, :] for idx in np.array_split(np.arange(T), S)]
    half = S // 2
    logits: list[float] = []
    for is_sel in combinations(range(S), half):
        in_set = set(is_sel)
        is_perf = np.concatenate([blocks[b] for b in range(S) if b in in_set]).sum(axis=0)
        oos_perf = np.concatenate([blocks[b] for b in range(S) if b not in in_set]).sum(axis=0)
        best = int(np.argmax(is_perf))
        oos_rank = int(np.sum(oos_perf <= oos_perf[best]))  # 1 = worst, N = best OOS
        w = min(max(oos_rank / (N + 1.0), 1e-6), 1.0 - 1e-6)
        logits.append(math.log(w / (1.0 - w)))

    if not logits:
        return float(degenerate_value)
    return float(np.mean(np.asarray(logits, dtype=np.float64) <= 0.0))
