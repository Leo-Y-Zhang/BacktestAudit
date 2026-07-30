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
from scipy.stats import kurtosis, norm, rankdata, skew

__all__ = [
    "EULER_MASCHERONI",
    "annualized_sharpe",
    "deflated_sharpe_ratio",
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
    if sr_var <= 0.0:  # extreme skew/kurtosis can drive the estimator variance <= 0
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
    :func:`lyravalidate.evaluate.evaluate` lets the search be measured directly
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
    which MinTRL is the exact algebraic inverse in the sample length:
    ``T >= MinTRL`` if and only if ``PSR(sr_benchmark) >= confidence``.

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
           rather than waved through. :func:`lyravalidate.evaluate.evaluate`
           does exactly this.

    Returns
    -------
    float
        Estimated PBO in ``[0, 1]`` (or ``degenerate_value``).
    """
    M: FloatArray = np.asarray(performance, dtype=np.float64)
    if M.ndim != 2 or M.shape[1] < 2 or M.shape[0] < _MIN_OBS:
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
