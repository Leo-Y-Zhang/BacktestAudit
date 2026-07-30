"""Property-based invariants of the published statistics (Hypothesis).

Where the other test modules anchor on hand-computed numbers, this module
encodes the mathematical *contract* the library must honour for arbitrary
valid inputs:

* PSR, DSR and PBO are probabilities: always in ``[0, 1]``.
* PSR is monotonically non-increasing in the benchmark Sharpe.
* DSR is non-increasing in ``n_trials``, and ``DSR <= PSR`` for ``n_trials > 1``
  (deflation can only ever lower the probability).
* PBO is invariant under column permutation of the candidate matrix (CSCV must
  not care which order the configurations were tried in).
* The purged, embargoed walk-forward splitter never leaves a training index
  within ``label_horizon`` of the evaluation window, for arbitrary valid
  window parameters.

The Hypothesis profile is deterministic (``derandomize=True``) with bounded
example counts, so the suite stays fast and reproducible run-to-run.
"""

from __future__ import annotations

import math

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

from lyravalidate.crossval import PurgedWalkForwardSplitter
from lyravalidate.stats import (
    cluster_trials,
    cross_trial_sharpe_std,
    deflated_sharpe_ratio,
    deflated_sharpe_ratio_from_trials,
    effective_trials,
    probabilistic_sharpe_ratio,
    probability_of_backtest_overfitting,
    sharpe_ratio,
)

settings.register_profile(
    "lyravalidate",
    max_examples=40,
    derandomize=True,
    deadline=None,
)
settings.load_profile("lyravalidate")

# Per-period returns bounded to a generous +/-50%; NaN/inf are injected
# explicitly because the statistics are contracted to drop non-finite entries.
_return_value = st.one_of(
    st.floats(min_value=-0.5, max_value=0.5, allow_nan=False, allow_infinity=False),
    st.just(float("nan")),
    st.just(float("inf")),
    st.just(float("-inf")),
)
_returns_series = st.lists(_return_value, min_size=0, max_size=120)
_benchmark = st.floats(min_value=-3.0, max_value=3.0, allow_nan=False, allow_infinity=False)

# T x N candidate matrices for PBO (T >= 4 rows, N >= 2 configuration columns).
_candidate_matrix = hnp.arrays(
    dtype=np.float64,
    shape=st.tuples(st.integers(min_value=4, max_value=36), st.integers(min_value=2, max_value=6)),
    elements=st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
)


# ── probabilities stay probabilities ──────────────────────────────────────────


@given(returns=_returns_series, benchmark=_benchmark)
def test_psr_always_in_unit_interval(returns: list[float], benchmark: float) -> None:
    psr = probabilistic_sharpe_ratio(returns, benchmark)
    assert 0.0 <= psr <= 1.0


@given(returns=_returns_series, n_trials=st.integers(min_value=1, max_value=1000))
def test_dsr_always_in_unit_interval(returns: list[float], n_trials: int) -> None:
    dsr = deflated_sharpe_ratio(returns, n_trials)
    assert 0.0 <= dsr <= 1.0


@given(perf=_candidate_matrix, n_splits=st.integers(min_value=2, max_value=10))
def test_pbo_always_in_unit_interval(perf: np.ndarray, n_splits: int) -> None:
    pbo = probability_of_backtest_overfitting(perf, n_splits)
    assert 0.0 <= pbo <= 1.0


# ── deflation only ever lowers a probability ──────────────────────────────────


@given(returns=_returns_series, b1=_benchmark, b2=_benchmark)
def test_psr_non_increasing_in_benchmark(returns: list[float], b1: float, b2: float) -> None:
    lo, hi = min(b1, b2), max(b1, b2)
    assert probabilistic_sharpe_ratio(returns, lo) >= probabilistic_sharpe_ratio(returns, hi)


@given(
    returns=_returns_series,
    n1=st.integers(min_value=1, max_value=500),
    n2=st.integers(min_value=1, max_value=500),
)
def test_dsr_non_increasing_in_trials(returns: list[float], n1: int, n2: int) -> None:
    lo, hi = min(n1, n2), max(n1, n2)
    assert deflated_sharpe_ratio(returns, lo) >= deflated_sharpe_ratio(returns, hi)


@given(returns=_returns_series, n_trials=st.integers(min_value=2, max_value=1000))
def test_dsr_never_exceeds_psr(returns: list[float], n_trials: int) -> None:
    assert deflated_sharpe_ratio(returns, n_trials) <= probabilistic_sharpe_ratio(returns, 0.0)


# ── CSCV must not care about configuration order ──────────────────────────────


@given(
    matrix_seed=st.integers(min_value=0, max_value=2**32 - 1),
    perm_seed=st.integers(min_value=0, max_value=2**32 - 1),
    T=st.integers(min_value=8, max_value=48),
    N=st.integers(min_value=2, max_value=6),
    n_splits=st.integers(min_value=2, max_value=8),
)
def test_pbo_invariant_under_column_permutation(
    matrix_seed: int, perm_seed: int, T: int, N: int, n_splits: int
) -> None:
    """Permuting the candidate columns must not change the PBO.

    Matrices are drawn from a seeded continuous distribution because the
    invariant holds for tie-free data: with *exactly* tied in-sample sums the
    winning configuration (``argmax``) is order-dependent by construction, and
    ties have probability zero for continuous performance figures.
    """
    perf = np.random.default_rng(matrix_seed).standard_normal((T, N))
    perm = np.random.default_rng(perm_seed).permutation(N)
    base = probability_of_backtest_overfitting(perf, n_splits)
    shuffled = probability_of_backtest_overfitting(perf[:, perm], n_splits)
    assert shuffled == base


# ── walk-forward CV never leaks into the evaluation window ────────────────────


@given(
    train_size=st.integers(min_value=1, max_value=20),
    valid_size=st.integers(min_value=1, max_value=10),
    test_size=st.integers(min_value=1, max_value=10),
    embargo_size=st.integers(min_value=0, max_value=10),
    label_horizon=st.integers(min_value=1, max_value=10),
    extra=st.integers(min_value=0, max_value=60),
)
def test_walk_forward_never_leaks_into_eval_window(
    train_size: int,
    valid_size: int,
    test_size: int,
    embargo_size: int,
    label_horizon: int,
    extra: int,
) -> None:
    splitter = PurgedWalkForwardSplitter(
        train_size=train_size,
        valid_size=valid_size,
        test_size=test_size,
        embargo_size=embargo_size,
        label_horizon=label_horizon,
    )
    n = splitter.window + extra
    for train_idx, valid_idx, test_idx in splitter.split(n):
        assert train_idx.size > 0  # fully-purged folds are skipped, not emptied
        eval_start = int(valid_idx.min())
        # Purging: no training observation's forward-label window may reach the
        # evaluation window (positional, per the label_horizon contract).
        assert np.all(train_idx + label_horizon < eval_start)
        # Embargo: the last embargo_size bars before the eval window are gone.
        assert np.all(train_idx < eval_start - embargo_size)
        # Windows are ordered and disjoint, and all indices are in range.
        assert int(train_idx.max()) < eval_start
        assert int(valid_idx.max()) < int(test_idx.min())
        assert int(test_idx.max()) < n


# ── effective trials / matrix-faithful deflation ──────────────────────────────


@given(perf=_candidate_matrix)
def test_effective_trials_bounded_by_column_count(perf: np.ndarray) -> None:
    K = effective_trials(perf)
    assert 0 <= K <= perf.shape[1]


@given(perf=_candidate_matrix)
def test_cluster_trials_is_a_partition_of_columns(perf: np.ndarray) -> None:
    clusters = cluster_trials(perf)
    members = [j for cluster in clusters for j in cluster]
    assert len(members) == len(set(members))  # disjoint
    assert all(0 <= j < perf.shape[1] for j in members)
    assert all(cluster for cluster in clusters)  # never an empty cluster


@given(perf=_candidate_matrix)
def test_cross_trial_sharpe_std_is_non_negative(perf: np.ndarray) -> None:
    sigma = cross_trial_sharpe_std(perf)
    assert sigma >= 0.0  # inf (unmeasurable, fail-closed) satisfies this too


@given(perf=_candidate_matrix)
def test_matrix_faithful_dsr_is_a_probability_capped_by_psr(perf: np.ndarray) -> None:
    # The matrix-derived benchmark SR* is always >= 0, so the matrix-faithful
    # deflation can never *raise* the probability above the undeflated PSR.
    dsr = deflated_sharpe_ratio_from_trials(perf)
    assert 0.0 <= dsr <= 1.0
    best = int(np.argmax([sharpe_ratio(perf[:, j]) for j in range(perf.shape[1])]))
    psr = probabilistic_sharpe_ratio(perf[:, best], 0.0)
    assert dsr <= psr + 1e-12


@given(
    matrix_seed=st.integers(min_value=0, max_value=2**32 - 1),
    perm_seed=st.integers(min_value=0, max_value=2**32 - 1),
    T=st.integers(min_value=8, max_value=48),
    N=st.integers(min_value=2, max_value=6),
)
def test_effective_trials_invariant_under_column_permutation(
    matrix_seed: int, perm_seed: int, T: int, N: int
) -> None:
    """The trial count and the matrix-faithful DSR must not care which order
    the configurations were tried in (same rationale as the PBO invariant;
    continuous seeded data keeps the clustering tie-free)."""
    perf = np.random.default_rng(matrix_seed).standard_normal((T, N))
    perm = np.random.default_rng(perm_seed).permutation(N)
    shuffled = perf[:, perm]
    assert effective_trials(shuffled) == effective_trials(perf)
    base = deflated_sharpe_ratio_from_trials(perf)
    assert math.isclose(
        deflated_sharpe_ratio_from_trials(shuffled), base, rel_tol=1e-9, abs_tol=1e-12
    )
