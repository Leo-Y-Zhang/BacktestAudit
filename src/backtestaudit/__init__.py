"""BacktestAudit -- an honest verdict on whether a backtest edge is real.

A standalone *measurement* tool. Point it at your own backtest results and it
re-implements the published statistics of Bailey & Lopez de Prado -- the Deflated
Sharpe Ratio and the Probability of Backtest Overfitting (CSCV) -- plus a purged,
embargoed walk-forward splitter, to tell you how much of an apparent edge is
likely real versus overfit. It does not generate signals or promise returns.
"""

from __future__ import annotations

from ._version import __version__
from .crossval import PurgedWalkForwardSplitter, default_walk_forward_splitter
from .evaluate import Thresholds, Verdict, evaluate
from .report import render_html, render_markdown, write_report
from .stats import (
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

__all__ = [
    "PurgedWalkForwardSplitter",
    "Thresholds",
    "Verdict",
    "__version__",
    "annualized_sharpe",
    "cluster_trials",
    "cross_trial_sharpe_std",
    "default_walk_forward_splitter",
    "deflated_sharpe_ratio",
    "deflated_sharpe_ratio_from_trials",
    "effective_trials",
    "evaluate",
    "expected_max_sharpe_benchmark",
    "hit_rate",
    "information_coefficient",
    "max_drawdown",
    "minimum_backtest_length",
    "minimum_track_record_length",
    "probabilistic_sharpe_ratio",
    "probability_of_backtest_overfitting",
    "rank_information_coefficient",
    "render_html",
    "render_markdown",
    "sharpe_ratio",
    "sharpe_standard_error",
    "write_report",
]
