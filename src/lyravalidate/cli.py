"""Command-line entry point for ``lyra-validate``.

Reads a CSV of per-period returns (one column = one strategy; multiple columns =
a candidate matrix) and prints an honest, default-deny verdict. No network access
is ever performed -- everything runs locally on the file you provide.

Usage::

    lyra-validate returns.csv [--trials N] [--report out.html] [--json]

The process exits ``0`` when the verdict is ``DEPLOYABLE`` and non-zero otherwise,
so it composes cleanly inside a research pipeline or a CI gate.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from . import __version__
from .evaluate import Thresholds, Verdict, evaluate
from .report import CITATIONS, DISCLAIMER, write_report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lyra-validate",
        description=DISCLAIMER,
        epilog=CITATIONS,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "returns_csv",
        nargs="?",
        help="Path to a CSV of per-period returns. A single returns column (optionally "
        "with a leading date column), or multiple numeric columns = a T x N candidate "
        "matrix.",
    )
    parser.add_argument(
        "--column",
        default=None,
        help="Evaluate only this named column (treated as a single strategy).",
    )
    parser.add_argument(
        "--trials",
        "--n-trials",
        dest="n_trials",
        type=int,
        default=None,
        metavar="N",
        help="Number of configurations tried during research (selection-bias count). "
        "Defaults to the number of columns for a matrix, else 1.",
    )
    parser.add_argument(
        "--periods-per-year",
        type=int,
        default=252,
        help="Annualisation factor for the Sharpe ratio (default: 252).",
    )
    parser.add_argument(
        "--pbo-splits",
        type=int,
        default=16,
        metavar="S",
        help="Number of CSCV blocks S for the PBO estimate (forced even, >= 2; default: "
        "16). Larger S is honoured but costs C(S, S/2) partitions, which grows fast "
        "(C(16,8)=12870), so raise it deliberately.",
    )
    parser.add_argument(
        "--report",
        default=None,
        metavar="PATH",
        help="Also write a one-page report to PATH. The format is inferred from the "
        "suffix (.md/.markdown -> Markdown, anything else -> self-contained HTML).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the verdict as machine-readable JSON instead of human text.",
    )
    parser.add_argument("--min-deflated-sharpe", type=float, default=0.95)
    parser.add_argument("--min-sharpe", type=float, default=0.75)
    parser.add_argument("--max-pbo", type=float, default=0.5)
    parser.add_argument(
        "--about",
        action="store_true",
        help="Print the methodology disclaimer and citations, then exit.",
    )
    parser.add_argument("--version", action="version", version=f"lyra-validate {__version__}")
    return parser


def _load_returns(path: str, column: str | None) -> npt.NDArray[np.float64]:
    frame = pd.read_csv(path)
    numeric = frame.select_dtypes(include=["number"])
    if numeric.shape[1] == 0:
        raise ValueError(f"No numeric columns found in {path!r}.")
    if column is not None:
        if column not in numeric.columns:
            available = ", ".join(map(str, numeric.columns))
            raise ValueError(
                f"Column {column!r} not found among numeric columns ({available})."
            )
        return np.asarray(numeric[column].to_numpy(), dtype=np.float64)
    if numeric.shape[1] == 1:
        return np.asarray(numeric.iloc[:, 0].to_numpy(), dtype=np.float64)
    return np.asarray(numeric.to_numpy(), dtype=np.float64)


def _finite_or_none(value: float | None) -> float | None:
    """Map non-finite floats (NaN / inf) to ``None`` so the JSON stays valid."""
    if value is None:
        return None
    return value if math.isfinite(value) else None


def _verdict_to_dict(verdict: Verdict) -> dict[str, Any]:
    """A JSON-serialisable view of a verdict (no NaN/inf, which break JSON)."""
    return {
        "classification": verdict.classification,
        "deployable": verdict.deployable,
        "deflated_sharpe": _finite_or_none(verdict.deflated_sharpe),
        "annualised_sharpe": _finite_or_none(verdict.sharpe),
        "pbo": _finite_or_none(verdict.pbo),
        "probabilistic_sharpe": _finite_or_none(verdict.probabilistic_sharpe),
        "n_trials": verdict.n_trials,
        "n_periods": verdict.n_periods,
        "periods_per_year": verdict.periods_per_year,
        "min_track_record": _finite_or_none(verdict.min_track_record),
        "min_backtest_years": _finite_or_none(verdict.min_backtest_years),
        "oos_sharpe": _finite_or_none(verdict.oos_sharpe),
        "oos_information_coefficient": _finite_or_none(verdict.oos_information_coefficient),
        "reasons": list(verdict.reasons),
        "disclaimer": DISCLAIMER,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.about:
        print(DISCLAIMER)
        print()
        print(CITATIONS)
        return 0

    if not args.returns_csv:
        parser.error("the following argument is required: returns_csv (or use --about)")

    try:
        data = _load_returns(args.returns_csv, args.column)
    except (OSError, ValueError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    thresholds = Thresholds(
        min_deflated_sharpe=args.min_deflated_sharpe,
        min_sharpe=args.min_sharpe,
        max_pbo=args.max_pbo,
    )
    try:
        verdict = evaluate(
            data,
            n_trials=args.n_trials,
            periods_per_year=args.periods_per_year,
            thresholds=thresholds,
            pbo_splits=args.pbo_splits,
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(_verdict_to_dict(verdict), indent=2))
    else:
        print(DISCLAIMER)
        print()
        print(verdict.summary())
        if not args.report:
            print(
                "\n(run with --report out.html for a plain-English explanation of "
                "each number)"
            )

    if args.report:
        try:
            written = write_report(verdict, args.report, thresholds=thresholds)
        except OSError as exc:
            print(f"error: could not write report: {exc}", file=sys.stderr)
            return 2
        # Keep stdout clean (it may be JSON); the confirmation goes to stderr.
        print(f"Report written to {written}", file=sys.stderr)

    return 0 if verdict.deployable else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
