"""Command-line entry point for ``backtest-validate``.

Reads a CSV of per-period returns (one column = one strategy; multiple columns =
a candidate matrix) and prints an honest, default-deny verdict. No network access
is ever performed -- everything runs locally on the file you provide.

Usage::

    backtest-validate returns.csv [--trials N] [--report out.html] [--json]

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


def _positive_int(text: str) -> int:
    """Argparse type for counts that must be >= 1.

    ``--trials 0`` (or a negative typo) used to be accepted and silently
    floored to 1 downstream, which both disabled the deflation and switched
    off the matrix-measured benchmark (any explicit value routes to the
    published raw-count path) -- an expensive typo, so it is rejected here.
    """
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"invalid int value: {text!r}") from None
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be a positive integer (got {value})")
    return value


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="backtest-validate",
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
        type=_positive_int,
        default=None,
        metavar="N",
        help="Number of configurations tried during research (selection-bias count). "
        "Defaults to the number of columns for a matrix, else 1. When omitted for a "
        "matrix, the deflation benchmark is measured from the matrix itself (cross-trial "
        "Sharpe dispersion across effective, correlation-clustered trials); supplying it "
        "asserts the search size and keeps the published raw-count deflation.",
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
    parser.add_argument("--version", action="version", version=f"backtest-validate {__version__}")
    return parser


def _not_returns_reason(name: object, values: npt.NDArray[np.float64]) -> str | None:
    """Why ``values`` cannot be a per-period return series, or None if it could be.

    This tool exists to refuse things, so the one input it must never accept
    silently is a column that is not returns at all. ``select_dtypes(["number"])``
    alone accepts the row counter that ``DataFrame.to_csv()`` writes by default,
    and a counter rises every period with no drawdown, which is the highest
    Sharpe any column can have. Left unchecked it is selected as the in-sample
    best and certified DEPLOYABLE -- the exact inversion of the product.

    Both tests are deliberately blunt, because a false rejection is a loud error
    the user can fix with --column while a false acceptance is a silent lie:
    a monotone run is only called out once it is long enough that no real return
    series would be monotone by chance, and the magnitude bound is set where a
    "return" is unarguably a price, level or date instead.
    """
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return "every value is NaN or infinite"
    if finite.size >= 8:
        steps = np.diff(finite)
        if np.all(steps > 0):
            return (
                "it increases at every one of its steps, so it is a row counter, "
                "index, date or cumulative level rather than per-period returns"
            )
        if np.all(steps < 0):
            return "it decreases at every one of its steps, so it is a level, not returns"
    biggest = float(np.max(np.abs(finite)))
    if biggest > 10.0:
        return (
            f"its largest magnitude is {biggest:.6g}, which as a per-period return "
            f"would be a {biggest * 100:.0f}% move; that is a price, level or date, "
            f"not a return"
        )
    return None


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
        chosen = np.asarray(numeric[column].to_numpy(), dtype=np.float64)
        reason = _not_returns_reason(column, chosen)
        if reason is not None:
            raise ValueError(f"Column {column!r} cannot be per-period returns: {reason}.")
        return chosen

    # pandas writes the frame index as an unnamed leading column, so a file
    # produced by a plain to_csv() arrives here with a counter beside the data.
    dropped: list[str] = []
    for name in list(numeric.columns):
        if isinstance(name, str) and name.startswith("Unnamed:"):
            numeric = numeric.drop(columns=[name])
            dropped.append(str(name))

    rejected: list[str] = []
    for name in list(numeric.columns):
        reason = _not_returns_reason(name, np.asarray(numeric[name].to_numpy(), dtype=np.float64))
        if reason is not None:
            numeric = numeric.drop(columns=[name])
            rejected.append(f"{name!r} ({reason})")

    if numeric.shape[1] == 0:
        detail = "; ".join(rejected) if rejected else "none were usable"
        raise ValueError(
            f"No column in {path!r} looks like per-period returns: {detail}. "
            f"Pass --column to name the one that is."
        )
    if rejected or dropped:
        skipped = ", ".join(dropped + rejected)
        print(
            f"note: ignored {len(dropped) + len(rejected)} column(s) that are not "
            f"per-period returns: {skipped}",
            file=sys.stderr,
        )

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
        "effective_trials": verdict.effective_trials,
        "cross_trial_sharpe_std": _finite_or_none(verdict.cross_trial_sharpe_std),
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
