"""Runnable demo for OverfitCheck -- see it work in one shot.

This script is fully offline and deterministic (seeded). It:

1. Generates a sample ``sample_returns.csv`` next to this file (if missing),
2. Evaluates that honest single strategy and prints the verdict,
3. Writes a self-contained HTML report and a Markdown report beside it,
4. Re-judges the *same* strategy on only its first 90 days, to show the
   evidence-gap report: the verdict says how much more track record the
   record falls short of, not just "no",
5. As a cautionary tale, evaluates a 50-configuration noise matrix to show how
   OverfitCheck flags a search that found nothing but luck.

Run it with::

    python examples/run_example.py

Nothing here trades, connects to a broker, or promises returns -- OverfitCheck
only *measures* how much of an apparent edge is real versus overfit.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from overfitcheck import evaluate
from overfitcheck.report import render_markdown, write_report

HERE = Path(__file__).resolve().parent
SAMPLE_CSV = HERE / "sample_returns.csv"
HTML_REPORT = HERE / "sample_report.html"
MARKDOWN_REPORT = HERE / "sample_report.md"


def make_sample_csv(path: Path) -> None:
    """Write a deterministic ``date,returns`` sample of daily returns."""
    rng = np.random.default_rng(4)
    n = 1500
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    # A modest, genuine drift with realistic daily volatility.
    returns = 0.0008 + 0.008 * rng.standard_normal(n)
    frame = pd.DataFrame({"date": dates.strftime("%Y-%m-%d"), "returns": returns})
    frame.to_csv(path, index=False)


def main() -> int:
    if not SAMPLE_CSV.exists():
        make_sample_csv(SAMPLE_CSV)
    print(f"Sample returns: {SAMPLE_CSV}")

    returns = pd.read_csv(SAMPLE_CSV)["returns"].to_numpy(dtype=float)

    print("\n=== 1) An honest single strategy ===")
    verdict = evaluate(returns, n_trials=1)
    print(verdict.summary())

    write_report(verdict, str(HTML_REPORT))
    MARKDOWN_REPORT.write_text(render_markdown(verdict), encoding="utf-8")
    print(f"\nHTML report     -> {HTML_REPORT}")
    print(f"Markdown report -> {MARKDOWN_REPORT}")

    print("\n=== 2) The same strategy, judged on only its first 90 days ===")
    young = evaluate(returns[:90], n_trials=1)
    print(young.summary())
    print(
        "\nLesson: the same edge that is DEPLOYABLE on 1500 observations is "
        "indistinguishable from luck on 90. The evidence gap quantifies exactly "
        "how far short the record falls instead of just saying no."
    )

    print("\n=== 3) A cautionary tale: 50 noise configurations ===")
    rng = np.random.default_rng(0)
    candidates = 0.01 * rng.standard_normal((750, 50))
    # Picks the in-sample best, computes PBO, and -- because no n_trials is
    # given -- measures the deflation benchmark from the matrix itself
    # (cross-trial Sharpe dispersion across the effective trials).
    overfit = evaluate(candidates)
    print(overfit.summary())
    print(
        "\nLesson: search hard enough over noise and *something* will look great in-sample. "
        "The Deflated Sharpe and PBO are what tell you it is not real."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
