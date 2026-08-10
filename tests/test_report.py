"""Tests for the one-page report renderer."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

from backtestaudit import evaluate, render_html, render_markdown, write_report
from backtestaudit.evaluate import Thresholds, Verdict
from backtestaudit.report import CITATIONS, DISCLAIMER

_FIXED = datetime(2026, 6, 30, 12, 0, tzinfo=timezone.utc)


def _deployable_verdict() -> Verdict:
    rng = np.random.default_rng(4)
    returns = 0.0008 + 0.008 * rng.standard_normal(1500)
    return evaluate(returns, n_trials=1)


def _overfit_verdict() -> Verdict:
    rng = np.random.default_rng(0)
    candidates = 0.01 * rng.standard_normal((750, 50))
    return evaluate(candidates)


def test_markdown_contains_verdict_and_gates() -> None:
    verdict = _deployable_verdict()
    md = render_markdown(verdict, generated_at=_FIXED)
    assert verdict.classification in md
    assert "Deflated Sharpe Ratio (DSR)" in md
    assert "Annualised Sharpe" in md
    assert "Probability of Backtest Overfitting" in md
    assert "| Metric | Value | Cutoff | Status |" in md
    assert "What this verdict means" in md
    assert DISCLAIMER in md
    # References are listed in plain English (no broker/IP, just published methods).
    assert "Deflated Sharpe Ratio (DSR)." in md
    assert "2026-06-30 12:00 UTC" in md


def test_markdown_shows_cutoffs_from_thresholds() -> None:
    verdict = _deployable_verdict()
    md = render_markdown(verdict, thresholds=Thresholds(min_deflated_sharpe=0.99))
    assert ">= 0.99" in md


def test_html_is_self_contained_document() -> None:
    verdict = _deployable_verdict()
    page = render_html(verdict, generated_at=_FIXED)
    assert page.startswith("<!DOCTYPE html>")
    assert "</html>" in page.strip().splitlines()[-1] or page.strip().endswith("</html>")
    assert verdict.classification in page
    # No external assets of any kind: no scripts, stylesheets, images or remote URLs.
    lowered = page.lower()
    for forbidden in ("<script", "<link", "<img", "http://", "https://", "src="):
        assert forbidden not in lowered, f"report should be self-contained: found {forbidden!r}"
    # The methodology disclaimer is always present.
    assert "measurement tool" in lowered


def test_html_pbo_row_present_for_matrix() -> None:
    verdict = _overfit_verdict()
    assert np.isfinite(verdict.pbo)
    page = render_html(verdict, generated_at=_FIXED)
    assert f"{verdict.pbo:.3f}" in page
    assert "PROBABLY_OVERFIT" in page
    # The amber banner colour is used for the overfit verdict.
    assert "#b45309" in page


def test_html_escapes_dynamic_text() -> None:
    verdict = Verdict(
        deployable=False,
        classification="NOT_DEPLOYABLE",
        deflated_sharpe=0.1,
        pbo=float("nan"),
        sharpe=0.0,
        n_trials=1,
        reasons=["danger <script>alert('x')</script> & friends"],
    )
    page = render_html(verdict)
    assert "<script>alert" not in page
    assert "&lt;script&gt;" in page
    assert "&amp;" in page


def test_write_report_html_by_suffix(tmp_path: Path) -> None:
    verdict = _deployable_verdict()
    out = tmp_path / "report.html"
    returned = write_report(verdict, str(out), generated_at=_FIXED)
    assert returned == str(out)
    assert out.exists()
    text = out.read_text(encoding="utf-8")
    assert text.startswith("<!DOCTYPE html>")
    assert text == render_html(verdict, generated_at=_FIXED)


def test_write_report_markdown_by_suffix(tmp_path: Path) -> None:
    verdict = _deployable_verdict()
    out = tmp_path / "report.md"
    write_report(verdict, str(out), generated_at=_FIXED)
    text = out.read_text(encoding="utf-8")
    assert text.startswith("# BacktestAudit")
    assert text == render_markdown(verdict, generated_at=_FIXED)


def test_write_report_format_override(tmp_path: Path) -> None:
    verdict = _deployable_verdict()
    # A .txt suffix would default to HTML, but the explicit override wins.
    out = tmp_path / "report.txt"
    write_report(verdict, str(out), fmt="markdown", generated_at=_FIXED)
    assert out.read_text(encoding="utf-8").startswith("# BacktestAudit")


def test_write_report_rejects_unknown_format(tmp_path: Path) -> None:
    verdict = _deployable_verdict()
    with pytest.raises(ValueError, match="Unknown report format"):
        write_report(verdict, str(tmp_path / "r.out"), fmt="pdf")


def test_citations_mention_published_methods() -> None:
    assert "Bailey" in CITATIONS
    assert "Deflated Sharpe" in CITATIONS
    assert "CSCV" in CITATIONS


def test_citations_mention_min_trl_and_min_btl() -> None:
    assert "MinTRL" in CITATIONS
    assert "MinBTL" in CITATIONS


def test_reports_surface_the_evidence_gap() -> None:
    # Single strategy: the MinTRL diagnostic row appears in both renderings.
    verdict = _deployable_verdict()
    md = render_markdown(verdict, generated_at=_FIXED)
    page = render_html(verdict, generated_at=_FIXED)
    assert "MinTRL" in md
    assert "MinTRL" in page
    # A searched matrix additionally shows the MinBTL row.
    overfit = _overfit_verdict()
    assert overfit.n_trials > 1
    assert "MinBTL" in render_markdown(overfit, generated_at=_FIXED)
    assert "MinBTL" in render_html(overfit, generated_at=_FIXED)


def test_reports_show_effective_trials_for_matrix_faithful_verdict() -> None:
    # The default matrix path measures the search from the matrix itself, and
    # both renderings surface the effective-trials diagnostic row.
    overfit = _overfit_verdict()
    assert overfit.effective_trials is not None
    assert "Effective trials" in render_markdown(overfit, generated_at=_FIXED)
    assert "Effective trials" in render_html(overfit, generated_at=_FIXED)
