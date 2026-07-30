"""Render a clean, one-page Lyra Validate report from a :class:`Verdict`.

Two faithful renderings of the same content are provided:

* :func:`render_markdown` -- a plain-text Markdown report, ideal for a terminal,
  a pull request, or a research log.
* :func:`render_html` -- a single, self-contained HTML page (inline CSS, **no**
  external assets, scripts, fonts or images), ideal for emailing or archiving.

Both lead with the big verdict, then show each gate (Deflated Sharpe vs its
cutoff, PBO, annualised Sharpe) with a PASS/FAIL/N-A flag, and then explain in
plain English what every number means and what the verdict implies. The tone is
deliberately honest: this is a *measurement* of overfitting risk, never a promise
of returns.
"""

from __future__ import annotations

import html
import math
from dataclasses import dataclass
from datetime import datetime, timezone

from ._version import __version__
from .evaluate import Thresholds, Verdict

__all__ = [
    "CITATIONS",
    "DISCLAIMER",
    "render_html",
    "render_markdown",
    "write_report",
]

DISCLAIMER = (
    "Lyra Validate is a MEASUREMENT tool, not a money-maker. It estimates whether a "
    "backtested edge is statistically real or likely overfit; it does NOT generate "
    "signals, trade, size positions, or guarantee returns. Methods are the published "
    "Deflated Sharpe Ratio and Probability of Backtest Overfitting (CSCV) of Bailey & "
    "Lopez de Prado. Nothing here is investment advice."
)

CITATIONS = (
    "References:\n"
    "  - Bailey & Lopez de Prado (2012), The Sharpe Ratio Efficient Frontier "
    "(PSR, MinTRL).\n"
    "  - Bailey & Lopez de Prado (2014), The Deflated Sharpe Ratio (DSR).\n"
    "  - Bailey, Borwein, Lopez de Prado & Zhu (2014), Pseudo-Mathematics and "
    "Financial Charlatanism (MinBTL).\n"
    "  - Bailey, Borwein, Lopez de Prado & Zhu (2017), The Probability of Backtest "
    "Overfitting (PBO via CSCV).\n"
    "  - Lopez de Prado (2018), Advances in Financial Machine Learning, ch. 7 "
    "(purged walk-forward CV)."
)

_REFERENCES: tuple[str, ...] = (
    "Bailey & Lopez de Prado (2012), The Sharpe Ratio Efficient Frontier (PSR, MinTRL).",
    "Bailey & Lopez de Prado (2014), The Deflated Sharpe Ratio (DSR).",
    "Bailey, Borwein, Lopez de Prado & Zhu (2014), Pseudo-Mathematics and Financial "
    "Charlatanism (MinBTL).",
    "Bailey, Borwein, Lopez de Prado & Zhu (2017), The Probability of Backtest "
    "Overfitting (PBO via CSCV).",
    "Lopez de Prado (2018), Advances in Financial Machine Learning, ch. 7 "
    "(purged walk-forward CV).",
)

# Plain-English headline beneath the big verdict word.
_SUBTITLE: dict[str, str] = {
    "DEPLOYABLE": "The track record is statistically consistent with a real edge.",
    "PROBABLY_OVERFIT": "This looked good in-sample but does not survive honest deflation.",
    "NOT_DEPLOYABLE": "The evidence does not support a real, repeatable edge.",
}

# Longer "what this means" paragraph per classification.
_MEANING: dict[str, str] = {
    "DEPLOYABLE": (
        "Every gate cleared: the Sharpe is statistically significant even after "
        "deflating for the number of configurations tried and for non-normal returns, "
        "and the selection process does not look overfit. This is NOT a promise of "
        "future profit -- it means the record is consistent with a genuine edge rather "
        "than luck. Confirm it forward with paper or live trading before committing "
        "capital, and re-validate as new data arrives."
    ),
    "PROBABLY_OVERFIT": (
        "The raw backtest looked attractive, but it does not survive an honest "
        "correction for selection bias and non-normality (a low Deflated Sharpe) "
        "and/or the cross-validated overfitting probability (PBO) is high. In plain "
        "terms: the configuration that won in-sample is the kind that tends to lose "
        "out-of-sample. Treat the headline Sharpe as optimistic and do not deploy on "
        "this evidence."
    ),
    "NOT_DEPLOYABLE": (
        "The record does not show a risk-adjusted edge strong enough to clear the "
        "policy bars. This is the expected outcome for noise or for a strategy whose "
        "costs eat its signal. No deployment is warranted on this evidence."
    ),
}

# Banner colours (background, foreground) keyed by classification.
_BANNER: dict[str, tuple[str, str]] = {
    "DEPLOYABLE": ("#0f7b3f", "#ffffff"),
    "PROBABLY_OVERFIT": ("#b45309", "#ffffff"),
    "NOT_DEPLOYABLE": ("#b91c1c", "#ffffff"),
}

_STATUS_COLOR: dict[str, str] = {
    "PASS": "#0f7b3f",
    "FAIL": "#b91c1c",
    "N/A": "#6b7280",
}


@dataclass(frozen=True)
class _Metric:
    """One gate, ready to render identically in Markdown or HTML."""

    name: str
    value: str
    cutoff: str
    status: str  # "PASS" | "FAIL" | "N/A"
    explanation: str


def _build_metrics(verdict: Verdict, thresholds: Thresholds) -> list[_Metric]:
    """Derive the (DSR, Sharpe, PBO) gate rows from a verdict and policy."""
    dsr_pass = verdict.deflated_sharpe >= thresholds.min_deflated_sharpe
    sharpe_pass = verdict.sharpe > thresholds.min_sharpe
    metrics = [
        _Metric(
            name="Deflated Sharpe Ratio (DSR)",
            value=f"{verdict.deflated_sharpe:.3f}",
            cutoff=f">= {thresholds.min_deflated_sharpe:.2f}",
            status="PASS" if dsr_pass else "FAIL",
            explanation=(
                "Probability your true Sharpe is positive AFTER correcting for the number "
                "of configurations tried, the length of the record, and non-normal "
                "(fat-tailed) returns. Higher is better; below the cutoff the track record "
                "could plausibly be luck."
            ),
        ),
        _Metric(
            name="Annualised Sharpe",
            value=f"{verdict.sharpe:.3f}",
            cutoff=f"> {thresholds.min_sharpe:.2f}",
            status="PASS" if sharpe_pass else "FAIL",
            explanation=(
                "Risk-adjusted return scaled to one year. This is the raw, undeflated "
                "figure -- on its own it says nothing about overfitting, which is exactly "
                "why the Deflated Sharpe exists."
            ),
        ),
    ]
    if math.isfinite(verdict.pbo):
        metrics.append(
            _Metric(
                name="Probability of Backtest Overfitting (PBO)",
                value=f"{verdict.pbo:.3f}",
                cutoff=f"<= {thresholds.max_pbo:.2f}",
                status="PASS" if verdict.pbo <= thresholds.max_pbo else "FAIL",
                explanation=(
                    "Across many in-sample / out-of-sample reshuffles (CSCV), how often the "
                    "configuration that looked best in-sample fell into the WORSE half "
                    "out-of-sample. Near or above 0.5 means the selection process is fitting "
                    "noise."
                ),
            )
        )
    else:
        metrics.append(
            _Metric(
                name="Probability of Backtest Overfitting (PBO)",
                value="n/a",
                cutoff=f"<= {thresholds.max_pbo:.2f}",
                status="N/A",
                explanation=(
                    "Not computable for a single strategy. Supply a T x N matrix of the "
                    "candidate configurations you searched over to estimate it."
                ),
            )
        )
    return metrics


def _timestamp(generated_at: datetime | None) -> str:
    when = generated_at or datetime.now(timezone.utc)
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    return when.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def _diagnostics(verdict: Verdict) -> list[tuple[str, str]]:
    """Supplementary, non-gating figures shown in a small details block."""
    rows: list[tuple[str, str]] = [
        ("Configurations assumed tried (n_trials)", str(verdict.n_trials)),
        ("Observations", str(verdict.n_periods)),
        ("Periods per year", str(verdict.periods_per_year)),
    ]
    if math.isfinite(verdict.probabilistic_sharpe):
        rows.append(("Probabilistic Sharpe (vs 0)", f"{verdict.probabilistic_sharpe:.3f}"))
    if not math.isnan(verdict.min_track_record):
        min_trl = (
            "unreachable at the observed moments"
            if math.isinf(verdict.min_track_record)
            else str(math.ceil(verdict.min_track_record))
        )
        rows.append(("Minimum track record length (MinTRL, obs)", min_trl))
    if verdict.n_trials > 1 and not math.isnan(verdict.min_backtest_years):
        min_btl = (
            "unattainable (Sharpe not positive)"
            if math.isinf(verdict.min_backtest_years)
            else f"{verdict.min_backtest_years:.1f}"
        )
        rows.append(("Minimum backtest length (MinBTL, years)", min_btl))
    if verdict.oos_sharpe is not None:
        rows.append(("Walk-forward OOS Sharpe", f"{verdict.oos_sharpe:.3f}"))
    if verdict.oos_information_coefficient is not None:
        rows.append(
            ("Walk-forward OOS info. coefficient", f"{verdict.oos_information_coefficient:.3f}")
        )
    return rows


# ── Markdown ──────────────────────────────────────────────────────────────────


def render_markdown(
    verdict: Verdict,
    *,
    thresholds: Thresholds | None = None,
    title: str = "Lyra Validate - Backtest Verdict",
    generated_at: datetime | None = None,
) -> str:
    """Render a one-page Markdown report for ``verdict``.

    Parameters
    ----------
    verdict:
        The :class:`~lyravalidate.evaluate.Verdict` to report on.
    thresholds:
        The policy the verdict was judged against (for the cutoff column).
        Defaults to :class:`~lyravalidate.evaluate.Thresholds` defaults.
    title:
        Heading for the report.
    generated_at:
        Timestamp to stamp the report with; defaults to "now" in UTC.
    """
    thr = thresholds or Thresholds()
    metrics = _build_metrics(verdict, thr)
    subtitle = _SUBTITLE.get(verdict.classification, "")
    meaning = _MEANING.get(verdict.classification, "")

    lines: list[str] = [
        f"# {title}",
        "",
        f"## Verdict: {verdict.classification}",
        "",
        f"_{subtitle}_",
        "",
        f"Generated {_timestamp(generated_at)} | Lyra Validate v{__version__}",
        "",
        "## Gates",
        "",
        "| Metric | Value | Cutoff | Status |",
        "| --- | --- | --- | --- |",
    ]
    lines += [f"| {m.name} | {m.value} | {m.cutoff} | {m.status} |" for m in metrics]
    lines += [
        "",
        "## What this verdict means",
        "",
        meaning,
        "",
        "## What each number means",
        "",
    ]
    lines += [f"- **{m.name}** -- {m.explanation}" for m in metrics]

    diagnostics = _diagnostics(verdict)
    if diagnostics:
        lines += ["", "## Diagnostics", ""]
        lines += [f"- {label}: {value}" for label, value in diagnostics]

    if verdict.reasons:
        lines += ["", "## Reasons", ""]
        lines += [f"- {reason}" for reason in verdict.reasons]

    lines += [
        "",
        "---",
        "",
        DISCLAIMER,
        "",
        "### References",
        "",
    ]
    lines += [f"- {ref}" for ref in _REFERENCES]
    lines.append("")
    return "\n".join(lines)


# ── HTML ──────────────────────────────────────────────────────────────────────

_CSS = """
:root { color-scheme: light; }
* { box-sizing: border-box; }
body {
  margin: 0; padding: 2rem 1rem;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  color: #1f2937; background: #f3f4f6; line-height: 1.55;
}
.page {
  max-width: 820px; margin: 0 auto; background: #ffffff;
  border: 1px solid #e5e7eb; border-radius: 12px; overflow: hidden;
  box-shadow: 0 1px 3px rgba(0,0,0,0.08);
}
.banner { padding: 1.4rem 1.8rem; }
.banner .label { font-size: 0.8rem; letter-spacing: 0.12em; text-transform: uppercase;
  opacity: 0.85; margin: 0 0 0.25rem; }
.banner h1 { margin: 0; font-size: 2.1rem; letter-spacing: -0.01em; }
.banner p { margin: 0.4rem 0 0; font-size: 1.02rem; opacity: 0.95; }
.body { padding: 1.6rem 1.8rem; }
.meta { color: #6b7280; font-size: 0.85rem; margin: 0 0 1.2rem; }
h2 { font-size: 1.15rem; margin: 1.6rem 0 0.6rem; border-bottom: 1px solid #e5e7eb;
  padding-bottom: 0.3rem; }
table { width: 100%; border-collapse: collapse; margin: 0.4rem 0 0.6rem; }
th, td { text-align: left; padding: 0.55rem 0.6rem; border-bottom: 1px solid #eef0f2; }
th { font-size: 0.75rem; letter-spacing: 0.06em; text-transform: uppercase; color: #6b7280; }
td.value, td.cutoff { font-variant-numeric: tabular-nums; white-space: nowrap; }
.status { font-weight: 700; font-size: 0.8rem; letter-spacing: 0.04em; }
dl { margin: 0.4rem 0; }
dt { font-weight: 600; margin-top: 0.7rem; }
dd { margin: 0.15rem 0 0; color: #374151; }
ul { margin: 0.4rem 0; padding-left: 1.2rem; }
.diag { display: grid; grid-template-columns: 1fr auto; gap: 0.2rem 1rem;
  font-size: 0.92rem; }
.diag .k { color: #6b7280; } .diag .v { font-variant-numeric: tabular-nums; text-align: right; }
.footer { margin-top: 1.6rem; padding-top: 1rem; border-top: 1px solid #e5e7eb;
  color: #6b7280; font-size: 0.82rem; }
.footer .disclaimer { color: #374151; }
.footer ol { margin: 0.5rem 0 0; padding-left: 1.2rem; }
""".strip()


def _esc(text: str) -> str:
    return html.escape(text, quote=True)


def render_html(
    verdict: Verdict,
    *,
    thresholds: Thresholds | None = None,
    title: str = "Lyra Validate - Backtest Verdict",
    generated_at: datetime | None = None,
) -> str:
    """Render a self-contained, one-page HTML report for ``verdict``.

    The output is a complete ``<!DOCTYPE html>`` document with inline CSS and no
    external assets (no scripts, fonts, images or stylesheets), so it can be
    saved or emailed as a single file.
    """
    thr = thresholds or Thresholds()
    metrics = _build_metrics(verdict, thr)
    bg, fg = _BANNER.get(verdict.classification, ("#374151", "#ffffff"))
    subtitle = _SUBTITLE.get(verdict.classification, "")
    meaning = _MEANING.get(verdict.classification, "")

    rows = "\n".join(
        "      <tr>"
        f"<td>{_esc(m.name)}</td>"
        f'<td class="value">{_esc(m.value)}</td>'
        f'<td class="cutoff">{_esc(m.cutoff)}</td>'
        f'<td class="status" style="color:{_STATUS_COLOR[m.status]}">{_esc(m.status)}</td>'
        "</tr>"
        for m in metrics
    )

    defs = "\n".join(
        f"        <dt>{_esc(m.name)}</dt>\n        <dd>{_esc(m.explanation)}</dd>"
        for m in metrics
    )

    diag_rows = "\n".join(
        f'        <div class="k">{_esc(label)}</div><div class="v">{_esc(value)}</div>'
        for label, value in _diagnostics(verdict)
    )
    diag_block = (
        f'      <h2>Diagnostics</h2>\n      <div class="diag">\n{diag_rows}\n      </div>\n'
        if diag_rows
        else ""
    )

    reasons_block = ""
    if verdict.reasons:
        items = "\n".join(f"        <li>{_esc(r)}</li>" for r in verdict.reasons)
        reasons_block = f"      <h2>Reasons</h2>\n      <ul>\n{items}\n      </ul>\n"

    refs = "\n".join(f"        <li>{_esc(ref)}</li>" for ref in _REFERENCES)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_esc(title)}</title>
<style>
{_CSS}
</style>
</head>
<body>
  <main class="page">
    <header class="banner" style="background:{bg};color:{fg}">
      <p class="label">Lyra Validate &middot; Backtest Verdict</p>
      <h1>{_esc(verdict.classification)}</h1>
      <p>{_esc(subtitle)}</p>
    </header>
    <div class="body">
      <p class="meta">Generated {_esc(_timestamp(generated_at))} &middot; \
Lyra Validate v{_esc(__version__)}</p>

      <h2>Gates</h2>
      <table>
        <thead>
          <tr><th>Metric</th><th>Value</th><th>Cutoff</th><th>Status</th></tr>
        </thead>
        <tbody>
{rows}
        </tbody>
      </table>

      <h2>What this verdict means</h2>
      <p>{_esc(meaning)}</p>

      <h2>What each number means</h2>
      <dl>
{defs}
      </dl>

{diag_block}{reasons_block}      <div class="footer">
        <p class="disclaimer">{_esc(DISCLAIMER)}</p>
        <p>References:</p>
        <ol>
{refs}
        </ol>
      </div>
    </div>
  </main>
</body>
</html>
"""


# ── file helper ───────────────────────────────────────────────────────────────


def write_report(
    verdict: Verdict,
    path: str,
    *,
    thresholds: Thresholds | None = None,
    fmt: str | None = None,
    title: str = "Lyra Validate - Backtest Verdict",
    generated_at: datetime | None = None,
) -> str:
    """Render ``verdict`` and write it to ``path``; return the path written.

    The format is chosen from ``fmt`` if given (``"html"`` or ``"markdown"``),
    else inferred from the file suffix (``.md`` / ``.markdown`` -> Markdown,
    anything else -> HTML).
    """
    suffix = path.lower().rsplit(".", 1)[-1] if "." in path else ""
    chosen = fmt or ("markdown" if suffix in {"md", "markdown"} else "html")
    if chosen == "html":
        text = render_html(
            verdict, thresholds=thresholds, title=title, generated_at=generated_at
        )
    elif chosen == "markdown":
        text = render_markdown(
            verdict, thresholds=thresholds, title=title, generated_at=generated_at
        )
    else:
        raise ValueError(f"Unknown report format {chosen!r}; expected 'html' or 'markdown'.")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)
    return path
