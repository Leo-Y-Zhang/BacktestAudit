# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/), and this project adheres to
[Semantic Versioning](https://semver.org/).

## [0.2.0] - 2026-07-30

The evidence-gap release: when a track record is not statistically significant,
Lyra Validate now says *how much more evidence* would be needed, not just "no".

### Added
- Minimum Track Record Length (MinTRL, Bailey & López de Prado 2012) and
  Minimum Backtest Length (MinBTL, Bailey, Borwein, López de Prado & Zhu 2014,
  *Pseudo-Mathematics and Financial Charlatanism*) as public statistics, plus
  the Sharpe-estimator standard error they build on.
- Evidence-gap reporting threaded through the verdict: on a significance
  failure, `evaluate(...)` reports how many observations (and years, via
  `periods_per_year`) the record falls short of MinTRL at the observed moments;
  on a pass it confirms the observed length against MinTRL. MinBTL is reported
  for searched records (`n_trials > 1`). Surfaced as the `Verdict` fields
  `min_track_record` and `min_backtest_years`, in `summary()`, in the CLI JSON
  output, and as diagnostics rows in the HTML/Markdown reports.
- Property-based invariant test layer (Hypothesis, a dev-only dependency, with
  a deterministic, bounded profile): PSR/DSR/PBO stay in [0, 1], PSR is
  non-increasing in the benchmark, deflation only ever lowers a probability,
  PBO is invariant under column permutation, and the purged walk-forward
  splitter never leaks a training label into the evaluation window.

### Changed
- Nothing numeric: every existing statistic and verdict output is unchanged.
  The additions are strictly additive (new fields, reasons, and report rows).

## [0.1.0] - 2026-06-30

First release. A default-deny verdict on whether a backtest's edge is statistically
real or an artefact of luck, non-normality, or searching over many configurations.

### Added
- Probabilistic Sharpe Ratio (PSR) and Deflated Sharpe Ratio (DSR) with the
  Bailey–López de Prado non-normality and multiple-testing corrections.
- Probability of Backtest Overfitting (PBO) via Combinatorially-Symmetric
  Cross-Validation (CSCV).
- Purged + embargoed walk-forward cross-validation (positional purging).
- A default-deny `evaluate(...)` verdict (DEPLOYABLE / NOT_DEPLOYABLE /
  PROBABLY_OVERFIT) over the deflated Sharpe, Sharpe, and PBO gates.
- `lyra-validate` CLI with CI-gate exit codes, JSON output, and self-contained
  HTML/Markdown reports.
- Full offline operation (numpy / pandas / scipy only; no network access).
