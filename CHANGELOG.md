# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/), and this project adheres to
[Semantic Versioning](https://semver.org/).

## [0.2.1] - 2026-07-30

### Changed
- The worked example (`examples/run_example.py`) gained a middle scene: the
  same demo strategy judged on only its first 90 days, producing a
  `NOT_DEPLOYABLE` verdict whose reasons quantify the evidence gap. The README
  worked-example section now pastes the real output of all three scenes,
  reasons included. No library behaviour changed.

### Fixed
- The evidence gap now counts only finite observations. `n_periods`, the
  MinTRL shortfall arithmetic in the reasons, `summary()`, the report
  "Observations" row and the CLI JSON previously used the raw input length,
  so NaN/inf rows (e.g. blank CSV cells, which load as NaN) were credited as
  track record: the reported shortfall was understated and `summary()` could
  call a record "sufficient" whose deflated Sharpe failed the bar. The
  deployable/classification result itself was unaffected by that path.
- PSR, DSR, MinTRL and the Sharpe standard error now fail closed on
  near-constant returns. Scipy's skew/kurtosis go NaN under catastrophic
  cancellation and the NaN slipped past the non-positive-variance guard,
  leaking NaN probabilities into `evaluate(...)` — which could wave a
  constant series through the deflated-Sharpe gate (NaN fails no `<`
  comparison). Such records are now degenerate, exactly as documented.
- An infinite MinTRL is no longer always blamed on "Sharpe does not exceed
  the benchmark": the failure reasons distinguish a confidence bar of 1, a
  record too degenerate to measure, and a Sharpe below the deflation
  benchmark, and `summary()` uses neutral fail-closed wording.
- The `minimum_track_record_length` docstring scoped its equivalence claim:
  `T >= MinTRL` iff `PSR >= confidence` is exact for `confidence > 0.5`
  (every realistic bar), not for all confidence levels.

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
