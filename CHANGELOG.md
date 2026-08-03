# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/), and this project adheres to
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Changed - renamed, breaking for importers and CLI callers
- The project is now **OverfitCheck** (previously Lyra Validate, then briefly
  BacktestValidator - neither name was ever released under this heading). The
  import package is `overfitcheck` (was `lyravalidate`), the distribution name
  is `overfitcheck`, and the console script is `overfitcheck` (was
  `lyra-validate`). No statistic, threshold, verdict or report figure changed:
  the rename is textual, and the full suite passes unchanged.
- Entries below are left as they were published under the old name; they are a
  record of what shipped, not a description of the current API.

### Fixed
- The README's test count said 157; the suite is 164.

## [0.3.0] - 2026-07-31

The matrix-faithful deflation release: when the candidate matrix is the whole
search, the Deflated Sharpe benchmark is *measured* from it (López de Prado &
Lewis 2019) instead of approximated from the selected column alone.

### Added
- Effective trials via deterministic ONC-style correlation clustering
  (López de Prado & Lewis 2019, *Detection of False Investment Strategies
  Using Unsupervised Learning Methods*): `cluster_trials`, `effective_trials`,
  `cross_trial_sharpe_std` and `deflated_sharpe_ratio_from_trials` as public,
  cited statistics. Average-linkage hierarchical clustering on the paper's
  distance `sqrt((1 - rho) / 2)` with mean-silhouette model selection
  (Rousseeuw 1987) and a per-cluster Kaufman & Rousseeuw 0.25 cohesion bound;
  fully deterministic (the paper's randomised k-means is deliberately
  replaced) and permutation-invariant, with the deviations documented in the
  docstrings.
- `evaluate()` on a `T x N` matrix **without** an explicit `n_trials` now
  builds the DSR benchmark from the measured cross-trial Sharpe dispersion
  across the effective (correlation-clustered) trials. MinTRL shares the same
  benchmark, so `T >= MinTRL` iff the DSR gate clears — the equivalence is
  preserved and tested. An explicit `n_trials` asserts the search size and
  keeps the published raw-count deflation; the 1-D path is byte-identical to
  0.2.1; MinBTL deliberately stays on the raw trials-tried count (it is a
  published bound defined for that count); the predictions/targets OOS path
  deliberately does not use the matrix benchmark (its units belong to the
  in-sample search, not the walk-forward series).
- `Verdict.effective_trials` and `Verdict.cross_trial_sharpe_std`, surfaced
  in `summary()`, the reasons, the CLI JSON, and the HTML/Markdown report
  diagnostics; the López de Prado & Lewis citation added to reports and
  `--about`.

### Changed - deliberate numeric change on the default matrix path
All figures measured by running this code on 2026-07-31 and pinned in
regression tests (`tests/test_evaluate.py`):
- Planted 12-column matrix that is really 3 correlated families (pairwise
  rho 0.9, T = 500): DSR 0.786238 -> 0.961623, MinTRL 2146 -> 433
  observations, effective trials 3. Twelve correlated variants are not twelve
  independent trials; the raw-count deflation overstated the search. The
  verdict stays PROBABLY_OVERFIT (PBO 0.551 > 0.5).
- 60 iid noise columns (600 rows): DSR 0.415233 -> 0.426080, effective
  trials 60 — on genuinely independent trials the measured dispersion
  reproduces the null approximation and the number barely moves.
- Worked example scene 3 (750 x 50 noise): DSR 0.519 -> 0.411, effective
  trials 50, and MinTRL is now unreachable (the measured benchmark exceeds
  the selected Sharpe, so no track record length would certify it).

### The trust model - read this before dropping `n_trials`
Without `n_trials` the matrix **is** the whole search: every configuration
tried must be a column. This measurement cannot see trials that are not in
evidence — submit only the winner of a hidden search plus near-copies of it
and the copies collapse to *one* effective trial, no deflation applies, and
the DSR equals the winner's undeflated PSR, which can clear every gate (the
old raw-count default happened to punish that shape incidentally; the
measured default, faithful to the paper, does not). The verdict reasons carry
an explicit caveat whenever the measurement collapses to one trial or the
measured dispersion is near zero, and passing the true `n_trials` restores
the published raw-count deflation. This limitation is pinned in
`test_hidden_search_near_duplicates_pin_the_documented_trust_model`.

### Fixed (adversarial review of the feature)
- A single correlated family now collapses to one cluster. `k = 1` was
  unreachable by the silhouette search and the duplicate test fired only at
  pairwise correlation `>= 1 - 2e-12`, so a parameter sweep around one idea —
  the most common real submission — reported `N` effective trials where the
  truth is 1 (the DSR stayed conservative; the diagnostic was wrong).
  Measured after the fix (T = 500, N = 12, 10 seeds per rho): one cluster
  10/10 at pairwise rho 0.8, 0.9 and 0.95; rho <= 0.6 splits into singletons
  10/10 (the conservative direction); rho 0.7 sits exactly on the
  homogeneity bound and collapsed 6/10.
- Near-duplicate columns (pairwise correlation >= 0.999) are duplicates for
  trial-counting purposes; on that scale, distance differences are float
  noise, not structure.
- Short or wide matrices are no longer clustered: fewer than 100 complete
  rows, or no more rows than columns, fails closed to the published
  raw-count deflation. The unguarded search invented families on iid noise
  (measured, N = 10, 20 seeds per length: at T = 8 the true count was
  recovered 6/20, at T = 40 19/20), under-counting the trials and weakening
  the deflation exactly where the evidence is thinnest.
- Silhouette scoring vectorised: one `effective_trials` call on a 500 x 200
  matrix drops from 13.9 s to 0.3 s (measured 2026-07-31). An explicit
  `n_trials` remains the documented escape for very wide matrices.
- `cross_trial_sharpe_std` validates a caller-supplied `clusters` argument:
  negative or out-of-range indices, empty clusters and overlapping clusters
  raise `ValueError` instead of silently wrapping/double-counting under
  numpy indexing.
- The CLI rejects `--trials 0` and negative values (previously accepted and
  silently floored to 1 — a typo that disabled both the deflation and the
  matrix measurement).

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
