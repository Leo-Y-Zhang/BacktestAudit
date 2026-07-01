# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/), and this project adheres to
[Semantic Versioning](https://semver.org/).

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
