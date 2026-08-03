# PRD — BacktestValidator

**Status:** built (retrospective — written after v0.3.0 shipped)
**Date:** 2026-08-03 · **Repo:** BacktestValidator · **Related:** [TDD](TDD.md), [Design Brief](DESIGN_BRIEF.md)

This document was written after the fact, from the code. Where it and the README
disagree, the code wins.

## Problem

A backtest is the easiest number in finance to fake without meaning to. Search
over two hundred parameter combinations, keep the best, and the winner's Sharpe
ratio is inflated by the search itself — by an amount nobody eyeballs correctly.
The corrections for this exist and are published (Bailey and López de Prado's
Probabilistic and Deflated Sharpe Ratios; the Probability of Backtest
Overfitting via CSCV), but they are scattered across six papers, each with
notational traps — per-period versus annualised Sharpe, excess versus non-excess
kurtosis, calendar-day versus bar-count purging — and getting any of them wrong
silently makes the answer more flattering, never less.

Concretely: the author's own systematic-trading research kept producing
backtests that looked deployable and had no honest referee. The person with this
problem is the person who wrote the strategy, which is exactly the person least
able to judge it.

## Who it is for

The researcher who already has a return series and is about to decide whether to
risk money on it — primarily the author, secondarily any quant or systematic
trader who wants a second opinion before showing a track record to an allocator.
It is not for someone looking for a strategy; it has none.

## Success looks like

- [x] A verdict on a single return series, or on a `T x N` matrix of candidate
      configurations, from one function call or one shell command.
- [x] Every statistic traceable to a named published paper, with the deviations
      from the paper written down in the docstring where they exist.
- [x] Default-deny: `DEPLOYABLE` only when *every* gate clears; degenerate or
      unmeasurable input is rejected, never waved through.
- [x] A rejection says how much more evidence would close the gap (MinTRL /
      MinBTL), not just "no".
- [x] Exit code `0` / `1` so it drops into a research pipeline or CI gate.
- [x] No network access, ever — the input is somebody's proprietary track record.

## Requirements

**Must**
- PSR, DSR, PBO (CSCV), MinTRL, MinBTL, and purged + embargoed walk-forward CV,
  faithful to the cited papers.
- A single `evaluate(...)` entry point returning one typed, frozen `Verdict`.
- Fail-closed behaviour on every degenerate path: probabilities collapse to 0,
  required-evidence statistics to infinity, an unrankable matrix to PBO = 1.
- Refuse input that is not a return series at all. A row counter rises every
  period with no drawdown, which is the highest Sharpe any column can have; left
  unchecked it wins the in-sample selection and is certified `DEPLOYABLE`. That
  single failure inverts the entire product, so the CSV loader screens for it.
- Offline. No telemetry, no phone-home, no upload.

**Should**
- A self-contained one-page report (HTML and Markdown) explaining each number in
  plain English, for archiving or emailing.
- Thresholds exposed as policy, not baked in as constants.
- Search-size measured from the candidate matrix where possible, rather than
  approximated from the winning column alone.

**Won't (this time)**
- Signal generation, parameter optimisation, position sizing, execution, or any
  broker connection.
- Multi-strategy portfolio construction or capital allocation.
- A hosted service, an account system, or any data leaving the machine.
- Reading a broker or vendor data feed. The user brings a CSV or an array.

## Explicitly out of scope

**Making money.** This tool has no signals, no optimiser, and no trading engine,
and it will not acquire them. It is a referee, not a player. The distinction is
load-bearing: the moment a validator also proposes strategies, its verdicts on
those strategies are worthless, and users are right to stop trusting the
verdicts on their own.

**Inventing statistics.** Every gate is somebody else's published mathematics,
cited in the code, the README, the reports and `--about`. A novel in-house
overfitting metric would be unfalsifiable marketing. Where the implementation
does deviate from a paper — deterministic average-linkage clustering in place of
López de Prado & Lewis's randomised k-means, for instance — the deviation is
documented in the docstring rather than quietly adopted.

**Deciding for the user.** The default thresholds (deflated Sharpe ≥ 0.95,
annualised Sharpe > 0.75, PBO ≤ 0.5) are a demanding *policy*, not published
constants, and they are exposed for tuning. The tool must not pretend its policy
is mathematics.

**Auditing an input the user did not supply.** No statistic can see trials that
are not in evidence. Submit only the winner of a large search plus near-copies
of it and the copies collapse to one effective trial, earning no deflation. This
is a real limit, not a bug, and the verdict says so in its own reasons whenever
the measurement collapses. Honest inputs are the price of measured deflation.

## Safety and privacy

- **Personal data:** none is collected. The input is a return series, which is
  commercially sensitive rather than personal — it can reveal a live strategy.
- **Who can see it:** only the local process. No network call is made anywhere
  in the package; dependencies are `numpy`, `pandas`, `scipy`. Nothing is
  written to disk except the report file the user explicitly names via
  `--report`. There is no cache, no log file, no config file, no history.
- **Revocation:** not applicable — there are no accounts, sessions or shared
  state to revoke. This is a stateless local computation.
- **Worst outcome if it is wrong:** a false `DEPLOYABLE` certifies an overfit
  strategy and someone commits capital to noise. That single asymmetry sets the
  whole design: every ambiguous, degenerate or unmeasurable case must resolve to
  a rejection. A false rejection costs a user some time; a false acceptance
  costs them money and, worse, teaches them to trust the tool.
- **The user's own honesty is inside the trust boundary.** See the last item
  under "out of scope". The tool states this limit in its output rather than
  implying an authority it does not have.

## Open questions

None outstanding for v0.3.0. Two things are deliberately unresolved rather than
unanswered:

- Whether the default thresholds should differ by asset class or holding period.
  They are policy; the answer is "the desk decides", and the knobs exist.
- Whether the effective-trials clustering should be validated against the
  paper's randomised k-means on a shared benchmark. It is currently justified by
  determinism and permutation-invariance, both machine-checked, not by
  agreement with the original implementation.

## Not doing / rejected alternatives

| Considered | Rejected because |
|---|---|
| Ship it as a module inside the author's trading engine | A validator inside the thing it validates has no credibility, and the coupling would drag a heavy research stack into a tool whose value is being small and auditable. |
| Annualise the Sharpe inside PSR/DSR | The papers are written in per-period units and the estimator variance term does not survive naive rescaling. Annualisation happens only at the reporting boundary. |
| Excess kurtosis (normal = 0) in the estimator variance | Bailey & López de Prado use non-excess kurtosis (normal = 3). Using excess silently inflates the deflated Sharpe — the flattering direction. |
| Calendar-day purging in walk-forward CV | On a business-day index a five-*calendar*-day purge removes three or four bars and leaks the tail of the label window. Purging is positional (bar count) for this reason. |
| The paper's randomised k-means for effective trials | Non-deterministic: the same matrix could earn different deflation on two runs, which is indefensible in a tool whose output is a go/no-go. Replaced by deterministic average-linkage on the paper's distance, with the deviation documented. |
| PBO's published default of `0.0` for degenerate input | `0.0` reads as "no overfitting" — optimistic, not fail-closed. The library keeps the published default for the standalone statistic, and `evaluate` passes `degenerate_value=1.0` so an unrankable matrix is rejected. |
| Accept any numeric CSV column as returns | `select_dtypes(["number"])` accepts the row counter `DataFrame.to_csv()` writes by default, and a counter has the highest Sharpe of any column. It would have been selected as the in-sample best and certified. Columns are now screened, and a false rejection is preferred because it is loud and fixable with `--column`. |
| A web UI or hosted service | Someone's live strategy returns would have to leave their machine. The offline guarantee is worth more than the convenience. |
| Silently falling back to the in-sample series when walk-forward CV yields no usable fold | A caller who asked to be judged out of sample would have been judged in sample and could be told `DEPLOYABLE`. For a tool that exists to refuse overfitted strategies this is the one failure it must not have; it now refuses instead. |
