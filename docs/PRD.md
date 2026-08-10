# BacktestAudit — product requirements

Recorded after v0.3.0 shipped, from the code. Where this and the README
disagree, the code wins. See [TDD](TDD.md) and [Design Brief](DESIGN_BRIEF.md).

## The easiest number in finance to fake

Fake without meaning to, that is. Search over two hundred parameter
combinations, keep the best, and the winner's Sharpe ratio is inflated by the
search itself — by an amount nobody eyeballs correctly.

The corrections exist and are published: Bailey and López de Prado's
Probabilistic and Deflated Sharpe Ratios, and the Probability of Backtest
Overfitting via CSCV. But they are scattered across six papers, each with its own
notational traps — per-period versus annualised Sharpe, excess versus non-excess
kurtosis, calendar-day versus bar-count purging — and getting any of them wrong
silently makes the answer more flattering, never less.

Concretely: the author's own systematic-trading research kept producing
backtests that looked deployable and had no honest referee. The person with that
problem is the person who wrote the strategy, which is exactly the person least
able to judge it.

## A row counter would have been certified DEPLOYABLE

This is not hypothetical, and it is the sharpest thing this project has learned
about itself.

`DataFrame.to_csv()` writes an index column by default. A row counter rises every
period and never draws down, which is the highest Sharpe any column can possibly
have. An early loader used `select_dtypes(["number"])` to find return columns, so
the counter was eligible; it would have won the in-sample selection and come back
`DEPLOYABLE`.

One accepted CSV convention, and the product inverts. So the loader now screens
columns for whether they are plausibly a return series at all, and a false
rejection is preferred to a false acceptance — a rejection is loud and fixable
with `--column`.

## Everything ambiguous resolves toward refusal

A false `DEPLOYABLE` certifies an overfit strategy and someone commits capital to
noise. A false rejection costs a user some time. That asymmetry is the whole
design, and it shows up everywhere:

- `DEPLOYABLE` only when *every* gate clears. Default deny.
- Degenerate paths collapse in the pessimistic direction: probabilities to 0,
  required-evidence statistics to infinity, an unrankable matrix to PBO = 1.
- Where the published default is optimistic, it is overridden at the call site
  rather than in the library. PBO's published degenerate default is `0.0`, which
  reads as "no overfitting"; the standalone statistic keeps it, and `evaluate`
  passes `degenerate_value=1.0`.
- When walk-forward CV yields no usable fold, it refuses. Silently falling back
  to the in-sample series would judge a caller in sample who asked to be judged
  out of sample, and could return `DEPLOYABLE`. For a tool that exists to refuse
  overfitted strategies, that is the one failure it must not have.

A false acceptance costs money and, worse, teaches the user to trust the tool.

## Requirements

**Must**

- PSR, DSR, PBO (CSCV), MinTRL, MinBTL, and purged plus embargoed walk-forward
  CV, faithful to the cited papers.
- A single `evaluate(...)` entry point returning one typed, frozen `Verdict`.
- Fail-closed behaviour on every degenerate path.
- Refuse input that is not a return series at all.
- Offline. No telemetry, no phone-home, no upload.

**Should**

- A self-contained one-page report, HTML and Markdown, explaining each number in
  plain English, for archiving or emailing.
- Thresholds exposed as policy rather than baked in as constants.
- Search size measured from the candidate matrix where possible, rather than
  approximated from the winning column alone.

**Deliberately absent**

- Signal generation, parameter optimisation, position sizing, execution, or any
  broker connection.
- Multi-strategy portfolio construction or capital allocation.
- A hosted service, an account system, or any data leaving the machine.
- Reading a broker or vendor data feed. The user brings a CSV or an array.

## Four things it must never become

**A player.** No signals, no optimiser, no trading engine, and it will not
acquire them. It is a referee. The distinction is load-bearing: the moment a
validator also proposes strategies, its verdicts on those strategies are
worthless, and users are right to stop trusting the verdicts on their own.

**An inventor of statistics.** Every gate is somebody else's published
mathematics, cited in the code, the README, the reports and `--about`. A novel
in-house overfitting metric would be unfalsifiable marketing. Where the
implementation does deviate from a paper — deterministic average-linkage
clustering in place of López de Prado and Lewis's randomised k-means, for
instance — the deviation is documented in the docstring rather than quietly
adopted.

**A decision-maker.** The default thresholds (deflated Sharpe ≥ 0.95, annualised
Sharpe > 0.75, PBO ≤ 0.5) are a demanding *policy*, not published constants, and
they are exposed for tuning. The tool must not pretend its policy is
mathematics.

**An auditor of what it was not shown.** No statistic can see trials that are not
in evidence. Submit only the winner of a large search plus near-copies of it, and
the copies collapse to one effective trial, earning no deflation. That is a real
limit rather than a bug, and the verdict says so in its own reasons whenever the
measurement collapses. Honest inputs are the price of measured deflation — the
user's own honesty sits inside the trust boundary, and the tool states that limit
in its output instead of implying an authority it does not have.

## Who brings the data, and where it goes

The researcher who already has a return series and is about to decide whether to
risk money on it. Primarily the author; secondarily any quant or systematic
trader who wants a second opinion before showing a track record to an allocator.
It is not for someone looking for a strategy, because it has none.

No personal data is collected. The input is a return series — commercially
sensitive rather than personal, since it can reveal a live strategy. Only the
local process ever sees it: no network call is made anywhere in the package, the
dependencies are `numpy`, `pandas` and `scipy`, and nothing is written to disk
except the report file the user explicitly names with `--report`. No cache, no
log file, no config file, no history. Revocation does not apply, because there
are no accounts, sessions or shared state; this is a stateless local
computation.

## Marks of done

- [x] A verdict on a single return series, or on a `T x N` matrix of candidate
      configurations, from one function call or one shell command.
- [x] Every statistic traceable to a named published paper, with any deviations
      written down in the docstring.
- [x] Default-deny, with degenerate or unmeasurable input rejected rather than
      waved through.
- [x] A rejection says how much more evidence would close the gap (MinTRL,
      MinBTL), not just "no".
- [x] Exit code `0` / `1`, so it drops into a research pipeline or a CI gate.
- [x] No network access, ever.

## Where a defensible-looking choice was refused

| The choice | Why it was refused |
|---|---|
| Ship it as a module inside the author's trading engine | A validator inside the thing it validates has no credibility, and the coupling would drag a heavy research stack into a tool whose value is being small and auditable. |
| Annualise the Sharpe inside PSR/DSR | The papers are written in per-period units and the estimator variance term does not survive naive rescaling. Annualisation happens only at the reporting boundary. |
| Excess kurtosis (normal = 0) in the estimator variance | Bailey & López de Prado use non-excess kurtosis (normal = 3). Using excess silently inflates the deflated Sharpe — the flattering direction. |
| Calendar-day purging in walk-forward CV | On a business-day index a five-*calendar*-day purge removes three or four bars and leaks the tail of the label window. Purging is positional, by bar count, for this reason. |
| The paper's randomised k-means for effective trials | Non-deterministic: the same matrix could earn different deflation on two runs, which is indefensible in a tool whose output is a go/no-go. Replaced by deterministic average-linkage on the paper's distance, with the deviation documented. |
| A web UI or hosted service | Someone's live strategy returns would have to leave their machine. The offline guarantee is worth more than the convenience. |

## Left unresolved on purpose

Neither blocks v0.3.0, and both are unresolved on purpose rather than
unanswered.

Whether the default thresholds should differ by asset class or holding period.
They are policy; the answer is that the desk decides, and the knobs exist.

Whether the effective-trials clustering should be validated against the paper's
randomised k-means on a shared benchmark. It is currently justified by
determinism and permutation-invariance, both machine-checked, rather than by
agreement with the original implementation.
