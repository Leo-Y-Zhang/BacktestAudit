# BacktestAudit — technical design

v0.3.0 as implemented, derived from the source rather than the README. Line
references are to `src/backtestaudit/`. Requirements: [PRD.md](PRD.md).

## Shape

A pure-computation Python library with a thin CLI on top. No database, no
server, no state surviving the process, no network call anywhere in the package.

One public entry point, `evaluate(...)`, reduces whatever the caller supplies to
a single judged return series, computes six published statistics over it, applies
three policy gates, and returns one frozen `Verdict` dataclass. Everything else —
the CLI, the HTML and Markdown reports — is a rendering of that dataclass.

| Module | Lines | Responsibility |
|---|---|---|
| `stats.py` | 949 | The published statistics. No policy, no I/O. PSR, DSR, PBO (CSCV), MinTRL, MinBTL, Sharpe estimator standard error, effective-trials clustering, cross-trial dispersion. |
| `crossval.py` | 170 | `PurgedWalkForwardSplitter` — positional purging plus embargo, and an auto-sized default. |
| `evaluate.py` | 553 | Policy. `Thresholds`, `Verdict`, `evaluate(...)`. The only module that decides anything. |
| `report.py` | 499 | Rendering of a `Verdict` to Markdown or self-contained HTML. Also holds `DISCLAIMER` and `CITATIONS`, which the CLI reuses. |
| `cli.py` | 302 | Argument parsing, CSV loading and column screening, exit codes. |
| `_version.py` | 10 | Single source of the version, separate so `report.py` can import it without a circular import through `__init__`. |

Dependency direction is strictly one way — `cli → report → evaluate → {crossval,
stats}` — and `stats.py` imports nothing from the package.

## Every degenerate path, and where it lands

For a tool built on "a false accept is worse than a false reject", this table is
the design rather than an appendix to it.

| What breaks | Who notices | How it is detected | How it is undone |
|---|---|---|---|
| Fewer than 4 finite observations, or zero variance | the caller | `_sharpe_moments` returns `None`; PSR/DSR collapse to `0.0`, `sharpe_standard_error` to `inf`, MinTRL to `inf` | rejection with an explicit "record is too degenerate to measure" reason |
| Extreme skew/kurtosis drives the estimator variance non-positive or `NaN` | the caller | same guard — the check demands finite *and* positive, because a bare `<= 0` lets `NaN` through | same |
| Candidate matrix unrankable (`N < 2`, `T < 4`, no usable partition) | the caller | PBO returns `degenerate_value`, which `evaluate` sets to `1.0` | PBO gate fails; verdict is `PROBABLY_OVERFIT` |
| Matrix too short to measure a search (`< 100` complete rows, or rows ≤ columns) | the caller, via `effective_trials` being `None` | `cluster_trials` declines | falls back to the published raw-count deflation — never a *weaker* assumed search |
| Configurations are near-duplicates, so the measured deflation is nearly nil | the caller, via the reasons | measured cross-trial dispersion below half the selected series' own Sharpe estimator noise | a caveat reason naming the trust model and telling the caller to pass `n_trials` |
| Walk-forward CV yields no usable fold after purging | the caller | `_walk_forward_oos` returns `None` | verdict fails closed with a reason saying every figure shown is in-sample and is *not* evidence about held-out performance |
| A supplied CSV column is a row counter, index, date or price level | the caller | `_not_returns_reason`: strictly monotone over ≥ 8 finite steps, or a magnitude above 10.0 | column dropped with a note to stderr; if nothing survives, exit 2 telling the user to name the column with `--column` |
| `--trials 0` or a negative typo | the caller | `_positive_int` argparse type | usage error, exit 2. Previously it floored to 1 downstream, which both disabled deflation and silently switched off the matrix-measured benchmark |
| Confidence bar set to 1.0 | the caller | `Phi^-1(1)` is infinite | MinTRL `inf` with a reason naming the bar, distinct from the "Sharpe below benchmark" case |
| Report path unwritable | the caller | `OSError` around `write_report` | exit 2 after the verdict has already printed |

There is no alerting and no telemetry, by design. Detection is the caller reading
the output, plus CI.

## Input contract

No database, no migrations, no persisted schema. Two structures carry
everything, and the first is what the caller hands over.

| Shape | Interpretation | Consequences |
|---|---|---|
| 1-D array | one strategy's per-period returns | PBO is `NaN` (not computable); `n_trials` defaults to `1` |
| 2-D `T x N`, `N >= 2` | candidate matrix, one column per configuration | PBO computed across columns; the column with the highest annualised Sharpe is selected and judged; `n_trials` defaults to `N` |
| 2-D with `N <= 1` | flattened to the 1-D case | |
| anything else | `ValueError` | |

**Non-finite entries are not evidence.** Every statistic drops `NaN` and `inf`
before computing, `Verdict.n_periods` counts only finite observations, and the
evidence-gap arithmetic uses that same count — so padding a CSV with blank rows
cannot shrink the reported shortfall.

PBO is the one statistic that drops whole *rows* rather than per-column entries,
because CSCV ranks columns against each other and a single blank cell would
otherwise make `np.argmax` select the `NaN` column as the in-sample best in every
partition.

## The Verdict

Frozen dataclass, `evaluate.py`.

| Field | Type | Meaning / null case |
|---|---|---|
| `deployable` | `bool` | true only if no gate failed |
| `classification` | `'DEPLOYABLE' \| 'NOT_DEPLOYABLE' \| 'PROBABLY_OVERFIT'` | |
| `deflated_sharpe` | `float` | gate 1 |
| `sharpe` | `float` | annualised; gate 2 |
| `pbo` | `float` | gate 3; **`NaN` when a single series was supplied** — the gate is then skipped, not failed |
| `n_trials` | `int` | trials assumed for the deflation |
| `reasons` | `list[str]` | human-readable, always populated |
| `probabilistic_sharpe` | `float` | diagnostic, not a gate |
| `n_periods` | `int` | finite observations judged |
| `periods_per_year` | `int` | annualisation factor, default 252 |
| `min_track_record` | `float` | MinTRL. `inf` = unreachable (three distinct causes, distinguished in `reasons`); `NaN` = not computed |
| `min_backtest_years` | `float` | MinBTL. `inf` = observed Sharpe not positive |
| `effective_trials` | `int \| None` | **`None` unless** a matrix was judged without an explicit `n_trials` |
| `cross_trial_sharpe_std` | `float \| None` | `None` when only one effective trial exists — a single trial has no dispersion |
| `oos_sharpe`, `oos_information_coefficient` | `float \| None` | `None` unless `predictions`/`targets` were supplied and at least one fold survived purging |

The three `| None` fields are absent on most verdicts. Consumers — `report.py`
and the CLI JSON included — must branch on them, and do.

## `evaluate()`, and four contracts the signature does not show

```python
evaluate(
    returns,                       # 1-D series or T x N matrix
    predictions=None, targets=None,# optional paired signal + forward returns
    *,
    n_trials: int | None = None,   # None => inferred; explicit => raw-count deflation
    periods_per_year: int = 252,   # must be > 0
    thresholds: Thresholds | None = None,
    pbo_splits: int = 16,          # CSCV blocks, forced even and >= 2
    splitter: PurgedWalkForwardSplitter | None = None,
) -> Verdict
```

**`n_trials=None` on a matrix changes the mathematics, not just a default.**
Omitted, the matrix is taken to *be* the whole search and the DSR benchmark is
measured from it — correlation clusters as effective trials, cross-trial Sharpe
dispersion as the scale (López de Prado & Lewis 2019). Supplied, the caller is
asserting a search larger than the matrix, and the published raw-count
approximation is used instead. Both are correct; they answer different questions.

**MinTRL shares the DSR's benchmark `SR*`.** That is deliberate, and it is what
makes `n_periods >= min_track_record` equivalent to the deflated-Sharpe gate
clearing, for any confidence above 0.5. Computing MinTRL against a different
benchmark would produce an evidence gap that contradicts the verdict.

**MinBTL deliberately uses the raw trials-tried count**, not the effective count.
The published bound is defined for the number of configurations tried.

**Supplying `predictions` and `targets` *is* the request to be gated out of
sample.** The OOS series replaces the in-sample one as the basis for every
statistic, and the matrix-measured benchmark is discarded, because its units
belong to the in-sample search rather than to the walk-forward series. The two
must arrive together: one without the other is that same request with no way to
honour it, so `evaluate` raises `ValueError` rather than quietly gating the
in-sample series and answering a different question.

`Thresholds(min_deflated_sharpe=0.95, min_sharpe=0.75, max_pbo=0.5)` is policy —
frozen, injectable. `probability_of_backtest_overfitting(...)` keeps the
published `degenerate_value=0.0` as a standalone statistic, and `evaluate` calls
it with `degenerate_value=1.0` so an unrankable matrix is rejected.

## CLI

`backtestaudit PATH [--column NAME] [--trials N] [--periods-per-year K]
[--pbo-splits S] [--report OUT] [--json] [--min-deflated-sharpe X]
[--min-sharpe X] [--max-pbo X] [--about] [--version]`

| Exit | When |
|---|---|
| `0` | verdict is `DEPLOYABLE`; also `--about` and `--version` |
| `1` | any other verdict — the CI-gate signal |
| `2` | input could not be read or judged: missing/unreadable file, unparseable CSV, no numeric column, named column absent, no column that could be returns, `ValueError` from `evaluate`, or the report file could not be written; also argparse usage errors |

Stdout carries the verdict, or the JSON. The report-written confirmation and the
ignored-columns note go to **stderr**, so `--json --report` still emits parseable
JSON on stdout. JSON output maps every non-finite float to `null`, because `NaN`
and `Infinity` are not valid JSON.

## Security properties, in the absence of access control

There is no server, no database, no authentication, no session, no multi-tenancy
and no shared state. The security boundary is the local process, and four
properties describe it.

No network access anywhere in the package: `numpy`, `pandas` and `scipy` are the
only dependencies, and none is used for I/O beyond `pandas.read_csv` on a local
path.

No implicit writes. The only file written is the one named by `--report`. No
cache, log, config or history file.

No `eval`, no pickle, no dynamic import. Input is parsed as CSV by pandas and
coerced to `float64`.

Supply chain is the residual risk — three pinned-by-floor dependencies and a
gitleaks job in CI. Dependency advisories must be re-swept, never quoted from the
last run.

## The 0.3.0 numeric change, and rolling back across it

Nothing is persisted, so rollback is otherwise trivial and total: `pip install`
the previous version, or `git revert`. No schema exists, and no other process
holds state produced by this one. Reports already written are static files and
stay valid — each stamps the version that produced it, which is why
`_version.py` exists as a separate module.

The exception is recorded in `CHANGELOG.md` for 0.3.0. Matrices judged without an
explicit `n_trials` now deflate against a *measured* benchmark rather than an
approximated one, so the same input can return a different verdict across that
version boundary. The 1-D path is unchanged. Reports stamped `v0.3.0` were
produced against the measured benchmark and should not be compared directly with
pre-0.3.0 ones.

## What the 166 tests pin down

166 tests, `pytest -q`. CI runs `ruff check src tests`, `mypy src` (strict,
`python_version = 3.12`) and `pytest -q` on Python 3.13, with a 15-minute job
timeout and a gitleaks job.

**Anchored.** `test_stats.py` (672 lines) pins hand-computed values from the
source papers, so a refactor that changes a formula fails loudly rather than
drifting.

**Negative and fail-closed.** Degenerate records, near-constant returns,
unrankable matrices, the `--trials 0` rejection, the not-returns column screen,
and the "OOS requested but unavailable" refusal each have a test that would pass
if the guard were removed and the tool merely got quieter. That is the class of
regression this suite exists to catch.

**Property-based.** `test_properties.py` uses Hypothesis on a deterministic
profile (`derandomize=True`, dev-only) to machine-check the contract on arbitrary
valid inputs with `NaN` and `inf` injected: PSR, DSR and PBO stay in `[0, 1]`;
PSR is non-increasing in the benchmark; `DSR <= PSR` for `n_trials > 1`; PBO is
invariant under column permutation; `cluster_trials` always returns a partition
of the usable columns; the matrix-faithful DSR never exceeds the selected
column's undeflated PSR; and the splitter never leaves a training index within
`label_horizon` of the evaluation window.

**Rendering.** `test_report.py` and `test_cli.py` cover exit codes, JSON validity
with non-finite values present, format inference by suffix, and stdout/stderr
separation.

## The fourth document, and why it is not here

The estate standard asks for four documents. This project has three: this TDD,
the [PRD](PRD.md), and the [Design Brief](DESIGN_BRIEF.md). There is no App Flow,
deliberately.

An App Flow enumerates screens, states, and every transition between them. There
is one non-interactive invocation here — arguments in, one verdict out, process
exits — with no screens, no navigation, no session and no authorisation state.
The transitions such a document would capture are the input-rejection paths, the
three verdicts and the exit codes, and all three are already tabulated above in
*Every degenerate path, and where it lands* and *CLI*. Restating those tables
under UI headings would create a second copy to keep true, and the copy would be
the one that rots, because the tables above are the ones the tests are written
against.

The Design Brief closes with the same statement from the other side; this
paragraph exists so a reader who only opens the TDD does not have to infer that
the omission was a decision rather than an oversight.

## A known gap

`default_walk_forward_splitter` can auto-size a window longer than the data — any
`n <= 7` — and `PurgedWalkForwardSplitter.split` then raises `ValueError` out of
`evaluate`. That is the same situation as "no usable fold", in that the requested
out-of-sample basis does not exist, but it is handled by raising rather than by
the documented fail-closed refusal, and `evaluate`'s docstring does not mention
that it can raise from the splitter.

Library callers only; the CLI never reaches this path. Reproduced 2026-08-03 with
a 7-observation series:
`ValueError: Not enough observations (7) for one fold (need at least 8)`.
