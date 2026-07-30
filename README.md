# Lyra Validate

**An honest verdict on whether your backtest edge is real or overfit.**

Lyra Validate is a standalone Python tool you point at *your own* backtest
results. It reports how much of an apparent edge is statistically real versus an
artefact of luck, non-normal returns, or searching over many configurations.

> **Lyra Validate is a measurement tool, not a money-maker.** It does **not**
> generate trading signals, size positions, connect to a broker, or guarantee
> returns. It only tells you how much to trust a track record. Nothing here is
> investment advice.

## Who it's for

Quants, systematic traders, and researchers who already have a backtest and want
an honest second opinion *before* risking capital or showing results to an
allocator. If you have searched over many configurations, optimised parameters,
or simply want to know whether a promising Sharpe is signal or luck, point Lyra
Validate at your returns and get a default-deny verdict in seconds.

It is built entirely on **published, peer-reviewed mathematics** and cites it:

| Statistic | Reference |
|---|---|
| Probabilistic Sharpe Ratio (PSR) | Bailey & López de Prado (2012), *The Sharpe Ratio Efficient Frontier* |
| Deflated Sharpe Ratio (DSR) | Bailey & López de Prado (2014), *The Deflated Sharpe Ratio* |
| Probability of Backtest Overfitting (PBO via CSCV) | Bailey, Borwein, López de Prado & Zhu (2017), *The Probability of Backtest Overfitting* |
| Minimum Track Record Length (MinTRL) | Bailey & López de Prado (2012), *The Sharpe Ratio Efficient Frontier* |
| Minimum Backtest Length (MinBTL) | Bailey, Borwein, López de Prado & Zhu (2014), *Pseudo-Mathematics and Financial Charlatanism* |
| Purged + embargoed walk-forward CV | López de Prado (2018), *Advances in Financial Machine Learning*, ch. 7 |

Dependencies are limited to `numpy`, `pandas`, and `scipy`. No network access is
ever performed.

## Install

```bash
python -m pip install -e .
```

## Library usage

```python
import numpy as np
from lyravalidate import evaluate

rng = np.random.default_rng(0)
# Your strategy's per-period (e.g. daily) returns:
returns = 0.0008 + 0.01 * rng.standard_normal(1000)

verdict = evaluate(returns, n_trials=1)
print(verdict.classification)   # DEPLOYABLE / NOT_DEPLOYABLE / PROBABLY_OVERFIT
# This particular random sample misses both bars (annualised Sharpe 0.52 <= 0.75,
# deflated Sharpe 0.85 < 0.95) -> NOT_DEPLOYABLE, and summary() reports the
# evidence gap: 2537 observations needed at these moments, 1000 held.
# The worked example further down generates a stronger series that comes back DEPLOYABLE.
print(verdict.summary())
```

Pass a **`T x N` matrix** of candidate configurations (columns) to also get a PBO
estimate and an honest, search-aware deflation:

```python
candidates = rng.standard_normal((500, 50))   # 50 configs you tried
verdict = evaluate(candidates)                 # picks the best, deflates by 50
# best-of-50 pure noise: PBO 0.33 and the deflated Sharpe drops to 0.80,
# well under the 0.95 bar -> PROBABLY_OVERFIT
print(verdict.pbo, verdict.classification)
```

Or supply a signal's `predictions` and the `targets` it forecast for a purged
walk-forward out-of-sample check:

```python
verdict = evaluate(returns, predictions=preds, targets=fwd_returns)
print(verdict.oos_sharpe)
```

### The verdict

`evaluate(...)` returns a frozen `Verdict` dataclass:

```text
deployable: bool
classification: 'DEPLOYABLE' | 'NOT_DEPLOYABLE' | 'PROBABLY_OVERFIT'
deflated_sharpe: float    # probability the edge survives deflation
pbo: float                # probability of backtest overfitting (NaN if N/A)
sharpe: float             # annualised Sharpe
n_trials: int             # configurations assumed tried
n_periods: int            # finite observations judged (NaN/inf rows are not evidence)
min_track_record: float   # MinTRL: observations needed at the observed moments
min_backtest_years: float # MinBTL: years needed to beat a best-of-n_trials noise search
reasons: list[str]
```

The decision is **default-deny**: a result is only `DEPLOYABLE` when the deflated
Sharpe, the annualised Sharpe, and (when computable) the PBO all clear their bars.

### The evidence gap

A rejection also says *how much more evidence* would be needed. When the verdict
fails on significance, the reasons quantify the gap: how many observations (and
years, via `periods_per_year`) the record falls short of the closed-form
**Minimum Track Record Length** (MinTRL) — the length at which the PSR against
the same benchmark the deflation used would reach the significance bar, holding
the observed Sharpe, skew and kurtosis fixed. When the Sharpe does not exceed
that benchmark at all, no length ever suffices and the verdict says so
(`min_track_record` is `inf`). When significance passes, the verdict confirms
the observed length against MinTRL instead. For a searched record
(`n_trials > 1`), the **Minimum Backtest Length** (MinBTL) additionally reports
how many years of backtest are needed before the observed annualised Sharpe
even exceeds the expected best of that many pure-noise trials.

Only finite observations count as evidence: every statistic drops NaN/inf
entries (blank CSV cells load as NaN), and `n_periods` and the shortfall
arithmetic exclude them too, so padding a record with blank rows cannot shrink
the reported gap.

## CLI

Point it at a CSV of returns. A single returns column is treated as one strategy
(an optional leading `date` column is ignored); multiple numeric columns are
treated as a `T x N` matrix of candidate configurations.

The commands below use the demo CSV that ships in this repo. No CSV yet? Run
`python examples/run_example.py` to (re)generate `examples/sample_returns.csv`.

```bash
lyra-validate examples/sample_returns.csv                       # human-readable verdict
lyra-validate examples/sample_returns.csv --trials 20           # you searched over 20 configs
lyra-validate examples/sample_returns.csv --column returns      # evaluate one named column
lyra-validate examples/sample_returns.csv --report verdict.html # also write a one-page report
lyra-validate examples/sample_returns.csv --report verdict.md   # ...as Markdown (by suffix)
lyra-validate examples/sample_returns.csv --json                # machine-readable output
lyra-validate --about                                           # methodology + citations
```

The CLI exits `0` when the verdict is `DEPLOYABLE`, otherwise `1`, so it drops
straight into a research pipeline or a CI gate. The `--report` file is fully
self-contained (inline CSS, no external assets) and shows the big verdict, each
gate against its cutoff, and a plain-English explanation of what every number
means.

## Worked example

A runnable, fully-offline demo lives in [`examples/run_example.py`](examples/run_example.py).
It generates a deterministic `examples/sample_returns.csv`, evaluates it (and
writes an HTML and a Markdown report), re-judges the *same* strategy on only its
first 90 days to show the evidence-gap report, and finally -- as a cautionary
tale -- evaluates a 50-configuration noise matrix to show how a hard search over
noise gets flagged:

```bash
python examples/run_example.py
```

```text
=== 1) An honest single strategy ===
Verdict: DEPLOYABLE (deployable=True)
  Deflated Sharpe : 1.000 (probability the edge is real after deflation)
  Annualised Sharpe: 1.660
  PBO             : n/a
  Trials assumed  : 1
  MinTRL          : 253 obs needed; have 1500 (sufficient)
  Reasons:
    - MinTRL confirmed: 1500 observations against a minimum of 253 for the 0.95 bar at the observed moments.
    - Clears the deflated-Sharpe, Sharpe and PBO bars.

=== 2) The same strategy, judged on only its first 90 days ===
Verdict: NOT_DEPLOYABLE (deployable=False)
  Deflated Sharpe : 0.586 (probability the edge is real after deflation)
  Annualised Sharpe: 0.366
  PBO             : n/a
  Trials assumed  : 1
  MinTRL          : 5070 obs needed; have 90 (short 4980 obs ~ 19.8 years)
  Reasons:
    - Deflated Sharpe 0.586 < 0.95: the Sharpe is not significant after deflating for trials and non-normality.
    - Evidence gap: 5070 observations are needed at the observed moments to reach the 0.95 bar; the record has 90 - short about 4980 observations (~19.8 years).
    - Annualised Sharpe 0.366 <= 0.75: insufficient risk-adjusted return.

=== 3) A cautionary tale: 50 noise configurations ===
Verdict: PROBABLY_OVERFIT (deployable=False)
  Deflated Sharpe : 0.519 (probability the edge is real after deflation)
  Annualised Sharpe: 1.355
  PBO             : 0.495
  Trials assumed  : 50
  MinTRL          : 914221 obs needed; have 750 (short 913471 obs ~ 3624.9 years)
  MinBTL          : 2.8 years needed for 50 trials; have 3.0
  Reasons:
    - Deflated Sharpe 0.519 < 0.95: the Sharpe is not significant after deflating for trials and non-normality.
    - Evidence gap: 914221 observations are needed at the observed moments to reach the 0.95 bar; the record has 750 - short about 913471 observations (~3624.9 years).
```

Scene 2 is the evidence-gap report doing its job: the same edge that is
`DEPLOYABLE` on 1500 observations is indistinguishable from luck on 90, and
instead of a bare "no" the verdict says the record is about 4980 observations
(~19.8 years) short of certifying the modest Sharpe it has shown *so far*. Weak
observed Sharpes need enormous track records -- that is the closed-form MinTRL
arithmetic, not an opinion.

Scene 3 still shows a healthy-looking annualised Sharpe of 1.36, but a deflated
Sharpe near a coin flip and a PBO near 0.5 -- exactly the signature of
overfitting. Its MinTRL makes the evidence gap concrete: at these moments, a
best-of-50 selection would need thousands of years of data before this Sharpe
became significant.

## Thresholds are policy, not law

The default bars (`deflated_sharpe >= 0.95`, `sharpe > 0.75`, `pbo <= 0.5`) are a
sensible, demanding *policy* — not published constants. Tune them via the
`Thresholds` dataclass or the corresponding CLI flags.

## Numerical conventions

Faithful to the source papers: per-period (non-annualised) Sharpe inside PSR/DSR;
*non-excess* kurtosis (a normal sample is 3); biased Fisher–Pearson skew; the
Euler–Mascheroni constant in the expected-maximum-Sharpe benchmark; natural-log
logits and `rank / (N + 1)` in CSCV; **positional** (bar-count) purging so a
business-day index is handled correctly. Degenerate inputs **fail closed**:
probabilities return 0, and required-evidence statistics (MinTRL/MinBTL) return
infinity.

## Machine-checked invariants

The suite is 116 tests. Most anchor on hand-computed values from the source
papers; on top of those, a property-based layer
([`tests/test_properties.py`](tests/test_properties.py), Hypothesis as a
dev-only dependency) machine-checks the mathematical contract on arbitrary
valid inputs -- the return-series cases with NaN/inf entries injected, since
the statistics are contracted to drop non-finite observations:

- PSR, DSR and PBO are probabilities: always in `[0, 1]`.
- PSR is monotonically non-increasing in the benchmark Sharpe.
- DSR is non-increasing in `n_trials`, and `DSR <= PSR` for `n_trials > 1`:
  deflation can only ever lower a probability.
- PBO is invariant under column permutation of the candidate matrix (CSCV must
  not care which order the configurations were tried in).
- The purged, embargoed walk-forward splitter never leaves a training index
  within `label_horizon` of the evaluation window, for arbitrary valid window
  parameters.

The Hypothesis profile is deterministic (`derandomize=True`) with bounded
example counts, so the suite stays fast and reproducible run-to-run.

## FAQ

**Does this make money or find strategies?**
No. Lyra Validate has no signals, no optimiser, no broker, and no trading engine.
It *measures* whether a track record you already have is statistically real or
likely overfit. It is a referee, not a player.

**Are these your own statistics?**
No — and that is the point. The hard mathematics (Probabilistic Sharpe, Deflated
Sharpe, PBO via CSCV, purged walk-forward CV) is published, peer-reviewed work by
Bailey, Borwein, López de Prado and Zhu. Lyra Validate is a clean, faithful,
tested re-implementation of those public methods, cited above.

**What's the difference between the Sharpe and the *Deflated* Sharpe?**
The raw Sharpe says nothing about how many things you tried. The Deflated Sharpe
is the probability the true Sharpe is positive *after* correcting for the number
of configurations searched, the length of the record, and non-normal (fat-tailed)
returns. A great-looking Sharpe with a low Deflated Sharpe is the classic
fingerprint of overfitting.

**Why is the verdict "default-deny"?**
Because the costly error in research is deploying a false edge, not skipping a
real one. A result is only `DEPLOYABLE` when *every* gate clears; anything
ambiguous is rejected.

**What does a `DEPLOYABLE` verdict promise?**
Nothing about the future. It means the record is statistically *consistent with*
a real edge rather than luck. Always confirm forward with paper or live trading
before committing capital.

**Does it phone home or need the internet?**
Never. Everything runs locally on the file you provide. Dependencies are limited
to `numpy`, `pandas`, and `scipy`.

**Can I change the thresholds?**
Yes — they are policy, not law. Tune them via the `Thresholds` dataclass or the
`--min-deflated-sharpe`, `--min-sharpe`, and `--max-pbo` CLI flags.

## License

Proprietary — All Rights Reserved. See [LICENSE](LICENSE). The underlying
published mathematics is not claimed; this implementation and product are.
