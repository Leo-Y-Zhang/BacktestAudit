# Design Brief — the verdict report

Scope: the one-page report produced by `--report` (`report.py`, `render_html`
and `render_markdown`) and the terminal summary printed by `Verdict.summary()`.
There is no application UI; the last section says why. See [PRD.md](PRD.md) and
[TDD.md](TDD.md).

## The reader is motivated to misread it

Someone who has just spent weeks on a strategy, hoping for a pass. That is the
single fact this page is designed around, and it produces two failure modes:
reading the big green banner and stopping, or seeing a rejection and dismissing
the tool.

The layout answers both by making the *reasons* unavoidable. Verdict word, then
the gate table with each value against its cutoff, then plain-English
explanations of what each number means, then the reasons list, then the
disclaimer and the citations. Nothing is behind a toggle.

## A pathology report, not a dashboard

The reader should feel they are being told the truth by something with no stake
in the answer. Calm, dense, factual, finished — a page you could print, staple
to a research log, and defend to an allocator six months later.

It must never feel like a trading product. No gauges, no sparklines, no gradient
hero, no green upward arrow, no confetti on a pass. Every visual flourish that
says "you have made money" is a lie in a tool that measures whether you can trust
a number, and the verdict most users get is a rejection.

## Refusals

**Colour as the only carrier of the verdict.** Enforced: the classification is
the `<h1>` text, and each gate row carries the literal word `PASS`, `FAIL` or
`N/A`, so a greyscale print or a colour-blind reader loses nothing.

**External assets.** No `<script>`, no web font, no image, no stylesheet link, no
tracking pixel. The HTML is a single self-contained document, because anything
fetched would leak the fact that someone opened a report about their strategy.
That is a privacy property enforced by the renderer's structure, not a
preference.

**Softening a rejection.** `PROBABLY_OVERFIT` gets amber rather than red, and
that is the entire concession. The subtitle still reads "This looked good
in-sample but does not survive honest deflation."

**Precision theatre.** Three decimal places on probabilities, one on years. More
digits would imply an accuracy the estimator does not have.

**A summary that omits the caveats.** The trust-model caveat — near-duplicate
configurations earning almost no deflation — is rendered in the same list as
every other reason, at the same weight. It is the finding most damaging to the
tool's apparent authority, so it must not be demoted.

## Borrowed from

A clinical lab report: the result leads, the reference range sits beside every
value, and the interpretation is written out in words. That reference range is
the thing being borrowed — a number without its cutoff is unreadable, so the gate
table carries `Value` and `Cutoff` in adjacent columns.

A compiler diagnostic: it names the rule, the observed value, and what to do
next. The `reasons` list follows that shape. "Evidence gap: 5070 observations are
needed at the observed moments to reach the 0.95 bar; the record has 90 — short
about 4980 observations (~19.8 years)" is a diagnostic, not an error message.

A paper's methods section: citations on the artefact itself rather than on a
website. Every report carries the six references and the disclaimer inline, so
the file stays self-explanatory when it is forwarded with no context.

## Content states

There are no interactive states — no links, no buttons, no scripts, no form
controls — so hover, focus, active, disabled and loading have nothing to bite on.
What exists is content.

| State | Rendered as |
|---|---|
| `DEPLOYABLE` | green banner, subtitle "The track record is statistically consistent with a real edge", and a meaning paragraph that explicitly refuses to promise future profit |
| `PROBABLY_OVERFIT` | amber banner, "This looked good in-sample but does not survive honest deflation" |
| `NOT_DEPLOYABLE` | red banner, "The evidence does not support a real, repeatable edge" |
| PBO not computable (single series) | the gate row is still shown, valued `n/a`, status `N/A`, and the explanation says how to make it computable — supply a `T x N` matrix. An absent row would read as a passed gate |
| Diagnostics absent (`effective_trials`, OOS figures) | rows are omitted rather than shown empty; the block itself is dropped if nothing survives |
| No reasons | the section is omitted. In practice `evaluate` always populates at least one reason, including on a pass |

## Colour, measured

Roles first. The report is deliberately light-only (`color-scheme: light`),
because it is an archival document that will be printed and emailed as often as
it is viewed. Ratios were computed against the actual hex values in `report.py`
on 2026-08-03.

| Role | Value | On | Contrast | Floor |
|---|---|---|---|---|
| Page surround | `#f3f4f6` | — | — | — |
| Card surface | `#ffffff` | — | — | — |
| Body text | `#1f2937` | `#ffffff` | **14.68:1** | 4.5 |
| Body text | `#1f2937` | `#f3f4f6` | **13.34:1** | 4.5 |
| Definition text | `#374151` | `#ffffff` | **10.31:1** | 4.5 |
| Muted (meta, table headers, diagnostics labels) | `#6b7280` | `#ffffff` | **4.83:1** | 4.5 |
| Status PASS | `#0f7b3f` | `#ffffff` | **5.35:1** | 4.5 |
| Status FAIL | `#b91c1c` | `#ffffff` | **6.47:1** | 4.5 |
| Status N/A | `#6b7280` | `#ffffff` | **4.83:1** | 4.5 |
| Banner `DEPLOYABLE` | `#ffffff` on `#0f7b3f` | | **5.35:1** | 4.5 |
| Banner `PROBABLY_OVERFIT` | `#ffffff` on `#b45309` | | **5.02:1** | 4.5 |
| Banner `NOT_DEPLOYABLE` | `#ffffff` on `#b91c1c` | | **6.47:1** | 4.5 |
| Banner fallback (unknown class) | `#ffffff` on `#374151` | | **10.31:1** | 4.5 |

Every text pair clears 4.5:1, including the two closest to the line — `#6b7280`
muted text at 4.83:1 and the amber banner at 5.02:1. Those are the two to
re-check if anyone ever "brightens" the palette.

Hairlines (`#e5e7eb` borders at 1.24:1, `#eef0f2` row rules at 1.14:1) do **not**
meet the 3:1 UI-boundary floor. Accepted deliberately: they separate rows that
are already separated by whitespace and by their own text, and no information
depends on perceiving them. Raising them would give the page a gridded,
spreadsheet-like feel that works against the intent. A conscious exception, not
an oversight.

## Type and layout

One family: the platform UI stack (`-apple-system`, `Segoe UI`, `Roboto`,
Helvetica, Arial, sans-serif). A downloaded font would break the
no-external-assets rule, and a document that renders in the reader's own system
face reads as a report rather than as a brand.

Body 1rem at line-height 1.55, measure capped at 820px. Verdict `<h1>` at
2.1rem with tight tracking (-0.01em). Section `<h2>` at 1.15rem with a hairline
rule beneath. Overlines and table headers 0.75–0.8rem, uppercase, 0.06–0.12em
tracking. `font-variant-numeric: tabular-nums` on the value and cutoff cells and
on the diagnostics values, so figures align down the column and a changed digit
is visible at a glance.

A single centred column on a neutral surround, with 1.4–1.8rem of padding inside
the card and sections separated by a 1.6rem top margin plus a hairline. The
diagnostics block is the one two-dimensional element: a `1fr auto` grid, label
muted and left-aligned, value tabular and right-aligned — the densest honest way
to show a dozen supplementary figures without implying they are gates.

## Accessibility, and one unverified risk

**Contrast** — every text pair measured above; all clear 4.5:1, hairlines a
documented exception.

**Colour never the only signal** — verdict word in the `<h1>`, and
`PASS`/`FAIL`/`N/A` as literal text in every gate row.

**Keyboard** — nothing to operate; the document is scroll-only, and no
`outline: none` appears anywhere in the CSS.

**Reduced motion** — no motion, no transition, no animation.

**Semantics** — `<main>`, `<header>`, a real `<table>` with `<thead>` and `<th>`
headers for the gates, `<dl>`/`<dt>`/`<dd>` for the explanations, `<ol>` for the
references, `lang="en"`, `<meta name="viewport">`, and a `<title>` carrying the
project name and the verdict.

**Zoom to 200%** — `max-width: 820px` caps rather than forces, so the single
column narrows with the viewport instead of scrolling sideways, subject to the
risk below.

That risk, not yet verified in a browser: the gates `<table>` has no
`overflow-x` wrapper, and the value and cutoff cells are `white-space: nowrap`.
At a 320px viewport the available content width is roughly 230px after the body
and card padding, while the table's minimum content width — the longest
unbreakable word of "Probability of Backtest Overfitting (PBO)" plus three nowrap
columns plus cell padding — is close to that. It is the first thing to check at
320px, and the fix if it overflows is a wrapping `div` with `overflow-x: auto`,
not narrower text.

## The other two surfaces

**Markdown.** Same content, same order, same wording, no styling: heading,
verdict, gate table, meaning, per-number explanations, diagnostics, reasons,
disclaimer, references. It exists because a research log, a pull request and a
terminal are where this output most often lands, and a rendering that dropped the
caveats to fit would defeat the point. The two renderers share `_build_metrics`
and `_diagnostics` so they cannot drift apart.

**The terminal summary.** `Verdict.summary()` is the surface most users see most
often, and its whole design is one decision: fixed-width labels padded to a
common column (`Deflated Sharpe `, `Annualised Sharpe`, `PBO             `) so
the numbers form a straight edge in a monospace terminal, with the reasons
underneath at a two-space indent. No colour, no escape codes, so it survives
being piped to a file or captured by CI.

## Done means

- [x] Matches intent — a pathology report, not a dashboard
- [x] Every content state designed, including the not-computable gate
- [x] Contrast checked with the real hex values, not eyeballed
- [x] Keyboard path — none required; no interactive elements, no suppressed focus
- [ ] Checked at 320px in a real browser — **outstanding**, see the risk above
- [x] Self-contained: no external asset of any kind

## Why there is no App Flow document

There is no interactive flow to document. The CLI is a single non-interactive
invocation: arguments in, one verdict out, process exits. No screens, no
navigation, no sessions, no authorisation states, no transitions between views.
The states a flow document would capture — the input-rejection paths, the three
verdicts, and the exit codes — are enumerated in the TDD's degenerate-path table
and CLI contract, where they belong. A flow document here would restate those
tables under UI headings and add nothing.
