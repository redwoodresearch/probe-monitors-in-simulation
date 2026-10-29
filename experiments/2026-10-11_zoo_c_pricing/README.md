# 2026-10-11_zoo_c_pricing

2026-10-11 · part of [Earlier work: do monitors catch simulated attacks as often as real ones?](../../docs/experiments.md#earlier-work-do-monitors-catch-simulated-attacks-as-often-as-real-ones)

## Question

The credential-monitor study: how many more tasks with two same-batch real GPT-5.6-Sol honest runs would the credential gate need to reach an 80% chance of a clear answer, what would they cost, and do unused real runs already exist?

## What we did

\$0 pricing from the gate's eight counts and Sol token usage in real log headers.

## What we found

README: the committed rule reproduces (0.9987 x 0.7005 = 0.6996); the labelled cohort already holds 251 disjoint same-batch pairs over 201 tasks, projecting 0.92 at a design effect of 1.25, but only with a plan amendment.

## In the post?

Not in the post.

## Status

Current

## Builds on

[2026-10-11_credential_count](../2026-10-11_credential_count/) (gate_counts_v3.json)

## Main scripts

- [`price_zoo_c.py`](price_zoo_c.py)
- [`read_header_usage.py`](read_header_usage.py)

Result files the index names:

- `zoo_c_price_2026-10-11_134610.json` (not copied: result left out to keep the release under 120 MB)

## Files here

2 code and configuration files and 0 result files, copied byte for byte from `experiments/2026-10-11_zoo_c_pricing/` at commit `4a0b823cf688c59de74e5f3a39a8d0650b2bda18` of our internal repository. 9 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
