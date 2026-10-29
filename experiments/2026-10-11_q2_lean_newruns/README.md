# 2026-10-11_q2_lean_newruns

2026-10-11 · part of [Earlier work: do monitors catch simulated attacks as often as real ones?](../../docs/experiments.md#earlier-work-do-monitors-catch-simulated-attacks-as-often-as-real-ones)

## Question

What would new simulated and real runs cost to test, within plus or minus 5 points, whether the data-transfer and destructive monitors catch slightly less in simulation (the pooled lean)?

## What we did

\$0 sizing that reproduced the pooled lean test and projected the chance of a clear answer by number of new tasks, then purchase rules; after stage 1 (50 simulated runs) a post hoc re-pricing.

## What we found

README: 230 tasks needed at 1% (\$2,965 with buffer, \$2,366 sharing the credential-monitor comparison's real runs); re-pricing gave at most a 0.70 chance of a clear answer, so the purchase stopped after stage 1 (\$301.66); those runs stay sealed.

## In the post?

Not in the post.

## Status

Current (sizing; the purchase it priced stopped after stage 1)

## Builds on

[2026-10-11_q2_pooled_lean](../2026-10-11_q2_pooled_lean/)

## Main scripts

- [`analyze_lean_sizing.py`](analyze_lean_sizing.py)
- [`analyze_reprice_POSTHOC.py`](analyze_reprice_POSTHOC.py)
- [`analyze_reprice_variant_POSTHOC.py`](analyze_reprice_variant_POSTHOC.py)

Result files the index names:

- [`lean_sizing_2026-10-11_153740.json`](results/lean_sizing_2026-10-11_153740.json)
- [`reprice_POSTHOC_2026-10-11_191730.json`](results/reprice_POSTHOC_2026-10-11_191730.json)
- `reprice_variant_POSTHOC_2026-10-11_230018.json` (not copied: result left out to keep the release under 120 MB)

## Files here

3 code and configuration files and 2 result files, copied byte for byte from `experiments/2026-10-11_q2_lean_newruns/` at commit `4e79ee54fef77cd1192d19c5aa933442f71bb758` of our internal repository. 6 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
