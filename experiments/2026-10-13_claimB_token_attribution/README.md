# 2026-10-13_claimB_token_attribution

2026-10-13 · part of [Can a probe tell simulated runs from real ones?](../../docs/experiments.md#can-a-probe-tell-simulated-runs-from-real-ones)

## Question

Which tokens in honest replies push the simulated-vs-real direction hardest toward 'simulated', as a way to find giveaways nobody has named?

## What we did

Scored every token's push along the Qwen3-32B fake-vs-real direction and a planted-slip direction on 91 matched sites, 30 planted gutter pairs and 10 whole runs, behind a pre-registered gate that the planted tokens must rank in the top 1% better than random directions. One Modal H100 job, \$3.90.

## What we found

RESULT.md (script-printed): the gate failed (planted tokens' median best rank fraction 0.123 along the planted direction vs 0.029 to 0.071 along random directions; bar 0.01), so no token list was computed; post hoc, the planted direction does transfer at its training position (planted above twin in 30 of 30 pairs). Reading: per-token push does not localize giveaways; thread closed.

## In the post?

Not in the post.

## Status

Exploratory (closed: the method failed its pre-registered gate)

## Builds on

[2026-10-12_claimB_meaning_judge](../2026-10-12_claimB_meaning_judge/) (sites); halves design

## Main scripts

- [`run_token_attr_modal.py`](run_token_attr_modal.py)
- [`analyze_token_attr.py`](analyze_token_attr.py)
- [`analyze_POSTHOC_transfer.py`](analyze_POSTHOC_transfer.py)
- [`make_result_md.py`](make_result_md.py)

Result files the index names:

- [`token_attr_2026-10-13_142630.json`](results/token_attr_2026-10-13_142630.json)
- `POSTHOC_transfer_2026-10-13_142927.json` (not copied: result left out to keep the release under 120 MB)
- `throughput_2026-10-13_143024.json` (not copied: result left out to keep the release under 120 MB)
- [`marks_counts_2026-10-13_132713.json`](results/marks_counts_2026-10-13_132713.json)

## Files here

8 code and configuration files and 2 result files, copied byte for byte from `experiments/2026-10-13_claimB_token_attribution/` at commit `7aa494ed887389861df1f2153b4472b2a81cc098` of our internal repository. 11 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
