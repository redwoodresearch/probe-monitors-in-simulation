# 2026-10-11_q2_e1_sentences

2026-10-11 · part of [Side question: deception and sabotage probes](../../docs/experiments.md#side-question-deception-and-sabotage-probes)

## Question

Hide-or-mention test, new design: does the frozen deception probe read higher on the report sentences about the changed file when the agent was told to hide the edit than when told to mention it?

## What we did

\$0 adaptive reanalysis of stage 1's generations and per-token scores: selected the change-statement sentences with a script that never opened scores, then compared hide and mention reports with swap tests and length adjustment; also planned the real-versus-simulated follow-up.

## What we found

README: pass (adaptive reanalysis); D +0.99 honest SD [+0.52, +1.54] on 14 of 30 runs (swap p 0.0003); without the changed path's tokens +0.92 [+0.34, +1.61].

## In the post?

Not in the post.

## Status

Superseded by [2026-10-11_q2_e1_confirm](../2026-10-11_q2_e1_confirm/) (the fresh-run confirmation is the hide-or-mention test's result of record)

## Builds on

[2026-10-11_q2_e1_honest](../2026-10-11_q2_e1_honest/)

## Main scripts

- [`select_sentences.py`](select_sentences.py)
- [`analyze_sentences.py`](analyze_sentences.py)
- [`sensitivity_POSTHOC_floor.py`](sensitivity_POSTHOC_floor.py)
- [`sensitivity_POSTHOC_matcher.py`](sensitivity_POSTHOC_matcher.py)

Result files the index names:

- [`e1_sentences_2026-10-11_155202.json`](results/e1_sentences_2026-10-11_155202.json)
- `sentence_masks_2026-10-11_155002.json` (not copied: result left out to keep the release under 120 MB)
- `matcher_sensitivity_POSTHOC_2026-10-11_175120.json` (not copied: result left out to keep the release under 120 MB)
- [`floor_sensitivity_POSTHOC_2026-10-11_181859.json`](results/floor_sensitivity_POSTHOC_2026-10-11_181859.json)

## Files here

4 code and configuration files and 2 result files, copied byte for byte from `experiments/2026-10-11_q2_e1_sentences/` at commit `70babbf806a06d331a1e19cd93b8ebb4ad4ed9ff` of our internal repository. 14 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
