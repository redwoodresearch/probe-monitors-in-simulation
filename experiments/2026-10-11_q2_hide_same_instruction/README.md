# 2026-10-11_q2_hide_same_instruction

2026-10-11 · part of [Side question: deception and sabotage probes](../../docs/experiments.md#side-question-deception-and-sabotage-probes)

## Question

Row 11: under the same 'hide' instruction, does the deception probe read higher on reports that actually hid the changed file than on reports that named it anyway (reacting to the act of hiding rather than the instruction)?

## What we did

Pilot of 40 runs (320 sampled reports) on Modal, with an LLM judge (GPT-5.6-Sol) classifying hid versus named, then sizing and one allowed rescue design with a matched per-run length.

## What we found

README: closed, too few to measure; only 37 usable samples and 6 of 40 runs with both groups; power at most 0.13 even at 24 generated reports per run; the rescue reached 0.717 against a 0.80 gate; no hid-minus-named mean computed.

## In the post?

Not in the post.

## Status

Void (closed: too few)

## Builds on

[2026-10-11_q2_e1_confirm](../2026-10-11_q2_e1_confirm/)

## Main scripts

- [`make_pilot_index.py`](make_pilot_index.py)
- [`run_hide_pilot.py`](run_hide_pilot.py)
- [`build_readout.py`](build_readout.py)
- [`run_judges.py`](run_judges.py)
- [`analyze_pilot.py`](analyze_pilot.py)
- [`analyze_sizing_b.py`](analyze_sizing_b.py)
- [`diagnose_sizing_POSTHOC.py`](diagnose_sizing_POSTHOC.py)
- [`diagnose_looser_readout_POSTHOC.py`](diagnose_looser_readout_POSTHOC.py)

Result files the index names:

- [`pilot_sizing_2026-10-11_224031.json`](results/pilot_sizing_2026-10-11_224031.json)
- [`sizing_b_2026-10-11_232203.json`](results/sizing_b_2026-10-11_232203.json)
- `looser_readout_POSTHOC_2026-10-11_230016.json` (not copied: result left out to keep the release under 120 MB)
- [`judgments_2026-10-11_224016.json`](results/judgments_2026-10-11_224016.json)

## Files here

9 code and configuration files and 4 result files, copied byte for byte from `experiments/2026-10-11_q2_hide_same_instruction/` at commit `8156c2bf04fa4729dedb56a7c1ef1007da49477d` of our internal repository. 12 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
