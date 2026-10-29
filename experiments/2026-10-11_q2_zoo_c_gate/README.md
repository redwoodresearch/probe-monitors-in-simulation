# 2026-10-11_q2_zoo_c_gate

2026-10-11 · part of [Earlier work: do monitors catch simulated attacks as often as real ones?](../../docs/experiments.md#earlier-work-do-monitors-catch-simulated-attacks-as-often-as-real-ones)

## Question

The credential-monitor study: does the committed positive-count gate give at least an 80% chance of a clear real-versus-simulated answer for a credential monitor on GPT-5.6-Sol's runs?

## What we did

\$0: applied the committed gate (normal approximation, p = 0.5, design effect at least 1.25) to the gate counts, then an amendment using all disjoint same-batch real pairs.

## What we found

README: too few to tell, predicted chance of agreement 0.6996 vs 0.8 (limited by the real-versus-real control); amendment 1 with 251 pairs still too few, 0.443 at the committed design effect 2.69.

## In the post?

Not in the post.

## Status

Current

## Builds on

[2026-10-11_credential_count](../2026-10-11_credential_count/); [2026-10-11_q2_zoo_plans](../2026-10-11_q2_zoo_plans/)

## Main scripts

- [`analyze_zoo_c_gate.py`](analyze_zoo_c_gate.py)
- [`analyze_zoo_c_gate_pairs.py`](analyze_zoo_c_gate_pairs.py)

Result files the index names:

- `zoo_c_gate_2026-10-11_121752.json` (not copied: result left out to keep the release under 120 MB)
- `zoo_c_gate_pairs_2026-10-11_135108.json` (not copied: result left out to keep the release under 120 MB)

## Files here

2 code and configuration files and 0 result files, copied byte for byte from `experiments/2026-10-11_q2_zoo_c_gate/` at commit `8bb959fcf97217d05405318500dc32e4730b4a53` of our internal repository. 4 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
