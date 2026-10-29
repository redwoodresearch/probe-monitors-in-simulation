# 2026-10-13_q3_loeo

2026-10-13 · part of [Side question: probes trained on simulated runs](../../docs/experiments.md#side-question-probes-trained-on-simulated-runs)

## Question

Does the frozen simulation-trained recipe (simulated/simulated training, full repair, averaged layer-48 readout) transfer to a LinuxArena environment it never saw?

## What we did

Leave one environment out: trained on 9 of the 10 environments and tested on the 10th, for Qwen3-32B and Seed-OSS-36B reading GPT-5.6-Sol runs, with the real-trained comparator on the same splits; pooled reading on real catch, simulated-minus-real gap and the second-real-run catch difference against the count-matched real-trained detector. Modal.

## What we found

loeo9s_2026-10-14_001947.json: Qwen3-32B reads 'transfers to an unseen environment' at 1% and 5% under every threshold variant; Seed-OSS-36B 'transfers' at 5% but 'fails (ii)' (gap outside +/-5) at 1%.

## In the post?

Not in the post.

## Status

Exploratory

## Builds on

[2026-10-13_q3_sce_reply_probe](../2026-10-13_q3_sce_reply_probe/) (recipe, Amendments 21e to 21g); [2026-10-13_why_steps](../2026-10-13_why_steps/) (packs on another working session's Modal volume)

## Main scripts

- [`run_loeo.py`](run_loeo.py)
- [`run_loeo9.py`](run_loeo9.py)
- [`run_loeo9s.py`](run_loeo9s.py)
- [`check_loeo9s_gate.py`](check_loeo9s_gate.py)

Result files the index names:

- `loeo_2026-10-13_232840.json` (not copied: result left out to keep the release under 120 MB)
- `loeo9_2026-10-14_000238.json` (not copied: result left out to keep the release under 120 MB)
- `loeo9s_2026-10-14_001947.json` (not copied: result left out to keep the release under 120 MB)
- `check_loeo9s_gate_2026-10-14_003141.json` (not copied: result left out to keep the release under 120 MB)

## Files here

4 code and configuration files and 0 result files, copied byte for byte from `experiments/2026-10-13_q3_loeo/` at commit `9db895e914e2ae222252f26354f6ef757070ea27` of our internal repository. 10 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: medium). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
