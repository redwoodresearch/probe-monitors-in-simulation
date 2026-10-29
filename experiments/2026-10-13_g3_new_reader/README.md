# 2026-10-13_g3_new_reader

2026-10-13 · part of [Earlier work: the simulator's formatting slips and tool-output monitors](../../docs/experiments.md#earlier-work-the-simulators-formatting-slips-and-tool-output-monitors)

## Question

Does the planting forecast (planting the simulator's known formatting slips into real runs predicts the simulated-vs-real gap of a tool-output detector) hold for a reader from a new model family?

## What we did

Screened nine open models and picked Seed-OSS-36B-Instruct; extracted its activations on GPT-5.6-Sol's real honest, first and second real attack, planted and simulated runs (939 task cells), fit a frozen tool-output detector on real arms only, computed the planted shift x and froze a forecast before scoring the simulated runs (y = real run 2 minus simulated catch at 1%). Modal GPU, about \$100 to \$175 in extraction.

## What we found

README (from g3_y_2026-10-13_124203.json): y = +13.63 points [10.76, 16.87] against the frozen forecast 7.05 (cutoff 3.201, error 6.58), so the verdict is 'does not hold'; x = +5.62 [4.33, 6.94]; selection-aware null AUROC 0.955 vs null max 0.563. Independently recomputed by a separate reviewer and Codex.

## In the post?

Not in the post.

## Status

Exploratory

## Builds on

[2026-10-12_plantslip_heldout](../2026-10-12_plantslip_heldout/) (planting forecast and gate rows); the attack detector's cohort of 939 cells

## Main scripts

- [`screen_readers.py`](screen_readers.py)
- [`g3_freeze.py`](g3_freeze.py)
- [`g3_x.py`](g3_x.py)
- [`g3_y.py`](g3_y.py)
- [`g3_score_frozen.py`](g3_score_frozen.py)
- [`g3_score_sim.py`](g3_score_sim.py)
- [`g3_null_gate.py`](g3_null_gate.py)
- [`run_q3_attack_g3.patch`](run_q3_attack_g3.patch) (extractor run_extract_g3.py on S3)

Result files the index names:

- `forecast.json` (not copied: JSON outside results/ and config/)
- `g3_y_2026-10-13_124203.json` (not in this folder at the source commit)
- `g3_x_2026-10-13_112514.json` (not in this folder at the source commit)
- `s3_manifest_v8.json` (not copied: strings over 300 characters)
- `verify_y_09_2026-10-13_134137.json` (not copied: result left out to keep the release under 120 MB)
- `step7_projection_2026-10-13_113731.json` (not copied: result left out to keep the release under 120 MB)

## Files here

22 code and configuration files and 1 result files, copied byte for byte from `experiments/2026-10-13_g3_new_reader/` at commit `77a1c55713e998bb987f270fd34657f41cdda97c` of our internal repository. 27 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
