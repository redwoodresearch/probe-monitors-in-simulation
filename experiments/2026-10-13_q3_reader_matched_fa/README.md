# 2026-10-13_q3_reader_matched_fa

2026-10-13 · part of [Earlier work: the simulator's formatting slips and tool-output monitors](../../docs/experiments.md#earlier-work-the-simulators-formatting-slips-and-tool-output-monitors)

## Question

With Olmo-3.1-32B as the reader, does the editor-output formatting repair raise false alarms on simulated honest runs, and at matched false alarms (each world calibrated on its own honest runs) does repaired simulated catch match the second real run?

## What we did

Olmo-3.1-32B read 828 honest transcripts (437 simulated honest, 381 repaired, 10 real), scored with the frozen Olmo-3.1-32B detector fit ([2026-10-12_q2_olmo_reader](../2026-10-12_q2_olmo_reader/)); a false-alarm comparison with an equivalence margin derived from a power calculation, and a matched-false-alarm comparison over 939 cells. A Llama-3.3-70B leg passed its smoke test but was held when its price (\$94.23) exceeded the \$80 stop. Modal, \$12.01 billed.

## What we found

README: the repair leaves false alarms stable (+0.11 points [-0.79, +1.43] at 5%, margin 3.0); matched false alarms, edited minus real second run -1.84 [-9.37, +5.18] at 5% (planted check later found not detected, see [2026-10-13_q3_matchedfa_designs](../2026-10-13_q3_matchedfa_designs/)).

## In the post?

Not in the post.

## Status

Exploratory (Llama leg held)

## Builds on

[2026-10-12_q2_olmo_reader](../2026-10-12_q2_olmo_reader/) (frozen fit); [2026-10-12_q3_edit_honest_fa](../2026-10-12_q3_edit_honest_fa/); the repaired runs of all 939 cells

## Main scripts

- [`run_score_simhonest_olmo.py`](run_score_simhonest_olmo.py)
- [`run_score_simhonest_llama.py`](run_score_simhonest_llama.py)
- [`analyze_reader_edit_honest.py`](analyze_reader_edit_honest.py)
- [`analyze_reader_matched_fa.py`](analyze_reader_matched_fa.py)
- [`power_reader.py`](power_reader.py)
- [`reader_inputs.py`](reader_inputs.py)

Result files the index names:

- `simhonest_scoring_olmo_2026-10-13_021213.json` (not copied: result left out to keep the release under 120 MB)
- `power_olmo_2026-10-13_021316.json` (not copied: result left out to keep the release under 120 MB)
- [`extract_check_olmo_2026-10-13_021051.json`](results/extract_check_olmo_2026-10-13_021051.json)
- [`price_llama_2026-10-13_022003.json`](results/price_llama_2026-10-13_022003.json)

## Files here

13 code and configuration files and 3 result files, copied byte for byte from `experiments/2026-10-13_q3_reader_matched_fa/` at commit `acc8e318fc2c6f6ac1ad0ce0847ca2dfe74bed91` of our internal repository. 19 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
