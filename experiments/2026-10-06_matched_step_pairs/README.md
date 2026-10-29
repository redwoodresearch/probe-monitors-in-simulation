# 2026-10-06_matched_step_pairs

2026-10-06 · part of [Can a probe tell simulated runs from real ones?](../../docs/experiments.md#can-a-probe-tell-simulated-runs-from-real-ones)

## Question

Holding the real history fixed, does the simulator's output for the same command look fake? (This separates fake-looking outputs from the agent behaving differently.)

## What we did

For 200 real steps (100 from first real attack runs and 100 from real honest runs, over 8 environments), the September version of the GPT-6-Astra simulator wrote the output of the real command given the same real history; TF-IDF text classifiers and Qwen3-32B probes (the frozen real-versus-simulated probe from the first research question and a new within-pair probe, with label-shuffle nulls) tried to tell real from simulated outputs, and the second research question's impactful-tool-call probes were checked at the next call.

## What we found

README: text classifier AUROC 0.608 [0.576, 0.644], frozen probe 0.551 [0.521, 0.580], fresh probe 0.520 [0.497, 0.544] against a shuffled-label cutoff of 0.526; 57 of 200 simulated outputs byte-identical to the real one.

## In the post?

Not in the post.

## Status

Current

## Builds on

[2026-10-06_q1_text_baselines](../2026-10-06_q1_text_baselines/) (frozen text classifier); [2026-10-06_q1_segment_end](../2026-10-06_q1_segment_end/) (frozen probe); [2026-10-06_q2_impactful_call_transfer](../2026-10-06_q2_impactful_call_transfer/) (impactful-call probes); config cites [2026-10-06_astra_sim_honest](../2026-10-06_astra_sim_honest/)

## Main scripts

- [`run_select_pairs.py`](run_select_pairs.py)
- [`run_simulate_pairs.py`](run_simulate_pairs.py)
- [`run_read_pairs.py`](run_read_pairs.py)
- [`analyze_pairs.py`](analyze_pairs.py)
- [`analyze_probes.py`](analyze_probes.py)
- [`analyze_selection_null.py`](analyze_selection_null.py)
- [`analyze_layer_max_null.py`](analyze_layer_max_null.py)
- [`analyze_text_frozen.py`](analyze_text_frozen.py)
- [`analyze_checks.py`](analyze_checks.py)
- [`plot_layers.py`](plot_layers.py)

Result files the index names:

- `pairs_2026-10-06_175844.json` (not copied: result left out to keep the release under 120 MB)
- `metrics_2026-10-06_194622.json` (not copied: result left out to keep the release under 120 MB)
- [`selection_null_null_2026-10-07_103409.json`](results/selection_null_null_2026-10-07_103409.json)
- [`layer_max_null_2026-10-07_015553.json`](results/layer_max_null_2026-10-07_015553.json)
- `figure_layers_2026-10-06_194622.png` (not copied: result file type not copied (.png))

## Files here

11 code and configuration files and 2 result files, copied byte for byte from `experiments/2026-10-06_matched_step_pairs/` at commit `3820d9e51ea0ce38eb8506838408d9fd851f10cf` of our internal repository. 17 other files in that folder were not copied; [`../manifest.tsv`](../manifest.tsv) gives each one's reason (for example activations, run records, logs, figures, notes, or result files over 1 MB or holding long text).

## About this page

The sections above are this experiment's entry in our experiment index, written from the folder's notes, logs and result files and checked against them by a second model (index confidence: high). Numbers are as the index states them. Commits and branches refer to our internal repository; it, our experiment log and our internal status page, which some entries mention, are not published. The code is as it was in our repository, so its comments use internal working labels; [the glossary](../README.md#glossary) explains the common ones.
