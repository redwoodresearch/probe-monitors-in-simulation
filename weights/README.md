# weights/: every probe in the analysis

Each file here is byte for byte the file the post's scores were computed with; [`manifest.json`](manifest.json) gives its sha256,
size, arrays and shapes, the layer, which fold copies use it, and the script, commit and fit record that wrote it.
[`../pipeline/score.py`](../pipeline/score.py) scores activation rows with them, and
[`../pipeline/train_probes.py`](../pipeline/train_probes.py) and [`../pipeline/published_probes/`](../pipeline/published_probes/)
rebuild them.

| Folder | Probe | Layout |
|---|---|---|
| `on_policy_probe/`, `on_policy_probe_25pct_tasks/`, `on_policy_probe_10pct_tasks/` | The on-policy probe, and its versions trained on 25% and 10% of the tasks | `{commands,tool_outputs}/fold{0-4}.safetensors`: one file per fold copy, layer 48; arrays `W` (5120 × 1), `b`, `mean`, `std`, all float64 |
| `off_policy_probe/` | The off-policy probe (trained on GPT-5.6-Sol's runs) | `commands/layer63.npz` and `tool_outputs/layer{34,45,46,47,48,49}.npz`: one file per layer, holding all five fold copies as `a__f<k>__W` and `__b` (float64), `__mean` and `__std` (float32) |
| `extra_probe_simulated_attack_vs_real_honest/` | Appendix D's simulated-attack probe | As the off-policy probe, with arrays named `c_asis__f<k>__...` |
| `published/` | The deception probe (FAR AI's weights, with their license and notice), the harmfulness and refusal directions, and the high-stakes probe | See [`published/manifest.json`](published/manifest.json) |
| `on_policy_folds.csv`, `off_policy_folds.csv` | Which fold each main task is in (901 and 536 main tasks) | A unit is scored by the copy whose held-out fold holds its main task; a unit whose main task is in no fold (off-policy and simulated-attack probes only) is scored by all five copies, which `../analysis/main_analysis.py` combines |

For tool outputs, each fold copy of the off-policy and simulated-attack probes averages the logits of two layers: copies 0 to 4
use layers [48, 45], [47, 49], [48, 45], [47, 46] and [47, 34]. The array names inside the `.npz` files are the original ones,
kept so that the files stay byte-identical.

## How we checked them

On Modal, we re-scored 23,500 stored units with `pipeline/score.py` and these files, over all ten probe-and-input combinations
and all five kinds of run (real and simulated honest runs and the three attack runs): 20,000 drawn uniformly at random with a
fixed seed, plus 3,500 chosen to cover units whose main task is in no fold and units from the first 196 tasks. Every score matched
the score the post used to floating-point rounding (largest relative difference 4.6 × 10⁻¹³; for each combination, 95% to 98% of
the scores bit for bit). Every file's sha256 equals the one recorded when the probe was fit.
