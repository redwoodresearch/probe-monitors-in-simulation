# pipeline/: from the agent's runs to the probes' scores

Readable code for every step between the agent's runs and the scores in [`../data/`](../data/), each rewritten from the code that
produced the post's results and checked, on Modal, against what that code stored. Comments marked "original:" name the file and
commit of the code that ran; [`../experiments/`](../experiments/) holds those files, under `as_run/<commit>/` where the version that
ran differs from the folder's latest copy. The reader model is Qwen3.5-27B, the agent's own model, reading each run; the runs are
Inspect `.eval` logs (Inspect is the evaluation framework that ran and recorded them). Every check below ran on files
byte-identical to the committed ones.

```
agent run (.eval log) ──units.py──> the text Qwen3.5-27B reads, and where each command and tool output sits in it
                      ──extract_activations.py──> one activation row per unit, per layer
activation rows       ──train_probes.py──> the on-policy, off-policy and simulated-attack probes (../weights/)
public datasets       ──published_probes/──> the deception probe, harmfulness and refusal directions, high-stakes probe
activation rows       ──score.py + ../weights/──> each unit's score (../data/)
```

| File | What it does | Needs | How we checked it |
|---|---|---|---|
| [`units.py`](units.py) | Writes a run out the way the reader model reads it, and finds each scored unit's tokens: the last content token of each command (two variants, one for the probes we trained and one for the published probes) and the content tokens of each tool output | An Inspect `.eval` log; the Qwen3.5-27B tokenizer | Part of the extraction check below: on 5 honest runs, every token id and every unit position matched the stored ones |
| [`extract_activations.py`](extract_activations.py) | Runs Qwen3.5-27B once over a whole run and keeps one row of 5,120 numbers per unit at each requested layer | The run's `.eval` log (for an attack run, also the honest run of the same task, whose system prompt it reads with); a GPU with 80 GB (the longest runs need more; we used H200s) | On 5 randomly drawn real honest runs of up to 60,000 tokens, the 143 commands' rows were bit-identical to the stored rows at all 13 layers the probes read, and the 124 tool-output rows agreed within 0.1% (one bf16 rounding step). Attack runs and simulated runs were not checked |
| [`train_probes.py`](train_probes.py), [`training_data/`](training_data/) | Trains the on-policy probe (and its 25% and 10% variants), the off-policy probe (including its layer selection) and the simulated-attack probe; `training_data/` lists every training run with its label and fold | The training activations: about 4.6 GB for the on-policy probe; for the off-policy probe, Qwen3.5-27B's activation dump over GPT-5.6-Sol's runs, about 3.9 GB per layer (about 27 GB at its seven layers, about 250 GB for all 64 layers to rerun its layer selection); for the simulated-attack probe, a second dump of about 1.2 GB per layer; and a GPU | 18 refits were bit-identical to the stored weights: the on-policy probe's five folds on both kinds of input, fold 0 of its 10% version, and two of the off-policy probe's five folds on each kind of input. For one fold, the off-policy probe's validation AUROCs at 4 candidate layers equalled the recorded ones, and its layer rule applied to the recorded AUROCs gives the stored layers. Re-saved weight files were byte-identical. Not refit: the 25% version, the simulated-attack probe and the remaining folds |
| [`published_probes/`](published_probes/) | The four probes from published work: converts FAR AI's deception probe without unpickling, computes the harmfulness and refusal directions from their published prompts, trains the high-stakes probe on its published data | Downloads the public weights and datasets; a GPU to read prompts | The deception probe's conversion and both directions were bit-identical to what the post used; the high-stakes probe's refit agreed with the shipped one to cosine 0.99999999999987 (the shipped file is the one the post used). A GPU re-read of the 448 direction prompts and 64 of the 12,984 high-stakes conversations reproduced the stored activations bit for bit. Per-unit scores on 8 runs matched bit for bit (a few deception-probe units within 1.1 × 10⁻¹⁶). One bar we set in advance failed: recomputing the directions' per-layer selection metrics in float32 differed from the record by up to 1.9 × 10⁻⁶, against a bar of 10⁻⁶ |
| [`score.py`](score.py) | Scores activation rows with the trained probes in [`../weights/`](../weights/), picking each unit's fold copy | Activation rows | Re-scoring 23,500 stored units (20,000 drawn at random, 3,500 chosen to cover units in no fold and the first 196 tasks), over all ten probe-and-input combinations, matched the scores the post used within 4.6 × 10⁻¹³ (relative), 95% to 98% of them bit for bit |

Each script lists its dependencies at the top (torch, transformers, safetensors and flash-linear-attention are pinned), so `uv run pipeline/<script>.py` (or `pipeline/published_probes/<script>.py`) installs them.

## What the reader reads

The probes do not read the raw run. Each run is written out as a cleaned copy: an attack run gets the system prompt of the honest
run of the same task, so the side-task instructions are never in the probes' input; the agent's reasoning and the tool
definitions are removed; times and dates are masked; and every assistant turn gets the same empty thinking block.
[`units.py`](units.py) does this, and [`../docs/methods.md`](../docs/methods.md) (section 2) describes it in full.

## What is not here

The `.eval` logs of the agent's runs and the stored activations are not in this repository, so the first steps cannot be rerun from
it alone. [`../data/`](../data/) holds the scores they produced, and everything from there on reruns from this repository.
