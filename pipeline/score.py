"""Score activations with the trained probes in weights/, exactly as the post's scoring code did.

The five trained probes (the on-policy probe and its versions trained on 25% and 10% of the tasks, the off-policy probe, and
Appendix D's simulated-attack probe) are logistic regressions on the activations of Qwen/Qwen3.5-27B, whose hidden size is 5120.
"Layer L" is the output of the model's decoder layer L (model.model.layers[L], before the final norm). A probe scores units:

  - a command: the activation row at the last content token of the tool call;
  - a tool output: the mean of the activation rows over the output's content tokens (one row).

We stored every row as float16 (a tool output's row is its mean, rounded to float16); pass the rows as stored.

Each probe comes in five copies, one per fold of main tasks: copy k was trained without the main tasks of fold k, and a unit is
scored by the copy whose held-out fold holds the unit's main task. weights/on_policy_folds.csv gives each main task's fold for the
three on-policy probes; weights/off_policy_folds.csv does the same for the off-policy and simulated-attack probes. One copy's score
at one layer (a logit) is

    ((x - mean) / std) @ W + b

with that copy's stored mean, std, W and b. The standardization runs in the dtype of the stored mean and std: float32 for the
off-policy and simulated-attack probes, float64 for the on-policy probes. A copy's score is the average of these logits over its
layers: for tool outputs, the off-policy and simulated-attack probes use two layers, a pair per copy (OFF_POLICY_TOOL_OUTPUT_LAYERS);
every other case uses one layer (63 for those two probes' commands, 48 for the on-policy probes). A unit whose main task is in no
fold gets all five copies' scores instead; analysis/main_analysis.py (step 1) standardizes and averages them. In the post this
happens only for the off-policy and simulated-attack probes, whose folds cover 536 main tasks.

Example, with acts[L] an (n, 5120) array of rows at layer L and main_tasks the n units' main tasks:

    probe = load_probe("off_policy_probe", "tool_outputs")
    acts = {L: rows_at_layer(L) for L in probe["layers"]}
    score, fold, fold_scores = score_units(probe, acts, main_tasks)
"""

import csv
import json
from pathlib import Path

import numpy as np

WEIGHTS = Path(__file__).resolve().parent.parent / "weights"
PROBES = ["on_policy_probe", "on_policy_probe_25pct_tasks", "on_policy_probe_10pct_tasks", "off_policy_probe",
          "extra_probe_simulated_attack_vs_real_honest"]
KINDS = ["commands", "tool_outputs"]
ON_POLICY = {"on_policy_probe", "on_policy_probe_25pct_tasks", "on_policy_probe_10pct_tasks"}
# off-policy and simulated-attack probes, tool outputs: the two layers of fold copies 0 to 4, chosen on each copy's training tasks
OFF_POLICY_TOOL_OUTPUT_LAYERS = [[48, 45], [47, 49], [48, 45], [47, 46], [47, 34]]
# their weight files hold all five copies for one layer, as arrays <prefix>__f<k>__W, __b, __mean and __std
ARRAY_PREFIX = {"off_policy_probe": "a", "extra_probe_simulated_attack_vs_real_honest": "c_asis"}


def read_safetensors(path):
    """The arrays of a .safetensors file (an 8-byte header length, a JSON header, then the raw little-endian arrays)."""
    raw = Path(path).read_bytes()
    n = int.from_bytes(raw[:8], "little")
    header = json.loads(raw[8:8 + n])
    dtypes = {"F64": "<f8", "F32": "<f4", "F16": "<f2"}
    out = {}
    for name, h in header.items():
        if name == "__metadata__":
            continue
        start, end = h["data_offsets"]
        out[name] = np.frombuffer(raw[8 + n + start:8 + n + end], dtype=dtypes[h["dtype"]]).reshape(h["shape"])
    return out


def read_folds(path):
    """{main task: fold} from a fold file."""
    with open(path, newline="") as f:
        return {r["main_task"]: int(r["fold"]) for r in csv.DictReader(f)}


def load_probe(probe, kind):
    """One probe for one kind of input ("commands" or "tool_outputs"): its five copies (per copy, its layers and per layer the
    arrays W, b, mean and std), the layers they need, and the fold of each main task."""
    assert probe in PROBES and kind in KINDS, (probe, kind)
    copies = []
    if probe in ON_POLICY:
        for k in range(5):
            w = read_safetensors(WEIGHTS / probe / kind / f"fold{k}.safetensors")
            copies.append({"layers": [48], "arrays": {48: w}})
        folds = read_folds(WEIGHTS / "on_policy_folds.csv")
    else:
        layers = OFF_POLICY_TOOL_OUTPUT_LAYERS if kind == "tool_outputs" else [[63]] * 5
        files = {L: np.load(WEIGHTS / probe / kind / f"layer{L}.npz") for L in sorted({L for ls in layers for L in ls})}
        p = ARRAY_PREFIX[probe]
        for k in range(5):
            copies.append({"layers": layers[k],
                           "arrays": {L: {a: files[L][f"{p}__f{k}__{a}"] for a in ("W", "b", "mean", "std")} for L in layers[k]}})
        folds = read_folds(WEIGHTS / "off_policy_folds.csv")
    return {"probe": probe, "kind": kind, "copies": copies, "folds": folds,
            "layers": sorted({L for c in copies for L in c["layers"]})}


def logit(x, w):
    """One copy's logit at one layer for rows x (n x 5120): standardize in the dtype of the stored mean and std, dot with W, add b."""
    # off-policy and simulated-attack probes: run_sola_uncut_score.py @ 2f1b932af (experiments/2026-10-14_q35_sol_avg_POSTHOC), l.306 and l.311
    # the same for the 196 pilot tasks: run_sold_cutfull_score.py @ 687696e61 (same folder), l.591 and l.596
    # on-policy probes: run_own_all.py @ 63096838c (experiments/2026-10-14_p2_family_expansion), l.690 and l.803 (rows: run_avg_transfer.py @ 4d1eec572, l.373, l.1036)
    x = np.asarray(x).astype(w["mean"].dtype)  # float16 rows become float32 or float64 exactly
    return ((x - w["mean"]) / w["std"]) @ w["W"].reshape(-1) + float(w["b"].reshape(-1)[0])


def copy_score(copy, acts):
    """One copy's score: the mean of its logits over its layers (a single layer gives that layer's logit unchanged)."""
    # run_sola_uncut_score.py @ 2f1b932af, l.318 (tool outputs: the fold's two layers) and l.471 (commands: layer 63 alone)
    return np.mean([logit(acts[L], copy["arrays"][L]) for L in copy["layers"]], axis=0)


def score_units(probe, acts, main_tasks):
    """Score n units of one probe and kind of input. acts: {layer: (n, 5120) rows} for every layer in probe["layers"];
    main_tasks: the n units' main tasks.

    Returns (score, fold, fold_scores): score[i] is the score from the copy that held out unit i's main task (NaN if that main task
    is in no fold); fold[i] is that fold (0 to 4, or -1 for none); fold_scores is n x 5, all five copies' scores for the units with
    fold -1 (NaN elsewhere)."""
    # run_sola_uncut_score.py @ 2f1b932af, l.315-318 and l.326-328; run_own_all.py @ 63096838c, l.810-820
    fold = np.array([probe["folds"].get(m, -1) for m in main_tasks], dtype=np.int64)
    n = len(fold)
    assert all(np.shape(acts[L]) == (n, 5120) for L in probe["layers"]), "acts[L] must be (number of units, 5120) for every layer"
    score, fold_scores = np.full(n, np.nan), np.full((n, 5), np.nan)
    for k in range(5):
        sel = fold == k
        if sel.any():
            score[sel] = copy_score(probe["copies"][k], {L: np.asarray(acts[L])[sel] for L in probe["layers"]})
    sel = fold == -1
    if sel.any():
        for k in range(5):
            fold_scores[sel, k] = copy_score(probe["copies"][k], {L: np.asarray(acts[L])[sel] for L in probe["layers"]})
    return score, fold, fold_scores
