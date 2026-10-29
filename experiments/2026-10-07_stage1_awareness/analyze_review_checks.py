"""The $0 review checks on Stage 1 (2026-10-08), recomputed from saved records; no activations, no model calls.

1. Selection-matched nulls (#171): per null draw k, pick the layer the way the observed layer was picked (the null's own
   selection statistic: null_mean, null_em_val or null_held_out; ties to the deeper layer, None as 0, as in
   analyze_fit_stage1.py), then read that layer's secondary null at the same k. Compared with the committed
   max-over-layers cut.
2. Twin folds (#222): GroupKFold(5) over sources and the 100 seeded source-label permutations, as run_fit_stage1.py's
   twin(): how many test folds hold both labels, observed and per null draw.
3. asst_last rows (#169, #221): from the Needham row index, rows on a turn that also has a command row, and rows from
   transcripts over the 32k reading limit. The index has no transcript lengths, so "over the limit" uses items.jsonl's
   n_tokens, counted with the Qwen3-32B tokenizer, not Stage 1's Qwen3.5-27B: that count is approximate.
4. Needham extraction failures (#163): the failure reasons, tallied.

Inputs are the fit record and verdicts (S3, results/stage1_manifest_2026-10-08.json) and three metadata files from the
Stage 1 Volumes: needham/index.jsonl and needham/failures.jsonl (probe-shift_aswin-surya_sub-stage1_acts_kill-2026-11-06,
run 2026-10-07_stage1_qwen35-27b_v1) and inputs/items.jsonl (probe-shift_aswin-surya_sub-q3evalaw_data_kill-2026-11-06).

  uv run --no-project --python 3.12 --with numpy==2.3.3 --with scikit-learn==1.7.2 python analyze_review_checks.py \
      --fit <fit_stage1_...json> --verdicts <fit_verdicts_...json> --index <index.jsonl> --failures <failures.jsonl> \
      --items <items.jsonl>
"""

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
from sklearn.model_selection import GroupKFold

HERE = Path(__file__).parent
AIDER = ("aider_full", "aider_commit", "aider_multi", "aider_manual")
CUT = 32768


def seed_of(*parts) -> int:
    return int(hashlib.sha256(json.dumps(parts).encode()).hexdigest()[:8], 16)


def p95(v):
    return float(np.percentile([x for x in v if x is not None], 95))


def pick(vals):  # argmax, ties to the deeper layer, None as 0
    vals = {l: (v if v is not None else 0) for l, v in vals.items()}
    top = max(vals.values())
    return max(l for l, v in vals.items() if v == top)


def main():
    ap = argparse.ArgumentParser()
    for a in ("--fit", "--verdicts", "--index", "--failures", "--items"):
        ap.add_argument(a, required=True)
    args = ap.parse_args()
    L = {r["layer"]: r for r in json.loads(Path(args.fit).read_text())["layers"]}
    V = json.loads(Path(args.verdicts).read_text())["probes"]
    lay = sorted(L)
    out = {"inputs": {Path(f).name: hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in (args.fit, args.verdicts, args.index, args.failures, args.items)}}

    def matched(sel, sec, layers):
        return [(lambda x: x if x is not None else 0)(sec(pick({l: sel(l, k) for l in layers}), k)) for k in range(100)]

    checks = {}
    for p in ("monitoring", "stakes"):
        obs = L[V[p]["layer"]][p]["implicit"]["auroc"]
        m = matched(lambda l, k: L[l][p]["null_mean"][k], lambda l, k: L[l][p]["implicit"]["null"][k], lay)
        checks[f"{p} implicit"] = (obs, V[p]["implicit"]["null_p95_max_over_layers"], p95(m), V[p]["passes_validity"])
    m = matched(lambda l, k: L[l]["tvc"]["null_mean"][k], lambda l, k: L[l]["tvc"]["within_aider"]["null"][k], lay)
    checks["tvc within_aider"] = (V["tvc"]["within_aider"]["auroc"], V["tvc"]["within_aider"]["null_p95_max_over_layers"], p95(m), V["tvc"]["passes_validity"])
    vl = [l for l in lay if "vuln" in L[l]]
    m = matched(lambda l, k: L[l]["vuln"]["null_em_val"][k], lambda l, k: L[l]["vuln"]["null_safecoder"][k], vl)
    checks["vuln safecoder"] = (V["vuln"]["safecoder"], V["vuln"]["null_p95_max_over_layers"], p95(m), V["vuln"]["passes_validity"])
    for key in ("twin", "twin_mixed"):
        m = matched(lambda l, k: L[l][key]["null_held_out"][k], lambda l, k: L[l][key]["null_within_aider"][k], lay)
        checks[f"{key} within_aider"] = (L[V[key]["layer"]][key]["within_aider"], V[key]["within_aider_null_p95"], p95(m), V[key]["passes_validity"])
    out["1 selection-matched nulls"] = {k: {"observed": round(o, 4), "max_over_layers_p95": round(c, 4), "selection_matched_p95": round(s, 4),
                                            "above_null_before": o > c, "above_null_matched": o > s, "probe_passes_validity": pv}
                                        for k, (o, c, s, pv) in checks.items()}

    meta = {}
    for line in Path(args.items).read_text().splitlines():
        it = json.loads(line)
        meta[it["uid"]] = it
    items = defaultdict(lambda: {"cmd_steps": [], "al": None})
    for line in Path(args.index).read_text().splitlines():
        r = json.loads(line)
        if r["kind"] == "command_end":
            items[r["item"]]["cmd_steps"].append(r["step"])
        else:
            items[r["item"]]["al"] = r["step"]
    uids = sorted(u for u in items if u in meta)
    src = np.array([meta[u]["source"] for u in uids]); lab = np.array([int(meta[u]["label"]) for u in uids])
    has_calls = np.array([bool(items[u]["cmd_steps"]) for u in uids]); has_al = np.array([items[u]["al"] is not None for u in uids])
    aider_all = np.isin(src, AIDER)
    folds_out = {}
    for name, keep in (("twin", has_calls), ("twin_mixed", has_calls | has_al)):
        train = np.flatnonzero(keep & ~aider_all)
        folds = list(GroupKFold(n_splits=5).split(np.zeros(len(train)), lab[train], src[train]))

        def valid(labels):
            return sum(len(set(labels[train[tr]])) == 2 and len(set(labels[train[te]])) == 2 for tr, te in folds)

        sources = np.unique(src[train]); src_label = {s: int(lab[src == s][0]) for s in sources}
        nv = []
        for k in range(100):
            r = np.random.default_rng(seed_of("twin", name, k))
            perm = dict(zip(sources.tolist(), r.permutation([src_label[s] for s in sources]).tolist()))
            nv.append(valid(np.array([perm.get(s, lab[i]) for i, s in enumerate(src)])))
        obs_v = valid(lab)
        mx = [max((L[l][name]["null_held_out"][k] or 0) for l in lay) for k in range(100)]
        same = [mx[k] for k in range(100) if nv[k] == obs_v]
        folds_out[name] = {"train_items": int(len(train)), "sources": int(len(sources)), "observed_valid_folds_of_5": int(obs_v),
                           "null_valid_folds": dict(sorted(Counter(nv).items())),
                           "observed_held_out": L[V[name]["layer"]][name]["held_out_by_source"], "null_p95_all_draws": p95(mx),
                           "null_p95_same_fold_count": p95(same) if same else None, "draws_with_same_fold_count": len(same)}
    out["2 twin folds"] = folds_out

    al = {u: v["al"] for u, v in items.items() if v["al"] is not None}
    on_call = {u for u in al if al[u] in items[u]["cmd_steps"]}
    cut = {u for u in al if meta[u]["n_tokens"] > CUT}
    out["3 asst_last rows"] = {"rows": len(al), "on a turn with a command row (lower bound)": len(on_call),
                               "from transcripts over the 32k cut (approximate: Qwen3-32B token counts)": len(cut),
                               "in both groups (approximate)": len(on_call & cut),
                               "not on the final assistant turn": sum(al[u] < meta[u]["n_assistant_turns"] - 1 for u in al)}
    fails = [json.loads(x) for x in Path(args.failures).read_text().splitlines()]
    out["4 Needham extraction failures"] = dict(Counter(re.sub(r"\d+", "N", f["error"])[:80] for f in fails))
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    path = HERE / "results" / f"review_checks_{stamp}.json"
    with path.open("x") as fh:
        fh.write(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    print("->", path)


if __name__ == "__main__":
    main()
