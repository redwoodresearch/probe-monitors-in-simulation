"""POST HOC (RULE.md Amendment 5), Modal CPU, counts and cosines only:
1. The full prefix gate (Codex on the scorer): for EVERY run x version whose reply map lists a changed output, every kept
   pre_call and call_end row at or before the turn of the first changed output (no content filter) equals the
   original's, at L48 and L63, cosine >= 0.999; coverage is reported (pairs checked, positions per pair).
2. 09's think-block question: per arm, how many of the original staged runs carry a harness user message after the first
   assistant turn, and how many staged assistant messages contain a think block (the reader's messages; counts only).

  PYTHONPATH=src uv run --no-project --python 3.12 --with modal==1.6.1 --with numpy modal run \\
      experiments/2026-10-14_onpolicy_patching/check_POSTHOC_prefix_and_think.py --draw <draw json> --reply-map <json>
"""

import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal
import numpy as np

HERE = Path(__file__).parent
REPO = HERE.parent.parent
APP = "probe-shift_aswin-surya_sub-patching_check_kill-2026-11-14"
ACTS = "probe-shift_aswin-surya_sub-la-activations"
DUMP, INPUTS = "2026-10-14_q2_onpolicy_patching_qwen3-32b_l7_v1", "2026-10-14_q2_onpolicy_patching_inputs_v1"
VERSIONS = ("rewritten", "bash", "edits", "createview", "placeholder")
LAYERS = (48, 63)

app = modal.App(APP)
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy==2.3.3")


@app.function(image=image, cpu=(8, 8), memory=(16384, 16384), timeout=1800, retries=0,
              volumes={"/acts": modal.Volume.from_name(ACTS, version=2).read_only()})
def check(keys: dict, changed: dict) -> dict:
    root = Path("/acts")
    metas = {}
    for v in ("original",) + VERSIONS:
        for p in (root / DUMP / "meta" / f"patch-{v}").glob("*.json"):
            m = json.loads(p.read_text())
            metas[(v, m["key"])] = m

    def rows(m, idx, layer):
        return np.load(root / DUMP / "by_layer" / f"L{layer}" / f"{m['chunk']}.npy", mmap_mode="r")[idx].astype(np.float32)

    gate = {"pairs_with_changes": 0, "pairs_checked": 0, "positions": [], "min_cosine": {}, "non_finite": 0}
    for key in keys:
        o = metas[("original", key)]
        for v in VERSIONS:
            ch = changed.get(key, {}).get(v, [])
            if not ch:
                continue
            gate["pairs_with_changes"] += 1
            first = min(st for st, _ in ch)
            m = metas[(v, key)]
            sel = lambda mm: {(p["kind"], p["step"], p["action"]): mm["row_start"] + p["row"] for p in mm["positions"]
                              if p["kind"] in ("pre_call", "call_end") and p["kept"] and p["step"] <= first}
            so, sv = sel(o), sel(m)
            common = sorted(set(so) & set(sv))
            if set(so) != set(sv) or not common:
                raise ValueError(f"prefix positions differ or are empty ({len(so)} vs {len(sv)})")
            gate["pairs_checked"] += 1
            gate["positions"].append(len(common))
            for layer in LAYERS:
                a, b = rows(o, [so[k] for k in common], layer), rows(m, [sv[k] for k in common], layer)
                c = (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))
                gate["non_finite"] += int((~np.isfinite(c)).sum())
                gate["min_cosine"][f"L{layer}"] = min(gate["min_cosine"].get(f"L{layer}", 1.0), float(np.nanmin(c)))
    gate["positions"] = {"min": min(gate["positions"]), "max": max(gate["positions"]), "total": sum(gate["positions"])}
    gate["pass"] = gate["non_finite"] == 0 and all(x >= 0.999 for x in gate["min_cosine"].values())

    think = {}
    for line in (root / INPUTS / "inputs" / "shard_0.jsonl").read_text().splitlines():
        r = json.loads(line)
        if r["arm"] != "patch-original" or r["key"] not in keys:
            continue
        arm = keys[r["key"]]
        msgs = r["messages"]
        first_asst = next((i for i, m in enumerate(msgs) if m["role"] == "assistant"), len(msgs))
        late_user = sum(m["role"] == "user" for m in msgs[first_asst + 1:])
        text = lambda c: c if isinstance(c, str) else "".join(x.get("text") or "" for x in c or [] if isinstance(x, dict))
        n_think = sum(m["role"] == "assistant" and ("<think>" in text(m.get("content")) or bool(m.get("reasoning_content"))) for m in msgs)
        t = think.setdefault(arm, {"runs": 0, "runs_with_late_user_message": 0, "runs_with_think_blocks": 0, "assistant_messages_with_think": 0})
        t["runs"] += 1
        t["runs_with_late_user_message"] += late_user > 0
        t["runs_with_think_blocks"] += n_think > 0
        t["assistant_messages_with_think"] += n_think
    return {"prefix_gate": gate, "think_counts": think}


@app.local_entrypoint()
def main(draw: str, reply_map: str):
    git = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True, cwd=REPO, check=True).stdout.strip()
    if git("status", "--porcelain", "--untracked-files=no"):
        raise SystemExit("refusing to run from a dirty tree")
    dr, rm_raw = json.loads(Path(draw).read_bytes()), Path(reply_map).read_bytes()
    rm = json.loads(rm_raw)["keys"]
    keys = {p["rel"].replace("/", "__"): p["arm"] for p in dr["picked"]}
    changed = {k: {v: x["changed"] for v, x in e["versions"].items()} for k, e in rm.items()}
    res = check.remote(keys, changed)
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    rec = {"git": git("rev-parse", "HEAD"), "reply_map_sha256": hashlib.sha256(rm_raw).hexdigest(), "dump": DUMP, "inputs": INPUTS} | res
    p = HERE / "results" / f"check_POSTHOC_prefix_and_think_{stamp}.json"
    with p.open("x") as fh:
        fh.write(json.dumps(rec, indent=1))
    print(json.dumps(res), flush=True)
