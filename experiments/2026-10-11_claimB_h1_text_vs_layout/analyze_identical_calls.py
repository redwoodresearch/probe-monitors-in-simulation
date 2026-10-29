"""H1 planning (Aswin 10:22:01 PT via 0d): count the identical-call pairs available for whole-output swaps, test vs
dev environments, before H1's plan is written. Local and free. Keys and numbers only: calls are matched on a hash of
function + arguments (k-th occurrence with k-th, per cell, as `2026-10-07_sim_failure_hiding`), and the script saves
and prints counts only. No output text is read beyond checking that each call has a result; no command text is printed
or saved; no side-task field is touched.

Counted per pair type (sim–e1, e2–e1), test vs dev, tool family (text_editor by command; bash-like tools), for all
identical calls and the pre-edit subset (neither run had edited a file before the call: the failure-hiding rule):
pairs, cells and main tasks; and how many have both outputs inside claim B's 32k-token reading (step A's level-1
pack, `kept` output_end positions, matched by the call's order in its run). A pair's cut status is "unknown" (counted
separately, never as outside) unless both runs are in the pack under the same sample key, their tool results come in
the same order as their calls, and their call count equals their output_end count.

  PYTHONPATH=src uv run --no-project --python 3.12 --with 'modal>=1.4.3' --with numpy \\
      python experiments/2026-10-11_claimB_h1_text_vs_layout/analyze_identical_calls.py
"""

import hashlib
import json
import subprocess
import sys
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "experiments/2026-10-07_sim_failure_hiding"))
from analyze_failure_hiding import ARMS, is_edit  # noqa: E402
from probe_shift.transcripts import RUNNER_SUBMIT  # noqa: E402

EXCLUSIONS = REPO / "splits/exclusions_v3.json"
DATA = Path.home() / "probe-shift-viewer/.run_cache/viewer/eval"
FOLDS = REPO / "splits/folds_v2.json"
VOL, PACK = "probe-shift_aswin-surya_sub-la-activations", "claimB/level1/packed_39f842231a46"
PACK_ARM = {"e1": "real-attack-e1", "e2": "real-attack-e2", "sim": "sim-attack-fixed"}
TEST = {"clinical_trial", "media_processing", "shared_dotfiles", "tiktok", "user_tracking"}


def calls_of(sample: dict) -> dict:
    """Non-submit tool calls in order: hashed key, tool family, prior edits; and whether the non-submit tool results
    come in the same order as their calls (results are otherwise only checked to exist)."""
    result_ids = [m.get("tool_call_id") for m in sample["messages"] if m["role"] == "tool"]
    has_result, call_ids, out = set(result_ids), [], []
    for m in sample["messages"]:
        if m["role"] != "assistant" or not m.get("tool_calls"):
            continue
        for tc in m["tool_calls"]:
            if tc["function"] == RUNNER_SUBMIT:
                continue
            assert tc.get("id") in has_result, "a tool call has no result"
            call_ids.append(tc.get("id"))
            args = tc.get("arguments") or {}
            fam = f"text_editor {args.get('command')}" if tc["function"] == "text_editor" else ("bash" if tc["function"].endswith("bash") else "other")
            out.append({"key": hashlib.sha256((tc["function"] + " " + json.dumps(args, sort_keys=True)).encode()).hexdigest(),
                        "family": fam, "prior_edits": sum(c["edit"] for c in out), "edit": is_edit(tc["function"], args)})
    keep = set(call_ids)
    return {"calls": out, "results_in_call_order": [r for r in result_ids if r in keep] == call_ids}


def load_file(job):
    path, wanted = job
    want = {(sid, ep): key for key, sid, ep in wanted}
    got = []
    with zipfile.ZipFile(path) as z:
        for n in z.namelist():
            if n.startswith("samples/") and n.endswith(".json"):
                x = json.loads(z.read(n))
                if (x["id"], x["epoch"]) in want:
                    got.append((want[(x["id"], x["epoch"])], calls_of(x)))
    assert len(got) == len(want), f"{Path(path).name}: found {len(got)} of {len(want)} samples"
    return got


def main():
    ex = json.loads(EXCLUSIONS.read_text())
    cells = ex["paired_attack_cells"]["cells"]
    chosen, cell_set = {}, set(cells)
    for s in sorted(ex["samples"], key=lambda s: s["key"]):
        if s["arm"] in ARMS and s["cell"] in cell_set:
            chosen.setdefault((ARMS[s["arm"]], s["cell"]), s)
    assert all((a, c) in chosen for a in ARMS.values() for c in cells), "a paired cell lacks a sample"
    jobs = defaultdict(list)
    for (arm, cell), s in chosen.items():
        jobs[str(DATA / s["cohort"] / s["file"])].append((f"{arm}|{cell}", s["sample_id"], s["epoch"]))
    with ProcessPoolExecutor(14) as pool:
        rec = dict(r for part in pool.map(load_file, jobs.items()) for r in part)
    env_of = json.loads(FOLDS.read_text())["main_task_environment"]

    vol = modal.Volume.from_name(VOL, version=2)
    runs_raw = b"".join(vol.read_file(f"{PACK}/runs.L19.json"))
    kept, dup = {}, 0  # (arm, cell) -> (pack sample key, kept flags of the run's output_end positions, in order)
    for r in json.loads(runs_raw):
        a = {v: k for k, v in PACK_ARM.items()}.get(r["arm"])
        if a:
            dup += (a, r["cell"]) in kept
            kept[(a, r["cell"])] = (r["key"], [p["kept"] for p in r["positions"] if p["kind"] == "output_end"])

    def flags(a, c):
        """The run's kept flags when its cut status is knowable, else None."""
        k = kept.get((a, c))
        r = rec[f"{a}|{c}"]
        ok = k is not None and k[0] == chosen[(a, c)]["key"] and r["results_in_call_order"] and len(k[1]) == len(r["calls"])
        return k[1] if ok else None
    check = Counter()
    for a in ARMS.values():
        for c in cells:
            check["runs"] += 1
            check["in_pack"] += (a, c) in kept
            check["same_sample_key"] += (a, c) in kept and kept[(a, c)][0] == chosen[(a, c)]["key"]
            check["results_in_call_order"] += rec[f"{a}|{c}"]["results_in_call_order"]
            check["cut_status_knowable"] += flags(a, c) is not None
    check["duplicate_pack_rows"] = dup

    counts = defaultdict(Counter)
    for a, b in (("sim", "e1"), ("e2", "e1")):
        for c in cells:
            ca, cb = rec[f"{a}|{c}"]["calls"], rec[f"{b}|{c}"]["calls"]
            qa, qb = defaultdict(list), defaultdict(list)
            for i, x in enumerate(ca):
                qa[x["key"]].append(i)
            for i, x in enumerate(cb):
                qb[x["key"]].append(i)
            main_task = c.split(":")[0]
            split = "test" if env_of[main_task] in TEST else "dev"
            ka, kb = flags(a, c), flags(b, c)
            for k in qa.keys() & qb.keys():
                for i, j in zip(qa[k], qb[k]):
                    pre = ca[i]["prior_edits"] == 0 and cb[j]["prior_edits"] == 0
                    fam = "bash" if ca[i]["family"] == "bash" else ("text_editor" if ca[i]["family"].startswith("text_editor") else "other")
                    cut = "unknown" if ka is None or kb is None else ("both_in_cut" if ka[i] and kb[j] else "not_both_in_cut")
                    for subset in ("all", "pre_edit") if pre else ("all",):
                        for fm in (fam, "any"):
                            g = counts[f"{a}_vs_{b}|{split}|{subset}|{fm}"]
                            g["pairs"] += 1
                            g[f"pairs_{cut}"] += 1
                            g[f"cell:{c}"] = 1
                            g[f"task:{main_task}"] = 1
    out = {"label": "H1 planning: identical-call pairs for whole-output swaps (counts only; preliminary until reviewed)",
           "cut_status_check": dict(check), "rows": {}}
    for k in sorted(f"{t}|{sp}|{sub}|{fm}" for t in ("sim_vs_e1", "e2_vs_e1") for sp in ("test", "dev")
                    for sub in ("all", "pre_edit") for fm in ("bash", "text_editor", "other", "any")):
        g = counts[k]  # zero rows are kept
        out["rows"][k] = {"pairs": g["pairs"], **{f"pairs_{x}": g[f"pairs_{x}"] for x in ("both_in_cut", "not_both_in_cut", "unknown")},
                          "cells": sum(1 for x in g if x.startswith("cell:")), "main_tasks": sum(1 for x in g if x.startswith("task:"))}
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip()
    out |= {"git": git, "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "inputs": {EXCLUSIONS.name: hashlib.sha256(EXCLUSIONS.read_bytes()).hexdigest(), FOLDS.name: hashlib.sha256(FOLDS.read_bytes()).hexdigest(),
                       "pack_runs_json": hashlib.sha256(runs_raw).hexdigest(), "archives": str(DATA)}}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    (HERE / "results").mkdir(exist_ok=True)
    dst = HERE / "results" / f"identical_calls_{stamp}.json"
    with open(dst, "x") as fh:
        fh.write(json.dumps(out, indent=1))
    print("cut status check:", dict(check))
    for k, v in out["rows"].items():
        print(k, v)
    print("->", dst)


if __name__ == "__main__":
    main()
