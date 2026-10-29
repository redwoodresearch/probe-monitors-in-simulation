"""Run length beside the attempt-rate record (the orchestrator's follow-up, approved about 23:35 PT 10-15): per labelled
run, its tool calls, model turns and staged tokens, counted from the pinned staged inputs; median and IQR per arm (sim,
run1, run2) for attempted vs not-attempted (unclear or declined) runs, per set and pooled. Infra runs are left out, as
in the rate record. Keys and counts only: one Modal CPU container reads the staged files and keeps only lengths.

Staged files: S_new and nm360 from their label records' staged_sets (sha256 -> world, staged set) on the chain Volume;
T196 from the corrected-B manifest's epochs (sha256-pinned) on the q4 Volume. A (world, cell) found in more than one
staged file is ambiguous (the label builder merged batches by key, so the counted run can't be told apart): counted and
left out of the medians. Prediction (a1, before counting): sim runs have about 30% fewer tool calls than real runs;
not-attempted runs are shorter than attempted ones in every arm.

  PYTHONPATH=src uv run --no-project --python 3.12 --with "modal==1.6.1" --with "numpy==2.3.3" modal run \\
      experiments/2026-10-15_attempt_rate/run_run_length.py
"""

import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).resolve().parent
APP = "probe-shift_aswin-surya_sub-attemptrate_len_kill-2026-11-15"
VOLS = {"chain": "probe-shift_aswin-surya_sub-q3su311_acts_kill-2026-11-14", "q4": "probe-shift_aswin-surya_sub-q4own_evals_kill-2026-11-13"}
SRC = {"labels311": ("806b7a63a6", "experiments/2026-10-14_q3_overnight_chain/results/hygiene/attempt_labels_311.json", "22fc502fc3faa1323e5bee5927a2d6c8ef0ec158e55e76186b54425a2293356b"),
       "labels360": ("f60fb6cff", "experiments/2026-10-14_q3_overnight_chain/results/hygiene/attempt_labels_nm360.json", "5d28a4f51dbdbfb79883553068ae06b29329117e49deeeff6cbed60e7482a415"),
       "m710": ("32abe4120", "experiments/2026-10-15_fpr_sweep_507_prep/config/membership710.json", "6516d15fab77d7f344497e72db10b60c5f709b35c3a1859a2c9e016d2fc6e4ad")}
LOCAL = {"t196": (HERE / "data" / "claimc_labels_reduced.json", "d00e88773fc2a71bef51e1ff0ccc2559932f5c3a306a8f5604c5248e797dc995"),
         "manifest": (HERE / "data" / "correctedb_manifest_v1.json", "bf847d425b80d95ebb75a9ed1d524fc68c9efcd014710ddc18e445ed9d373ebc")}
RATE_RECORD = "experiments/2026-10-15_attempt_rate/results/attempt_rate_2026-10-15_234954.json"
ARMS, LAB = ("sim", "run1", "run2"), ("attempted", "unclear", "declined")

app = modal.App(APP)
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy==2.3.3")
vols = {k: modal.Volume.from_name(v, version=2).read_only() for k, v in VOLS.items()}


@app.function(image=image, cpu=4, memory=16384, timeout=1200, retries=0, volumes={f"/{k}": v for k, v in vols.items()})
def count(files: list, classes: dict) -> dict:
    """files: [set, world, volume, path, sha256, keys or None]; classes: {set: {world|cell: label}}. Returns lengths by key."""
    import traceback

    import numpy as np
    try:
        seen, runs = {}, {}
        for s, world, vol, path, sha, keys in files:
            raw = (Path(f"/{vol}") / path).read_bytes()
            if hashlib.sha256(raw).hexdigest() != sha:
                raise RuntimeError(f"{s} {world}: a staged file does not match its pinned sha256")
            for line in raw.decode().splitlines():
                row = json.loads(line)
                if keys is not None and row["key"] not in keys:
                    continue
                cell = row.get("task") or row.get("cell")
                calls = row.get("calls") or []
                k = (s, f"{world}|{cell}")
                seen[k] = seen.get(k, 0) + 1
                ends = [c for c in calls if c.get("kind") == "call_end"]  # one per retained tool call (output_end carries the receiving turn's irid)
                runs[k] = {"tool_calls": len(ends), "model_turns": len({c["irid"] for c in ends}),
                           "tool_calls_in_cut": sum(c.get("readout") is not None for c in ends), "staged_tokens": len(row.get("ids") or [])}
                del row
        ambiguous = {f"{s}|{k}" for (s, k), n in seen.items() if n > 1}
        q = lambda v: {"n": len(v), "median": float(np.median(v)), "iqr": [float(np.percentile(v, 25)), float(np.percentile(v, 75))]} if v else {"n": 0}
        labelled = {(s, k) for s in classes for k, lab in classes[s].items() if lab in LAB}
        out = {"ambiguous": sorted(ambiguous), "per_run (labelled, unambiguous)": {f"{s}|{k}": v for (s, k), v in sorted(runs.items())
                                                                                if (s, k) in labelled and f"{s}|{k}" not in ambiguous}, "summary": {}}
        for view in [*classes, "pooled"]:
            sel = [s for s in classes] if view == "pooled" else [view]
            res = {}
            for a in ARMS:
                for grp, want in (("attempted", ("attempted",)), ("not attempted", ("unclear", "declined"))):
                    ks = [(s, k) for s in sel for k, lab in classes[s].items() if k.startswith(f"{a}|") and lab in want]
                    have = [(s, k) for s, k in ks if (s, k) in runs and f"{s}|{k}" not in ambiguous]
                    res[f"{a} {grp}"] = {m: q([runs[x][m] for x in have]) for m in ("tool_calls", "model_turns", "tool_calls_in_cut", "staged_tokens")} | {
                        "labelled": len(ks), "no staged row": sum((s, k) not in runs for s, k in ks), "ambiguous": sum(f"{s}|{k}" in ambiguous for s, k in ks)}
            out["summary"][view] = res
        return out
    except Exception as e:  # no row content may leave: the type and this file's line numbers only
        raise RuntimeError(f"{type(e).__name__} at lines {[f.lineno for f in traceback.extract_tb(e.__traceback__) if f.filename.endswith('run_run_length.py')]}") from None


@app.local_entrypoint()
def main():
    git = lambda *a: subprocess.run(["git", *a], capture_output=True, check=True, cwd=HERE).stdout
    if git("status", "--porcelain", "--untracked-files=no").strip() or not git("branch", "-r", "--contains", "HEAD").strip():
        raise SystemExit("commit and push first")
    src = {}
    for name, (sha, path, want) in SRC.items():
        b = git("show", f"{sha}:{path}")
        if hashlib.sha256(b).hexdigest() != want:
            raise SystemExit(f"{name}: sha256 differs from its pin")
        src[name] = json.loads(b)
    for name, (p, want) in LOCAL.items():
        if hashlib.sha256(p.read_bytes()).hexdigest() != want:
            raise SystemExit(f"{name}: sha256 differs from its pin")
        src[name] = json.loads(p.read_bytes())
    # labels per run, by the rate record's rules (S_new: as labelled; nm360: after exclusions and folder coverage; T196: reduced file)
    m = src["m710"]
    scored = {k for v in m["folder_keys"].values() for k in v}
    classes = {"S_new": {k: v for k, v in src["labels311"]["labels"].items() if k.split("|")[0] in ARMS},
               "nm360": {k: v for k, v in src["labels360"]["labels"].items() if k.split("|")[0] in ARMS and k not in m["excluded_runs"] and k in scored},
               "T196": {k: v for k, v in src["t196"]["labels"].items() if k.split("|")[0] in ARMS}}
    files = []
    for s, name in (("S_new", "labels311"), ("nm360", "labels360")):
        for sha, v in src[name]["staged_sets"].items():
            if v["world"] in ARMS:
                files.append([s, v["world"], "chain", v["staged_set"], sha, None])
    by_file = {}
    for e in src["manifest"]["epochs"]:
        if e["world"] in ARMS:
            by_file.setdefault((e["world"], e["staged_inputs_file"], e["staged_inputs_sha256"]), []).append(e["key"])
    files += [["T196", w, "q4", f, sha, ks] for (w, f, sha), ks in sorted(by_file.items())]
    res = count.remote(files, classes)
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    head = git("rev-parse", "HEAD").decode().strip()
    rec = {"label": "POST HOC descriptive; PRELIMINARY; run length beside the attempt-rate record", "git": head, "stamp (PT)": stamp, "app": APP,
           "rate_record": RATE_RECORD, "sources": {k: {"commit": v[0], "path": v[1], "sha256": v[2]} for k, v in SRC.items()},
           "local_sources": {k: {"file": p.name, "sha256": w} for k, (p, w) in LOCAL.items()}, "staged_files": len(files),
           "definitions": {"tool_calls": "retained staged calls: call_end entries (staging drops submit and some other calls; the whole run if the stager lists calls past the cut)",
                           "model_turns": "distinct irid among the call_end entries (turns with a retained call)",
                           "tool_calls_in_cut": "call_end entries with a readout inside the 32,768-token cut", "staged_tokens": "len(ids), cut at 32,768"}} | res
    dst = HERE / "results" / f"run_length_{stamp}.json"
    dst.open("x").write(json.dumps(rec, indent=1))
    lines = [f"staged files {len(files)}; ambiguous runs {len(res['ambiguous'])}"]
    for view, r in res["summary"].items():
        lines.append(f"[{view}]")
        for g, v in r.items():
            t = v["tool_calls"]
            lines.append(f"  {g}: labelled {v['labelled']}, counted {t['n']}, no staged row {v['no staged row']}, ambiguous {v['ambiguous']}; tool calls median "
                         f"{t.get('median', float('nan')):.0f} IQR {t.get('iqr')}; model turns median {v['model_turns'].get('median', float('nan')):.0f}; "
                         f"in-cut calls median {v['tool_calls_in_cut'].get('median', float('nan')):.0f}; staged tokens median {v['staged_tokens'].get('median', float('nan')):.0f}")
    (HERE / "results" / f"run_length_{stamp}.printout.txt").open("x").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("->", dst)
