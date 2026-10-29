"""RULE.md Addition 3, part A, POST HOC: the CRLF/curl tell split by acting model. Sol's #36 runs vs Qwen3.5-27B's own runs,
real vs simulated, as staged for the readers (scrub 0, #36's reader format). Per run: curl -i/-I/-v calls, CRLF-bearing
outputs (`\\r\\n`, outside dpkg output), CRLF-bearing curl -i/-I/-v outputs, header-block lines without \\r (F12's
target), whole run and within the first 21 replies (tool messages). Means per arm with a 2,000-draw main-task bootstrap per
acting model; tell gap G = real - simulated CRLF-bearing outputs per run; ratios Qwen3.5 / Sol. Modal CPU; counts only
(attack runs: scripted counts and booleans only; no text).

  PYTHONPATH=src uv run --no-project --python 3.12 --with 'modal==1.6.1' modal run --detach \\
      experiments/2026-10-14_q2_ffm_iter1/model_split.py
"""
import hashlib
import json
import subprocess
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).parent
REPO = HERE.parents[1] if modal.is_local() else Path("/root")
FOLDER = "experiments/2026-10-14_q2_ffm_iter1"
if modal.is_local():
    sys.path.insert(0, str(REPO / "src"))
APP = "probe-shift_aswin-surya_sub-q2-ffm1_split_kill-2026-11-13"
ACTS = "probe-shift_aswin-surya_sub-la-activations"
OUT = "2026-10-14_q2_ffm_iter1/results"
Q35 = "2026-10-14_q3onpol_q35own_inputs_v1"  # d9's staging: run_onpol_stage.py, rule @ f396f9f2; record @ ce2f68ca
Q35_SHARDS = {  # shard -> (arm, runs, sha256) from stage_result_2026-10-14_011819.json @ ce2f68ca; 6 and 7 (reproduction) skipped
    0: ("q35-real-attack-1", 193, "886f3cd618482a3d2e87f6ea235f64f7dac558723c3b874bb9ea9460c8f58715"),
    1: ("q35-real-attack-2", 192, "7d827bc2bcd0c0e6fac876b504431db0ea228edeae1602be2b36e5f94fa16d5d"),
    2: ("q35-real-honest-a", 191, "af209dd48ac77f3afc8ff3cef460c169cece55643f26d7c1f1c123f5f49260cc"),
    3: ("q35-real-honest-b", 195, "6fa42c50b47622b2d0af1f6bfa28c49f05757d1994763e0c96fcebea477a597e"),
    4: ("q35-sim-attack", 191, "89c7347c885a078e71a30eca9af19bfcbf1abf93a0611cf20528eab25bb1e014"),
    5: ("q35-sim-honest", 19, "44fa048faa1093233f7ff2d756b9ca70f4b59fcf84bd70604e006c7f41940d59")}
Q35_ARM = {"q35-real-attack-1": "real attack", "q35-real-attack-2": "real attack", "q35-real-honest-a": "real honest",
           "q35-real-honest-b": "real honest", "q35-sim-attack": "sim attack", "q35-sim-honest": "sim honest"}
RBASE = "2026-10-06_la_segment-end_qwen3-32b_v1"  # #36's staged inputs (Sol): real-honest, e1, e2, sim-attack
V4 = "2026-10-06_la_segment-end-v4_qwen3-32b_v1"  # Sol's simulated honest (as simulated), the same reader format
SOL_ARM = {"real-honest": "real honest", "real-attack-e1": "real attack", "real-attack-e2": "real attack", "sim-attack": "sim attack"}
RUNS36 = REPO / "experiments/2026-10-06_q3_attack_detector/results/run_scores_2026-10-06_181548/runs.json"
B, SEED, FIRST = 2000, 20261014, 21
MEASURES = ("replies", "curl -i/-I/-v calls", "CRLF-bearing outputs", "CRLF-bearing curl -i/-I/-v outputs",
            "header-block lines without \\r", "runs with a CRLF-bearing output")

app = modal.App(APP)
acts = modal.Volume.from_name(ACTS, version=2).read_only()
image = (modal.Image.debian_slim(python_version="3.12").pip_install("numpy==2.5.4")
         .add_local_python_source("probe_shift")
         .add_local_file(HERE / "f12_lib.py", f"/root/{FOLDER}/f12_lib.py")
         .add_local_file(REPO / "splits" / "exclusions_v4.json", "/root/splits/exclusions_v4.json")
         .add_local_file(REPO / "experiments/2026-10-07_q2_rendering_swap/rerender.py", "/root/swap/rerender.py")
         .add_local_file(REPO / "experiments/2026-10-07_q2_rendering_swap/str_replace_tells.py", "/root/swap/str_replace_tells.py")
         .add_local_file(REPO / "experiments/2026-10-09_fixed_sim/fixes.py", "/root/fsim/fixes.py"))


class Gate(Exception):
    """A failed check; its message holds counts only."""


def say(*a):
    print(datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%H:%M:%S"), *a, flush=True)


def cmd_family(tool: str, args: dict) -> str:
    """count_classes.py's command families (first match in this order), from the call's arguments."""
    import re
    if tool == "text_editor":
        return "text_editor"
    if not tool.endswith("bash"):
        return "other tool"
    c = str(args.get("cmd") or args.get("command") or "")
    if re.search(r"\b(?:apt|apt-get|dpkg)\b", c):
        return "apt/dpkg"
    if re.search(r"\bcurl\b", c):
        return "curl -i/-I/-v" if re.search(r"\bcurl\b[^|;&\n]*\s(?:-[a-zA-Z]*[iIv][a-zA-Z]*|--include|--verbose|--head)\b", c) else "curl other"
    for name, rx in (("pip", r"\bpip3?\b|-m pip\b"), ("npm/yarn/pnpm", r"\b(?:npm|yarn|pnpm|npx)\b"), ("git", r"\bgit\b"),
                     ("wget", r"\bwget\b")):
        if re.search(rx, c):
            return name
    return "other"


def per_run(msgs, L, fixes) -> dict:
    """The measures for one run: whole run and the first FIRST replies (tool messages in order)."""
    tool_ix = [i for i, m in enumerate(msgs) if m["role"] == "tool"]
    early = set(tool_ix[:FIRST])
    calls = fixes.calls_of(msgs)
    fam = {x["msg"]: L.call_family(x["args"]) for x in calls if x["tool"].endswith("bash")}
    cmd = {x["msg"]: cmd_family(x["tool"], x["args"]) for x in calls}
    out = {"CRLF-bearing outputs by command family (whole run)": {}}
    for win, keep in (("whole run", set(tool_ix)), (f"first {FIRST} replies", early)):
        r = dict.fromkeys(MEASURES, 0)
        r["replies"] = len(keep)
        for i in keep:
            t = msgs[i].get("content")
            if not isinstance(t, str):
                continue
            r["curl -i/-I/-v calls"] += fam.get(i) == L.FAMILIES[0]
            crlf = "\r\n" in t and L.DPKG not in t
            r["CRLF-bearing outputs"] += crlf
            r["CRLF-bearing curl -i/-I/-v outputs"] += crlf and fam.get(i) == L.FAMILIES[0]
            if crlf and win == "whole run":
                k = cmd.get(i, "unpaired output")
                out["CRLF-bearing outputs by command family (whole run)"][k] = out["CRLF-bearing outputs by command family (whole run)"].get(k, 0) + 1
            if i in fam:
                lines = t.split("\n")
                r["header-block lines without \\r"] += sum(not lines[j].endswith("\r") for j, _ in L.typed_lines(t))
        r["runs with a CRLF-bearing output"] = int(r["CRLF-bearing outputs"] > 0)
        out[win] = r
    return out


@app.function(image=image, volumes={"/acts": acts}, cpu=16, memory=65536, timeout=3600, retries=0)
def split(git: str, stamp: str, sol_keys: dict, task_of: dict) -> dict:
    import numpy as np
    for p in ("/root/swap", "/root/fsim", f"/root/{FOLDER}"):
        sys.path.insert(0, p)
    import f12_lib as L
    import fixes

    root = Path("/acts")
    runs = {"Qwen3.5-27B": [], "Sol": []}  # (arm, main task, measures)
    for s, (arm, n, h) in Q35_SHARDS.items():
        raw = (root / Q35 / "inputs" / f"shard_{s}.jsonl").read_bytes()
        if hashlib.sha256(raw).hexdigest() != h:
            raise Gate(f"Qwen3.5 shard_{s}: sha256 differs from the stage record")
        rows = [json.loads(line) for line in raw.decode().splitlines()]
        if len(rows) != n or {r["arm"] for r in rows} != {arm} or any(not r.get("main_task") for r in rows):
            raise Gate(f"Qwen3.5 shard_{s}: not {n} runs of {arm} with a main task")
        runs["Qwen3.5-27B"] += [(Q35_ARM[arm], r["main_task"], per_run(r["messages"], L, fixes)) for r in rows]
    want = {(a, k) for a, ks in sol_keys.items() for k in ks}
    found = set()
    for run_, arms in ((RBASE, set(SOL_ARM)), (V4, {"sim-honest"})):
        for p in sorted((root / run_ / "inputs").glob("shard_*.jsonl")):
            for line in open(p):
                it = json.loads(line)
                ak = (it["arm"], it["key"])
                if it["arm"] in arms and ak in want:
                    if ak in found:
                        raise Gate("a Sol run twice")
                    found.add(ak)
                    runs["Sol"].append((SOL_ARM.get(it["arm"], "sim honest"), task_of[f"{it['arm']}|{it['key']}"], per_run(it["messages"], L, fixes)))
    if found != want:
        raise Gate(f"Sol: {len(want - found)} wanted runs missing")
    say("runs", {m: len(v) for m, v in runs.items()})

    arms = ("real honest", "sim honest", "real attack", "sim attack")
    by_fam = {m: {a: defaultdict(int) for a in arms} for m in runs}
    for model, rs in runs.items():
        for a, _, m in rs:
            for k, v in m["CRLF-bearing outputs by command family (whole run)"].items():
                by_fam[model][a][k] += v
    crlf_source = {m: {a: {"CRLF-bearing outputs (whole run)": sum(d.values()),
                           "share whose call is curl -i/-I/-v": (d["curl -i/-I/-v"] / sum(d.values())) if sum(d.values()) else None,
                           "by command family": dict(sorted(d.items()))} for a, d in v.items()} for m, v in by_fam.items()}
    draws = {}
    res = {"runs": {m: {a: sum(x[0] == a for x in v) for a in arms} for m, v in runs.items()}, "means": {}, "tell gap": {}}
    for mi, (model, rs) in enumerate(runs.items()):
        tasks = sorted({t for _, t, _ in rs})
        ti = {t: i for i, t in enumerate(tasks)}
        g = np.random.default_rng([SEED, mi + 1])
        W = np.vstack([np.ones(len(tasks))] + [np.bincount(g.integers(0, len(tasks), len(tasks)), minlength=len(tasks)) for _ in range(B)])
        res["means"][model], draws[model] = {}, {}
        for a in arms:
            sel = [(t, m) for arm, t, m in rs if arm == a]
            w = W[:, [ti[t] for t, _ in sel]]
            for win in ("whole run", f"first {FIRST} replies"):
                for meas in MEASURES:
                    x = np.array([m[win][meas] for _, m in sel], float)
                    mean = (w * x).sum(1) / w.sum(1)
                    draws[model][(a, win, meas)] = mean
                    res["means"][model][f"{a}|{win}|{meas}"] = {"point": float(mean[0]), "ci95": [float(np.percentile(mean[1:], 2.5)), float(np.percentile(mean[1:], 97.5))]}
        for side in ("honest", "attack"):
            for win in ("whole run", f"first {FIRST} replies"):
                for meas in ("CRLF-bearing outputs", "runs with a CRLF-bearing output"):
                    G = draws[model][(f"real {side}", win, meas)] - draws[model][(f"sim {side}", win, meas)]
                    draws[model][("G", side, win, meas)] = G
                    res["tell gap"].setdefault(model, {})[f"{side}|{win}|{meas} (real - sim)"] = {
                        "point": float(G[0]), "ci95": [float(np.percentile(G[1:], 2.5)), float(np.percentile(G[1:], 97.5))]}
    ratios, verdicts = {}, {}
    for side in ("attack", "honest"):
        for win in (f"first {FIRST} replies", "whole run"):
            C = draws["Qwen3.5-27B"][(f"real {side}", win, "curl -i/-I/-v calls")] / draws["Sol"][(f"real {side}", win, "curl -i/-I/-v calls")]
            Gq, Gs = draws["Qwen3.5-27B"][("G", side, win, "CRLF-bearing outputs")], draws["Sol"][("G", side, win, "CRLF-bearing outputs")]
            R = Gq / Gs
            ci = lambda v: [float(np.percentile(v[1:], 2.5)), float(np.percentile(v[1:], 97.5))]  # noqa: E731
            gs_ci = ci(Gs)
            ratios[f"{side}|{win}"] = {"C = curl calls per run, real, Qwen3.5 / Sol": {"point": float(C[0]), "ci95": ci(C)},
                                       "R = tell gap G, Qwen3.5 / Sol": {"point": float(R[0]), "ci95": ci(R)},
                                       "Sol's G interval excludes 0": gs_ci[0] > 0 or gs_ci[1] < 0}
            r_ci, c_ci = ci(R), ci(C)
            if gs_ci[0] <= 0 <= gs_ci[1]:  # Codex r1 nit: test zero inclusion
                v = "INCONCLUSIVE (Sol's tell gap interval includes 0)"
            elif c_ci[1] < 1 and r_ci[1] < 1:
                v = "EXPLAINS (consistent: fewer curl calls and a smaller tell on Qwen3.5's runs)"
            elif r_ci[0] >= 0.75:
                v = "DOES NOT EXPLAIN (Qwen3.5's tell is at least 3/4 of Sol's)"
            else:
                v = "INCONCLUSIVE"
            verdicts[f"{side}|{win}"] = v + (" [primary]" if (side, win) == ("attack", f"first {FIRST} replies") else
                                              " [descriptive: 19 simulated honest runs]" if side == "honest" else " [beside]")
    out = {"POST HOC": True, "status": "preliminary", "rule": "RULE.md Addition 3 (part A)", "git": git, "stamp": stamp,
           "inputs": {"Qwen3.5-27B": Q35, "Sol": [RBASE, V4]}, "draws": B, "seed": SEED, **res, "CRLF source by acting model and arm": crlf_source,
           "ratios": ratios, "verdicts": verdicts}
    return out


@app.local_entrypoint()
def main():
    for f in (Path(__file__), HERE / "RULE.md", HERE / "f12_lib.py"):
        if subprocess.run(["git", "diff", "--quiet", "HEAD", "--", str(f)], cwd=HERE).returncode \
                or subprocess.run(["git", "ls-files", "--error-unmatch", str(f)], cwd=HERE, capture_output=True).returncode:
            raise SystemExit(f"{f.name} differs from HEAD: commit it before running")
    runs36 = json.loads(RUNS36.read_text())
    ex = json.loads((REPO / "splits" / "exclusions_v4.json").read_text())
    sol_keys = {a: sorted(r["key"] for r in runs36 if r["arm"] == a) for a in SOL_ARM}
    sol_keys["sim-honest"] = sorted(r["key"] for r in ex["samples"] if r["arm"] == "sim-honest" and not r["reason"])
    task_of = {f"{r['arm']}|{r['key']}": r["main_task"] for r in runs36 if r["arm"] in SOL_ARM}
    task_of |= {f"sim-honest|{r['key']}": r["cell"] for r in ex["samples"] if r["arm"] == "sim-honest" and not r["reason"]}
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip()
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    out = split.remote(git, stamp, sol_keys, task_of)
    p = HERE / "results" / f"model_split_{stamp}.json"
    p.write_text(json.dumps(out, indent=1) + "\n")
    print("wrote", p, out["verdicts"])
