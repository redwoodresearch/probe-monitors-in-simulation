"""H1 amendment 2 (PLAN.md on branch aswin/claimb-h-plans): the paired-swap pilot on the dev environments. Whole-output
swaps on identical bash calls in B-dev's evened runs, read through the swapped output by B-dev's reader, scored by
B-dev's frozen per-environment L63 refits (both imported unchanged and checked by hash). Keys and numbers only: no
output or command text is printed, saved or returned.

  --mode cells    CPU: match identical bash calls on the level-1 staged inputs (frozen manifest), recompute gate 1 on it
                  from d9's stored per-output scores, draw the pilot pairs and 20 gate slots; cells.json.
  --mode extract  2 H100 shards: 40 S<-R reads (real e1 output in the sim slot), 40 R<-S reads (sim output in the e1
                  slot), 20 unswapped gate reads, each through the slot only.
  --mode score    1 H100: gate checks (pack row bitwise, stored score within 1e-5), then the paired noise and the
                  projected SE of S (amendment 2's number); results/h1_swap_pilot_<stamp>.json.

  PYTHONPATH=src uv run --no-project --python 3.12 --with 'modal>=1.4.3' --with numpy \\
      modal run --detach experiments/2026-10-11_claimB_h1_text_vs_layout/run_h1_swap_modal.py --mode cells|extract|score
"""

import hashlib
import json
import random
import re
import subprocess
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).parent
BDEV = HERE.parent / "2026-10-11_claimB_slips_evened_probe"
sys.path.insert(0, str(BDEV) if modal.is_local() else "/root")
import run_slips_evened_modal as B  # noqa: E402  (B-dev's reader, evening, refit and staged inputs)

B_SCRIPT = "0abee4a4d8730c0d3330e4702d179dfd8d3a11a228279e18e84fc4baa1d4cd06"
B_DEPS = "33d539e9a646c5adfeb3b8069c708c3f696f1ac794e09c26a281448bf0c6a65d"
M = B.M
OUT, RECORDS = "claimB/h1_swap_v1", "claimB/q11_remainder_v1/records.json"
N_PILOT, N_GATE, SHARDS, DRAWS, SE_PROCEED, MIN_TASKS = 40, 20, 2, 2000, 0.12, 8
SIM, E1 = M.SIM, M.REAL
TE_WRITES = {"create", "str_replace", "insert"}  # the failure-hiding edit rule, copied
BASH_WRITE = re.compile(r"sed -i|\btee\b|cat\s*>|cat\s*<<|\bpatch\b|(?:echo|printf)[^|\n]*>\s*[^&\s]")
SCRIPT_SHA = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

app = modal.App("probe-shift_aswin-surya_sub-claimb-h1_swap_kill-2026-11-11")
image = B.image.add_local_file(BDEV / "run_slips_evened_modal.py", "/root/run_slips_evened_modal.py")


def check_b():
    if (B.SCRIPT_SHA, B.DEPS_SHA) != (B_SCRIPT, B_DEPS):
        raise ValueError("B-dev's code here differs from the code that ran")


def tool_calls(msgs):
    """Per tool message index: (function, sorted-JSON arguments hash, prior detected edits in this run)."""
    out, pending, edits = {}, [], 0
    for i, m in enumerate(msgs):
        if m["role"] == "assistant":
            pending = list(m.get("tool_calls") or [])
        elif m["role"] == "tool" and pending:
            fn = pending.pop(0)["function"]
            a = fn.get("arguments")
            a = json.loads(a) if isinstance(a, str) else a or {}
            name = fn["name"]
            out[i] = (name, hashlib.sha256((name + " " + json.dumps(a, sort_keys=True)).encode()).hexdigest(), edits)
            cmd = str(a.get("cmd") or a.get("command") or "") if isinstance(a, dict) else ""
            edits += (a.get("command") in TE_WRITES) if name == "text_editor" else (name.endswith("bash") and bool(BASH_WRITE.search(cmd)))
    return out


def step_action(msgs) -> dict:
    """tool-message index -> (step, action), the reader's counting (as d9's rescore)."""
    out, step, action = {}, -1, 0
    for i, m in enumerate(msgs):
        if m["role"] == "assistant":
            step, action = step + 1, 0
        elif m["role"] == "tool":
            out[i] = (step, action)
            action += 1
    return out


def evened_runs():
    runs = B.dev_runs()
    return {k: B.evened(r["messages"]) + (r,) for k, r in runs.items()}  # (evened msgs, changed outputs, range idx, row)


@app.function(image=image, cpu=8, memory=32768, volumes={"/acts": B.P.vol}, timeout=15 * 60, retries=0)
def cells() -> dict:
    import numpy as np

    check_b()
    ev = evened_runs()
    after = {(x["cell"], x["arm"], x["step"], x["action"]): x["after"] for x in json.loads((Path("/acts") / RECORDS).read_text())}
    pairs, dropped = [], defaultdict(int)
    for cell in sorted({c for c, _ in ev}):
        if not all((cell, a) in ev for a in (SIM, E1, "real-attack-e2")):
            continue
        for a, b in ((SIM, E1), ("real-attack-e2", E1)):
            ca, cb = tool_calls(ev[(cell, a)][0]), tool_calls(ev[(cell, b)][0])
            sa_a, sa_b = step_action(ev[(cell, a)][0]), step_action(ev[(cell, b)][0])
            qa, qb = defaultdict(list), defaultdict(list)
            for i, (n, k, _) in sorted(ca.items()):
                qa[k].append(i)
            for i, (n, k, _) in sorted(cb.items()):
                qb[k].append(i)
            for k in sorted(qa.keys() & qb.keys()):
                for i, j in zip(qa[k], qb[k]):
                    if not ca[i][0].endswith("bash"):
                        continue
                    ga, gb = after.get((cell, a, *sa_a[i])), after.get((cell, b, *sa_b[j]))
                    if ga is None or gb is None:  # outside the cut, or excluded at score level by B-dev
                        dropped[f"{a}|{b}"] += 1
                        continue
                    pairs.append({"id": len(pairs), "cell": cell, "main_task": cell.split(":")[0], "a": a, "b": b, "ia": i, "ib": j,
                                  "sa": list(sa_a[i]), "sb": list(sa_b[j]), "pre_edit": ca[i][2] == 0 and cb[j][2] == 0, "G": ga - gb})
    se = [p for p in pairs if p["a"] == SIM]
    tasks = sorted({p["main_task"] for p in se})
    by = defaultdict(list)
    for p in se:
        by[p["main_task"]].append(p["G"])
    sums, ns = np.array([sum(by[t]) for t in tasks]), np.array([len(by[t]) for t in tasks])
    w = np.random.default_rng(20261011).multinomial(len(tasks), np.full(len(tasks), 1 / len(tasks)), size=DRAWS)
    bG = (w @ sums) / (w @ ns)
    gate1 = {"G": float(sums.sum() / ns.sum()), "ci95": [float(x) for x in np.percentile(bG, [2.5, 97.5])], "pairs": len(se), "main_tasks": len(tasks)}
    gate1["pass"] = gate1["ci95"][0] > 0
    order = list(tasks)
    random.Random("h1-swap-pilot").shuffle(order)
    pilot_tasks, pilot = [], []  # at most N_PILOT // MIN_TASKS pairs per task, so the pilot spans at least MIN_TASKS tasks
    for t in order:
        if len(pilot) >= N_PILOT:
            break
        pilot_tasks.append(t)
        pilot += [p["id"] for p in se if p["main_task"] == t][: N_PILOT // MIN_TASKS]
    pilot = pilot[:N_PILOT]
    unchanged = sorted(c for (c, a), v in ev.items() if a == SIM and v[1] == 0)
    rng = random.Random("h1-swap-gate")
    gate = []
    for c in rng.sample(unchanged, len(unchanged)):
        sa = step_action(ev[(c, SIM)][0])
        bash = [i for i, (n, _, _) in sorted(tool_calls(ev[(c, SIM)][0]).items()) if n.endswith("bash") and (c, SIM, *sa[i]) in after]
        if bash:
            i = rng.choice(bash)
            gate.append({"cell": c, "i": i, "sa": list(sa[i])})
        if len(gate) == N_GATE:
            break
    plan = {"script_sha256": SCRIPT_SHA, "b_script": B.SCRIPT_SHA, "pairs": pairs, "pairs_dropped": dict(dropped), "gate1": gate1,
            "pilot_pairs": pilot, "pilot_tasks": pilot_tasks, "gate_slots": gate}
    (Path("/acts") / OUT).mkdir(parents=True, exist_ok=True)
    (Path("/acts") / OUT / "cells.json").write_text(json.dumps(plan))
    B.P.vol.commit()
    return {"pairs": {f"{a}|{b}": sum(1 for p in pairs if (p["a"], p["b"]) == (a, b)) for a, b in ((SIM, E1), ("real-attack-e2", E1))},
            "pairs_dropped": dict(dropped), "gate1": gate1, "pilot_pairs": len(pilot), "pilot_tasks": len(pilot_tasks), "gate_slots": len(gate)}


def load_plan():
    raw = (Path("/acts") / OUT / "cells.json").read_bytes()
    plan = json.loads(raw)
    if plan["script_sha256"] != SCRIPT_SHA:
        raise ValueError("cells.json was made by another version of this script")
    return plan, hashlib.sha256(raw).hexdigest()[:12]  # the plan's tag namespaces every read's file


def jobs_for(shard: int):
    """(cell, arm, version, swapped-or-gate messages, [], unswapped truncated messages, full evened messages)."""
    plan, tag = load_plan()
    if not plan["gate1"]["pass"]:
        raise ValueError("binding gate 1 failed on the manifest: no reads")
    ev = evened_runs()
    pairs = {p["id"]: p for p in plan["pairs"]}
    jobs = []
    for pid in plan["pilot_pairs"]:
        p = pairs[pid]
        sim, e1 = ev[(p["cell"], SIM)][0], ev[(p["cell"], E1)][0]
        sr = [dict(m) for m in sim[: p["ia"] + 1]]
        sr[-1] = sr[-1] | {"content": e1[p["ib"]]["content"]}  # the real e1 output in the sim slot
        rs = [dict(m) for m in e1[: p["ib"] + 1]]
        rs[-1] = rs[-1] | {"content": sim[p["ia"]]["content"]}  # the sim output in the e1 slot
        jobs += [(p["cell"], SIM, f"SR|{pid}|{tag}", sr, [], sim[: p["ia"] + 1], sim), (p["cell"], E1, f"RS|{pid}|{tag}", rs, [], e1[: p["ib"] + 1], e1)]
    for g in plan["gate_slots"]:
        full = ev[(g["cell"], SIM)][0]
        jobs.append((g["cell"], SIM, f"gate|{g['i']}|{tag}", [dict(m) for m in full[: g["i"] + 1]], [], full[: g["i"] + 1], full))
    return jobs[shard::SHARDS]


@app.function(image=image, gpu="H100!", cpu=8, memory=65536, volumes={"/acts": B.P.vol, "/hf": B.P.hf}, timeout=35 * 60, retries=0)
def extract(shard: int, model: str, revision: str) -> dict:
    from transformers import AutoTokenizer

    from probe_shift.reader import segment_end_positions

    check_b()
    jobs = jobs_for(shard)
    # cutting a run after the slot must not change how the earlier turns render (the chat template can depend on later
    # messages): the cut, unswapped prefix's token ids must be a prefix of the full run's ids. Checked for every job.
    tok = AutoTokenizer.from_pretrained(model, revision=revision)
    bad = []
    for cell, arm, version, _, _, cut, full in jobs:
        sc, sf = segment_end_positions(tok, cut, M.MAX_TOKENS), segment_end_positions(tok, full, M.MAX_TOKENS)
        ends = [p["token"] for p in sc["positions"] if p["kind"] == "output_end" and p["kept"]]
        if not ends:
            continue  # the slot is outside the cut; the pair is dropped at scoring
        t = max(ends)  # the slot's own readout token; template tokens after it may differ and are not read
        if sc["ids"][: t + 1] != sf["ids"][: t + 1]:
            bad.append(version)
    if bad:
        raise ValueError(f"{len(bad)} cut prefixes render differently from their full runs; no reads")
    B.OUT = OUT  # B-dev's reader writes under OUT/runs; its files keep B-dev's script hash
    return B._read([j[:5] for j in jobs], model, revision, f"shard{shard}") | {"prefix_checks": len(jobs)}


@app.function(image=image, gpu="H100!", cpu=8, memory=65536, volumes={"/acts": B.P.vol}, timeout=15 * 60, retries=0)
def score(env_to_fold: dict) -> dict:
    import numpy as np
    import torch

    from probe_shift.q1_eval import step_table

    check_b()
    B.P.vol.reload()
    plan, tag = load_plan()
    pruns = json.loads((Path("/acts") / M.PACK / "runs.L63.json").read_text())
    t = step_table(pruns, "output_end")
    arm_of = np.array([pruns[i]["arm"] for i in t["run"]])
    efold = np.array([env_to_fold[pruns[i]["environment"]] for i in t["run"]])
    XP = {63: np.load(Path("/acts") / M.PACK / "L63_output_end.npy")}
    probes, fits = {}, {}
    for e in B.DEV:
        probes[e], fits[e] = B.refit(e, 63, env_to_fold, arm_of, efold, XP)

    def s(e, X):
        mean, std, W, b = probes[e]
        return ((((torch.from_numpy(np.asarray(X)).cuda().double() - mean) / std * M.C**0.5) @ W + b)[:, 0]).cpu().numpy()

    ev = evened_runs()
    env_of = {c: r[3]["environment"] for (c, _), r in ev.items()}
    after = {(x["cell"], x["arm"], x["step"], x["action"]): x["after"] for x in json.loads((Path("/acts") / RECORDS).read_text())}

    def slot(cell, arm, version, sa):
        """The read's row at the slot (its last kept output), or None if the slot fell outside the cut."""
        f = Path("/acts") / OUT / "runs" / f"{hashlib.sha256(f'{cell}|{arm}|{version}|{tag}'.encode()).hexdigest()[:16]}.npz"
        z = np.load(f)
        meta = json.loads(str(z["meta"]))
        if (meta["version"], meta["script_sha256"]) != (f"{version}|{tag}", B.SCRIPT_SHA):
            raise ValueError(f"{f.name}: made for another plan or reader")
        steps = [tuple(x) for x in meta["steps"]]
        return z["L63"][steps.index(tuple(sa))].astype(np.float32) if tuple(sa) in steps else None

    key_of = {(pruns[i]["cell"], pruns[i]["arm"]): i for i in range(len(pruns))}
    rows_of = {}
    for k in range(len(t["run"])):
        rows_of.setdefault(int(t["run"][k]), {})[(int(t["step"][k]), int(t["action"][k]))] = k
    gate = []
    for g in plan["gate_slots"]:
        x = slot(g["cell"], SIM, f"gate|{g['i']}", g["sa"])
        k = rows_of[key_of[(g["cell"], SIM)]].get(tuple(g["sa"]))
        bitwise = x is not None and k is not None and np.array_equal(x.astype(np.float16), XP[63][k])
        sc = float(s(env_of[g["cell"]], x[None])[0]) if x is not None else None
        stored = after.get((g["cell"], SIM, *g["sa"]))
        gate.append({"bitwise": bool(bitwise), "score_diff": None if sc is None or stored is None else abs(sc - stored)})
    gate_ok = len(gate) == N_GATE and all(x["bitwise"] and x["score_diff"] is not None and x["score_diff"] < 1e-5 for x in gate)
    fits_ok = all(v["converged"] for v in fits.values())
    pairs = {p["id"]: p for p in plan["pairs"]}
    rec, lost = [], 0
    for pid in plan["pilot_pairs"]:
        p = pairs[pid]
        xs, xr = slot(p["cell"], SIM, f"SR|{pid}", p["sa"]), slot(p["cell"], E1, f"RS|{pid}", p["sb"])
        if xs is None or xr is None:
            lost += 1
            continue
        e = env_of[p["cell"]]
        a_sim, a_e1 = after[(p["cell"], SIM, *p["sa"])], after[(p["cell"], E1, *p["sb"])]
        T, R = a_sim - float(s(e, xs[None])[0]), float(s(e, xr[None])[0]) - a_e1
        rec.append({"task": p["main_task"], "T": T, "R": R, "u": (T + R) / 2, "G": a_sim - a_e1})
    tasks = sorted({r["task"] for r in rec})
    if not tasks:
        return {"proceed": False, "reason": "no pilot pair survived", "gate": {"slots": gate, "pass": gate_ok}, "pilot_pairs_lost_to_cut": lost}
    U = np.array([sum(r["u"] for r in rec if r["task"] == tk) for tk in tasks])
    Gs = np.array([sum(r["G"] for r in rec if r["task"] == tk) for tk in tasks])
    w = np.random.default_rng(20261011).multinomial(len(tasks), np.full(len(tasks), 1 / len(tasks)), size=DRAWS)
    S_star = (w @ U) / (w @ Gs)
    rest = [p for p in plan["pairs"] if p["a"] == SIM and p["main_task"] not in set(plan["pilot_tasks"])]
    k_rest = len({p["main_task"] for p in rest})
    se_pilot = float(np.std(S_star)) if np.isfinite(S_star).all() else float("nan")
    proj = se_pilot * (len(tasks) / k_rest) ** 0.5 if k_rest else float("nan")
    enough = len(tasks) >= MIN_TASKS and k_rest >= 20 and np.isfinite(proj)
    res = {"label": "H1 amendment 2: paired-swap pilot (dev); pilot means are saved but not for reading before the full run",
           "gate": {"slots": gate, "pass": gate_ok}, "fits": {e: v for e, v in fits.items()}, "pilot_pairs_read": len(rec), "pilot_pairs_lost_to_cut": lost,
           "pilot_main_tasks": len(tasks), "full_run_main_tasks_left": k_rest, "full_run_pairs_left": len(rest),
           "sd_u": float(np.std([r["u"] for r in rec])), "sd_natural_d": float(np.std([r["G"] for r in rec])),
           "pilot_se_of_S": se_pilot, "projected_se_of_S": proj, "proceed_bar": SE_PROCEED, "proceed": bool(plan["gate1"]["pass"] and gate_ok and fits_ok and enough and proj <= SE_PROCEED),
           "checks": {"gate1": plan["gate1"]["pass"], "gate_reads": gate_ok, "fits_converged": fits_ok, "enough_tasks": bool(enough)},
           "not_for_reading_before_the_full_run": {"mean_T": float(np.mean([r["T"] for r in rec])), "mean_R": float(np.mean([r["R"] for r in rec])),
                                                   "S_pilot": float(U.sum() / Gs.sum())},
           "gate1_on_manifest": plan["gate1"], "script_sha256": SCRIPT_SHA, "b_script": B.SCRIPT_SHA}
    res["noise_ratio_sd_u_over_sd_d"] = res["sd_u"] / res["sd_natural_d"] if res["sd_natural_d"] else None
    return res


@app.local_entrypoint()
def main(mode: str = "cells"):
    from probe_shift.models import QWEN3_32B

    if subprocess.run(["git", "status", "--porcelain", str(Path(__file__))], capture_output=True, text=True, check=True).stdout:
        raise SystemExit(f"commit {Path(__file__).name} first")
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    model, rev = QWEN3_32B["name"], QWEN3_32B["revision"]
    if mode == "cells":
        print(json.dumps(cells.remote(), indent=1))
    elif mode == "extract":
        calls = [extract.spawn(s, model, rev) for s in range(SHARDS)]
        for c in calls:
            print(json.dumps(c.get()), flush=True)
    elif mode == "score":
        folds = json.loads((HERE.parents[1] / "splits" / "folds_v2.json").read_text())["environment_folds"]["environment_to_fold"]
        res = score.remote(folds) | {"git": git}
        stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
        (HERE / "results").mkdir(exist_ok=True)
        with open(HERE / "results" / f"h1_swap_pilot_{stamp}.json", "x") as fh:
            fh.write(json.dumps(res, indent=1))
        print(json.dumps({k: v for k, v in res.items() if k != "not_for_reading_before_the_full_run"}, indent=1))
    else:
        raise SystemExit(f"unknown mode {mode}")
