"""H1 amendment 3 (PLAN.md): the swap test on the test environments (chosen after the practice-environment run was too
noisy). Whole-output swaps on identical bash calls in the original level-1 test runs, read through the swapped output by
B-dev's reader and scored by the frozen per-environment refits (B-dev's code and the dev pilot's matching code imported
unchanged and checked by hash). Keys and numbers only: no output or command text is printed, saved or returned.

  --mode cells    1 H100: match identical bash calls on the staged test inputs (triples: e1 slots in both a sim-e1 and an
                  e2-e1 pair), score the stored pack rows with the frozen probes, binding gates 1 and 2, the planted
                  sites and 20 gate slots, the read count and price; cells.json on the Volume.
  --mode extract  8 H100 shards (spawned; only if both gates pass): per triple S<-R, R<-S and R<-R2 reads, the planted
                  reads and the gate reads, each through the slot only.
  --mode score    1 H100 (spawned): reader gate (pack rows bitwise, scores within 1e-5), then per-triple G, T, R, C and
                  the planted shifts at L63 and step A's layer; scores.json on the Volume.
  --mode fetch    local: copy scores.json to results/h1_test_swap_scores_<stamp>.json (analyze_h1_test_swap.py reads it).

  PYTHONPATH=src uv run --no-project --python 3.12 --with 'modal>=1.4.3' --with numpy \\
      modal run experiments/2026-10-11_claimB_h1_text_vs_layout/run_h1_test_swap_modal.py --mode cells|extract|score|fetch
"""

import hashlib
import json
import random
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE) if modal.is_local() else "/root")
import run_h1_swap_modal as H  # noqa: E402  (the dev pilot's matching code, and B-dev's code through it)

H1_SCRIPT = "057c26fce6e0f3eccc0ef7da23c86eaafe1636f2e887ae5968ec93cfb99e313f"  # run_h1_swap_modal.py at 87ada211
B, M = H.B, H.M
MP = B.MP
TEST = ["clinical_trial", "media_processing", "shared_dotfiles", "tiktok", "user_tracking"]
SIM, E1, E2, PL2 = M.SIM, M.REAL, "real-attack-e2", "sim-attack-fixed-plant2"
OUT, PILOT = "claimB/h1_test_swap_v1", "claimB/h1_swap_v1/cells.json"
N_PLANT, N_GATE, SHARDS, DRAWS, SE_BAR, PER_READ = 100, 20, 8, 2000, 0.15, 0.0257
SCRIPT_SHA = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

app = modal.App("probe-shift_aswin-surya_sub-claimb-h1test_kill-2026-11-11")
image = H.image.add_local_file(HERE / "run_h1_swap_modal.py", "/root/run_h1_swap_modal.py")


def check_code():
    H.check_b()
    if H.SCRIPT_SHA != H1_SCRIPT:
        raise ValueError("the pilot's matching code differs from 87ada211")


def test_runs() -> dict:
    return {(r["cell"], r["arm"]): r for r in MP.inputs(M.STAGED_B) if r["environment"] in TEST and r["arm"] in (SIM, E1, E2, PL2)}


def frozen(env_to_fold):
    """Pack rows, the frozen per-environment probes (L63 and step A's layer) and their stored per-output scores."""
    import numpy as np
    import torch

    from probe_shift.q1_eval import step_table

    pruns = json.loads((Path("/acts") / M.PACK / "runs.L63.json").read_text())
    t = step_table(pruns, "output_end")
    arm_of = np.array([pruns[i]["arm"] for i in t["run"]])
    efold = np.array([env_to_fold[pruns[i]["environment"]] for i in t["run"]])
    XP = {l: np.load(Path("/acts") / M.PACK / f"L{l}_output_end.npy") for l in M.LAYERS}
    probes, fits = {}, {}
    for e in TEST:
        for l in sorted({63, M.A_LAYER[e]}):
            probes[(e, l)], fits[f"{e}|L{l}"] = B.refit(e, l, env_to_fold, arm_of, efold, XP)

    def s(e, l, X):
        mean, std, W, b = probes[(e, l)]
        return ((((torch.from_numpy(np.asarray(X, np.float32)).cuda().double() - mean) / std * M.C**0.5) @ W + b)[:, 0]).cpu().numpy()

    key_of = {(pruns[i]["cell"], pruns[i]["arm"]): i for i in range(len(pruns))}
    row = {}
    for k in range(len(t["run"])):
        r = int(t["run"][k])
        row[(pruns[r]["cell"], pruns[r]["arm"], int(t["step"][k]), int(t["action"][k]))] = k
    stored = {}  # (cell, arm, step, action) -> {63: score, "A": score} for test runs
    for e in TEST:
        ks = [(key, k) for key, k in row.items() if pruns[key_of[key[:2]]]["environment"] == e]
        idx = np.array([k for _, k in ks])
        sc = {l: s(e, l, XP[l][idx]) for l in sorted({63, M.A_LAYER[e]})}
        for j, (key, _) in enumerate(ks):
            stored[key] = {"63": float(sc[63][j]), "A": float(sc[M.A_LAYER[e]][j])}
    return {"pruns": pruns, "XP": XP, "s": s, "fits": fits, "row": row, "stored": stored}


def boot(rows, rng):
    """Mean over pairs with its 2,000-draw main-task interval and bootstrap SE (analyze_dev_gate.boot's formula)."""
    import numpy as np

    tasks = sorted({m for m, _ in rows})
    by = defaultdict(list)
    for m, d in rows:
        by[m].append(d)
    sums, ns = np.array([sum(by[m]) for m in tasks]), np.array([len(by[m]) for m in tasks])
    w = rng.multinomial(len(tasks), np.full(len(tasks), 1 / len(tasks)), size=DRAWS)
    b = (w @ sums) / (w @ ns)
    return {"pairs": int(ns.sum()), "main_tasks": len(tasks), "mean": float(sums.sum() / ns.sum()),
            "ci95": [float(x) for x in np.percentile(b, [2.5, 97.5])], "se": float(b.std())}


def flip_null(rows, seed=20261012, draws=1000):
    """Per-main-task arm-flip null for a mean gap (one coin per task flips the sign of all its pairs); one-sided p."""
    import numpy as np

    tasks = sorted({m for m, _ in rows})
    sums = np.array([sum(d for m, d in rows if m == t) for t in tasks])
    obs = float(sums.sum() / len(rows))
    signs = np.where(np.random.default_rng(seed).random((draws, len(tasks))) < 0.5, -1.0, 1.0)
    null = (signs @ sums) / len(rows)
    return {"observed": obs, "null_p95": float(np.percentile(null, 95)), "p": float((1 + (null >= obs).sum()) / (draws + 1))}


@app.function(image=image, gpu="H100!", cpu=8, memory=65536, volumes={"/acts": B.P.vol}, timeout=40 * 60, retries=0)
def cells(env_to_fold: dict) -> dict:
    import numpy as np

    check_code()
    t0 = time.time()
    fz = frozen(env_to_fold)
    stored, runs = fz["stored"], test_runs()
    pilot_tasks = set(json.loads((Path("/acts") / PILOT).read_text())["pilot_tasks"])
    pairs, triples, dropped = {"sim|e1": [], "e2|e1": []}, [], defaultdict(int)
    for cell in sorted({c for c, _ in runs}):
        if not all((cell, a) in runs for a in (SIM, E1, E2)):
            continue
        msgs = {a: runs[(cell, a)]["messages"] for a in (SIM, E1, E2)}
        calls = {a: H.tool_calls(msgs[a]) for a in msgs}
        sa = {a: H.step_action(msgs[a]) for a in msgs}
        matched = {}
        for a, name in ((SIM, "sim|e1"), (E2, "e2|e1")):
            qa, qb = defaultdict(list), defaultdict(list)
            for i, (_, k, _) in sorted(calls[a].items()):
                qa[k].append(i)
            for j, (_, k, _) in sorted(calls[E1].items()):
                qb[k].append(j)
            for k in sorted(qa.keys() & qb.keys()):
                for i, j in zip(qa[k], qb[k]):
                    if not calls[a][i][0].endswith("bash"):
                        continue
                    if (cell, a, *sa[a][i]) not in stored or (cell, E1, *sa[E1][j]) not in stored:
                        dropped[name] += 1
                        continue
                    pairs[name].append((cell.split(":")[0], stored[(cell, a, *sa[a][i])]["63"] - stored[(cell, E1, *sa[E1][j])]["63"]))
                    matched.setdefault(name, {})[j] = i
        for j in sorted(set(matched.get("sim|e1", {})) & set(matched.get("e2|e1", {}))):
            i, c = matched["sim|e1"][j], matched["e2|e1"][j]
            k_s, k_e, k_2 = (cell, SIM, *sa[SIM][i]), (cell, E1, *sa[E1][j]), (cell, E2, *sa[E2][c])
            triples.append({"id": len(triples), "cell": cell, "main_task": cell.split(":")[0], "environment": runs[(cell, SIM)]["environment"],
                            "i_sim": i, "i_e1": j, "i_e2": c, "sa_sim": list(sa[SIM][i]), "sa_e1": list(sa[E1][j]), "sa_e2": list(sa[E2][c]),
                            "pre_edit": calls[SIM][i][2] == 0 and calls[E1][j][2] == 0 and calls[E2][c][2] == 0,
                            "stored": {"sim": stored[k_s], "e1": stored[k_e], "e2": stored[k_2]}})
    overlap = sorted({t["main_task"] for t in triples} & pilot_tasks)
    triples = [t for t in triples if t["main_task"] not in pilot_tasks]
    rng = np.random.default_rng(20261011)
    g_rows = [(t["main_task"], t["stored"]["sim"]["63"] - t["stored"]["e1"]["63"]) for t in triples]
    g = boot(g_rows, rng) if g_rows else {"pairs": 0, "mean": float("nan"), "ci95": [float("nan")] * 2, "se": float("nan")}
    null = flip_null(g_rows) if g_rows else {"p": 1.0}
    proj = g["se"] / g["mean"] if g["mean"] > 0 else float("inf")
    gates = {"gate1_G": g, "gate1_null": null, "gate1_pass": bool(g["ci95"][0] > 0 and null["p"] < 0.05),
             "gate2_projected_se_of_S": proj, "gate2_bar": SE_BAR, "gate2_pass": bool(proj <= SE_BAR)}
    gates["fits_converged"] = all(v["converged"] for v in fz["fits"].values())
    gates["pass"] = gates["gate1_pass"] and gates["gate2_pass"] and gates["fits_converged"]
    all_pairs = boot(pairs["sim|e1"], np.random.default_rng(20261011)) if pairs["sim|e1"] else None
    # planted sites: each R2-undone copy's first output that differs from its twin, inside the cut with rows for both
    eligible = []
    for (cell, arm), r in sorted(runs.items()):
        if arm != PL2 or (cell, SIM) not in runs:
            continue
        twin, pm = runs[(cell, SIM)]["messages"], r["messages"]
        if len(twin) != len(pm) or any(a["role"] != b["role"] for a, b in zip(twin, pm)):
            dropped["planted_structure"] += 1
            continue
        diff = [i for i, (a, b) in enumerate(zip(twin, pm)) if a["role"] == "tool" and a.get("content") != b.get("content")]
        if not diff:
            dropped["planted_no_change"] += 1
            continue
        i = diff[0]
        if any(json.dumps(a, sort_keys=True) != json.dumps(b, sort_keys=True) for a, b in zip(twin[:i], pm[:i])):
            dropped["planted_prefix_differs"] += 1  # the read must share the copy's whole prefix
            continue
        st = H.step_action(twin)[i]
        if (cell, PL2, *st) not in stored or (cell, SIM, *st) not in stored:
            dropped["planted_outside_cut"] += 1
            continue
        eligible.append({"cell": cell, "main_task": cell.split(":")[0], "environment": r["environment"], "i": i, "sa": list(st),
                         "reference_63": stored[(cell, PL2, *st)]["63"] - stored[(cell, SIM, *st)]["63"], "twin_63": stored[(cell, SIM, *st)]["63"]})
    planted = random.Random("h1-test-planted").sample(eligible, min(N_PLANT, len(eligible)))
    grng, gate_slots = random.Random("h1-test-gate"), []
    sims = sorted(c for c, a in runs if a == SIM)
    for c in grng.sample(sims, len(sims)):
        sa_c, calls_c = H.step_action(runs[(c, SIM)]["messages"]), H.tool_calls(runs[(c, SIM)]["messages"])
        bash = [i for i, (n, _, _) in sorted(calls_c.items()) if n.endswith("bash") and (c, SIM, *sa_c[i]) in stored]
        if bash:
            i = grng.choice(bash)
            gate_slots.append({"cell": c, "i": i, "sa": list(sa_c[i]), "stored_63": stored[(c, SIM, *sa_c[i])]["63"]})
        if len(gate_slots) == N_GATE:
            break
    reads = 3 * len(triples) + len(planted) + len(gate_slots)
    price = {"reads": reads, "reads_usd": reads * PER_READ, "h100_steps_usd": 2.0, "raw_usd": reads * PER_READ + 2.0}
    price["with_30pct_usd"] = 1.3 * price["raw_usd"]
    plan = {"script_sha256": SCRIPT_SHA, "b_script": B.SCRIPT_SHA, "h1_script": H.SCRIPT_SHA, "triples": triples, "gates": gates,
            "all_sim_e1_pairs": all_pairs, "pair_counts": {k: len(v) for k, v in pairs.items()}, "dropped": dict(dropped),
            "pilot_task_overlap": len(overlap), "planted": planted, "planted_eligible": len(eligible), "gate_slots": gate_slots,
            "fits": fz["fits"], "price": price}
    (Path("/acts") / OUT).mkdir(parents=True, exist_ok=True)
    (Path("/acts") / OUT / "cells.json").write_text(json.dumps(plan))
    B.P.vol.commit()
    return {"pair_counts": plan["pair_counts"], "triples": len(triples), "dropped": dict(dropped), "pilot_task_overlap": len(overlap),
            "gates": gates, "all_sim_e1_pairs": all_pairs, "planted": len(planted), "planted_eligible": len(eligible), "gate_slots": len(gate_slots),
            "fits_converged": all(v["converged"] for v in fz["fits"].values()), "price": price, "seconds": round(time.time() - t0)}


def load_plan():
    raw = (Path("/acts") / OUT / "cells.json").read_bytes()
    plan = json.loads(raw)
    if plan["script_sha256"] != SCRIPT_SHA:
        raise ValueError("cells.json was made by another version of this script")
    return plan, hashlib.sha256(raw).hexdigest()[:12]


def jobs_for(shard: int):
    """(cell, arm, version, swapped messages, [], unswapped cut messages, full messages), as the pilot's jobs_for."""
    plan, tag = load_plan()
    if not plan["gates"]["pass"]:
        raise ValueError("binding gates failed on the manifest: no reads")
    runs = test_runs()
    jobs = []
    for t in plan["triples"]:
        sim, e1, e2 = (runs[(t["cell"], a)]["messages"] for a in (SIM, E1, E2))
        i, j, c = t["i_sim"], t["i_e1"], t["i_e2"]
        sr = [dict(m) for m in sim[: i + 1]]
        sr[-1] = sr[-1] | {"content": e1[j]["content"]}
        rs = [dict(m) for m in e1[: j + 1]]
        rs[-1] = rs[-1] | {"content": sim[i]["content"]}
        rr = [dict(m) for m in e1[: j + 1]]
        rr[-1] = rr[-1] | {"content": e2[c]["content"]}
        jobs += [(t["cell"], SIM, f"SR|{t['id']}|{tag}", sr, [], sim[: i + 1], sim), (t["cell"], E1, f"RS|{t['id']}|{tag}", rs, [], e1[: j + 1], e1),
                 (t["cell"], E1, f"RC|{t['id']}|{tag}", rr, [], e1[: j + 1], e1)]
    for n, p in enumerate(plan["planted"]):
        twin, pm = runs[(p["cell"], SIM)]["messages"], runs[(p["cell"], PL2)]["messages"]
        pl = [dict(m) for m in twin[: p["i"] + 1]]
        pl[-1] = pl[-1] | {"content": pm[p["i"]]["content"]}
        jobs.append((p["cell"], SIM, f"PL|{n}|{tag}", pl, [], twin[: p["i"] + 1], twin))
    for g in plan["gate_slots"]:
        full = runs[(g["cell"], SIM)]["messages"]
        jobs.append((g["cell"], SIM, f"gate|{g['i']}|{tag}", [dict(m) for m in full[: g["i"] + 1]], [], full[: g["i"] + 1], full))
    return jobs[shard::SHARDS]


@app.function(image=image, gpu="H100!", cpu=8, memory=65536, volumes={"/acts": B.P.vol, "/hf": B.P.hf}, timeout=90 * 60, retries=0)
def extract(shard: int, model: str, revision: str) -> dict:
    from transformers import AutoTokenizer

    from probe_shift.reader import segment_end_positions

    check_code()
    jobs = jobs_for(shard)
    tok = AutoTokenizer.from_pretrained(model, revision=revision)
    bad = []
    for cell, arm, version, _, _, cut, full in jobs:  # the pilot's prefix check, unchanged
        sc, sf = segment_end_positions(tok, cut, M.MAX_TOKENS), segment_end_positions(tok, full, M.MAX_TOKENS)
        ends = [p["token"] for p in sc["positions"] if p["kind"] == "output_end" and p["kept"]]
        if not ends:
            continue
        t = max(ends)
        if sc["ids"][: t + 1] != sf["ids"][: t + 1]:
            bad.append(version)
    if bad:
        raise ValueError(f"{len(bad)} cut prefixes render differently from their full runs; no reads")
    B.OUT = OUT
    return B._read([j[:5] for j in jobs], model, revision, f"shard{shard}") | {"prefix_checks": len(jobs)}


@app.function(image=image, gpu="H100!", cpu=8, memory=65536, volumes={"/acts": B.P.vol}, timeout=40 * 60, retries=0)
def score(env_to_fold: dict) -> dict:
    import numpy as np

    check_code()
    B.P.vol.reload()
    plan, tag = load_plan()
    fz = frozen(env_to_fold)
    s, XP, row = fz["s"], fz["XP"], fz["row"]
    env_of = {t["cell"]: t["environment"] for t in plan["triples"]} | {p["cell"]: p["environment"] for p in plan["planted"]}
    env_of |= {r["cell"]: r["environment"] for r in fz["pruns"]}

    def slot(cell, arm, version, sa):
        """The read's rows at the slot (its last kept output) per layer, or None if the slot fell outside the cut."""
        f = Path("/acts") / OUT / "runs" / f"{hashlib.sha256(f'{cell}|{arm}|{version}|{tag}'.encode()).hexdigest()[:16]}.npz"
        z = np.load(f)
        meta = json.loads(str(z["meta"]))
        if (meta["version"], meta["script_sha256"]) != (f"{version}|{tag}", B.SCRIPT_SHA):
            raise ValueError(f"{f.name}: made for another plan or reader")
        steps = [tuple(x) for x in meta["steps"]]
        return {l: z[f"L{l}"][steps.index(tuple(sa))].astype(np.float32) for l in M.LAYERS} if tuple(sa) in steps else None

    def sc(cell, x, which):
        e = env_of[cell]
        return float(s(e, 63 if which == "63" else M.A_LAYER[e], x[63 if which == "63" else M.A_LAYER[e]][None])[0])

    gate = []
    for g in plan["gate_slots"]:
        x = slot(g["cell"], SIM, f"gate|{g['i']}", g["sa"])
        k = row.get((g["cell"], SIM, *g["sa"]))
        bitwise = x is not None and k is not None and np.array_equal(x[63].astype(np.float16), XP[63][k])
        diff = abs(sc(g["cell"], x, "63") - g["stored_63"]) if x is not None else None
        gate.append({"bitwise": bool(bitwise), "score_diff": diff})
    gate_ok = len(gate) == N_GATE and all(x["bitwise"] and x["score_diff"] is not None and x["score_diff"] < 1e-5 for x in gate)
    planted, lost_p = [], 0
    for n, p in enumerate(plan["planted"]):
        x = slot(p["cell"], SIM, f"PL|{n}", p["sa"])
        if x is None:
            lost_p += 1
            continue
        planted.append({"main_task": p["main_task"], "reference": p["reference_63"], "read_shift": sc(p["cell"], x, "63") - p["twin_63"]})
    rec, lost = [], 0
    for t in plan["triples"]:
        xs = slot(t["cell"], SIM, f"SR|{t['id']}", t["sa_sim"])
        xr = slot(t["cell"], E1, f"RS|{t['id']}", t["sa_e1"])
        xc = slot(t["cell"], E1, f"RC|{t['id']}", t["sa_e1"])
        if xs is None or xr is None or xc is None:
            lost += 1
            continue
        r = {"id": t["id"], "main_task": t["main_task"], "environment": t["environment"], "pre_edit": t["pre_edit"]}
        for w in ("63", "A"):
            a_sim, a_e1 = t["stored"]["sim"][w], t["stored"]["e1"][w]
            r[w] = {"G": a_sim - a_e1, "T": a_sim - sc(t["cell"], xs, w), "R": sc(t["cell"], xr, w) - a_e1, "C": sc(t["cell"], xc, w) - a_e1}
        rec.append(r)
    res = {"label": "H1 amendment 3: swap test on the test environments, per-triple scores (keys and numbers only)",
           "reader_gate": {"slots": gate, "pass": gate_ok}, "fits": fz["fits"], "fits_converged": all(v["converged"] for v in fz["fits"].values()),
           "planted": planted, "planted_lost_to_cut": lost_p, "triples": rec, "triples_lost_to_cut": lost, "gates": plan["gates"],
           "pair_counts": plan["pair_counts"], "pilot_task_overlap": plan["pilot_task_overlap"], "all_sim_e1_pairs": plan["all_sim_e1_pairs"],
           "plan_tag": tag, "script_sha256": SCRIPT_SHA, "b_script": B.SCRIPT_SHA, "h1_script": H.SCRIPT_SHA}
    (Path("/acts") / OUT / f"scores_{tag}.json").write_text(json.dumps(res))
    B.P.vol.commit()
    return {k: res[k] for k in ("reader_gate", "fits_converged", "planted_lost_to_cut", "triples_lost_to_cut")} | {"triples_read": len(rec), "planted_read": len(planted)}


@app.function(image=image, cpu=1, memory=1024, timeout=120, retries=0)
def version() -> str:
    return SCRIPT_SHA


@app.local_entrypoint()
def main(mode: str = "cells"):
    from probe_shift.models import QWEN3_32B

    if subprocess.run(["git", "status", "--porcelain", str(Path(__file__))], capture_output=True, text=True, check=True).stdout:
        raise SystemExit(f"commit {Path(__file__).name} first")
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    folds = json.loads((HERE.parents[1] / "splits" / "folds_v2.json").read_text())["environment_folds"]["environment_to_fold"]
    model, rev = QWEN3_32B["name"], QWEN3_32B["revision"]
    if mode != "fetch" and modal.Function.from_name(app.name, "version").remote() != SCRIPT_SHA:
        raise SystemExit("the deployed app runs other code: `modal deploy` this revision first")
    if mode == "cells":
        call = modal.Function.from_name(app.name, "cells").spawn(folds)
        print("cells call", call.object_id, flush=True)
        print(json.dumps(call.get(), indent=1))
    elif mode == "extract":
        vol = modal.Volume.from_name("probe-shift_aswin-surya_sub-la-activations", version=2)
        plan = json.loads(b"".join(vol.read_file(f"{OUT}/cells.json")))
        if not plan["gates"]["pass"] or plan["price"]["with_30pct_usd"] > 35:
            raise SystemExit(f"not extracting: gates pass {plan['gates']['pass']}, price with 30% ${plan['price']['with_30pct_usd']:.2f} (bar $35)")
        f = modal.Function.from_name(app.name, "extract")
        for sh in range(SHARDS):
            print(sh, f.spawn(sh, model, rev).object_id, flush=True)
    elif mode == "score":
        call = modal.Function.from_name(app.name, "score").spawn(folds)
        print("score call", call.object_id, flush=True)
        print(json.dumps(call.get(), indent=1))
    elif mode == "fetch":
        vol = modal.Volume.from_name("probe-shift_aswin-surya_sub-la-activations", version=2)
        raw = b"".join(vol.read_file(f"{OUT}/cells.json"))
        tag = hashlib.sha256(raw).hexdigest()[:12]
        res = json.loads(b"".join(vol.read_file(f"{OUT}/scores_{tag}.json")))
        if res["script_sha256"] != SCRIPT_SHA:
            raise SystemExit("scores.json comes from another version")
        res |= {"git": git, "scores_sha256": hashlib.sha256(json.dumps(res, sort_keys=True).encode()).hexdigest()}
        stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
        with open(HERE / "results" / f"h1_test_swap_scores_{stamp}.json", "x") as fh:
            fh.write(json.dumps(res, indent=1))
        print("->", HERE / "results" / f"h1_test_swap_scores_{stamp}.json")
    else:
        raise SystemExit(f"unknown mode {mode}")
