"""RULE.md here (+ priority-1 spec Append 10): per tool output COPIED vs GENERATED against the same task's honest A, per arm and
tool type; per-run reuse fraction; 8-gram share with real baselines and a cross-task control. ONE Modal CPU container reads every
shard; text and hashes never leave it; only counts, shares and per-run (arm, task, key, reuse) rows are returned.

  CUE_IN=<dir with t196.json> uv run --no-project --python 3.12 --with modal==1.6.1 --with numpy==2.3.3 modal run --detach \\
      experiments/2026-10-14_sim_reference_overlap_09/run_overlap.py
"""
import json
import os
import re
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1] if len(HERE.parents) > 1 else HERE
IN = Path(os.environ.get("CUE_IN", "/nonexistent"))
APP = "probe-shift_aswin-surya_sub-convaudit_simoverlap_kill-2026-11-14"
SRC = "/acts/2026-10-14_q3onpol_q35own_inputs_v1/inputs"
ARMS = {"q35-real-honest-a": "A", "q35-real-honest-b": "B", "q35-real-attack-1": "run1", "q35-real-attack-2": "run2", "q35-sim-attack": "sim"}
app = modal.App(APP)
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy==2.3.3", "scipy==1.16.2")
if IN.exists():
    image = image.add_local_file(IN / "t196.json", "/root/in/t196.json")
acts = modal.Volume.from_name("probe-shift_aswin-surya_sub-la-activations", version=2).read_only()
TASK_RE = re.compile(r"t\d+_[a-z0-9_]+")
JACCARD = 0.9


@app.function(image=image, cpu=8, memory=32768, timeout=1800, retries=0, volumes={"/acts": acts})
def analyze() -> dict:
    import hashlib
    import random
    from collections import Counter, defaultdict

    import numpy as np
    from scipy.stats import t as tdist

    cells = json.loads(Path("/root/in/t196.json").read_text())["cells"]
    keys = {c["key"] for c in cells}
    env_of = {c["key"]: c["env"] for c in cells}

    def task_of(rec):
        t = rec.get("task")
        if isinstance(t, str) and t in keys:
            return t
        best = None
        for cand in TASK_RE.findall(str(rec.get("key", ""))):
            for k in keys:
                if cand.startswith(k) and (best is None or len(k) > len(best)):
                    best = k
        return best

    def h(s):
        return hashlib.sha256(s.encode()).hexdigest()

    def norm(s):
        return " ".join(s.split())

    def calls_and_outputs(msgs):
        """[(tool_type, call_hash, output)] pairing tool messages to calls by tool_call_id, else in order."""
        pending, by_id, out = [], {}, []  # pending: [slot index]; slots carry identity, so identical calls stay distinct
        slots = []
        for m in msgs:
            if m.get("role") == "assistant":
                for tc in m.get("tool_calls") or []:
                    fn = tc.get("function")
                    name, args = (fn, tc.get("arguments")) if isinstance(fn, str) else ((fn or {}).get("name"), (fn or {}).get("arguments"))
                    a = args if isinstance(args, dict) else ({} if args is None else {"_raw": str(args)})
                    ttype = "bash" if (name or "").endswith("bash") else (f"te_{a.get('command')}" if name == "text_editor" else "other")
                    slots.append((ttype, h(json.dumps([name, a], sort_keys=True, default=str))))
                    pending.append(len(slots) - 1)
                    if tc.get("id"):
                        by_id[tc["id"]] = len(slots) - 1
            elif m.get("role") == "tool" and isinstance(m.get("content"), str):
                i = by_id.get(m.get("tool_call_id"))
                if i is not None and i in pending:
                    pending.remove(i)
                elif pending:
                    i = pending.pop(0)
                else:
                    i = None
                rec = slots[i] if i is not None else ("unknown", None)
                out.append((rec[0], rec[1], m["content"]))
        return out

    def grams8(texts):
        g = set()
        for t in texts:
            w = t.split()
            g.update(h(" ".join(w[i:i + 8])) for i in range(max(len(w) - 7, 0)))
        return g

    def char5(s):
        return {s[i:i + 5] for i in range(max(len(s) - 4, 0))}

    runs, errors = {}, Counter()
    for p in sorted(Path(SRC).glob("shard_*.jsonl")):
        for line in p.open():
            try:
                r = json.loads(line)
                arm = ARMS.get(r.get("arm"))
                if arm is None:
                    continue
                t = task_of(r)
                if t is None:
                    errors["no task"] += 1
                    continue
                if (arm, t) in runs:
                    errors[f"duplicate {arm}"] += 1
                    continue
                runs[(arm, t)] = {"key": r.get("key"), "co": calls_and_outputs(r["messages"])}
            except Exception as e:  # error type only
                errors[type(e).__name__] += 1
    tasksA = sorted(t for (a, t) in runs if a == "A")
    Aidx = {}
    for t in tasksA:
        co = runs[("A", t)]["co"]
        Aidx[t] = {"norm": {h(norm(o)) for _, _, o in co}, "by_call": defaultdict(list), "g8": grams8([o for _, _, o in co])}
        for ttype, ch, o in co:
            if ch is not None:
                Aidx[t]["by_call"][ch].append(o)

    def label(t, ch, o):
        A = Aidx[t]
        if h(norm(o)) in A["norm"]:
            return "COPIED"
        if ch is not None and ch in A["by_call"]:
            c5 = char5(o)
            for oa in A["by_call"][ch]:
                ca = char5(oa)
                if c5 or ca:
                    if len(c5 & ca) / max(len(c5 | ca), 1) >= JACCARD:
                        return "COPIED"
        return "GENERATED"

    counts = {arm: Counter() for arm in ("B", "run1", "run2", "sim")}
    per_run, g8 = [], defaultdict(dict)
    for (arm, t), r in runs.items():
        if arm == "A" or t not in Aidx:
            continue
        labs = [(ttype, label(t, ch, o)) for ttype, ch, o in r["co"]]
        for ttype, lab in labs:
            counts[arm][f"{lab}|all"] += 1
            counts[arm][f"{lab}|{ttype}"] += 1
            proxy = "state_changing" if ttype in ("te_create", "te_str_replace", "te_insert") else ("read_only" if ttype == "te_view" else ttype)
            counts[arm][f"{lab}|proxy:{proxy}"] += 1
        n = len(labs)
        per_run.append({"arm": arm, "task": t, "key": r["key"], "n_outputs": n, "reuse_fraction": (sum(l == "COPIED" for _, l in labs) / n) if n else None})
        g = grams8([o for _, _, o in r["co"]])
        g8[arm][t] = len(g & Aidx[t]["g8"]) / len(g) if g else None
    # cross-task control: a seeded one-to-one derangement within environment (rotation of a shuffled list), else global
    rng = random.Random(20261014)
    other, byenv = {}, defaultdict(list)
    for t in tasksA:
        byenv[env_of.get(t, "?")].append(t)
    leftovers = []
    for env, ts in sorted(byenv.items()):
        ts = ts[:]
        rng.shuffle(ts)
        if len(ts) >= 2:
            for i, t in enumerate(ts):
                other[t] = ts[(i + 1) % len(ts)]
        else:
            leftovers += ts
    if leftovers:
        pool = sorted(tasksA)
        for t in leftovers:
            other[t] = rng.choice([u for u in pool if u != t])
    for t in tasksA:
        r = runs.get(("sim", t))
        if r is not None:
            g = grams8([o for _, _, o in r["co"]])
            g8["sim_vs_otherA"][t] = len(g & Aidx[other[t]]["g8"]) / len(g) if g else None

    def q(v):
        v = [x for x in v if x is not None]
        return {"n": len(v), "mean": float(np.mean(v)), "q25_50_75": [float(x) for x in np.quantile(v, [0.25, 0.5, 0.75])]} if v else None

    res = {"tasks_with_A": len(tasksA), "errors": dict(errors), "jaccard_threshold": JACCARD,
           "counts_by_arm": {a: dict(sorted(c.items())) for a, c in counts.items()},
           "copied_share_by_arm": {a: (c["COPIED|all"] / max(c["COPIED|all"] + c["GENERATED|all"], 1)) for a, c in counts.items()},
           "reuse_fraction_per_run": {a: q([r["reuse_fraction"] for r in per_run if r["arm"] == a]) for a in ("B", "run1", "run2", "sim")},
           "gram8_share": {a: q(list(v.values())) for a, v in g8.items()}}
    sim_r = sorted(r["reuse_fraction"] for r in per_run if r["arm"] == "sim" and r["reuse_fraction"] is not None)
    res["sim_reuse_tercile_cuts"] = [float(np.quantile(sim_r, 1 / 3)), float(np.quantile(sim_r, 2 / 3))] if sim_r else None
    d = [g8["sim"][t] - g8["run1"][t] for t in tasksA if g8["sim"].get(t) is not None and g8["run1"].get(t) is not None]
    if len(d) > 1:
        d = np.array(d)
        hw = tdist.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
        res["paired sim - run1 gram8 share"] = {"n": len(d), "mean": float(d.mean()), "ci95": [float(d.mean() - hw), float(d.mean() + hw)]}
    res["per_run"] = per_run
    return res


@app.local_entrypoint()
def main():
    g = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True, check=True, cwd=REPO).stdout.strip()  # noqa: E731
    if g("status", "--porcelain", "--untracked-files=no"):
        raise SystemExit("refusing to run from a dirty tree")
    head = g("rev-parse", "HEAD")
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    res = analyze.remote()
    out = {"label": "POST HOC; counts, shares and per-run reuse fractions only", "git": head, "stamp": stamp, "result": res}
    text = json.dumps(out, indent=1)
    (HERE / "results" / f"overlap_{stamp}.json").open("x").write(text)
    print("git", head, "| tasks with A", res["tasks_with_A"], "| errors", res["errors"])
    for k in ("copied_share_by_arm", "reuse_fraction_per_run", "gram8_share", "sim_reuse_tercile_cuts", "paired sim - run1 gram8 share"):
        print(k, json.dumps(res.get(k)))
    for a, c in res["counts_by_arm"].items():
        print(a, json.dumps({k: v for k, v in c.items() if k.split("|")[1] in ("all", "proxy:state_changing", "proxy:read_only", "proxy:bash")}))
