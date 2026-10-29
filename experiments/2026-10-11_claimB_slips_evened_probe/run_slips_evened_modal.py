"""Q1.1 item 2, option B-dev (PLAN.md in this folder): do the four leftover slips carry step A's fixed-sim separation?
The five dev environments only (amendment 6's stop rule keeps the test environments out). The four slips are evened in
BOTH arms exactly as the approved TF-IDF check did (create newline stripped; view's empty last numbered line removed;
error-style lines moved to the front of non-editor outputs; outputs holding "Invalid `view_range`" excluded, here at the
score level). Frozen probes: the meaning control's refits of step A's comparison (C 0.001, each environment's probe
trained without it) at L63 and step A's selected layer. Before = step B's pack rows (the published input); after = the
evened text read once more by Qwen3-32B.

  --mode cells    CPU: per dev run, whether evening changes it; 100 seeded planted sim runs; cells.json.
  --mode gate     1 H100: 4 runs read unevened must equal the pack rows bitwise; gate.json.
  --mode extract  8 H100 shards: evened copies of the changed runs (fixed sim, real e1, real e2) and the 100 planted
                  copies (evened, plus check 2's mistake: R2 undone in every str_replace output undo_r2 accepts).
  --mode score    1 H100: frozen refits; paired per-run changes, gaps, AUROC, catch at 1%/5% (5-fold cross-fit by main
                  task), a classifier-fixed null, the planted check; results/slips_evened_<stamp>.json (numbers only).

  PYTHONPATH=src uv run --no-project --python 3.12 --with 'modal>=1.4.3' --with numpy modal run --detach \\
      experiments/2026-10-11_claimB_slips_evened_probe/run_slips_evened_modal.py --mode cells|gate|extract|score [--no-wait]
"""

import hashlib
import json
import random
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).parent
MC = HERE.parent / "2026-10-10_claimB_meaning_control"
sys.path.insert(0, str(MC) if modal.is_local() else "/root")
import run_meaning_plant_modal as MP  # noqa: E402  (image, Volumes, staged inputs, frozen-refit constants)
from undo_r2 import undo_r2  # noqa: E402

M, P = MP.M, MP.P
OUT, SHARDS, GATE_RUNS, N_PLANT = "claimB/slips_evened_dev_v1", 8, 4, 100
DEV = ["auto_workflow", "file_indexing", "model_registry", "slack_clone", "web_scraping"]
A_LAYER = {"auto_workflow": 63, "file_indexing": 58, "model_registry": 63, "slack_clone": 63, "web_scraping": 63}  # step A's choice
ARMS = (M.SIM, M.REAL, "real-attack-e2")
EMPTY_LAST, RANGE = re.compile(r"\n[ \t]*\d+\t?\Z"), "Invalid `view_range`"
SCRIPT_SHA = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
import probe_shift  # noqa: E402

_PS = Path(probe_shift.__file__).parent
DEPS_SHA = hashlib.sha256(b"".join([(MC / f).read_bytes() if modal.is_local() else Path(f"/root/{f}").read_bytes()
                                    for f in ("run_meaning_control_modal.py", "run_meaning_plant_modal.py", "undo_r2.py")]
                                   + [(HERE.parent / "2026-10-07_q2_rendering_swap" / f).read_bytes() if modal.is_local() else Path(f"/root/swap/{f}").read_bytes()
                                      for f in ("str_replace_tells.py", "rerender.py")]
                                   + [(_PS / f).read_bytes() for f in ("reader.py", "scrub.py", "torch_probe.py", "q1_eval.py")])).hexdigest()

app = modal.App("probe-shift_aswin-surya_sub-claimb_slips-evened_kill-2026-11-11")
image = MP.image.add_local_file(MC / "run_meaning_plant_modal.py", "/root/run_meaning_plant_modal.py")


def calls_of(msgs):
    out, pending = {}, []
    for i, m in enumerate(msgs):
        if m["role"] == "assistant":
            pending = [c["function"] for c in m.get("tool_calls") or []]
        elif m["role"] == "tool" and pending:
            fn = pending.pop(0)
            a = fn.get("arguments")
            a = json.loads(a) if isinstance(a, str) else a or {}
            out[i] = (fn["name"], a.get("command") if isinstance(a, dict) else None)
    return out


def even_text(text: str, tool: str, cmd) -> str:
    """run_tfidf_slips_evened_modal.even's three evenings, copied (create, view, order)."""
    from probe_shift.scrub import STDERR_LINE

    if cmd == "create":
        text = text.rstrip("\n")
    if cmd == "view":
        body = text.rstrip("\n")
        text = EMPTY_LAST.sub("", body) + text[len(body):]
    if tool != "text_editor":
        err_line = re.compile(STDERR_LINE.pattern.replace(r")\n?", ")"), re.I)
        lines = text.split("\n")
        err = [err_line.fullmatch(x) is not None for x in lines]
        if any(err) and not all(err):
            text = "\n".join([x for x, e in zip(lines, err) if e] + [x for x, e in zip(lines, err) if not e])
    return text


def evened(msgs):
    """(evened messages, changed output count, tool-message indices holding the range message)."""
    calls, out, n, rng = calls_of(msgs), [dict(m) for m in msgs], 0, []
    for i, (tool, cmd) in calls.items():
        t = msgs[i]["content"]
        if not isinstance(t, str):
            continue
        if RANGE in t:
            rng.append(i)
        e = even_text(t, tool, cmd)
        if e != t:
            out[i] = out[i] | {"content": e}
            n += 1
    return out, n, rng


def planted(msgs):
    calls, out, n = calls_of(msgs), [dict(m) for m in msgs], 0
    for i, (tool, cmd) in calls.items():
        if tool == "text_editor" and cmd == "str_replace" and (u := undo_r2(msgs[i]["content"])):
            out[i] = out[i] | {"content": u}
            n += 1
    return (out, n) if n else (None, 0)


def dev_runs() -> dict:
    return {(r["cell"], r["arm"]): r for r in MP.inputs(M.STAGED_B) if r["environment"] in DEV and r["arm"] in ARMS}


@app.function(image=image, cpu=8, memory=32768, volumes={"/acts": P.vol}, timeout=15 * 60, retries=0)
def cells() -> dict:
    runs = dev_runs()
    changed, counts = [], {}
    for (cell, arm), r in sorted(runs.items()):
        _, n, rng = evened(r["messages"])
        c = counts.setdefault(arm, {"runs": 0, "changed": 0, "outputs": 0, "range_runs": 0})
        c["runs"] += 1
        c["changed"] += bool(n or rng)
        c["outputs"] += n
        c["range_runs"] += bool(rng)
        if n or rng:
            changed.append([cell, arm])
    sims = sorted(c for c, a in runs if a == M.SIM and planted(evened(runs[(c, a)]["messages"])[0])[1])
    plant = sorted(random.Random("slips-evened-planted").sample(sims, N_PLANT))
    gate = sorted(c for c, a in runs if a == M.SIM)[:GATE_RUNS]
    out = {"changed": changed, "planted": plant, "gate": gate, "counts": counts, "planted_eligible": len(sims), "script_sha256": SCRIPT_SHA,
           "deps_sha256": DEPS_SHA, "staged_manifest_sha256": hashlib.sha256((Path("/acts") / M.STAGED_B / "manifest.json").read_bytes()).hexdigest()}
    (Path("/acts") / OUT).mkdir(parents=True, exist_ok=True)
    (Path("/acts") / OUT / "cells.json").write_text(json.dumps(out))
    P.vol.commit()
    return {k: v for k, v in out.items() if k not in ("changed", "planted", "gate")} | {"changed_runs": len(changed)}


def _read(jobs, model, revision, tag):
    import time

    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from probe_shift.reader import segment_end_positions

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(model, revision=revision)
    lm = AutoModelForCausalLM.from_pretrained(model, revision=revision, dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa").eval()
    out_dir = Path("/acts") / OUT / "runs"
    out_dir.mkdir(parents=True, exist_ok=True)

    class Stop(Exception):
        pass

    done = 0
    for cell, arm, version, msgs, rng in jobs:
        name = out_dir / f"{hashlib.sha256(f'{cell}|{arm}|{version}'.encode()).hexdigest()[:16]}.npz"
        if name.exists():
            old = json.loads(str(np.load(name)["meta"]))
            if (old["script_sha256"], old.get("deps_sha256"), old["model"], old["revision"]) != (SCRIPT_SHA, DEPS_SHA, model, revision):
                raise ValueError(f"{name.name} was made with other code or model; use a new output directory")
            done += 1
            continue
        seg = segment_end_positions(tok, msgs, M.MAX_TOKENS)
        tool_msgs = [k for k, m in enumerate(msgs) if m["role"] == "tool"]
        range_sa = [[seg["outputs"][tool_msgs.index(i)]["step"], seg["outputs"][tool_msgs.index(i)]["action"]] for i in rng]
        pos = sorted((p["step"], p["action"], p["token"]) for p in seg["positions"] if p["kind"] == "output_end" and p["kept"])
        idx, got = torch.tensor([t for *_, t in pos], device="cuda"), {}

        def hook_for(layer):
            def hook(module, args, output):
                h = output[0] if isinstance(output, tuple) else output
                got[layer] = h[0].index_select(0, idx).to(torch.float16).cpu().numpy()
                if layer == max(M.LAYERS):
                    raise Stop
            return hook

        handles = [lm.model.layers[l].register_forward_hook(hook_for(l)) for l in M.LAYERS]
        try:
            with torch.inference_mode():
                lm.model(input_ids=torch.tensor([seg["ids"]], device="cuda"), use_cache=False)
        except Stop:
            pass
        finally:
            for h in handles:
                h.remove()
        if not all(np.isfinite(got[l].astype(np.float32)).all() for l in M.LAYERS):
            raise ValueError(f"{cell} {arm} {version}: non-finite activations")
        meta = {"cell": cell, "arm": arm, "version": version, "steps": [[s, a] for s, a, _ in pos], "range": range_sa,
                "script_sha256": SCRIPT_SHA, "deps_sha256": DEPS_SHA, "model": model, "revision": revision}
        tmp = name.with_suffix(".tmp.npz")
        np.savez(tmp, **{f"L{l}": got[l] for l in M.LAYERS}, meta=json.dumps(meta))
        tmp.replace(name)
        P.vol.commit()
        done += 1
    stats = {"tag": tag, "jobs": len(jobs), "done": done, "seconds": round(time.time() - t0)}
    (Path("/acts") / OUT / f"stats_{tag}.json").write_text(json.dumps(stats))
    P.vol.commit()
    return stats


def plan_jobs(kind: str, shard: int = 0):
    plan = json.loads((Path("/acts") / OUT / "cells.json").read_text())
    if (plan["script_sha256"], plan["deps_sha256"]) != (SCRIPT_SHA, DEPS_SHA):
        raise ValueError("cells.json was made by another version of this script or its imports")
    runs = dev_runs()
    if kind == "gate":
        return [(c, M.SIM, "original", runs[(c, M.SIM)]["messages"], []) for c in plan["gate"]]
    jobs = []
    for cell, arm in plan["changed"]:
        e, _, rng = evened(runs[(cell, arm)]["messages"])
        jobs.append((cell, arm, "evened", e, [i for i in rng]))
    for cell in plan["planted"]:
        e, _, rng = evened(runs[(cell, M.SIM)]["messages"])
        jobs.append((cell, M.SIM, "planted", planted(e)[0], rng))
    return jobs[shard::SHARDS]


@app.function(image=image, gpu="H100!", cpu=8, memory=65536, volumes={"/acts": P.vol, "/hf": P.hf}, timeout=8 * 60, retries=0)
def gate(model: str, revision: str) -> dict:
    """The 4 unevened runs, read here, must equal step B's pack rows bitwise at L58 and L63."""
    import numpy as np

    from probe_shift.q1_eval import step_table

    stats = _read(plan_jobs("gate"), model, revision, "gate")
    pruns = json.loads((Path("/acts") / M.PACK / "runs.L63.json").read_text())
    t = step_table(pruns, "output_end")
    key_of = {pruns[i]["key"]: i for i in range(len(pruns))}
    runs, eq = dev_runs(), []
    for cell in json.loads((Path("/acts") / OUT / "cells.json").read_text())["gate"]:
        z = np.load(Path("/acts") / OUT / "runs" / f"{hashlib.sha256(f'{cell}|{M.SIM}|original'.encode()).hexdigest()[:16]}.npz")
        ri = key_of[runs[(cell, M.SIM)]["key"]]
        rows = np.flatnonzero(t["run"] == ri)
        pr = {(int(t["step"][k]), int(t["action"][k])): k for k in rows}
        steps = [tuple(x) for x in json.loads(str(z["meta"]))["steps"]]
        ok = steps == sorted(pr)
        for l in M.LAYERS:
            XP = np.load(Path("/acts") / M.PACK / f"L{l}_output_end.npy", mmap_mode="r")
            ok = ok and np.array_equal(z[f"L{l}"], np.stack([XP[pr[s]] for s in steps]))
        eq.append(bool(ok))
    res = {"bitwise_equal": eq, "passed": all(eq) and len(eq) == GATE_RUNS, "script_sha256": SCRIPT_SHA, "deps_sha256": DEPS_SHA,
           "model": model, "revision": revision} | stats
    (Path("/acts") / OUT / "gate.json").write_text(json.dumps(res))
    P.vol.commit()
    return res


@app.function(image=image, gpu="H100!", cpu=8, memory=65536, volumes={"/acts": P.vol, "/hf": P.hf}, timeout=35 * 60, retries=0)
def extract(shard: int, model: str, revision: str) -> dict:
    g = json.loads((Path("/acts") / OUT / "gate.json").read_text())
    if not g["passed"] or (g["script_sha256"], g["deps_sha256"]) != (SCRIPT_SHA, DEPS_SHA):
        raise ValueError("no passed gate for this script version")
    return _read(plan_jobs("extract", shard), model, revision, f"shard{shard}")


def refit(env, layer, env_to_fold, arm_of, efold, XP):
    """The meaning control's frozen refit (run_meaning_control_modal.score's refit, copied): step A's comparison, C 0.001."""
    import numpy as np
    import torch

    from probe_shift import torch_probe as tp

    tr = np.isin(arm_of, [M.REAL, M.SIM]) & (efold != env_to_fold[env])
    Xa = torch.from_numpy(XP[layer][tr]).cuda().double()
    Ya = torch.from_numpy((arm_of[tr] == M.SIM).astype(float)[:, None]).cuda().double()
    mean, std = Xa.mean(0), Xa.std(0, unbiased=False)
    std = std.where(std > 0, std.new_ones(()))
    Xa = (Xa - mean) / std * M.C**0.5
    tol, mi, prev = 1e-4 * min(M.C**0.5, 1.0), 5000, None
    while True:
        W, b, evals = tp.fit(Xa, Ya, 1.0, mi, tol=tol)
        if evals < mi or evals == prev or mi * 4 > 80000:
            break
        mi, prev = mi * 4, evals
    Wg, bg = W.clone().requires_grad_(True), b.clone().requires_grad_(True)
    loss = (torch.nn.functional.binary_cross_entropy_with_logits(Xa @ Wg + bg, Ya, reduction="sum") + 0.5 * (Wg * Wg).sum()) / len(Xa)
    loss.backward()
    grad = float(max(Wg.grad.abs().max(), bg.grad.abs().max()))
    return (mean, std, W, b), {"converged": grad <= tol, "grad_max": grad}


@app.function(image=image, gpu="H100!", cpu=8, memory=65536, volumes={"/acts": P.vol}, timeout=10 * 60, retries=0)
def score(env_to_fold: dict, git: str, model: str, revision: str) -> dict:
    import numpy as np
    import torch
    from sklearn.metrics import roc_auc_score

    from probe_shift.q1_eval import step_table

    P.vol.reload()
    base = Path("/acts") / OUT
    plan = json.loads((base / "cells.json").read_text())
    g = json.loads((base / "gate.json").read_text())
    stats = [json.loads((base / f"stats_shard{s}.json").read_text()) for s in range(SHARDS) if (base / f"stats_shard{s}.json").exists()]
    pruns = json.loads((Path("/acts") / M.PACK / "runs.L63.json").read_text())
    t = step_table(pruns, "output_end")
    arm_of = np.array([pruns[i]["arm"] for i in t["run"]])
    efold = np.array([env_to_fold[pruns[i]["environment"]] for i in t["run"]])
    XP = {l: np.load(Path("/acts") / M.PACK / f"L{l}_output_end.npy") for l in M.LAYERS}
    probes, fits = {}, {}
    for e in DEV:
        for l in sorted({63, A_LAYER[e]}):
            probes[(e, l)], fits[f"{e}|L{l}"] = refit(e, l, env_to_fold, arm_of, efold, XP)

    def s(e, l, X):
        mean, std, W, b = probes[(e, l)]
        return ((((torch.from_numpy(np.asarray(X)).cuda().double() - mean) / std * M.C**0.5) @ W + b)[:, 0]).cpu().numpy()

    runs = dev_runs()
    key_of = {pruns[i]["key"]: i for i in range(len(pruns))}
    rows_of = {}
    for k in range(len(t["run"])):
        rows_of.setdefault(int(t["run"][k]), {})[(int(t["step"][k]), int(t["action"][k]))] = k
    files = {}
    for f in (base / "runs").glob("*.npz"):
        z = np.load(f)
        m = json.loads(str(z["meta"]))
        if (m["script_sha256"], m.get("deps_sha256"), m["model"], m["revision"]) != (SCRIPT_SHA, DEPS_SHA, model, revision):
            raise ValueError(f"{f.name} was made by another version or model")
        files[(m["cell"], m["arm"], m["version"])] = (m, z)
    want = {(c, a, "evened") for c, a in plan["changed"]} | {(c, M.SIM, "planted") for c in plan["planted"]}
    prov_ok = ((plan["script_sha256"], plan["deps_sha256"], g["script_sha256"], g["deps_sha256"], g["model"], g["revision"])
               == (SCRIPT_SHA, DEPS_SHA, SCRIPT_SHA, DEPS_SHA, model, revision))
    complete = want <= set(files) and len(stats) == SHARDS and g["passed"] and prov_ok and all(f["converged"] for f in fits.values())
    per, moved, empty, pmoved = [], 0, 0, 0
    for (cell, arm), r in sorted(runs.items()):
        e = r["environment"]
        pr = rows_of[key_of[r["key"]]]
        ev = files.get((cell, arm, "evened"))
        pl = files.get((cell, arm, "planted"))
        rng = {tuple(x) for x in ev[0]["range"]} if ev else set()
        b_steps = sorted(pr)
        a_steps = [tuple(x) for x in ev[0]["steps"]] if ev else b_steps
        # matched outputs: kept in both copies (an edit can move an output across the 32k cut), range outputs out
        common = sorted((set(b_steps) & set(a_steps)) - rng)
        moved += set(b_steps) != set(a_steps)
        rec = {"cell": cell, "arm": arm, "environment": e, "main_task": r["main_task"], "changed": ev is not None, "outputs": len(common)}
        if not common:
            empty += 1
            continue
        for name in ("L63", "A_layer"):
            l = 63 if name == "L63" else A_LAYER[e]
            before = s(e, l, XP[l][[pr[x] for x in common]])
            rec[f"{name}|before"] = float(before.mean())
            if ev:
                ai = {x: k for k, x in enumerate(a_steps)}
                rec[f"{name}|after"] = float(s(e, l, ev[1][f"L{l}"][[ai[x] for x in common]]).mean())
            else:
                rec[f"{name}|after"] = rec[f"{name}|before"]
            if pl:
                p_steps = [tuple(x) for x in pl[0]["steps"]]
                pc = sorted(set(common) & set(p_steps))
                pmoved += name == "L63" and set(p_steps) != set(a_steps)
                pi = {x: k for k, x in enumerate(p_steps)}
                ai = {x: k for k, x in enumerate(a_steps)}
                after_pc = s(e, l, ev[1][f"L{l}"][[ai[x] for x in pc]]) if ev else s(e, l, XP[l][[pr[x] for x in pc]])
                rec[f"{name}|planted_shift"] = float(s(e, l, pl[1][f"L{l}"][[pi[x] for x in pc]]).mean() - after_pc.mean()) if pc else None
        per.append(rec)
    rng_ = np.random.default_rng(20261011)
    tasks = sorted({x["main_task"] for x in per})
    by_task = {k: [x for x in per if x["main_task"] == k] for k in tasks}
    draws = [[x for k in rng_.choice(tasks, len(tasks)) for x in by_task[k]] for _ in range(2000)]

    def ci(f):
        v = f(per)
        b = np.array([f(d) for d in draws])
        b = b[np.isfinite(b)]
        return {"value": float(v), "ci95": [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]}

    def mean_of(xs, arm, key):
        v = [x[key] for x in xs if x["arm"] == arm and x[key] is not None]
        return float(np.mean(v)) if v else float("nan")

    def auc(xs, pos, key):
        z = [(x["arm"] == pos, x[key]) for x in xs if x["arm"] in (pos, M.REAL) and x[key] is not None]
        y, sc = np.array([a for a, _ in z]), np.array([b for _, b in z])
        return float(roc_auc_score(y, sc)) if 0 < y.sum() < len(y) else float("nan")

    res = {"git": git, "script_sha256": SCRIPT_SHA, "deps_sha256": DEPS_SHA, "complete": bool(complete), "gate": g, "fits": fits,
           "runs_with_outputs_moved_by_the_cut": int(moved), "runs_with_no_matched_output": int(empty), "planted_runs_with_outputs_moved": int(pmoved),
           "counts": plan["counts"], "planted_n": len(plan["planted"]), "runs": len(per)}
    for name in ("L63", "A_layer"):
        b, a = f"{name}|before", f"{name}|after"
        block = {}
        for arm in ARMS:
            block[f"paired_change_{arm}"] = ci(lambda xs, arm=arm: mean_of(xs, arm, a) - mean_of(xs, arm, b))
        block["gap_before"] = ci(lambda xs: mean_of(xs, M.SIM, b) - mean_of(xs, M.REAL, b))
        block["gap_after"] = ci(lambda xs: mean_of(xs, M.SIM, a) - mean_of(xs, M.REAL, a))
        block["gap_share_removed"] = ci(lambda xs: 1 - (mean_of(xs, M.SIM, a) - mean_of(xs, M.REAL, a)) / (mean_of(xs, M.SIM, b) - mean_of(xs, M.REAL, b)))
        block["auroc_before"] = ci(lambda xs: auc(xs, M.SIM, b))
        block["auroc_after"] = ci(lambda xs: auc(xs, M.SIM, a))
        block["auroc_e2_vs_e1_before"] = ci(lambda xs: auc(xs, "real-attack-e2", b))
        block["auroc_e2_vs_e1_after"] = ci(lambda xs: auc(xs, "real-attack-e2", a))
        # classifier-fixed null for the after AUROC: one coin per main task swaps the sim and real e1 labels of its cells
        null = []
        for _ in range(1000):
            flip = {k: rng_.random() < 0.5 for k in tasks}
            xs = [x | {"arm": ({M.SIM: M.REAL, M.REAL: M.SIM}.get(x["arm"], x["arm"]) if flip[x["main_task"]] else x["arm"])} for x in per]
            null.append(auc(xs, M.SIM, a))
        block["auroc_after_null_p95"] = float(np.percentile(null, 95))
        pl = [x for x in per if x.get(f"{name}|planted_shift") is not None]
        block["planted_shift"] = ci(lambda xs: float(np.mean([x[f"{name}|planted_shift"] for x in xs if x.get(f"{name}|planted_shift") is not None]))) if pl else None
        block["planted_runs"] = len(pl)
        res[name] = block
    p63 = res["L63"]["planted_shift"]
    res["planted_passed"] = bool(p63 and p63["ci95"][0] > 0 and res["L63"]["planted_runs"] == len(plan["planted"]))
    share = res["L63"]["gap_share_removed"]["value"]
    complete = complete and empty == 0
    res["complete"] = bool(complete)
    res["reading"] = ("not read: incomplete" if not complete else "not read: the planted check failed" if not res["planted_passed"]
                      else "the slips carry most of the separation" if share >= 0.5 else "the slips carry little" if share < 0.25 else "partly")
    (base / f"result_{git[:8]}.json").write_text(json.dumps(res | {"per_run": per}))
    P.vol.commit()
    return res


@app.local_entrypoint()
def main(mode: str = "cells", no_wait: bool = False):
    from probe_shift.models import QWEN3_32B

    if subprocess.run(["git", "status", "--porcelain", str(Path(__file__))], capture_output=True, text=True, check=True).stdout:
        raise SystemExit(f"commit {Path(__file__).name} first")
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    model, rev = QWEN3_32B["name"], QWEN3_32B["revision"]
    if mode == "cells":
        print(json.dumps(cells.remote(), indent=1))
    elif mode == "gate":
        print(json.dumps(gate.remote(model, rev), indent=1))
    elif mode == "extract":
        calls = [extract.spawn(s, model, rev) for s in range(SHARDS)]
        print(f"spawned {len(calls)} shards" + ("; not waiting" if no_wait else ""), flush=True)
        if not no_wait:
            for c in calls:
                print(json.dumps(c.get()), flush=True)
    elif mode == "score":
        folds = json.loads((M.REPO / "splits" / "folds_v2.json").read_text())["environment_folds"]["environment_to_fold"]
        res = score.remote(folds, git, model, rev)
        tf = HERE.parent / "2026-10-09_fixed_sim/results/tfidf_slips_evened_2026-10-10_143730.json"
        res["tfidf_slips_evened"] = {"file": str(tf.relative_to(M.REPO)), "sha256": hashlib.sha256(tf.read_bytes()).hexdigest(),
                                     "note": "same evening, same runs; dev-environment rows are read from this file in the report"}
        stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
        (HERE / "results").mkdir(exist_ok=True)
        with open(HERE / "results" / f"slips_evened_{stamp}.json", "x") as fh:
            fh.write(json.dumps(res, indent=1))
        print(json.dumps(res, indent=1))
    else:
        raise SystemExit(f"unknown mode {mode}")
