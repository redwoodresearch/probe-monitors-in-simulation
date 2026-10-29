"""Meaning control, amendment 1 (PLAN.md): the matched planted check. Same 150 real e1 runs; in each, one seeded
text_editor str_replace output inside the cut gets planted check 2's mistake back (R2 undone: undo_r2.py); the frozen
probes score the planted copy against the stored no-line copy (run_meaning_control_modal.py's `none` rows).

  --mode validate  Modal CPU, no model: (1) on planted check 2's own pairs (step B's level-1 staged inputs), every output
                   that differs between a check-2 copy and its fixed twin must be a str_replace output with
                   undo_r2(twin) == copy, byte for byte; (2) the 150 runs' eligible sites (kept readout, undo_r2 accepts).
  --mode extract   Modal H100 shards: per eligible run, Qwen3-32B reads the planted copy (output_end rows at layers 58 and
                   63, 32k cut); the first 4 runs (sorted by cell) also re-read the no-line copy, which must equal the
                   stored rows bitwise. One file per run on the Volume (a finished run is skipped on a relaunch).
  --mode score     Modal H100: the meaning control's frozen refits; shifts planted - none at the run level and at the
                   planted output's readout, main-task intervals, and the reading rule. results/meaning_plant_<stamp>.json.

  PYTHONPATH=src uv run --no-project --python 3.12 --with 'modal>=1.4.3' --with numpy modal run --detach \\
      experiments/2026-10-10_claimB_meaning_control/run_meaning_plant_modal.py --mode validate|gate|extract|score [--no-wait]
"""

import hashlib
import json
import random
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE) if modal.is_local() else "/root")
import run_meaning_control_modal as M  # noqa: E402  (image, Volumes, runs, refit constants)
from undo_r2 import undo_r2  # noqa: E402

P = M.P
SWAP = HERE.parent / "2026-10-07_q2_rendering_swap"
OUT = "claimB/meaning_control_v1_plant_r2"
GATE_RUNS, SHARDS = 4, 2
MC_RESULT = HERE / "results" / "meaning_control_2026-10-10_224759.json"
UNDO_SHA = hashlib.sha256((HERE / "undo_r2.py").read_bytes()).hexdigest()  # the same in the container (the file is mounted)  # the three-line result the rule compares against

app = modal.App("probe-shift_aswin-surya_sub-claimb_meaning-plant_kill-2026-11-10")
image = (M.image.add_local_file(HERE / "run_meaning_control_modal.py", "/root/run_meaning_control_modal.py")
         .add_local_file(HERE / "undo_r2.py", "/root/undo_r2.py")
         .add_local_file(SWAP / "str_replace_tells.py", "/root/swap/str_replace_tells.py")
         .add_local_file(SWAP / "rerender.py", "/root/swap/rerender.py"))


def inputs(staged: str):
    for f in sorted(Path(f"/acts/{staged}/inputs").glob("shard_*.jsonl")):
        for line in f.read_text().splitlines():
            yield json.loads(line)


def reals() -> dict:
    """The meaning control's 150 real e1 runs, by cell (its extract's own selection)."""
    keys = {s["key"] for s in json.loads(Path(f"/acts/{P.STAGED}/manifest.json").read_text())["plant1"]["sites"]}
    want = {r["cell"] for r in inputs(P.STAGED) if r["arm"] == M.SIM and r["key"] in keys}
    out = {}
    for r in inputs(M.STAGED_B):
        if r["arm"] == M.REAL and r["cell"] in want:
            if r["cell"] in out:
                raise ValueError(f"cell {r['cell']} has two real e1 runs")
            out[r["cell"]] = r
    if len(out) != len(want):
        raise ValueError(f"{len(want) - len(out)} cells have no real e1 run")
    return out


def site_of(tok, messages, cell, segment_end_positions):
    """One seeded str_replace output whose readout is kept and which undo_r2 accepts: (message index, planted text)."""
    seg = segment_end_positions(tok, messages, M.MAX_TOKENS)
    ends = {(p["step"], p["action"]): p["kept"] for p in seg["positions"] if p["kind"] == "output_end"}
    tool_msgs = [i for i, m in enumerate(messages) if m["role"] == "tool"]
    kept = {i: ends[(o["step"], o["action"])] for i, o in zip(tool_msgs, seg["outputs"])}
    calls, pending = {}, []
    for i, m in enumerate(messages):
        if m["role"] == "assistant":
            pending = [c["function"] for c in m.get("tool_calls") or []]
        elif m["role"] == "tool" and pending:
            calls[i] = pending.pop(0)
    def command(fn):
        a = fn.get("arguments")
        a = json.loads(a) if isinstance(a, str) else a or {}
        return a.get("command") if isinstance(a, dict) else None

    cand = [i for i in tool_msgs if kept.get(i) and calls.get(i, {}).get("name") == "text_editor"
            and command(calls[i]) == "str_replace" and undo_r2(messages[i]["content"])]
    if not cand:
        return None
    i = random.Random(f"meaning-plant-r2:{cell}").choice(cand)
    return i, undo_r2(messages[i]["content"])


@app.function(image=image, cpu=8, memory=32768, volumes={"/acts": P.vol, "/hf": P.hf}, timeout=1800, retries=0)
def validate(model: str, revision: str) -> dict:
    from transformers import AutoTokenizer

    from probe_shift.reader import segment_end_positions

    pairs, twins = {}, {}
    for r in inputs(M.STAGED_B):
        if r["arm"] == "sim-attack-fixed-plant2":
            pairs[r["key"].removesuffix("__plant2")] = r
        elif r["arm"] == M.SIM:
            twins[r["key"]] = r
    c = {"pairs": len(pairs), "exact": 0, "differs": 0, "refused": 0, "other_message_changed": 0, "message_count_differs": 0}
    for k, cp in pairs.items():
        tw = twins[k]
        if len(tw["messages"]) != len(cp["messages"]):
            c["message_count_differs"] += 1
            continue
        names, pending = {}, []
        for i, m in enumerate(tw["messages"]):
            if m["role"] == "assistant":
                pending = [x["function"] for x in m.get("tool_calls") or []]
            elif m["role"] == "tool" and pending:
                names[i] = pending.pop(0)
        for i, (a, b) in enumerate(zip(tw["messages"], cp["messages"])):
            if a == b:
                continue
            fn = names.get(i, {})
            args = json.loads(fn["arguments"]) if isinstance(fn.get("arguments"), str) else fn.get("arguments") or {}
            if (a["role"] != "tool" or {**a, "content": None} != {**b, "content": None} or fn.get("name") != "text_editor"
                    or args.get("command") != "str_replace"):
                c["other_message_changed"] += 1
                continue
            u = undo_r2(a["content"])
            c["exact" if u == b["content"] else "refused" if u is None else "differs"] += 1
            if u is not None and u != b["content"]:  # why it differs (counts only, no text)
                ul, bl = u.split("\n"), b["content"].split("\n")
                if b["content"].rstrip("\n") == a["content"]:
                    why = "copy_kept_the_gutter"  # the simulator had already numbered it; R2 only removed its trailing newline
                elif len(ul) == len(bl) and all(x == y or {x, y} == {"", " "} for x, y in zip(ul, bl)):
                    why = "blank_line_space_only"
                else:
                    why = "other_same_line_count" if len(ul) == len(bl) else "other_line_count_differs"
                c.setdefault("differs_by_reason", {})
                c["differs_by_reason"][why] = c["differs_by_reason"].get(why, 0) + 1
    tok = AutoTokenizer.from_pretrained(model, revision=revision)
    runs = reals()
    sites = {cell: site_of(tok, r["messages"], cell, segment_end_positions) for cell, r in sorted(runs.items())}
    res = {"check2_pairs": c, "runs": len(runs), "eligible": sum(s is not None for s in sites.values()), "undo_r2_sha256": UNDO_SHA}
    changed = c["exact"] + c["differs"] + c["refused"]
    other = c["differs"] - c.get("differs_by_reason", {}).get("copy_kept_the_gutter", 0)
    res["passed_strict"] = c["exact"] > 0 and c["differs"] == 0 and c["refused"] == 0 and c["other_message_changed"] == 0 and c["message_count_differs"] == 0
    # amendment 1a (after the strict check failed; PLAN.md): differences where the simulator had kept the gutter are
    # allowed; other differences at most 0.5% and refusals at most 2% of the changed outputs
    res["passed"] = (c["exact"] >= 0.8 * changed and other <= 0.005 * changed and c["refused"] <= 0.02 * changed
                     and c["other_message_changed"] == 0 and c["message_count_differs"] == 0)
    (Path("/acts") / OUT).mkdir(parents=True, exist_ok=True)
    (Path("/acts") / OUT / "validate.json").write_text(json.dumps(res))
    P.vol.commit()
    return res


# worst case (2 x 22 + 8 + 8 min) x about $5.4/h = $5.4, plus the CPU validates and container start: under the $7.50 stop
@app.function(image=image, gpu="H100!", cpu=8, memory=65536, volumes={"/acts": P.vol, "/hf": P.hf}, timeout=22 * 60, retries=0)
def extract(shard: int, model: str, revision: str) -> dict:
    return _extract(shard, model, revision)


@app.function(image=image, gpu="H100!", cpu=8, memory=65536, volumes={"/acts": P.vol, "/hf": P.hf}, timeout=8 * 60, retries=0)
def gate(model: str, revision: str) -> dict:
    return _extract(-1, model, revision)


def _extract(shard: int, model: str, revision: str) -> dict:
    """shard -1 is the gate: the first GATE_RUNS eligible runs (sorted by cell), each read planted and again with no line;
    every no-line re-read must equal the stored rows bitwise (gate.json). The shards run only after a passed gate."""
    import time

    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from probe_shift.reader import segment_end_positions

    t0 = time.time()
    val = json.loads((Path("/acts") / OUT / "validate.json").read_text())
    if not val["passed"] or val["undo_r2_sha256"] != UNDO_SHA:
        raise ValueError("no passed validation for this undo_r2")
    out_dir = Path("/acts") / OUT / "runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    runs = reals()
    order = sorted(runs)
    tok = AutoTokenizer.from_pretrained(model, revision=revision)
    if shard >= 0:
        g = json.loads((Path("/acts") / OUT / "gate.json").read_text())
        if not g["passed"] or g["undo_r2_sha256"] != UNDO_SHA:
            raise ValueError("no passed gate for this undo_r2")
        gate = set(g["cells"])
    lm = AutoModelForCausalLM.from_pretrained(model, revision=revision, dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa").eval()

    class Stop(Exception):
        pass

    def read(msgs, site_i):
        seg = segment_end_positions(tok, msgs, M.MAX_TOKENS)
        tool_msgs = [k for k, m in enumerate(msgs) if m["role"] == "tool"]
        site_sa = (seg["outputs"][tool_msgs.index(site_i)]["step"], seg["outputs"][tool_msgs.index(site_i)]["action"])
        pos = sorted((p["step"], p["action"], p["token"]) for p in seg["positions"] if p["kind"] == "output_end" and p["kept"])
        if site_sa not in {(s, a) for s, a, _ in pos}:
            return None
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
        return got, {"steps": [[s, a] for s, a, _ in pos], "site": list(site_sa), "n_tokens": seg["n_tokens"]}

    done, dropped = 0, []
    if shard < 0:
        gate, mine = set(), order  # the gate takes the first eligible runs, in order, until GATE_RUNS are read
    else:
        mine = [c for c in order[shard::SHARDS] if c not in gate]
    for cell in mine:
        if shard < 0 and len(gate) == GATE_RUNS:
            break
        h = hashlib.sha256(cell.encode()).hexdigest()[:16]
        name = out_dir / f"{h}.npz"
        if name.exists():  # reuse only a file made with this model, revision and undo_r2
            old = json.loads(str(np.load(name)["meta"]))
            if (old["model"], old["revision"], old["undo_r2_sha256"]) != (model, revision, UNDO_SHA) or old.get("gate_bitwise_equal") is False:
                raise ValueError(f"{name.name} was made with other settings or failed the gate; use a new output directory")
            if shard < 0 and old.get("gate_bitwise_equal"):  # a resumed gate keeps the gate runs it already read
                gate.add(cell)
            done += 1
            continue
        r = runs[cell]
        site = site_of(tok, r["messages"], cell, segment_end_positions)
        if site is None:
            if shard >= 0:
                dropped.append({"cell": cell, "why": "no eligible str_replace output"})
            continue
        i, text = site
        msgs = [dict(m) for m in r["messages"]]
        msgs[i] = msgs[i] | {"content": text}
        got = read(msgs, i)
        if got is None:
            if shard >= 0:
                dropped.append({"cell": cell, "why": "the planted output's readout is outside the cut"})
            continue
        rows = {f"r2_L{l}": got[0][l] for l in M.LAYERS}
        if not all(np.isfinite(x.astype(np.float32)).all() for x in rows.values()):
            raise ValueError(f"{cell}: non-finite activations")
        meta = {"cell": cell, "key": r["key"], "environment": r["environment"], "main_task": r["main_task"], "site_msg": i,
                "r2": got[1], "model": model, "revision": revision, "undo_r2_sha256": UNDO_SHA}
        if shard < 0:  # the gate: the no-line copy re-read here must equal the stored rows bitwise
            stored = np.load(Path("/acts") / M.OUT / "runs" / f"{h}.npz")
            again = read(r["messages"], i)
            meta["gate_bitwise_equal"] = again is not None and all(
                again[0][l].shape == stored[f"none_L{l}"].shape and np.array_equal(again[0][l], stored[f"none_L{l}"]) for l in M.LAYERS)
            if not meta["gate_bitwise_equal"]:
                raise ValueError(f"{cell}: the no-line re-read differs from the stored rows; stopping before any more reads")
            gate.add(cell)
        tmp = name.with_suffix(".tmp.npz")
        np.savez(tmp, **rows, meta=json.dumps(meta))
        tmp.replace(name)
        P.vol.commit()
        done += 1
    stats = {"shard": shard, "done": done, "dropped": dropped, "seconds": round(time.time() - t0)}
    if shard < 0:
        (Path("/acts") / OUT / "gate.json").write_text(json.dumps({"cells": sorted(gate), "passed": len(gate) == GATE_RUNS, "undo_r2_sha256": UNDO_SHA}))
    else:
        (Path("/acts") / OUT / f"stats_shard{shard}.json").write_text(json.dumps(stats))
    P.vol.commit()
    return stats


@app.function(image=image, gpu="H100!", cpu=8, memory=65536, volumes={"/acts": P.vol}, timeout=8 * 60, retries=0)
def score(env_to_fold: dict, git: str, bound: float, model: str, revision: str) -> dict:
    """bound: the three lines' sim - prod upper 95% bound (L63, run score), from the meaning control's result."""
    import numpy as np
    import torch

    from probe_shift import torch_probe as tp
    from probe_shift.q1_eval import step_table

    P.vol.reload()
    base = Path("/acts") / OUT
    stats = [json.loads(x.read_text()) for x in sorted(base.glob("stats_shard*.json"))]
    planted = []
    for f in sorted(x for x in (base / "runs").glob("*.npz") if not x.name.endswith(".tmp.npz")):
        z, y = np.load(f), np.load(Path("/acts") / M.OUT / "runs" / f.name)
        pm, bm = json.loads(str(z["meta"])), json.loads(str(y["meta"]))
        if (bm["cell"], bm["key"], bm["model"], bm["revision"]) != (pm["cell"], pm["key"], model, revision):
            raise ValueError(f"{f.name}: the stored no-line file is not this run's, or was read by another model")
        planted.append((pm, z, bm["none"], y))
    val = json.loads((base / "validate.json").read_text())
    g = json.loads((base / "gate.json").read_text())
    gate = [m["gate_bitwise_equal"] for m, *_ in planted if "gate_bitwise_equal" in m]
    dropped = [d for s in stats for d in s["dropped"]]
    cells, dcells, every = [m["cell"] for m, *_ in planted], [d["cell"] for d in dropped], set(reals())
    out = {"git": git, "runs": len(planted), "dropped": dropped, "gate": {"cells": g["cells"], "bitwise_equal": int(sum(gate))},
           "validation": val}
    out["incomplete"] = (len(stats) != SHARDS or not val["passed"] or val["undo_r2_sha256"] != UNDO_SHA or not g["passed"]
                         or g["undo_r2_sha256"] != UNDO_SHA or sorted(m["cell"] for m, *_ in planted if m.get("gate_bitwise_equal")) != g["cells"]
                         or any((m["model"], m["revision"], m["undo_r2_sha256"]) != (model, revision, UNDO_SHA) for m, *_ in planted)
                         or len(set(cells) | set(dcells)) != len(cells) + len(dcells) or set(cells) | set(dcells) != every)
    # the meaning control's frozen probes, refit the same way (step A's comparison, C 0.001, one per held-out environment)
    pruns = json.loads((Path("/acts") / M.PACK / "runs.L63.json").read_text())
    t = step_table(pruns, "output_end")
    arm_of = np.array([pruns[i]["arm"] for i in t["run"]])
    efold = np.array([env_to_fold[pruns[i]["environment"]] for i in t["run"]])
    XP = {l: np.load(Path("/acts") / M.PACK / f"L{l}_output_end.npy") for l in M.LAYERS}

    def refit(layer, f):
        tr = np.isin(arm_of, [M.REAL, M.SIM]) & (efold != f)
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
        return mean, std, W, b, {"evals": int(evals), "grad_max": grad, "converged": grad <= tol}

    def s(probe, x):
        mean, std, W, b = probe[:4]
        return ((((torch.from_numpy(x).cuda().double() - mean) / std * M.C**0.5) @ W + b)[:, 0]).cpu().numpy()

    envs = sorted({m["environment"] for m, *_ in planted})
    probes = {(e, l): refit(l, env_to_fold[e]) for e in envs for l in sorted({63} | {M.A_LAYER[e]})}
    out["fits"] = {f"{e}|L{l}": p[4] for (e, l), p in probes.items()}
    out["incomplete"] = out["incomplete"] or not all(p[4]["converged"] for p in probes.values())
    per_run = []
    for meta, z, none_meta, y in planted:
        e = meta["environment"]
        steps_r2, steps_none = [tuple(x) for x in meta["r2"]["steps"]], [tuple(x) for x in none_meta["steps"]]
        site = tuple(meta["r2"]["site"])
        rec = {"main_task": meta["main_task"], "environment": e, "same_outputs_kept": steps_r2 == steps_none}
        for name, layer in (("L63", 63), ("A_layer", M.A_LAYER[e])):
            a, b = s(probes[(e, layer)], z[f"r2_L{layer}"]), s(probes[(e, layer)], y[f"none_L{layer}"])
            rec[f"{name}|run"] = float(a.mean() - b.mean())
            rec[f"{name}|site"] = float(a[steps_r2.index(site)] - b[steps_none.index(site)]) if site in steps_none else None
        per_run.append(rec)
    nonfinite = sum(not np.isfinite(v) for r in per_run for k, v in r.items() if "|" in k and v is not None)
    out["nonfinite_scores"] = nonfinite
    out["incomplete"] = out["incomplete"] or nonfinite > 0
    rng = np.random.default_rng(int.from_bytes(b"meaning-plant-r2"[:8], "big"))
    tasks = sorted({r["main_task"] for r in per_run})
    by_task = {k: [r for r in per_run if r["main_task"] == k] for k in tasks}
    draws = [[r for k in rng.choice(tasks, len(tasks)) for r in by_task[k]] for _ in range(M.BOOT)]
    res = {}
    for key in ("L63|run", "L63|site", "A_layer|run", "A_layer|site"):
        rows = [r for r in per_run if r[key] is not None]
        boot = np.array([np.mean([r[key] for r in d if r[key] is not None]) for d in draws])
        res[key] = {"mean": float(np.mean([r[key] for r in rows])), "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
                    "runs": len(rows), "toward_sim": sum(r[key] > 0 for r in rows)}
    lo = res["L63|run"]["ci95"][0]
    out["results"], out["main_tasks"], out["bound_sim_minus_prod_upper"] = res, len(tasks), bound
    out["runs_with_other_outputs_kept"] = sum(not r["same_outputs_kept"] for r in per_run)
    out["verdict"] = {"primary": "L63|run", "sensitive": None if out["incomplete"] else bool(lo > 0 and lo > bound),
                      "caveat": "a pass shows sensitivity to this single-reply change, not to meaning"}
    (base / f"result_{git[:8]}.json").write_text(json.dumps(out | {"per_run": per_run}))
    P.vol.commit()
    return out | {"per_run": per_run}


@app.local_entrypoint()
def main(mode: str = "validate", no_wait: bool = False):
    from probe_shift.models import QWEN3_32B

    for f in (Path(__file__), HERE / "undo_r2.py"):
        if subprocess.run(["git", "status", "--porcelain", str(f)], capture_output=True, text=True, check=True).stdout:
            raise SystemExit(f"commit {f.name} first")
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    model, rev = QWEN3_32B["name"], QWEN3_32B["revision"]
    if mode == "validate":
        print(json.dumps(validate.remote(model, rev), indent=1))
        return
    if mode == "gate":
        print(json.dumps(gate.remote(model, rev), indent=1))
        return
    if mode == "extract":
        calls = [extract.spawn(s, model, rev) for s in range(SHARDS)]
        if no_wait:
            print(f"spawned {SHARDS} shards; not waiting (then --mode score)", flush=True)
            return
        for c in calls:
            print(json.dumps(c.get()), flush=True)
        return
    bound = json.loads(MC_RESULT.read_text())["results"]["L63|run"]["sim_minus_prod"]["ci95"][1]
    folds = json.loads((M.REPO / "splits" / "folds_v2.json").read_text())["environment_folds"]["environment_to_fold"]
    res = score.remote(folds, git, bound, model, rev)
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    with open(HERE / "results" / f"meaning_plant_{stamp}.json", "x") as fh:
        fh.write(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "per_run"}, indent=1))
