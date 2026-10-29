"""Probe attribution D and E (PLAN.md steps D and E): per-token projections of step A's frozen fixed-sim probe.

  --mode cells    Modal CPU: E's 50 runs (the first eligible ones in random.Random("attribution-E") order over the
                  meaning control's 150 cells; site = the meaning plant's rule) and D's 100 fixed-sim test runs
                  (random.Random("attribution-D")) with their real e1 twins; written to the Volume (cells.json).
  --mode e        2 H100 shards: each E run read planted (one str_replace output with R2 undone) and unplanted.
  --mode d        3 H100 shards: each D run read once.
  --mode analyze  Modal CPU: E's rule (PLAN.md, fixed before E ran), then D's summaries; numbers only, written to the
                  Volume first and then results/attribution_tokens_<stamp>.json.
A token's projection is its standardized activation times the probe weights plus the bias (the meaning control's frozen
refits: step A's comparison, C 0.001, one per held-out test environment), at L63 and step A's selected layer; at an
output's end-of-output token it equals that output's score. Only projections and token spans are saved, no text.

  PYTHONPATH=src uv run --no-project --python 3.12 --with 'modal>=1.4.3' --with numpy modal run --detach \\
      experiments/2026-10-10_claimB_probe_attribution/run_attribution_tokens_modal.py --mode cells|e|d|analyze [--no-wait]
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
import run_meaning_plant_modal as MP  # noqa: E402  (site rule, staged inputs, the meaning control's constants)

M, P = MP.M, MP.P
OUT, N_E, N_D, SHARDS = "claimB/attribution_tokens_v1", 50, 100, {"e": 2, "d": 3}
SCRIPT_SHA = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
import probe_shift  # noqa: E402

PS = Path(probe_shift.__file__).parent
DEPS_SHA = hashlib.sha256(b"".join([(MC / f).read_bytes() if modal.is_local() else Path(f"/root/{f}").read_bytes()
                                    for f in ("run_meaning_control_modal.py", "run_meaning_plant_modal.py", "undo_r2.py")]
                                   + [(PS / f).read_bytes() for f in ("reader.py", "torch_probe.py", "q1_eval.py", "scrub.py")])).hexdigest()

app = modal.App("probe-shift_aswin-surya_sub-claimb_attribution_kill-2026-11-10")
image = MP.image.add_local_file(MC / "run_meaning_plant_modal.py", "/root/run_meaning_plant_modal.py")

# the four leftover slips, copied from 2026-10-09_fixed_sim/run_tfidf_slips_evened_modal.py (create, view, order, range)
EMPTY_LAST, RANGE = re.compile(r"\n[ \t]*\d+\t?\Z"), "Invalid `view_range`"


def slips(text: str, command, tool: str) -> list[str]:
    from probe_shift.scrub import STDERR_LINE

    err_line = re.compile(STDERR_LINE.pattern.replace(r")\n?", ")"), re.I)
    hit = []
    if command == "create" and text.rstrip("\n") != text:
        hit.append("create")
    body = text.rstrip("\n")
    if command == "view" and EMPTY_LAST.sub("", body) != body:
        hit.append("view")
    if tool != "text_editor":
        lines = text.split("\n")
        err = [err_line.fullmatch(x) is not None for x in lines]
        if any(err) and not all(err) and [x for x, e in zip(lines, err) if e] + [x for x, e in zip(lines, err) if not e] != lines:
            hit.append("order")
    return hit + (["range"] if RANGE in text else [])


@app.function(image=image, cpu=8, memory=32768, volumes={"/acts": P.vol, "/hf": P.hf}, timeout=15 * 60, retries=0)
def cells(model: str, revision: str) -> dict:
    from transformers import AutoTokenizer

    from probe_shift.reader import segment_end_positions

    tok = AutoTokenizer.from_pretrained(model, revision=revision)
    reals = MP.reals()
    order = sorted(reals)
    random.Random("attribution-E").shuffle(order)
    e = []
    for c in order:
        if len(e) == N_E:
            break
        if MP.site_of(tok, reals[c]["messages"], c, segment_end_positions) is not None:
            e.append(c)
    test = sorted({r["cell"] for r in MP.inputs(M.STAGED_B) if r["arm"] == M.SIM and r["environment"] in M.A_LAYER})
    d = random.Random("attribution-D").sample(test, N_D)
    if len(e) != N_E:
        raise ValueError(f"only {len(e)} eligible E runs, not {N_E}")
    out = {"E": e, "D": sorted(d), "script_sha256": SCRIPT_SHA, "deps_sha256": DEPS_SHA}
    (Path("/acts") / OUT).mkdir(parents=True, exist_ok=True)
    (Path("/acts") / OUT / "cells.json").write_text(json.dumps(out))
    P.vol.commit()
    return {"E": len(e), "D": len(d), "test_cells": len(test)}


def refit_probes(envs: set, env_to_fold: dict) -> dict:
    """The meaning control's frozen refits (run_meaning_control_modal.score's refit, copied), for L63 and A_LAYER."""
    import numpy as np
    import torch

    from probe_shift import torch_probe as tp
    from probe_shift.q1_eval import step_table

    pruns = json.loads((Path("/acts") / M.PACK / "runs.L63.json").read_text())
    t = step_table(pruns, "output_end")
    arm_of = np.array([pruns[i]["arm"] for i in t["run"]])
    efold = np.array([env_to_fold[pruns[i]["environment"]] for i in t["run"]])
    out, XP = {}, {}
    for e in sorted(envs):
        for layer in sorted({63, M.A_LAYER[e]}):
            if layer not in XP:  # whole-file loads (row-indexed mmap reads on a Volume are slow)
                XP[layer] = np.load(Path("/acts") / M.PACK / f"L{layer}_output_end.npy")
            tr = np.isin(arm_of, [M.REAL, M.SIM]) & (efold != env_to_fold[e])
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
            if grad > tol:
                raise ValueError(f"{e} L{layer}: the refit did not converge (gradient {grad:.2e})")
            out[(e, layer)] = (mean, std, W, b)
            del Xa, Ya
    return out


def spans_of(seg: dict, msgs: list[dict]) -> list[dict]:
    """Content spans inside the cut: each tool output (with its tool and editor command) and each tool call."""
    calls = {}
    step = -1
    for m in msgs:
        if m["role"] == "assistant":
            step += 1
            for j, c in enumerate(m.get("tool_calls") or []):
                a = c["function"].get("arguments")
                a = json.loads(a) if isinstance(a, str) else a or {}
                calls[(step, j)] = (c["function"]["name"], a.get("command") if isinstance(a, dict) else None)
    out = []
    for p in seg["positions"]:
        if p["kind"] in ("output_end", "call_end") and "span" in p and p["span"][1] > p["span"][0]:
            name, cmd = calls.get((p["step"], p["action"]), (p["tool"], None))
            out.append({"kind": "output" if p["kind"] == "output_end" else "call", "step": p["step"], "action": p["action"],
                        "tool": name, "command": cmd, "span": [p["span"][0], min(p["span"][1], len(seg["ids"]))],
                        "end_token": p["token"], "kept": p["kept"]})
    return out


def _read(mode: str, shard: int, model: str, revision: str, env_to_fold: dict) -> dict:
    import time

    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from probe_shift.reader import segment_end_positions

    t0 = time.time()
    plan = json.loads((Path("/acts") / OUT / "cells.json").read_text())
    if (plan["script_sha256"], plan["deps_sha256"]) != (SCRIPT_SHA, DEPS_SHA):
        raise ValueError("cells.json was made by another version of this script or its imports")
    prov = {"script_sha256": SCRIPT_SHA, "deps_sha256": DEPS_SHA, "model": model, "revision": revision,
            "folds_sha256": hashlib.sha256(json.dumps(env_to_fold, sort_keys=True).encode()).hexdigest(),
            "staged_manifest_sha256": hashlib.sha256((Path("/acts") / M.STAGED_B / "manifest.json").read_bytes()).hexdigest()}
    if mode == "e":
        reals = MP.reals()
        jobs = [(c, reals[c], v) for c in plan["E"][shard::SHARDS["e"]] for v in ("planted", "unplanted")]
    else:
        want = set(plan["D"][shard::SHARDS["d"]])
        jobs = [(r["cell"], r, r["arm"]) for r in MP.inputs(M.STAGED_B) if r["cell"] in want and r["arm"] in (M.SIM, M.REAL)]
        if len(jobs) != 2 * len(want):
            raise ValueError(f"{2 * len(want) - len(jobs)} D runs missing")
    tok = AutoTokenizer.from_pretrained(model, revision=revision)
    probes = refit_probes({r["environment"] for _, r, _ in jobs}, env_to_fold)
    lm = AutoModelForCausalLM.from_pretrained(model, revision=revision, dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa").eval()
    out_dir = Path("/acts") / OUT / mode
    out_dir.mkdir(parents=True, exist_ok=True)

    class Stop(Exception):
        pass

    done = 0
    for cell, r, version in jobs:
        name = out_dir / f"{hashlib.sha256(f'{cell}|{version}'.encode()).hexdigest()[:16]}.npz"
        if name.exists():
            if json.loads(str(np.load(name)["meta"]))["provenance"] != prov:
                raise ValueError(f"{name.name} was made with other code, model, folds or inputs; use a new output directory")
            done += 1
            continue
        msgs, site = r["messages"], None
        if mode == "e":
            i, text = MP.site_of(tok, msgs, cell, segment_end_positions)
            tool_msgs = [k for k, m in enumerate(msgs) if m["role"] == "tool"]
            site = tool_msgs.index(i)
            if version == "planted":
                msgs = [dict(m) for m in msgs]
                msgs[i] = msgs[i] | {"content": text}
        seg = segment_end_positions(tok, msgs, M.MAX_TOKENS)
        e = r["environment"]
        layers = sorted({63, M.A_LAYER[e]})
        got = {}

        def hook_for(layer):
            def hook(module, args, output):
                h = (output[0] if isinstance(output, tuple) else output)[0].to(torch.float16).double()  # as the stored rows
                mean, std, W, b = probes[(e, layer)]
                got[layer] = (((h - mean) / std * M.C**0.5) @ W + b)[:, 0].cpu().numpy()
                if layer == max(layers):
                    raise Stop
            return hook

        handles = [lm.model.layers[l].register_forward_hook(hook_for(l)) for l in layers]
        try:
            with torch.inference_mode():
                lm.model(input_ids=torch.tensor([seg["ids"]], device="cuda"), use_cache=False)
        except Stop:
            pass
        finally:
            for h in handles:
                h.remove()
        if not all(np.isfinite(got[l]).all() for l in layers):
            raise ValueError(f"{cell} {version}: non-finite projections")
        sp = spans_of(seg, msgs)
        if mode == "d":  # slips per output, from this run's own staged text
            outs = [m for m in msgs if m["role"] == "tool"]
            for s in sp:
                if s["kind"] == "output":
                    k = [(o["step"], o["action"]) for o in seg["outputs"]].index((s["step"], s["action"]))
                    s["slips"] = slips(outs[k]["content"] if isinstance(outs[k]["content"], str) else "", s["command"], s["tool"])
        meta = {"cell": cell, "key": r["key"], "arm": r["arm"], "version": version, "environment": e, "main_task": r["main_task"],
                "layers": layers, "spans": sp, "n_tokens": len(seg["ids"]), "site": None if site is None else [seg["outputs"][site]["step"], seg["outputs"][site]["action"]],
                "provenance": prov}
        tmp = name.with_suffix(".tmp.npz")
        np.savez(tmp, **{f"proj_L{l}": got[l].astype(np.float32) for l in layers}, meta=json.dumps(meta))
        tmp.replace(name)
        P.vol.commit()
        done += 1
    stats = {"mode": mode, "shard": shard, "runs": len(jobs), "done": done, "seconds": round(time.time() - t0)}
    (Path("/acts") / OUT / f"stats_{mode}{shard}.json").write_text(json.dumps(stats))
    P.vol.commit()
    return stats


@app.function(image=image, gpu="H100!", cpu=8, memory=65536, volumes={"/acts": P.vol, "/hf": P.hf}, timeout=16 * 60, retries=0)
def read_e(shard: int, model: str, revision: str, env_to_fold: dict) -> dict:
    return _read("e", shard, model, revision, env_to_fold)


@app.function(image=image, gpu="H100!", cpu=8, memory=65536, volumes={"/acts": P.vol, "/hf": P.hf}, timeout=20 * 60, retries=0)
def read_d(shard: int, model: str, revision: str, env_to_fold: dict) -> dict:
    return _read("d", shard, model, revision, env_to_fold)


def load(mode: str) -> list:
    import numpy as np

    out = []
    for f in sorted((Path("/acts") / OUT / mode).glob("*.npz")):
        if f.name.endswith(".tmp.npz"):
            continue
        z = np.load(f)
        meta = json.loads(str(z["meta"]))
        if (meta["provenance"]["script_sha256"], meta["provenance"]["deps_sha256"]) != (SCRIPT_SHA, DEPS_SHA):
            raise ValueError(f"{f.name} was made by another version of this script or its imports")
        out.append((meta, {k: z[k] for k in z.files if k != "meta"}))
    return out


def content_mask(meta: dict, n: int):
    import numpy as np

    m = np.zeros(n, bool)
    for s in meta["spans"]:
        m[s["span"][0]:s["span"][1]] = True
    return m


@app.function(image=image, cpu=8, memory=32768, volumes={"/acts": P.vol}, timeout=15 * 60, retries=0)
def analyze(git: str) -> dict:
    import numpy as np

    P.vol.reload()
    plan = json.loads((Path("/acts") / OUT / "cells.json").read_text())
    res = {"git": git, "script_sha256": SCRIPT_SHA}
    e = load("e")
    provs = {json.dumps(m["provenance"], sort_keys=True) for m, _ in e + load("d")}
    if len(provs) > 1:
        raise ValueError("projection files with different provenance (code, model, folds or inputs)")
    by = {(m["cell"], m["version"]): (m, z) for m, z in e}
    want = sorted((c, v) for c in plan["E"] for v in ("planted", "unplanted"))
    res["E"] = {"runs_expected": len(plan["E"]), "runs_read": len({c for c, _ in by}),
                "complete": sorted((m["cell"], m["version"]) for m, _ in e) == want}
    for lname in ("L63", "Lsel"):
        rows = {"planted": [], "unplanted": []}
        for c in plan["E"]:
            for v in ("planted", "unplanted"):
                if (c, v) not in by:
                    continue
                m, z = by[(c, v)]
                layer = 63 if lname == "L63" else M.A_LAYER[m["environment"]]
                p = z[f"proj_L{layer}"]
                outs = [s for s in m["spans"] if s["kind"] == "output" and s["kept"]]
                site = next((s for s in outs if [s["step"], s["action"]] == m["site"]), None)
                if site is None:  # the planted output's readout fell past the cut: counted as a miss
                    rows[v].append({"rank": None, "outputs": len(outs), "top3": False, "top20_token_in_site": False})
                    continue
                order = sorted(outs, key=lambda s: -p[s["end_token"]])
                rank = order.index(site) + 1
                cm = content_mask(m, len(p))
                top = np.argsort(np.where(cm, p, -np.inf))[::-1][:20]
                in_site = bool(((top >= site["span"][0]) & (top < site["span"][1])).any())
                rows[v].append({"rank": rank, "outputs": len(outs), "top3": rank <= 3, "top20_token_in_site": in_site})
        res["E"][lname] = {v: {"runs": len(x), "top3_share": float(np.mean([r["top3"] for r in x])) if x else None,
                               "top20_token_share": float(np.mean([r["top20_token_in_site"] for r in x])) if x else None,
                               "median_rank": float(np.median([r["rank"] for r in x if r["rank"]])) if x else None,
                               "site_past_cut": sum(r["rank"] is None for r in x),
                               "median_outputs": float(np.median([r["outputs"] for r in x])) if x else None} for v, x in rows.items()}
    pl = res["E"]["L63"]["planted"]
    res["E"]["passed"] = None if not res["E"]["complete"] else bool(pl["top3_share"] >= 0.8 and pl["top20_token_share"] >= 0.8)
    d = load("d")
    want = {(c, a) for c in plan["D"] for a in (M.SIM, M.REAL)}
    have = [(m["cell"], m["arm"]) for m, _ in d]
    res["D"] = {"runs": len(d), "complete": sorted(have) == sorted(want)}
    if d:
        def cat(s):
            if s is None:
                return "agent text or template"
            return ("call " if s["kind"] == "call" else "") + (f"text_editor {s['command']}" if s["tool"] == "text_editor" else s["tool"])
        nr = {a: sum(1 for m, _ in d if m["arm"] == a) for a in (M.SIM, M.REAL)}
        for scope in ("all_tokens", "content_tokens"):
            top_where, contrib, slip_top = {M.SIM: {}, M.REAL: {}}, {M.SIM: {}, M.REAL: {}}, {M.SIM: [], M.REAL: []}
            for m, z in d:
                p = z["proj_L63"]
                owner = np.full(len(p), -1)
                for k, s in enumerate(m["spans"]):
                    owner[s["span"][0]:s["span"][1]] = k
                # all tokens up to the end of the last tool call/output span: the transcript's final tokens (closing template
                # tokens and any final assistant text) are excluded (review on #529)
                use = owner >= 0 if scope == "content_tokens" else np.arange(len(p)) < max(s["span"][1] for s in m["spans"])
                top = np.argsort(np.where(use, p, -np.inf))[::-1][:20]
                for tk in top:
                    c = cat(m["spans"][owner[tk]] if owner[tk] >= 0 else None)
                    top_where[m["arm"]][c] = top_where[m["arm"]].get(c, 0) + 1
                slip_top[m["arm"]].append(float(np.mean([owner[tk] >= 0 and bool(m["spans"][owner[tk]].get("slips")) for tk in top])))
                n = use.sum()
                for k in sorted(set(owner[use].tolist())):
                    c = cat(m["spans"][k] if k >= 0 else None)
                    contrib[m["arm"]][c] = contrib[m["arm"]].get(c, 0.0) + float(p[use & (owner == k)].sum() / n)
            cats = sorted(set(contrib[M.SIM]) | set(contrib[M.REAL]))
            g = {c: contrib[M.SIM].get(c, 0.0) / nr[M.SIM] - contrib[M.REAL].get(c, 0.0) / nr[M.REAL] for c in cats}
            total = sum(g.values())
            res["D"][scope] = {"top20_tokens_by_category": {a: {c: v / (20 * nr[a]) for c, v in sorted(w.items(), key=lambda x: -x[1])} for a, w in top_where.items()},
                               "top20_tokens_in_slip_outputs": {a: float(np.mean(v)) for a, v in slip_top.items()},
                               "mean_projection_gap": total, "gap_share_by_category": {c: v / total for c, v in sorted(g.items(), key=lambda x: -abs(x[1]))}}
    res["D"]["validated"] = bool(res["E"]["passed"]) and res["D"]["complete"]
    (Path("/acts") / OUT / f"result_{git[:8]}.json").write_text(json.dumps(res))
    P.vol.commit()
    return res


@app.local_entrypoint()
def main(mode: str = "cells", no_wait: bool = False):
    from probe_shift.models import QWEN3_32B

    if subprocess.run(["git", "status", "--porcelain", str(Path(__file__))], capture_output=True, text=True, check=True).stdout:
        raise SystemExit(f"commit {Path(__file__).name} first")
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    model, rev = QWEN3_32B["name"], QWEN3_32B["revision"]
    folds = json.loads((M.REPO / "splits" / "folds_v2.json").read_text())["environment_folds"]["environment_to_fold"]
    if mode == "cells":
        print(json.dumps(cells.remote(model, rev)))
        return
    if mode in ("e", "d"):
        fn = read_e if mode == "e" else read_d
        calls = [fn.spawn(s, model, rev, folds) for s in range(SHARDS[mode])]
        if no_wait:
            print(f"spawned {SHARDS[mode]} {mode} shards; not waiting (then --mode analyze)", flush=True)
            return
        for c in calls:
            print(json.dumps(c.get()), flush=True)
        return
    res = analyze.remote(git)
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    (HERE / "results").mkdir(exist_ok=True)
    with open(HERE / "results" / f"attribution_tokens_{stamp}.json", "x") as fh:
        fh.write(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
