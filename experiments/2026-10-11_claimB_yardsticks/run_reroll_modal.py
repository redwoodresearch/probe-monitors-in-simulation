"""I3, the sim-vs-sim reroll yardstick (PRICES.md item 1; 0d's GO relayed after 22:33 PT 10-11: 97 test cells, $3.9 with
30%, stop $5; header check passed, results/reroll_headers_2026-10-11_224554.json). Reuses, unchanged: fixed_sim's rule
and clock code (fixes.apply R1/R2/R5, clock R6 with donors = dev real e1 runs, as run_fixed_sim_modal.apply_shard), the
staging scrub (run_stage_probe_inputs_modal.scrubbed, copied), B-dev's reader (run_slips_evened_modal._read, copied) and
the meaning control's frozen refit (run_slips_evened_modal.refit, copied: pack claimB/level1/packedB_2cfd1aec393c, step
A's comparison, C 0.001, each test environment's probe trained without its fold).

  --mode stage  CPU: from the uploaded samples (stage_reroll_upload.py), build reader rows (the honest system prompt S1
                was read with), apply R1/R2/R5 and R6, scrub to level 1. Gates, all on the S1 samples of the same 97 cells
                (each must hold for every cell or nothing is written): the reader rows equal the stored inputs
                (2026-10-06_la_segment-end_qwen3-32b_v1); the fixed rows equal 2026-10-09_la_fixed-sim_v2 with the same
                clock donor; the level-1 rows equal step B's staged inputs (the pack's source). Writes S2's level-1 rows.
  --mode read   1 H100: bitwise gate first (4 seeded S1 cells re-read from step B's staged rows must equal the pack's rows
                at L58 and L63), then the 97 S2 reads (output_end, L58/L63). Resumable per cell.
  --mode score  1 H100: the frozen refits; per-run mean output_end scores for R1 (e1), R2 (e2), S1 (fixed sim) from the
                pack and S2 from the reads, at L63 (primary) and step A's layer; numbers only.
  --mode fetch  local: the scores file into data/ and its summary into results/.

  PYTHONPATH=src uv run --no-project --python 3.12 --with 'modal>=1.4.3' --with numpy modal deploy <this file>
  PYTHONPATH=src uv run --no-project --python 3.12 --with 'modal>=1.4.3' --with numpy python <this file> stage|read|score|fetch
"""

import hashlib
import json
import random
import subprocess
import sys
from pathlib import Path

import modal

_HERE = Path(__file__).resolve().parent
_MC = _HERE.parent / "2026-10-10_claimB_meaning_control"
LOCAL = _MC.exists()  # in the container this file is /root/run_reroll_modal.py and _MC does not exist
sys.path.insert(0, str(_MC) if LOCAL else "/root")
import run_meaning_plant_modal as MP  # noqa: E402  (image, Volumes, staged inputs, frozen-refit constants)

M, P = MP.M, MP.P
NAME = "probe-shift_aswin-surya_sub-claimb-reroll_kill-2026-11-11"
OUT, INPUTS, FIXED = "claimB/reroll_yardstick_v1", "2026-10-06_la_segment-end_qwen3-32b_v1", "2026-10-09_la_fixed-sim_v2"
GATE_RUNS, SEED = 4, "reroll-yardstick-2026-10-11"
SCRIPT_SHA = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
N_CELLS = 97


def deps_sha(kind: str) -> str:
    """sha256 over the code each container step depends on (computed inside the container)."""
    import probe_shift

    ps = Path(probe_shift.__file__).parent
    if kind == "read":
        files = [ps / f for f in ("reader.py", "scrub.py", "torch_probe.py", "q1_eval.py")] + [Path(m.__file__) for m in (MP, M, P)]
    else:
        files = [ps / f for f in ("transcripts.py", "scrub.py")] + [Path(f) for f in (
            "/root/fixed_sim/fixes.py", "/root/fixed_sim/clock.py", "/root/2026-10-07_q2_rendering_swap/rerender.py",
            "/root/2026-10-07_q2_rendering_swap/str_replace_tells.py")]
    return hashlib.sha256(b"".join(f.read_bytes() for f in files)).hexdigest()

app = modal.App(NAME)
read_image = MP.image  # B-dev's image chain, unchanged (the bitwise gate depends on it)
# The staging image: P.image's packages plus boto3 (probe_shift.transcripts imports it), then the same files the
# meaning-control chain mounts (this module imports it) and fixed_sim's rule and clock code.
stage_image = (modal.Image.debian_slim(python_version="3.12").uv_pip_install(*P.PIP, "boto3")
               .env({"HF_HOME": "/hf", "HF_HUB_OFFLINE": "1"}))
if LOCAL:
    _FS, _SWAP = _HERE.parent / "2026-10-09_fixed_sim", _HERE.parent / "2026-10-07_q2_rendering_swap"
    read_image = MP.image.add_local_file(_MC / "run_meaning_plant_modal.py", "/root/run_meaning_plant_modal.py")
    stage_image = (stage_image.add_local_python_source("probe_shift")
                   .add_local_file(_FS / "run_plant_token_pilot_modal.py", "/root/run_plant_token_pilot_modal.py")
                   .add_local_file(_MC / "run_meaning_control_modal.py", "/root/run_meaning_control_modal.py")
                   .add_local_file(_MC / "undo_r2.py", "/root/undo_r2.py")
                   .add_local_file(_SWAP / "str_replace_tells.py", "/root/swap/str_replace_tells.py")
                   .add_local_file(_SWAP / "rerender.py", "/root/swap/rerender.py")
                   .add_local_file(_MC / "run_meaning_plant_modal.py", "/root/run_meaning_plant_modal.py")
                   .add_local_file(_FS / "fixes.py", "/root/fixed_sim/fixes.py")
                   .add_local_file(_FS / "clock.py", "/root/fixed_sim/clock.py")
                   .add_local_file(_SWAP / "rerender.py", "/root/2026-10-07_q2_rendering_swap/rerender.py")
                   .add_local_file(_SWAP / "str_replace_tells.py", "/root/2026-10-07_q2_rendering_swap/str_replace_tells.py"))


def rows_of(run: str, keep):
    for f in sorted(Path(f"/acts/{run}/inputs").glob("shard_*.jsonl")):
        with open(f) as fh:
            for line in fh:
                r = json.loads(line)
                if keep(r):
                    yield r


def scrubbed(messages: list[dict], level: int) -> list[dict]:
    """run_stage_probe_inputs_modal.scrubbed, copied unchanged."""
    from probe_shift.scrub import scrub

    def args(v):
        if isinstance(v, str):
            return scrub(v, level)
        if isinstance(v, bool) or v is None:
            return v
        if isinstance(v, (int, float)):
            return "<n>" if level >= 2 else v
        if isinstance(v, list):
            return [args(x) for x in v]
        if isinstance(v, dict):
            return {k: args(x) for k, x in v.items()}
        return v

    out = []
    for m in messages:
        if m["role"] == "tool":
            m = m | {"content": scrub(m["content"], level)}
        elif m["role"] == "assistant":
            m = m | {"content": scrub(m["content"], level) if m["content"] else m["content"],
                     "tool_calls": [c | {"function": c["function"] | {"arguments": args(c["function"]["arguments"])}}
                                    for c in m.get("tool_calls") or []]}
        out.append(m)
    return out


def outputs_of(messages: list[dict]) -> list[str]:
    return [m["content"] for m in messages if m["role"] == "tool" and isinstance(m["content"], str)]


def fixed(messages, cell, donors, clock, fixes):
    """run_fixed_sim_modal.apply_shard's simulated-run branch, copied: R1/R2/R5, then R6 with the cell's donor."""
    new, recs = fixes.apply(messages, ("R1", "R2", "R5"))
    now = clock.now_of(outputs_of(messages))
    clock_rec = {"now": None, "donor_cell": None, "donor_now": None, "offset_s": None}
    if now is not None:
        donor, donor_now = clock.donor_for(cell, donors)
        offset = donor_now - now
        clock_rec = {"now": now.isoformat(), "donor_cell": donor, "donor_now": donor_now.isoformat(),
                     "offset_s": round(offset.total_seconds())}
        for i, m in enumerate(new):
            if m["role"] == "tool" and isinstance(m["content"], str):
                shifted, _ = clock.shift_text(m["content"], offset, now)
                if shifted != m["content"]:
                    new[i] = m | {"content": shifted}
                    recs.append({"msg": i, "rule": "R6", "category": "shifted", "changed": True})
    return new, recs, clock_rec


@app.function(image=stage_image, cpu=8, memory=65536, volumes={"/acts": P.vol}, timeout=45 * 60, retries=0)
def stage(cells939: list[str], dev: list[str]) -> dict:
    import gzip
    from collections import Counter
    from datetime import datetime

    sys.path[:0] = ["/root/fixed_sim", "/root/2026-10-07_q2_rendering_swap"]
    import clock
    import fixes

    from probe_shift.transcripts import reader_messages

    P.vol.reload()
    if Path(f"/acts/{OUT}/stage.json").exists():
        raise FileExistsError("stage.json exists; staging runs once")
    samples = [json.loads(l) for l in gzip.open(f"/acts/{OUT}/samples.jsonl.gz", "rt")]
    cells = sorted({s["cell"] for s in samples})
    pairs = Counter((s["cell"], s["arm"]) for s in samples)
    if len(cells) != N_CELLS or set(pairs) != {(c, a) for c in cells for a in ("sim-attack", "sim-reroll")} or max(pairs.values()) != 1:
        raise ValueError(f"expected one S1 and one S2 sample for each of {N_CELLS} cells")
    want, all939 = set(cells), set(cells939)
    inp = {r["cell"]: r for r in rows_of(INPUTS, lambda r: r["arm"] == "sim-attack" and r["cell"] in want)}
    fx = {r["cell"]: r for r in rows_of(FIXED, lambda r: r["arm"] == "sim-attack" and r["cell"] in want)}
    stb = {r["cell"]: r for r in MP.inputs(M.STAGED_B) if r["arm"] == M.SIM and r["cell"] in want}
    donor_rows = list(rows_of(INPUTS, lambda r: r["arm"] == "real-attack-e1" and r["environment"] in set(dev) and r["cell"] in all939))
    assert len({r["cell"] for r in donor_rows}) == len(donor_rows), "two dev real e1 runs share a cell"
    donors = {r["cell"]: n for r in donor_rows if (n := clock.now_of(outputs_of(r["messages"]))) is not None}
    if not (set(inp) == set(fx) == set(stb) == want):
        raise ValueError("a stored S1 row is missing for some cell")
    gate, count, s2_rows = Counter(), Counter(), []
    for s in samples:
        c = s["cell"]
        prompt = inp[c]["messages"][0]["content"]
        msgs = reader_messages({"id": s["id"], "epoch": s["epoch"], "messages": s["messages"]}, prompt)
        new, recs, clock_rec = fixed(msgs, c, {k: v for k, v in donors.items()}, clock, fixes)
        lv1 = scrubbed(new, 1)
        if s["arm"] == "sim-attack":
            gate["reader_equal"] += msgs == inp[c]["messages"]
            gate["fixed_equal"] += new == fx[c]["messages"]
            gate["donor_equal"] += clock_rec["donor_cell"] == fx[c]["fixed_sim"]["clock"]["donor_cell"]
            gate["level1_equal"] += lv1 == stb[c]["messages"]
            gate["cells"] += 1
        else:
            count["s2_runs"] += 1
            count["s2_outputs_changed"] += len({x["msg"] for x in recs if x["changed"]})
            count["s2_runs_with_now"] += clock_rec["now"] is not None
            s2_rows.append({"cell": c, "arm": "sim-reroll", "environment": s["environment"], "main_task": s["main_task"],
                            "key": f"reroll__{c}", "messages": lv1, "fixed_sim": {"clock": clock_rec}})
    passed = all(gate[k] == gate["cells"] == len(cells) for k in ("reader_equal", "fixed_equal", "donor_equal", "level1_equal"))
    passed = passed and count["s2_runs"] == N_CELLS
    res = {"gates": dict(gate), "passed": passed, "counts": dict(count), "cells": len(cells), "donors": len(donors),
           "script_sha256": SCRIPT_SHA, "deps_sha256": deps_sha("stage"), "at": datetime.now().isoformat()}
    with open(f"/acts/{OUT}/stage.json", "x") as fh:
        fh.write(json.dumps(res))
    if passed:
        data = "".join(json.dumps(r) + "\n" for r in sorted(s2_rows, key=lambda r: r["cell"]))
        with open(f"/acts/{OUT}/inputs_s2_level1.jsonl", "x") as fh:
            fh.write(data)
        res["inputs_sha256"] = hashlib.sha256(data.encode()).hexdigest()
    P.vol.commit()
    if not passed:
        raise ValueError(f"staging gates failed: {dict(gate)}")
    return res


def _read(jobs, model, revision, tag):
    """B-dev's reader (run_slips_evened_modal._read), copied; one npz per (cell, arm, version)."""
    import time

    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from probe_shift.reader import segment_end_positions

    t0, deps = time.time(), deps_sha("read")
    tok = AutoTokenizer.from_pretrained(model, revision=revision)
    lm = AutoModelForCausalLM.from_pretrained(model, revision=revision, dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa").eval()
    out_dir = Path("/acts") / OUT / "runs"
    out_dir.mkdir(parents=True, exist_ok=True)

    class Stop(Exception):
        pass

    done = 0
    for cell, arm, version, msgs in jobs:
        name = out_dir / f"{hashlib.sha256(f'{cell}|{arm}|{version}'.encode()).hexdigest()[:16]}.npz"
        if name.exists():
            old = json.loads(str(np.load(name)["meta"]))
            if (old["script_sha256"], old.get("deps_sha256"), old["model"], old["revision"]) != (SCRIPT_SHA, deps, model, revision):
                raise ValueError(f"{name.name} was made with other code or model; use a new output directory")
            done += 1
            continue
        seg = segment_end_positions(tok, msgs, M.MAX_TOKENS)
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
        meta = {"cell": cell, "arm": arm, "version": version, "steps": [[s, a] for s, a, _ in pos],
                "script_sha256": SCRIPT_SHA, "deps_sha256": deps, "model": model, "revision": revision}
        tmp = name.with_suffix(".tmp.npz")
        np.savez(tmp, **{f"L{l}": got[l] for l in M.LAYERS}, meta=json.dumps(meta))
        tmp.replace(name)
        P.vol.commit()
        done += 1
        print(f"[{time.time() - t0:6.0f}s] {tag} {done}/{len(jobs)}", flush=True)
    return {"tag": tag, "jobs": len(jobs), "done": done, "seconds": round(time.time() - t0)}


def pack_rows():
    """The pack's run list and its output_end step table, with each run's rows by (step, action)."""
    from probe_shift.q1_eval import step_table

    pruns = json.loads((Path("/acts") / M.PACK / "runs.L63.json").read_text())
    t = step_table(pruns, "output_end")
    rows_of_run = {}
    for k in range(len(t["run"])):
        rows_of_run.setdefault(int(t["run"][k]), {})[(int(t["step"][k]), int(t["action"][k]))] = k
    return pruns, t, rows_of_run


@app.function(image=read_image, gpu="H100!", cpu=8, memory=65536, volumes={"/acts": P.vol, "/hf": P.hf}, timeout=60 * 60, retries=0)
def read(model: str, revision: str, stage_deps: str) -> dict:
    """Bitwise gate on 4 seeded S1 cells (step B's staged level-1 rows re-read must equal the pack's rows), then S2."""
    import numpy as np

    P.vol.reload()
    st = json.loads((Path("/acts") / OUT / "stage.json").read_text())
    if not st["passed"] or (st["script_sha256"], st["deps_sha256"]) != (SCRIPT_SHA, stage_deps):
        raise ValueError("no passed staging for this script and the current staging code (stage_deps, from the laptop)")
    s2 = [json.loads(l) for l in (Path("/acts") / OUT / "inputs_s2_level1.jsonl").read_text().splitlines()]
    cells = sorted(r["cell"] for r in s2)
    if len(cells) != N_CELLS or len(set(cells)) != N_CELLS:
        raise ValueError("staged S2 rows are not one per expected cell")
    gate_cells = random.Random(SEED).sample(cells, GATE_RUNS)
    stb = {r["cell"]: r for r in MP.inputs(M.STAGED_B) if r["arm"] == M.SIM and r["cell"] in set(gate_cells)}
    g = _read([(c, M.SIM, "gate", stb[c]["messages"]) for c in gate_cells], model, revision, "gate")
    pruns, t, rows_of_run = pack_rows()
    key_of = {pruns[i]["key"]: i for i in range(len(pruns))}
    eq = []
    for c in gate_cells:
        z = np.load(Path("/acts") / OUT / "runs" / f"{hashlib.sha256(f'{c}|{M.SIM}|gate'.encode()).hexdigest()[:16]}.npz")
        pr = rows_of_run[key_of[stb[c]["key"]]]
        steps = [tuple(x) for x in json.loads(str(z["meta"]))["steps"]]
        ok = steps == sorted(pr)
        for l in M.LAYERS:
            XP = np.load(Path("/acts") / M.PACK / f"L{l}_output_end.npy", mmap_mode="r")
            ok = ok and np.array_equal(z[f"L{l}"], np.stack([XP[pr[s]] for s in steps]))
        eq.append(bool(ok))
    gate = {"cells": gate_cells, "bitwise_equal": eq, "passed": all(eq) and len(eq) == GATE_RUNS, "script_sha256": SCRIPT_SHA,
            "deps_sha256": deps_sha("read"), "stage_deps_sha256": stage_deps, "model": model, "revision": revision} | g
    (Path("/acts") / OUT / "gate.json").write_text(json.dumps(gate))
    P.vol.commit()
    if not gate["passed"]:
        raise ValueError(f"bitwise gate failed: {eq}")
    r = _read([(x["cell"], x["arm"], "s2", x["messages"]) for x in s2], model, revision, "s2")
    return {"gate": gate, "s2": r}


def refit(env, layer, env_to_fold, arm_of, efold, XP):
    """The meaning control's frozen refit (run_slips_evened_modal.refit, copied): step A's comparison, C 0.001."""
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


@app.function(image=read_image, gpu="H100!", cpu=8, memory=65536, volumes={"/acts": P.vol}, timeout=20 * 60, retries=0)
def score(env_to_fold: dict, model: str, revision: str) -> dict:
    import numpy as np
    import torch
    from sklearn.metrics import roc_auc_score

    P.vol.reload()
    deps = deps_sha("read")
    g = json.loads((Path("/acts") / OUT / "gate.json").read_text())
    if not g["passed"] or (g["script_sha256"], g["deps_sha256"], g["model"], g["revision"]) != (SCRIPT_SHA, deps, model, revision):
        raise ValueError("no passed gate for this script, its code and this model")
    if (Path("/acts") / OUT / "scores.json").exists():
        raise FileExistsError("scores.json exists")
    s2 = {r["cell"]: r for r in map(json.loads, (Path("/acts") / OUT / "inputs_s2_level1.jsonl").read_text().splitlines())}
    reads = {}
    for c in s2:  # preflight: every S2 read exists, is non-empty and comes from this code and model, before any fit
        f = Path("/acts") / OUT / "runs" / f"{hashlib.sha256(f'{c}|sim-reroll|s2'.encode()).hexdigest()[:16]}.npz"
        if not f.exists():
            raise ValueError(f"{c}: S2 read missing")
        m = json.loads(str(np.load(f)["meta"]))
        if (m["script_sha256"], m.get("deps_sha256"), m["model"], m["revision"]) != (SCRIPT_SHA, deps, model, revision) or not m["steps"]:
            raise ValueError(f"{c}: S2 read empty or from other code")
        reads[c] = f
    if len(reads) != N_CELLS:
        raise ValueError("not one S2 read per expected cell")
    pruns, t, rows_of_run = pack_rows()
    arm_of = np.array([pruns[i]["arm"] for i in t["run"]])
    efold = np.array([env_to_fold[pruns[i]["environment"]] for i in t["run"]])
    XP = {l: np.load(Path("/acts") / M.PACK / f"L{l}_output_end.npy") for l in M.LAYERS}
    probes, fits = {}, {}
    for e in M.A_LAYER:
        for l in sorted({63, M.A_LAYER[e]}):
            probes[(e, l)], fits[f"{e}|L{l}"] = refit(e, l, env_to_fold, arm_of, efold, XP)
    if not all(f["converged"] for f in fits.values()):
        raise ValueError(f"a frozen refit did not converge: {fits}")

    def s(e, l, X):
        mean, std, W, b = probes[(e, l)]
        return ((((torch.from_numpy(np.asarray(X, dtype=np.float32)).cuda().double() - mean) / std * M.C**0.5) @ W + b)[:, 0]).cpu().numpy()

    by_cell_arm = {(pruns[i]["cell"], pruns[i]["arm"]): i for i in range(len(pruns)) if pruns[i]["environment"] in M.A_LAYER}
    out = {"fits": fits, "runs": {}, "sanity": {}}
    for layer_kind in ("L63", "A"):
        runs = {}
        for c, r in sorted(s2.items()):
            e = r["environment"]
            l = 63 if layer_kind == "L63" else M.A_LAYER[e]
            rec = {"environment": e, "main_task": r["main_task"]}
            for name, arm in (("R1", M.REAL), ("R2", "real-attack-e2"), ("S1", M.SIM)):
                ri = by_cell_arm.get((c, arm))
                if ri is None:
                    raise ValueError(f"{c}: no pack run for {arm}")
                rows = sorted(rows_of_run[ri].values())
                rec[name] = float(s(e, l, XP[l][rows]).mean())
            z = np.load(reads[c])
            m = json.loads(str(z["meta"]))
            rec["S2"] = float(s(e, l, z[f"L{l}"]).mean())
            rec["S2_outputs"] = len(m["steps"])
            runs[c] = rec
        out["runs"][layer_kind] = runs
        # sanity: S1 vs R1 AUROC over every test-environment pack cell (the frozen probes' own comparison)
        ys, ss = [], []
        for (c, arm), ri in by_cell_arm.items():
            if arm in (M.REAL, M.SIM):
                e = pruns[ri]["environment"]
                l = 63 if layer_kind == "L63" else M.A_LAYER[e]
                ys.append(arm == M.SIM)
                ss.append(float(s(e, l, XP[l][sorted(rows_of_run[ri].values())]).mean()))
        out["sanity"][layer_kind] = {"S1_vs_R1_auroc_all_test_cells": float(roc_auc_score(ys, ss)), "runs": len(ys)}
    out |= {"script_sha256": SCRIPT_SHA, "deps_sha256": deps, "model": model, "revision": revision}
    with open(Path("/acts") / OUT / "scores.json", "x") as fh:
        fh.write(json.dumps(out))
    P.vol.commit()
    return {"fits_converged": all(f["converged"] for f in fits.values()), "sanity": out["sanity"], "cells": len(s2)}


def local(mode: str):
    from datetime import datetime
    from zoneinfo import ZoneInfo

    from probe_shift.models import QWEN3_32B

    repo = _HERE.parents[1]
    if subprocess.run(["git", "status", "--porcelain", str(Path(__file__))], capture_output=True, text=True, check=True).stdout:
        raise SystemExit(f"commit {Path(__file__).name} first")
    model, rev = QWEN3_32B["name"], QWEN3_32B["revision"]
    fn = lambda n: modal.Function.from_name(NAME, n)  # noqa: E731
    if mode == "stage":
        excl = json.loads((repo / "splits/exclusions_v4.json").read_text())
        dev = json.loads((repo / "experiments/2026-10-09_fixed_sim/split.json").read_text())["dev"]
        print(json.dumps({"call": fn("stage").spawn(excl["paired_attack_cells"]["cells"], dev).object_id}))
    elif mode == "read":
        ps, fs, sw = repo / "src/probe_shift", repo / "experiments/2026-10-09_fixed_sim", repo / "experiments/2026-10-07_q2_rendering_swap"
        files = [ps / "transcripts.py", ps / "scrub.py", fs / "fixes.py", fs / "clock.py", sw / "rerender.py", sw / "str_replace_tells.py"]
        if subprocess.run(["git", "status", "--porcelain", *map(str, files)], capture_output=True, text=True, check=True).stdout:
            raise SystemExit("staging dependencies have uncommitted changes")
        stage_deps = hashlib.sha256(b"".join(f.read_bytes() for f in files)).hexdigest()  # the same files as deps_sha("stage")
        print(json.dumps({"call": fn("read").spawn(model, rev, stage_deps).object_id}))
    elif mode == "score":
        folds = json.loads((repo / "splits/folds_v2.json").read_text())["environment_folds"]["environment_to_fold"]
        print(json.dumps({"call": fn("score").spawn(folds, model, rev).object_id}))
    elif mode == "fetch":
        raw = b"".join(P.vol.read_file(f"{OUT}/scores.json"))
        stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
        dst = _HERE / "data" / f"reroll_scores_{stamp}.json"
        dst.parent.mkdir(exist_ok=True)
        with open(dst, "xb") as fh:
            fh.write(raw)
        print(json.dumps({"file": str(dst), "sha256": hashlib.sha256(raw).hexdigest()}))
    else:
        raise SystemExit(f"unknown mode {mode}")


if __name__ == "__main__":
    local(*sys.argv[1:])
