"""Token attribution (RULE.md in this folder): every token's score along claim B's fake-vs-real directions, for the staged
honest items (stage_inputs.py). One Modal H100 container, run detached; the result is written to the Volume as it goes.

1. Directions. Claim B's halves probes (run_halves_modal.py @ a2494dc42, sha 5b07509a98fc) were saved without weights,
   so the four used here are refit at the frozen selection (argmax of the stored val_auroc, column 0, as
   analyze_halves.py selects): rows assembled as that runner assembles them (its code, copied below; the assembled run
   list must equal its stored runs.json), labels equal to its stored Y and use, every label column fit as it fit them,
   with the project's fitter and refit loop. Pre-condition: in column 0, the refit's held-out run scores equal the stored
   test_run_score at the selection within 1e-6 (max abs) for all four, or the job stops before the reader loads.
     main    fixed_vs_e1 (real-attack-e1 vs sim-attack-fixed), dev_to_test and test_to_dev;
     plant2  sim-attack-fixed vs its R2-undone copies, dev_to_test and test_to_dev.
   An item is scored with the directions trained on the other environment half. In standardized units a direction is
   v = sqrt(C) W[:, 0] with offset b[0]; 5 random directions each: seeded unit vectors in the same standardized space,
   scaled to |v|, no offset (seed "token-attr-random|<half>|<comparison>|<i>").
2. Reader: Qwen3-32B (pinned), bf16, sdpa, the 32k cut, hooks after blocks 58 and 63, the residual cast to fp16 as the
   training rows were (B-dev's _read). Per token, float32: ((x - mean) / std) @ [v, random 1-5] (+ b for v).
3. Saved to the Volume (OUT): directions.npz (weights; they stay there), repro.json (the pre-condition, written and
   committed before the reader loads), one npz per item (token ids and the 12 scores of its window: site, plant, twin
   from the token before the reply to the end; whole runs every token), then scores.npz (all items) and done.json.
   Numbers and token ids only. A relaunch of the same script reuses the checked directions and the finished items (both
   bound to this script's sha and the staged inputs' sha) and returns done.json if the job finished.

  PYTHONPATH=src uv run --no-project --python 3.12 --with 'modal==1.6.1' --with numpy modal run --detach \\
      experiments/2026-10-13_claimB_token_attribution/run_token_attr_modal.py
"""

import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import modal

HERE = Path(__file__).parent
H3 = HERE.parent / "2026-10-11_claimB_h3_fix_to_plateau"
sys.path[:0] = [str(H3), str(HERE.parent / "2026-10-11_claimB_slips_evened_probe")] if modal.is_local() else ["/root"]
import run_h3_small_modal as R  # noqa: E402  (H3's runner, unchanged since the halves run: B-dev's code, constants)

HALVES_OUT, H3_RUNNER_SHA12 = "claimB/halves_v1/5b07509a98fc", "04c3953b139e"
B, M = R.B, R.M
SIM, E1, E2, PL2 = M.SIM, M.REAL, "real-attack-e2", "sim-attack-fixed-plant2"
TEST = list(R.TEST)
DEV = ["auto_workflow", "file_indexing", "model_registry", "slack_clone", "web_scraping"]
DIRS = {"dev_to_test": (DEV, TEST), "test_to_dev": (TEST, DEV)}
COMPS = {"fixed_vs_e1": (E1, SIM), "plant2": (SIM, PL2)}  # (negative arm, positive arm), as the halves runner
K, KP, VAL_FOLD, N_RANDOM, TOL, CUT = 101, 21, 0, 5, 1e-6, 32768
STAGE_RESULT = HERE / "results/stage_2026-10-13_132114.json"
SCRIPT_SHA = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
OUT = f"claimB/token_attr_v1/{SCRIPT_SHA[:12]}"

app = modal.App("probe-shift_aswin-surya_sub-claimb-token-attr_kill-2026-11-12")
image = R.image.add_local_file(H3 / "run_h3_small_modal.py", "/root/run_h3_small_modal.py")


def directions(cfg, log):
    """The four refits and the pre-condition. Rows, labels and fits copied from run_halves_modal.py @ a2494dc42."""
    import numpy as np
    import torch
    import torch.nn.functional as F

    from probe_shift import torch_probe as tp
    from probe_shift.q1_eval import step_table

    if (B.SCRIPT_SHA, R.SCRIPT_SHA[:12]) != (R.B_SCRIPT, H3_RUNNER_SHA12):
        raise ValueError("B-dev's or H3's runner differs from the halves run's")
    for base in (R.OUT, R.DEV_READS):
        if not json.loads((Path("/acts") / base / "gate.json").read_text()).get("passed"):
            raise ValueError(f"{base}: gate not passed")
    pruns = json.loads((Path("/acts") / M.PACK / "runs.L63.json").read_text())
    t = step_table(pruns, "output_end")
    XP = {l: np.load(Path("/acts") / M.PACK / f"L{l}_output_end.npy") for l in R.LAYERS}
    rows_of = {}
    for k in range(len(t["run"])):
        rows_of.setdefault(int(t["run"][k]), {})[(int(t["step"][k]), int(t["action"][k]))] = k

    def files(base, version):
        out = {}
        for f in (Path("/acts") / base / "runs").glob("*.npz"):
            z = np.load(f)
            m = json.loads(str(z["meta"]))
            if m["version"] != version:
                continue
            # the halves runner compared with B.SCRIPT_SHA and B.DEPS_SHA; the pinned values are compared here, since
            # torch_probe.py has since gained an optional argument (its default keeps the old behaviour)
            if (m["script_sha256"], m.get("deps_sha256"), m["model"], m["revision"]) != (R.B_SCRIPT, R.B_DEPS, cfg["model"], cfg["revision"]):
                raise ValueError(f"{f.name} was made by another version or model")
            out[(m["cell"], m["arm"])] = (m, z)
        return out

    test_reads, dev_reads, dev_planted = files(R.OUT, "evened"), files(R.DEV_READS, "evened"), files(R.DEV_READS, "planted")
    dev_plan = json.loads((Path("/acts") / R.DEV_READS / "cells.json").read_text())

    def read_rows(meta, z):
        rng = {tuple(x) for x in meta["range"]}
        keep = [k for k, s in enumerate(meta["steps"]) if tuple(s) not in rng]
        return {l: np.asarray(z[f"L{l}"][keep], np.float16) for l in R.LAYERS}, len(keep)

    runs, rr, Xl = [], [], {l: [] for l in R.LAYERS}
    env_of = {m["cell"]: m["environment"] for m in pruns}

    def add(entry, rows, n):
        if n == 0:
            raise ValueError(f"{entry['cell']} {entry['arm']}: no rows")
        for l in R.LAYERS:
            Xl[l].append(rows[l])
        rr.extend([len(runs)] * n)
        runs.append(entry | {"rows": n})

    for i, m in enumerate(pruns):
        if m["arm"] not in (SIM, E1, E2, PL2):
            continue
        src = (test_reads if m["environment"] in TEST else dev_reads).get((m["cell"], m["arm"]))
        if src:
            rows, n = read_rows(*src)
            how = "evened read"
        else:
            ix = sorted(rows_of.get(i, {}).values())
            rows, n = {l: XP[l][ix] for l in R.LAYERS}, len(ix)
            how = "pack rows"
        add({"cell": m["cell"], "arm": m["arm"], "environment": m["environment"], "main_task": m["main_task"], "source": how}, rows, n)
    for cell in dev_plan["planted"]:
        rows, n = read_rows(*dev_planted[(cell, SIM)])
        add({"cell": cell, "arm": PL2, "environment": env_of[cell], "main_task": cell.split(":")[0], "source": "B-dev planted read"}, rows, n)
    stored = json.loads((Path("/acts") / HALVES_OUT / "runs.json").read_text())["runs"]
    if [{k: r[k] for k in ("cell", "arm", "environment", "main_task", "source", "rows")} for r in runs] != stored:
        raise ValueError("the assembled runs differ from the halves run's runs.json")
    rr = np.array(rr)
    X = {l: np.concatenate(Xl[l]) for l in R.LAYERS}
    del XP
    Xl.clear()
    log(f"rows assembled: {len(rr)} rows, {len(runs)} runs (equal to the halves runs.json)")
    env = np.array([r["environment"] for r in runs])
    arm = np.array([r["arm"] for r in runs])
    task = np.array([r["main_task"] for r in runs])
    cell = np.array([r["cell"] for r in runs])
    pl2_cells = set(cell[arm == PL2])
    dev = torch.device("cuda")

    def labels(comp):
        neg, pos = COMPS[comp]
        if comp == "plant2":
            use = ((arm == PL2) | (arm == SIM)) & np.isin(cell, list(pl2_cells))
            unit, kk, tag = cell, KP, "plant2"
        else:
            use = (arm == neg) | (arm == pos)
            unit, kk, tag = task, K, "real"
        y = (arm == pos).astype(np.float32)
        Y = np.zeros((len(runs), kk), np.float32)
        Y[:, 0] = y
        units = sorted(set(unit[use])) if comp == "plant2" else sorted(set(task[np.isin(arm, [SIM, E1, E2])]))
        for k in range(1, kk):
            rngk = np.random.default_rng(R.seed_of("halves-shuffle", tag, k))
            flip = dict(zip(units, rngk.random(len(units)) < 0.5))
            Y[:, k] = [1 - v if (use[i] and flip[unit[i]]) else v for i, v in enumerate(y)]
        return use, Y

    def linear(Xa, Xb, Ya, C, cap):  # the halves runner's linear(), also returning the fit and the standardization
        Xd = Xa.double()
        mean, std = Xd.mean(0), Xd.std(0, unbiased=False)
        std = std.where(std > 0, std.new_ones(()))
        Xa, Xb = tp.standardize(Xd, Xb.double())
        Xa, Xb = Xa * C**0.5, Xb * C**0.5
        tol, mi, prev = 1e-4 * min(C**0.5, 1.0), R.MAX_ITER, None
        while True:
            W, b, evals = tp.fit(Xa, Ya, 1.0, mi, tol=tol)
            Wg, bg = W.clone().requires_grad_(True), b.clone().requires_grad_(True)
            loss = (F.binary_cross_entropy_with_logits(Xa @ Wg + bg, Ya, reduction="sum") + 0.5 * (Wg * Wg).sum()) / len(Xa)
            loss.backward()
            gk = torch.maximum(Wg.grad.abs().amax(0), bg.grad.abs()).cpu().numpy()
            if float(gk.max()) <= tol or evals < mi or evals == prev or mi * 4 > cap:
                return tp.decision(Xb, W, b), gk, W, b, mean, std
            mi, prev = mi * 4, evals

    dirs, repro = {}, {}
    for dname, (train_envs, test_envs) in DIRS.items():
        for comp in COMPS:
            z = np.load(Path("/acts") / HALVES_OUT / dname / f"{comp}.npz")
            subs, V = json.loads(str(z["subs"])), z["val_auroc"]
            if not np.isfinite(V).all():
                raise ValueError(f"{dname} {comp}: non-finite validation AUROC")  # as analyze_halves.py
            s = int(V[:, 0].argmax())
            l, C = subs[s]
            use, Y = labels(comp)
            if not (np.array_equal(use, z["use"]) and np.array_equal(Y, z["Y"])):
                raise ValueError(f"{dname} {comp}: labels differ from the halves run's")
            cap = R.MAX_ITER if comp == "plant2" else R.MAX_ITER_CAP
            u_r, e_r = use[rr], env[rr]
            ia, ib = np.flatnonzero(u_r & np.isin(e_r, train_envs)), np.flatnonzero(u_r & np.isin(e_r, test_envs))
            S, gk, W, b, mean, std = linear(torch.from_numpy(np.asarray(X[l][ia], np.float32)).to(dev),
                                            torch.from_numpy(np.asarray(X[l][ib], np.float32)).to(dev),
                                            torch.from_numpy(Y[rr[ia]]).to(dev).double(), C, cap)
            kk_, inv = np.unique(rr[ib], return_inverse=True)
            tot = torch.zeros(len(kk_), Y.shape[1], dtype=torch.float64, device=dev).index_add_(0, torch.from_numpy(inv).to(dev), S.double())
            Mv = (tot / torch.from_numpy(np.bincount(inv)).to(dev)[:, None]).cpu().numpy()
            diff = np.abs(Mv[:, 0] - z["test_run_score"][s, kk_, 0])
            repro[f"{dname}|{comp}"] = {"layer": l, "C": C, "val_auroc": float(V[s, 0]), "held_runs": len(kk_), "max_abs_diff": float(diff.max()),
                                        "max_abs_diff_all_columns": float(np.abs(Mv - z["test_run_score"][s, kk_]).max()),
                                        "grad_col0": float(gk[0]), "pass": bool(diff.max() <= TOL)}
            v = (W[:, 0] * C**0.5).double()
            rnd = []
            for i in range(N_RANDOM):
                g = np.random.default_rng(R.seed_of("token-attr-random", dname, comp, i)).standard_normal(len(v))
                rnd.append(torch.from_numpy(g / np.linalg.norm(g)).to(dev) * v.norm())
            dirs[(dname, comp)] = {"layer": l, "mean": mean.float(), "std": std.float(), "V": torch.stack([v] + rnd, 1).float(), "b": float(b[0])}
            log(f"{dname} {comp}: L{l} C {C} max abs diff {diff.max():.3g}")
            del S, tot
    torch.cuda.empty_cache()
    return dirs, repro


@app.function(image=image, gpu="H100!", cpu=8, memory=131072, volumes={"/acts": B.P.vol, "/hf": B.P.hf}, timeout=90 * 60, retries=0)
def run(cfg: dict, items: list[dict]) -> dict:
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from probe_shift.reader import _layout, segment_end_positions

    t0 = time.time()
    log = lambda m: print(f"[token-attr] {time.time() - t0:6.0f}s {m}", flush=True)  # noqa: E731
    B.P.vol.reload()
    out = Path("/acts") / OUT
    (out / "items").mkdir(parents=True, exist_ok=True)
    if (out / "done.json").exists():
        done = json.loads((out / "done.json").read_text())
        if done["staged_sha256"] != cfg["staged_sha256"]:
            raise ValueError("done.json is from other staged inputs")
        return done
    if (out / "repro.json").exists() and json.loads((out / "repro.json").read_text())["pass"]:  # a resume: reuse the checked directions
        z, repro = np.load(out / "directions.npz"), json.loads((out / "repro.json").read_text())["directions"]
        bb, ll = json.loads(str(z["b"])), json.loads(str(z["layer"]))
        dirs = {(d, c): {"layer": ll[f"{d}|{c}"], "b": bb[f"{d}|{c}"], **{k: torch.from_numpy(z[f"{d}|{c}|{k}"]).cuda() for k in ("mean", "std", "V")}}
                for d in DIRS for c in COMPS}
        log("directions reused from the checked refit")
    else:
        dirs, repro = directions(cfg, log)
        ok = all(r["pass"] for r in repro.values())
        np.savez(out / "directions.npz", **{f"{d}|{c}|{k}": x[k].cpu().numpy() for (d, c), x in dirs.items() for k in ("mean", "std", "V")},
                 b=json.dumps({f"{d}|{c}": x["b"] for (d, c), x in dirs.items()}), layer=json.dumps({f"{d}|{c}": x["layer"] for (d, c), x in dirs.items()}))
        (out / "repro.json").write_text(json.dumps({"tolerance": TOL, "pass": ok, "directions": repro, "script_sha256": SCRIPT_SHA}))
        B.P.vol.commit()
        if not ok:
            raise ValueError(f"refit does not reproduce the stored scores within {TOL}: {repro}")
    ok = True
    tok = AutoTokenizer.from_pretrained(cfg["model"], revision=cfg["revision"])
    lm = AutoModelForCausalLM.from_pretrained(cfg["model"], revision=cfg["revision"], dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa").eval()
    log("reader loaded")

    class Stop(Exception):
        pass

    order = [f"{c}|{k}" for c in COMPS for k in ["v"] + [f"r{i + 1}" for i in range(N_RANDOM)]]
    for it in items:
        name = out / "items" / f"{it['item']:04d}.npz"
        if name.exists():
            old = json.loads(str(np.load(name)["meta"]))
            if (old["script_sha256"], old["staged_sha256"]) != (SCRIPT_SHA, cfg["staged_sha256"]):
                raise ValueError(f"{name.name} was made from other code or inputs")
            continue
        half = "dev_to_test" if it["environment"] in TEST else "test_to_dev"  # the directions trained on the other half
        if it["environment"] not in TEST + DEV:
            raise ValueError(f"item {it['item']}: unknown environment")
        seg = segment_end_positions(tok, it["messages"], CUT)
        if seg["truncated"] and it["kind"] != "whole":
            raise ValueError(f"item {it['item']}: past the cut")
        last = seg["outputs"][-1]
        w0 = 0 if it["kind"] == "whole" else last["start"] - 1
        _, found, enc = _layout(tok, it["messages"])  # the window adds a token that holds the reply's last character and
        lt = [f for f in found if f[0] == "tool_output"][-1]  # runs on into the template (marks.layout's rule)
        b_, end, off = lt[5], last["end"], enc["offset_mapping"]
        if enc["input_ids"][: len(seg["ids"])] != seg["ids"]:
            raise ValueError(f"item {it['item']}: layout disagrees")
        if b_ > lt[4] and end < len(off) and off[end][0] < b_ < off[end][1]:
            end += 1
        got = {}

        def hook_for(layer):
            def hook(module, args, output):
                h = output[0] if isinstance(output, tuple) else output
                x = h[0, w0:].to(torch.float16).float()
                for comp in COMPS:
                    d = dirs[(half, comp)]
                    if d["layer"] == layer:
                        got[comp] = ((x - d["mean"]) / d["std"]) @ d["V"] + torch.tensor([d["b"]] + [0.0] * N_RANDOM, device=x.device)
                if layer == max(R.LAYERS):
                    raise Stop
            return hook

        handles = [lm.model.layers[l].register_forward_hook(hook_for(l)) for l in R.LAYERS]
        try:
            with torch.inference_mode():
                lm.model(input_ids=torch.tensor([seg["ids"]], device="cuda"), use_cache=False)
        except Stop:
            pass
        finally:
            for h in handles:
                h.remove()
        sc = torch.cat([got[c] for c in COMPS], 1).cpu().numpy()
        if not np.isfinite(sc).all():
            raise ValueError(f"item {it['item']}: non-finite scores")
        meta = {"item": it["item"], "half": half, "w0": w0, "reply": [last["start"], last["end"]], "window": [last["start"], end],
                "n_tokens": len(seg["ids"]), "columns": order, "script_sha256": SCRIPT_SHA, "staged_sha256": cfg["staged_sha256"]}
        tmp = name.with_suffix(".tmp.npz")
        np.savez(tmp, ids=np.array(seg["ids"][w0:], np.int32), scores=sc.astype(np.float32), meta=json.dumps(meta))
        tmp.replace(name)
        if it["item"] % 20 == 0:
            B.P.vol.commit()
            log(f"item {it['item']} of {len(items)}")
    B.P.vol.commit()
    files = [np.load(out / "items" / f"{it['item']:04d}.npz") for it in items]
    lens = np.array([len(f["ids"]) for f in files])
    np.savez(out / "scores.npz", ids=np.concatenate([f["ids"] for f in files]), scores=np.concatenate([f["scores"] for f in files]),
             starts=np.concatenate([[0], np.cumsum(lens)[:-1]]), lens=lens, meta=json.dumps([json.loads(str(f["meta"])) for f in files]))
    done = {"items": len(items), "tokens_scored": int(lens.sum()), "tokens_read": int(sum(min(it["n_tokens"], CUT) for it in items)),
            "repro_pass": ok, "seconds": round(time.time() - t0), "script_sha256": SCRIPT_SHA, "staged_sha256": cfg["staged_sha256"]}
    (out / "done.json").write_text(json.dumps(done))
    B.P.vol.commit()
    log(f"done: {done}")
    return done


@app.local_entrypoint()
def main():
    from probe_shift.models import QWEN3_32B

    if subprocess.run(["git", "status", "--porcelain", str(Path(__file__))], capture_output=True, text=True, check=True).stdout:
        raise SystemExit(f"commit {Path(__file__).name} first")
    staged = HERE / "data/staged.jsonl"
    raw = staged.read_bytes()
    if hashlib.sha256(raw).hexdigest() != json.loads(STAGE_RESULT.read_text())["staged_sha256"]:
        raise SystemExit("data/staged.jsonl differs from the committed staging result")
    items = [json.loads(x) for x in raw.decode().splitlines()]
    folds = json.loads((HERE.parents[1] / "splits/folds_v2.json").read_text())
    cfg = {"main_task_to_fold": folds["task_folds"]["main_task_to_fold"], "model": QWEN3_32B["name"], "revision": QWEN3_32B["revision"],
           "staged_sha256": hashlib.sha256(raw).hexdigest()}
    print(f"OUT {OUT}", flush=True)
    print(json.dumps(run.remote(cfg, items)), flush=True)
