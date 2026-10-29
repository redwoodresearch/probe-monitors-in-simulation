"""Item D (RULE.md, frozen at be1e5090): matched training on the 2 x 2 of evaluation and simulation cues inside real
honest LA runs, read by Qwen3.5-27B on Modal. Only numbers, hashed run keys, main tasks and environments leave.

- `dry` (Modal CPU): the splits (d_lib.select), every run's four versions built and rendering-checked, token and row
  counts, and the CPU time of building the versions.
- `time` (one H200): two versions read twice (cold, warm); a batched fit (21 label sets) at each C, and a 40-column one,
  timed on real layer-63 rows from those reads, tiled to the dry run's token-max row count, with random labels (a slow
  case), with each fit's max gradient after every step.
- `full` (one H200), resumable after the fits:
  1. development passes: both targets' slot tokens, 13 background tokens per version (seed 20261013, as every row
     draw), and every scored call's span mean, at layers 2, 4, 8, 16, 32, 48 and 63;
  2. fits per target, layer and readout: rows standardized once; per C one batched L-BFGS over 21 label sets (0 real,
     1-20 the inference null's swaps), and at layer 2, C 1e-3, token-max 40 more (21-40 balanced, 41-60 independent
     swaps: the diagnostic); float64; a final max gradient above 1e-5 stops the job. Saved to the Volume;
  3. selection and test passes: every probe applied; per version, each probe's run score (token-max: max logit over
     eligible content positions; downstream: mean over scored calls of each call's span-mean logit).
  `full` launches only if the dry and timing files match this code and the roster, and the projection is under $56.3.

  PYTHONPATH=src uv run --no-project --python 3.12 --with modal modal deploy experiments/2026-10-12_matched_training_d/run_matched_d.py
  ... python experiments/2026-10-12_matched_training_d/run_matched_d.py launch dry | time --dry <f> | full --dry <f> --time <f> --prior <$> [--resume <stamp>]
  ... collect <stamp>
"""

import hashlib
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))
import d_lib as L  # noqa: E402

APP = "probe-shift_aswin-surya_sub-matched-d_kill-2026-11-12"
VOL, LA = "probe-shift_aswin-surya_sub-evalaware_acts_kill-2026-11-09", "probe-shift_aswin-surya_sub-la-activations"
STAGED = "/in/2026-10-06_la_segment-end-v4_qwen3-32b_v1/inputs"
LAYERS, CS, N_NULL, N_BG, N_FIX = (2, 4, 8, 16, 32, 48, 63), (1e-4, 1e-3, 1e-2, 1e-1), 20, 13, 10_000
DIAG = (2, 1e-3)  # RULE.md: the diagnostic's fixed configuration (token-max)
STOP, RATE, N_USED, TIMEOUT = 56.3, 6.1, 211, 21600  # N_USED: results/pool_counts.json's "used tasks found"
PIP = ["torch==2.8.0", "transformers==5.19.0", "accelerate>=1.1.0", "numpy", "huggingface_hub", "flash-linear-attention==0.5.2"]
app = modal.App(APP)
vol = modal.Volume.from_name(VOL, version=2)
la = modal.Volume.from_name(LA, version=2).read_only()
hf = modal.Volume.from_name("probe-shift_aswin-surya_hf-cache").read_only()
image = (modal.Image.debian_slim(python_version="3.12").uv_pip_install(*PIP).env({"HF_HOME": "/hf", "HF_HUB_OFFLINE": "1"})
         .add_local_python_source("probe_shift", "d_lib"))
h16 = lambda s: hashlib.sha256(s.encode()).hexdigest()[:16]


def code_sha():
    """This script, d_lib, and the shared reader and model pin."""
    import importlib.util
    pkg = Path(importlib.util.find_spec("probe_shift").origin).parent
    return hashlib.sha256(b"".join(p.read_bytes() for p in (Path(__file__), Path(L.__file__), pkg / "reader.py", pkg / "models.py"))).hexdigest()


def guarded(fn):
    """stdout and stderr discarded (flushed and restored after); failures leave as type and line numbers only."""
    def run(*a):
        import traceback
        saved, null = (os.dup(1), os.dup(2)), os.open(os.devnull, os.O_WRONLY)
        os.dup2(null, 1), os.dup2(null, 2)
        try:
            if code_sha() != a[-1]["code_sha256"]:
                raise RuntimeError("the deployed code is not the launching commit's: redeploy")
            return fn(*a)
        except BaseException as e:
            lines = [(Path(f.filename).name, f.lineno) for f in traceback.extract_tb(e.__traceback__) if f.filename.endswith(("run_matched_d.py", "d_lib.py"))]
            raise RuntimeError(f"{fn.__name__} failed: {type(e).__name__} at {lines}") from None
        finally:
            sys.stdout.flush(), sys.stderr.flush()
            os.dup2(saved[0], 1), os.dup2(saved[1], 2)
            for fd in (*saved, null):
                os.close(fd)
    return run


def tokenizer():
    from transformers import AutoTokenizer

    from probe_shift.models import QWEN35_27B
    return AutoTokenizer.from_pretrained(QWEN35_27B["name"], revision=QWEN35_27B["revision"])


def staged():
    for p in sorted(Path(STAGED).glob("shard_*.jsonl")):
        yield from map(json.loads, p.read_text().splitlines())


def roster_sha(roster):
    return hashlib.sha256(json.dumps(roster).encode()).hexdigest()


@app.function(image=image, volumes={"/in": la, "/hf": hf}, cpu=8, memory=32768, timeout=3600, retries=0)
def dry(used_runs: list, used_tasks: list, prov: dict) -> dict:
    return guarded(_dry)(used_runs, used_tasks, prov)


def _dry(used_runs, used_tasks, prov):
    task_of = {it["key"]: it["main_task"] for it in staged() if it["arm"] == "real-honest"}
    if any(k not in task_of for k in used_runs):
        raise ValueError("a used run is not in the staging")
    used = {task_of[k] for k in used_runs} | set(used_tasks)
    if len(used) != N_USED:
        raise ValueError("the used tasks are not count_d_pool's")
    tok, t0 = tokenizer(), time.time()
    L.check_words(tok)
    s, skips = L.select(tok, STAGED, used)
    select_s, t0 = time.time() - t0, time.time()
    n_tok, place, slot_rows, down_rows, bg_rows = {}, {}, {"eval": 0, "sim": 0}, 0, 0
    for k, runs in s.items():
        for it in runs:
            vs, p = L.versions(tok, it["messages"], k, it["main_task"])
            n_tok.setdefault(k, []).extend(len(v["ids"]) for v in vs)
            place[f"{k}|{p}"] = place.get(f"{k}|{p}", 0) + 1
            if k == "dev":
                slot_rows["eval"] += sum(len(v["e"]) for v in vs)
                slot_rows["sim"] += sum(len(v["s"]) for v in vs)
                down_rows += sum(len(v["calls"]) for v in vs)
                bg_rows += sum(min(N_BG, len(set(v["eligible"]) - set(v["e"]) - set(v["s"]))) for v in vs)
    roster = {k: [h16(it["key"]) for it in v] for k, v in s.items()}
    return {"counts": {k: len(v) for k, v in s.items()}, "skips": skips, "placements": place, "roster": roster, "roster_sha256": roster_sha(roster),
            "tasks": {k: [it["main_task"] for it in v] for k, v in s.items()}, "environments": {k: [it["environment"] for it in v] for k, v in s.items()},
            "tokens": {k: sum(v) for k, v in n_tok.items()}, "total tokens": sum(sum(v) for v in n_tok.values()),
            "max tokens": max(max(v) for v in n_tok.values()), "slot rows": slot_rows, "token-max rows": {t: n + 2 * N_FIX for t, n in slot_rows.items()},
            "downstream rows": down_rows, "background pool": bg_rows, "select s": round(select_s), "versions s (all runs once)": round(time.time() - t0), "provenance": prov}


@app.function(image=image, gpu="H200", volumes={"/in": la, "/hf": hf, "/acts": vol}, cpu=4, memory=98304, timeout=TIMEOUT,
              startup_timeout=600, retries=0)
def read(phase: str, prov: dict) -> dict:
    return guarded(_read)(phase, prov)


def _read(phase, prov):
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM

    from probe_shift.models import QWEN35_27B
    t0, tok = time.time(), tokenizer()
    by_h = {h16(it["key"]): it for it in staged() if it["arm"] == "real-honest"}
    s = {k: [by_h[h] for h in v] for k, v in prov["roster"].items()}
    if roster_sha(prov["roster"]) != prov["roster_sha256"]:
        raise ValueError("the roster is not the dry run's")
    lm = AutoModelForCausalLM.from_pretrained(QWEN35_27B["name"], revision=QWEN35_27B["revision"], dtype=torch.bfloat16,
                                              device_map="cuda", attn_implementation="sdpa").eval()
    load_s, pad = round(time.time() - t0), (tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id)

    class Stop(Exception):
        pass

    def forward(ids):
        acts = {}

        def hook_for(layer):
            def hook(module, args, output):
                acts[layer] = (output[0] if isinstance(output, tuple) else output)[0]
                if layer == LAYERS[-1]:
                    raise Stop
            return hook
        hs = [lm.model.layers[l].register_forward_hook(hook_for(l)) for l in LAYERS]
        try:
            with torch.inference_mode():
                lm.model(input_ids=torch.tensor([ids + [pad] * (-len(ids) % 4096)], device="cuda"), use_cache=False)
        except Stop:
            pass
        finally:
            for h in hs:
                h.remove()
        return acts

    if phase == "time":
        reads, seen, real = [], set(), []
        for split, it in [("sel", s["sel"][0]), ("test", s["test"][0])] * 2:
            v = L.versions(tok, it["messages"], split, it["main_task"])[0][3]
            b = -(-len(v["ids"]) // 4096)
            t1 = time.time()
            acts = forward(v["ids"])
            torch.cuda.synchronize()
            reads.append({"tokens": len(v["ids"]), "seconds": round(time.time() - t1, 2), "warm": b in seen})
            seen.add(b)
            if len(real) < 2:
                real.append(acts[63][torch.tensor(v["eligible"], device="cuda")].double())
        n = max(prov["token-max rows"].values())
        X = torch.cat(real)
        X = X[torch.arange(n, device="cuda") % len(X)]
        Z = (X - X.mean(0)) / X.std(0).clamp_min(1e-6)
        Y = (torch.rand(n, 21, generator=torch.Generator("cuda").manual_seed(int(L.SEED)), device="cuda") < 0.35).double()
        fit_s, traces = {}, {}
        for C in CS:
            traces[str(C)], t1 = [], time.time()
            L.fit(Z, Y, C, check=False, trace=traces[str(C)])
            torch.cuda.synchronize()
            fit_s[str(C)] = round(time.time() - t1, 1)
        t1 = time.time()
        _, g40 = L.fit(Z, torch.cat([Y, Y[torch.randperm(n, device="cuda")][:, :19]], 1), DIAG[1], check=False)  # a diagnostic-sized solve
        torch.cuda.synchronize()
        return {"load_s": load_s, "rows": reads, "fit_s by C": fit_s, "fit traces (iterations, seconds, max gradient)": traces,
                "fit40_s": round(time.time() - t1, 1), "fit rows": n, "fit distinct rows": len(torch.cat(real)), "fit40 max gradient": g40,
                "gpu": torch.cuda.get_device_properties(0).name}

    ck = Path("/acts") / "matched_d" / prov["stamp"]
    ck.mkdir(parents=True, exist_ok=True)
    fp = {k: prov[k] for k in ("code_sha256", "roster_sha256", "stamp")}
    fits_file, info_file, fp_file = ck / "fits.pt", ck / "fits_info.json", ck / "fingerprint.json"
    if fp_file.exists() and json.loads(fp_file.read_text()) != fp:
        raise ValueError("this checkpoint folder was made by another launch")
    if fits_file.exists() != fp_file.exists() or info_file.exists() != fp_file.exists() or (any(ck.glob("row_*.json")) and not fp_file.exists()):
        raise ValueError("an incomplete checkpoint: not resumed")
    T = ("eval", "sim")
    if fp_file.exists():
        P, fit_info = torch.load(fits_file), json.loads(info_file.read_text()) | {"resumed": True}
    else:  # 1. development passes
        rng = np.random.default_rng(int(L.SEED))
        slot = {t: {l: [] for l in LAYERS} for t in T}
        slot_meta = {t: [] for t in T}  # (task index, 1 if the target's cue else 0 for its matched neutral)
        bg, down, down_meta = {l: [] for l in LAYERS}, {l: [] for l in LAYERS}, []  # down_meta: (task index, eval present, sim present)
        for ti, it in enumerate(s["dev"]):
            vs, _ = L.versions(tok, it["messages"], "dev", it["main_task"])
            for (e, sm), v in zip(L.VERSIONS, vs):
                acts = forward(v["ids"])
                in_slot = set(v["e"]) | set(v["s"])
                others = [i for i in v["eligible"] if i not in in_slot]
                pick = sorted(rng.choice(others, size=min(N_BG, len(others)), replace=False).tolist())
                for l in LAYERS:
                    a = acts[l]
                    slot["eval"][l].append(a[v["e"]].cpu())  # bf16 as read (fp16 could overflow)
                    slot["sim"][l].append(a[v["s"]].cpu())
                    bg[l].append(a[pick].cpu())
                    down[l].append(torch.stack([a[i0:i1].float().mean(0) for i0, i1 in v["calls"]]).cpu())
                slot_meta["eval"] += [(ti, e)] * len(v["e"])
                slot_meta["sim"] += [(ti, sm)] * len(v["s"])
                down_meta += [(ti, e, sm)] * len(v["calls"])
        n_dev, dm = len(s["dev"]), np.array(down_meta)
        rb = np.random.default_rng(20261015)
        coins = {"null": np.random.default_rng(20261014).integers(0, 2, size=(N_NULL, n_dev)),
                 "balanced": np.stack([rb.permutation(np.arange(n_dev) < n_dev // 2) for _ in range(N_NULL)]).astype(int),
                 "independent": np.random.default_rng(20261016).integers(0, 2, size=(N_NULL, n_dev))}
        bgs = {l: torch.cat(bg[l]) for l in LAYERS}
        keep_bg = np.sort(rng.choice(len(bgs[LAYERS[0]]), size=N_FIX, replace=False))  # exactly N_FIX, or the job stops
        P, fit_info = {}, {}
        for t in T:  # 2. fits
            o = "sim" if t == "eval" else "eval"
            sm_t, n_o = np.array(slot_meta[t]), len(slot_meta[o])
            keep_o = np.sort(rng.choice(n_o, size=N_FIX, replace=False))
            n_fix = len(keep_o) + len(keep_bg)

            def labels(sets, kind):
                if kind == "tok":  # a swapped task's cue tokens become negatives and its matched neutral's positives
                    cols = [sm_t[:, 1] == 1] + [np.where(c[sm_t[:, 0]] == 1, sm_t[:, 1] == 0, sm_t[:, 1] == 1) for c in sets]
                    return torch.tensor(np.concatenate([np.stack(cols, 1), np.zeros((n_fix, len(cols)), bool)]), device="cuda")
                y = dm[:, 1 if t == "eval" else 2] == 1  # a swapped task's versions flip, target present with absent
                return torch.tensor(np.stack([y] + [np.where(c[dm[:, 0]] == 1, ~y, y) for c in sets], 1), device="cuda")
            for l in LAYERS:
                for kind in ("tok", "down"):
                    X = (torch.cat([torch.cat(slot[t][l]), torch.cat(slot[o][l])[keep_o], bgs[l][keep_bg]]) if kind == "tok"
                         else torch.cat(down[l])).cuda().double()
                    mu, sd = X.mean(0), X.std(0).clamp_min(1e-6)
                    Z = (X - mu) / sd
                    Ws, gs = [], []
                    for C in CS:
                        W, g = L.fit(Z, labels(coins["null"], kind), C)
                        if kind == "tok" and (l, C) == DIAG:
                            Wd, gd = L.fit(Z, labels(np.concatenate([coins["balanced"], coins["independent"]]), kind)[:, 1:], C)
                            W, g = torch.cat([W, Wd], 1), max(g, gd)
                        Ws.append(W.cpu())
                        gs.append(g)
                    P[(t, l, kind)] = {"mu": mu.cpu(), "sd": sd.cpu(), "W": Ws}
                    fit_info[f"{t}|{l}|{kind}"] = {"rows": int(len(X)), "positives": int(labels([], kind)[:, 0].sum()), "max gradient": max(gs)}
                    del X, Z
        del slot, bg, bgs, down
        torch.save(P, fits_file)
        info_file.write_text(json.dumps(fit_info))
        fp_file.write_text(json.dumps(fp))
        vol.commit()
    fit_s = round(time.time() - t0)
    P = {k: {"mu": p["mu"].cuda(), "sd": p["sd"].cuda(), "W": [w.cuda() for w in p["W"]]} for k, p in P.items()}
    done = 0
    for split in ("sel", "test"):  # 3. selection and test passes
        for it in s[split]:
            vs, place = L.versions(tok, it["messages"], split, it["main_task"])
            for (e, sm), v in zip(L.VERSIONS, vs):
                rid = h16(f"{it['key']}|{e}{sm}")
                if (ck / f"row_{rid}.json").exists():
                    done += 1
                    continue
                acts = forward(v["ids"])
                el = torch.tensor(v["eligible"], device="cuda")
                sc = {}
                for (t, l, kind), p in P.items():
                    if kind == "tok":
                        Z = (acts[l][el].double() - p["mu"]) / p["sd"]
                        sc[f"{t}|{l}|tok"] = [(Z @ W[:-1] + W[-1]).max(0).values.tolist() for W in p["W"]]
                    else:
                        Z = (torch.stack([acts[l][i0:i1].float().mean(0) for i0, i1 in v["calls"]]).double() - p["mu"]) / p["sd"]
                        sc[f"{t}|{l}|down"] = [(Z @ W[:-1] + W[-1]).mean(0).tolist() for W in p["W"]]
                if not all(math.isfinite(x) for vals in sc.values() for col in vals for x in col):
                    raise ValueError("a non-finite score")
                row = {"split": split, "run": h16(it["key"]), "task": it["main_task"], "environment": it["environment"], "placement": place,
                       "eval": e, "sim": sm, "tokens": len(v["ids"]), "scored calls": len(v["calls"]), "scores": sc}
                (ck / f".row_{rid}.tmp").write_text(json.dumps(row))
                (ck / f".row_{rid}.tmp").rename(ck / f"row_{rid}.json")
            vol.commit()
    return {"load_s": load_s, "fits_done_s": fit_s, "resumed rows": done, "seconds": round(time.time() - t0), "fits": fit_info,
            "gpu": torch.cuda.get_device_properties(0).name}


@app.function(image=image, volumes={"/acts": vol}, cpu=2, memory=8192, timeout=900, retries=0)
def rows(stamp: str) -> list:
    vol.reload()
    return [json.loads(f.read_text()) for f in sorted((Path("/acts") / "matched_d" / stamp).glob("row_*.json"))]


if __name__ == "__main__":
    git = lambda *c: subprocess.run(["git", *c], capture_output=True, text=True, cwd=ROOT)
    cache = ROOT / ".run_cache" / "matched_d"
    cache.mkdir(parents=True, exist_ok=True)
    arg = lambda k: sys.argv[sys.argv.index(k) + 1]
    if sys.argv[1] == "launch":
        st = git("status", "--porcelain", "--untracked-files=no")
        if st.returncode or st.stdout.strip() or git("ls-files", "--error-unmatch", __file__, L.__file__).returncode:
            raise SystemExit("commit first: the run must record the commit of the code that made it")
        if sys.argv[2] not in ("dry", "time", "full"):
            raise SystemExit("phase must be dry, time or full")
        stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
        prov = {"git": git("rev-parse", "HEAD").stdout.strip(), "stamp": stamp, "phase": sys.argv[2], "code_sha256": code_sha()}
        if sys.argv[2] == "dry":  # the used tasks exactly as count_d_pool.py built them
            ex = json.loads((ROOT / "experiments/2026-10-10_eval_awareness_probe/results/retrain_frozen_inputs.json").read_text())
            cue = json.loads((ROOT / "experiments/2026-10-12_cue_pilot/results/dry_2026-10-12_150605.json").read_text())["result"]
            call = modal.Function.from_name(APP, "dry").spawn(ex["runs"] + (ex.get("checker runs") or []), [t for v in cue["tasks"].values() for t in v], prov)
        else:
            if "--resume" in sys.argv:
                prov["stamp"] = arg("--resume")
            d = json.loads((HERE / arg("--dry")).read_text())
            if d["provenance"]["code_sha256"] != prov["code_sha256"]:
                raise SystemExit("the dry run was made by other code: run it again")
            r = d["result"]
            if r["counts"]["test"] < 50 or r["counts"]["sel"] < 40 or r["counts"]["dev"] != 200 or min(*r["slot rows"].values(), r["background pool"]) < N_FIX:
                raise SystemExit(f"counts {r['counts']}, slot rows {r['slot rows']}, background pool {r['background pool']}: under RULE.md's floors (test 50, selection 40, development 200, "
                                 f"{N_FIX} other-target slot rows, {N_FIX} background): stop for an amendment")
            prov |= {"roster": r["roster"], "roster_sha256": r["roster_sha256"], "token-max rows": r["token-max rows"], "dry": arg("--dry")}
            if sys.argv[2] == "full":
                tm = json.loads((HERE / arg("--time")).read_text())
                if tm["provenance"]["code_sha256"] != prov["code_sha256"] or tm["provenance"]["roster_sha256"] != prov["roster_sha256"]:
                    raise SystemExit("the timing file was made by other code or another roster: rerun it")
                t = tm["result"]
                warm, cold = [x for x in t["rows"] if x["warm"]], [x for x in t["rows"] if not x["warm"]]
                if not warm or not cold:
                    raise SystemExit("the timing check has no warm or no cold pass: rerun it")
                tps = sum(x["tokens"] for x in warm) / sum(x["seconds"] for x in warm)
                warmup = max(0.0, *(x["seconds"] - x["tokens"] / tps for x in cold))
                if any(not tr[-1][2] < 1e-5 for tr in t["fit traces (iterations, seconds, max gradient)"].values()) or not t["fit40 max gradient"] < 1e-5:
                    raise SystemExit("a slow-case timing fit did not reach 1e-5: bring the amendment question to 0d")
                n_fits = 2 * len(LAYERS) * 2  # 21-column solves per C (target x layer x readout), plus 2 diagnostic 40-column ones
                # passes at the warm rate, a warm-up per length bucket, every solve at the timed slow case, the versions'
                # CPU time twice, two loads, and 30 min for scoring, commits and start-up
                secs = (r["total tokens"] / tps + -(-r["max tokens"] // 4096) * warmup + n_fits * sum(t["fit_s by C"].values()) + 2 * t["fit40_s"]
                        + 2 * r["versions s (all runs once)"] + 2 * t["load_s"] + 1800)
                proj, prior = secs / 3600 * RATE * 1.45, float(arg("--prior"))  # --prior: D's spend so far (dry, time, earlier attempts)
                if not (math.isfinite(prior) and prior >= 0 and math.isfinite(proj) and proj > 0):
                    raise SystemExit("--prior must be a finite, nonnegative dollar amount (and the projection finite)")
                prov |= {"time": arg("--time"), "projected $": round(proj, 2), "prior $": prior}
                if 1.3 * secs > TIMEOUT:
                    raise SystemExit(f"{secs / 3600:.2f} h projected (+30%) does not fit the {TIMEOUT / 3600:.0f} h timeout: re-plan before launching")
                if prior + proj > STOP:
                    raise SystemExit(f"${prior:.2f} spent + ${proj:.2f} projected is over the ${STOP} stop: re-price before launching")
            call = modal.Function.from_name(APP, "read").spawn(sys.argv[2], prov)
        (cache / f"launch_{stamp}.json").write_text(json.dumps({"call": call.object_id, "prov": prov}))
        print(json.dumps({"call": call.object_id, **{k: v for k, v in prov.items() if k != "roster"}}, indent=1))
    elif sys.argv[1] == "collect":
        rec = json.loads((cache / f"launch_{sys.argv[2]}.json").read_text())
        out = {"provenance": rec["prov"], "result": modal.FunctionCall.from_id(rec["call"]).get()}
        if rec["prov"]["phase"] == "full":
            rs = modal.Function.from_name(APP, "rows").remote(rec["prov"]["stamp"])
            f = cache / f"rows_{rec['prov']['stamp']}.json"
            f.write_text(json.dumps(rs))
            out["rows"] = {"file": f.name, "n": len(rs), "sha256": hashlib.sha256(f.read_bytes()).hexdigest()}
        path = HERE / "results" / f"{rec['prov']['phase']}_{sys.argv[2]}.json"
        if path.exists():
            raise FileExistsError(path)
        path.write_text(json.dumps(out, indent=1) + "\n")
        print(json.dumps(out, indent=1)[:3000], f"\n-> {path}")
