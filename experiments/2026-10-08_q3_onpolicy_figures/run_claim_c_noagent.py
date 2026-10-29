"""POST-HOC stripped-input test (plan Amendment 11): claim C's detector retrained on readings with the agent's own reasoning
and chat removed from the input (an off-policy ablation). Deployed Modal app:
- strip_stage (CPU): one staged arm file -> `<staging>_noagent/inputs/<arm>.jsonl`. In every assistant turn, by
  special-token ids and whitespace-only vocabulary ids (no transcript text is decoded): the think block and every
  maximal non-call run holding a non-whitespace token are deleted; malformed turns follow Amendment 11's addition.
  Request lengths and readouts are remapped through the kept mask; every request hash is recomputed. Checks per epoch
  (any failure stops the stage): staged hashes first; nothing deleted outside assistant interiors; every call span kept;
  every remaining readout kept with its token unchanged; turn markers unchanged in number. The 1% readout-loss limit is
  checked per arm, across its files, by the `stage` client before any extraction.
- cache (CPU): every readout (output_end, call_end) at layers 39, 51, 63 of the stripped extractions (run 1, run 2,
  sim, honest_a, honest_b) into /score/claimcna_cache_v1/.
- grid (H100, one per layer and readout): exactly run_claim_c.grid on that cache; records under /score/claimcna_<tag>/.
  `timing`: L51 output_end, C 1e-6, 1e-2 and 1, every column a shuffle, never analyzed.

  modal deploy run_claim_c_noagent.py
  uv run ... python run_claim_c_noagent.py stage | cache | timing <labels> | grid <labels>   (spawns on the deployed app)
"""

import hashlib
import json
import os
import sys
import time
from pathlib import Path

import modal

HERE = Path(__file__).parent
REPO = HERE.parents[1] if len(HERE.parents) > 1 else HERE
APP = "probe-shift_aswin-surya_sub-q3onpol_claimcna_kill-2026-11-10"
SCORE_VOL, ACTS_VOL, HF_CACHE = ("probe-shift_aswin-surya_sub-q3onpol_score_kill-2026-11-08", "probe-shift_aswin-surya_sub-q3onpol_acts_kill-2026-11-08",
                                 "probe-shift_aswin-surya_hf-cache")
CACHE, MAX_DROP = "claimcna_cache_v1", 0.01
T196 = "experiments/2026-10-07_phase2_attack_check/config/pilot_tasks_amendment9_all196.json"
BASES = ["2026-10-09_rerun_strict_segends_v1_scrub1", "2026-10-10_scaleup_canary_segends_v1_scrub1",
         "2026-10-10_scaleup_l2_segends_v1_scrub1", "2026-10-10_scaleup_topup_segends_v1_scrub1"]
SOURCES = ([(f"{s}_swapsys", "attack") for s in ("rerun_strict_segends_v2_scrub1", "scaleup_canary_segends_v1_scrub1", "scaleup_l2_segends_v1_scrub1",
                                                 "scaleup_topup_segends_v1_scrub1")] + [("second_attack_segends_v1_scrub1_swapsys", "attack")]
           + [(f"{s}_swapsys2", "sim_attack@september") for s in ("rerun_strict_segends_v2_scrub1", "scaleup_canary_segends_v1_scrub1",
                                                                   "scaleup_l2_segends_v1_scrub1", "scaleup_topup_segends_v1_scrub1")]
           + [(s, h) for s in ("rerun_strict_segends_v2_scrub1", "scaleup_canary_segends_v1_scrub1", "scaleup_l2_segends_v1_scrub1",
                               "scaleup_topup_segends_v1_scrub1") for h in ("honest_a", "honest_b")])  # (staging, arm) files stripped
RUNS = ([(f"{b}_swapsys_noagent", "attack", "run1") for b in BASES] + [("2026-10-10_second_attack_segends_v1_scrub1_swapsys_noagent", "attack", "run2")]
        + [(f"{b}_swapsys2_noagent", "sim_attack@september", "sim") for b in BASES]
        + [(f"{b}_noagent", h, h) for b in BASES for h in ("honest_a", "honest_b")])  # (stripped extraction run, arm, world)
LAYERS, KINDS, C_GRID = [39, 51, 63], ["output_end", "call_end"], [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]
N_NULL, MAX_ITER, MAX_ITER_CAP = 100, 5000, 80000

app = modal.App(APP)
score_vol, acts_vol = modal.Volume.from_name(SCORE_VOL, version=2), modal.Volume.from_name(ACTS_VOL, version=2)
hf_vol = modal.Volume.from_name(HF_CACHE).read_only()
image = (modal.Image.debian_slim(python_version="3.12").uv_pip_install("torch==2.8.0", "numpy", "scipy", "transformers==5.19.0")
         .env({"HF_HOME": "/hf", "HF_HUB_OFFLINE": "1"})
         .add_local_python_source("probe_shift").add_local_file(REPO / T196, f"/root/{T196}"))


def tol_of(C: float) -> float:
    return 1e-4 * min(C**0.5, 1.0)


def frozen_folds(d: dict) -> dict:  # run_score_onpolicy.frozen_folds (amendment 25), on a parsed task file
    h = lambda s: hashlib.sha256((d["salt"] + s).encode()).hexdigest()
    envs, by_env = sorted(d["eligible_mains_by_env"], key=h), {}
    for c in d["cells"]:
        by_env.setdefault(c["env"], []).append(c["main"])
    return {c["key"]: (by_env[c["env"]].index(c["main"]) + envs.index(c["env"])) % 5 for c in d["cells"]}


@app.function(image=image, cpu=4, memory=32768, timeout=3600, retries=0, volumes={"/acts": acts_vol, "/hf": hf_vol})
def strip_stage(staging: str, arm: str) -> dict:
    import numpy as np
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-27B", revision="fc05daec18b0a78c049392ed2e771dde82bdf654")
    sid = {s: tok.convert_tokens_to_ids(s) for s in ("<|im_start|>", "<|im_end|>", "<think>", "</think>", "<tool_call>", "</tool_call>")}
    if any(v is None or v == tok.unk_token_id for v in sid.values()):
        raise ValueError(f"a template marker is not a single token: {sid}")
    role = tok.encode("assistant", add_special_tokens=False)
    if len(role) != 1:
        raise ValueError("the assistant role is not one token")
    ws = np.array([tok.decode([i]).strip() == "" for i in range(len(tok))])
    sha = lambda ids: hashlib.sha256(np.asarray(ids, np.int32).tobytes()).hexdigest()
    T, Tc, C, Cc = sid["<think>"], sid["</think>"], sid["<tool_call>"], sid["</tool_call>"]
    out_eps, n = [], {"epochs": 0, "tokens_before": 0, "tokens_after": 0, "malformed_no_call": 0, "malformed_kept_spans": 0,
                      "malformed_left_out": 0, "readouts_dropped": {}, "readouts": {}}
    for line in (Path("/acts") / staging / "inputs" / f"{arm}.jsonl").read_text().splitlines():
        ep = json.loads(line)
        ids = np.asarray(ep["ids"])
        for c in ep["calls"]:
            if sha(ids[: c["n"]]) != c["sha256"]:
                raise ValueError(f"{ep['key']}: the staged ids differ from the staged hash")
        keep, interior, spans = np.ones(len(ids), bool), np.zeros(len(ids), bool), np.zeros(len(ids), bool)
        rd = np.array([c["readout"] for c in ep["calls"] if c["readout"] is not None], int)
        drop_rd = set()
        for s in np.flatnonzero(ids == sid["<|im_start|>"]):
            ends = np.flatnonzero(ids[s + 1:] == sid["<|im_end|>"])
            e = s + 1 + (ends[0] if len(ends) else len(ids) - s - 1)
            if s + 1 >= e or ids[s + 1] != role[0]:
                continue
            a0 = s + 2  # the interior after the role token
            t = ids[a0:e]
            interior[a0:e] = True
            state, bad = None, False
            for v in t[np.isin(t, [T, Tc, C, Cc])]:
                if v in (T, C) and state is None:
                    state = v
                elif (state, v) in ((T, Tc), (C, Cc)):
                    state = None
                else:
                    bad = True
            bad = bad or state is not None
            if not bad:
                in_think = np.cumsum((t == T).astype(int) - np.r_[0, (t == Tc)[:-1]].astype(int)) > 0  # tags included
                in_call = np.cumsum((t == C).astype(int) - np.r_[0, (t == Cc)[:-1]].astype(int)) > 0
                drop = in_think.copy()
                other = ~in_think & ~in_call
                i = 0
                while i < len(t):  # each maximal run of other tokens holding a non-whitespace token is chat: deleted
                    if other[i]:
                        j = i
                        while j < len(t) and other[j]:
                            j += 1
                        if (~ws[t[i:j]]).any():
                            drop[i:j] = True
                        i = j
                    else:
                        i += 1
                call = in_call
            else:  # Amendment 11's addition: spans pair each </tool_call> with the nearest unpaired <tool_call> before it
                call, opened = np.zeros(len(t), bool), []
                for i in np.flatnonzero(np.isin(t, [C, Cc])):
                    if t[i] == C:
                        opened.append(i)
                    elif opened:
                        call[opened.pop():i + 1] = True
                inside = rd[(rd >= a0) & (rd < e)] - a0
                if not np.isin(t, [C, Cc]).any():
                    n["malformed_no_call"] += 1
                    drop = np.ones(len(t), bool)
                elif call.any() and call[inside].all():
                    n["malformed_kept_spans"] += 1
                    drop = ~call
                else:
                    n["malformed_left_out"] += 1
                    drop, call = np.ones(len(t), bool), np.zeros(len(t), bool)
                    drop_rd |= set((inside + a0).tolist())
            keep[a0:e] = ~drop
            spans[a0:e] = call
        if (~keep & ~interior).any() or not keep[spans].all():
            raise ValueError(f"{ep['key']}: a deletion outside an assistant interior, or inside a call span")
        new = ids[keep]
        if (new == sid["<|im_start|>"]).sum() != (ids == sid["<|im_start|>"]).sum() or (new == sid["<|im_end|>"]).sum() != (ids == sid["<|im_end|>"]).sum():
            raise ValueError(f"{ep['key']}: the turn markers changed")
        pos = np.cumsum(keep) - 1  # the new position of each kept token
        calls = []
        for c in ep["calls"]:
            c = dict(c)
            kind = c.get("kind", "command")
            if c["readout"] is not None:
                n["readouts"][kind] = n["readouts"].get(kind, 0) + 1
                if c["readout"] in drop_rd:
                    n["readouts_dropped"][kind] = n["readouts_dropped"].get(kind, 0) + 1
                    c["readout"] = None
                else:
                    if not keep[c["readout"]] or new[pos[c["readout"]]] != ids[c["readout"]]:
                        raise ValueError(f"{ep['key']}: a readout was deleted or its token changed")
                    c["readout"] = int(pos[c["readout"]])
            c["n"] = int(keep[: c["n"]].sum())
            c["sha256"] = sha(new[: c["n"]])
            calls.append(c)
        n["epochs"] += 1
        n["tokens_before"] += len(ids)
        n["tokens_after"] += len(new)
        out_eps.append(ep | {"ids": new.tolist(), "calls": calls, "strip": {"deleted": int(len(ids) - len(new))}})
    d = Path("/acts") / f"{staging}_noagent" / "inputs"
    d.mkdir(parents=True, exist_ok=True)
    f = d / f"{arm}.jsonl"
    if f.exists():
        raise FileExistsError(f"{f} exists")
    b = "".join(json.dumps(e) + "\n" for e in out_eps).encode()
    f.write_bytes(b)
    acts_vol.commit()
    return {"staging": staging, "arm": arm, "sha256": hashlib.sha256(b).hexdigest()} | n


@app.function(image=image, cpu=8, memory=65536, timeout=3600, retries=0, volumes={"/score": score_vol, "/acts": acts_vol.read_only()})
def cache(layer: int) -> dict:
    import numpy as np

    score_vol.reload()
    acts_vol.reload()
    d = Path("/score") / CACHE
    d.mkdir(exist_ok=True)
    rows, parts, counted = [], [], {}
    for run, arm, world in RUNS:
        root = Path("/acts") / run
        done = json.loads((root / "complete.json").read_text())
        if len(list((root / "meta").glob("*/*.json"))) != done["epochs"]:  # every arm of the run (honest runs hold two)
            raise ValueError(f"{run}: the extracted epochs differ from the staged count {done['epochs']}")
        metas = [json.loads(f.read_text()) for f in sorted((root / "meta" / arm).glob("*.json"))]
        picks = []
        for m in metas:
            if m.get("redrawn_because") or m.get("attempt_kind") == "rerun":
                continue
            counted.setdefault((world, m["task"]), set()).add(m["run"])
            for c in m["calls"]:
                if c.get("kind") in KINDS and c.get("row") is not None:
                    rows.append({"world": world, "task": m["task"], "kind": c["kind"], "order": [c["turn"], c.get("call_index", 0)]})
                    picks.append((m["chunk"], m["row_start"] + c["row"]))
        X = np.empty((len(picks), 5120), np.float16)
        for chunk in sorted({p[0] for p in picks}):
            a = np.load(root / "by_layer" / f"L{layer}" / f"{chunk}.npy")
            idx = [i for i, p in enumerate(picks) if p[0] == chunk]
            X[idx] = a[[picks[i][1] for i in idx]]
        parts.append(X)
    if any(len(v) > 1 for v in counted.values()):
        raise ValueError("several counted runs for one world and task")
    X = np.concatenate(parts)
    if len(X) != len(rows) or not np.isfinite(X.astype(np.float32)).all():
        raise ValueError("rows and features disagree, or non-finite activations")
    np.save(d / f"on_L{layer}.npy", X)
    (d / f"on_rows.L{layer}.json").write_text(json.dumps(rows))
    score_vol.commit()
    return {"layer": layer, "rows": len(rows), "by_world": {w: sum(r["world"] == w for r in rows) for w in ("run1", "run2", "sim", "honest_a", "honest_b")}}


@app.function(image=image, gpu="H100!", cpu=8, memory=65536, timeout=2 * 3600, retries=0, volumes={"/score": score_vol}, scaledown_window=5)
def grid(tag: str, layer: int, kind: str, attempted: list, timing: bool = False) -> dict:
    import numpy as np
    import torch
    import torch.nn.functional as F
    from scipy.stats import rankdata

    from probe_shift import torch_probe as tp

    score_vol.reload()
    fold_of = frozen_folds(json.loads((Path("/root") / T196).read_text()))
    att, t0, out = set(attempted), time.time(), {}
    d = Path("/score") / CACHE
    rows = json.loads((d / f"on_rows.L{layer}.json").read_text())
    kn = np.flatnonzero([r["kind"] == kind for r in rows])
    X = torch.from_numpy(np.load(d / f"on_L{layer}.npy")[kn]).cuda()
    runs, n_run = np.unique([f"{r['world']}|{r['task']}" for r in rows], return_inverse=True)
    n_run = n_run[kn]
    world = np.array([u.split("|", 1)[0] for u in runs])
    task = np.array([u.split("|", 1)[1] for u in runs])
    r_fold = np.array([fold_of[t] for t in task])
    train = (world == "honest_a") | ((world == "run1") & np.isin(runs, sorted(att)))
    y = (world == "run1").astype(float)
    Y = np.zeros((len(runs), 1 + N_NULL))
    Y[:, 0] = y
    for c in range(0 if timing else 1, 1 + N_NULL):  # timing: column 0 is a shuffle too (no real labels)
        rng = np.random.default_rng([20261010, c + (1000 if timing and c == 0 else 0)])
        yc = y.copy()
        for t in sorted(set(task[train])):
            j = np.flatnonzero(train & (task == t))
            yc[j] = y[rng.permutation(j)]
        Y[:, c] = yc
    K = Y.shape[1]
    for C in ([1e-6, 1e-2, 1.0] if timing else C_GRID):
        p = Path("/score") / f"claimcna_{tag}{'_TIMING' if timing else ''}" / f"L{layer}_{kind}_C{C:g}.npz"
        if p.exists():
            continue
        r = {"val_auroc": np.full((5, K), np.nan), "grad_full": np.zeros((5, K)), "grad_val": np.zeros((5, K)),
             "evals": np.zeros((5, 2), int), "on_score": np.full((len(runs), K), np.nan)}
        for f in range(5):
            v = (f + 1) % 5
            for name, tr in (("full", train & (r_fold != f)), ("val", train & (r_fold != f) & (r_fold != v))):
                a = np.flatnonzero(tr[n_run])
                b_ = np.flatnonzero(r_fold[n_run] == f) if name == "full" else np.flatnonzero((train & (r_fold == v))[n_run])
                Xa, Xb = tp.standardize(X[a].double(), X[b_].double())
                Xa.mul_(C**0.5), Xb.mul_(C**0.5)
                Ya = torch.from_numpy(Y[n_run[a]]).cuda()
                mi, prev = MAX_ITER, None
                while True:  # refit only fits that used all their iterations; stop when a refit repeats
                    W, b, evals = tp.fit(Xa, Ya, 1.0, mi, tol=tol_of(C))
                    Wg, bg = W.clone().requires_grad_(True), b.clone().requires_grad_(True)
                    loss = (F.binary_cross_entropy_with_logits(Xa @ Wg + bg, Ya, reduction="sum") + 0.5 * (Wg * Wg).sum()) / len(Xa)
                    loss.backward()
                    g = torch.maximum(Wg.grad.abs().amax(0), bg.grad.abs()).cpu().numpy()
                    if g.max() <= tol_of(C) or evals < mi or evals == prev or mi * 4 > MAX_ITER_CAP:
                        break
                    mi, prev = mi * 4, evals
                r[f"grad_{name}"][f], r["evals"][f, int(name == "val")] = g, evals
                S = tp.decision(Xb, W, b).cpu().numpy()
                u, inv = np.unique(n_run[b_], return_inverse=True)
                M = np.stack([np.bincount(inv, weights=S[:, c]) for c in range(K)], 1) / np.bincount(inv)[:, None]
                if name == "val":
                    R, Yv = rankdata(M, axis=0), Y[u]
                    n1 = Yv.sum(0)
                    r["val_auroc"][f] = ((R * Yv).sum(0) - n1 * (n1 + 1) / 2) / (n1 * (len(Yv) - n1))
                else:
                    r["on_score"][u] = M
                del Xa, Xb
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p.with_suffix(".tmp"), "wb") as fh:
            np.savez(fh, on_runs=runs, on_fold=r_fold, train=train, Y=Y, tol=tol_of(C), **r)
        os.replace(p.with_suffix(".tmp"), p)
        score_vol.commit()
        out[f"{C:g}"] = round(time.time() - t0)
        print(f"L{layer} {kind} C {C:g} done at {out[f'{C:g}']} s; evals max {r['evals'].max()}", flush=True)
    return out


def attempted_run1(labels: str) -> list:
    att = []
    for f in labels.split(","):  # run 1 is every label folder's attack arm except the second batch's
        if "second_attack" in Path(f).name:
            continue
        att += [f"run1|{r['task']}" for r in map(json.loads, Path(f).read_text().splitlines()) if r["arm"] == "attack" and r.get("label") == "attempted"]
    if len(att) != len(set(att)):
        raise ValueError("a run-1 task is labelled twice")
    return sorted(att)


if __name__ == "__main__":  # spawns on the deployed app
    cmd = sys.argv[1]
    if cmd not in ("stage", "cache", "timing", "grid") or (cmd in ("timing", "grid") and len(sys.argv) != 3):
        raise SystemExit("usage: run_claim_c_noagent.py stage | cache | timing <labels> | grid <labels>")
    if cmd == "stage":
        calls = [modal.Function.from_name(APP, "strip_stage").spawn(s, a) for s, a in SOURCES]
        res = [c.get() for c in calls]
        world = lambda r: {"attack": "run2" if r["staging"].startswith("second_attack") else "run1", "sim_attack@september": "sim"}.get(r["arm"], r["arm"])
        arms = {}
        for r in res:  # Amendment 11's 1% limit, per arm across its files: checked here, before any extraction
            a = arms.setdefault(world(r), {"readouts": {}, "readouts_dropped": {}})
            for k in ("readouts", "readouts_dropped"):
                for kind, v in r[k].items():
                    a[k][kind] = a[k].get(kind, 0) + v
        for w, a in arms.items():
            a["share_dropped"] = {kind: a["readouts_dropped"].get(kind, 0) / a["readouts"][kind] for kind in ("call_end", "output_end") if a["readouts"].get(kind)}
            a["stop"] = any(v > MAX_DROP for v in a["share_dropped"].values())
        print(json.dumps({"files": res, "arms": arms, "STOP_before_extraction": any(a["stop"] for a in arms.values())}, indent=1))
    elif cmd == "cache":
        print(json.dumps({l: modal.Function.from_name(APP, "cache").spawn(l).object_id for l in LAYERS}))
    else:
        att = attempted_run1(sys.argv[2])
        tag = hashlib.sha256((HERE / "run_claim_c_noagent.py").read_bytes() + json.dumps(att).encode()).hexdigest()[:12]
        fn = modal.Function.from_name(APP, "grid")
        jobs = [(51, "output_end", True)] if cmd == "timing" else [(l, k, False) for l in LAYERS for k in KINDS]
        print(json.dumps({"tag": tag, "attempted_run1": len(att), "calls": {f"L{l}_{k}": fn.spawn(tag, l, k, att, t).object_id for l, k, t in jobs}}))
