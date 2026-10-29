"""q2's review-note swap on the Qwen3.5 reader (2026-10-08): score the same 100 simulated runs, under each insert-output
variant, with frozen Qwen3.5 Q2 pre_call probes.

q2's design (PR #88, experiments/2026-10-07_q2_rendering_swap): only the simulator's successful insert outputs change.
- V1: as simulated (`sim-attack`).
- V2: re-rendered as the real tool's `cat -n` snippet plus the review note (`sim-attack-realstyle`).
- V3: V2 without the note (`sim-attack-snippet-nonote`).
- V4: the diff with the note appended (`sim-attack-diff-note`).
- V5: the diff without its trailing newline (`sim-attack-diff-nonewline`).
Staged runs: q2's, on the shared Volume, read only.

Two steps on Modal, per readout position (--positions; default pre_call, the `<tool_call>` template token; its
content-token controls are prev_content and call_end, the readouts Q2 defines in run_q2_alllayers_qwen35.positions_of):
  1. refit (A10G): the matched level-0 attack-only probes trained on real-attack-e1, one per outer fold at that
     fold's 64-layer headline pick, refit exactly as run_q2_alllayers_qwen35.py fit them (same packs, standardization,
     C grid, strict-maximum C choice, 1% threshold on validation), now keeping the weights. Also checks the refit test
     scores against the committed fits.
  2. score (H200): each variant transcript read by Qwen3.5 (pinned revision, bf16, sdpa, batch 1, 4,096-token buckets,
     the reader's segment-end positions), the forward stopped after the last needed layer, and every call's readouts
     scored on the GPU with its own fold's frozen probes. Returns scores only.
The analysis (closure, matched-FPR recall, frozen-threshold FPR) is analyze_q2_review_note_qwen35.py.

  uv run modal run experiments/2026-10-06_qwen35_replication/run_q2_review_note_qwen35.py [--positions prev_content,call_end]
"""

import gzip
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).parent
ROOT = HERE.parent.parent
APP = "probe-shift_aswin-surya_sub-qwen35_review-note_kill-2026-11-06"
FEATS = "probe-shift_aswin-surya_sub-qwen35_q2feats_kill-2026-11-06"
INPUTS = "probe-shift_aswin-surya_sub-la-activations"  # q2's staged swap runs (read only)
FR = "2026-10-06_q2_qwen35_matched_eb014307fd1aab31_as_labeled"  # the committed matched features run
COMMITTED = HERE / "results" / "q2_alllayers_q2_alllayers_qwen35_matched_2026-10-06_235812.json"
STAGED = {"2026-10-07_q2_swap_inputs_v1": ["sim-attack", "sim-attack-realstyle"],
          "2026-10-07_q2_swap_note_inputs_v1": ["sim-attack-snippet-nonote", "sim-attack-diff-note"],
          "2026-10-07_q2_swap_v5_inputs_v1": ["sim-attack-diff-nonewline"]}
TRAIN, SHARDS = "real-attack-e1", 3
PIP = ["torch==2.8.0", "transformers==5.19.0", "accelerate>=1.1.0", "numpy", "huggingface_hub", "flash-linear-attention==0.5.2"]

app = modal.App(APP)
feats_vol = modal.Volume.from_name(FEATS, version=2).read_only()
ckpt_vol = modal.Volume.from_name(FEATS, version=2)  # the same Volume, writable, for scoring checkpoints only
CKPT = f"{FR}/review_note_qwen35"  # <stamp>/shard<s>/batch<n>.json: scored transcripts, 20 per file, each renamed into place
in_vol = modal.Volume.from_name(INPUTS, version=2).read_only()
hf_cache = modal.Volume.from_name("probe-shift_aswin-surya_hf-cache")
fit_image = (modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy", "scikit-learn==1.7.2", "torch==2.8.0")
             .add_local_python_source("probe_shift"))
gpu_image = (modal.Image.debian_slim(python_version="3.12").uv_pip_install(*PIP).env({"HF_HOME": "/hf", "HF_HUB_OFFLINE": "1"})
             .add_local_python_source("probe_shift"))


@app.function(image=fit_image, gpu="A10G", volumes={"/feats": feats_vol}, memory=16384, timeout=1800)
def refit(cfg: dict, layers_by_fold: dict, position: str) -> dict:
    """Per outer fold f: the probe at layer layers_by_fold[f], fit as run_q2_alllayers_qwen35.fit did (validation fold
    f+1 mod 5, C by strict maximum of validation AUROC, threshold at cfg['fpr'] on validation negatives)."""
    import platform

    import numpy as np
    import torch
    from sklearn.metrics import roc_auc_score

    from probe_shift.probes import fit_logreg
    from probe_shift.transfer import threshold_at_fpr

    d = Path("/feats") / FR
    man = json.loads((d / "manifest.json").read_text())
    y, fold, arm = (np.array([m[k] for m in man]) for k in ("y", "fold", "arm"))
    in_train = arm == TRAIN
    test_dom = np.isin(arm, cfg["test_domains"])
    out = {"folds": {}, "scores": {}, "gpu": torch.cuda.get_device_properties(0).name, "host": platform.node()}
    for f in range(5):
        L = layers_by_fold[str(f)]
        X = torch.from_numpy(np.load(d / "alllayers" / f"L{L}" / f"{position}.npy")).cuda().float()
        v = (f + 1) % 5
        train, val, test = (np.flatnonzero(in_train & (fold != f) & (fold != v)), np.flatnonzero(in_train & (fold == v)),
                            np.flatnonzero(fold == f))
        mu, sd = X[train].mean(0), X[train].std(0).clamp_min(1e-6)
        Z = (X - mu) / sd
        ytr = torch.tensor(y[train], dtype=torch.float32, device="cuda")
        best = None
        for C in cfg["C_grid"]:
            w, g = fit_logreg(Z[train], ytr, C, cfg["max_iter"])
            s = (Z @ w[:-1] + w[-1]).cpu().numpy()
            auc = float(roc_auc_score(y[val], s[val]))
            if best is None or auc > best[0]:  # the committed fits' rule (strict maximum: the first C on ties)
                best = (auc, C, w, s)
        s = best[3]
        t = threshold_at_fpr(s[val][y[val] == 0], cfg["fpr"])
        committed = np.load(d / "alllayers" / "fits" / f"{position}__{TRAIN}__L{L}__real.npz")
        rows = np.flatnonzero(test_dom)  # the committed npz holds the test-domain rows, in manifest order
        in_f = fold[rows] == f
        out["folds"][str(f)] = {"layer": L, "C": best[1], "val_auroc": best[0], "threshold": float(t),
                                "w": best[2][:-1].cpu().numpy().tolist(), "b": float(best[2][-1]),
                                "mu": mu.cpu().numpy().tolist(), "sd": sd.cpu().numpy().tolist(),
                                "repro_max_abs_score_diff": float(np.abs(s[rows][in_f] - committed["score"][in_f]).max()),
                                "repro_threshold_diff": float(abs(t - float(committed["threshold"][in_f][0])))}
        for i in test:
            out["scores"][f"{man[i]['arm']}|{man[i]['key']}|{man[i]['step']}|{man[i]['action']}"] = float(s[i])
        del X, Z
        torch.cuda.empty_cache()
    return out


@app.function(image=gpu_image, gpu="H200", volumes={"/in": in_vol, "/hf": hf_cache, "/ckpt": ckpt_vol}, memory=65536, timeout=3 * 3600)
def score(cfg: dict, shard: int, probes: dict, fold_of: dict, stamp: str) -> dict:
    """Every call of this shard's variant transcripts (a kept pre_call), its readouts at each position in `probes`
    ({position: {fold: probe}}) scored with its own fold's frozen probe. Every 20 scored
    transcripts are written to /ckpt/<CKPT>/<stamp>/shard<s>/batch<n>.json (a temporary file renamed into place, so a
    batch is whole or absent); a rerun with the same stamp skips the transcripts already saved."""
    import os
    from concurrent.futures import ThreadPoolExecutor

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from probe_shift.models import QWEN35_27B
    from probe_shift.reader import segment_end_positions

    items = []
    for run, arms in STAGED.items():
        for line in (Path("/in") / run / "inputs" / "shard_0.jsonl").read_text().splitlines():
            it = json.loads(line)
            if it["arm"] in arms:
                items.append(it)
    items.sort(key=lambda it: (-it["tokens"], it["arm"], it["key"]))  # contiguous length blocks: few tuning buckets each
    total, before, mine = sum(it["tokens"] for it in items), 0, []
    for it in items:
        if min(before * SHARDS // total, SHARDS - 1) == shard:
            mine.append(it)
        before += it["tokens"]
    ckpt = Path("/ckpt") / CKPT / stamp / f"shard{shard}"
    ckpt.mkdir(parents=True, exist_ok=True)
    batches = sorted(ckpt.glob("batch*.json"))  # only renamed files: a write cut short leaves a .tmp, never read
    done = [d for b in batches for d in json.loads(b.read_text())]
    seen = {(d["arm"], d["key"]) for d in done}
    mine = [it for it in mine if (it["arm"], it["key"]) not in seen]
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(QWEN35_27B["name"], revision=QWEN35_27B["revision"])
    lm = AutoModelForCausalLM.from_pretrained(QWEN35_27B["name"], revision=QWEN35_27B["revision"], dtype=torch.bfloat16,
                                              device_map="cuda", attn_implementation="sdpa").eval()
    load_s = time.time() - t0
    layers = sorted({p["layer"] for per in probes.values() for p in per.values()})
    W = {pos: {f: {k: torch.tensor(p[k], device="cuda", dtype=torch.float32) for k in ("w", "mu", "sd")} | {"b": p["b"], "layer": p["layer"]}
               for f, p in per.items()} for pos, per in probes.items()}

    def readouts(r):
        """(position, step, action, token) of every call's kept readouts, as run_q2_alllayers_qwen35.positions_of."""
        at = {(q["kind"], q["step"], q["action"]): q for q in r["positions"]}
        last_output = {}
        for q in r["positions"]:
            if q["kind"] == "output_end":
                last_output[q["step"]] = max(last_output.get(q["step"], -1), q["action"])
        out = []
        for c in (q for q in r["positions"] if q["kind"] == "pre_call" and q["kept"]):
            s, a = c["step"], c["action"]
            for pos in probes:
                q = (at.get(("call_end", s, a - 1)) if a > 0 else at.get(("output_end", s - 1, last_output[s - 1])) if s - 1 in last_output
                     else None) if pos == "prev_content" else at.get((pos, s, a))
                if q is not None and q["kept"]:
                    out.append((pos, s, a, q["token"]))
        return out
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    class Stop(Exception):
        pass

    out, stats, pending = [r for d in done for r in d["rows"]], {"transcripts": 0, "tokens": 0, "forward_s": 0.0, "resumed": len(done)}, []

    def save():
        n = len(list(ckpt.glob("batch*.json")))
        tmp = ckpt / f"batch{n:05d}.json.tmp"
        tmp.write_text(json.dumps(pending))
        os.replace(tmp, ckpt / f"batch{n:05d}.json")
        pending.clear()
        ckpt_vol.commit()
    positions = lambda it: segment_end_positions(tok, it["messages"], None)
    with ThreadPoolExecutor(1) as pool:  # the next transcript's tokens and positions while the GPU reads this one
        ahead = pool.submit(positions, mine[0]) if mine else None
        for n, it in enumerate(mine):
            r = ahead.result()
            ahead = pool.submit(positions, mine[n + 1]) if n + 1 < len(mine) else None
            reads = readouts(r)
            idx = torch.tensor([t for *_, t in reads], device="cuda", dtype=torch.long)
            acts = {}

            def hook_for(layer):
                def hook(module, args, output):
                    h = output[0] if isinstance(output, tuple) else output
                    acts[layer] = h[0].index_select(0, idx).to(torch.float16).float()  # fp16 like the packs, then float32
                    if layer == layers[-1]:
                        raise Stop  # nothing after the last needed layer is read
                return hook

            handles = [lm.model.layers[l].register_forward_hook(hook_for(l)) for l in layers]
            ids = r["ids"] + [pad] * (-len(r["ids"]) % 4096)  # the extraction's 4,096-token buckets
            t1 = time.time()
            try:
                with torch.inference_mode():
                    lm.model(input_ids=torch.tensor([ids], device="cuda"), use_cache=False)
            except Stop:
                pass
            finally:
                for h in handles:
                    h.remove()
            torch.cuda.synchronize()
            stats["forward_s"] += time.time() - t1
            f = str(fold_of[it["key"]])
            rows = []
            for pos in probes:
                p, j = W[pos][f], [i for i, x in enumerate(reads) if x[0] == pos]
                with torch.inference_mode():
                    s = ((((acts[p["layer"]][j] - p["mu"]) / p["sd"]) @ p["w"]) + p["b"]).cpu().tolist()
                rows += [{"arm": it["arm"], "key": it["key"], "position": pos, "step": reads[i][1], "action": reads[i][2], "fold": int(f),
                          "layer": p["layer"], "score": x} for i, x in zip(j, s)]
            out += rows
            pending.append({"arm": it["arm"], "key": it["key"], "rows": rows})
            if len(pending) == 20:
                save()
            stats["transcripts"] += 1
            stats["tokens"] += len(r["ids"])
    if pending:
        save()
    props = torch.cuda.get_device_properties(0)
    return {"rows": out, "stats": stats | {"load_s": round(load_s, 1), "gpu": props.name, "shard": shard,
                                           "tokens_per_s": round(stats["tokens"] / max(stats["forward_s"], 1e-9))}}


@app.local_entrypoint()
def main(resume: str = "", positions: str = "pre_call"):
    """--resume <stamp>: reuse that run's saved refit and the shards' scoring checkpoints (pass the same --positions)."""
    committed = json.loads(COMMITTED.read_text())
    cfg = committed["config"]
    positions = positions.split(",")
    layers_by_fold = {pos: {str(p["fold"]): p["layer"] for p in committed["positions"][pos]["headline_layers"][TRAIN]} for pos in positions}
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True, cwd=ROOT).stdout.strip())
    attempt = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")  # names this invocation's result
    stamp = resume or attempt  # names the run: its refit and scoring checkpoints, shared by every resume
    cache = ROOT / ".run_cache" / "q2_review_note"  # probe weights and per-shard scores, saved as they arrive
    cache.mkdir(parents=True, exist_ok=True)
    saved = cache / f"refit_{stamp}.json"
    if resume:
        fits = json.loads(saved.read_text())  # the run's own probes, so resumed shards score with the same weights
        if "folds" in fits:  # saved before --positions existed: that run (2026-10-08_011223) completed and is not resumable
            raise SystemExit(f"run {stamp} predates --positions and cannot be resumed")
        if sorted(fits) != sorted(positions):
            raise SystemExit(f"run {stamp} refit {sorted(fits)}, not --positions {sorted(positions)}")
    else:
        fits = dict(zip(positions, list(refit.starmap([(cfg, layers_by_fold[pos], pos) for pos in positions]))))  # list: drain the
        # generator, or zip leaves it open and Modal's client prints an aclose() RuntimeError on exit
        saved.write_text(json.dumps(fits))  # before any scoring: a failed shard loses nothing else
    for pos, fit in fits.items():
        print("refit", pos, {f: {k: v[k] for k in ("layer", "C", "repro_max_abs_score_diff", "repro_threshold_diff")} for f, v in fit["folds"].items()},
              "on", fit["gpu"], flush=True)
    man = json.loads(b"".join(feats_vol.read_file(f"{FR}/manifest.json")))
    fold_of = {m["key"]: m["fold"] for m in man if m["arm"] == "sim-attack"}
    probes = {pos: {f: {k: v[k] for k in ("w", "mu", "sd", "b", "layer")} for f, v in fit["folds"].items()} for pos, fit in fits.items()}
    parts, failed = [], []
    for part in score.starmap([(cfg, s, probes, fold_of, stamp) for s in range(SHARDS)], order_outputs=False, return_exceptions=True):
        if isinstance(part, BaseException):
            failed.append(repr(part)[:300])
            print("shard failed:", failed[-1], flush=True)
            continue
        (cache / f"shard{part['stats']['shard']}_{stamp}_{attempt}.json").write_text(json.dumps(part))
        parts.append(part)
        print("shard", part["stats"], flush=True)
    path = HERE / "results" / f"q2_review_note_scores_qwen35_{attempt}.json.gz"  # a new file per invocation (append-only)
    path.write_bytes(gzip.compress(json.dumps({"git": git, "dirty": dirty, "run": stamp, "resumed": bool(resume), "features_run": FR, "train": TRAIN, "positions": positions,
                                "layers_by_fold": layers_by_fold, "staged": STAGED, "weights": str((cache / f"refit_{stamp}.json").relative_to(ROOT)),
                                "refit": {pos: {f: {k: v[k] for k in ("layer", "C", "val_auroc", "threshold", "repro_max_abs_score_diff",
                                                                      "repro_threshold_diff")} for f, v in fit["folds"].items()} for pos, fit in fits.items()},
                                "refit_gpu": {pos: fit["gpu"] for pos, fit in fits.items()}, "real_scores": {pos: fit["scores"] for pos, fit in fits.items()}, "shards": [p["stats"] for p in parts],
                                "failed_shards": failed, "rows": [r for p in parts for r in p["rows"]]}).encode()))
    if failed:
        raise SystemExit(f"{len(failed)} shard(s) failed; the others are saved in {cache} and {path}")
    print(f"-> {path}")
