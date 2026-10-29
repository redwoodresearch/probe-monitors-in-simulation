"""The aimed slip-vs-no-slip probe (PLAN.md here, committed before any number): planted check 2's sites, read by
Qwen3.5-27B and, beside it, Qwen3-32B, at layers 16, 32, 48 and 63. One Modal GPU container reads the stored
activations read-only and returns counts and AUROCs only (no text, no activation leaves it).

  PYTHONPATH=src uv run --no-project --python 3.12 --with "modal==1.6.1" modal run \\
      experiments/2026-10-12_q2_aimed_slip_probe/run_aimed_slip_probe.py
"""
import json
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).parent
APP = "probe-shift_aswin-surya_sub-q2-aimslip_probe_kill-2026-11-12"
LA, CB = "probe-shift_aswin-surya_sub-la-activations", "probe-shift_aswin-surya_sub-claimb-reader_acts_kill-2026-12-10"
INPUTS_A, INPUTS_B = "2026-10-10_la_fixed-sim-v2_probe-inputs-l1_v1", "2026-10-10_la_fixed-sim-v2_probe-inputs-b-l1_v1"
READERS = {"Qwen3.5-27B": ("/cb", "2026-10-10_la_fixed-sim-v2-b-scrub1_qwen35-27b_alllayers_v1"),
           "Qwen3-32B (positive control, beside)": ("/la", "2026-10-10_la_fixed-sim-v2-scrub1_qwen3-32b_alllayers_v1")}
TWIN, PLANT = "sim-attack-fixed", "sim-attack-fixed-plant2"
LAYERS, C, MAX_ITER, N_NULL, SEED = (16, 32, 48, 63), 0.01, 1000, 200, 20261012

app = modal.App(APP)
la = modal.Volume.from_name(LA, version=2).read_only()
cb = modal.Volume.from_name(CB, version=2).read_only()
image = (modal.Image.debian_slim(python_version="3.12").uv_pip_install("torch==2.8.0", "numpy")
         .add_local_python_source("probe_shift"))


def calls_of(messages):
    """fixes.calls_of's pairing (../2026-10-09_fixed_sim/fixes.py), args dropped: each tool output with its call's step
    and action, paired by order."""
    out, step, pending = [], -1, []
    for i, m in enumerate(messages):
        if m["role"] == "assistant":
            step += 1
            pending = [(step, j) for j, _ in enumerate(m.get("tool_calls") or [])]
        elif m["role"] == "tool" and pending:
            s, j = pending.pop(0)
            out.append({"step": s, "action": j, "msg": i})
    return out


def text(m):
    c = m.get("content")
    return c if isinstance(c, str) else json.dumps(c)


def auroc(pos, neg):
    import numpy as np
    s = np.r_[pos, neg]; r = s.argsort(kind="mergesort").argsort(kind="mergesort").astype(float) + 1
    u, inv = np.unique(s, return_inverse=True); rr = np.zeros(len(u)); np.add.at(rr, inv, r); cnt = np.bincount(inv); r = (rr / cnt)[inv]
    return float((r[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


@app.function(image=image, gpu="A10G", cpu=8, memory=32768, volumes={"/la": la, "/cb": cb}, timeout=3600, retries=0)
def run() -> dict:
    try:
        return _run()
    except Exception as e:  # nothing from the exception (which may quote staged text) crosses the boundary
        raise RuntimeError(f"aimed probe failed: {type(e).__name__}") from None


def _run():
    from collections import Counter, defaultdict
    from concurrent.futures import ThreadPoolExecutor

    import numpy as np
    import torch

    from probe_shift import torch_probe as tp

    def staged(run):
        return {(it["arm"], it["key"]): it for f in sorted(Path(f"/la/{run}/inputs").glob("shard_*.jsonl"))
                for it in map(json.loads, f.read_text().splitlines()) if it["arm"] in (TWIN, PLANT)}
    A, Bst = staged(INPUTS_A), staged(INPUTS_B)
    man = json.loads(Path(f"/la/{INPUTS_A}/manifest.json").read_text())
    sites = man["plant2"]["sites"]
    counts = Counter({"pairs in the manifest": len(sites)})
    # step B copied step A's planted copies and twins unchanged: checked here, for the 150 pairs
    for s in sites:
        for arm, key in ((TWIN, s["key"]), (PLANT, s["key"] + "__plant2")):
            counts["copies identical in step A's and step B's inputs"] += A[(arm, key)]["messages"] == Bst[(arm, key)]["messages"]
    # sites: (pair, step, action), typed by text
    site_rows = []
    for p, s in enumerate(sites):
        tw, pl = A[(TWIN, s["key"])], A[(PLANT, s["key"] + "__plant2")]
        sa = {c["msg"]: (c["step"], c["action"]) for c in calls_of(tw["messages"])}
        first = min(s["msgs"])
        for m in s["msgs"]:
            a, b = text(tw["messages"][m]), text(pl["messages"][m])
            if m not in sa or a == b:
                counts["sites unmapped or unchanged (dropped)"] += 1
                continue
            kind = "newline only" if b.rstrip("\r\n") == a.rstrip("\r\n") else "format and newline"
            site_rows.append({"pair": p, "key": s["key"], "env": tw["environment"], "step": sa[m][0], "action": sa[m][1],
                              "type": kind, "first": m == first})
    counts["sites in the manifest"] = sum(len(s["msgs"]) for s in sites)
    out = {"counts": dict(counts), "readers": {}}
    g = np.random.default_rng(SEED)
    flips = g.random((N_NULL, len(sites))) < 0.5  # one coin per run pair, drawn once, used at every layer and reader
    for reader, (mnt, run) in READERS.items():
        d = Path(mnt) / run
        meta = {}
        for arm, suffix in ((TWIN, ""), (PLANT, "__plant2")):
            for f in (d / "meta" / arm).glob("*.json"):
                m = json.loads(f.read_text()); meta[(arm, m["key"])] = m
        idx = {}
        for (arm, key), m in meta.items():
            for q in m["positions"]:
                if q["kind"] == "output_end" and q["kept"]:
                    idx[(arm, key, q["step"], q["action"])] = (m["chunk"], m["row_start"] + q["row"])
        kept = [r for r in site_rows if (TWIN, r["key"], r["step"], r["action"]) in idx and (PLANT, r["key"] + "__plant2", r["step"], r["action"]) in idx]
        rc = Counter({"sites typed": len(site_rows), "sites with a kept readout in both copies": len(kept)})
        rc |= Counter(f"kept sites: {r['type']}" for r in kept); rc["kept sites: first site of their pair"] = sum(r["first"] for r in kept)
        rc["pairs with a kept site"] = len({r["pair"] for r in kept})
        # rows: for each kept site, the twin row then the planted row
        need = [idx[(TWIN, r["key"], r["step"], r["action"])] for r in kept] + [idx[(PLANT, r["key"] + "__plant2", r["step"], r["action"])] for r in kept]
        by_chunk = defaultdict(list)
        for j, (c, row) in enumerate(need):
            by_chunk[c].append((j, row))
        n = len(kept); env = np.array([r["env"] for r in kept]); pair = np.array([r["pair"] for r in kept])
        typ = np.array([r["type"] for r in kept]); first = np.array([r["first"] for r in kept])
        res = {"counts": dict(rc), "layers": {}}
        for layer in LAYERS:
            def load(item):
                c, rows = item
                X = np.load(d / "by_layer" / f"L{layer}" / f"{c}.npy")  # whole file: row-indexed mmap is slow on a Volume
                return [(j, X[r].astype(np.float32)) for j, r in rows]
            Xall = None
            with ThreadPoolExecutor(8) as pool:
                for got in pool.map(load, by_chunk.items()):
                    for j, v in got:
                        if Xall is None:
                            Xall = np.zeros((len(need), len(v)), np.float32)
                        Xall[j] = v
            if not np.isfinite(Xall).all():
                raise ValueError("non-finite activations")
            Xt, Xp = Xall[:n], Xall[n:]
            # label columns: 0 = real (planted 1, twin 0); 1..N_NULL = each pair's labels flipped or not
            lab_p = np.ones((n, N_NULL + 1), bool); lab_p[:, 1:] = ~flips[:, pair].T  # planted row's label per column
            sc_t, sc_p = np.full((n, N_NULL + 1), np.nan), np.full((n, N_NULL + 1), np.nan)
            iters = []
            for e in sorted(set(env)):
                tr, te = env != e, env == e
                X = torch.from_numpy(np.vstack([Xt[tr], Xp[tr]])).cuda()
                Y = torch.from_numpy(np.vstack([~lab_p[tr], lab_p[tr]]).astype(np.float32)).cuda()
                Xs, Xte_t, Xte_p = tp.standardize(X, torch.from_numpy(Xt[te]).cuda(), torch.from_numpy(Xp[te]).cuda())
                W, b, it = tp.fit(Xs, Y, C, MAX_ITER)
                iters.append(int(it))
                sc_t[te] = tp.decision(Xte_t, W, b).double().cpu().numpy(); sc_p[te] = tp.decision(Xte_p, W, b).double().cpu().numpy()

            def stat(mask):  # per column: site-weighted mean of within-environment AUROCs, and the win share
                aucs, wins = np.zeros(N_NULL + 1), np.zeros(N_NULL + 1)
                tot = 0
                for e in sorted(set(env[mask])):
                    mm = mask & (env == e); k = int(mm.sum()); tot += k
                    for col in range(N_NULL + 1):
                        pos = np.where(lab_p[mm, col], sc_p[mm, col], sc_t[mm, col]); neg = np.where(lab_p[mm, col], sc_t[mm, col], sc_p[mm, col])
                        aucs[col] += k * auroc(pos, neg); wins[col] += (pos > neg).sum()
                return aucs / tot, wins / tot, tot
            layer_out = {"loss evaluations by fold": iters, "folds that may have hit the cap (evaluations >= max_iter)": sum(i >= MAX_ITER for i in iters)}
            for name, mask in (("all sites", np.ones(n, bool)), ("newline only", typ == "newline only"),
                               ("format and newline", typ == "format and newline"), ("first site of each pair", first)):
                if not mask.any():
                    layer_out[name] = {"sites": 0}; continue
                a, w, tot = stat(mask)
                layer_out[name] = {"sites": tot, "AUROC": float(a[0]), "planted beats twin": float(w[0]),
                                   "null AUROC 2.5/50/95/97.5 percentiles": [float(x) for x in np.percentile(a[1:], [2.5, 50, 95, 97.5])],
                                   "null draws": a[1:].tolist()}
            res["layers"][str(layer)] = layer_out
        for name in ("all sites", "newline only", "format and newline", "first site of each pair"):
            L = [res["layers"][str(l)][name] for l in LAYERS]
            if any(x["sites"] == 0 for x in L):
                continue
            M = max(x["AUROC"] for x in L); Mnull = np.max(np.array([x["null draws"] for x in L]), axis=0); M95 = float(np.percentile(Mnull, 95))
            reading = ("decodable" if M >= 0.75 else "inconclusive") if M > M95 else "no evidence"
            verdict = reader == "Qwen3.5-27B" and name == "all sites"  # PLAN.md: the only verdict; the rest is beside
            res.setdefault("max over layers", {})[name] = {"M": M, "M95": M95,
                                                           ("reading by the rule" if verdict else "same rules applied (beside, no verdict)"): reading}
        for l in LAYERS:
            for name in res["layers"][str(l)]:
                if isinstance(res["layers"][str(l)][name], dict):
                    res["layers"][str(l)][name].pop("null draws", None)
        out["readers"][reader] = res
    return out


@app.local_entrypoint()
def main():
    for f in (Path(__file__).resolve(), HERE / "PLAN.md"):
        if subprocess.run(["git", "ls-files", "--error-unmatch", str(f)], cwd=HERE, capture_output=True).returncode \
                or subprocess.run(["git", "diff", "--quiet", "HEAD", "--", str(f)], cwd=HERE).returncode:
            raise SystemExit(f"{f.name} differs from HEAD: commit it before running")
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip()
    res = run.remote()
    q = res["readers"]["Qwen3.5-27B"].get("max over layers", {}).get("all sites", {})
    out = {"plan": "PLAN.md (this folder)", "git": head, "layers": list(LAYERS), "C": C, "null shuffles": N_NULL, "seed": SEED,
           "reading (Qwen3.5-27B, all sites)": {"decodable": "planted-vs-twin status is decodable at the sites", "inconclusive": "inconclusive",
                                                "no evidence": "no evidence that it is"}.get(q.get("reading by the rule"), "not computed")} | res
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    (HERE / "results").mkdir(exist_ok=True)
    with open(dest := HERE / "results" / f"aimed_slip_{stamp}.json", "x") as fh:
        fh.write(json.dumps(out, indent=1) + "\n")
    print(dest)
