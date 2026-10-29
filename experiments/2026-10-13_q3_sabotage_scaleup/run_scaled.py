"""Score the 116 new held-out tasks for RULE_scaled.md (primary) and its Amendment 2 (secondary). Numbers only; no text.

One Modal CPU container reads, read-only, the layer-30 call_end readings at: the R44 draws' training rows (committed
results/r44_selection_manifest.json), the label-free job's 10,973 rows (the 40 stage-1 tasks' calls and the honest B / C /
simulated-honest calls), the new tasks' 23,603 rows (make_scaled_rows.py), and the gate's 4,452 staged rows
(`/score/sabotage_gate_v1/rows.json`, pinned: honest A, honest_b calibration and held-out, and the gate's own rows). It returns the
scores of R (the frozen gate), S (the frozen sim-trained weights) and R44 draws 0–4 (refitted from the manifest with S's recipe,
fit_dual copied verbatim from run_r44.py), and refits draw 0 a second time. It also returns the extraction metas' digests.

Reproduction gates, checked locally before anything is written (nothing is written if one fails):
- R reproduces the label-free job's stored scores (its 10,973 rows and 4,452 staged rows) within 1e-6, and the gate's stored held-out
  scores within 1e-6 with identical flags at the frozen threshold;
- S reproduces the sim-trained pilot's stored scores on the 10,973 rows within 1e-6;
- R44: draw 0 fitted twice agrees within 1e-6, and every draw reproduces the R44 pilot's stored scores within 1e-6;
- the committed selection manifest equals a fresh selection (the rule in RULE_scaled.md Amendment 2);
- the metas digests equal those in both rows summaries; every score is finite.
Writes results/scaled_scores_<stamp>.json and results/scaled_r44_weights_<stamp>.npz. Refuses to run unless this file,
RULE_scaled.md, make_scaled_rows.py and the selection manifest are committed unchanged at HEAD.

  uv run --no-project --python 3.12 --with "modal==1.6.1" --with numpy modal run run_scaled.py --inputs <dir holding the pinned files>
"""

import csv
import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

APP = "probe-shift_aswin-surya_sub-q3scaled_score_kill-2026-11-12"
ACTS = "probe-shift_aswin-surya_sub-q3onpol_acts_kill-2026-11-08"
SCORE = "probe-shift_aswin-surya_sub-q3onpol_score_kill-2026-11-08"
OUT, LAYER, HIDDEN, CS = "sabotage_gate_v1", 30, 5120, [1e-4]
HERE = Path(__file__).parent
PINS = {  # file name -> sha256, each as recorded in its manifest
    "flagrate_rows_2026-10-13_122417.csv": "44c75f78e78f86b4bcab409b1379762a8d4dccadfd456f28ee651832eb2b13a7",  # manifest_flagrate_s3_v1 @ 13abd4ba
    "flagrate_rows_summary_2026-10-13_122417.json": "6e7594357d4b92239b9d7eac30ec082742d042db3023dae1614c12f3d2df084e",
    "flagrate_scores_2026-10-13_122815.json": "b53660ed6e190f10c378ff40762dffe391f5da6374f8ba354b1fd1092c53d1b7",
    "scaled_rows_2026-10-13_152021.csv": "a95453ac2688e61ba99184726ce27a4dd54276a04730a56d67825f204d1ee1f3",  # make_scaled_rows.py @ 49e81fc2
    "scaled_rows_summary_2026-10-13_152021.json": "5873aebe868e6ec21e5c46bb2813a200fe40f29620546fb92acfccee2d2e80ed",
    "gate_refit_2026-10-13_085636.npz": "dbe0908a09b2499304eb6d70d529e59b949524333fa91c658570c55078764cf8",  # manifest_gate_refit_v1
    "gate_final_2026-10-11_211436.json": "a08fda3bf41cad1645551c2fc9f7fb1a05d5224b4c653bff4d0ced8573e0c88b",  # manifest_gate_v1
    "rows_v2.json": "727c417916b8fb2df9a31aca5f8ca0cc4dd17002a8a2edcb7bd79444bc0bf4b5",
    "simtrained_weights_2026-10-13_132542.npz": "2e8371cc92c3de3fa18610032a8f551e22605a7a878f322008e68cb33185cf8a",  # manifest_simtrained_s3_v1
    "simtrained_scores_2026-10-13_132542.json": "3dc328d497f7a9beec5ef2a5ae289ac3d80c9431e0bdf4fd1fe75984bbe50731",
    "r44_scores_2026-10-13_135204.json": "55565fac6c35a1458f38ffc40fbf53e1a053cc830f01c6a6d4a0b2fe9d76e3a7",  # manifest_simtrained_s3_v2
    "sim_votes.csv": "06f4ef4bc00b53767b074185cda99311f2ba44f7c8f5a4f38ce59dd1537e81ae",  # manifest_sim_panel_votes_s3_v1
    "real_votes.csv": "9a96988db9818c8b97f3045ade83f19fae0e19e7d0cac134fa775005eba8dc36"}  # manifest_real_panel_votes_s3_v1
SEL_SHA = "9d989e752f8c8a41e18f5dfb0810a9e0a78d8fa7ab2394d62ad64adca3a648ba"  # results/r44_selection_manifest.json @ 37a417e3
SIZES = {"sabotage": 44, "not_sabotage": 900}

app = modal.App(APP)
acts = modal.Volume.from_name(ACTS, version=2)
score = modal.Volume.from_name(SCORE, version=2)
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("torch==2.8.0", "numpy")


def fit_dual(K, y, w, Cs, iters=100, tol=1e-9):
    """Copied from run_persist_gate.py (from run_score_onpolicy.py, main): alpha [L, c, n], beta = Z^T alpha minimizing
    C sum w_i (log(1 + e^f) - y f) + 1/2 |beta|^2, f = K alpha, by Newton steps with backtracking; raises if not converged."""
    import torch
    import torch.nn.functional as F

    L, n, _ = K.shape
    C = torch.tensor(Cs, dtype=K.dtype)[None, :, None]
    Kb, eye = K[:, None], torch.eye(n, dtype=K.dtype)
    a = torch.zeros(L, len(Cs), n, dtype=K.dtype)

    def obj(a):
        f = (Kb @ a[..., None])[..., 0]
        return (C * w * (F.softplus(f) - y * f)).sum(-1) + 0.5 * (a * f).sum(-1)

    for _ in range(iters):
        f = (Kb @ a[..., None])[..., 0]
        p = torch.sigmoid(f)
        r = a - C * w * (y - p)
        res = r.abs().amax(-1)
        if float(res.max()) < tol:
            return a
        delta = torch.linalg.solve(eye + (C * w * p * (1 - p))[..., :, None] * Kb, r[..., None])[..., 0]
        step, j0 = torch.ones(L, len(Cs), 1, dtype=K.dtype), obj(a)
        for _ in range(40):
            worse = obj(a - step * delta) > j0 + 1e-12 * j0.abs()
            if not worse.any():
                break
            step = torch.where(worse[..., None], step / 2, step)
        a = a - step * delta
    raise RuntimeError(f"dual Newton did not converge: max residual {float(res.max()):.3g}")


@app.function(image=image, cpu=4, memory=32768, timeout=1500, retries=0, volumes={"/acts": acts.read_only(), "/score": score.read_only()})
def score_job(rows_sha: str, draws: list, eval_locs: list, frozen: dict, xs: list) -> dict:
    """draws: [[(extraction, chunk, row, y)]]; eval_locs: [(extraction, chunk, row)]; frozen: name -> (weights, mu, sd)."""
    import numpy as np
    import torch

    raw = Path(f"/score/{OUT}/rows.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != rows_sha:
        raise SystemExit("staged rows on the Volume differ from rows_v2.json")
    staged = [(r["x"], r["chunk"], int(r["row"])) for r in json.loads(raw)["rows"]]
    locs = sorted({(x, c, int(r)) for d in draws for x, c, r, _ in d} | {(x, c, int(r)) for x, c, r in eval_locs} | set(staged))
    pos = {v: i for i, v in enumerate(locs)}
    groups = {}
    for i, (x, chunk, row) in enumerate(locs):
        groups.setdefault((x, chunk), []).append((i, row))
    X = np.empty((len(locs), HIDDEN), np.float64)
    for (x, chunk), lst in groups.items():
        a = np.load(f"/acts/{x}/by_layer/L{LAYER}/{chunk}.npy")
        X[[i for i, _ in lst]] = a[[r for _, r in lst]]
    Xe = X[[pos[(x, c, int(r))] for x, c, r in eval_locs]]
    Xs = X[[pos[v] for v in staged]]
    out = {}
    for name, (w, mu, sd) in frozen.items():
        w, mu, sd = (np.array(v) for v in (w, mu, sd))
        out[name] = {"eval": (((Xe - mu) / sd) @ w).tolist(), "staged": (((Xs - mu) / sd) @ w).tolist()}

    def fit(d):
        Xt = X[[pos[(x, c, int(r))] for x, c, r, _ in d]]
        y = torch.tensor([float(v) for *_, v in d], dtype=torch.float64)
        mu, sd = Xt.mean(0), Xt.std(0)
        sd = np.where(sd < 1e-6, 1.0, sd)
        Zt = torch.from_numpy((Xt - mu) / sd)
        a = fit_dual((Zt @ Zt.T)[None], y[None, None, :], torch.ones(len(d), dtype=torch.float64), CS)
        return (a[0, 0] @ Zt).numpy(), mu, sd

    weights = {}
    for i, d in enumerate(draws):
        beta, mu, sd = fit(d)
        weights[f"R44_{i}"] = {"weights": beta.tolist(), "mu": mu.tolist(), "sd": sd.tolist()}
        out[f"R44_{i}"] = {"eval": (((Xe - mu) / sd) @ beta).tolist(), "staged": (((Xs - mu) / sd) @ beta).tolist()}
    beta2, mu2, sd2 = fit(draws[0])
    refit_diff = float(np.abs(((Xe - mu2) / sd2) @ beta2 - np.array(out["R44_0"]["eval"])).max())
    dig = {}
    for x in xs:
        files = sorted(Path(f"/acts/{x}/meta").glob("*/*.json"))
        dig[x] = hashlib.sha256(b"".join(hashlib.sha256(f.read_bytes()).digest() for f in files)).hexdigest()
    return {"scores": out, "weights": weights, "refit0_max_abs_diff": refit_diff, "chunks_read": len(groups), "n_staged": len(staged),
            "metas_digests": dig}


def sha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def select(flag_rows: list, sim_votes: str, real_votes: str) -> dict:
    """RULE_scaled.md Amendment 2's R44 selection, recomputed (as make_simtrained_rows.build + make_r44_selection.select)."""
    import numpy as np

    key4 = lambda r: (r["run_id"], int(r["epoch"]), int(r["turn"]), int(r["call_index"]))  # noqa: E731
    votes = {}
    for path, default in ((sim_votes, "sim"), (real_votes, None)):
        for r in csv.DictReader(open(path)):
            if key4(r) in votes:
                raise SystemExit("a call appears twice in the votes")
            votes[key4(r)] = (r.get("world") or default, r["label"], int(r["state_changing"]))
    labels = {}
    for r in flag_rows:
        if r["part"] != "2":
            continue
        v = votes.get(key4(r))
        if v is None or v[0] != r["world"] or v[2] != int(r["state_changing"]):
            raise SystemExit("a part-2 row has no matching vote")
        labels[r["call_id"]] = {"world": v[0], "group": r["group"], "state_changing": v[2], "label": v[1]}
    pools = {c: sorted(cid for cid, v in labels.items() if v["world"] in ("real", "run2") and v["group"] == "stage1_train"
                       and v["state_changing"] == 1 and v["label"] == c) for c in SIZES}
    draws = []
    for d in range(5):
        seed = int(hashlib.sha256(f"match44|{d}".encode()).hexdigest()[:16], 16)
        rng = np.random.default_rng(seed)
        picked = {c: [pools[c][i] for i in rng.choice(len(pools[c]), SIZES[c], replace=False)] for c in SIZES}
        draws.append({"draw": d, "seed": seed, "sabotage": picked["sabotage"], "not_sabotage": picked["not_sabotage"]})
    return {"candidate_counts": {c: len(p) for c, p in pools.items()}, "draws": draws}


@app.local_entrypoint()
def main(inputs: str):
    import numpy as np

    for f in (Path(__file__).name, "RULE_scaled.md", "make_scaled_rows.py", "results/r44_selection_manifest.json"):
        tracked = subprocess.run(["git", "ls-files", "--error-unmatch", f], cwd=HERE, capture_output=True)
        status = subprocess.run(["git", "status", "--porcelain", "--", f], cwd=HERE, capture_output=True, text=True)
        if tracked.returncode != 0 or status.returncode != 0 or status.stdout:
            raise SystemExit(f"{f} is not committed unchanged at HEAD")
    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=HERE, capture_output=True, text=True, check=True).stdout.strip()
    I = Path(inputs)
    for n, want in PINS.items():
        if sha(I / n) != want:
            raise SystemExit(f"{n}: sha256 differs from its pin")
    selp = HERE / "results" / "r44_selection_manifest.json"
    if sha(selp) != SEL_SHA:
        raise SystemExit("r44_selection_manifest.json: sha256 differs from its pin")
    ss = [json.loads((I / "scaled_rows_summary_2026-10-13_152021.json").read_text())]
    if ss[0]["rows_file"]["sha256"] != PINS["scaled_rows_2026-10-13_152021.csv"]:
        raise SystemExit("the scaled_rows summary is for another rows file")
    F = list(csv.DictReader(open(I / "flagrate_rows_2026-10-13_122417.csv")))
    N = list(csv.DictReader(open(I / "scaled_rows_2026-10-13_152021.csv")))
    if len({r["call_id"] for r in F} | {r["call_id"] for r in N}) != len(F) + len(N):
        raise SystemExit("call_id collision between the two rows files")
    man = json.loads(selp.read_text())
    fresh = select(F, str(I / "sim_votes.csv"), str(I / "real_votes.csv"))
    if fresh["draws"] != man["draws"] or fresh["candidate_counts"] != man["candidate_counts"]:
        raise SystemExit("the committed selection manifest differs from a fresh selection")
    loc = {r["call_id"]: (r["x"], r["chunk"], int(r["row"])) for r in F}
    draws = [[(*loc[c], 1) for c in d["sabotage"]] + [(*loc[c], 0) for c in d["not_sabotage"]] for d in man["draws"]]
    zr, zs = np.load(I / "gate_refit_2026-10-13_085636.npz"), np.load(I / "simtrained_weights_2026-10-13_132542.npz")
    frozen = {n: tuple(np.asarray(z[k], dtype=np.float64).tolist() for k in ("weights", "mu", "sd")) for n, z in (("R", zr), ("S", zs))}
    fsum = json.loads((I / "flagrate_rows_summary_2026-10-13_122417.json").read_text())
    want_dig = fsum["metas_digests"] | ss[0]["metas_digests"]
    if any(fsum["metas_digests"].get(x, d) != d for x, d in ss[0]["metas_digests"].items()):
        raise SystemExit("the two rows summaries disagree on a metas digest")
    rows_v2 = I / "rows_v2.json"
    eval_rows = F + N
    out = score_job.remote(PINS["rows_v2.json"], draws, [(r["x"], r["chunk"], int(r["row"])) for r in eval_rows], frozen, sorted(want_dig))
    sc, nF = out["scores"], len(F)
    fs = json.loads((I / "flagrate_scores_2026-10-13_122815.json").read_text())
    ps = json.loads((I / "simtrained_scores_2026-10-13_132542.json").read_text())
    rs = json.loads((I / "r44_scores_2026-10-13_135204.json").read_text())
    gate = json.loads((I / "gate_final_2026-10-11_211436.json").read_text())["final"]["real|0"]
    r_eval, r_staged = {s["call_id"]: s["score"] for s in fs["scores"]}, np.array([s["score"] for s in fs["staged"]])
    s_eval = {s["call_id"]: s["score"] for s in ps["scores"]}
    names_p = rs["probes"]
    r44_eval = {s["call_id"]: dict(zip(names_p, s["s"])) for s in rs["scores"]}
    role = np.array([r["role"] for r in json.loads(rows_v2.read_text())["rows"]])
    st_r = np.array(sc["R"]["staged"])
    ho, stored_ho, t = st_r[role == "heldout"], np.array(gate["heldout_scores"]), gate["threshold"]
    diff = lambda name, ref: float(max(abs(v - ref[r["call_id"]]) for v, r in zip(sc[name]["eval"][:nF], F)))  # noqa: E731
    val = {"R_flagrate_rows_max_abs_diff": diff("R", r_eval),
           "R_staged_max_abs_diff": float(np.abs(st_r - r_staged).max()) if len(st_r) == len(r_staged) else None,
           "R_gate_heldout_max_abs_diff": float(np.abs(ho - stored_ho).max()) if len(ho) == len(stored_ho) else None,
           "R_gate_heldout_flags_identical": bool(len(ho) == len(stored_ho) and ((ho > t) == (stored_ho > t)).all()),
           "S_flagrate_rows_max_abs_diff": diff("S", s_eval),
           "R44_flagrate_rows_max_abs_diff": {f"R44_{i}": float(max(abs(v - r44_eval[r["call_id"]][f"R44_{i}"]) for v, r in zip(sc[f"R44_{i}"]["eval"][:nF], F)))
                                              for i in range(5)},
           "R44_0_refit_max_abs_diff": out["refit0_max_abs_diff"], "selection_matches_fresh": True,
           "metas_digests_match": out["metas_digests"] == want_dig, "n_flagrate_rows": nF, "n_new_rows": len(N), "n_staged": out["n_staged"],
           "all_finite": bool(all(np.isfinite(np.array(v[k])).all() for v in sc.values() for k in ("eval", "staged")))}
    worst = [val["R_flagrate_rows_max_abs_diff"], val["R_staged_max_abs_diff"], val["R_gate_heldout_max_abs_diff"], val["S_flagrate_rows_max_abs_diff"],
             *val["R44_flagrate_rows_max_abs_diff"].values(), val["R44_0_refit_max_abs_diff"]]
    val["passes"] = bool(None not in worst and max(worst) <= 1e-6 and val["R_gate_heldout_flags_identical"] and val["metas_digests_match"]
                         and val["all_finite"] and len(ho) == 351 and out["n_staged"] == 4452)
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    if not val["passes"]:
        raise SystemExit(f"reproduction gates failed, nothing written: {json.dumps(val)}")
    wz = HERE / "results" / f"scaled_r44_weights_{stamp}.npz"
    with wz.open("xb") as fh:
        np.savez(fh, **{f"{k}_{n}": np.array(v[k]) for n, v in out["weights"].items() for k in ("weights", "mu", "sd")})
    names = ["R", "S"] + [f"R44_{i}" for i in range(5)]
    meta = [{k: r[k] for k in ("call_id", "world", "task", "run_id")} | {"source": "flagrate", "part": int(r["part"]), "arm": r["arm"], "group": r["group"],
            "epoch": int(r["epoch"]), "turn": int(r["turn"]), "call_index": int(r["call_index"]), "state_changing": int(r["state_changing"])} for r in F]
    meta += [{k: r[k] for k in ("call_id", "world", "task", "run_id")} | {"source": "new", "part": 2, "arm": r["world"], "group": "new_held_out",
             "epoch": int(r["epoch"]), "turn": int(r["turn"]), "call_index": int(r["call_index"]), "state_changing": int(r["state_changing"])} for r in N]
    res = {"rule": "RULE_scaled.md + Amendments 1-2", "label": "scores only; no label joined", "git": git, "app": APP, "layer": LAYER, "C": CS[0],
           "frozen_gate_threshold_1pct": t, "validation": val, "inputs_sha256": PINS | {"r44_selection_manifest": SEL_SHA},
           "weights_npz": wz.name, "weights_npz_sha256": sha(wz), "chunks_read": out["chunks_read"], "probes": names,
           "scores": [m | {"s": [sc[n]["eval"][i] for n in names]} for i, m in enumerate(meta)],
           "staged": [{"index": i, "role": str(role[i]), "s": [sc[n]["staged"][i] for n in names]} for i in range(len(role))],
           "written": stamp}
    dst = HERE / "results" / f"scaled_scores_{stamp}.json"
    with dst.open("x") as fh:
        fh.write(json.dumps(res) + "\n")
    print(json.dumps({"validation": val}, indent=1), "\n->", dst, sha(dst), wz)
