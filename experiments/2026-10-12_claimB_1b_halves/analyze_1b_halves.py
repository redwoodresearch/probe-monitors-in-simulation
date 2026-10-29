"""Llama-3.2-1B through the halves test, the summary (RULE.md here; written before any Llama score exists). Reads
run_1b_refit_modal.py's arrays and, for the comparison, halves' Qwen3-32B arrays (claimB/halves_v1/5b07509a98fc and its
POSTHOC_e2 direct-yardstick rerun, aswin/claimb-halves @53da7f48). Per RULE.md:
- selection per reader and direction: the best validation AUROC (real-label column) among candidates whose fits converged
  (RULE.md amendment 2, step A's rule: experiments/2026-10-09_fixed_sim/maxb_core.py choose() @96e4c6a1); the unmasked
  argmax (halves' analyze_halves.py) beside it;
- the frozen nominal-1% threshold per direction on all held-out real e1 runs (halves' thr), achieved e1 rate reported;
  sim, direct e2 (e2 scored by the sim-trained probe) and plant-2 copies flagged; pooled over directions;
- 2,000 main-task draws per direction with halves' seed and construction, the threshold recomputed in each draw;
- the plant-2 gate per direction: AUROC >= 0.910 AND p < 0.05 against the 20 refit shuffles (halves' rule);
- the paired reader difference: Llama's pooled direct gap minus the probe's, over the same draws, with the same rule;
- beside it, the TF-IDF halves result (aswin/claimb-tfidf-halves @faafd24c) and Llama's earlier planted checks.
Gate: the probe's unmasked numbers must reproduce item (A)'s direct counts (33/473, 8/473, 18/466, 7/466).

  PYTHONPATH=<asrun>/src uv run --no-project --python 3.12 --with 'modal>=1.4.3' --with numpy --with scikit-learn \\
      --with scipy python experiments/2026-10-12_claimB_1b_halves/analyze_1b_halves.py
"""

import hashlib
import io
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal
import numpy as np
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).parent
sys.path[:0] = [str(HERE), str(HERE.parent / "2026-10-11_claimB_h3_fix_to_plateau"), str(HERE.parent / "2026-10-11_claimB_slips_evened_probe")]
import run_1b_refit_modal as R  # noqa: E402

DRAWS, SEED, PLANT_FLOOR = 2000, 20261011, 0.910  # analyze_halves.py @53da7f48 line 42
PROBE_OUT, PROBE_SHA = "claimB/halves_v1/5b07509a98fc", "5b07509a98fc80968fb508ab5da569f72ed417b4044c05ed2d4b1813ca1c8e85"
POSTHOC_RUNNER = ("53da7f48", "experiments/2026-10-11_claimB_halves/run_POSTHOC_halves_e2_modal.py")
PROBE_GATE = {"dev_to_test": (33, 473, 8, 473), "test_to_dev": (18, 466, 7, 466)}  # halves_e2_direct_POSTHOC_2026-10-11_172057.json
TFIDF = ("faafd24c", "experiments/2026-10-12_claimB_tfidf_halves/results/tfidf_halves_POSTHOC_2026-10-12_112423.json")
LLAMA_EARLIER = {"source": "experiments/2026-10-10_claimB_small_lm/PLAN.md @6d7caaf5 (lines 75-106)",
                 "plant1": {"auroc": 0.637, "bar": 0.95}, "plant2": {"auroc": 0.862, "bar": 0.90}, "fixed_vs_e1_auroc": 0.705}


def thr(neg):  # analyze_halves.thr: the alarm sits at the floor(1%)-th highest real e1 score
    s = np.sort(neg)[::-1]
    return s[int(np.floor(0.01 * len(s)))]


def tol(C):
    return 1e-4 * min(C**0.5, 1.0)


def pick(z, col, masked):
    """Index of the chosen candidate for label column col: best validation AUROC, among converged candidates if masked."""
    V = z["val_auroc"][:, col]
    if not masked and not np.isfinite(V).all():
        raise SystemExit("non-finite validation AUROC under the unmasked rule")
    if masked:
        subs = json.loads(str(z["subs"]))
        conv = z["converged"] if "converged" in z.files else z["grad"] <= np.array([tol(C) for _, C in subs])[:, None, None]
        V = np.where(conv[:, 0, col] & conv[:, 1, col] & np.isfinite(V), V, -np.inf)  # both the full and the validation fit
        if not np.isfinite(V.max()):
            return None
    return int(np.argmax(V))


class Reader:
    def __init__(self, read, out, run_sha, fixed_files, plant_files=None, array_sha=None, read_sha=None):
        raw = read(f"{out}/runs.json")
        meta = json.loads(raw)
        if meta["script_sha256"] != run_sha:
            raise SystemExit(f"{out}/runs.json from another runner version")
        self.inputs = {f"{out}/runs.json": hashlib.sha256(raw).hexdigest()}
        runs = meta["runs"]
        self.env, self.arm, self.task, self.cell = (np.array([r[k] for r in runs]) for k in ("environment", "arm", "main_task", "cell"))
        self.z, self.zp = {}, {}
        for d in R.DIRS:
            b = read(fixed_files[d])
            self.inputs[fixed_files[d]] = hashlib.sha256(b).hexdigest()
            self.z[d] = np.load(io.BytesIO(b))
            if str(self.z[d]["script_sha256"]) != (array_sha or run_sha):
                raise SystemExit(f"{fixed_files[d]}: made by another runner version")
            if read_sha and str(self.z[d]["read_script_sha256"]) != read_sha:
                raise SystemExit(f"{fixed_files[d]}: rows from another read runner version")
            if plant_files:
                b = read(plant_files[d])
                self.inputs[plant_files[d]] = hashlib.sha256(b).hexdigest()
                self.zp[d] = np.load(io.BytesIO(b))
                if str(self.zp[d]["script_sha256"]) != run_sha:
                    raise SystemExit(f"{plant_files[d]}: made by another runner version")
        self.held = {d: np.isin(self.env, R.DIRS[d][1]) for d in R.DIRS}
        self.halves = {d: sorted(set(self.task[self.held[d] & np.isin(self.arm, [R.SIM, R.E1, R.E2])])) for d in R.DIRS}

    def score(self, d, masked):
        b = pick(self.z[d], 0, masked)
        return None if b is None else self.z[d]["test_run_score"][b, :, 0]

    def flags(self, d, s, dr=None):
        w = np.zeros(len(self.arm))
        if dr is None:
            w[self.held[d]] = 1.0
        else:
            pos = {t: j for j, t in enumerate(self.halves[d])}
            cnt = np.bincount(dr, minlength=len(self.halves[d]))
            for i in np.flatnonzero(self.held[d]):
                w[i] = cnt[pos[self.task[i]]] if self.task[i] in pos else 0
        e1 = self.held[d] & (self.arm == R.E1) & (w > 0)
        return s > thr(np.repeat(s[e1], w[e1].astype(int))), w

    def rates(self, S, drs=None):
        """{arm: flagged share} per direction and pooled; S = {d: scores}, drs = {d: draw} or None."""
        out, tot = {}, {}
        for d in R.DIRS:
            f, w = self.flags(d, S[d], None if drs is None else drs[d])
            out[d] = {}
            for nm, ar in (("sim", R.SIM), ("e2", R.E2), ("e1", R.E1), ("plant2", R.PL2)):
                m = (self.arm == ar) & self.held[d]
                if not np.isfinite(S[d][m]).all():  # an arm this reader did not score (the probe's plant copies): unavailable
                    out[d][nm] = None
                    tot[nm] = None
                    continue
                c, n = float((w * f * m).sum()), float((w * m).sum())
                out[d][nm] = c / n if n else float("nan")
                if nm in tot and tot[nm] is None:
                    continue
                tot.setdefault(nm, [0.0, 0.0])
                tot[nm][0] += c
                tot[nm][1] += n
        out["pooled"] = {nm: (None if v is None else v[0] / v[1] if v[1] else float("nan")) for nm, v in tot.items()}
        for k in out:
            out[k]["gap"] = out[k]["sim"] - out[k]["e2"]
        return out

    def counts(self, S):
        res = {}
        for d in R.DIRS:
            f, _ = self.flags(d, S[d])
            res[d] = {}
            for nm, ar in (("sim", R.SIM), ("e2", R.E2), ("e1", R.E1), ("plant2", R.PL2)):
                m = (self.arm == ar) & self.held[d]
                res[d][nm] = [int((f & m).sum()), int(m.sum())] if np.isfinite(S[d][m]).all() else None
        res["pooled"] = {nm: (None if any(res[d][nm] is None for d in R.DIRS) else [sum(res[d][nm][0] for d in R.DIRS), sum(res[d][nm][1] for d in R.DIRS)])
                         for nm in res[next(iter(R.DIRS))]}
        return res

    def plant_gate(self, d, masked):
        z = self.zp[d]
        i = np.flatnonzero(z["use"] & self.held[d])
        Y = z["Y"] > 0.5
        aucs = []
        for k in range(Y.shape[1]):
            b = pick(z, k, masked)
            aucs.append(np.nan if b is None else roc_auc_score(Y[i, k], z["test_run_score"][b, i, k]))
        obs, null = aucs[0], np.array(aucs[1:])
        if not np.isfinite(obs) or not np.isfinite(null).all():  # an unavailable fit: incomplete, no p-value
            return {"pairs": int(Y[i, 0].sum()), "auroc": None if not np.isfinite(obs) else float(obs), "p": None,
                    "null_columns_valid": int(np.isfinite(null).sum()), "pass": False, "incomplete": True}
        p = float((1 + (null >= obs).sum()) / (1 + len(null)))
        return {"pairs": int(Y[i, 0].sum()), "auroc": float(obs), "p": p, "null_columns_valid": len(null),
                "pass": bool(obs >= PLANT_FLOOR and p < 0.05), "incomplete": False}


def ci(v):
    return [float(x) for x in np.percentile(np.asarray(v, float), [2.5, 97.5])]


def main():
    vol = modal.Volume.from_name("probe-shift_aswin-surya_sub-la-activations", version=2)
    read = lambda p: b"".join(vol.read_file(p))  # noqa: E731
    L = Reader(read, R.OUT, R.SCRIPT_SHA, {d: f"{R.OUT}/{d}/fixed_vs_e1.npz" for d in R.DIRS}, {d: f"{R.OUT}/{d}/plant2.npz" for d in R.DIRS}, read_sha=R.READ_SHA)
    ph = hashlib.sha256(subprocess.run(["git", "show", f"{POSTHOC_RUNNER[0]}:{POSTHOC_RUNNER[1]}"], capture_output=True, check=True, cwd=HERE).stdout).hexdigest()
    P = Reader(read, PROBE_OUT, PROBE_SHA, {d: f"{PROBE_OUT}/POSTHOC_e2/{d}/fixed_vs_e1.npz" for d in R.DIRS}, array_sha=ph)
    if L.halves != P.halves:
        raise SystemExit("the two readers' held-out task lists differ")
    inv = lambda X: sorted(zip(X.env.tolist(), X.task.tolist(), X.cell.tolist(), X.arm.tolist()))  # noqa: E731
    if inv(L) != inv(P):
        raise SystemExit("the two readers' run inventories (environment, main task, cell, arm) differ")
    out = {"label": "POST HOC (RULE.md): Llama-3.2-1B through the halves test; descriptive, preliminary until reviewed", "inputs": L.inputs | P.inputs}
    Sp_u = {d: P.score(d, False) for d in R.DIRS}
    pc = P.counts(Sp_u)
    got = {d: (pc[d]["sim"][0], pc[d]["sim"][1], pc[d]["e2"][0], pc[d]["e2"][1]) for d in R.DIRS}
    if got != PROBE_GATE:
        raise SystemExit(f"gate: the probe's unmasked counts {got} differ from item (A)'s {PROBE_GATE}")
    out["gate_probe_reproduces_item_A"] = {"pass": True, "counts": got}
    rng = np.random.default_rng(SEED)  # halves' draws: per direction in DIRS order
    draws = {d: [rng.integers(0, len(L.halves[d]), len(L.halves[d])) for _ in range(DRAWS)] for d in R.DIRS}
    res = {}
    for rule, masked in (("converged_only", True), ("unmasked", False)):
        SL = {d: L.score(d, masked) for d in R.DIRS}
        SP = {d: P.score(d, masked) for d in R.DIRS}
        if any(v is None for v in list(SL.values()) + list(SP.values())):
            res[rule] = {"error": "a direction has no converged candidate for the real-label column"}
            continue
        r = {"llama": {"counts": L.counts(SL), "rates": L.rates(SL)}, "probe": {"counts": P.counts(SP), "rates": P.rates(SP)},
             "plant2_gate_llama": {d: L.plant_gate(d, masked) for d in R.DIRS}}
        bl, bp = [], []
        for b in range(DRAWS):
            jd = {d: draws[d][b] for d in R.DIRS}
            bl.append(L.rates(SL, jd))
            bp.append(P.rates(SP, jd))
        for who, bs in (("llama", bl), ("probe", bp)):
            r[who]["gap_ci95"] = {k: ci([x[k]["gap"] for x in bs]) for k in list(R.DIRS) + ["pooled"]}
        diff = [x["pooled"]["gap"] - y["pooled"]["gap"] for x, y in zip(bl, bp)]  # the same draws, the same cells
        r["paired_difference_llama_minus_probe"] = {"pooled_gap_difference": r["llama"]["rates"]["pooled"]["gap"] - r["probe"]["rates"]["pooled"]["gap"],
                                                    "ci95": ci(diff)}
        res[rule] = r
    tf = json.loads(subprocess.run(["git", "show", f"{TFIDF[0]}:{TFIDF[1]}"], capture_output=True, check=True, cwd=HERE).stdout)["results"]["pooled"]
    out |= {"results": res, "tfidf_pooled": {k: tf[k] for k in ("sim_flagged", "n_sim", "e2_flagged", "n_e2", "gap", "gap_ci95")} | {"source": f"{TFIDF[1]}@{TFIDF[0]}"},
            "llama_earlier_planted_checks": LLAMA_EARLIER, "draws": DRAWS, "seed": SEED,
            "git": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip(),
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "refit_script_sha256": R.SCRIPT_SHA, "read_script_sha256": R.READ_SHA}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    (HERE / "results").mkdir(exist_ok=True)
    dst = HERE / "results" / f"1b_halves_{stamp}.json"
    with open(dst, "x") as fh:
        fh.write(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k not in ("inputs",)}, indent=1)[:6000])
    print("->", dst)


if __name__ == "__main__":
    main()
