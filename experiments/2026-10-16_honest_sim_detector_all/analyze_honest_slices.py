"""POST HOC boring-explanation checks on the honest detector record (0d's go, 10-16), from its stored out-of-fold run scores: no refit.
Per cell (all four), run-level AUROC with the record's own 2,000-draw main-task bootstrap (seed 20261015, mains drawn with replacement,
every run of a drawn main taken with multiplicity, one-class draws counted as failed), on these subsets:
  (1) epoch imbalance: sim vs B restricted to epoch 1 (and, beside it, sim vs B epoch 2);
  (2) set mix: sim vs B within each set; and sim vs all B with each B run weighted by (sim's share of its set) / (B's share of its set);
      the same per-set AUROCs for the control (A vs B);
  (3) S_old's launch effect on the control: A vs B without S_old's runs.
Positive control of this code: on the full problems it must reproduce the record's run AUROC and bootstrap interval (|diff| <= 1e-12),
else STOP. Scores and labels come from the record by run name (class prefix sim| / A| / B|); set and epoch from its run_set / run_epoch,
main from run_main. One Modal CPU container.

  PYTHONPATH=src uv run --no-project --python 3.12 --with "modal==1.6.1" --with "numpy==2.3.3" modal run \\
      experiments/2026-10-16_honest_sim_detector_all/analyze_honest_slices.py
"""

import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).resolve().parent
APP = "probe-shift_aswin-surya_sub-honsimdet_all_slices_kill-2026-11-15"
RECORD = ("8b0b18e2a", "experiments/2026-10-16_honest_sim_detector_all/results/honest_all_2026-10-16_120353.json",
          "c9ec2530e368c990136d536aa5ff74aefc3be44615698ed00f913f7b52d97b14")
BOOT_B, BOOT_SEED, TOL = 2000, 20261015, 1e-12
SETS = ("S_old", "S_new", "S_newmains", "S_paper493")

app = modal.App(APP)
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy==2.3.3")


@app.function(image=image, cpu=8, memory=8192, timeout=1800, retries=0)
def slices(raw: bytes) -> dict:
    import numpy as np

    rec = json.loads(raw)
    rset, rep, rmain = rec["run_set"], rec["run_epoch"], rec["run_main"]

    def wauc(sp, wp, sn, wn):
        """Weighted Mann-Whitney AUROC (ties one half): sum_i,j wp_i wn_j [sp_i > sn_j] / (sum wp sum wn)."""
        o = np.argsort(sn, kind="stable")
        sn, wn = sn[o], wn[o]
        cw = np.concatenate([[0.0], np.cumsum(wn)])
        lo, hi = np.searchsorted(sn, sp, "left"), np.searchsorted(sn, sp, "right")
        return float((wp * (cw[lo] + 0.5 * (cw[hi] - cw[lo]))).sum() / (wp.sum() * wn.sum()))

    def measure(scores: dict, pos: list, neg: list, w: dict | None = None) -> dict:
        """AUROC of pos vs neg run names, with the record's main-task bootstrap (the same draws as its boot_pairs on these runs)."""
        if not pos or not neg:  # (review r1, optional) an empty class has no AUROC
            raise ValueError(f"empty class: {len(pos)} vs {len(neg)}")
        names = pos + neg
        s = np.array([scores[n] for n in names], np.float64)
        y = np.array([1] * len(pos) + [0] * len(neg))
        wt = np.array([1.0 if w is None else w.get(n, 1.0) for n in names])
        keys = sorted({rmain[n] for n in names})
        mi = np.array([keys.index(rmain[n]) for n in names])
        obs = wauc(s[y == 1], wt[y == 1], s[y == 0], wt[y == 0])
        rng = np.random.default_rng(BOOT_SEED)
        vals, failed = [], 0
        for _ in range(BOOT_B):
            m = np.bincount(rng.integers(0, len(keys), len(keys)), minlength=len(keys))[mi] * wt
            if m[y == 1].sum() == 0 or m[y == 0].sum() == 0:
                failed += 1
                continue
            vals.append(wauc(s[y == 1], m[y == 1], s[y == 0], m[y == 0]))
        if not vals:
            raise ValueError("every bootstrap draw failed")
        v = np.array(vals)
        return {"auroc": obs, "ci95": [float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))], "n_pos": len(pos), "n_neg": len(neg),
                "n_mains": len(keys), "failed_draws": failed, "weighted": w is not None}

    out = {}
    for cell, c in rec["cells"].items():
        sc = {p: c["problems"][p]["run_scores"] for p in ("simh_vs_b", "a_vs_b")}
        cls = lambda p, k: sorted(n for n in sc[p] if n.startswith(k + "|"))  # noqa: E731
        sim, b_s, a, b_a = cls("simh_vs_b", "sim"), cls("simh_vs_b", "B"), cls("a_vs_b", "A"), cls("a_vs_b", "B")
        if len(sim) + len(b_s) != len(sc["simh_vs_b"]) or len(a) + len(b_a) != len(sc["a_vs_b"]) or b_s != b_a:
            raise ValueError(f"{cell}: run names outside sim| / A| / B|, or the B runs differ between the problems")
        r = {"full": {"simh_vs_b": measure(sc["simh_vs_b"], sim, b_s), "a_vs_b": measure(sc["a_vs_b"], a, b_a)}}
        for p in ("simh_vs_b", "a_vs_b"):  # the positive control of this code: the record's own numbers
            want = c["problems"][p]
            got = r["full"][p]
            if abs(got["auroc"] - want["run_level"]["auroc"]) > TOL or any(abs(x - z) > TOL for x, z in zip(got["ci95"], want["run_auroc_boot"]["ci95"])):
                raise ValueError(f"{cell}|{p}: does not reproduce the record's run AUROC / interval: {got['auroc']} {got['ci95']} vs "
                                 f"{want['run_level']['auroc']} {want['run_auroc_boot']['ci95']}")
        by = lambda ns, f: [n for n in ns if f(n)]  # noqa: E731
        r["(1) sim vs B epoch 1"] = measure(sc["simh_vs_b"], sim, by(b_s, lambda n: rep[n] == 1))
        r["(1) sim vs B epoch 2"] = measure(sc["simh_vs_b"], sim, by(b_s, lambda n: rep[n] == 2))
        r["(2) per set"] = {}
        for s_ in SETS:
            ins = lambda n: rset[n] == s_  # noqa: E731
            r["(2) per set"][s_] = {"simh_vs_b": measure(sc["simh_vs_b"], by(sim, ins), by(b_s, ins)),
                                    "a_vs_b": measure(sc["a_vs_b"], by(a, ins), by(b_a, ins))}
        share = {k: {s_: sum(rset[n] == s_ for n in ns) / len(ns) for s_ in SETS} for k, ns in (("sim", sim), ("B", b_s))}
        w = {n: share["sim"][rset[n]] / share["B"][rset[n]] for n in b_s}
        r["(2) sim vs B, B reweighted to sim's set shares"] = measure(sc["simh_vs_b"], sim, b_s, w) | {"set_shares": share}
        no_old = lambda n: rset[n] != "S_old"  # noqa: E731
        r["(3) A vs B without S_old"] = measure(sc["a_vs_b"], by(a, no_old), by(b_a, no_old))
        out[cell] = r
    return out


@app.local_entrypoint()
def main():
    git = lambda *a: subprocess.run(["git", *a], capture_output=True, check=True, cwd=HERE).stdout  # noqa: E731
    if git("status", "--porcelain", "--untracked-files=no") or not git("branch", "-r", "--contains", "HEAD"):
        raise SystemExit("commit and push first")
    if subprocess.run(["git", "ls-files", "--error-unmatch", "--", Path(__file__).name], capture_output=True, cwd=HERE).returncode:
        raise SystemExit("this script is not tracked: commit and push it first")  # (review r1, required) the record's commit must hold the code
    raw = git("show", f"{RECORD[0]}:{RECORD[1]}")
    if hashlib.sha256(raw).hexdigest() != RECORD[2]:
        raise SystemExit("the record differs from its pin")
    res = slices.remote(raw)
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    rec = {"label": "POST HOC boring-explanation checks (0d go, 10-16) from the stored out-of-fold run scores; no refit; PRELIMINARY until 0d's recheck",
           "git": git("rev-parse", "HEAD").decode().strip(), "stamp (PT)": stamp, "app": APP,
           "input": {"commit": RECORD[0], "path": RECORD[1], "sha256": RECORD[2]}, "boot": {"B": BOOT_B, "seed": BOOT_SEED},
           "positive_control": "the full-problem AUROC and interval reproduce the record's to 1e-12 in every cell (else the job stops)", "results": res}
    dst = HERE / "results" / f"honest_slices_{stamp}.json"
    dst.open("x").write(json.dumps(rec, indent=1))
    lines = []
    for cell, r in res.items():
        f = lambda m: f"{m['auroc']:.4f} [{m['ci95'][0]:.4f}, {m['ci95'][1]:.4f}] (n {m['n_pos']} vs {m['n_neg']}, mains {m['n_mains']}, failed {m['failed_draws']})"  # noqa: E731
        lines.append(f"[{cell}] full: sim vs B {f(r['full']['simh_vs_b'])}; A vs B {f(r['full']['a_vs_b'])}")
        for k in ("(1) sim vs B epoch 1", "(1) sim vs B epoch 2", "(2) sim vs B, B reweighted to sim's set shares", "(3) A vs B without S_old"):
            lines.append(f"  {k}: {f(r[k])}")
        for s_, v in r["(2) per set"].items():
            lines.append(f"  (2) {s_}: sim vs B {f(v['simh_vs_b'])}; A vs B {f(v['a_vs_b'])}")
    (HERE / "results" / f"honest_slices_{stamp}.printout.txt").open("x").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("->", dst)
