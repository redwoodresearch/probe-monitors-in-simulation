"""POST HOC follow-on to analyze_avg_vs_last.py (Codex on #1337): the paired contrast between readouts,
C = (mean-pooled arm - e2) - (last-token arm - e2), in points, for arm = as simulated, repaired and real e1 (control).

Both readouts score the same 939 cells and share one task-weight matrix W per reader (analyze_avg_vs_last.py, the
`default_rng(SEED)` line in main()), so each bootstrap draw gives a paired contrast. This script reuses that script's
loading, thresholds, catch and committed-reading gate unchanged, rebuilds W the same way, and first asserts that every
per-readout difference point and 95% range equals the as-run record (bd4a077e) exactly.
Rule, fixed before running: "the readouts differ" if C's 95% range excludes 0; otherwise "no difference shown".
1% leads; 5% is shown beside it because the as-run table has both rates.

  uv run --no-project --with numpy python experiments/2026-10-13_q2_avg_vs_last/analyze_paired_readouts.py \\
      --qwen DIR JSON DIR JSON --olmo DIR JSON DIR JSON --llama DIR JSON DIR JSON   (the as-run script's arguments)
"""
import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import analyze_avg_vs_last as A  # noqa: E402

AS_RUN = ("bd4a077e", "experiments/2026-10-13_q2_avg_vs_last/results/avg_vs_last_POSTHOC_2026-10-13_110005.json")
ARMS = {"as simulated": "as simulated - e2", "repaired": "repaired - e2", "real e1": "e1 - e2"}
RATES = (0.01, 0.05)
COLS = ["reader", "layers", "rate", "difference", "mean-pooled (points)", "last-token (points)", "C (points)", "C 95% low", "C 95% high", "reading"]


def main():
    me = Path(__file__).resolve()
    for p in (me, Path(A.__file__).resolve()):
        if subprocess.run(["git", "ls-files", "--error-unmatch", str(p)], cwd=HERE, capture_output=True).returncode \
                or subprocess.run(["git", "diff", "--quiet", "HEAD", "--", str(p)], cwd=HERE).returncode:
            raise SystemExit(f"{p.name} is not committed as is")
    git = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True, check=True, cwd=A.REPO).stdout  # noqa: E731
    asrun = json.loads(git("show", f"{AS_RUN[0]}:{AS_RUN[1]}"))
    ap = argparse.ArgumentParser()
    for r in A.READERS:
        ap.add_argument(f"--{r}", nargs=4, required=True)
    args = ap.parse_args()
    rel = lambda f: str(f.resolve().relative_to(next(q for q in f.resolve().parents if q.name == "experiments").parent))  # noqa: E731
    rows, checked = [], 0
    inputs = {f"sha256 of {rel(f)}": A.sha(f) for f in (Path(A.__file__), A.FIX939)}
    ci = lambda x: [float(v) for v in np.percentile(x, [2.5, 97.5])]  # noqa: E731
    for r in A.READERS:
        name = A.READERS[r]["name"]; runs, cfgs, _ = A.load(r, *getattr(args, r))
        for p in map(Path, getattr(args, r)):  # every input file, keyed by its repo path
            fs = [p, p.parent / json.loads(p.read_text())["scores file"]] if p.suffix == ".json" else \
                [p / "runs.json", *(p / f"scores_L{l}.npz" for l in A.READERS[r]["layers"])]
            inputs |= {f"sha256 of {rel(f)}": A.sha(f) for f in fs}
        if not set(asrun["readers"][name]["inputs sha256"].values()) <= set(inputs.values()):
            raise SystemExit(f"{name}: an input file differs from the as-run record's hashes")
        # the as-run main()'s population and W, line for line
        arm = np.array([x["arm"] for x in runs]); task = np.array([x["main_task"] for x in runs]); cell = np.array([x["cell"] for x in runs])
        fold = np.array([x["fold"] for x in runs]); hon = arm == "real-honest"
        sim_i = {c: i for i, c in enumerate(cell) if arm[i] == "sim-attack"}; e2_i = {c: i for i, c in enumerate(cell) if arm[i] == "real-attack-e2"}
        cells = sorted(set(sim_i) & set(e2_i)); ok = lambda S, i: np.isfinite(np.delete(S[i], fold[i])).all()  # noqa: E731
        assert len(cells) == 939
        si, ei = np.array([sim_i[x] for x in cells]), np.array([e2_i[x] for x in cells])
        e1i = np.array([i for i in range(len(runs)) if arm[i] == "real-attack-e1" and all(ok(c["S"], i) for c in cfgs)])
        hi = np.array([i for i in range(len(runs)) if hon[i] and all(ok(c["S"], i) for c in cfgs)])
        tasks = np.unique(task); tix = np.searchsorted(tasks, task)
        g = np.random.default_rng(A.SEED); W = np.stack([np.bincount(g.integers(0, len(tasks), len(tasks)), minlength=len(tasks)) for _ in range(A.B)]).astype(float)
        t = {"real e1": tix[e1i], "real e2": tix[ei], "as simulated": tix[si], "repaired": tix[si], "false alarms": tix[hi]}
        for alpha in RATES:
            base = {}
            for c in cfgs:
                T = A.thresholds(c, fold, hon, alpha)
                cc = {"real e1": A.catch(e1i, T, c["S"], fold), "real e2": A.catch(ei, T, c["S"], fold),
                      "as simulated": A.catch(si, T, c["S"], fold), "repaired": A.catch(si, T, c["Sf"], fold), "false alarms": A.catch(hi, T, c["S"], fold)}
                bt = {k: 100 * (W[:, t[k]] @ v) / W[:, t[k]].sum(1) for k, v in cc.items()}
                pts = {k: 100 * float(v.mean()) for k, v in cc.items()}
                A.gate(r, c, alpha, pts)
                want = asrun["readers"][name]["cells read"][f"{c['readout']} @ {c['layers']} @ {alpha}"]["differences"]
                for a_, dn in ARMS.items():
                    gap, b = pts[a_] - pts["real e2"], bt[a_] - bt["real e2"]
                    if [gap, ci(b)] != want[dn]["points, 95%"]:
                        raise SystemExit(f"gate: {name} {c['readout']} {c['layers']} {alpha} {dn} differs from the as-run record")
                    checked += 1; base[(c["readout"], c["layers"], a_)] = (gap, b)
            for lname in dict.fromkeys(c["layers"] for c in cfgs):
                for a_, dn in ARMS.items():
                    (gm, bm), (gl, bl) = base[("mean-pooled", lname, a_)], base[("last-token", lname, a_)]
                    rng_ = ci(bm - bl)
                    rows.append([name, lname, alpha, dn, gm, gl, gm - gl, *rng_, "the readouts differ" if rng_[0] > 0 or rng_[1] < 0 else "no difference shown"])
    head = git("rev-parse", "HEAD").strip()
    meta = {"POST HOC": True, "status": "preliminary until Codex", "git": head, "as-run record": f"{AS_RUN[0]}:{AS_RUN[1]}",
            "draws": A.B, "seed": A.SEED, "per-readout differences reproduced exactly": checked, "inputs": inputs,
            "rule": "the readouts differ if the contrast's 95% range excludes 0", "row columns": COLS}
    f1 = lambda x: f"{x:+.1f}"  # noqa: E731
    lines = ["POST HOC, preliminary until Codex. Paired contrast C = (mean-pooled difference) - (last-token difference), in points, same 939 cells,",
             f"shared main-task draws ({A.B}, seed {A.SEED}), thresholds fixed at the point estimate as in the as-run table ({AS_RUN[0]}).",
             f"All {checked} per-readout differences (points and 95% ranges) reproduce the as-run record exactly. 1% leads; 5% is beside it because the",
             "as-run table has both rates. Reading: 'the readouts differ' if C's 95% range excludes 0.", "",
             f"{'reader':<15}{'layers':<26}{'rate':<6}{'difference':<19}{'mean-pooled':>12}{'last-token':>11}  {'C [95%]':<22}reading"]
    for nm, lname, alpha, dn, gm, gl, c_, lo, hi_, reading in rows:
        cs = f"{f1(c_)} [{f1(lo)},{f1(hi_)}]"
        lines.append(f"{nm:<15}{lname:<26}{int(alpha * 100):>2}%   {dn:<19}{f1(gm):>12}{f1(gl):>11}  {cs:<22}{reading}")
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    body = "{\n \"meta\": " + json.dumps(meta, indent=1).replace("\n", "\n ") + ",\n \"rows\": [\n  " + ",\n  ".join(json.dumps(x) for x in rows) + "\n ]\n}\n"
    json.loads(body)
    with open(HERE / "results" / f"paired_readouts_POSTHOC_{stamp}.json", "x") as fh:
        fh.write(body)
    with open(HERE / "results" / f"paired_readouts_POSTHOC_{stamp}.txt", "x") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines)); print(HERE / "results" / f"paired_readouts_POSTHOC_{stamp}.json")


if __name__ == "__main__":
    main()
