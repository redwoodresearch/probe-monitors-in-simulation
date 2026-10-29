"""G3 follow-ups, PLAN.md Amendment 1 to (1) (POST HOC, preliminary until Codex): edit vs non-edit, newline-ending vs not within
non-edit outputs, and position5 within str_replace and within bash, for second real vs as simulated and the second real vs first
real control. Local, $0. The same rows record and engine as analyze_breakdown_g3.py (its decompose(), caught() and inputs,
imported unchanged); gate first: both pairs' G at 1% and 5% and the main pair's range equal the committed breakdown's
(breakdown_POSTHOC_2026-10-13_132818.json) within 1e-9, so the splits sit on the same numbers.

  uv run --no-project --python 3.12 --with numpy python experiments/2026-10-13_g3_followups/analyze_breakdown_amend1.py \\
      --rows results/rows_g3_2026-10-13_132325.json
"""
import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analyze_breakdown_g3 as AB  # noqa: E402

BREAKDOWN = ("breakdown_POSTHOC_2026-10-13_132818.json", "a5b87e9f")
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()  # noqa: E731


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--rows", required=True, type=Path); a = ap.parse_args()
    for f in (Path(__file__).resolve(), HERE / "PLAN.md", HERE / "analyze_breakdown_g3.py", HERE / "results" / BREAKDOWN[0]):
        if subprocess.run(["git", "ls-files", "--error-unmatch", str(f)], cwd=HERE, capture_output=True).returncode \
                or subprocess.run(["git", "diff", "--quiet", "HEAD", "--", str(f)], cwd=HERE).returncode:
            raise SystemExit(f"{f.name} differs from HEAD: commit it before running")
    bd = json.loads((HERE / "results" / BREAKDOWN[0]).read_text())
    if bd["inputs sha256"][a.rows.name] != sha(a.rows):
        raise SystemExit("not the rows record the breakdown read")
    for n, p in ((AB.Y_REC, AB.G3R / AB.Y_REC), (AB.X_REC, AB.G3R / AB.X_REC), (AB.SIM_REC, AB.G3R / AB.SIM_REC), ("forecast.json", AB.G3R.parent / "forecast.json")):
        if sha(p) != AB.PINS[n]:
            raise SystemExit(f"{n} is not the pinned file")
    fc = json.loads((AB.G3R.parent / "forecast.json").read_text())["frozen"]; x = json.loads((AB.G3R / AB.X_REC).read_text())
    sd = AB.G3R / f"run_scores_{AB.FIT}"
    runs = json.loads((sd / "runs.json").read_text()); arm = np.array([r["arm"] for r in runs]); fold = np.array([int(r["fold"]) for r in runs])
    N = {l: np.load(sd / f"scores_L{l}.npz")["output_end__mean__task_nested"] for l in set(AB.LAYER_BY_FOLD)}
    by = {ar_: {r["cell"]: i for i, r in enumerate(runs) if r["arm"] == ar_} for ar_ in ("real-attack-e1", "real-attack-e2")}
    cells = sorted(x["caught, real e2 by cell (R)"])
    ei, e1i = np.array([by["real-attack-e2"][c] for c in cells]), np.array([by["real-attack-e1"][c] for c in cells]); F = fold[ei]
    stored = lambda idx: np.stack([N[AB.LAYER_BY_FOLD[fold[i]]][fold[i], :, i] for i in idx])  # noqa: E731
    S_e2, S_e1 = stored(ei), stored(e1i)
    srec = json.loads((AB.G3R / AB.SIM_REC).read_text()); Z = np.load(AB.G3R / srec["scores file"])
    kc = {k: c for k, c in srec["keys and cells"]}; pos = {kc[k]: j for j, k in enumerate(Z["keys"])}
    S_sim = np.stack([Z[f"L{AB.LAYER_BY_FOLD[F[t]]}"][F[t], :, pos[c]] for t, c in enumerate(cells)])
    T1 = np.array([[np.nan if v is None else v for v in row] for row in fc["thresholds at 1% [f][v]"]], float)
    T5 = np.full((5, 5), np.nan); hon = arm == "real-honest"
    for f in range(5):
        for v in range(5):
            if v != f:
                T5[f, v] = AB.threshold(N[AB.LAYER_BY_FOLD[f]][f, v][hon & (fold == v)], 0.05)
    Ts = {0.01: T1, 0.05: T5}
    e2r = {r["cell"]: r for r in runs if r["arm"] == "real-attack-e2"}
    tasks = sorted({e2r[c]["main_task"] for c in cells}); tix = np.array([tasks.index(e2r[c]["main_task"]) for c in cells])
    g = np.random.default_rng(AB.SEED); W = np.stack([np.bincount(g.integers(0, len(tasks), len(tasks)), minlength=len(tasks)) for _ in range(AB.B)]).astype(float)
    rr = json.loads(a.rows.read_text()); X = np.load(a.rows.parent / rr["rows file"])
    if sha(a.rows.parent / rr["rows file"]) != rr["rows sha256"]:
        raise SystemExit("the rows file is not its record's")
    u = X["row unit"]; L = X["row logit"].astype(np.float64)
    uarm = np.array(rr["units"]["arm per unit"]); ucell = np.array(rr["units"]["cell per unit"])
    own = np.arange(5)[None, :] == F[ucell[u]][:, None]
    L0 = np.where(own, 0.0, L); rarm = uarm[u]; rcell = ucell[u]
    fam, end, p5 = X["row family"], X["row ending"], X["row position5"]
    edit, nonedit = np.isin(fam, (2, 3, 4)), np.isin(fam, (0, 1, 6))
    cats = {"edit {create, str_replace, insert}": edit, "non-edit {bash, view, other}": nonedit,
            "non-edit, ending newline": nonedit & (end == 1), "non-edit, not ending in a newline": nonedit & (end != 1)}
    for fk, fname in ((3, "str_replace"), (0, "bash")):
        for k, lab in enumerate(("first 5", "middle", "last 5")):
            cats[f"{fname}, {lab}"] = (fam == fk) & (p5 == k)
    out = {"POST HOC": True, "status": "preliminary until Codex", "plan": "PLAN.md Amendment 1 to (1)",
           "git": subprocess.run(["git", "rev-parse", "HEAD"], cwd=HERE, capture_output=True, text=True).stdout.strip(),
           "inputs sha256": {a.rows.name: sha(a.rows), BREAKDOWN[0]: sha(HERE / "results" / BREAKDOWN[0])},
           "note (newline row)": "Seed reads each output at its last non-whitespace content token, so an output's own trailing newline comes after its "
                                 "readout: it can move later outputs' scores through the history, never its own",
           "editor-other": "in neither set (0.1% of rows)", "pairs": {}}
    for pair, other, S_o in (("second real vs as simulated", "sim", S_sim), ("second real vs first real (control)", "e1", S_e1)):
        m = (rarm == "e2") | (rarm == other)
        res, _ = AB.decompose(L0[m], rarm[m] == "e2", rcell[m], F, {c: v[m] for c, v in cats.items()}, Ts, W, tix, S_e2, S_o)
        ref = bd["(1b, 1c) decomposition"][pair]
        for r_ in ("0.01", "0.05"):
            if abs(res[r_]["G (points)"]["point"] - ref[r_]["G (points)"]["point"]) > 1e-9 or \
                    max(abs(res[r_]["G (points)"]["95%"][i] - ref[r_]["G (points)"]["95%"][i]) for i in (0, 1)) > 1e-9:
                raise SystemExit(f"gate: {pair} at {r_}: G differs from the committed breakdown's")
        out["pairs"][pair] = res
    m = out["pairs"]["second real vs as simulated"]["0.01"]["categories"]
    pt = lambda c, k: m[c][k]["point"] if isinstance(m[c][k]["point"], float) else None  # noqa: E731
    out["predictions (PLAN.md Amendment 1, at 1%)"] = {
        "non-edit A/G in [0.15, 0.55]": [pt("non-edit {bash, view, other}", "A/G"), None if pt("non-edit {bash, view, other}", "A/G") is None else 0.15 <= pt("non-edit {bash, view, other}", "A/G") <= 0.55],
        "within non-edit, newline-ending and other mean shifts within 0.1 logits": [pt("non-edit, ending newline", "mean shift"), pt("non-edit, not ending in a newline", "mean shift"),
                                                                                    abs(pt("non-edit, ending newline", "mean shift") - pt("non-edit, not ending in a newline", "mean shift")) < 0.1],
        **{f"{t}: first 5 and last 5 shifts within 0.15 logits": [pt(f"{t}, first 5", "mean shift"), pt(f"{t}, last 5", "mean shift"),
                                                                  abs(pt(f"{t}, first 5", "mean shift") - pt(f"{t}, last 5", "mean shift")) < 0.15] for t in ("str_replace", "bash")}}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    with open(dest := HERE / "results" / f"breakdown_amend1_POSTHOC_{stamp}.json", "x") as fh:
        fh.write(json.dumps(out, indent=1, default=float) + "\n")
    for pair, res in out["pairs"].items():
        for r_ in ("0.01", "0.05"):
            print(f"== {pair} at {float(r_):.0%}: G {res[r_]['G (points)']['point']:+.2f}")
            for c, e in res[r_]["categories"].items():
                f2 = lambda s: "n/a" if not isinstance(s["point"], float) else f"{s['point']:+.3f}" + (f" [{s['95%'][0]:+.3f}, {s['95%'][1]:+.3f}]" if isinstance(s["95%"], list) else "")  # noqa: E731
                print(f"  {c:38s} rows {e['rows (second real, other)']} share {[round(100 * v, 1) for v in e['share of rows (second real, other)']]} shift {f2(e['mean shift'])} A {f2(e['A'])} A/G {f2(e['A/G'])}")
    print(json.dumps(out["predictions (PLAN.md Amendment 1, at 1%)"])); print(dest)


if __name__ == "__main__":
    main()
