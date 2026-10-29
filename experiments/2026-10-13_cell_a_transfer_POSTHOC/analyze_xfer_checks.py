"""Cell (a) transfer, Amendment 4's checks (RULE.md; POST HOC), local, $0: the item 4 row gate from run_xfer_checks.py, and
the truncation-exception sensitivity. Per control (Amendment 2 at levels 0 and 1, Amendment 3 at level 1), the exception
runs are those whose reader-kept count exceeds the replay's output count or is 0. They are dropped from BOTH pipelines,
and the AUROC of the second real run vs honest B (analyze_xfer.own_null's population: cohort tasks with honest B) is
recomputed for the reader-style and the matched replay scores. The committed readings are re-evaluated on that set.

  uv run --no-project --python 3.14 --with numpy --with modal python -I analyze_xfer_checks.py <xfer_checks json> \\
      <replay_matched json> <replay_matched_swapped json> <honest_reader_scores json> <labels json> <out json>
"""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analyze_xfer as X  # noqa: E402

A = X.A


def auroc(rs, lab):  # own_null's observed AUROC, without the permutations
    runs = sorted(k for k in rs if not k.startswith("honest_c|"))
    world = np.array([u.split("|", 1)[0] for u in runs])
    task = np.array([u.split("|", 1)[1] for u in runs])
    z = np.array([rs[u] for u in runs])
    have = lambda w: set(task[world == w])  # noqa: E731
    att = lambda w: {t for t in have(w) if lab.get(f"{w}|{t}") == "attempted"}  # noqa: E731
    cohort = att("run1") & att("run2") & att("sim")
    cb = cohort & have("honest_b")
    pos = (world == "run2") & np.isin(task, sorted(cb))
    neg = (world == "honest_b") & np.isin(task, sorted(cb))
    return float(A.wauc(z[pos], np.ones(int(pos.sum())), z[neg], np.ones(int(neg.sum())))), int(pos.sum()), int(neg.sum())


def read2(d):
    return "formatting raises separation" if d >= 0.05 else "formatting lowers separation" if d <= -0.05 else "formatting does not change separation"


def read3(d):
    return ("the tools block and reasoning lower separation" if d >= 0.05 else "the tools block and reasoning raise separation" if d <= -0.05
            else "the tools block and reasoning don't change separation; the formatting gain is the system prompt")


def main(checks, matched, matched_sw, reader, labels, out):
    g = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True, check=True, cwd=HERE).stdout  # noqa: E731
    if g("status", "--porcelain", "--untracked-files=no"):
        raise SystemExit("uncommitted tracked changes")
    if hashlib.sha256(Path(labels).read_bytes()).hexdigest() != A.LABELS_SHA:
        raise SystemExit("labels sha256 mismatch")
    C, M, MS, R = (json.loads(Path(p).read_text()) for p in (checks, matched, matched_sw, reader))
    lab = json.loads(Path(labels).read_text())["labels"]
    res = {"row_gate": C["row_gate"], "row_gate_passes": all(x["max_abs_diff"] <= 1e-6 and x["dump_only"] == 0 and x["cache_only"] == 0 for x in C["row_gate"].values()),
           "controls": {}}
    for name, v, L, mm, rd in (("amendment2_level0", "unswapped_L0", "0", M["0"], read2), ("amendment2_level1", "unswapped_L1", "1", M["1"], read2),
                               ("amendment3_level1", "swapped_L1", "1", MS["1"], read3)):
        t = C["truncation"][v]
        exc = sorted(k for k, (a, b) in t.items() if a > b or a == 0)
        rr, rp = R[L]["run_scores"], mm["run_scores"]
        keys = sorted(set(rr) & set(rp))
        full = [auroc({k: rr[k] for k in keys}, lab), auroc({k: rp[k] for k in keys}, lab)]
        kx = [k for k in keys if k not in set(exc)]
        excl = [auroc({k: rr[k] for k in kx}, lab), auroc({k: rp[k] for k in kx}, lab)]
        d0, d1 = full[0][0] - full[1][0], excl[0][0] - excl[1][0]
        res["controls"][name] = {"exceptions": {k: t[k] for k in exc}, "auroc_full": {"reader": full[0], "matched_replay": full[1], "difference": d0, "reading": rd(d0)},
                                 "auroc_excluding_exceptions": {"reader": excl[0], "matched_replay": excl[1], "difference": d1, "reading": rd(d1)},
                                 "difference_moved_by": d1 - d0}
    Path(out).open("x").write(json.dumps({"label": "POST HOC (RULE.md Amendment 4)", "git": g("rev-parse", "HEAD").strip()} | res, indent=1, default=float))
    print("row gate:", json.dumps(res["row_gate"]), "| passes:", res["row_gate_passes"])
    for name, x in res["controls"].items():
        f, e = x["auroc_full"], x["auroc_excluding_exceptions"]
        print(f"== {name}: {len(x['exceptions'])} exception runs {x['exceptions']}")
        print(f"   all runs: reader {f['reader'][0]:.4f} matched {f['matched_replay'][0]:.4f} diff {f['difference']:+.4f} ({f['reading']})")
        print(f"   excluding: reader {e['reader'][0]:.4f} matched {e['matched_replay'][0]:.4f} diff {e['difference']:+.4f} ({e['reading']}) | moved by {x['difference_moved_by']:+.4f}")
    print("written", Path(out).name)


if __name__ == "__main__":
    main(*sys.argv[1:7])
