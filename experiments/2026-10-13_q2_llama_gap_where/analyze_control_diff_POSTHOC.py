"""POST HOC (ADDENDUM_DIFF_POSTHOC.md here): D = all-rows mean shift of repaired-plus minus that of the first real run,
both against the second real run's rows, over the SAME registered 2,000 main-task draws (seed 20261011). One Modal CPU
container. The arms are built as analyze_control_POSTHOC.py builds them, and the shift is decompose()'s "mean shift"
for "all rows" (analyze_meanpool_POSTHOC.py), copied without change of formula. Gate: each arm reproduces the committed
control record's 1% all-rows shift (point, 95%, valid draws) within 1e-12, else nothing is read.

Inputs are the as-run files, staged flat in one local folder and checked by sha256 in the container:
  INPUTS=<folder> uv run --no-project --with modal modal run experiments/2026-10-13_q2_llama_gap_where/analyze_control_diff_POSTHOC.py
"""
import hashlib
import importlib.util
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).resolve().parent  # modal run imports this file by a relative path
APP = "probe-shift_aswin-surya_sub-q2-gwdiff_job_kill-2026-11-13"
CONTROL = "control_POSTHOC_2026-10-13_131340.json"
PINS = {  # results/s3_manifest_v1.json (this folder) and the control record's "sha256 of inputs"
    CONTROL: "4276ab62890c97299fcf3f2394491e56cba26298d2272c21e19e95329132cdc9",
    "analyze_gap_where.py": "8b2aae2ba23e92f23d58bab5b65fb21019d18d19e3e1ebca2214437709949258",
    "rows_2026-10-13_110759.json": "a61956fd25f2344a9994750bebfed8d040d2bee9a9c5e7464eec7ddd3c87b81c",
    "rows_2026-10-13_110759_manifest.json": "8d848210cd509b68ecb409ff5916f889b83e7024c50daa8faae7a72b2e26d161",
    "rows_e1_POSTHOC_2026-10-13_131102.json": "ca7453802770eb41feeefa5f116fc53ddf87682590c57dfaa75cf3dab77f249c",
    "plus_reading_llama_2026-10-13_024135.json": "5d1182c2528efb1062884e6bfd9ec4c79d421396493acfdad773d49cca9b7258",
    "fix_scores_olmo_2026-10-12_153542.json": "f77dc81cc81e95f93ce2a2fd90a76423faf20a452298cbe2919760730e4dbd65",
    "plus_scores_llama_2026-10-13_024006.json": "9093ab786091cd01b6bf6facba96a44d67dcba57ecfbbb1aabb02fad703f4641"}
ARM_KEYS = ("repaired-plus", "first real run")
image = modal.Image.debian_slim(python_version="3.12").pip_install("numpy==2.3.3")
if os.environ.get("INPUTS"):
    image = image.add_local_dir(os.environ["INPUTS"], "/in")
app = modal.App(APP, image=image)


@app.function(cpu=2, memory=8192, timeout=600, retries=0)
def diff():
    import numpy as np
    IN = Path("/in")
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()  # noqa: E731
    bad = [n for n, h in PINS.items() if sha(IN / n) != h]
    if bad:
        raise SystemExit(f"inputs not their pinned sha256: {bad}")
    ctl = json.loads((IN / CONTROL).read_text())
    if any(ctl["sha256 of inputs"].get(n) != PINS[n] for n in PINS if n not in (CONTROL, "analyze_gap_where.py")):
        raise SystemExit("the control record was computed from other inputs")
    spec = importlib.util.spec_from_file_location("registered", IN / "analyze_gap_where.py")
    reg = importlib.util.module_from_spec(spec); spec.loader.exec_module(reg)
    reg_rows, e1_rows = IN / "rows_2026-10-13_110759.json", IN / "rows_e1_POSTHOC_2026-10-13_131102.json"
    if sha(reg_rows) != json.loads((IN / "rows_2026-10-13_110759_manifest.json").read_text())["rows record sha256"]:
        raise SystemExit("the registered rows record isn't its manifest's")
    rrec, erec = json.loads(reg_rows.read_text()), json.loads(e1_rows.read_text())
    Xr, X1 = np.load(IN / rrec["rows file"]), np.load(IN / erec["rows file"])
    if sha(IN / rrec["rows file"]) != rrec["rows sha256"] or sha(IN / erec["rows file"]) != erec["sha256 of the rows file"] \
            or erec["sha256 of inputs"][reg_rows.name] != sha(reg_rows):
        raise SystemExit("a rows file isn't its record's, or the first real run's rows were built from other registered rows")
    lay = tuple(rrec["readers (probe run, layer per test fold)"]["llama"][1])
    if tuple(erec["layer per test fold"]) != lay:
        raise SystemExit("the first real run's rows use other layers")
    R = reg.reader_llama_olmo("llama", IN, IN / "fix_scores_olmo_2026-10-12_153542.json", IN / "plus_scores_llama_2026-10-13_024006.json",
                              json.loads((IN / "plus_reading_llama_2026-10-13_024135.json").read_text()), lay)
    runs = R["runs"]; F, ctix = R["F"], R["ctix"]
    cpos = {runs[int(i)]["cell"]: c for c, i in enumerate(R["si"])}
    # the registered rows: second real run (ar) and repaired-plus (~ar), as analyze_control_POSTHOC.py lines 125-130
    U = rrec["units (per reader: runs.json index, key sha256, source, fold; arrays' unit indexes this)"]["llama"]
    upos = {int(i): ("e2", c) for c, i in enumerate(R["ei"])} | {int(i): ("rep", c) for c, i in enumerate(R["si"])}
    u_e2 = np.array([upos[i][0] == "e2" for i in U["runs.json index"]]); u_cell = np.array([upos[i][1] for i in U["runs.json index"]])
    u = Xr["llama__unit"]; ar, ce = u_e2[u], u_cell[u]
    own = np.arange(5)[None, :] == F[ce][:, None]
    L0 = np.where(own, 0.0, Xr["llama__logit"].astype(np.float64))
    # the first real run's rows, as analyze_control_POSTHOC.py lines 150-164
    E = erec["units (runs.json index, arm, fold; arrays' unit indexes this)"]
    e_idx, e_arm, e_fold = np.array(E["runs.json index"]), np.array(E["arm"]), np.array(E["fold"])
    if any(runs[int(i)]["arm"] != {"e1": "real-attack-e1", "e2": "real-attack-e2"}[a_] for i, a_ in zip(e_idx, e_arm)):
        raise SystemExit("the first-real-run job's units aren't the runs.json arms they name")
    e_cell = np.array([cpos[runs[int(i)]["cell"]] for i in e_idx])
    if not (e_fold == F[e_cell]).all():
        raise SystemExit("a first-real-run unit's fold differs from its cell's")
    u1 = X1["e1__unit"]
    if not (e_arm[u1] == "e1").all():
        raise SystemExit("a first-real-run row from a second-real-run unit")
    c1 = e_cell[u1]; own1 = np.arange(5)[None, :] == F[c1][:, None]
    L1 = X1["e1__logit"].astype(np.float64)
    if not np.isfinite(L1[~own1]).all():
        raise SystemExit("a non-finite first-real-run logit")
    L1 = np.where(own1, 0.0, L1)
    arms = {"repaired-plus": (ar, ce, L0),
            "first real run": (np.r_[np.ones(ar.sum(), bool), np.zeros(len(u1), bool)], np.r_[ce[ar], c1], np.r_[L0[ar], L1])}

    def shift_fn(a_, ce_, L0_):
        """decompose()'s all-rows "mean shift" (sums(), offsets(allc) and the shift line), formulas unchanged."""
        allm = np.ones(len(a_), bool)
        nce, ncr = (np.bincount(ce_[allm & a], minlength=939).astype(float) for a in (a_, ~a_))
        sce, scr = (np.stack([np.bincount(ce_[allm & a], weights=L0_[allm & a, v], minlength=939) for v in range(5)], 1) for a in (a_, ~a_))
        if (nce == 0).any() or (ncr == 0).any():
            raise SystemExit("a run with no rows")
        keep = np.ones(939, bool); m = ~np.eye(5, dtype=bool)

        def at(wc):
            de = np.bincount(F, weights=wc * keep * nce, minlength=5); dr = np.bincount(F, weights=wc * keep * ncr, minlength=5)
            if (de == 0).any() or (dr == 0).any():
                return None
            me = np.stack([np.bincount(F, weights=wc * keep * sce[:, v], minlength=5) for v in range(5)], 1) / de[:, None]
            mr = np.stack([np.bincount(F, weights=wc * keep * scr[:, v], minlength=5) for v in range(5)], 1) / dr[:, None]
            O, wt = me - mr, np.minimum(de, dr)
            return float((-O * wt[:, None])[m].sum() / np.broadcast_to(wt[:, None], (5, 5))[m].sum())
        return at

    ones = np.ones(len(R["draws"][0]))
    out = {"POST HOC": True, "addendum": "ADDENDUM_DIFF_POSTHOC.md (this folder)", "sha256 of inputs": PINS,
           "conventions": "registered: by cell, 2,000 main-task draws, seed 20261011, the same draws for both arms; "
                          "shift = arm - second real run (logits), all rows; D = shift(repaired-plus) - shift(first real run)",
           "arms": {}, "gate": {}}
    pts, per = {}, {}
    for name in ARM_KEYS:
        at = shift_fn(*arms[name])
        pts[name] = at(ones[ctix].astype(float))
        per[name] = [at(w[ctix].astype(float)) for w in R["draws"]]
        v = [x for x in per[name] if x is not None]
        got = {"point": pts[name], "95%": [float(x) for x in np.percentile(v, [2.5, 97.5])], "valid draws": len(v)}
        want = ctl["arms"][name]["0.01"]["categories"]["all rows"]["mean shift"]
        ok = got["valid draws"] == want["valid draws"] and abs(got["point"] - want["point"]) < 1e-12 \
            and all(abs(g - w) < 1e-12 for g, w in zip(got["95%"], want["95%"]))
        out["arms"][name] = got; out["gate"][f"{name}: reproduces the committed 1% all-rows mean shift (1e-12)"] = want
        if not ok:
            raise SystemExit(f"gate failed for {name}: {got} vs committed {want}")
    both = [(a, b) for a, b in zip(per[ARM_KEYS[0]], per[ARM_KEYS[1]]) if a is not None and b is not None]
    d = np.array([a - b for a, b in both])
    D = {"point": pts[ARM_KEYS[0]] - pts[ARM_KEYS[1]], "valid draws": len(d)}
    D["95%"] = [float(x) for x in np.percentile(d, [2.5, 97.5])] if len(d) >= reg.MIN_VALID else f"not reported: {len(d)} valid draws"
    out["D = shift(repaired-plus) - shift(first real run)"] = D
    out["across draws (descriptive)"] = {"correlation of the two arms' shifts": float(np.corrcoef(np.array(both).T)[0, 1]),
                                         "sd of D": float(d.std(ddof=1)), "sd of each arm's shift": [float(np.std([x[i] for x in both], ddof=1)) for i in (0, 1)]}
    if not isinstance(D["95%"], list):
        out["reading (addendum)"] = "not read: too few valid draws"
    elif D["95%"][1] < 0 or D["95%"][0] > 0:
        out["reading (addendum)"] = "simulation-specific (descriptive); the difference's interval excludes 0"
    else:
        out["reading (addendum)"] = "the repaired-plus shift's interval excludes 0 and the first real run's doesn't; their difference includes 0"
    return out


@app.local_entrypoint()
def main():
    me = Path(__file__).resolve()
    for f in (me, HERE / "ADDENDUM_DIFF_POSTHOC.md"):
        if subprocess.run(["git", "ls-files", "--error-unmatch", str(f)], cwd=HERE, capture_output=True).returncode \
                or subprocess.run(["git", "diff", "--quiet", "HEAD", "--", str(f)], cwd=HERE).returncode:
            raise SystemExit(f"{f.name} differs from HEAD: commit it before running")
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip()
    out = diff.remote()
    out = {"git": head, "app": APP} | out
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    with open(dest := HERE / "results" / f"control_diff_POSTHOC_{stamp}.json", "x") as fh:
        fh.write(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: out[k] for k in ("arms", "D = shift(repaired-plus) - shift(first real run)", "across draws (descriptive)", "reading (addendum)")}, indent=1))
    print(dest)
