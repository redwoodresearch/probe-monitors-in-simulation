"""09 / 2e (10-16 ~19:2x), P1 (c): are the epoch-2 units of the 5 declared E12 pairs that enter NO pooled pool (S_new / S_newmains H_A:
honest_a e2 vs its epoch-1 replicate on nm107, nm134, t361, t411, t413) byte-identical to their epoch-1 replicate's? Compared the way gate 2
compares: per probe score file (uncut, the locked INPUTS_FULL 110447's per-set and pools_ep2 overrides, sha-checked), a run's units as
(designated score, fold, five model scores) in file order, canonicalized like run_peraction311.pw_canon (float64, one NaN, -0 as 0, int64,
shape-prefixed), then sha256. Controls on the same machinery: honest_b nm305 (gate 2 found e1 == e2: must come out identical) and honest_b nm107
(compared by gate 2, no group: must come out different).
Reads score arrays ONLY to hash them, on Modal CPU; returns keys, counts, booleans and hashes, never a value.
usage: <modal> run run_e12_ident.py --ovdir <dir with the six override JSONs> --inputs-full <locked INPUTS_FULL json> --out <json>"""
import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

app = modal.App("probe-shift_aswin-surya_sub-fpr507prep_e12ident_kill-2026-11-14")
VOL = "probe-shift_aswin-surya_sub-convaudit_work_kill-2026-11-14"
image = modal.Image.debian_slim(python_version="3.12").pip_install("numpy==2.3.3")
TARGETS = [  # (world, cell, per-set override set, pools_ep2 arm prefixes, role)
    ("honest_a", "nm107_media_processing", "S_newmains", ("ep2_nm_",), "target"),
    ("honest_a", "nm134_media_processing", "S_newmains", ("ep2_nm_",), "target"),
    ("honest_a", "t361_file_indexing", "S_new", ("ep2_311_l1_", "ep2_311_l2_"), "target"),
    ("honest_a", "t411_media_processing", "S_new", ("ep2_311_l1_", "ep2_311_l2_"), "target"),
    ("honest_a", "t413_media_processing", "S_new", ("ep2_311_l1_", "ep2_311_l2_"), "target"),
    ("honest_b", "nm305_user_tracking", "S_newmains", ("ep2_nm_",), "control: gate 2 found identical"),
    ("honest_b", "nm107_media_processing", "S_newmains", ("ep2_nm_",), "control: gate 2 compared, no group"),
]


def canon(u):
    import numpy as np

    parts = []
    for a in u:
        a = np.asarray(a)
        if a.dtype.kind == "f":
            a = a.astype(np.float64)
            a = np.where(np.isnan(a), np.nan, a + 0.0)
        elif a.dtype.kind in "iub":
            a = a.astype(np.int64)
        else:
            raise SystemExit(f"unexpected dtype {a.dtype}")
        parts.append(repr(a.shape).encode() + b":" + np.ascontiguousarray(a).tobytes())
    return b"|".join(parts)


@app.function(image=image, cpu=(2, 4), memory=(4096, 16384), timeout=1800, retries=0,
              volumes={"/conv": modal.Volume.from_name(VOL, version=2).read_only()})
def hash_units(batch: list) -> list:
    """batch: [{path, sha256, keys: [world|cell, ...]}] -> per file and wanted key: units found (count) and the canonical sha256 (never values)."""
    import io

    import numpy as np

    out = []
    for f in batch:
        b = (Path("/conv") / f["path"].lstrip("/")).read_bytes()
        if hashlib.sha256(b).hexdigest() != f["sha256"]:
            out.append({"path": f["path"], "error": "sha256 differs"})
            continue
        z = np.load(io.BytesIO(b))
        k = z["key"].astype(str)
        n = len(k)
        fo = z["fold"].astype(np.int64) if "fold" in z.files else np.zeros(n, np.int64)
        sbm = (z["score_by_model"].astype(np.float64) if "score_by_model" in z.files
               else z["score_by_fold"].astype(np.float64) if "score_by_fold" in z.files else np.full((n, 5), np.nan))
        s = z["score"].astype(np.float64) if "score" in z.files else np.where(fo >= 0, sbm[np.arange(n), np.clip(fo, 0, 4)], np.nan)
        for want in f["keys"]:
            w, c = want.split("|", 1)
            ix = np.flatnonzero((k == f"{w}|{c}") | (k == f"{w}_e2|{c}"))
            keys_seen = sorted(set(k[ix].tolist()))
            out.append({"path": f["path"], "want": want, "n_units": int(len(ix)), "keys_seen": keys_seen,
                        "sha256": hashlib.sha256(canon((s[ix], fo[ix], sbm[ix]))).hexdigest() if len(ix) else None})
    return out


@app.local_entrypoint()
def main(ovdir: str, inputs_full: str, out: str, dry: bool = False):
    IF = json.loads(Path(inputs_full).read_bytes())
    ov = {}
    for k in ("S_new|C", "S_new|R", "S_newmains|C", "S_newmains|R", "pools_ep2|C", "pools_ep2|R"):
        b = (Path(ovdir) / f"ov_{k.replace('|', '_')}.json").read_bytes()
        if hashlib.sha256(b).hexdigest() != IF["overrides"][k]["sha256"]:
            raise SystemExit(f"{k}: override copy differs from the locked pin")
        ov[k] = json.loads(b)
    want = {}  # (path, sha) -> set of wanted keys
    plan = []  # (target, readout, file name, side, arm, path)
    for w, c, st, pre, role in TARGETS:
        for ro in ("C", "R"):
            e2 = {a: fs for a, fs in ov[f"pools_ep2|{ro}"]["arms"].items() if a.endswith(f"__{w}") and a.startswith(pre)}
            e1 = {a: fs for a, fs in ov[f"{st}|{ro}"]["arms"].items() if a.endswith(f"__{w}")}
            for side, arms in (("e2", e2), ("e1", e1)):
                for a, fs in arms.items():
                    for fn, e in fs.items():
                        want.setdefault((e["path"], e["sha256"]), set()).add(f"{w}|{c}")
                        plan.append({"world": w, "cell": c, "role": role, "readout": ro, "file": fn, "side": side, "arm": a, "path": e["path"]})
    files = [{"path": p, "sha256": h, "keys": sorted(ks)} for (p, h), ks in sorted(want.items())]
    if dry:  # the plan only (keys and counts), nothing read on Modal
        print(json.dumps({"files": len(files), "plan_rows": len(plan), "by_target": {f"{w}|{c}": sum(1 for p in plan if (p["world"], p["cell"]) == (w, c))
                                                                                 for w, c, *_ in TARGETS}}, indent=1))
        return
    batches = [files[i::32] for i in range(32)]
    res = [r for rs in hash_units.map([b for b in batches if b]) for r in rs]
    errs = [r for r in res if "error" in r]
    got = {(r["path"], r["want"]): r for r in res if "error" not in r}
    rows = []
    for (w, c, ro, fn), grp in sorted(_group(plan).items()):
        sides = {"e1": [], "e2": []}
        for p in grp:
            r = got.get((p["path"], f"{w}|{c}"))
            if r and r["n_units"]:
                sides[p["side"]].append({"arm": p["arm"], "n_units": r["n_units"], "keys_seen": r["keys_seen"], "sha256": r["sha256"]})
        e1s, e2s = {x["sha256"] for x in sides["e1"]}, {x["sha256"] for x in sides["e2"]}
        rows.append({"world": w, "cell": c, "role": grp[0]["role"], "readout": ro, "file": fn, "e1": sides["e1"], "e2": sides["e2"],
                     "compared": bool(e1s and e2s), "identical": bool(e1s and e1s == e2s)})
    summ = {}
    for r in rows:
        s_ = summ.setdefault(f"{r['world']}|{r['cell']}", {"role": r["role"], "files_compared": 0, "files_identical": 0, "files_one_side_only": 0})
        s_["files_compared"] += r["compared"]
        s_["files_identical"] += r["identical"]
        s_["files_one_side_only"] += (not r["compared"]) and bool(r["e1"] or r["e2"])
    rec = {"what": run_doc(), "made_pt": datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d %H:%M:%S %Z"),
           "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
           "inputs_full_sha256": hashlib.sha256(Path(inputs_full).read_bytes()).hexdigest(),
           "overrides": {k: IF["overrides"][k] for k in ov}, "files_read": len(files), "errors": errs, "summary": summ, "rows": rows}
    Path(out).write_text(json.dumps(rec, indent=1) + "\n")
    print(json.dumps({"files_read": len(files), "errors": len(errs), "summary": summ}, indent=1))


def run_doc():
    return (__doc__ or "").split("\n")[0]


def _group(plan):
    g = {}
    for p in plan:
        g.setdefault((p["world"], p["cell"], p["readout"], p["file"]), []).append(p)
    return g
