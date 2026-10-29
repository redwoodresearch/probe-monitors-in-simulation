"""S_paper493 supersession markers, v2 (Amendment 49a; 09 via 2e 10-16 03:3x). The Sol launcher's supersede_sp493.py (@ 9b0a5ee8a) stays unchanged;
its run stopped (results/superseded_sp493_2026-10-16_030149.json, nothing written) because the six ATTACK pairs' shared rows' score bytes differ.
The diagnostics show why: the activations are byte-identical (all 482 shared epochs at L63; all 64 layers for two keys) and the scores differ by at
most 1.78e-14, from the scorer's row grouping. This tool marks the same 14 pairs, read from that record and tools_sp493/supersede_pairs_r4.json @
9b0a5ee8a, with one change for the attack pairs only: on shared rows the score fields (score, score_by_model) may differ by at most 1.8e-14, both
finite (a non-finite score must be byte-identical). Everything else is Amendment 49's byte-exact rule, for every pair: the key sets differ by
recovered cells only, nothing removed, folds equal on shared keys, every compared file is its manifest's (sha256), row identities (key, unit,
call_index) unique and equal, every other field identical. The 8 honest pairs keep the byte rule for every field and get the ordinary Amendment 49
marker; the 6 attack pairs get 09's marker text. All or nothing: every pair is checked first; if any fails, no marker is written; if a write fails,
the markers this run created are removed again (Codex sv2 r1). --dry writes no marker (only its own local check record).
No score is returned or written (counts and difference magnitudes only).

  PYTHONPATH=src <modal> run experiments/2026-10-16_supersede_diag/supersede_sp493_v2.py --dry   # check only
  PYTHONPATH=src <modal> run experiments/2026-10-16_supersede_diag/supersede_sp493_v2.py         # check, then write the 14 markers
"""

import hashlib
import io
import json
import re
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).resolve().parent
SRC_COMMIT = "9b0a5ee8a"
Q35 = "experiments/2026-10-14_q35_sol_avg_POSTHOC"
STOPPED = f"{Q35}/results/superseded_sp493_2026-10-16_030149.json"
PAIRS_R4 = f"{Q35}/tools_sp493/supersede_pairs_r4.json"
DIAG = [{"commit": "5641f4c79", "path": "experiments/2026-10-16_supersede_diag/results/supersede_diag_2026-10-16_032201.json",
         "sha256": "0b46da1cf0318840f91c953204bded7b4a71ee2c4b96a96e2072f4a18966bc6c"},
        {"commit": "2652c4c15", "path": "experiments/2026-10-16_supersede_diag/results/supersede_diag_l63_2026-10-16_032848.json",
         "sha256": "954f978399b46c7edfcb9b4aa15c764306b291fcb7992eaa6ed120c0435dec09"}]
ATTACK_MARKER = "superseded (re-extracted; activations byte-identical; scores differ ≤ 1.8e-14 from scorer row grouping)"  # 09's text, exactly
TOL = 1.8e-14  # attack pairs, score fields only (09 via 2e 03:3x)
SCORE_FIELDS = ("score", "score_by_model")
OUTVOL = "probe-shift_aswin-surya_sub-convaudit_work_kill-2026-11-14"
SP493 = re.compile(r"^sp493_(?:a|b|c)_(?:main_v1(?:_try[0-9]+)?__(?:attack|honest_a|honest_b)|run2_v1(?:_try[0-9]+)?__attack)$")  # as v1
RECOVERABLE = {"t619_model_registry", "t699_model_registry", "t699_model_registry_r2"}  # as v1 (09)
FILES = {"sol": ("manifest_sol.json", ("a_tight__avg_top2.npz", "a_tight__cmd_A.npz", "c_asis_tight__avg_top2.npz", "c_asis_tight__cmd_A.npz")),
         "own_a": ("manifest_owna.json", ("own_a_tight__avg_top2.npz", "own_a_tight__cmd_A.npz"))}
app = modal.App("probe-shift_aswin-surya_sub-supersedev2_kill-2026-11-14")
out_vol = modal.Volume.from_name(OUTVOL, version=2)
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy==2.3.3")


def try_n(folder: str) -> int:
    m = re.search(r"_v1(?:_try([0-9]+))?__", folder)
    return int(m.group(1) or 1)


def compare(old_dir: Path, new_dir: Path, sub: str, recovered: set, tol: float) -> dict:
    """By key, old vs new (v1's rule); tol > 0 only for the attack pairs' score fields. Keys, counts and magnitudes only."""
    import numpy as np

    mf, npzs = FILES[sub]
    mo, mn = json.loads((old_dir / mf).read_bytes()), json.loads((new_dir / mf).read_bytes())
    ko, kn = set(mo["folds"]), set(mn["folds"])
    added, removed = sorted(kn - ko), sorted(ko - kn)
    bad_keys = [k for k in added if k.split("|", 1)[1] not in recovered] + removed  # only recovered cells may be ADDED; nothing may be removed
    shared = ko & kn
    fold_diff = sorted(k for k in shared if mo["folds"][k] != mn["folds"][k])
    per_file = {}
    for fn in npzs:
        bo, bn = (old_dir / fn).read_bytes(), (new_dir / fn).read_bytes()
        if hashlib.sha256(bo).hexdigest() != (mo.get("files") or {}).get(fn) or hashlib.sha256(bn).hexdigest() != (mn.get("files") or {}).get(fn):
            per_file[fn] = {"file_not_its_manifests": True, "passes": False}
            continue
        zo, zn = np.load(io.BytesIO(bo)), np.load(io.BytesIO(bn))
        if sorted(zo.files) != sorted(zn.files) or any(len(z[f]) != len(z["key"]) for z in (zo, zn) for f in z.files):
            per_file[fn] = {"fields_differ_or_ragged": True, "passes": False}
            continue

        def index(z):
            ids = list(zip(z["key"].astype(str).tolist(), z["unit"].tolist(), z["call_index"].tolist()))
            return ids, {u: i for i, u in enumerate(ids)}
        io_, xo = index(zo)
        in_, xn = index(zn)
        dup = len(xo) != len(io_) or len(xn) != len(in_)
        so = sorted(u for u in xo if u[0] in shared)
        sn = {u for u in xn if u[0] in shared}
        unexpected = sorted({u[0] for u in xn if u[0] not in shared and u[0] not in added} | {u[0] for u in xo if u[0] not in shared and u[0] not in removed})
        diff, within = [], {}
        if not dup and set(so) == sn:
            ia, ib = np.array([xo[u] for u in so], int), np.array([xn[u] for u in so], int)
            for f in zo.files:
                a_, b_ = zo[f], zn[f]
                if (a_.dtype.kind in "USO") != (b_.dtype.kind in "USO") or (a_.dtype.kind not in "USO" and (a_.dtype != b_.dtype or a_.shape[1:] != b_.shape[1:])):
                    diff.append(f)
                    continue
                if a_.dtype.kind in "USO":
                    if any(str(x) != str(y) for x, y in zip(a_[ia], b_[ib])):
                        diff.append(f)
                    continue
                A, B = a_[ia], b_[ib]
                rowbytes = np.array([x.tobytes() != y.tobytes() for x, y in zip(A, B)], bool)
                if not rowbytes.any():
                    continue
                if tol > 0 and f in SCORE_FIELDS and A.dtype.kind == "f":  # Amendment 49a: attack score fields within tol where both finite
                    Af, Bf = A.astype(np.float64), B.astype(np.float64)
                    fin = np.isfinite(Af) & np.isfinite(Bf)
                    eb_ = (np.ascontiguousarray(A).view(np.uint8).reshape(A.shape + (-1,)) != np.ascontiguousarray(B).view(np.uint8).reshape(B.shape + (-1,))).any(-1)
                    nf_bad = bool((~fin & eb_).any())  # a non-finite value on either side must be byte-identical
                    gap = float(np.max(np.where(fin, np.abs(Af - Bf), 0.0)))
                    within[f] = {"rows_differing": int(rowbytes.sum()), "max_abs_diff": gap, "non_finite_differs": nf_bad}
                    if nf_bad or not gap <= tol:
                        diff.append(f)
                    continue
                diff.append(f)
        per_file[fn] = {"rows_old_shared": len(so), "rows_new_shared": len(sn), "duplicate_row_identity": dup, "row_identities_equal": set(so) == sn,
                        "fields_differing_on_shared_rows": sorted(diff), "unexpected_keys": unexpected,
                        "passes": not dup and set(so) == sn and not diff and not unexpected} | ({"score_fields_within_tolerance": within} if tol > 0 else {})
    return {"keys_old": len(ko), "keys_new": len(kn), "added": added, "removed": removed, "keys_not_allowed": bad_keys,
            "fold_differs_on_shared_keys": fold_diff, "files": per_file,
            "passes": not bad_keys and not fold_diff and all(v["passes"] for v in per_file.values())} | ({"tolerance_on_score_fields": tol} if tol > 0 else {})  # honest: v1's fields


@app.function(image=image, cpu=(2, 2), memory=(8192, 8192), timeout=900, retries=0, volumes={"/work": out_vol})
def mark(pairs: list, stamp: str, git: str, dry: bool) -> dict:
    """Two phases (as v1): every pair is checked first; if ANY fails, nothing is written; then every marker is created exclusively."""
    out_vol.reload()
    res, ok = [], True
    if len({(p["old"], p["sub"]) for p in pairs}) != len(pairs):
        return {"written": 0, "stop": "a destination twice in the batch", "results": []}
    for p in pairs:
        od, nd = Path("/work") / "scaleup" / p["old"] / p["sub"], Path("/work") / "scaleup" / p["new"] / p["sub"]
        mf = FILES[p["sub"]][0]
        r = {"old": p["old"], "new": p["new"], "sub": p["sub"], "kind": p["kind"]}
        if not (od / mf).is_file() or not (nd / mf).is_file() or (od / "SUPERSEDED.json").exists() \
                or hashlib.sha256((nd / mf).read_bytes()).hexdigest() != p["new_manifest_sha256"]:
            res.append(r | {"stop": "a manifest missing, the old folder already marked, or the replacement manifest not the committed record's"})
            ok = False
            continue
        cmp_ = compare(od, nd, p["sub"], set(p["recovered"]), TOL if p["kind"] == "attack" else 0.0)
        res.append(r | {"by_key": cmp_, **({} if cmp_["passes"] else {"stop": "the replacement differs from the superseded folder beyond Amendment 49 / 49a"})})
        ok = ok and cmp_["passes"]
    if not ok:
        return {"written": 0, "stop": "a pair failed: nothing written", "results": res}
    if dry:
        return {"written": 0, "dry": True, "results": res}
    made = []
    try:
        for p, r in zip(pairs, res):
            made.append(write_marker(p, r, stamp, git))
    except Exception as e:  # remove the markers this run created (each created exclusively, so ours), then report: no partial set stays
        for m_ in made:
            m_.unlink()
        for r in res:
            if r.pop("written", None):
                r["rolled_back"] = True
        out_vol.commit()
        return {"written": 0, "stop": f"a marker write failed ({type(e).__name__}: {e}); the {len(made)} markers this run created were removed", "results": res}
    out_vol.commit()
    return {"written": len(pairs), "results": res}


def write_marker(p: dict, r: dict, stamp: str, git: str) -> Path:
    """One SUPERSEDED.json, created exclusively; a write that fails after the file exists removes it before raising."""
    od = Path("/work") / "scaleup" / p["old"] / p["sub"]
    mf = FILES[p["sub"]][0]
    marker = {"superseded": True, "superseded_by": {"path": f"/scaleup/{p['new']}/{p['sub']}/", "manifest": mf, "manifest_sha256": p["new_manifest_sha256"],
                                                    "record": p["record"]},
              "this_manifest_sha256": hashlib.sha256((od / mf).read_bytes()).hexdigest(), "reason": "d9's re-stage at a new _try with the widened index-donor "
              "fallback (recovers t619 / t699 / t699_r2); refused by name in d9's names record", "recovered_cells": p["recovered"], "by_key_check": r["by_key"],
              "rule": "09 via 2e 10-16 01:1x: the replacement differs from this folder only in the recovered cells, by key (Amendment 49)", "stamp": stamp, "git": git}
    if p["kind"] == "attack":  # Amendment 49a: 09's text, the tolerance, and both diagnostics
        marker |= {"marker": ATTACK_MARKER, "diagnostics": DIAG,
                   "rule": ("09 via 2e 10-16 03:3x (Amendment 49a): the replacement differs from this folder only in the recovered cells, by key; on shared "
                            f"rows the score fields differ by at most {TOL:g} (scorer row grouping; activations byte-identical); everything else byte-identical")}
    mb = json.dumps(marker, indent=1).encode()
    dst = od / "SUPERSEDED.json"
    fh = dst.open("xb")  # exclusive: this run owns the file from here on
    try:
        fh.write(mb)
        fh.close()  # the flush on close is covered too (Codex sv2 r2)
    except BaseException:
        try:  # cleanup only: the original error is re-raised below
            fh.close()
        except Exception:  # noqa: S110
            pass
        dst.unlink()
        raise
    r["written"], r["marker_sha256"] = True, hashlib.sha256(mb).hexdigest()
    return dst


@app.local_entrypoint()
def main(dry: bool = False):
    git = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip()  # noqa: E731
    gitb = lambda *a: subprocess.run(["git", *a], capture_output=True, check=True, cwd=HERE).stdout  # noqa: E731
    if git("status", "--porcelain", "--untracked-files=no"):
        raise SystemExit("uncommitted tracked changes")
    head = git("rev-parse", "HEAD")
    if head != git("ls-remote", "origin", "aswin/supersede-diag-2e").split()[0]:
        raise SystemExit("HEAD is not pushed")
    for d in DIAG:  # the cited diagnostics, by sha256 at their commits
        if hashlib.sha256(gitb("show", f"{d['commit']}:{d['path']}")).hexdigest() != d["sha256"]:
            raise SystemExit(f"diagnostic {d['path']} @ {d['commit']}: sha256 differs")
    stopped = json.loads(gitb("show", f"{SRC_COMMIT}:{STOPPED}"))
    r4 = json.loads(gitb("show", f"{SRC_COMMIT}:{PAIRS_R4}"))
    ps = [dict(p) for p in stopped["pairs"]]
    sel_ = ("old", "new", "sub", "record", "recovered")
    if [{k: p[k] for k in sel_} for p in ps] != [{k: p[k] for k in sel_} for p in r4] or len(ps) != 14 or stopped.get("written") != 0:
        raise SystemExit("the stopped record's pairs differ from supersede_pairs_r4.json, are not 14, or the stopped run wrote a marker")
    for p in ps:  # as v1's main: the replacement's committed landing record (at 9b0a5ee8a) names its manifest sha256; the same set and world, a later _try
        rn = Path(p["record"]).name
        rec = json.loads(gitb("show", f"{SRC_COMMIT}:{Q35}/results/{rn}"))
        o = rec.get("outputs") or {}
        if p["sub"] == "sol":
            sha_ = o.get("manifest_sha256") if (rn.startswith(f"scaleup507_{p['new']}_") and rec.get("arm") == p["new"] and rec.get("stage") == "done"
                                                and o.get("published") is True and o.get("path") == f"{OUTVOL}:/scaleup/{p['new']}/sol/") else None
        else:
            hits = [a for a in rec.get("arms") or [] if a.get("arm") == p["new"]]
            sha_ = hits[0].get("manifest_sha256") if (rn.startswith("owna_sp493_") and rec.get("stage") == "done" and len(hits) == 1 and hits[0].get("published") is True
                                                       and rec.get("label_sealed_sp493")) else None
        strip = lambda n: re.sub(r"_v1(?:_try[0-9]+)?__", "_v1__", n)  # noqa: E731
        if not SP493.fullmatch(p["old"]) or not SP493.fullmatch(p["new"]) or strip(p["old"]) != strip(p["new"]) or try_n(p["new"]) <= try_n(p["old"]) \
                or p["sub"] not in FILES or not p["recovered"] or not set(p["recovered"]) <= RECOVERABLE or not sha_ or sha_ != p["new_manifest_sha256"]:
            raise SystemExit(f"pair {p['old']} -> {p['new']} ({p['sub']}): names, _try, recovered cells or the landing record's manifest sha256 do not hold")
        p["kind"] = "attack" if p["old"].endswith("__attack") else "honest"
    if sum(p["kind"] == "attack" for p in ps) != 6:
        raise SystemExit("not 6 attack pairs and 8 honest pairs")
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    res = mark.remote(ps, stamp, head, dry)
    out = {"what": ("S_paper493 supersession markers v2 (Amendment 49a): every pair checked by key first, then SUPERSEDED.json (all or nothing); "
                    "attack pairs: score fields within the tolerance, 09's marker text; honest pairs: Amendment 49's byte rule and marker"),
           "git": head, "stamp": stamp, "dry": dry, "source": {"commit": SRC_COMMIT, "stopped_record": STOPPED, "pairs": PAIRS_R4}, "diagnostics": DIAG,
           "attack_marker": ATTACK_MARKER, "tolerance": TOL, "pairs": ps, "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()} | res | {"of": len(ps)}
    dest = HERE / "results" / f"superseded_sp493_v2{'_DRY' if dry else ''}_{stamp}.json"
    dest.open("x").write(json.dumps(out, indent=1) + "\n")
    print(json.dumps({"record": dest.name, "sha256": hashlib.sha256(dest.read_bytes()).hexdigest(), "written": out["written"], "of": out["of"],
                      "stop": out.get("stop"), "stops": [r.get("stop") for r in out.get("results") or [] if r.get("stop")]}))
