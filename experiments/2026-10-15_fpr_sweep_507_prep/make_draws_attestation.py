"""Draws attestation for FL-RENDER's 32k-vs-full-length table: the 32k final and the full-length final use the same draw streams (seeds,
universe, strata, order), compared BY CONTENT. No score value is read or printed: only meta fields, row keys and hashes.

The 32k final was written by code without per-row draws hashes, so its draws are REPLAYED: the same job (same inputs, cohort, variants)
re-run with --draws-only on a commit that is the 32k final's commit plus the draws-hash patch only; each row then returns only
draws_sha256 = sha256(column map + the whole per-draw multiplicity matrix W). The full-length final's rows carry the same field.
Pass needs all of:
  (1) the same base inputs sha256 in all three records;  (2) the same cohort-list ref;
  (3) the same seed fields (key paths and values) in the 32k and full-length metas;
  (4) the replay's commit differs from the 32k final's only by the REVIEWED draws patch: git diff (run_peraction311.py only) has exactly the
      +/- lines of the pinned patch file, in order (the patch is Codex-reviewed and committed before the replay);
  (5) every non-withheld row of both records carries draws_sha256; each full-length row equals the replay row of its registered label and
      origin (retrains via full_variant_of; '<family>|uncut' -> '<family>').

  python make_draws_attestation.py --rec32 <c>:<path>@<sha> --replay32 <c>:<path>@<sha> --recfull <c>:<path>@<sha> --patch <c>:<path>@<sha>
  -> results/draws_attestation_<stamp>.json (exclusive create); prints its path, sha256 and pass; exit 1 unless pass.
"""

import argparse
import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent
RUNNER = "experiments/2026-10-15_fpr_sweep_507_prep/run_peraction311.py"


def git(*a):
    return subprocess.run(["git", *a], capture_output=True, check=True, cwd=HERE).stdout


def load(ref):
    cp, h = ref.rsplit("@", 1)
    c, p = cp.split(":", 1)
    b = git("show", f"{c}:{p}")
    if hashlib.sha256(b).hexdigest() != h:
        raise SystemExit(f"{p} @ {c}: sha256 differs from the pin")
    return json.loads(b), {"commit": c, "path": p, "sha256": h}


def seeds(x, pre=""):
    out = {}
    if isinstance(x, dict):
        for k, v in x.items():
            kp = f"{pre}.{k}" if pre else k
            if "seed" in k.lower() and not isinstance(v, dict):
                out[kp] = v
            else:
                out |= seeds(v, kp)
    return out


def reg_label(lb, r):
    """A full-length row's registered (32k) job label: a retrain's full_variant_of, else '<family>|uncut <ro>' -> '<family> <ro>'."""
    if r.get("full_variant_of"):
        return r["full_variant_of"]
    fam, ro = lb.rsplit(" ", 1)
    return f"{fam.removesuffix('|uncut')} {ro}"


def hashes(rec, full):
    """{(registered label, origin): hash}, plus the non-withheld rows without a hash (a partial record fails). W may differ by probe within an
    origin, so the comparison is per (label, origin): every full-length row against the replay row of its registered label."""
    out, missing = {}, []
    for lb, by_o in (rec.get("rows") or {}).items():
        for o, r in by_o.items():
            if not isinstance(r, dict):  # Codex r10: a malformed row never passes silently
                raise SystemExit(f"{lb} | {o}: not a row")
            if "withheld" in r or "unavailable_rule" in r:
                continue
            if not r.get("draws_sha256"):
                missing.append(f"{lb} | {o}")
                continue
            out.setdefault((reg_label(lb, r) if full else lb, o), set()).add(r["draws_sha256"])
    return out, missing


def main():
    ap = argparse.ArgumentParser()
    for k in ("--rec32", "--replay32", "--recfull", "--patch"):
        ap.add_argument(k, required=True)
    a = ap.parse_args()
    (r32, p32), (rrp, prp), (rfl, pfl) = load(a.rec32), load(a.replay32), load(a.recfull)
    m32, mrp, mfl = r32["meta"], rrp["meta"], rfl["meta"]
    if m32.get("full_length") or mrp.get("full_length") or not mfl.get("full_length"):
        raise SystemExit("--rec32 / --replay32 must be 32k records and --recfull a full-length one")
    ck = {}
    ins = [(m.get("inputs_final") or {}).get("sha256") for m in (m32, mrp, mfl)]
    ck["1_same_base_inputs"] = {"rec32": ins[0], "replay32": ins[1], "recfull": ins[2], "pass": bool(ins[0]) and len(set(ins)) == 1}
    coh = [(m.get("cohort_list") or {}).get("ref") for m in (m32, mrp, mfl)]
    ck["2_same_cohort_list"] = {"rec32": coh[0], "replay32": coh[1], "recfull": coh[2], "pass": bool(coh[0]) and len(set(coh)) == 1}
    s32, sfl = seeds(m32), seeds(mfl)
    ck["3_seeds"] = {"keys_32k": sorted(s32), "only_32k": sorted(set(s32) - set(sfl)), "only_full": sorted(set(sfl) - set(s32)),
                     "differ": sorted(k for k in set(s32) & set(sfl) if s32[k] != sfl[k])}
    ck["3_seeds"]["pass"] = bool(s32) and set(s32) == set(sfl) and not ck["3_seeds"]["differ"]
    c32, crp = m32["git"], mrp["git"]
    pc, pp = a.patch.rsplit("@", 1)[0].split(":", 1)
    pb = git("show", f"{pc}:{pp}")
    if hashlib.sha256(pb).hexdigest() != a.patch.rsplit("@", 1)[1]:
        raise SystemExit("--patch: sha256 differs from the pin")
    pm = lambda t: [ln for ln in t.splitlines() if ln[:1] in "+-" and not ln.startswith(("+++", "---"))]  # noqa: E731
    files = git("diff", "--name-only", c32, crp).decode().split()
    lines, want = pm(git("diff", "-U0", c32, crp, "--", RUNNER).decode()), pm(pb.decode())
    ck["4_replay_is_32k_plus_reviewed_patch"] = {"rec32_commit": c32, "replay_commit": crp, "patch": {"commit": pc, "path": pp}, "files_changed": files,
                                                 "changed_lines": len(lines), "patch_lines": len(want),
                                                 "pass": files == [RUNNER] and bool(want) and lines == want}
    hrp, mis_rp = hashes(rrp, False)
    hfl, mis_fl = hashes(rfl, True)
    multi = {f"{w}:{k[0]}|{k[1]}": len(v) for w, d in (("replay", hrp), ("full", hfl)) for k, v in d.items() if len(v) != 1}
    nopair = sorted(f"{k[0]} | {k[1]}" for k in set(hfl) - set(hrp))  # every full-length row needs its registered label's replay row
    differ = sorted(f"{k[0]} | {k[1]}" for k in set(hfl) & set(hrp) if hrp[k] != hfl[k])
    ck["5_row_draw_hashes"] = {"full_rows": len(hfl), "replay_rows": len(hrp), "rows_without_hash": {"replay": mis_rp, "full": mis_fl},
                               "more_than_one_hash": multi, "full_rows_without_a_replay_row": nopair[:20], "differ": differ[:20],
                               "origins_full": sorted({k[1] for k in hfl})}
    ck["5_row_draw_hashes"]["pass"] = bool(hfl) and not mis_rp and not mis_fl and not multi and not nopair and not differ
    ok = all(v["pass"] for v in ck.values())
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    out = {"what": "draws attestation: the 32k final and the full-length final use the same draw streams (by content; no score value read)",
           "rec32": p32, "replay32": prp, "recfull": pfl, "checks": ck, "pass": ok, "made_pt": stamp,
           "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    dest = HERE / "results" / f"draws_attestation_{stamp}.json"
    b = json.dumps(out, indent=1).encode()
    dest.open("xb").write(b)
    print(json.dumps({"file": str(dest.relative_to(HERE)), "sha256": hashlib.sha256(b).hexdigest(), "pass": ok,
                      "failing": [k for k, v in ck.items() if not v["pass"]]}, indent=1))
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
