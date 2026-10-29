"""Per-set pool membership of the nm172 / nm305 runs (09's GO item 3, via 2e 17:5x; CUT glance fixes 1-2 taken 18:4x; NOT RUN until CUT's re-glance).
KEYS AND PATHS ONLY, on the laptop (CUT: keys-only reads are within 0d's laptop scope; if the permission classifier blocks a step, STOP and report).
For each per-set full-length record (A 95dc8602..., PLANT 57f46d77..., R37 bce504ee..., each sha256-checked after decompression), the WHOLE record is
walked (rows, meta, pooled, npz, coverage, every top-level section): every object key or string value that CONTAINS nm172_shared_dotfiles or
nm305_user_tracking (any form: <world>|<cell>, tagged _e2a / _e2b / _r<k> / _redraw<k>, runs/ prefixes, E12 forms, eval paths) is reported as its
distinct form with the paths where it appears. A path keeps its top-level section; under rows, <label> and <origin> are collapsed and counted; list
indices are []; a key equal to the form is {form}. A form with whitespace or over 300 characters is reported as its length and sha256, never its
text. No value of any other field is read out. Beside it, the exact membership of five run keys in rows[<label>][<origin>], counted over labels.
Usage: python check_perset_membership.py --out <new json> (reads the records from git; the commits must be in the local object store)"""
import argparse
import gzip
import hashlib
import json
import re
import subprocess
from collections import defaultdict

NEEDLES = ("nm172_shared_dotfiles", "nm305_user_tracking")
KEYS = ["simh_nm360|nm172_shared_dotfiles", "honest_a_e2|nm172_shared_dotfiles", "honest_a_e2|nm305_user_tracking", "honest_b_e2|nm305_user_tracking",
        "honest_b|nm305_user_tracking"]
P = "experiments/2026-10-15_fpr_sweep_507_prep/results/"
RECORDS = {  # name -> (commit, path, sha256 of the decompressed record)
    "A": ("a05ea8959", P + "alltasks_CRD_FULL_r2_allorigins_e12_2026-10-16_113538.json", "95dc8602e86e845ef60394c6e4d361cf306e6cea5b50ad9f2379e0011e8ac4e8"),
    "PLANT": ("caaa1f671", P + "alltasks_CRD_FULL_plant_r2_allorigins_e12_2026-10-16_113537.json",
              "57f46d776618e9f78d3aec1876890d82d8e1a88036b8a5df9d47243fd30d51b5"),
    "R37": ("caaa1f671", P + "alltasks_CRD_FULL_r2_allorigins_rule37_e12_2026-10-16_113538.json.gz",
            "bce504ee8ddf228d9da14649aec1b8ea252d6e2525db29892e51e86daf7752e4"),
}
IDV = re.compile(r"\S{1,300}")


def form_of(s):  # the reported form of a matching string: itself if identifier-like, else its length and sha256
    return s if IDV.fullmatch(s) else f"<text: {len(s)} chars, sha256 {hashlib.sha256(s.encode()).hexdigest()[:16]}>"


def walk(x, path, out):  # out: form -> path pattern -> set of (label, origin) or {None}; path is a list of segments
    def emit(s, segs):
        f_ = form_of(s)
        if segs[:1] == ["rows"] and len(segs) >= 3:
            out[f_][".".join(["rows", "{label}", "{origin}"] + ["{form}" if g == s else g for g in segs[3:]])].add((segs[1], segs[2]))
        else:
            out[f_][".".join("{form}" if g == s else g for g in segs)].add(None)

    if isinstance(x, dict):
        for k, v in x.items():
            k = str(k)
            if any(n in k for n in NEEDLES):
                emit(k, path + [k])
            walk(v, path + [k], out)
    elif isinstance(x, list):
        for v in x:
            walk(v, path[:-1] + [path[-1] + "[]"] if path else ["[]"], out)
    elif isinstance(x, str) and any(n in x for n in NEEDLES):
        emit(x, path)


def where(x, path, out):  # every path at which one of the five KEYS appears exactly (as a value or an object key); indices collapsed to []
    if isinstance(x, dict):
        for k, v in x.items():
            if k in KEYS:
                out[k].add(f"{path}.{{key}}" if path else "{key}")
            where(v, f"{path}.{k}" if path else k, out)
    elif isinstance(x, list):
        for v in x:
            where(v, f"{path}[]", out)
    elif isinstance(x, str) and x in KEYS:
        out[x].add(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    res = {"needles": NEEDLES, "keys": KEYS, "records": {}}
    for name, (commit, path, want) in RECORDS.items():
        b = subprocess.run(["git", "show", f"{commit}:{path}"], capture_output=True, check=True).stdout
        b = gzip.decompress(b) if path.endswith(".gz") else b
        got = hashlib.sha256(b).hexdigest()
        assert got == want, f"STOP: {name} record sha256 {got[:16]} != {want[:16]}"
        rec = json.loads(b)
        forms = defaultdict(lambda: defaultdict(set))
        walk(rec, [], forms)
        per = {k: defaultdict(lambda: defaultdict(list)) for k in KEYS}  # key -> origin -> path -> labels
        for lab, row in rec["rows"].items():
            for org, r in row.items():
                out = defaultdict(set)
                where(r, "", out)
                for k, ps in out.items():
                    for p_ in ps:
                        per[k][org][p_].append(lab)
        res["records"][name] = {
            "commit": commit, "path": path, "sha256": got, "labels": len(rec["rows"]), "sections": sorted(rec),
            "forms": {f_: {p_: ({"labels": len({x[0] for x in s}), "origins": sorted({x[1] for x in s})} if None not in s else {"n": 1})
                           for p_, s in sorted(d.items())} for f_, d in sorted(forms.items())},
            "sections_with_forms": {n: sorted({p_.split(".", 1)[0].removesuffix("[]") for f_, d in forms.items() if n in f_ or f_.startswith("<text")
                                                for p_ in d}) for n in NEEDLES},
            "membership_rows": {k: {o: {p_: {"labels": len(ls), "of": len(rec["rows"])} for p_, ls in sorted(d.items())} for o, d in sorted(v.items())}
                                for k, v in per.items()},
            "absent_in_rows": [k for k in KEYS if not per[k]]}
    with open(a.out, "x") as fh:
        json.dump(res, fh, indent=1)
    for name, r in res["records"].items():
        print(name, r["sha256"][:16], {n: sorted(f_ for f_ in r["forms"] if n in f_) for n in NEEDLES}, "sections:", r["sections_with_forms"],
              "absent_in_rows:", r["absent_in_rows"])


if __name__ == "__main__":
    main()
