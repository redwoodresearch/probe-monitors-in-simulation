"""Builds the --pw-authorized file (RULE item 46: authorized overlaps listed by key) from a pooled catalogue-stage manifest and its catalogue.
Keys and metadata only: it reads per_label_combo[*].cross_pool_overlaps.roles_by_key and the catalogue's world / sets / roles per run.
09 (via 507-prep, 10-16 ~18:3x): the ONLY authorized cross-pool overlap is S_old's honest_c, registered both in H_B (B ∪ C) and as S_old's H_A.
So in EVERY label x combo the overlap map must equal the catalogue's honest_c runs (192, sets [S_old] only, roles A and B), and every catalogued
world must be its key's world token, else it STOPs for 09; the runner's drive repeats this before any row (pw_auth_overlap_check). Every citation is verified here before writing: the commit resolves, the line (range) is
in the file (a range: both ends), and the first line holds the quoted words. The runner's matcher takes ONE reference per key, so (09, 10-16
via 507-prep) `rule` is READ_LOG @ 4ca9769a4:936, the one line stating both roles, and the other citations (with the 395-407 range) are in `note`.
usage: python3 make_authorized.py <local manifest copy> <local catalogue copy> <manifest path@sha256 on the volume> <catalogue path@sha256> <out>"""
import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

N, WORLD, SETS, ROLES = 192, "honest_c", ["S_old"], ("A", "B")
SPEC = "experiments/2026-10-14_priority1_spec_09/"
CITES = [  # (file, commit, first line, last line, words the first line must hold): 09's citations, 10-16 via 507-prep
    ("RULE.md", "efdfcc8b2", 395, 407, "## Append 6"),  # S_old's tau_real pool is honest B ∪ C, never honest A: honest_c is in H_B on S_old
    ("RULE.md", "24fd07866", 4163, 4163, "(H_B, or B ∪ C on S_old)"),
    ("RULE.md", "cd1afb33e", 3623, 3623, "pooled is S_new with H_A and S_old with honest C"),  # honest C plays S_old's H_A role
    ("READ_LOG.md", "4ca9769a4", 936, 936, "S_old pools are B u C; H_A = honest_c"),
    ("READ_LOG.md", "d6cec31b5", 1180, 1180, "H_A = honest_c of B u C, not cell-paired"),
    ("RULE.md", "ca621714d", 4890, 4890, "H_A is never mixed into H_B"),  # consistency: keeps honest A out of tau_real; on S_old H_A IS honest_c
]
RULE = "READ_LOG @ 4ca9769a4:936"  # 09's PRIMARY reference (both roles in one line); the runner's matcher re-checks its commit and line bound


def ref(c):
    f, sha, a, b, _ = c
    return f"{f.removesuffix('.md')} @ {sha}:{a}" + (f"-{b}" if b != a else "")


def verify(c):
    f, sha, a, b, words = c
    p = subprocess.run(["git", "show", f"{sha}:{SPEC}{f}"], capture_output=True, cwd=Path(__file__).parent)
    lines = p.stdout.decode().splitlines()
    if p.returncode != 0 or not 1 <= a <= b <= len(lines) or words not in lines[a - 1]:
        raise SystemExit(f"citation {ref(c)} does not verify (commit, line bounds or quoted words '{words}') (STOP)")


def main(man_local, cat_local, man_ref, cat_ref, out):
    if cat_ref.rsplit("@", 1)[0] != man_ref.rsplit("@", 1)[0].removesuffix("_manifest.json") + "_catalogue.json":
        raise SystemExit("the catalogue reference is not the catalogue beside the manifest (STOP)")
    mb, cb = Path(man_local).read_bytes(), Path(cat_local).read_bytes()
    if hashlib.sha256(mb).hexdigest() != man_ref.rsplit("@", 1)[1] or hashlib.sha256(cb).hexdigest() != cat_ref.rsplit("@", 1)[1]:
        raise SystemExit("a local copy's sha256 differs from its reference (STOP)")
    runs = json.loads(cb)["runs"]
    bw = sorted(k for k, e in runs.items() if e["world"] != k.split("|", 1)[0])  # Astra r9 #2: the catalogued world is the key's world token
    hc = {k for k, e in runs.items() if e["world"] == WORLD}
    want = {k: ROLES for k in hc}
    odd = sorted(k for k in hc if runs[k]["sets"] != SETS or tuple(sorted(runs[k]["roles"])) != ROLES)
    per = {lc: {k: tuple(sorted(rs)) for k, rs in v["cross_pool_overlaps"]["roles_by_key"].items()} for lc, v in json.loads(mb)["per_label_combo"].items()}
    off = sorted(lc for lc, o in per.items() if o != want)  # Astra r9 #1: EVERY label x combo on its own, never a union
    if bw or len(hc) != N or odd or off or not per:
        d = per[off[0]] if off else want
        raise SystemExit(f"STOP for 09: every label x combo's overlaps must be exactly the {N} S_old honest_c runs in H_A and H_B: {len(hc)} "
                         f"honest_c runs; world differs from the key {bw[:6]}; outside S_old or roles A, B {odd[:6]}; label x combos off ({len(off)}) "
                         f"{off[:4]}; in the first, other or differing {sorted(k for k in d if want.get(k) != d[k])[:6]}, missing {sorted(set(want) - set(d))[:6]}")
    for c in CITES:
        verify(c)
    if RULE not in map(ref, CITES):
        raise SystemExit("the primary rule is not one of the verified citations (STOP)")
    note = ("S_old's honest_c is registered in both roles, H_B (tau_real on B ∪ C) and S_old's H_A (honest C). Primary (`rule`): " + RULE + ". 09's "
            "other citations (paths in " + SPEC + "): " + "; ".join(f"{ref(c)} (\"{c[4]}\")" for c in CITES if ref(c) != RULE) + ". The last is "
            "consistency: it keeps honest A out of tau_real; on S_old the H_A role IS honest_c")
    ents = [{"key": k, "roles": list(ROLES), "rule": RULE, "note": note} for k in sorted(hc)]
    Path(out).write_text(json.dumps({"what": "authorized cross-pool overlaps for the pooled scored stage (RULE item 46), by key: S_old's honest_c "
                                             "in H_A and H_B (09's citations); built by tools_pooledworld/make_authorized.py",
                                     "made_pt": datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d %H:%M:%S %Z"),
                                     "manifest": man_ref, "catalogue": cat_ref, "authorized_overlaps": ents}, indent=1, ensure_ascii=False) + "\n")
    print(f"{len(ents)} authorized overlaps (S_old honest_c, roles A & B); {len(CITES)} citations verified; rule {RULE}")
    return ents


if __name__ == "__main__":
    main(*sys.argv[1:6])
