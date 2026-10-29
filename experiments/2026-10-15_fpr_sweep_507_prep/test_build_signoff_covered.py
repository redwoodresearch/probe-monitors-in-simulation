"""Synthetic tests of build_signoff_covered.covered (no git, no record): S1 requires every scorer family entry's final harness file(s) and
its regression record; a missing family, a sha that does not re-hash, an untracked record, a regression record gone from HEAD, a
non-canonical path or an empty-input sha STOPs. The last three checks run git on this code file and a tree (no record).
- 09's families (via 2e, 10-18): Sol, p2, famexp and own_all_loo as two entries (own_all_rs, own_all_rs_uncut), all filled; own_all_simt
  and own_all_loo3 MUST appear with "dropped": 09's note (byte-exact; any other key null). A family is skipped only with that flag; a bare null, an absent
  entry, an unknown unflagged family or a wrong note STOPs (this replaces "extra family ignored").
- a03f5d93 O-1: the fake blob lookup resolves path aliases ('./x', 'x//y', 'a/./b', 'a/../x', '/x', 'x/') to the file git would read,
  so the R3 cases fail if the canonical-path check is removed (a mutant without it must die).

  uv run --no-project --python 3.12 python experiments/2026-10-15_fpr_sweep_507_prep/test_build_signoff_covered.py
"""

import copy
import gzip
import os
import subprocess
import tempfile
from pathlib import Path

import build_signoff_covered as B


GZ_TMP = os.environ.get("GZ_TEST_TMP") or None  # a63478 OPTIONAL-1: unset -> the real-git case is SKIPPED, never the system temp dir


def synthetic_gzip_repo():
    """2e (10-18): the real-git gzip case must not read a results blob. Build a throwaway repo under $GZ_TEST_TMP (required; the B fork runs
    it in its scratchpad) with plumbing only (hash-object -w, mktree, commit-tree --no-gpg-sign: no hook runs, no signing), holding one
    gzip blob z.json and one plain blob p.json. Nothing is deleted. -> (repo dir, commit)."""
    d = tempfile.mkdtemp(prefix="routec_gz_", dir=GZ_TMP)
    g = lambda *a, inp=None: subprocess.run(["git", "-C", d, *a], input=inp, capture_output=True, check=True).stdout  # noqa: E731
    g("init", "-q")
    oz = g("hash-object", "-w", "--stdin", inp=gzip.compress(b"{}\n")).decode().strip()
    op = g("hash-object", "-w", "--stdin", inp=b"{}\n").decode().strip()
    tree = g("mktree", inp=f"100644 blob {oz}\tz.json\n100644 blob {op}\tp.json\n".encode()).decode().strip()
    c = g("-c", "user.name=t", "-c", "user.email=t@t", "commit-tree", "--no-gpg-sign", tree, "-m", "synthetic gzip fixture").decode().strip()
    return d, c


def norm(p):
    """The file git resolves a tree path to: '.' and empty segments drop, '..' pops (an alias names the same blob)."""
    parts = []
    for s in p.split("/"):
        if s in ("", "."):
            continue
        if s == "..":
            if parts:
                parts.pop()
            continue
        parts.append(s)
    return "/".join(parts)


def fname(fam):
    return fam.replace(":", "_")


BLOBS = {("HEAD", "r/inputs_full.json"): "a" * 64, ("HEAD", "r/preflight_A.json"): "b" * 64}
FAM = {}
for i, fam in enumerate(B.S1_FAMILIES):
    BLOBS[(f"c{i}", f"h/{fname(fam)}_regress.py")] = f"{i}" * 64
    BLOBS[(f"c{i}", f"res/{fname(fam)}_regress.json")] = f"{i + 1}" * 64
    BLOBS[("HEAD", f"res/{fname(fam)}_regress.json")] = f"{i + 1}" * 64  # the regression record is still at HEAD with the same bytes
    FAM[fam] = {"harness": [{"commit": f"c{i}", "path": f"h/{fname(fam)}_regress.py", "sha256": f"{i}" * 64}],
                "regression": {"commit": f"c{i}", "path": f"res/{fname(fam)}_regress.json", "sha256": f"{i + 1}" * 64}}
DROPPED = {fam: {"dropped": B.DROP_NOTE} for fam in B.S1_DROPPED}  # 09 (10-18): the drop is recorded in the inputs, byte-exact
OK = {"records": ["r/inputs_full.json", "r/preflight_A.json"], "families": FAM | DROPPED}
LOO_RS, LOO_UN = "own_all_loo:own_all_rs", "own_all_loo:own_all_rs_uncut"


def blob_sha(c, p):
    return BLOBS.get((c, norm(p)))


def run(inp, kind="S1"):
    return B.covered(kind, inp, blob_sha)


def stops(inp, kind="S1"):
    return bool(run(inp, kind)[1])


def without(fam, part=None):
    x = copy.deepcopy(OK)
    if part:
        x["families"][fam].pop(part)
    else:
        x["families"].pop(fam)
    return x


BLOBS[("cs", "h/shared_regress.py")] = "9" * 64
BLOBS[("cx", "h/own_all_simt_regress.py")] = "7" * 64
BLOBS[("cx", "res/own_all_simt_regress.json")] = "8" * 64
BLOBS[("HEAD", "res/own_all_simt_regress.json")] = "8" * 64


def shared_harness():
    x = copy.deepcopy(OK)
    for fam in (LOO_RS, LOO_UN):
        x["families"][fam]["harness"] = [{"commit": "cs", "path": "h/shared_regress.py", "sha256": "9" * 64}]
    return x


def shared_record():
    x = copy.deepcopy(OK)
    x["families"][LOO_UN]["regression"] = dict(x["families"][LOO_RS]["regression"])
    return x


def alias_record():
    """own_all_rs_uncut cites own_all_rs's record through an alias path: without the canonical check it would pass the shared-record check."""
    x = copy.deepcopy(OK)
    r = x["families"][LOO_RS]["regression"]
    x["families"][LOO_UN]["regression"] = {"commit": r["commit"], "path": "res/../" + r["path"], "sha256": r["sha256"]}
    return stops(x)


def fams_with(**over):
    """OK with some family entries replaced (a value of ... removes the entry)."""
    x = copy.deepcopy(OK)
    for k, v in over.items():
        if v is ...:
            x["families"].pop(k)
        else:
            x["families"][k] = v
    return x


SIMT_DATA = {"harness": [{"commit": "cx", "path": "h/own_all_simt_regress.py", "sha256": "7" * 64}],
             "regression": {"commit": "cx", "path": "res/own_all_simt_regress.json", "sha256": "8" * 64}}


def with_reg(fam, drop_head=False, head_sha=None):
    x, b = copy.deepcopy(OK), dict(BLOBS)
    r = x["families"][fam]["regression"]
    if drop_head:
        b.pop(("HEAD", r["path"]))
    if head_sha:
        b[("HEAD", r["path"])] = head_sha
    return bool(B.covered("S1", x, lambda c, p: b.get((c, norm(p))))[1])


ROOT = Path(subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output=True, check=True, text=True).stdout.strip())
ME = "experiments/2026-10-15_fpr_sweep_507_prep/build_signoff_covered.py"
lst, bad = run(OK)
unk_lst, unk_bad = run(fams_with(own_all_frac={"dropped": B.DROP_NOTE}))
cases = {
    "the five required entries plus the two flagged drops: a sorted list of records + 2 items per entry, no problem": bad == []
    and lst == sorted(lst) and len(lst) == 2 + 2 * len(B.S1_FAMILIES) and not any("own_all_simt" in e or "own_all_loo3" in e for e in lst),
    "the entries required (09, 10-18): Sol, p2, famexp, own_all_loo:own_all_rs, own_all_loo:own_all_rs_uncut":
        B.S1_FAMILIES == ("Sol", "p2", "famexp", LOO_RS, LOO_UN),
    "the drops and 09's corrected note (10-18) are pinned, byte-exact": B.S1_DROPPED == ("own_all_simt", "own_all_loo3") and B.DROP_NOTE ==
        "not in route C S1 scope: in neither the CAT spec's 24 pilot-tagged labels (FACC clause 2 @ 366e0602e) nor the catalogue"
        " manifest's 18 labels (16682ef28); 09 ruling 10-18",
    "09: the superseded note (CUT v2's wording) now STOPs as a wrong note": stops(fams_with(own_all_simt={
        "dropped": "not in route C S1 scope: catalogue has exactly 24 labels (FACC clause 2 @ 366e0602e); 09 ruling 10-18"})),
    "09: a bare-null own_all_simt STOPs": stops(fams_with(own_all_simt=None)),
    "09: own_all_simt with data instead of the flag STOPs (no longer ignored)": stops(fams_with(own_all_simt=SIMT_DATA)),
    "09: own_all_loo3 absent (the drop not recorded) STOPs": stops(fams_with(own_all_loo3=...)),
    "09: a wrong note string (one character off) STOPs": stops(fams_with(own_all_simt={"dropped": B.DROP_NOTE + " "})),
    "09: the flag plus null harness and regression (CUT v2's form) passes": not stops(fams_with(own_all_loo3={"dropped": B.DROP_NOTE, "harness": None, "regression": None})),
    "09: the flag plus a NON-null key (data under a drop) STOPs": stops(fams_with(own_all_loo3={"dropped": B.DROP_NOTE} | SIMT_DATA)),
    "09: an unknown family without the flag STOPs": stops(fams_with(own_all_frac=SIMT_DATA)),
    "09: an unknown family WITH the exact flag is skipped (not listed, no STOP)": unk_bad == [] and unk_lst == lst,
    "09: a required entry marked dropped STOPs": stops(fams_with(famexp={"dropped": B.DROP_NOTE})),
    "09: a required entry as a bare null STOPs": stops(fams_with(p2=None)),
    "09: families not an object STOPs": stops(OK | {"families": [1, 2]}),
    "S1: one family missing (famexp) STOPs": stops(without("famexp")),
    "S1: the own_all_loo:own_all_rs entry missing STOPs": stops(without(LOO_RS)),
    "S1: a family's regression record missing STOPs": stops(without("p2", "regression")),
    "S1: a family's harness missing STOPs": stops(without("Sol", "harness")),
    "a cited harness sha that does not re-hash at its commit STOPs": stops(
        {"records": OK["records"], "families": FAM | {"Sol": FAM["Sol"] | {"harness": [{"commit": "c0", "path": "h/Sol_regress.py", "sha256": "f" * 64}]}}}),
    "a harness cited at a commit where it does not exist STOPs": stops(
        {"records": OK["records"], "families": FAM | {"Sol": FAM["Sol"] | {"harness": [{"commit": "zz", "path": "h/Sol_regress.py", "sha256": "0" * 64}]}}}),
    "a record not tracked at HEAD STOPs": stops(OK | {"records": OK["records"] + ["r/missing.json"]}),
    "an empty-input sha STOPs": (lambda: (BLOBS.__setitem__(("HEAD", "r/empty.json"), B.EMPTY), stops(OK | {"records": OK["records"] + ["r/empty.json"]}))[1])(),
    "2e (10-18): a .json.gz RECORD STOPs with the named message, though its blob is tracked at HEAD": (lambda: (
        BLOBS.__setitem__(("HEAD", "r/rec.json.gz"), "c" * 64),
        any("is .gz (decompressed-sha convention not supported" in b for b in run(OK | {"records": OK["records"] + ["r/rec.json.gz"]})[1]))[1])(),
    "2e (10-18): a .json.gz REGRESSION record STOPs with the named message, though it re-hashes at its commit and at HEAD": (lambda: (
        BLOBS.__setitem__(("c0", "res/Sol_regress.json.gz"), "d" * 64), BLOBS.__setitem__(("HEAD", "res/Sol_regress.json.gz"), "d" * 64),
        any("family Sol: item is .gz" in b for b in run({"records": OK["records"], "families": OK["families"] | {"Sol": FAM["Sol"] | {
            "regression": {"commit": "c0", "path": "res/Sol_regress.json.gz", "sha256": "d" * 64}}}})[1]))[2])(),
    "a63478 REQUIRED-1: every name variant is caught by the name test": all(B.gz(q) for q in (
        "r/x.json.GZ", "r/x.json.Gz", "r/x.json.gZ", "r/x.json.gz.", "r/x.json.gz ", "r/x.json%2Egz", "r/x.json%2egz", "r/x.json.gz\\",
        "r/x.tgz", "r/x.GZIP")) and not any(B.gz(q) for q in ("r/x.json", "r/x.gzip_notes.json", "r/gz/x.json", None)),
    "a63478 REQUIRED-1: a .GZ record (blob tracked at HEAD) STOPs end to end": (lambda: (
        BLOBS.__setitem__(("HEAD", "r/rec.json.GZ"), "c" * 64),
        any("is .gz (decompressed-sha convention" in b for b in run(OK | {"records": OK["records"] + ["r/rec.json.GZ"]})[1]))[1])(),
    "a63478 OPTIONAL-1: a record whose BLOB is gzip under a plain name STOPs (magic 1f 8b)": (lambda: (
        BLOBS.__setitem__(("HEAD", "r/plain.bin"), "c" * 64),
        any("record blob is gzip" in b for b in B.covered("S1", OK | {"records": OK["records"] + ["r/plain.bin"]}, blob_sha,
                                                         lambda c, p: norm(p) == "r/plain.bin")[1]))[1])(),
    "a63478 OPTIONAL-1: a family item whose blob is gzip STOPs (magic 1f 8b)": any(
        "family Sol: item blob is gzip" in b for b in B.covered("S1", OK, blob_sha, lambda c, p: norm(p) == "h/Sol_regress.py")[1]),
    "a63478 OPTIONAL-1: with no gzip blob, is_gzip changes nothing": B.covered("S1", OK, blob_sha, lambda c, p: False) == run(OK),
    "a63478 B: a tree is None (not a blob), and this builder's own blob is not gzip (False, not None)":
        B.git_is_gzip(ROOT, "HEAD", "experiments") is None and B.git_is_gzip(ROOT, "HEAD", ME) is False,
    "a63478 B: is_gzip returning None (git error) STOPs a record and a family item (fail closed)":
        any("record blob could not be read for the gzip check" in b for b in B.covered("S1", OK, blob_sha, lambda c, p: None)[1])
        and any("item blob could not be read for the gzip check" in b for b in B.covered("S1", OK, blob_sha, lambda c, p: None)[1]),
    "a63478 optional: a NUL byte in a record or family path STOPs, named": any("record path contains a NUL byte" in b for b in run(
        OK | {"records": OK["records"] + ["r/a\x00b.json"]})[1]) and any("path contains a NUL byte" in b for b in run(
        {"records": OK["records"], "families": OK["families"] | {"Sol": FAM["Sol"] | {"harness": [{"commit": "c0", "path": "h/S\x00.py",
                                                                                                  "sha256": "0" * 64}]}}})[1]),
    "S2 does not require the S1 families": not stops({"records": ["r/inputs_full.json"], "families": {}}, kind="S2"),
    "the two own_all_loo entries citing the SAME harness path@sha: OK, listed once": (lambda r: r[1] == [] and r[0].count(
        "h/shared_regress.py@" + "9" * 64) == 1)(run(shared_harness())),
    "two family entries citing the same regression RECORD: STOP": stops(shared_record()),
    "a family citing the same item twice: STOP": stops({"records": OK["records"], "families": FAM | {"Sol": FAM["Sol"] | {"harness": FAM["Sol"]["harness"] * 2}}}),
    "a record listed twice: STOP": stops(OK | {"records": OK["records"] + OK["records"][:1]}),
    "R1: a regression record deleted at HEAD: STOP": with_reg("p2", drop_head=True),
    "R1: a regression record changed at HEAD: STOP": with_reg("p2", head_sha="e" * 64),
    "O-1: the fake lookup resolves an alias to the same blob, as git does": blob_sha("HEAD", "./r//inputs_full.json") == "a" * 64,
    "R3: a record alias './r/...' of a listed record: STOP": stops(OK | {"records": OK["records"] + ["./r/inputs_full.json"]}),
    "R3: aliases of a tracked record ('/r/x', 'r//x', 'r/./x', 'r/../r/x', 'r/x/') and '' all STOP": all(
        stops(OK | {"records": [q]}) for q in ("", "/r/inputs_full.json", "r//inputs_full.json", "r/./inputs_full.json",
                                               "r/../r/inputs_full.json", "r/inputs_full.json/")),
    "R3: a family citing another family's regression record through an alias path STOPs": alias_record(),
    "R2: a tree (a directory path) never hashes": B.git_blob_sha(ROOT, "HEAD", "experiments") is None,
    "R2: an empty path (HEAD:) never hashes": B.git_blob_sha(ROOT, "HEAD", "") is None,
    "R2: a blob at HEAD hashes (this script)": len(B.git_blob_sha(ROOT, "HEAD", ME) or "") == 64,
}
if GZ_TMP:  # a63478 OPTIONAL-1: the real-git gzip case runs only when $GZ_TEST_TMP names a scratch dir
    cases["a63478 B: real git on a SYNTHETIC repo (no results blob read): gzip blob True, plain blob False, missing path None"] = (
        lambda d, c: B.git_is_gzip(d, c, "z.json") is True and B.git_is_gzip(d, c, "p.json") is False
        and B.git_is_gzip(d, c, "missing.json") is None)(*synthetic_gzip_repo())
else:
    print("SKIPPED: GZ_TEST_TMP unset (the real-git gzip case on a synthetic repo)")
for name, ok in cases.items():
    print(("PASS " if ok else "FAIL ") + name)
assert all(cases.values()), "covered-list builder tests failed"
print(f"{len(cases)} checks pass")
