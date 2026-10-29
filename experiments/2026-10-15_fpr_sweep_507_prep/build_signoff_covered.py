"""09's S1 / S2 covered list (route C skeleton steps 5a / 8a; keys only, local, git reads; no Modal, no record value). One "path@sha" per
covered item, sorted, the path repo-root-relative and the sha the sha256 of the committed bytes; written exclusive-create.

Inputs (a JSON file, committed): {"records": [repo-root paths committed at HEAD],
  "families": {<family>: {"harness": [{"commit", "path", "sha256"}, ...], "regression": {"commit", "path", "sha256"}}}}
S1 REQUIRES every scorer family entry in S1_FAMILIES (09, 10-18: Sol, p2, famexp, and own_all_loo as two entries, own_all_rs and
own_all_rs_uncut) with its FINAL regression-harness file(s) and its regression result record, each cited by
commit + path + sha256 that must re-hash at that commit (2e / 09, 10-17: no RULE.md amendment for the harness fixes provided S1 cites both).
STOPs (exit 1, nothing written): a family missing either item, a cited sha not the blob's at its commit, a record not tracked at HEAD, a
regression record not at HEAD with its cited bytes, a path that is not canonical (empty, leading "/", "//", "." or ".." segments, trailing
"/"), a path that is not a blob (a tree STOPs), an empty-input sha256 anywhere, or an existing output.

  uv run --no-project --python 3.12 python -I experiments/2026-10-15_fpr_sweep_507_prep/build_signoff_covered.py --kind S1 --inputs <inputs.json> \
      --out experiments/2026-10-15_fpr_sweep_507_prep/results/routec_signoff_inputs/S1_covered_<stamp>.json
"""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from urllib.parse import unquote

S1_FAMILIES = ("Sol", "p2", "famexp", "own_all_loo:own_all_rs", "own_all_loo:own_all_rs_uncut")  # 09 via 2e, 10-18 (was: Sol, p2, own_all loo / simt / loo3, famexp)
# own_all_loo has two regression records, so it is two family ENTRIES, one record each (09's preferred form). own_all_simt and own_all_loo3
# are dropped: not in route C's S1 scope, in neither the CAT spec's 24 pilot-tagged labels (FACC clause 2 @ 366e0602e) nor the catalogue
# manifest's 18 labels (16682ef28); 09 corrected the wording 10-18. c_asis needs no entry: Sol's record covers the c_asis files.
# 09 via 2e, 10-18 (replaces "extra family ignored"): for S1 every input family is one of the five REQUIRED entries (filled: harness and
# regression, never null, never flagged), or carries "dropped": DROP_NOTE (byte-exact) with every other key null (CUT's v2 form: harness and
# regression null). own_all_simt and own_all_loo3 MUST appear that way,
# so INP records the drop. A bare null, an absent required entry, any other unflagged family or a wrong note STOPs.
DROP_NOTE = "not in route C S1 scope: in neither the CAT spec's 24 pilot-tagged labels (FACC clause 2 @ 366e0602e) nor the catalogue manifest's 18 labels (16682ef28); 09 ruling 10-18"
S1_DROPPED = ("own_all_simt", "own_all_loo3")
EMPTY = hashlib.sha256(b"").hexdigest()


def canonical(p):
    """Astra signoff1 R3: one spelling per file, so the duplicate checks see aliases (no "./x", "x//y", "a/../x", "/x", "x/", "")."""
    return isinstance(p, str) and p != "" and not p.startswith("/") and all(s not in ("", ".", "..") for s in p.split("/"))


def gz(p):
    """2e, 10-18: 09's Amendment 15 hashes a .gz entry DECOMPRESSED; this builder hashes the committed bytes. Until a reviewed change adopts
    that convention, any gzip-named path STOPs (fail closed) instead of producing a sha 09 would not match. a63478 REQUIRED-1: the name test
    is case-insensitive and sees through %-escapes and trailing whitespace, dots and backslashes; .tgz and .gzip count too. The content
    test (gzip magic bytes, covered()'s is_gzip) catches every other spelling."""
    if not isinstance(p, str):
        return False
    q = unquote(p).strip().rstrip(". \t\\").lower()
    return q.endswith((".gz", ".tgz", ".gzip"))


def covered(kind, inputs, blob_sha, is_gzip=None):
    """Pure: blob_sha(commit, path) -> sha256 of that blob, or None if absent or not a blob; is_gzip(commit, path) -> True when that blob
    starts with the gzip magic bytes 1f 8b, False when not, None when it cannot be read (a63478: None STOPs, fail closed; main always
    passes it). -> (sorted "path@sha" list, problems)."""
    bad, out = [], []
    recs = list(inputs.get("records") or [])
    if len(set(recs)) != len(recs):
        bad.append("a record listed twice")
    for p in recs:
        if isinstance(p, str) and "\x00" in p:  # a63478 optional: a named STOP for a NUL byte in a path
            bad.append(f"record path contains a NUL byte: {p!r}")
            continue
        if gz(p):
            bad.append(f"record is .gz (decompressed-sha convention not supported; STOP): {p}")
            continue
        if not canonical(p):
            bad.append(f"record path not canonical: {p!r}")
            continue
        h = blob_sha("HEAD", p)
        g = is_gzip("HEAD", p) if (is_gzip and h is not None) else False
        if h is None:
            bad.append(f"record not tracked at HEAD: {p}")
        elif g is None:
            bad.append(f"record blob could not be read for the gzip check (fail closed; STOP): {p}")
        elif g:
            bad.append(f"record blob is gzip (magic 1f 8b; decompressed-sha convention not supported; STOP): {p}")
        else:
            out.append(f"{p}@{h}")
    fams = inputs.get("families") or {}
    if kind == "S1":  # 09 (10-18): skip a family ONLY when it carries "dropped": DROP_NOTE and nothing non-null
        if not isinstance(inputs.get("families"), dict):
            bad.append("families is not an object")
        for fam, f in (fams.items() if isinstance(fams, dict) else ()):
            flagged = isinstance(f, dict) and f.get("dropped") == DROP_NOTE and all(v is None for k, v in f.items() if k != "dropped")
            if fam in S1_FAMILIES:
                if not isinstance(f, dict) or "dropped" in f:
                    bad.append(f"family {fam}: a required entry must be filled (not null, not dropped)")
            elif not flagged:
                bad.append(f"family {fam}: not a required entry and not exactly {{\"dropped\": 09's note}} (unknown, null or a wrong note)")
        bad += [f"family {fam}: must appear as {{\"dropped\": 09's note}}" for fam in S1_DROPPED if fam not in fams]
    reg_of = {}  # 507-prep (a0575c03a review): families may SHARE a harness file (the two own_all_loo entries may share one); each needs its own record
    for fam in (S1_FAMILIES if kind == "S1" else tuple(fams)):
        f = (fams.get(fam) if isinstance(fams, dict) else None) or {}
        items = list(f.get("harness") or []) + ([f["regression"]] if isinstance(f.get("regression"), dict) else [])
        if not f.get("harness") or not isinstance(f.get("regression"), dict):
            bad.append(f"family {fam}: missing its final harness file(s) or its regression record")
        mine = []
        for it in items:  # each family's slots are checked on their own, every one must re-hash
            c, p, s = it.get("commit"), it.get("path"), it.get("sha256")
            if isinstance(p, str) and "\x00" in p:
                bad.append(f"family {fam}: path contains a NUL byte: {p!r}")
                continue
            if gz(p):
                bad.append(f"family {fam}: item is .gz (decompressed-sha convention not supported; STOP): {p}")
                continue
            if not canonical(p):
                bad.append(f"family {fam}: path not canonical: {p!r}")
                continue
            h = blob_sha(c, p) if c else None
            g = is_gzip(c, p) if (is_gzip and h is not None and h == s) else False
            if h is None or h != s:
                bad.append(f"family {fam}: {p} @ {c} does not hash to the cited sha256")
            elif g is None:
                bad.append(f"family {fam}: item blob could not be read for the gzip check (fail closed; STOP): {p}")
            elif g:
                bad.append(f"family {fam}: item blob is gzip (magic 1f 8b; decompressed-sha convention not supported; STOP): {p}")
            else:
                mine.append(f"{p}@{h}")
        if len(set(mine)) != len(mine):
            bad.append(f"family {fam}: the same item cited twice")
        if isinstance(f.get("regression"), dict) and canonical(f["regression"].get("path")):
            rg = f["regression"]
            reg_of.setdefault(rg["path"], []).append(fam)
            if blob_sha("HEAD", rg["path"]) != rg.get("sha256"):  # Astra signoff1 R1: the record must still be at HEAD, with the cited bytes
                bad.append(f"family {fam}: regression record {rg['path']} is not at HEAD with its cited sha256")
        out += mine
    shared_ = {p: fs for p, fs in reg_of.items() if len(fs) > 1}
    if shared_:
        bad.append(f"two families cite the same regression record (each needs its own): {shared_}")
    if any(x.endswith("@" + EMPTY) for x in out):
        bad.append("an empty-input sha256 in the list")
    return sorted(set(out)), bad  # one entry per item: a harness shared across families is listed once


def git_blob_sha(root, commit, path):
    """sha256 of the BLOB at commit:path, else None (Astra signoff1 R2: `git cat-file blob` fails on a tree, so a directory never hashes)."""
    if not (commit and canonical(path)):
        return None
    r = subprocess.run(["git", "cat-file", "blob", f"{commit}:{path}"], capture_output=True, cwd=root)
    return hashlib.sha256(r.stdout).hexdigest() if r.returncode == 0 else None


def git_is_gzip(root, commit, path):
    """True when the BLOB at commit:path starts with the gzip magic bytes 1f 8b, False when it does not, None on any git error or a non-blob
    (a63478: the caller STOPs on None, so the check fails closed). Reads only the first 2 bytes of the blob (a63478 OPTIONAL-A)."""
    if not (commit and canonical(path)):
        return None
    t = subprocess.run(["git", "cat-file", "-t", f"{commit}:{path}"], capture_output=True, cwd=root)
    if t.returncode != 0 or t.stdout.strip() != b"blob":
        return None
    p = subprocess.Popen(["git", "cat-file", "blob", f"{commit}:{path}"], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, cwd=root)
    head = p.stdout.read(2)
    if len(head) < 2:  # a short blob ends the stream: its exit status tells a 0-1 byte blob from an error
        p.stdout.close()
        return False if p.wait() == 0 else None
    p.stdout.close()
    p.kill()
    p.wait()
    return head == b"\x1f\x8b"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=("S1", "S2"), required=True)
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    root = Path(subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output=True, check=True).stdout.decode().strip())

    def blob_sha(commit, path):
        return git_blob_sha(root, commit, path)

    out = root / a.out
    if out.exists():
        sys.exit("STOP: the output exists (exclusive create)")
    lst, bad = covered(a.kind, json.loads(Path(a.inputs).read_text()), blob_sha, lambda c, p: git_is_gzip(root, c, p))
    if bad:
        print(json.dumps({"status": "STOP", "problems": bad}, indent=1))
        sys.exit(1)
    out.parent.mkdir(parents=True, exist_ok=True)
    b = json.dumps(lst, indent=1).encode()
    with out.open("xb") as fh:
        fh.write(b)
    print(json.dumps({"status": f"{a.kind}_COVERED_OK", "out": a.out, "sha256": hashlib.sha256(b).hexdigest(), "n": len(lst)}))


if __name__ == "__main__":
    main()
