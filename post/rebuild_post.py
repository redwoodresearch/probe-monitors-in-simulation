"""Rebuild the post's figures and printed numbers from the committed records in this folder.

    uv run --project post post/rebuild_post.py        (from the repository root; or: cd post && uv run rebuild_post.py)

No GPU, no API keys, no Modal and no private package: Python 3.12 with numpy and matplotlib. Steps:
  (a) every input in MANIFEST.json (records, copied scripts, render logs, reference figures, and this folder's rules, expected inventories
      and configs) is checked against its sha256, and the expected inventory against the pre-stated token counts in MANIFEST.json;
  (b) Figures 1-7 are redrawn (post_figures.py) and checked against the original render logs, Figure 3 against its reference PNG;
  (c) the four slot readouts are rerun and must equal the committed slot files, then every printed number is rebuilt (post_numbers.py);
  (d) the rebuilt numbers are compared with expected/expected_post_numbers.json.
Any difference exits non-zero, and so does a partial run (--skip-figures). Outputs go to --out: figures/, rebuilt_post_numbers.json
({token id: value as printed}) and report.json. The final count is of individual differences (one per token, file or slot key).
"""

import argparse
import gzip
import hashlib
import io
import json
import sys
from collections import Counter
from pathlib import Path

import post_figures
import post_numbers

HERE = Path(__file__).resolve().parent
CLASSES = ("R", "Q", "C", "S", "X", "H", "N")  # rebuilt, recorded, constant, structural, cited, hypothetical, not rebuildable
COMPARED = ("R", "Q")


def sha256(b):
    return hashlib.sha256(b).hexdigest()


class Records:
    """The committed records in records/, by MANIFEST name. Every read checks the sha256 (and, for .gz, the decompressed sha256) first."""

    def __init__(self, manifest):
        self.by_name = {r["name"]: r for r in manifest["records"]}
        self.pending = {p["name"]: p for p in manifest.get("pending", [])}
        self._json, self._npz = {}, {}

    def raw(self, name):
        r = self.by_name[name]
        f = HERE / r["file"]
        if not f.exists():
            raise post_numbers.RuleError(f"{name} missing: {r['file']}")
        b = f.read_bytes()
        if sha256(b) != r["sha256"]:
            raise SystemExit(f"STOP: {r['file']} does not match its sha256 in MANIFEST.json")
        if r["file"].endswith(".gz"):
            b = gzip.decompress(b)
            if sha256(b) != r["decompressed_sha256"]:
                raise SystemExit(f"STOP: {r['file']} decompressed does not match its decompressed_sha256 in MANIFEST.json")
        return b

    def json(self, name):
        if name not in self._json:
            self._json[name] = json.loads(self.raw(name))
        return self._json[name]

    def npz(self, name):
        if name not in self._npz:
            import numpy as np

            self._npz[name] = np.load(io.BytesIO(self.raw(name)))  # allow_pickle stays False
        return self._npz[name]

    def sha(self, name):  # the sha the original runs pinned: of the decompressed bytes for a .gz record
        r = self.by_name[name]
        return r.get("decompressed_sha256", r["sha256"])

    def pending_file(self, name):
        """A file added after the first pinned run (MANIFEST "pending"): its path once present and pinned, else None."""
        p = self.pending[name]
        f = HERE / p["file"]
        return f if p.get("sha256") and f.exists() else None


def verify(manifest):
    """(a) Every record, copied file and pinned file against MANIFEST.json. -> (fatal problems, missing pending files)."""
    fatal, missing = [], []
    for r in manifest["records"]:
        f = HERE / r["file"]
        if not f.exists():
            fatal.append(f"{r['name']} missing: {r['file']}")
            continue
        b = f.read_bytes()
        if sha256(b) != r["sha256"]:
            fatal.append(f"{r['file']}: sha256 differs")
        elif r["file"].endswith(".gz") and sha256(gzip.decompress(b)) != r["decompressed_sha256"]:
            fatal.append(f"{r['file']}: decompressed sha256 differs")
    for c in manifest["copied_files"] + manifest["pinned_files"]:
        f = HERE / c["file"]
        if not f.exists():
            fatal.append(f"{c['file']} missing")
        elif sha256(f.read_bytes()) != c["sha256"]:
            fatal.append(f"{c['file']}: sha256 differs")
    for p in manifest.get("pending", []):
        f = HERE / p["file"]
        if not (p.get("sha256") and f.exists()):
            missing.append(p["missing_message"])
        elif sha256(f.read_bytes()) != p["sha256"]:
            fatal.append(f"{p['file']}: sha256 differs")
    return fatal, missing


def check_inventory(expected, rules, manifest):
    """The expected inventory may not shrink or change silently: unique ids, known classes, non-empty printed strings, the exact class
    counts pre-stated in MANIFEST.json, and exactly one rule per R/Q token. -> problems."""
    toks = expected.get("tokens") or []
    ids = Counter(t.get("id") for t in toks)
    probs = [f"inventory: id {i!r} appears {n} times" for i, n in sorted(ids.items(), key=str) if n != 1]
    probs += [f"inventory: {t.get('id')!r} has unknown class {t.get('class')!r}" for t in toks if t.get("class") not in CLASSES]
    probs += [f"inventory: {t.get('id')!r} has no printed string" for t in toks if not isinstance(t.get("printed"), str) or not t["printed"]]
    want, got = manifest["token_counts"], Counter(t.get("class") for t in toks)
    if len(toks) != want["total"]:
        probs.append(f"inventory: {len(toks)} tokens, MANIFEST.json pre-states {want['total']}")
    probs += [f"inventory: {got.get(c, 0)} tokens of class {c}, MANIFEST.json pre-states {want[c]}" for c in CLASSES if got.get(c, 0) != want[c]]
    rq = {t.get("id") for t in toks if t.get("class") in COMPARED}
    probs += [f"inventory: {i}: an R/Q token without a rule" for i in sorted(rq - set(rules))]
    probs += [f"inventory: {i}: a rule without an R/Q token" for i in sorted(set(rules) - rq)]
    return probs


def compare(expected, rebuilt, errors):
    """(d) Every R or Q token must have a rebuilt value, no error, and equal the printed string exactly. -> (rows, problems)."""
    rows, probs = [], []
    rq = [t for t in expected["tokens"] if t["class"] in COMPARED]
    for t in rq:
        got, err = rebuilt.get(t["id"]), errors.get(t["id"])
        ok = t["id"] in rebuilt and err is None and got == t["printed"]
        rows.append({"id": t["id"], "line": t["line"], "class": t["class"], "printed": t["printed"], "rebuilt": got, "match": ok}
                    | ({"error": err} if err else {}))
        if not ok:
            probs.append(f"{t['id']} (line {t['line']}): printed {t['printed']!r}, rebuilt {got!r}" + (f" [{err}]" if err else ""))
    probs += [f"{i}: rebuilt but not an R/Q token" for i in sorted(set(rebuilt) - {t["id"] for t in rq})]
    return rows, probs


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=str(HERE / "rebuild_out"), help="output folder (created; files in it are overwritten)")
    ap.add_argument("--skip-figures", action="store_true", help="numbers only: a partial run, which always exits non-zero")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((HERE / "MANIFEST.json").read_text())
    report = {"problems": {}}

    print("(a) verifying inputs")
    fatal, report["problems"]["pending"] = verify(manifest)
    if fatal:  # nothing below is meaningful on unverified inputs
        print("\n".join(fatal))
        sys.exit(f"FAIL: {len(fatal)} input(s) missing or differing from MANIFEST.json")
    for m in report["problems"]["pending"]:
        print(f"  {m}")
    expected = json.loads((HERE / "expected" / "expected_post_numbers.json").read_text())
    rules = json.loads((HERE / "number_rules.json").read_text())
    rules.pop("_about", None)
    report["draft"] = expected.get("draft")
    report["problems"]["inventory"] = check_inventory(expected, rules, manifest)
    records = Records(manifest)

    if a.skip_figures:
        print("(b) figures SKIPPED (--skip-figures): this run cannot pass")
        report["figures_skipped"] = True
    else:
        print("(b) redrawing the figures")
        report["problems"]["figures"] = post_figures.redraw(records, out)

    print("(c) rerunning the slot readouts and rebuilding the numbers")
    slots, report["problems"]["slots"] = post_numbers.slot_readouts(records)
    rebuilt, errors = post_numbers.rebuild(records, slots, rules)
    (out / "rebuilt_post_numbers.json").write_text(json.dumps(rebuilt, indent=1, ensure_ascii=False))

    print("(d) comparing with expected/expected_post_numbers.json")
    rows, report["problems"]["numbers"] = compare(expected, rebuilt, errors)
    classes = Counter(t["class"] for t in expected["tokens"])
    report["numbers"] = {"tokens": len(expected["tokens"]), "by_class": dict(sorted(classes.items())),
                         "compared": len(rows), "matched": sum(r["match"] for r in rows), "rows": rows}
    bad = [p for ps in report["problems"].values() for p in ps]
    report["differences"] = len(bad)
    (out / "report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))

    n = report["numbers"]
    print(f"  numbers: {n['matched']} of {n['compared']} rebuilt tokens match the draft; not compared, by class: "
          + ", ".join(f"{c} {k}" for c, k in n["by_class"].items() if c not in COMPARED) + " (see expected_post_numbers.json)")
    for p in bad[:60]:
        print("  DIFF", p)
    if len(bad) > 60:
        print(f"  ... and {len(bad) - 60} more (report.json lists every one)")
    print(f"report: {out / 'report.json'}")
    if bad:
        sys.exit(f"FAIL: {len(bad)} individual difference(s)")
    if a.skip_figures:
        sys.exit("PARTIAL: every rebuilt number matches, but the figures were skipped; run without --skip-figures for a full check")
    print("PASS: every figure and every rebuilt number matches")


if __name__ == "__main__":
    main()
