"""Check every number in the write-up outline against its result file, by key.

Reads specs (one JSON object per number: outline line, quoted number, result file + commit or S3 url + sha256, JSON key
path, scale), loads each result with json.load, reads the key, and compares it with the stated number at the stated
rounding. Writes NUMBERS.md (mismatches first) and numbers.csv. Read-only on the repo; S3 reads use the laptop profile
and are refused for sealed paths.

  uv run --no-project python analyze_outline_numbers.py --specs specs_all.jsonl --repo <repo> --out <dir> [--s3-cache <dir>]
"""

import argparse
import csv
import hashlib
import json
import re
import subprocess
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

SEALED = ("oct11c", "lean230", "zooc_stage", "zooc-stage", "zoo_c_stage", "zoo-c-stage", "stage_records", "stage-records")  # sealed runs only; gate and sizing results are allowed
NUM = re.compile(r"[−+-]?\$?\d[\d,]*(?:\.\d+)?")
RATIO = re.compile(r"(\d+)\s+in\s+(\d+)")
BOUNDS = [  # first match wins: inclusive and negated forms before the strict ones they contain
    (re.compile(r"<=|≤|\b(at most|no more than|not more than|no greater than)\b"), lambda v, s: v <= s),
    (re.compile(r">=|≥|\b(at least|no less than|not less than|no fewer than)\b"), lambda v, s: v >= s),
    (re.compile(r"<(?!=)|\b(under|below|less than|fewer than)\b"), lambda v, s: v < s),
    (re.compile(r">(?!=)|\b(over|above|more than)\b"), lambda v, s: v > s)]


def git(repo, *args):
    r = subprocess.run(["git", "-C", repo, *args], capture_output=True, text=True)
    return r.returncode, r.stdout


def parse_number(s):
    m = NUM.search(s.replace("−", "-"))
    if not m:
        return None, None, False
    t = m.group(0).replace("$", "").replace(",", "")
    dec = len(t.split(".")[1]) if "." in t else 0
    return float(t), dec, t[:1] in "+-"


def half_up(x, dec):
    """x rounded half-up to dec places (the outline's convention), as a float."""
    return float(Decimal(str(round(x, 9))).quantize(Decimal(1).scaleb(-dec), rounding=ROUND_HALF_UP))


def load_json(spec, repo, cache):
    if spec.get("s3"):
        url = spec["s3"]["url"]
        if any(w in url.lower() for w in SEALED):
            raise ValueError("sealed path refused")
        dst = Path(cache) / hashlib.sha256(url.encode()).hexdigest()[:16]
        if not dst.exists():
            r = subprocess.run(["aws", "s3", "cp", "--profile", "redwood-sandbox", "--only-show-errors", url, str(dst)])
            if r.returncode:
                raise ValueError("s3 download failed")
        data = dst.read_bytes()
        want = spec["s3"].get("sha256", "")
        if not re.fullmatch(r"[0-9a-f]{64}", want):  # a full digest only: a prefix or an elision would verify nothing
            raise ValueError("s3 spec lacks a full sha256")
        if hashlib.sha256(data).hexdigest() != want:  # a stale cache copy: fetch once more, then fail loudly
            r = subprocess.run(["aws", "s3", "cp", "--profile", "redwood-sandbox", "--only-show-errors", url, str(dst)])
            data = dst.read_bytes() if r.returncode == 0 else b""
            if hashlib.sha256(data).hexdigest() != want:
                raise ValueError("s3 sha256 differs from manifest")
        return json.loads(data), url
    path, ref = spec["file"], spec["ref"]
    if any(w in path.lower() for w in SEALED):
        raise ValueError("sealed path refused")
    rc, out = git(repo, "show", f"{ref}:{path}")
    if rc:
        raise ValueError(f"git show failed for {ref}:{path}")
    return json.loads(out), f"{path}@{ref}"


def walk(obj, key):
    for k in key:
        obj = obj[k] if isinstance(obj, dict) else obj[int(k)]
    return obj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--specs", required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--s3-cache", default="/tmp/numbers_s3_cache")
    a = ap.parse_args()
    Path(a.s3_cache).mkdir(parents=True, exist_ok=True)
    rows = []
    seen = set()
    with open(a.specs) as fh:
        specs = fh.read().splitlines()
    for line in specs:
        if not line.strip():
            continue
        s = json.loads(line)
        sig = json.dumps([s.get("line"), s["number"], s.get("key"), s.get("md"), s.get("file"), s.get("ref"),
                          (s.get("s3") or {}).get("url"), s.get("scale"), s.get("denom_key")])
        if sig in seen:
            continue  # two locators covered the same line
        seen.add(sig)
        stated, dec, signed = parse_number(s["number"])
        row = {"line": s.get("line"), "block": s.get("block"), "number": s["number"], "quote": s.get("quote", "")[:90],
               "source": "", "key": "/".join(map(str, s.get("key") or [])), "file value": "", "verdict": "", "on main": "",
               "note": s.get("note", "")}
        if s.get("ref"):
            rc, _ = git(a.repo, "merge-base", "--is-ancestor", s["ref"], "origin/main")
            if rc not in (0, 1):  # 1 = not an ancestor; anything else is a git error, not "branch only"
                raise RuntimeError(f"git merge-base failed ({rc}) for {s['ref']}")
            row["on main"] = "yes" if rc == 0 else "NO"
        if not s.get("key") or (not s.get("file") and not s.get("s3")):
            row["verdict"] = "README ONLY" if s.get("md") else "NO SOURCE"
            row["source"] = s.get("md") or ""
            rows.append(row)
            continue
        try:
            data, src = load_json(s, a.repo, a.s3_cache)
            v = walk(data, s["key"])
            if s.get("denom_key"):  # a derived share: key / denom_key
                v = v / walk(data, s["denom_key"])
                row["key"] += " ÷ " + "/".join(map(str, s["denom_key"]))
            row["source"] = src
            ratio = RATIO.search(s["number"])
            bound = next((f for r, f in BOUNDS if r.search(s["number"].lower())), None)
            if ratio and isinstance(v, (int, float)) and v:
                # "1 in 6": the file's value as a share; match when a/value rounds to the stated denominator
                val = v * float(s.get("scale") or 1)
                a_, b_ = int(ratio.group(1)), int(ratio.group(2))
                row["file value"] = f"{val:.6g} (= {a_} in {a_ / val:.3g})"
                row["verdict"] = "MATCH" if round(a_ / val) == b_ else "MISMATCH"
            elif isinstance(v, (int, float)) and stated is not None and bound:
                # "under 3": the file's value must satisfy the inequality, not round to the bound
                val = v * float(s.get("scale") or 1)
                row["file value"] = f"{val:.6g}"
                row["verdict"] = "MATCH" if bound(val, stated) else "MISMATCH"
            elif isinstance(v, (int, float)) and stated is not None:
                val = v * float(s.get("scale") or 1)
                row["file value"] = f"{val:.6g}"
                if signed:  # a sign on the stated number itself is compared
                    ok = half_up(val, dec) == stated
                    row["verdict"] = "MATCH" if ok else "MISMATCH"
                else:  # unsigned: half-up magnitudes; a negative file value is flagged for a hand check
                    ok = half_up(abs(val), dec) == abs(stated)
                    row["verdict"] = ("MATCH" if val >= 0 else "MAGNITUDE ONLY") if ok else "MISMATCH"
            else:
                row["file value"] = str(v)[:60]
                row["verdict"] = "CHECK BY HAND"
        except (ValueError, KeyError, IndexError, TypeError, json.JSONDecodeError) as e:
            row["verdict"] = "KEY OR FILE ERROR"
            row["note"] = f"{e}; {row['note']}"
        rows.append(row)
    keyed = {(r["line"], r["number"]) for r in rows if r["verdict"] in ("MATCH", "MISMATCH", "CHECK BY HAND", "MAGNITUDE ONLY")}
    rows = [r for r in rows if not (r["verdict"] in ("README ONLY", "NO SOURCE") and (r["line"], r["number"]) in keyed)]
    order = {"MISMATCH": 0, "KEY OR FILE ERROR": 1, "NO SOURCE": 2, "CHECK BY HAND": 3, "MAGNITUDE ONLY": 4,
             "README ONLY": 5, "MATCH": 6}
    rows.sort(key=lambda r: (order[r["verdict"]], r["line"] or 0))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "numbers.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    counts = {k: sum(r["verdict"] == k for r in rows) for k in order}
    notmain = sum(r["on main"] == "NO" for r in rows)
    esc = lambda x: str(x).replace("|", "\\|")  # noqa: E731
    md = ["# Write-up outline numbers sheet", "",
          f"Every number in the outline tab, checked by `analyze_outline_numbers.py` (json.load by key). {len(rows)} numbers: "
          + ", ".join(f"{k} {v}" for k, v in counts.items() if v) + f". Sources not reachable from origin/main: {notmain}.", "",
          ("A MATCH means the file's value rounds half-up to the stated number at the stated precision (or satisfies a stated "
           "bound such as 'under 3'). MAGNITUDE ONLY means an unsigned outline number matches a negative file value. README "
           "ONLY means the number is stated in a committed README or plan but no JSON key holds it."), "",
          "| line | number | verdict | file value | key | source | on main | quote | note |", "|---|---|---|---|---|---|---|---|---|"]
    md += [(f"| {r['line']} | {esc(r['number'])} | {r['verdict']} | {esc(r['file value'])} | `{esc(r['key'])}` | `{esc(r['source'])}` | "
            f"{r['on main']} | {esc(r['quote'])} | {esc(r['note'])} |") for r in rows]
    (out / "NUMBERS.md").write_text("\n".join(md) + "\n")
    print(counts, "not on main:", notmain)


if __name__ == "__main__":
    main()
