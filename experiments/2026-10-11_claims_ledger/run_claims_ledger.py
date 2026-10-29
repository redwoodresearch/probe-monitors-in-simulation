"""Claims ledger: each headline number on the status page's answer lines, read back from a committed script-written file
(`git show <commit>:<path>`) and checked against the number the page prints. $0, local, read-only. Spec row sources:
- {"ref", "path", "key" ("/"-separated, or a list of parts), "scale"}, or "keys" + "expr" (arithmetic over several keys);
- {"s3_only", "manifest_ref", "manifest_path", "s3_key", "sha256"}: fails unless one entry of that committed manifest pairs
  the key with the sha256; {"none": reason}: no committed source.
A row matches when the read value, rounded half-up to the page's decimals, equals the page's number ("edge" within 1e-9 of
a boundary). Coverage: every number printed in a question's section is matched by value to its rows; the rest are listed.
Writes claims_ledger_<stamp>.json (one line: rows, coverage) and a short markdown summary; notes refs not on origin/main.

  uv run --no-project python run_claims_ledger.py --page inputs/answer_lines_v463.txt <spec.json> [...]   (inside the repo)
"""

import argparse
import ast
import decimal
import gzip
import hashlib
import json
import operator
import re
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).parent
_CACHE = {}
_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv, ast.USub: operator.neg}


def blob(ref: str, path: str):
    """The parsed JSON at ref:path (cached)."""
    if (ref, path) not in _CACHE:
        raw = subprocess.run(["git", "show", f"{ref}:{path}"], capture_output=True, check=True).stdout
        if path.endswith(".gz"):
            raw = gzip.decompress(raw)
        _CACHE[(ref, path)] = (json.loads(raw), hashlib.sha256(raw).hexdigest())
    return _CACHE[(ref, path)]


def get(obj, key):
    """key: a "/"-separated path, or a list of parts when a key name itself contains "/"."""
    for part in (key if isinstance(key, list) else key.split("/")):
        obj = obj[int(part)] if isinstance(obj, list) else obj[part]
    return obj


def evaluate(expr: str, env: dict) -> float:
    def ev(n):
        if isinstance(n, ast.Expression):
            return ev(n.body)
        if isinstance(n, ast.BinOp) and type(n.op) in _OPS:
            return _OPS[type(n.op)](ev(n.left), ev(n.right))
        if isinstance(n, ast.UnaryOp) and type(n.op) in _OPS:
            return _OPS[type(n.op)](ev(n.operand))
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)) and not isinstance(n.value, bool):
            return decimal.Decimal(repr(n.value))
        if isinstance(n, ast.Name):
            return env[n.id]
        raise ValueError(f"unsupported expression: {expr}")
    return ev(ast.parse(expr, mode="eval"))


def num(v) -> decimal.Decimal:
    """A file value as an exact decimal of its shortest repr (so 0.145 * 100 is 14.5, not 14.4999...)."""
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise TypeError(f"not a number: {type(v).__name__}")
    d = decimal.Decimal(repr(v))
    if not d.is_finite():
        raise ValueError(f"not finite: {v}")
    return d


def page_number(s: str) -> decimal.Decimal:
    return decimal.Decimal(s.replace("−", "-").replace(",", "").replace("+", "").rstrip("%"))


def manifest_has(man, key: str, sha: str) -> bool:
    """True if one manifest entry pairs the key with the sha256: a dict holding the sha and a string naming the key (its
    full path, or a path the key ends with), or a mapping from the key's name or path to the sha."""
    key = re.sub(r"^s3://[^/]+/", "", key)
    names = lambda v: isinstance(v, str) and len(v) > 4 and (key == v or key.endswith("/" + v.lstrip("/")) or v.endswith(key))
    stack = [man]
    while stack:
        x = stack.pop()
        if isinstance(x, dict):
            if sha in x.values() and any(names(v) for k, v in x.items() if v != sha):
                return True
            if any(v == sha and names(k) for k, v in x.items()):
                return True
            stack += list(x.values())
        elif isinstance(x, list):
            stack += x
    return False


def check(row: dict) -> dict:
    src, out = row["source"], dict(row)
    if "none" in src:
        return out | {"status": "no committed source", "match": None}
    if src.get("s3_only"):  # the cited manifest must exist at its commit and name the object's key and sha256
        try:
            man = json.loads(subprocess.run(["git", "show", f"{src['manifest_ref']}:{src['manifest_path']}"], capture_output=True, check=True).stdout)
            if not manifest_has(man, src["s3_key"], src["sha256"]):
                raise ValueError("no manifest entry pairs this key with this sha256")
        except Exception as e:
            return out | {"status": f"S3 manifest check failed: {type(e).__name__}: {e}", "match": False}
        return out | {"status": "S3, manifest sha only", "match": None}
    try:  # every failure is a finding for this row, never a crash of the ledger
        obj, sha = blob(src["ref"], src["path"])
        if "expr" in src:
            value = evaluate(src["expr"], {k: num(get(obj, v)) for k, v in src["keys"].items()})
        else:
            value = num(get(obj, src["key"]))
        value = value * decimal.Decimal(str(src.get("scale", 1)))
        want = page_number(row["page_value"])
        places = -want.as_tuple().exponent if want.as_tuple().exponent < 0 else 0
        q = decimal.Decimal(1).scaleb(-places)
        got = value.quantize(q, rounding=decimal.ROUND_HALF_UP)
        edge = abs(abs(value) % q - q / 2) < decimal.Decimal("1e-9")
    except Exception as e:
        return out | {"status": f"source error: {type(e).__name__}: {e}", "match": False}
    return out | {"status": "read", "read_value": str(value), "read_rounded": str(got), "match": got == want, "edge": bool(edge), "blob_sha256": sha}


NUM = re.compile(r"(?<![\w.-])[−+]?\d[\d,]*(?:\.\d+)?(?![\w])")  # an ASCII hyphen before a digit is an identifier (gpt-5-mini, Llama-3.2-1B)


def norm(s: str) -> str:
    return s.replace("−", "-").replace(",", "").lstrip("+").rstrip("%")


def coverage(page: str, rows: list) -> dict:
    """Every number printed in each question's section of the page, and which have no spec row in that question (by
    value; each row covers one occurrence). Uncovered numbers are listed with context: setup facts, sample sizes, or misses."""
    out, uncovered = {}, []
    for sec in re.split(r"\n## ", page)[1:]:
        q = sec.split()[0]
        have = {}
        for r in rows:
            if r["question"] == q:
                k = norm(r["page_value"])
                have[k] = have.get(k, 0) + 1
        found = list(NUM.finditer(sec))
        miss = 0
        for m in found:
            k = norm(m.group())
            if have.get(k, 0) > 0:
                have[k] -= 1
            else:
                miss += 1
                uncovered.append({"question": q, "number": m.group(), "context": sec[max(0, m.start() - 40):m.end() + 25].replace("\n", " ")})
        out[q] = {"numbers_on_page": len(found), "without_a_row": miss}
    return {"by_question": out, "uncovered": uncovered, "numbers_on_page": sum(v["numbers_on_page"] for v in out.values()),
            "without_a_row": len(uncovered)}


def main(specs: list, page: str):
    rows, spec_shas = [], {}
    for s in specs:
        raw = Path(s).read_bytes()
        spec_shas[str(Path(s).resolve())] = hashlib.sha256(raw).hexdigest()
        rows += json.loads(raw)["rows"]
    if len({r["id"] for r in rows}) != len(rows):
        raise SystemExit("duplicate row ids")
    res = [check(r) for r in rows]
    cov = coverage(Path(page).read_text(), rows)
    on_main = {}
    for r in res:  # is each pinned commit on origin/main, or only on a branch?
        ref = r["source"].get("ref") or r["source"].get("manifest_ref")
        if ref and ref not in on_main:
            on_main[ref] = subprocess.run(["git", "merge-base", "--is-ancestor", ref, "origin/main"], capture_output=True).returncode == 0
        r["ref_on_main"] = on_main.get(ref)
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    count = lambda rs: {"rows": len(rs), "read": sum(r["status"] == "read" for r in rs), "match": sum(r["match"] is True for r in rs),
                        "mismatch": sum(r["match"] is False for r in rs), "s3_only": sum(r["status"].startswith("S3") for r in rs),
                        "no_source": sum(r["status"] == "no committed source" for r in rs), "pinned_off_main": sum(r["ref_on_main"] is False for r in rs)}
    qs = list(dict.fromkeys(r["question"] for r in res))
    summary = count(res) | {"by_question": {q: count([r for r in res if r["question"] == q]) for q in qs}}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    out = {"label": "claims ledger: status page headline numbers read back from committed script outputs", "git": head, "specs": spec_shas,
           "page": {"file": Path(page).name, "sha256": hashlib.sha256(Path(page).read_bytes()).hexdigest()}, "summary": summary, "coverage": cov, "rows": res}
    (HERE / "results").mkdir(exist_ok=True)
    with (HERE / "results" / f"claims_ledger_{stamp}.json").open("x") as fh:  # one line: every row, source and value
        fh.write(json.dumps(out, ensure_ascii=False) + "\n")

    def where(r):
        s = r["source"]
        if s.get("s3_only"):
            return f"S3, manifest sha only: `{s.get('s3_key', '?')}` (manifest @{str(s.get('manifest_ref', '?'))[:8]})"
        if "none" in s:
            return f"none: {s['none']}"
        return f"`{s['path']}` @{str(s.get('ref', '?'))[:8]} `{s.get('key', s.get('expr'))}`"

    md = [f"Claims ledger {stamp} (PT), page answer lines v463; checker at {head[:8]}. From raw data: every 'read' value comes from",
          "`git show <commit>:<path>` at the pinned commit. Full rows: the JSON beside this file.", "",
          "| question | rows | match | mismatch | S3 only | no source | pinned off main | page numbers | without a row |",
          "|---|---|---|---|---|---|---|---|---|"]
    cq = cov["by_question"]
    for q, c in summary["by_question"].items():
        md.append(f"| {q} | {c['rows']} | {c['match']} | {c['mismatch']} | {c['s3_only']} | {c['no_source']} | {c['pinned_off_main']} | "
                  f"{cq.get(q, {}).get('numbers_on_page', '')} | {cq.get(q, {}).get('without_a_row', '')} |")
    c = summary
    md.append(f"| all | {c['rows']} | {c['match']} | {c['mismatch']} | {c['s3_only']} | {c['no_source']} | {c['pinned_off_main']} | "
              f"{cov['numbers_on_page']} | {cov['without_a_row']} |")
    md += ["", f"Coverage: {cov['numbers_on_page']} numbers on the page; {cov['without_a_row']} have no row (setup facts, sample sizes, "
           "margins, repeated mentions, or misses). Each is listed with its context in the JSON's `coverage.uncovered`."]
    for title, sel in (("Mismatches", lambda r: r["match"] is False), ("S3 only", lambda r: r["status"].startswith("S3")),
                       ("No committed source", lambda r: r["status"] == "no committed source")):
        md += ["", f"## {title}", "", "| id | claim | page | read | source |", "|---|---|---|---|---|"]
        md += [f"| {r['id']} | {r['claim']} | {r['page_value']} | {r.get('read_rounded', '')} | {where(r)} |" for r in res if sel(r)]
    (HERE / "results" / f"claims_ledger_{stamp}.md").write_text("\n".join(md) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "by_question"}), "| page numbers", cov["numbers_on_page"], "| without a row", cov["without_a_row"])
    for r in res:
        if r["match"] is False:
            print(f"MISMATCH {r['id']}: page {r['page_value']} vs read {r.get('read_rounded')} ({r['status'] if r['status'] != 'read' else r.get('read_value')}) — {r['claim']}")
    print("->", HERE / "results" / f"claims_ledger_{stamp}.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--page", required=True, help="the captured answer lines (inputs/answer_lines_*.txt)")
    ap.add_argument("specs", nargs="+")
    a = ap.parse_args()
    main(a.specs, a.page)
