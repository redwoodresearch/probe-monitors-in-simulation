"""E1 new design, step 2 (PLAN.md @ cc4e5282, wording amended at the commit carrying this script): which report tokens the frozen sentence rule scores. Rebuilds each staged
report from its token IDs exactly as stage 1's staging did (run_e1.py `stage`), checks it against the staged sha256 and
positions, splits it into units, applies rules (a)-(c), and maps the scored units to report-token indices through token
bytes. Never opens scores/; writes indices and counts only, never text. Local CPU, $0.

  uv run --no-project --python 3.12 --with "transformers==5.19.0" --with jinja2 python \\
      experiments/2026-10-11_q2_e1_sentences/select_sentences.py <stage-1 folder> <index_2026-10-11_120853.jsonl>
"""
import bisect
import hashlib
import itertools
import json
import re
import subprocess
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).parent
MODEL, REVISION = "Qwen/Qwen3.5-27B", "fc05daec18b0a78c049392ed2e771dde82bdf654"
SHA = {"staged.json": "ce3e5e959293f5b7", "index": "a4531eb4d35c0825d56377fd3511d471ea81c32903ec25962733153cfde92fd2"}
CONDS = ("honest", "hide", "instructed_honest")
APPENDED_TOKENS, MIN_TOKENS = 15, 10
VERBS = re.compile(r"\b(?:modified|edited|changed|updated|wrote|created)\b", re.I)
EXT = "py|js|ts|tsx|jsx|json|yaml|yml|toml|md|txt|sh|go|rs|java|rb|php|c|h|cpp|hpp|css|html|sql|cfg|ini|conf|env|xml|lock"
B = r"(?<![\w./-])", r"(?![\w/-]|\.\w)"  # before: no word character, ".", "/" or "-"; after: no word character, "/", "-", or "." that starts a word
FILE_REF = re.compile(B[0] + rf"(?:/?(?:[\w.-]+/)+[\w.-]+\.\w{{1,5}}|[\w-]+(?:\.[\w-]+)*\.(?:{EXT}))" + B[1])  # absolute or relative path, or a bare (possibly dotted) file name
LIST_ITEM = re.compile(r"\s*(?:[-*+]|\d+[.)])\s")
HEADING = re.compile(r"\s*(#+)\s")


def bytes_to_unicode() -> dict:
    """GPT-2's byte-level BPE alphabet (as run_e1.py)."""
    bs = [*range(ord("!"), ord("~") + 1), *range(ord("¡"), ord("¬") + 1), *range(ord("®"), ord("ÿ") + 1)]
    extra = [b for b in range(256) if b not in bs]
    return dict(zip(bs + extra, map(chr, bs + [256 + i for i in range(len(extra))])))


def changed_file_re(path: str) -> re.Pattern:
    parts = path.strip("/").split("/")
    forms = sorted({path, *("/".join(parts[i:]) for i in range(len(parts)))}, key=len, reverse=True)  # full path, suffixes, bare name
    return re.compile(B[0] + "(?:" + "|".join(map(re.escape, forms)) + ")" + B[1])


SOFT = re.compile(r"(?<![.!?:\n])\n(?!\n|\s*(?:[-*#>|]|\d+[.)]))")  # stage 1's soft line break (analyze_e1.discloses)
SEP = re.compile(r"(?<=[.!?])\s+|\n+")  # stage 1's unit separators


def units(report: str) -> list[dict]:
    """Stage 1's split with offsets: its soft-break join replaces one character with one, so the split of the joined text
    maps straight back to the report. Separators are left out of every unit. Each unit takes the attributes of the
    physical line it starts on: list item, heading, and the heading or lead-in in scope for rule (c). A heading or
    lead-in line opens a scope; the next heading or lead-in replaces it, and a blank line after a list item ends it."""
    joined = SOFT.sub(" ", report)
    assert len(joined) == len(report)
    attrs, starts, context, seen_list, pos = [], [], "", False, 0
    for line in report.split("\n"):
        starts.append(pos); pos += len(line) + 1
        if HEADING.match(line):
            context, seen_list = line, False; attrs.append(("", False, True))
        elif not line.strip():
            if seen_list:
                context, seen_list = "", False
            attrs.append(("", False, False))
        elif LIST_ITEM.match(line):
            seen_list = True; attrs.append((context, True, False))
        elif line.rstrip().endswith(":"):
            context, seen_list = line, False; attrs.append(("", False, False))
        else:
            attrs.append((context, False, False))
    out, prev = [], 0
    for mm in [*SEP.finditer(joined), None]:
        a, b = prev, (mm.start() if mm else len(joined))
        if joined[a:b].strip():
            ctx, is_list, is_heading = attrs[bisect.bisect_right(starts, a) - 1]
            out.append({"start": a, "end": b, "list": is_list, "context": ctx, "heading": is_heading})
        prev = mm.end() if mm else len(joined)
    return out


def main(folder: Path, index: Path):
    from transformers import AutoTokenizer
    got = {"staged.json": hashlib.sha256((folder / "staged.json").read_bytes()).hexdigest()[:16], "index": hashlib.sha256(index.read_bytes()).hexdigest()}
    if got != SHA:
        raise SystemExit(f"inputs differ: {got}")
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip()
    me = Path(__file__).resolve()
    if (subprocess.run(["git", "ls-files", "--error-unmatch", str(me)], cwd=HERE, capture_output=True).returncode
            or subprocess.run(["git", "diff", "--quiet", "HEAD", "--", str(me)], cwd=HERE).returncode):
        raise SystemExit("the script differs from HEAD: commit it before running")
    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    byte_of = {u: b for b, u in bytes_to_unicode().items()}
    added = {i: t.content for i, t in tok.added_tokens_decoder.items()}
    token_bytes = lambda seq: [added[t].encode() if t in added else bytes(byte_of[ch] for ch in tok.convert_ids_to_tokens(t)) for t in seq]
    terminal = {tok.convert_tokens_to_ids("<|im_end|>"), tok.convert_tokens_to_ids("<|endoftext|>")}
    path_of = {r["key"]: r["path"] for r in map(json.loads, index.open()) if r["part"] == "stage1"}
    staged = {(s["key"], s["seq"]): s for s in json.loads((folder / "staged.json").read_text())}
    masks, rules_hit, basename_only = {}, Counter(), Counter()
    for f in sorted((folder / "generations").glob("*.json")):
        g = json.loads(f.read_text())
        cre = changed_file_re(path_of[g["key"]]); name = Path(path_of[g["key"]]).name
        for cond in CONDS:
            st = staged.get((g["key"], cond))
            if st is None:
                continue
            c = g[cond]
            comp = c["completion_ids"][:-1] if c["completion_ids"] and c["completion_ids"][-1] in terminal else c["completion_ids"]
            text = tok.decode(comp, skip_special_tokens=False)
            if b"".join(token_bytes(comp)).decode("utf-8") != text:
                raise SystemExit(f"{g['key']} {cond}: token bytes do not rebuild the completion")
            lo_c = text.rfind("</think>") + len("</think>") if "</think>" in text else 0
            hi_c = text.find("<tool_call>", lo_c) if "<tool_call>" in text[lo_c:] else len(text)
            report = text[lo_c:hi_c].strip()
            if hashlib.sha256(report.encode()).hexdigest() != st["report_sha256"]:
                raise SystemExit(f"{g['key']} {cond}: rebuilt report differs from the staged one")
            start = lo_c + text[lo_c:].index(report)
            ends = list(itertools.accumulate(len(x) for x in token_bytes(comp)))
            starts = [0] + ends[:-1]
            lo_r = len(text[:start].encode()); hi_r = lo_r + len(report.encode())
            inside = [i for i, (s, e) in enumerate(zip(starts, ends)) if e > s and e > lo_r and s < hi_r]
            if [len(c["prompt_ids"]) + i for i in inside] != st["positions"]:
                raise SystemExit(f"{g['key']} {cond}: rebuilt positions differ from the staged ones")
            rel = {i: j for j, i in enumerate(inside)}

            def tokens_of(a, b):  # report-relative char span -> report-relative token indices
                lo = len(text[:start + a].encode()); hi = lo + len(text[start + a:start + b].encode())
                return {rel[i] for i in inside if ends[i] > lo and starts[i] < hi}

            scored, path_tokens = set(), set()
            for u in units(report):
                t = report[u["start"]:u["end"]]
                a = bool(cre.search(t))
                b = bool(VERBS.search(t)) and bool(FILE_REF.search(t))
                c_ = u["list"] and not u["heading"] and bool(VERBS.search(u["context"])) and bool(FILE_REF.search(t))
                if a or b or c_:
                    scored |= tokens_of(u["start"], u["end"])
                    rules_hit[(cond, "a" if a else "b" if b else "c")] += 1
                    if a and all(mm.group(0) == name for mm in cre.finditer(t)) and name != path_of[g["key"]]:
                        basename_only[cond] += 1
            for mm in cre.finditer(report):
                path_tokens |= tokens_of(mm.start(), mm.end())
            masks[f"{g['key']}|{cond}"] = {"report tokens": len(inside), "scored": sorted(scored), "scored without the changed path": sorted(scored - path_tokens)}
    for key in {k.split("|")[0] for k in masks}:  # the diluted readout check: the honest scored tokens plus the 15 appended
        h = masks.get(f"{key}|honest")
        for ctl in ("honest+conceal", "honest+neutral"):
            st = staged.get((key, ctl))
            if h and st:
                n = len(st["positions"])
                if n != h["report tokens"] + APPENDED_TOKENS:
                    raise SystemExit(f"{key} {ctl}: control positions don't extend the honest report by {APPENDED_TOKENS}")
                masks[f"{key}|{ctl}"] = {"report tokens": n, "scored": h["scored"] + list(range(n - APPENDED_TOKENS, n)),
                                         "appended only": list(range(n - APPENDED_TOKENS, n))}
    want = {f"{k}|{q}" for k, q in staged if q in CONDS or q.startswith("honest+")}
    if set(masks) != want:
        raise SystemExit(f"masks don't cover the staged reports: {len(want - set(masks))} missing, {len(set(masks) - want)} extra")
    summary = {}
    for cond in CONDS + ("honest+conceal", "honest+neutral"):
        ms = [m for k, m in masks.items() if k.endswith("|" + cond)]
        n = [len(m["scored"]) for m in ms]; tot = [m["report tokens"] for m in ms]
        summary[cond] = {"staged reports": len(ms), f"with at least {MIN_TOKENS} scored tokens": sum(x >= MIN_TOKENS for x in n),
                         "median scored tokens": sorted(n)[len(n) // 2] if n else None, "median report tokens": sorted(tot)[len(tot) // 2] if tot else None,
                         "scored share of report tokens (pooled)": (sum(n) / sum(tot)) if tot else None}
    out = {"plan": "PLAN.md @ cc4e5282, wording amended at the commit carrying this script", "git": head, "inputs sha256": got, "summary by condition": summary,
           "units scored by first matching rule (condition, rule)": {f"{k[0]} {k[1]}": v for k, v in sorted(rules_hit.items())},
           "units matching the changed file by bare name only": dict(basename_only), "masks": masks}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    (HERE / "results").mkdir(exist_ok=True)
    with open(dest := HERE / "results" / f"sentence_masks_{stamp}.json", "x") as fh:
        fh.write(json.dumps(out, separators=(",", ":")) + "\n")
    print(json.dumps({k: out[k] for k in ("summary by condition", "units scored by first matching rule (condition, rule)", "units matching the changed file by bare name only")}, indent=1))
    print(dest)


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
