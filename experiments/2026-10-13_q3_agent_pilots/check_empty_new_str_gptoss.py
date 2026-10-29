"""gpt-oss-120b's str_replace calls with an empty new_str: the model's own, or the strict-tools bug? (d9, 10-13). Read-only,
$0. Keys, lengths and hashes only: no argument values are written or printed.

  uv run --no-project --python 3.14 --with tokenizers python -I check_empty_new_str_gptoss.py \\
      --tokenizer <gpt-oss tokenizer.json at b5c939de> --runs <launch dir> [...]

Per policy model call, joined to the request-log proxy's 200 record by its x-irid header (as check_tool_outputs_pilots.py):
- raw: the record's generated token IDs, decoded with the pinned tokenizer; each `to=functions.text_editor ...
  <|message|>{json}<|call|>` block is one raw call (unparseable JSON skipped, counted);
- output: the str_replace calls in the model call's output, as Inspect received them from vLLM;
- executed: the text_editor tool event with the same tool call id (the call as run).
Each output str_replace is matched one-to-one to an unused raw str_replace in the same request with the same sha256 of
old_str and the same path (Codex, 10-13). Several unused candidates in different new_str states make it "ambiguous" (no
classification is assigned). Its executed new_str state (absent, empty = literal "", whitespace-only, non-empty) is
compared with the raw one:
- (a) both empty: the model's own deletion; (a') both whitespace-only;
- (b) raw non-empty or whitespace-only, executed empty or absent: the strict-tools bug's signature;
- (c) raw absent;
- unmatched: no unused raw str_replace with that old_str and path in the same request.
Writes results/empty_new_str_gptoss_<stamp>.json.
"""

import argparse
import glob
import gzip
import hashlib
import json
import re
import subprocess
import zipfile
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from tokenizers import Tokenizer

HERE = Path(__file__).parent
# the recipient comes before or after the channel: `to=functions.X<|channel|>commentary json<|message|>` or
# `<|channel|>commentary to=functions.X <|constrain|>json<|message|>`
CALL = re.compile(r"to=functions\.([A-Za-z_.]+)[^<]*(?:<\|(?:channel|constrain)\|>[^<]*)*<\|message\|>(.*?)<\|call\|>", re.S)


def state(a: dict) -> str:
    v = a.get("new_str", ...)
    return ("absent" if v is ... or v is None else "empty" if str(v) == "" else "whitespace-only" if not str(v).strip()
            else "non-empty")


def key(a: dict) -> tuple[str, str]:
    return hashlib.sha256(str(a.get("old_str") or "").encode()).hexdigest(), str(a.get("path"))


def raw_by_request(launch: Path, tok: Tokenizer) -> tuple[dict[str, list[dict]], Counter]:
    out, c = {}, Counter()
    for f in sorted((launch.parent / f"{launch.name}.request_logs").glob("requests_*.jsonl.gz")):
        with gzip.open(f, "rt") as fh:
            for line in fh:
                x = json.loads(line)
                if x.get("kind") == "header" or x.get("status") != 200 or not x.get("prompt_token_ids"):
                    continue
                calls = []
                for ch in x.get("choices") or []:
                    for name, body in CALL.findall(tok.decode(ch.get("token_ids") or [], skip_special_tokens=False)):
                        try:
                            a = json.loads(body)
                        except ValueError:
                            c["raw calls with unparseable JSON"] += 1
                            continue
                        if name == "text_editor" and isinstance(a, dict) and a.get("command") == "str_replace":
                            calls.append({"key": key(a), "state": state(a), "used": False})
                out[x["request_id"]] = calls
    return out, c


def main(tokenizer: str, runs: list[str], git: str | None = None):
    git = git or subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip()
    tok = Tokenizer.from_file(tokenizer)
    res = {"git": git, "vocab_source": "openai/gpt-oss-120b at revision b5c939de8f754692c1647ca79fbf85e8c1e70f8a",
           "vocab_file_sha256": hashlib.sha256(Path(tokenizer).read_bytes()).hexdigest(), "launches": {}}
    for run in map(Path, runs):
        raw, c = raw_by_request(run, tok)
        rows = []
        for f in sorted(glob.glob(f"{run}/honest_*/*/*.eval")):
            with zipfile.ZipFile(f) as z:
                names = [n for n in z.namelist() if n.startswith("samples/") and n.endswith(".json")]
                s = json.loads(z.read(names[0])) if names else {"events": []}
            att = s.get("attachments") or {}  # long strings in events are stored as attachment://<hash>
            resolve = lambda a: {k: att.get(v[len("attachment://"):], v) if isinstance(v, str) and v.startswith("attachment://") else v
                                 for k, v in a.items()}  # noqa: E731
            tools = {e.get("id"): e for e in s.get("events", []) if e.get("event") == "tool"}
            for e in s.get("events", []):
                if e.get("event") != "model" or not ((e.get("call") or {}).get("request") or {}).get("extra_headers"):
                    continue
                irid = e["call"]["request"]["extra_headers"].get("x-irid")
                if irid not in raw:
                    c["model calls without a 200 record"] += 1
                    continue
                for ch in (e.get("output") or {}).get("choices") or []:
                    for tc in (ch.get("message") or {}).get("tool_calls") or []:
                        a = tc.get("arguments") or {}
                        if tc.get("function") != "text_editor" or not isinstance(a, dict) or a.get("command") != "str_replace":
                            continue
                        a = resolve(a)
                        ex = tools.get(tc.get("id"))
                        ex_state = state(resolve(ex.get("arguments") or {})) if ex else "no tool event"
                        c[f"output new_str {state(a)}"] += 1
                        c[f"executed new_str {ex_state}"] += 1
                        cands = [x for x in raw[irid] if x["key"] == key(a) and not x["used"]]
                        r = None
                        if not cands:
                            case = "unmatched"
                        elif len({x["state"] for x in cands}) > 1:
                            case = "ambiguous"
                        else:
                            cands[0]["used"] = True
                            r = cands[0]["state"]
                            case = ("both non-empty" if r == ex_state == "non-empty" else "(a) both empty" if r == ex_state == "empty"
                                    else "(a') both whitespace-only" if r == ex_state == "whitespace-only"
                                    else "(b) raw has text, executed empty or absent" if r in ("non-empty", "whitespace-only") and ex_state in ("empty", "absent")
                                    else "(c) raw absent" if r == "absent" else f"other: raw {r}, executed {ex_state}")
                        c[case] += 1
                        if case == "both non-empty":
                            continue
                        rows.append({"run": "/".join(Path(f).parts[-3:-1]), "request_id": irid, "tool_call_id": tc.get("id"), "raw": r, "output": state(a),
                                     "executed": ex_state, "case": case, "old_str_len": len(str(a.get("old_str") or "")),
                                     "executed_error": ((ex or {}).get("error") or {}).get("type")})
        c["raw str_replace calls matched to no output call"] = sum(not x["used"] for v in raw.values() for x in v)
        res["launches"][run.name] = {"raw_str_replace_calls": sum(len(v) for v in raw.values()),
                                     "raw_by_state": dict(Counter(x["state"] for v in raw.values() for x in v)), "counts": dict(c), "rows": rows}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    dst = HERE / "results" / f"empty_new_str_gptoss_{stamp}.json"
    with dst.open("x") as fh:
        fh.write(json.dumps(res, indent=1))
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "rows"} for k, v in res["launches"].items()}, indent=1))
    print("->", dst)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--git", help="the commit, where git isn't available (the Modal rerun)")
    main(**vars(ap.parse_args()))
