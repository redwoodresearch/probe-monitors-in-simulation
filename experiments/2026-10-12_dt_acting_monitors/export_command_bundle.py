"""Blinded command bundle for Aswin's labelling (2026-10-12; $0). One scripted export pass, with Aswin's explicit OK.

Aswin, relayed by the orchestrator at about 12:27 PT: "Can the v2 key holder provide a local path to a blinded command
bundle—blind_id, tool, arguments, and tool definition only—or run an exporter I supply? I'll keep the held_key_v2 folder
unopened. Without that lookup, v2's IDs cannot be matched to commands safely."

For each row of the v2 blinded sheet, in its shuffled order, the call is joined through the held key: blind_id -> run_key,
irid, call_index. run_key maps to the run's .eval, irid to the acting model's request (x-irid), and call_index to the tool
call's position in that completion, as run_replay_readouts.py numbers it. Written per row: blind_id, tool, arguments
(the raw JSON string vLLM returned in the logged response, else the parsed arguments re-serialised, with the count of each
reported), tool_def_id. Tool definitions go to a separate deduplicated file keyed by tool_def_id.

Blinding check first: each tool's definition is hashed per arm. If a tool name has more than one definition, every row of
that tool gets ONE canonical definition (the most common by rows), so a definition id can't reveal the world.

This script never prints content: only counts, hashes and the normalisation outcome. The key's path is an argument and
is not echoed. No labelling, masking or credential logic.

  uv run --no-project --python 3.14 python -I export_command_bundle.py --key <held key csv> --sheet <v2 blind sheet> \\
      --evals <dir with the five launch folders> --out-dir <bundle dir>
"""

import argparse
import csv
import hashlib
import json
import re
import subprocess
import sys
import traceback
import zipfile
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).parent
SHEET_SHA = "707188aedd55fc7091c090e6a555bb8aa61b4e83f10897d06fc52eb94dda8385"
KEY_SHA = "8af4a987"  # prefix, as committed in results/manifest_dt_label_blind_v2.json
LAUNCHES = ["2026-10-09_000523_rerun_strict", "2026-10-09_173815_scaleup_canary", "2026-10-09_182505_scaleup",
            "2026-10-10_001314_scaleup_topup", "2026-10-10_001314_second_attack"]
SIM = {"sim_honest", "sim_attack"}
ACTING = "vllm/Qwen/Qwen3.5-27B"
REF = re.compile(r"attachment://[0-9a-f]{32}")


def resolved(x, att, depth=0):
    """Inspect's whole-value attachment references, at any depth and chained; a cycle or a missing one raises."""
    if depth > 20:
        raise ValueError("attachment reference chain too deep")
    if isinstance(x, str) and REF.fullmatch(x):
        return resolved(att[x.removeprefix("attachment://")], att, depth + 1)
    if isinstance(x, list):
        return [resolved(v, att, depth) for v in x]
    if isinstance(x, dict):
        return {k: resolved(v, att, depth) for k, v in x.items()}
    return x


def refs_left(x) -> bool:
    if isinstance(x, str):
        return bool(REF.fullmatch(x))
    if isinstance(x, list):
        return any(refs_left(v) for v in x)
    if isinstance(x, dict):
        return any(refs_left(v) for v in x.values())
    return False


def main(key: str, sheet: str, evals: str, out_dir: str):
    if subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True, check=True, cwd=HERE).stdout:
        raise SystemExit("the repo has uncommitted tracked changes")
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip()
    if hashlib.sha256(Path(sheet).read_bytes()).hexdigest() != SHEET_SHA:
        raise SystemExit("blind sheet sha256 mismatch")
    if not hashlib.sha256(Path(key).read_bytes()).hexdigest().startswith(KEY_SHA):
        raise SystemExit("key sha256 mismatch")
    order = [(r["blind_id"], r["tool"]) for r in csv.DictReader(open(sheet, newline=""))]
    keylist = list(csv.DictReader(open(key, newline="")))
    keyrows = {r["blind_id"]: r for r in keylist}
    if len(keyrows) != len(keylist):
        raise SystemExit("duplicate blind_ids in the key")
    if len(order) != len(keyrows) or {b for b, _ in order} != set(keyrows):
        raise SystemExit("sheet and key blind_ids differ")
    tool_of = dict(order)
    # run file index: path relative to its launch folder -> local file
    index = {}
    for launch in LAUNCHES:
        for p in (Path(evals) / launch).rglob("*.eval"):
            if "references" in p.relative_to(Path(evals) / launch).parts:
                continue
            rel = str(p.relative_to(Path(evals) / launch))
            if rel in index:
                raise SystemExit("a run path occurs in two launches")
            index[rel] = p
    by_run = defaultdict(list)
    for b, r in keyrows.items():
        by_run[r["run_key"].rsplit("#e", 1)[0]].append(b)
    missing = [run for run in by_run if run not in index]
    if missing:
        raise SystemExit(f"{len(missing)} runs have no local .eval")
    args_of, def_of, src, defs_by = {}, {}, Counter(), defaultdict(lambda: defaultdict(Counter))
    for run, bids in by_run.items():
        z = zipfile.ZipFile(index[run])
        names = [n for n in z.namelist() if n.startswith("samples/")]
        if len(names) != 1:
            raise SystemExit("a run file without exactly one sample")
        s = json.loads(z.read(names[0]))
        att = s.get("attachments") or {}
        events = {}
        for e in s.get("events") or []:
            irid = (((e.get("call") or {}).get("request") or {}).get("extra_headers") or {}).get("x-irid") if e.get("event") == "model" and e.get("model") == ACTING else None
            if irid is None:
                continue
            if irid in events:
                raise SystemExit("an x-irid occurs twice in one run")
            events[irid] = e
        for b in bids:
            r = keyrows[b]
            e = events.get(r["irid"])
            if e is None:
                raise SystemExit("a key row's request is not in its run")
            tcs = (((e.get("output") or {}).get("choices") or [{}])[0].get("message") or {}).get("tool_calls") or []
            ci = int(r["call_index"])
            if not 0 <= ci < len(tcs) or tcs[ci].get("function") != tool_of[b]:
                raise SystemExit("a key row's tool call doesn't match the sheet's tool")
            parsed = resolved(tcs[ci].get("arguments") or {}, att)
            resp = resolved((e.get("call") or {}).get("response") or {}, att)
            rtc = (((resp.get("choices") or [{}])[0].get("message") or {}).get("tool_calls") or [])
            raw = None
            if resp:  # a raw response was logged: its calls must agree with Inspect's parsed calls, or the run fails
                if [(t.get("function") or {}).get("name") for t in rtc] != [t.get("function") for t in tcs]:
                    raise SystemExit("raw response tool calls disagree with the parsed calls")
                raw = (rtc[ci].get("function") or {}).get("arguments")
                if not isinstance(raw, str) or json.loads(raw) != parsed:
                    raise SystemExit("a raw arguments string disagrees with the parsed arguments")
                src["raw response string"] += 1
            else:
                raw = json.dumps(parsed, ensure_ascii=False)
                src["no raw response logged: parsed arguments re-serialised"] += 1
            if refs_left(parsed) or REF.search(raw):
                raise SystemExit("an attachment reference is left in a call's arguments")
            args_of[b] = raw
            tools = resolved(e.get("tools") or [], att)
            if refs_left(tools) or REF.search(json.dumps(tools)):
                raise SystemExit("an attachment reference is left in a tool definition")
            d = [t for t in tools if t.get("name") == tool_of[b]]
            if len(d) != 1:
                raise SystemExit("a call's tool has no single definition in its request")
            dj = json.dumps(d[0], sort_keys=True, ensure_ascii=False)
            def_of[b] = dj
            defs_by[tool_of[b]][r["arm"]][hashlib.sha256(dj.encode()).hexdigest()] += 1
    # blinding check and normalisation
    canon, outcome = {}, {}
    for name, per_arm in defs_by.items():
        allc = sum(per_arm.values(), Counter())
        real = set().union(*(set(c) for a, c in per_arm.items() if a not in SIM)) if any(a not in SIM for a in per_arm) else set()
        sim = set().union(*(set(c) for a, c in per_arm.items() if a in SIM)) if any(a in SIM for a in per_arm) else set()
        outcome[name] = {"distinct_definitions": len(allc), "real_and_sim_differ": bool(real and sim and real != sim),
                         "normalized": len(allc) > 1}
        canon[name] = allc.most_common(1)[0][0]
    canon_json = {}
    for b, dj in def_of.items():
        h = canon[tool_of[b]]
        if hashlib.sha256(dj.encode()).hexdigest() == h:
            canon_json[h] = dj
    for name, h in canon.items():
        if h not in canon_json:
            raise SystemExit("a canonical definition was not captured")
    tid = {name: hashlib.sha256(canon_json[h].encode()).hexdigest()[:12] for name, h in canon.items()}
    od = Path(out_dir)
    od.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    bpath, dpath = od / f"dt_command_bundle_{stamp}.jsonl", od / f"dt_tool_definitions_{stamp}.jsonl"
    with bpath.open("x") as fb:
        for b, name in order:
            fb.write(json.dumps({"blind_id": b, "tool": name, "arguments": args_of[b], "tool_def_id": tid[name]}, ensure_ascii=False) + "\n")
    with dpath.open("x") as fd:
        for name in sorted(canon):
            fd.write(json.dumps({"tool_def_id": tid[name], "tool": name, "definition": json.loads(canon_json[canon[name]])}, ensure_ascii=False) + "\n")
    summary = {"git": git, "rows": len(order), "per_tool_rows": dict(Counter(n for _, n in order).most_common()),
               "arguments_source": dict(src), "tool_definitions": outcome,
               "definitions_differed_across_arms": any(o["real_and_sim_differ"] for o in outcome.values()),
               "bundle": bpath.name, "bundle_sha256": hashlib.sha256(bpath.read_bytes()).hexdigest(),
               "tool_definitions_file": dpath.name, "tool_definitions_sha256": hashlib.sha256(dpath.read_bytes()).hexdigest()}
    with (od / f"dt_command_bundle_summary_{stamp}.json").open("x") as fh:
        fh.write(json.dumps(summary, indent=1))
    print(json.dumps({k: summary[k] for k in ("rows", "per_tool_rows", "arguments_source", "definitions_differed_across_arms",
                                              "bundle", "bundle_sha256", "tool_definitions_file", "tool_definitions_sha256")}))
    print("tool definitions:", json.dumps(outcome))


class Parser(argparse.ArgumentParser):
    def error(self, message):  # no echo of the arguments: they include the held key's path
        sys.stderr.write("failed: bad arguments\n")
        sys.exit(2)


if __name__ == "__main__":
    ap = Parser()
    ap.add_argument("--key", required=True)
    ap.add_argument("--sheet", required=True)
    ap.add_argument("--evals", required=True)
    ap.add_argument("--out-dir", required=True)
    try:
        main(**{k.replace("-", "_"): v for k, v in vars(ap.parse_args()).items()})
    except SystemExit:
        raise
    except BaseException as exc:  # no message or path: the held key's location must not reach stderr
        line = traceback.extract_tb(exc.__traceback__)[-1].lineno
        print(f"failed: {type(exc).__name__} at line {line}", file=sys.stderr)
        sys.exit(1)
