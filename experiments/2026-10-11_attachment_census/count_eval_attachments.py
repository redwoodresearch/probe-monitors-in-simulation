"""Attachment census, part 1: unresolved `attachment://` references in Inspect .eval logs, as integer counts only.

Inspect stores a long string once per sample under `attachments[<hash>]` and writes `attachment://<hash>` in its place
in `messages` and `events`. A pipeline that reads `messages` without looking the hash up feeds the placeholder to the
reader instead of the text. Per cohort or arm folder this counts, over every sample of every .eval file:
- refs_in_messages: occurrences of the string "attachment://" in the serialized `messages`;
- unresolvable_in_messages: message strings that start with "attachment://" and whose hash is not in `attachments`;
- refs_in_reader_view: occurrences in `probe_shift.transcripts.reader_messages(sample, system_prompt(sample))`, which is
  what the off-policy (Qwen3-32B) staging builds from a sample (it resolves nothing);
- refs_in_reader_view_resolved: the same after replacing every resolvable reference by its text first (expected 0);
- refs_in_model_event_inputs: occurrences in the acting/simulator model events' `input` and `call.request`;
- attachment_keys: entries in `attachments`.
Attack folders are refused unless `--attack-go` names the go (Aswin's 10-11 10:19:53 rule: "No new pass over
attack-run text, scripted or not"; his go for this census, just before 16:57 PT via 0d: "Yes, counts only is fine").
Only integer counts, with folder labels and paths, are printed or written; never matched text or its context.

  PYTHONPATH=src uv run --no-project --python 3.12 --with boto3 --with pyyaml python \
      experiments/2026-10-11_attachment_census/count_eval_attachments.py --dirs label=<folder>,... --out <json>
"""

import argparse
import json
import subprocess
import zipfile
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

NEEDLE = "attachment://"
KEYS = ("files", "samples", "attachment_keys", "refs_in_messages", "samples_with_refs_in_messages", "unresolvable_in_messages",
        "refs_in_reader_view", "samples_with_refs_in_reader_view", "refs_in_reader_view_resolved", "model_events",
        "refs_in_model_event_inputs", "refs_in_model_event_requests")  # every key written, zeros included


def resolve(x, att: dict):
    if isinstance(x, str):
        return att.get(x.removeprefix(NEEDLE), x) if x.startswith(NEEDLE) else x
    if isinstance(x, list):
        return [resolve(v, att) for v in x]
    if isinstance(x, dict):
        return {k: resolve(v, att) for k, v in x.items()}
    return x


def unresolvable(x, att: dict) -> int:
    if isinstance(x, str):
        return int(x.startswith(NEEDLE) and x.removeprefix(NEEDLE) not in att)
    if isinstance(x, list):
        return sum(unresolvable(v, att) for v in x)
    if isinstance(x, dict):
        return sum(unresolvable(v, att) for v in x.values())
    return 0


def count_file(path: str) -> Counter:
    from probe_shift.transcripts import reader_messages, system_prompt

    c = Counter(files=1)
    with zipfile.ZipFile(path) as z:
        for name in z.namelist():
            if not (name.startswith("samples/") and name.endswith(".json")):
                continue
            s = json.loads(z.read(name))
            att, msgs = s.get("attachments") or {}, s.get("messages") or []
            c["samples"] += 1
            c["attachment_keys"] += len(att)
            n = json.dumps(msgs).count(NEEDLE)
            c["refs_in_messages"] += n
            c["samples_with_refs_in_messages"] += int(n > 0)
            c["unresolvable_in_messages"] += unresolvable(msgs, att)
            for e in s.get("events") or []:
                if e.get("event") == "model":
                    c["model_events"] += 1
                    c["refs_in_model_event_inputs"] += json.dumps(e.get("input")).count(NEEDLE)
                    c["refs_in_model_event_requests"] += json.dumps((e.get("call") or {}).get("request")).count(NEEDLE)
            try:
                k = json.dumps(reader_messages(s, system_prompt(s))).count(NEEDLE)
                c["refs_in_reader_view"] += k
                c["samples_with_refs_in_reader_view"] += int(k > 0)
            except (KeyError, TypeError, AttributeError) as e:  # counted, never silent
                c[f"reader_view_error_{type(e).__name__}"] += 1
            try:
                fixed = {**s, "messages": resolve(msgs, att)}
                c["refs_in_reader_view_resolved"] += json.dumps(reader_messages(fixed, system_prompt(fixed))).count(NEEDLE)
            except (KeyError, TypeError, AttributeError) as e:
                c[f"reader_view_resolved_error_{type(e).__name__}"] += 1
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", required=True, help="label=folder,... (honest folders only)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--attack-go", default="", help="the go that allows attack folders (quoted, with its source)")
    a = ap.parse_args()
    dirs = dict(x.split("=", 1) for x in a.dirs.split(","))
    dirs = {lab: str(Path(d).expanduser().resolve()) for lab, d in dirs.items()}
    missing = [d for d in dirs.values() if not Path(d).is_dir()]
    if missing:
        raise SystemExit(f"not a folder: {missing}")
    files = {lab: sorted(str(p.resolve()) for p in Path(d).rglob("*.eval")) for lab, d in dirs.items()}
    bad = ([p for fs in files.values() for p in fs if "attack" in p.lower()] + [d for d in dirs.values() if "attack" in d.lower()]
           + [lab for lab in dirs if "attack" in lab.lower()])
    if bad and not a.attack_go:
        raise SystemExit(f"refused: {len(bad)} attack paths (no attack-run pass without Aswin's go)")
    out = {}
    with ProcessPoolExecutor() as ex:
        for lab, fs in files.items():
            tot = Counter()
            for c in ex.map(count_file, fs, chunksize=4):
                tot += c
            out[lab] = {"folder": dirs[lab], "eval_files": len(fs), **{k: tot.get(k, 0) for k in KEYS},
                        **{k: v for k, v in sorted(tot.items()) if k not in KEYS}}
            print(lab, json.dumps({k: v for k, v in out[lab].items() if k != "folder"}), flush=True)
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True).stdout.strip())
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d %H:%M:%S %Z")
    Path(a.out).write_text(json.dumps({"git": {"commit": git, "dirty": dirty}, "written": stamp, "needle": NEEDLE,
                                       "attack_go": a.attack_go or None, "counts": out}, indent=1))
    print("->", a.out)


if __name__ == "__main__":
    main()
