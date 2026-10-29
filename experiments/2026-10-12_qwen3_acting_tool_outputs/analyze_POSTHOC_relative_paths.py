"""Relative-path misses: invented files or wrong path resolution? (POST HOC; rule in RULE_POSTHOC_relative_paths.md,
committed at 232062ab before counting.) Honest runs only, $0, saved messages (the last model call's input for 10-06
runs that saved none). Writes results/relative_paths_POSTHOC_<stamp>.json: counts, editor bases and bash directories
(directory names only, no transcript text).

  uv run --no-project --python 3.14 python -I analyze_POSTHOC_relative_paths.py --data <10-09 dir> --pilot <10-06 dir>
"""

import argparse
import json
import re
import subprocess
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).parent))
import check_tool_outputs as C  # noqa: E402

HERE = Path(__file__).parent
MISS = re.compile(r"The path (\S+) does not exist")
LISTING = re.compile(r"^\s*(ls|find|tree)\b")
B = r"""[\s'"/():,]"""


def token(s: str) -> re.Pattern:
    return re.compile(rf"(?:^|(?<={B})){re.escape(s)}(?=$|{B})", re.M)


def clean(text: str) -> str:
    return "\n".join(line for line in text.splitlines() if "No such file" not in line)


def messages_for(s: dict) -> list[dict]:
    msgs = s["messages"]
    if any(m["role"] == "assistant" for m in msgs):
        return msgs
    events = [e for e in s["events"] if e.get("event") == "model"]
    if not events:
        return []
    last = max(events, key=lambda e: sum(b - a for a, b in e.get("input_refs") or []))
    return [m for a, b in last.get("input_refs") or [] for m in s["events_data"]["messages"][a:b]]


def run_counts(s: dict, calls_c: Counter, uniq: dict, bases: Counter, key: str, conforming: bool = False) -> dict:
    """conforming=True follows the committed rule exactly (the two disclosed deviations off); the default is as run."""
    att = s.get("attachments") or {}
    msgs = messages_for(s)
    task_text = "\n".join(C.text_of(m["content"], att) for m in msgs if m["role"] in ("system", "user"))
    res = {m.get("tool_call_id"): m for m in msgs if m["role"] == "tool"}
    calls = []  # (function, args, output text, is_error) in order
    for m in msgs:
        for tc in (m.get("tool_calls") or []) if m["role"] == "assistant" else []:
            r = res.get(tc.get("id"))
            if r is not None:
                calls.append((tc.get("function"), tc.get("arguments") or {}, C.as_sent(r, att), bool(r.get("error"))))
    pwd = sorted({out.strip().splitlines()[0] for f, a, out, err in calls
                  if f == "bash" and str(a.get("cmd") or a.get("command") or "").strip() == "pwd" and not err and out.strip()})
    ok_abs = [a.get("path") for f, a, out, err in calls
              if f == "text_editor" and not err and a.get("command") != "create" and str(a.get("path") or "").startswith("/")]
    run_bases = set()
    for i, (f, a, out, err) in enumerate(calls):
        p = str(a.get("path") or "")
        if f != "text_editor" or a.get("command") not in ("view", "str_replace", "insert") or not p or p.startswith("/") or not err:
            continue
        rel = p.removeprefix("./")
        mp = next((x for x in MISS.findall(out) if x.endswith("/" + rel)), None)
        if mp is None:
            continue
        base = mp[: -(len(rel) + 1)] or "/"
        bases[base] += 1
        run_bases.add(base)
        names = {rel, rel.rsplit("/", 1)[-1]}
        earlier = calls[:i]
        confirmed = any(q.endswith("/" + rel) and (conforming or q != mp) for q in ok_abs)
        listed = any(token(n).search(o if conforming else clean(o)) for ff, aa, o, e in earlier for n in names
                     if ff == "bash" and (conforming or not e) and LISTING.match(str(aa.get("cmd") or aa.get("command") or "")))
        seen = any(token(n).search(clean(o) if (ff == "bash" or not conforming) else o) for ff, aa, o, e in earlier for n in names if not e)
        bucket = "exists_elsewhere" if (confirmed or listed) else ("never_seen" if not seen else "undetermined")
        calls_c[bucket] += 1
        calls_c["exists_elsewhere_by_successful_absolute_call"] += confirmed
        calls_c["exists_elsewhere_by_earlier_listing"] += listed
        calls_c["relative_misses"] += 1
        u = uniq.setdefault((key, rel), set())
        u.add(bucket)
        # added after the rule (not in it): did the name come from the task text (system and user messages)?
        u.add("task_has_path" if token(rel).search(task_text) else "task_lacks_path")
        u.add("task_has_basename" if token(rel.rsplit("/", 1)[-1]).search(task_text) else "task_lacks_basename")
    listing_calls = sum(f == "bash" and bool(LISTING.match(str(a.get("cmd") or a.get("command") or ""))) for f, a, out, err in calls)
    return {"pwd": pwd, "editor_bases": sorted(run_bases), "listing_calls": listing_calls}


def main(data: str, pilot: str, conforming: bool = False):
    if subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True, check=True, cwd=HERE).stdout:
        raise SystemExit("the repo has uncommitted tracked changes")
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip()
    out = {"git": git, "rule": "RULE_POSTHOC_relative_paths.md @ 232062ab", "conforming": conforming}
    for label, root, launches in (("1009", Path(data), C.RUNS_1009), ("1006", Path(pilot), C.PILOT_1006)):
        calls_c, uniq, bases, per_run = Counter(), {}, Counter(), {}
        for launch in launches:
            for ev in sorted(p for p in (root / launch).rglob("*.eval") if "references" not in p.parts):
                key = str(ev.relative_to(root).parent)
                per_run[key] = run_counts(C.load_sample(ev), calls_c, uniq, bases, key, conforming)
        u = Counter()
        for b in uniq.values():
            k = b & {"exists_elsewhere", "never_seen", "undetermined"}
            k = "exists_elsewhere" if "exists_elsewhere" in k else ("never_seen" if k == {"never_seen"} else "undetermined")
            u[k] += 1
            u[f"{k}_task_has_path"] += "task_has_path" in b
            u[f"{k}_task_has_basename"] += "task_has_basename" in b
        miss_runs = [r for r in per_run.values() if r["editor_bases"]]
        u["runs_with_a_relative_miss"] = len(miss_runs)
        u["of_those_runs_with_any_ls_find_tree_call"] = sum(r["listing_calls"] > 0 for r in miss_runs)
        cmp = Counter()
        for r in per_run.values():
            if not r["editor_bases"]:
                continue
            cmp["undetermined_no_pwd" if not r["pwd"] else ("same" if set(r["pwd"]) == set(r["editor_bases"]) else "different")] += 1
        out[label] = {"calls": dict(calls_c), "unique_run_paths": dict(u), "n_unique_run_paths": len(uniq),
                      "editor_bases": dict(bases), "bash_pwd_vs_editor_base_runs": dict(cmp), "per_run": per_run}
        print(label, "calls:", dict(calls_c))
        print(label, "unique run+path:", len(uniq), dict(u))
        print(label, "editor bases:", dict(bases), "| bash pwd vs editor base (runs with a relative miss):", dict(cmp))
        print(label, "bash pwd values:", dict(Counter(p for r in per_run.values() for p in r["pwd"])))
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    dst = HERE / "results" / f"relative_paths_POSTHOC_{'conforming_' if conforming else ''}{stamp}.json"
    with dst.open("x") as fh:
        fh.write(json.dumps(out))
    print("->", dst)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--pilot", required=True)
    ap.add_argument("--conforming", action="store_true", help="follow the committed rule exactly (#990 review)")
    main(**vars(ap.parse_args()))
