"""G2's local site count (PLAN.md here, committed before this runs). Local, $0, counts only.

For #36's 3,807 runs, every text_editor create, insert and str_replace output as the dump reads it (the direction count's
reader, ../2026-10-12_q2_whitespace_direction): outputs, outputs ending in "\\n", and harness-truncated ones
(`<START_TOOL_OUTPUT>`). An independent reference for the build's D4 site counts, not a gate (PLAN.md): the eval files
join multi-part text with a newline, as Inspect's `.text` does, so outputs with more than one text part are counted apart.

  PYTHONPATH=src uv run --no-project --python 3.14 python experiments/2026-10-13_q2_llama_leftover/count_local_sites.py
"""
import hashlib
import json
import subprocess
import zipfile
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).parent
REPO = HERE.parents[1]
RUNS = REPO / "experiments/2026-10-06_q3_attack_detector/results/run_scores_2026-10-06_181548/runs.json"
RUNS_SHA = "76b11486cc353ab1eec390d49ef7449d729d4039d1e85643541d6679d73c3b96"
SPLITS = REPO / "splits/exclusions_v3.json"
DATA = Path.home() / "probe-shift-viewer/.run_cache/viewer/eval"
CMDS = ("create", "insert", "str_replace")
TRUNCATED = "<START_TOOL_OUTPUT>"  # fixes.TRUNCATED


def text(c, att):
    r = lambda s: att[s[len("attachment://"):]] if isinstance(s, str) and s.startswith("attachment://") else s  # noqa: E731
    return r(c) if isinstance(c, str) else "\n".join(r(x.get("text")) or "" for x in c or [] if x.get("type") == "text")


def outputs(sample):
    """(command, text, multi-part) per text_editor output, as the dump reads it: the error message when present."""
    calls, out, att = {}, [], sample.get("attachments") or {}
    for m in sample["messages"]:
        if m["role"] == "assistant":
            for tc in m.get("tool_calls") or []:
                calls[tc.get("id")] = (tc["function"], (tc.get("arguments") or {}).get("command"))
        elif m["role"] == "tool":
            fn, cmd = calls[m.get("tool_call_id")]  # an output without its call fails loudly
            if fn != "text_editor" or cmd not in CMDS:
                continue
            e = m.get("error")
            body = text(m.get("content"), att)
            c = m.get("content")
            multi = isinstance(c, list) and sum(x.get("type") == "text" for x in c) > 1
            out.append((cmd, (text(e.get("message"), att) or body) if e else body, multi))
    return out


def load(s):
    with zipfile.ZipFile(DATA / s["cohort"] / s["file"]) as z:
        for n in z.namelist():
            if n.startswith("samples/") and n.endswith(".json"):
                x = json.loads(z.read(n))
                if (str(x["id"]), str(x["epoch"])) == (str(s["sample_id"]), str(s["epoch"])):
                    return outputs(x)
    raise KeyError(s["key"])


def main():
    for f in (Path(__file__).resolve(), HERE / "PLAN.md"):
        if subprocess.run(["git", "ls-files", "--error-unmatch", str(f)], cwd=HERE, capture_output=True).returncode \
                or subprocess.run(["git", "diff", "--quiet", "HEAD", "--", str(f)], cwd=HERE).returncode:
            raise SystemExit(f"{f.name} differs from HEAD: commit it before running")
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip()
    if hashlib.sha256(RUNS.read_bytes()).hexdigest() != RUNS_SHA:
        raise SystemExit("not #36's runs.json")
    runs = json.loads(RUNS.read_text())
    src = {s["key"]: s for s in json.loads(SPLITS.read_text())["samples"]}
    tot = {}
    runs_with_site = Counter()
    for r in runs:
        s = src[r["key"]]
        assert s["arm"] == r["arm"], r["key"]
        acc = tot.setdefault(r["arm"], Counter())
        acc["runs"] += 1
        site = False
        for cmd, t, multi in load(s):
            nl, tr = t.endswith("\n"), TRUNCATED in t
            acc[f"{cmd}: outputs"] += 1
            acc[f"{cmd}: ending in a newline"] += nl
            acc[f"{cmd}: harness-truncated"] += tr
            acc[f"{cmd}: more than one text part"] += multi
            acc[f"{cmd}: ending in a newline, not truncated (D4 sites)"] += nl and not tr
            site |= nl and not tr
        runs_with_site[r["arm"]] += site
    rec = {"plan": "PLAN.md (this folder)", "git": head, "runs.json sha256": RUNS_SHA,
           "splits sha256": hashlib.sha256(SPLITS.read_bytes()).hexdigest(),
           "counts by arm": {a: dict(sorted(v.items())) for a, v in sorted(tot.items())},
           "runs with at least one D4 site": dict(runs_with_site)}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    (HERE / "results").mkdir(exist_ok=True)
    with open(rp := HERE / "results" / f"local_sites_{stamp}.json", "x") as fh:
        fh.write(json.dumps(rec, indent=1) + "\n")
    print(json.dumps(rec, indent=1)); print(rp)


if __name__ == "__main__":
    main()
