"""Gate row 5's whitespace direction count (PLAN.md here, committed before this ran). Local, $0, counts only.

For #36's 939 cells (real second run vs simulated run), applies b7's W1, W2 and W3 (probe_shift.scrub's patterns) to every
tool output as the dump reads it, and counts per run, per cell class and per tool which arm has more of each.

  PYTHONPATH=src uv run --no-project --python 3.14 python experiments/2026-10-12_q2_whitespace_direction/count_whitespace_direction.py
"""
import hashlib
import json
import subprocess
import zipfile
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from probe_shift.scrub import BLANK_RUNS, TRAILING_SPACE

HERE = Path(__file__).parent
REPO = HERE.parents[1]
RUNS = REPO / "experiments/2026-10-06_q3_attack_detector/results/run_scores_2026-10-06_181548/runs.json"
RUNS_SHA = "76b11486cc353ab1eec390d49ef7449d729d4039d1e85643541d6679d73c3b96"
SPLITS = REPO / "splits/exclusions_v3.json"
DATA = Path.home() / "probe-shift-viewer/.run_cache/viewer/eval"
ARMS = ("real-attack-e2", "sim-attack")
OPS = {"W1": lambda t: TRAILING_SPACE.sub("", t), "W2": lambda t: BLANK_RUNS.sub("\n\n", t), "W3": lambda t: t.rstrip()}


def text(c, att):
    """Inspect's ChatMessage.text (text parts joined with a newline), with attachment:// references resolved as Inspect's
    reader does (sce's tool_result_text reads message.text)."""
    r = lambda s: att[s[len("attachment://"):]] if isinstance(s, str) and s.startswith("attachment://") else s  # noqa: E731
    return r(c) if isinstance(c, str) else "\n".join(r(x.get("text")) or "" for x in c or [] if x.get("type") == "text")


def outputs(sample):
    """(tool group, text) per tool output, as the dump reads it: the error message when present, submit dropped."""
    calls, out, att = {}, [], sample.get("attachments") or {}
    for m in sample["messages"]:
        if m["role"] == "assistant":
            for tc in m.get("tool_calls") or []:
                a = tc.get("arguments") or {}
                calls[tc.get("id")] = tc["function"] + (f":{a.get('command')}" if tc["function"] == "text_editor" else "")
        elif m["role"] == "tool":
            g = calls[m.get("tool_call_id")]  # an output without its call fails loudly
            if g == "submit":
                continue
            e = m.get("error")
            body = text(m.get("content"), att)
            out.append((g, (text(e.get("message"), att) or body) if e else body))
    return out


def load(s):
    with zipfile.ZipFile(DATA / s["cohort"] / s["file"]) as z:
        for n in z.namelist():
            if n.startswith("samples/") and n.endswith(".json"):
                x = json.loads(z.read(n))
                if (str(x["id"]), str(x["epoch"])) == (str(s["sample_id"]), str(s["epoch"])):
                    return outputs(x)
    raise KeyError(s["key"])


def group(g):
    return "text_editor" if g.startswith("text_editor") else "bash" if g == "bash" else "other"


def main():
    for f in (Path(__file__).resolve(), HERE / "PLAN.md"):
        if subprocess.run(["git", "ls-files", "--error-unmatch", str(f)], cwd=HERE, capture_output=True).returncode \
                or subprocess.run(["git", "diff", "--quiet", "HEAD", "--", str(f)], cwd=HERE).returncode:
            raise SystemExit(f"{f.name} differs from HEAD: commit it before running")
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip()
    if hashlib.sha256(RUNS.read_bytes()).hexdigest() != RUNS_SHA:
        raise SystemExit("not #36's runs.json")
    runs = [r for r in json.loads(RUNS.read_text()) if r["arm"] in ARMS]
    src = {s["key"]: s for s in json.loads(SPLITS.read_text())["samples"]}
    by_cell = defaultdict(dict)
    for r in runs:
        by_cell[r["cell"]][r["arm"]] = r
    cells = sorted(c for c, v in by_cell.items() if set(v) == set(ARMS))
    assert len(cells) == 939, len(cells)
    # per run: per (scope, op): removed chars, outputs changed, outputs; scope = "all", a tool group, or a text_editor command
    per = {}
    for c in cells:
        for a in ARMS:
            r = by_cell[c][a]; s = src[r["key"]]
            assert s["arm"] == a and s["cell"] == c
            acc = defaultdict(Counter)
            for g, t in load(s):
                for scope in {"all", group(g), g}:
                    acc[scope]["outputs"] += 1
                    acc[scope]["W3 outputs ending in whitespace"] += t != t.rstrip()
                    for op, fn in OPS.items():
                        d = len(t) - len(fn(t))
                        acc[scope][f"{op} chars removed"] += d; acc[scope][f"{op} outputs changed"] += d > 0
            per[(c, a)] = acc
    scopes = sorted({k for acc in per.values() for k in acc})
    res = {}
    for scope in scopes:
        cls = {op: Counter() for op in OPS}
        tot = {a: Counter() for a in ARMS}
        for c in cells:
            e, sm = per[(c, ARMS[0])].get(scope, Counter()), per[(c, ARMS[1])].get(scope, Counter())
            for a, acc in zip(ARMS, (e, sm)):
                tot[a].update(acc)
            for op in OPS:
                x, y = e[f"{op} chars removed"], sm[f"{op} chars removed"]
                cls[op]["real has more" if x > y else "simulated has more" if y > x else "equal"] += 1
        res[scope] = {"cells by class (chars removed per run)": {op: dict(v) for op, v in cls.items()},
                      "per-arm totals": {a: dict(v) for a, v in tot.items()},
                      "per-output rates (outputs changed / outputs)": {a: {op: (tot[a][f"{op} outputs changed"] / tot[a]["outputs"] if tot[a]["outputs"] else None) for op in OPS} for a in ARMS}}
    rec = {"plan": "PLAN.md (this folder)", "git": head, "runs.json sha256": RUNS_SHA, "splits sha256": hashlib.sha256(SPLITS.read_bytes()).hexdigest(),
           "cells": len(cells), "arms": {"real": ARMS[0], "simulated": ARMS[1]}, "scopes": res}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    (HERE / "results").mkdir(exist_ok=True)
    with open(rp := HERE / "results" / f"whitespace_direction_{stamp}.json", "x") as fh:
        fh.write(json.dumps(rec, indent=1) + "\n")
    print(rp)


if __name__ == "__main__":
    main()
