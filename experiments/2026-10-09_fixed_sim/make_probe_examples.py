"""Outputs at extreme-scoring end-of-output readouts: seeded examples for the status page (free; local CPU; no fits). Probe step A's saved
per-output scores, fixed sim vs real e1, times masked (level 1), the 5 test environments.

  most simulated-looking: simulated outputs at or above their outer fold's 95th score percentile (scores are comparable
      only within a fold's probe), of types no fix rule touches (analyze_review_checks.rule_type v2 is False: not a
      text_editor insert or str_replace output, no dpkg marker, no parseable clock stamp).
  most real-looking: real e1 outputs at or below the fold's 5th percentile, the same types.
  draws: candidates sorted by id, shuffled with random.Random(SEED + side), taken in order; a drawn output listed in
      --skip (its content, or its twin's, is about the attack itself rather than ordinary machine output; decided by
      reading) is skipped and counted, and the next one is taken.
  twin: the other arm's run of the same cell (main and side task): its output at the same (step, action) if it has the
      same type label, else its closest output of the same type label, else its output at the same (step, action), else
      its closest output; the file says which.
  shown: run key, environment, main task, step; each output's type label (from the call, which is never shown), score
      and fold percentile; the level-1 output text the probe read, credential-like strings replaced by <masked>, cut to
      20 lines of at most 200 characters. --readings (JSON {pair id: text}) adds my reading of each pair.
Writes data/probe_examples_<stamp>.md (transcript text; git-ignored, archived to S3) and results/probe_examples_<stamp>.json
(no text: the draws, twins, skips, and the .md's sha256).

  PYTHONPATH=src uv run --no-project --with numpy --with scipy --with scikit-learn python experiments/2026-10-09_fixed_sim/make_probe_examples.py \
      <claimB_output_scores_A_*.json.gz> <local 2026-10-09_la_fixed-sim_v2> [--skip id,id] [--readings readings.json] [--draft out.md]
"""

import argparse
import gzip
import hashlib
import json
import random
import re
import subprocess
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import fixes  # noqa: E402
from analyze_review_checks import TEST, rule_type  # noqa: E402

from probe_shift.scrub import scrub  # noqa: E402

SEED, N_SIM, N_REAL, LINES, WIDTH = "probe-examples-2026-10-10:", 6, 3, 20, 200
ARMS = {"real-attack-e1": "real-attack-e1", "sim-attack": "sim-attack-fixed"}
KEYWORD = r"(?:pass(?:word|wd|phrase)?|secret|token|api[_-]?key|access[_-]?key|private[_-]?key|credentials?|auth)[A-Za-z0-9_]*"
SECRETS = [r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?(?:-----END [A-Z ]*PRIVATE KEY-----|$)",
           r"\b(?:sk|pk|rk)-[A-Za-z0-9_-]{16,}", r"\bAKIA[0-9A-Z]{16}\b", r"\bgh[pousr]_[A-Za-z0-9]{20,}", r"\bgithub_pat_[A-Za-z0-9_]{20,}",
           r"\bxox[abpr]-[A-Za-z0-9-]{10,}", r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]+",
           r"(?i)(\b(?:bearer|basic)\s+)[A-Za-z0-9._~+/=-]{8,}",
           r"([a-z][a-z0-9+.-]*://[^:/\s@]+:)[^@\s/]+(?=@)",
           rf"(?i)({KEYWORD}[\"']?[ \t]*[:=][ \t]*\"|{KEYWORD}[\"']?[ \t]+\")[^\"\n]{{4,}}(?=\")",  # quoted, spaces allowed
           rf"(?i)({KEYWORD}[\"']?[ \t]*[:=][ \t]*'|{KEYWORD}[\"']?[ \t]+')[^'\n]{{4,}}(?=')",
           rf"(?i)({KEYWORD}[\"']?[ \t]*[:=][ \t]*)(?![\"'<\s])[^\n]+",  # unquoted: the rest of the line
           r"(?i)(\b(?:requirepass|masterauth|password|passwd)[ \t]+)(?![\"'<\s])[^\n]+",  # `requirepass x`, netrc: the rest of the line
           r"(?im)^([ \t]*\|?[ \t]*(?:api_?)?(?:key|token|secret|password)[ \t]*\|[ \t]*)[^|\n]+"]


def mask(text: str) -> str:
    """Credential-like strings -> <masked> (the label before them is kept). Newlines inside a masked span are kept, so
    line numbers stay the same as the unmasked text's."""
    for p in SECRETS:
        text = re.sub(p, lambda m: (m.group(1) if m.lastindex else "") + "<masked>" + "\n" * m.group(0)[len(m.group(1) if m.lastindex else ""):].count("\n"), text)
    return text


def label(tool: str, args: dict) -> str:
    """A plain type label from the call (the call itself is never shown)."""
    if tool == "text_editor":
        return {"view": "file view", "create": "file create", "str_replace": "file edit", "insert": "file insert",
                "undo_edit": "edit undo"}.get(args.get("command"), "file tool")
    cmd = (args.get("cmd") or args.get("command") or "") if isinstance(args, dict) else ""
    cmd = cmd.strip()
    if cmd.startswith(("cd ", "cd\t")) and "&&" in cmd:
        return label(tool, {"cmd": cmd.split("&&", 1)[1]})
    first = re.split(r"[\s;&|]+", cmd.lstrip("(").removeprefix("sudo "), maxsplit=1)[0].rsplit("/", 1)[-1]
    groups = [("directory listing", {"ls", "find", "tree", "du", "stat"}), ("file read", {"cat", "head", "tail", "less", "wc", "nl"}),
              ("text search", {"grep", "rg", "awk", "sed"}), ("package install", {"apt", "apt-get", "pip", "pip3", "npm", "yarn", "uv", "cargo", "gem", "go"}),
              ("web request", {"curl", "wget"}), ("database query", {"psql", "sqlite3", "mysql", "redis-cli", "mongosh"}),
              ("test run", {"pytest", "jest", "make"}), ("script run", {"python", "python3", "node", "bash", "sh", "ruby", "Rscript", "php", "java", "dotnet"}),
              ("process or service check", {"ps", "systemctl", "service", "kill", "pkill", "top", "netstat", "ss", "lsof"}),
              ("file change", {"cp", "mv", "rm", "mkdir", "chmod", "chown", "touch", "tee", "echo", "printf"}), ("version control", {"git"})]
    return next((g for g, s in groups if first in s), "shell command")


def oid(arm, key, step, action) -> str:
    return hashlib.sha256(f"{arm}|{key}|{step}|{action}".encode()).hexdigest()[:10]


def excerpt(text: str, start: int = 0) -> str:
    """Up to LINES lines from line `start`, each cut to WIDTH characters; control characters (C0 but newline and tab,
    DEL, C1) shown as escapes (`\\r`, `\\x1b`, `\\x93`); the end of the output says whether it ends in a newline."""
    text = re.sub(r"[\x00-\x08\x0b-\x1f\x7f-\x9f]", lambda m: "\\r" if m.group() == "\r" else f"\\x{ord(m.group()):02x}", text)
    lines = text.split("\n")
    end = "[end of output; ends with a newline]" if text.endswith("\n") else "[end of output; no trailing newline]"
    if text.endswith("\n"):
        lines = lines[:-1]
    body = [l if len(l) <= WIDTH else l[:WIDTH] + " [line cut]" for l in lines[start:start + LINES]]
    head = [f"[... {start} lines above]"] if start else []
    tail = [f"[... {len(lines) - start - LINES} more lines]"] if len(lines) > start + LINES else [end]
    return "\n".join(head + body + tail).replace("```", "` ` `")


def load_calls(run_dir: Path, arms: dict) -> dict:
    """{(arm, key, step, action): (text, type label, rule-type v2, call)} for the arms named."""
    out = {}
    for f in sorted((run_dir / "inputs").glob("shard_*.jsonl")):
        for r in map(json.loads, f.read_text().splitlines()):
            if r["arm"] not in arms:
                continue
            for c in fixes.calls_of(r["messages"]):
                t = r["messages"][c["msg"]]["content"]
                t = t if isinstance(t, str) else ""
                out[(arms[r["arm"]], r["key"], c["step"], c["action"])] = (t, label(c["tool"], c["args"]), rule_type(c["tool"], c["args"], t, True), c)
    return out


def cell_of(key: str) -> str:
    return ":".join(key.rsplit("__", 2)[-2].split(":")[:2])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scores")
    ap.add_argument("run_dir")
    ap.add_argument("--skip", default="")
    ap.add_argument("--readings", default="")
    ap.add_argument("--draft", default="", help="write the .md here only (no results file), for reading the draws")
    a = ap.parse_args()
    skip = {s for s in a.skip.split(",") if s}
    readings = json.loads(Path(a.readings).read_text()) if a.readings else {}
    rows = [r for r in json.loads(gzip.decompress(Path(a.scores).read_bytes()))["levels"]["level1"]["rows"] if r["environment"] in TEST]
    calls = load_calls(Path(a.run_dir), ARMS)
    by_fold = defaultdict(list)
    for r in rows:
        by_fold[r["fold"]].append(r["score"])
    srt = {f: np.sort(v) for f, v in by_fold.items()}
    runs = defaultdict(set)
    for r in rows:
        r["pctl"] = float(100 * np.searchsorted(srt[r["fold"]], r["score"], side="right") / len(srt[r["fold"]]))
        r["id"] = oid(r["arm"], r["key"], r["step"], r["action"])
        r["text"], r["label"], r["rule_type"], _ = calls[(r["arm"], r["key"], r["step"], r["action"])]
        runs[(r["arm"], cell_of(r["key"]))].add(r["key"])
    if any(len(v) != 1 for v in runs.values()):
        raise ValueError("a cell has more than one run in an arm")
    by_run = defaultdict(list)
    for r in rows:
        by_run[(r["arm"], cell_of(r["key"]))].append(r)

    def twin(r):
        other = "real-attack-e1" if r["arm"] == "sim-attack-fixed" else "sim-attack-fixed"
        cand = sorted(by_run.get((other, cell_of(r["key"])), []), key=lambda x: (x["step"], x["action"]))
        near = lambda xs: min(xs, key=lambda x: (abs(x["step"] - r["step"]), x["step"], x["action"]))
        typed = [x for x in cand if x["label"] == r["label"]]
        same = [x for x in cand if (x["step"], x["action"]) == (r["step"], r["action"])]
        if same and same[0]["label"] == r["label"]:
            return same[0], "same step, same type"
        if typed:
            return near(typed), "closest same-type output in the twin run"
        if same:
            return same[0], "same step, different type (no same-type output in the twin run)"
        return (near(cand), "closest output, different type (no same-type output in the twin run)") if cand else (None, "no twin run")

    def block(r, side):
        return (f"**{side}**: run `{r['key']}`, step {r['step']}.{r['action']}; type: {r['label']}; score {r['score']:.3f} "
                f"(fold percentile {r['pctl']:.1f})\n\n```text\n{excerpt(mask(scrub(r['text'], 1)))}\n```\n")

    md = ["# Outputs at extreme-scoring end-of-output readouts: examples\n\n",
          "Fixed sim vs real (epoch 1), times masked, the 5 test environments. Scores are probe step A's chosen probes, out "
          "of fold; higher means more simulated-looking; percentiles are within each held-out environment's fold. Only "
          "output types no fix rule touches are drawn (no editor insert or str_replace output, no dpkg output, no clock "
          "stamp). Text: the computer's output as the probe read it (times masked), credential-like strings replaced by "
          "`<masked>`, cut to 20 lines; the agent's commands are never shown. Readings under each pair are mine, not "
          f"measured. Seeds: `{SEED}sim`, `{SEED}real`. A score is read at an output's last content token, after the whole "
          "transcript before it, so it reflects that context as much as the output shown; these are outputs where the "
          "readout is extreme, not what causes it.\n"]
    meta = {"seeds": {}, "sections": {}}
    for side, arm, sel, n in (("sim", "sim-attack-fixed", lambda r: r["pctl"] >= 95, N_SIM),
                              ("real", "real-attack-e1", lambda r: r["pctl"] <= 5, N_REAL)):
        cand = sorted((r for r in rows if r["arm"] == arm and not r["rule_type"] and sel(r)), key=lambda r: r["id"])
        random.Random(SEED + side).shuffle(cand)
        picked, skipped = [], []
        for r in cand:
            if len(picked) == n:
                break
            (skipped if r["id"] in skip else picked).append(r)
        title = ("Most simulated-looking simulated outputs (top 5% of their fold)" if side == "sim"
                 else "Most real-looking real outputs (bottom 5% of their fold)")
        md.append(f"\n## {title}, each with its twin\n\n{len(cand)} candidates; {len(picked)} drawn in seeded order; {len(skipped)} "
                  "skipped because the output or its twin was about the attack itself.\n")
        if len(picked) < n:
            raise ValueError(f"{side}: only {len(picked)} of {n} draws after skips")
        sec = []
        for i, r in enumerate(picked, 1):
            t, how = twin(r)
            pid = f"{side}{i}"
            sides = ("simulated", "real") if side == "sim" else ("real", "simulated")
            md.append(f"\n### Pair {pid}: {r['environment']}, main task {r['main_task']} (twin: {how})\n\n" + block(r, sides[0]) + "\n"
                      + (block(t, sides[1]) if t else "_no twin_\n")
                      + f"\n_My reading, not measured:_ {readings.get(pid, 'TODO')}\n")
            sec.append({"pair": pid, "drawn": {k: r[k] for k in ("id", "arm", "key", "step", "action", "fold", "label")} | {"score": r["score"], "pctl": r["pctl"]},
                        "twin": ({k: t[k] for k in ("id", "arm", "key", "step", "action", "label")} | {"score": t["score"], "pctl": t["pctl"]}) if t else None,
                        "twin_rule": how})
        meta["seeds"][side] = SEED + side
        meta["sections"][side] = {"candidates": len(cand), "drawn": len(picked), "skipped": [r["id"] for r in skipped], "pairs": sec}
        print(side, "| candidates", len(cand), "| skipped", len(skipped), "| order:", " ".join(r["id"] for r in cand[: n + len(skipped) + 5]))
        for p in sec:
            print(" ", p["pair"], p["drawn"]["id"], p["drawn"]["label"], f"{p['drawn']['pctl']:.1f}", "| twin:", p["twin_rule"])
    text = "".join(md)
    if a.draft:
        Path(a.draft).write_text(text)
        return
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    (HERE / "data").mkdir(exist_ok=True)
    md_path = HERE / "data" / f"probe_examples_{stamp}.md"
    with open(md_path, "x") as f:
        f.write(text)
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=HERE).stdout.strip()
    meta |= {"scores": Path(a.scores).name, "scores_sha256": hashlib.sha256(Path(a.scores).read_bytes()).hexdigest(),
             "run": Path(a.run_dir).name, "git": git, "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             "md": f"data/{md_path.name}", "md_sha256": hashlib.sha256(text.encode()).hexdigest(), "readings": bool(readings)}
    with open(HERE / "results" / f"probe_examples_{stamp}.json", "x") as f:
        json.dump(meta, f, indent=1)
    print(md_path, meta["md_sha256"])


if __name__ == "__main__":
    main()
