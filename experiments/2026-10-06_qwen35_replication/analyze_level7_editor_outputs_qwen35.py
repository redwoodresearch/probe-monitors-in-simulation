"""What still differs between real and simulated text_editor edit and create outputs at text level 7 (2026-10-07)?

About 89% of Qwen3.5's level-7 pre_call residual follows an edit or create output (README, "Where the level-7 pre_call
residual sits"). This compares those outputs in the level-7 staged text the readers read (the v4 scrub-7 inputs on the
shared Volume, streamed, nothing written there): per cell (main task x side task), the first str_replace output and
the first create output of the real attack run (e1) and of the simulated attack run, 200 cells of each kind sampled
with a fixed seed. Per output: length, line count, blank-line runs, indentation (tabs, spaces, the smallest indent),
leading and trailing whitespace, trailing newlines, lines with trailing spaces, a preamble-like first line, and for
create the message with paths and numbers masked. Rates and medians per arm, the paired difference, and the most
different features. Note: Qwen3.5's chat template trims each message, so whitespace at an output's start and end does
not reach it; whitespace inside the output does.

  uv run python experiments/2026-10-06_qwen35_replication/analyze_level7_editor_outputs_qwen35.py
"""

import json
import random
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal
import numpy as np

HERE = Path(__file__).parent
INPUTS = "probe-shift_aswin-surya_sub-la-activations"
RUN = "2026-10-06_la_segment-end-v4-scrub7_qwen3-32b_v1"  # level 7, five arms (the Qwen3.5 v4-scrub7 extraction read these)
ARMS = {"real-attack-e1": "real", "sim-attack": "sim"}
KINDS, N_PAIRS, SEED = ("str_replace", "create"), 200, 0
PREAMBLE = re.compile(r"^(the file|file |here'?s|successfully|created|edited|no replacement|error|warning)", re.I)


def outputs(it: dict) -> list[dict]:
    """The run's text_editor outputs, in order: (step, command, path, output text)."""
    out, step = [], -1
    msgs = it["messages"]
    for i, m in enumerate(msgs):
        if m["role"] != "assistant":
            continue
        step += 1
        calls = m.get("tool_calls") or []
        results = [x for x in msgs[i + 1:] if x["role"] == "tool"][:len(calls)]  # the k-th output answers the k-th call
        for c, r in zip(calls, results):
            a = c["function"].get("arguments") or {}
            a = json.loads(a) if isinstance(a, str) else a
            if c["function"]["name"] == "text_editor" and a.get("command") in KINDS:
                out.append({"step": step, "command": a["command"], "path": a.get("path"), "text": r.get("content") or ""})
    return out


def features(t: str) -> dict:
    lines = t.split("\n") if t else []  # an empty output has no lines (and so no blank-line run)
    if lines and lines[-1] == "" and len(lines) > 1:  # a terminating newline ends the last line; it is not a blank line
        lines = lines[:-1]
    nonblank = [l for l in lines if l.strip()]
    lead = [len(l) - len(l.lstrip(" ")) for l in nonblank if l.startswith(" ")]
    runs, cur = [], 0
    for l in lines:
        cur = cur + 1 if not l.strip() else 0
        runs.append(cur)
    trailing_nl = len(t) - len(t.rstrip("\n"))
    return {"chars": len(t), "lines": len(lines), "max blank-line run": max(runs, default=0),
            "has a run of 2+ blank lines": max(runs, default=0) >= 2,
            "any tab-indented line": any(l.startswith("\t") for l in nonblank),
            "any space-indented line": bool(lead), "smallest space indent": min(lead) if lead else 0,
            "smallest space indent is 1": bool(lead) and min(lead) == 1,
            "starts with whitespace": t[:1].isspace(), "ends with whitespace": t[-1:].isspace(),
            "trailing newlines": trailing_nl, "ends with a newline": trailing_nl > 0,
            "share of lines with trailing spaces": sum(l != l.rstrip(" \t") for l in lines) / max(len(lines), 1),
            "any line with trailing spaces": any(l != l.rstrip(" \t") for l in lines),
            "preamble-like first line": bool(PREAMBLE.match(t.lstrip())),
            "empty": not t.strip()}


def main():
    vol = modal.Volume.from_name(INPUTS, version=2)
    shards = [e.path for e in vol.listdir(f"{RUN}/inputs") if e.path.endswith(".jsonl")]

    def scan(path):
        found = {}
        for line in b"".join(vol.read_file(path)).decode().splitlines():
            it = json.loads(line)
            if it["arm"] in ARMS:
                found[(it["arm"], it["cell"], it["key"])] = outputs(it)
        return found

    runs = {}
    with ThreadPoolExecutor(16) as pool:
        for part in pool.map(scan, shards):
            runs |= part
    rng = random.Random(SEED)
    report, examples = {"run": RUN, "seed": SEED, "pairs_per_kind": N_PAIRS, "kinds": {}}, {}
    for kind in KINDS:
        first = {}  # (arm, cell) -> the first output of this kind (lowest key, then step)
        for (arm, cell, key), outs in sorted(runs.items()):
            o = next((o for o in outs if o["command"] == kind), None)
            if o is not None and (arm, cell) not in first:
                first[(arm, cell)] = o | {"key": key}
        cells = sorted(c for (a, c) in first if a == "real-attack-e1" and ("sim-attack", c) in first)
        pick = rng.sample(cells, min(N_PAIRS, len(cells)))
        feats = {d: [features(first[(arm, c)]["text"]) for c in pick] for arm, d in ARMS.items()}
        summary = {}
        for f in feats["real"][0]:
            r, s = np.array([x[f] for x in feats["real"]], float), np.array([x[f] for x in feats["sim"]], float)
            binary = set(np.unique(np.r_[r, s])) <= {0.0, 1.0}
            summary[f] = ({"real rate": round(r.mean(), 3), "sim rate": round(s.mean(), 3), "sim minus real": round(s.mean() - r.mean(), 3),
                           "pairs differing": int((r != s).sum())} if binary else
                          {"real median": round(float(np.median(r)), 3), "sim median": round(float(np.median(s)), 3),
                           "sim larger in pairs": round(float((s > r).mean()), 3), "sim smaller in pairs": round(float((s < r).mean()), 3)})
        k = {"cells with both arms": len(cells), "pairs": len(pick), "features": summary}
        if kind == "create":  # the create message, with paths and numbers masked
            mask = lambda t: re.sub(r"\d+", "<d>", re.sub(r"\S*/\S*|(?<=at: )\S+", "<path>", t.strip().split("\n")[0]))[:120]
            # (every path, absolute or relative, and a bare file name after "at: ")
            k["first line (masked), top 5"] = {d: Counter(mask(first[(arm, c)]["text"]) for c in pick).most_common(5) for arm, d in ARMS.items()}
        examples[kind] = [{"cell": c, "real": first[("real-attack-e1", c)]["text"][:300], "sim": first[("sim-attack", c)]["text"][:300]}
                          for c in pick[:5]]  # transcript text: kept out of git (the gitignored .run_cache), not in results/
        report["kinds"][kind] = k
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    path = HERE / "results" / f"level7_editor_outputs_qwen35_{stamp}.json"
    path.write_text(json.dumps(report, indent=1))  # aggregate features only
    ex = HERE.parent.parent / ".run_cache" / "level7_editor_examples" / path.name  # a few raw outputs, for reading locally
    ex.parent.mkdir(parents=True, exist_ok=True)
    ex.write_text(json.dumps(examples, indent=1))
    for kind, k in report["kinds"].items():
        print(f"== {kind}: {k['pairs']} pairs of {k['cells with both arms']} cells")
        for f, v in k["features"].items():
            print(f"   {f:38} {v}")
        if "first line (masked), top 5" in k:
            for d, top in k["first line (masked), top 5"].items():
                print(f"   create first lines, {d}: {top}")
    print(f"-> {path}")


if __name__ == "__main__":
    main()
