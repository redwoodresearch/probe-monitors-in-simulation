"""The most simulated-looking fixed-sim outputs, for Aswin to read (exploratory, gathered at his explicit request at about
21:27 PT 10-10, relayed by 0d; his 10-09 stop on formatting checks is lifted for this). Free: local CPU, no fits.

Probe step A's saved per-output scores (fixed sim vs real e1, level 1 = times masked, the 5 test environments; scores are
comparable only within a fold, so each output is ranked by its fold percentile):
  pool    the 20 fixed-sim outputs with the highest fold percentiles (all output types);
  draws   the pool sorted by id and shuffled with random.Random(SEED), taken in order: 5 outputs, of which at least 2
          come before their run's first rule-type output (analyze_review_checks.rule_type v2: text_editor insert or
          str_replace, a dpkg marker or a parseable clock stamp; the outputs of step A's prefix test). If the shuffle's
          first 5 hold fewer than 2 such outputs, the last non-prefix draws give way to the next prefix outputs in the
          shuffled pool; if the pool holds fewer than 2, the next prefix outputs below the pool (by percentile) are added
          and the file says so. An output in --skip (its content, or its twin's, is the attack itself rather than
          ordinary machine output; decided by reading) is skipped, counted, and the next one taken.
  twin    the real e1 run of the same cell (make_probe_examples.py's rule: same step and type, else the closest
          same-type output, ...); the file says which.
Shown per output: run key, environment, step, type label (from the call, which is never shown), score, fold percentile,
and up to 20 lines of the level-1 output text the probe read, with credential-like values masked whole (the v8 mask).
These are attack runs (claim B has no honest arm), so only the tools' and the simulator's replies are shown.
Writes data/top_sim_examples_<stamp>.md (the 5 pairs; transcript text: git-ignored, S3 only),
data/top_sim_top20_<stamp>.md (all 20 pairs, for reading), and results/top_sim_examples_<stamp>.json (no text: pool,
draws, twins, prefix flags, skips, the masked-credential leak counts and the .md files' sha256).

  PYTHONPATH=src uv run --no-project --with numpy --with scipy --with scikit-learn python experiments/2026-10-09_fixed_sim/make_top_sim_examples.py \\
      <claimB_output_scores_A_*.json.gz> <local 2026-10-09_la_fixed-sim_v2> [--skip id,id] [--draft]
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
import make_probe_examples as M  # noqa: E402  (mask, labels, excerpts and the twin rule, unchanged)

SEED, POOL, N, MIN_PREFIX = "top-sim-examples-2026-10-10", 20, 5, 2


def leaks(text: str) -> int:
    """Credential-like spans still present after masking, inside the output text only (the ``` blocks; headers carry run
    keys and task names, which are never masked): each SECRETS pattern's matches whose value is not <masked>."""
    n, inside = 0, False
    for line in text.split("\n"):
        if line.startswith("```"):
            inside = not inside
            continue
        if inside:
            for p in M.SECRETS:
                for m in re.finditer(p, line):
                    n += not m.group(0)[len(m.group(1)) if m.lastindex else 0:].startswith("<masked>")
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scores")
    ap.add_argument("run_dir")
    ap.add_argument("--skip", default="")
    ap.add_argument("--draft", action="store_true", help="write only the two .md files, for reading the draws")
    ap.add_argument("--fair", action="store_true", help="pool = the top 20 outputs whose real twin is the same step and type and, where both "
                    "outputs name a file, the same file; 80-line excerpts (the judge's fairer set; Aswin about 22:15 PT 10-10)")
    a = ap.parse_args()
    skip = {s for s in a.skip.split(",") if s}
    rows = [r for r in json.loads(gzip.decompress(Path(a.scores).read_bytes()))["levels"]["level1"]["rows"] if r["environment"] in M.TEST]
    calls = M.load_calls(Path(a.run_dir), M.ARMS)
    first_rule = {}
    for (arm, key, step, action), (_, _, rt, _) in calls.items():
        if rt and (arm, key) not in first_rule or rt and (step, action) < first_rule[(arm, key)]:
            first_rule[(arm, key)] = (step, action)
    by_fold = defaultdict(list)
    for r in rows:
        by_fold[r["fold"]].append(r["score"])
    srt = {f: np.sort(v) for f, v in by_fold.items()}
    by_run = defaultdict(list)
    for r in rows:
        r["pctl"] = float(100 * np.searchsorted(srt[r["fold"]], r["score"], side="right") / len(srt[r["fold"]]))
        r["id"] = M.oid(r["arm"], r["key"], r["step"], r["action"])
        r["text"], r["label"], r["rule_type"], _ = calls[(r["arm"], r["key"], r["step"], r["action"])]
        fr = first_rule.get((r["arm"], r["key"]))
        r["prefix"] = fr is None or (r["step"], r["action"]) < fr
        by_run[(r["arm"], M.cell_of(r["key"]))].append(r)

    def twin(r):  # make_probe_examples.py's rule, for a fixed-sim output
        cand = sorted(by_run.get(("real-attack-e1", M.cell_of(r["key"])), []), key=lambda x: (x["step"], x["action"]))
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

    sim = sorted((r for r in rows if r["arm"] == "sim-attack-fixed"), key=lambda r: (-r["pctl"], r["id"]))
    if a.fair:
        M.LINES = 80

        def named(r):  # the file an output names in its first line, if any (views and creates)
            m = re.search(r"(?:running `cat -n` on|File created successfully at:|File already exists at:)\s*(\S+?)[:.]?(?:\s|$)", r["text"])
            return m.group(1) if m else None

        def fair(r):
            tw, how = twin(r)
            return tw is not None and how == "same step, same type" and named(r) == named(tw)

        sim = [r for r in sim if fair(r)]
    pool = sim[:POOL]
    order = sorted(pool, key=lambda r: r["id"])
    random.Random(SEED).shuffle(order)
    extended = []
    if sum(r["prefix"] for r in pool) < MIN_PREFIX:  # the pool is short of prefix outputs: the next ones by percentile
        extended = [r for r in sim[POOL:] if r["prefix"]][: MIN_PREFIX - sum(r["prefix"] for r in pool)]
        order += extended
    skipped = [r["id"] for r in order if r["id"] in skip]
    usable = [r for r in order if r["id"] not in skip]
    picked = usable[:N]
    if sum(r["prefix"] for r in picked) < MIN_PREFIX:
        extra = [r for r in usable[N:] if r["prefix"]][: MIN_PREFIX - sum(r["prefix"] for r in picked)]
        keep = [r for r in picked if r["prefix"]] + [r for r in picked if not r["prefix"]][: N - MIN_PREFIX]
        picked = sorted(keep + extra, key=usable.index)

    def block(r, side):
        return (f"**{side}**: run `{r['key']}` ({r['environment']}), step {r['step']}.{r['action']}; type: {r['label']}; "
                f"score {r['score']:.3f} (fold percentile {r['pctl']:.1f}){'; before any rewritten output' if r['prefix'] else ''}\n\n"
                f"```text\n{M.excerpt(M.mask(M.scrub(r['text'], 1)))}\n```\n")

    head = ("Step A's probe (Qwen3-32B, end-of-output readout, fixed sim vs real e1, times masked, 5 test environments) scores "
            "every output of every run; higher means more simulated-looking, and percentiles are within each held-out "
            "environment's fold. These are attack runs (claim B has no honest arm): only the tools' and the simulator's "
            "replies are shown, never the agent's commands. Credential-like values are masked whole. Exploratory, at "
            f"Aswin's request. Seed `{SEED}`; pool = the top {POOL} fixed-sim outputs by fold percentile.\n")
    pairs, md5, md20 = [], ["# The most simulated-looking fixed-sim outputs, each with its real twin\n\n", head], ["# All top-20 pairs (for reading)\n\n", head]
    for j, r in enumerate(sorted(pool + extended, key=lambda r: (-r["pctl"], r["id"]))):
        tw, how = twin(r)
        md20.append(f"\n## {j + 1}. `{r['id']}`\n\n" + block(r, "Simulated") + (block(tw, f"Real twin ({how})") if tw else "No twin.\n"))
    for j, r in enumerate(picked):
        tw, how = twin(r)
        md5.append(f"\n## Pair {j + 1}\n\n" + block(r, "Simulated") + (block(tw, f"Real twin ({how})") if tw else "No twin.\n"))
        pairs.append({"pair": j + 1, "drawn": {k: r[k] for k in ("id", "key", "environment", "step", "action", "label", "score", "pctl", "prefix")},
                      "twin": {k: tw[k] for k in ("id", "key", "step", "action", "label", "score", "pctl")} if tw else None, "twin_rule": how})
    texts = {"5": "".join(md5), "20": "".join(md20)}
    leak = {k: leaks(v) for k, v in texts.items()}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    tag = "fair_" if a.fair else ""
    paths = {"5": HERE / "data" / f"top_sim_examples_{tag}{stamp}.md", "20": HERE / "data" / f"top_sim_top20_{tag}{stamp}.md"}
    for k, p in paths.items():
        p.write_text(texts[k])
    for p in pairs:
        print(p["pair"], p["drawn"]["id"], p["drawn"]["label"], f"{p['drawn']['pctl']:.1f}", "prefix" if p["drawn"]["prefix"] else "", "| twin:", p["twin_rule"])
    print("pool prefix outputs:", sum(r["prefix"] for r in pool), "| extended:", len(extended), "| skipped:", skipped, "| leak counts:", leak)
    if a.draft:
        print("draft ->", paths)
        return
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    res = {"git": git, "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "seed": SEED, "fair": a.fair, "excerpt_lines": M.LINES,
           "pool": [{k: r[k] for k in ("id", "environment", "label", "pctl", "prefix")} for r in pool],
           "extended_below_pool": [r["id"] for r in extended], "skipped": skipped, "pairs": pairs, "credential_leak_counts": leak,
           "md_sha256": {k: hashlib.sha256(texts[k].encode()).hexdigest() for k in texts}, "md_files": {k: p.name for k, p in paths.items()}}
    with open(HERE / "results" / f"top_sim_examples_{tag}{stamp}.json", "x") as fh:
        fh.write(json.dumps(res, indent=1))
    print("->", paths, HERE / "results" / f"top_sim_examples_{stamp}.json")


if __name__ == "__main__":
    main()
