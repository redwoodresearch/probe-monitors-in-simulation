"""Figure 1 pairs, addendum 1 (PLAN.md here, committed before this ran). Local, $0, honest runs only. Over the selection's
tier-2 pools (130 str_replace, 9 insert pairs), counts which editor-format slips each real/simulated reply pair differs in,
with the repo's existing definitions; then, if any str_replace pair's gutter differs, draws one more pair from those by a
fresh default_rng(20261021). Raw text goes only to data/figure1_pairs/ (git-ignored); results/ gets counts, IDs and hashes.

  uv run --no-project --python 3.14 --with numpy python experiments/2026-10-12_q2_figure1_pairs/count_figure1_slips.py
"""
import hashlib
import json
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "2026-10-09_fixed_sim"))
import fixes  # noqa: E402  (imports the rendering swap's rerender)
import select_figure1_pairs as sel  # noqa: E402  (the selection's own loader, which asserts honest runs only)
from rerender import DIFF, PREAMBLE, REVIEW  # noqa: E402

SELECTION = HERE / "results/figure1_selection_2026-10-12_150044.json"


def gutter(r):
    if DIFF not in r[:300]:
        return "not a diff"
    return fixes.numbered_form(r.partition("+++ edited_version\n")[2]) or "plain"


def note(r):
    return r.rstrip().endswith(REVIEW)


def form(r):
    return "diff" if DIFF in r[:300] else "cat -n snippet" if PREAMBLE in r[:400] else "other"


def main():
    for f in (Path(__file__).resolve(), HERE / "PLAN.md"):
        if subprocess.run(["git", "ls-files", "--error-unmatch", str(f)], cwd=HERE, capture_output=True).returncode \
                or subprocess.run(["git", "diff", "--quiet", "HEAD", "--", str(f)], cwd=HERE).returncode:
            raise SystemExit(f"{f.name} differs from HEAD: commit it before running")
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip()
    sel_rec = json.loads(SELECTION.read_text())
    ex_p = sel.REPO / "splits/exclusions_v4.json"
    ex = json.loads(ex_p.read_text()); tasks = set(ex["paired_honest_cells"]["cells"])
    assert hashlib.sha256(ex_p.read_bytes()).hexdigest() == sel_rec["splits sha256"], "not the selection's splits"
    runs = defaultdict(list)
    for r in sorted((r for r in ex["samples"] if r["arm"] == "real-honest" and not r["reason"] and r["main_task"] in tasks), key=lambda r: r["key"]):
        runs[r["main_task"]].append(r)
    e1 = {t: v[0] for t, v in runs.items()}
    sims = {r["main_task"]: r for r in ex["samples"] if r["arm"] == "sim-honest" and not r["reason"] and r["main_task"] in e1}
    calls = {t: (sel.load(e1[t]), sel.load(sims[t])) for t in sorted(sims)}
    pools = {}
    for command in sel.COMMANDS:  # tier 2 exactly as the selection builds it
        pool = []
        for t, (a, b) in calls.items():
            cand = [(i, j) for i, x in enumerate(a) if sel.ok(x, command) for j, y in enumerate(b) if sel.ok(y, command) and x["cmd"] == y["cmd"] and x["cmd"] is not None]
            if cand:
                i, j = min(cand); pool.append((t, i, j))
        pools[command] = sorted(pool, key=lambda p: (p[0], e1[p[0]]["key"], p[1], p[2]))
        assert len(pools[command]) == sel_rec["eligible pairs by command and tier"][command]["2"], (command, len(pools[command]))
        first = sel_rec["pairs"][command]
        t, i, j = pools[command][first["draw index"]]
        assert (t, e1[t]["key"], sims[t]["key"], i, j) == (first["main task"], first["real"]["key"], first["simulated"]["key"], first["real"]["position k"], first["simulated"]["position k"]), command
    rep = lambda t, i, j: (calls[t][0][i]["reply"], calls[t][1][j]["reply"])  # noqa: E731
    # str_replace
    sr, gt, diffs = Counter(), Counter(), []
    for t, i, j in pools["str_replace"]:
        a, b = rep(t, i, j)
        ga, gb = gutter(a), gutter(b); gd, nd = ga != gb, a.endswith("\n") != b.endswith("\n")
        gt[f"{ga} | {gb}"] += 1
        sr["identical" if a == b else "both" if gd and nd else "gutter only" if gd else "newline only" if nd else "neither, other text differs"] += 1
        if gd:
            diffs.append((t, i, j))
    # insert
    ins, tabs = Counter(), {c: Counter() for c in ("review note", "form", "trailing newline")}
    for t, i, j in pools["insert"]:
        a, b = rep(t, i, j)
        ins["identical"] += a == b
        for c, f in (("review note", note), ("form", form), ("trailing newline", lambda r: r.endswith("\n"))):
            tabs[c][f"{f(a)} | {f(b)}"] += 1; ins[f"{c} differs"] += f(a) != f(b)
    rec = {"plan": "PLAN.md addendum 1 (this folder)", "git": head, "selection record": SELECTION.name, "pairs": {"str_replace": len(pools["str_replace"]), "insert": len(pools["insert"])},
           "str_replace: pair categories": dict(sr), "str_replace: gutter label (real | simulated)": dict(gt),
           "insert: pairs identical, and pairs differing per component": dict(ins), "insert: (real | simulated) per component": {c: dict(v) for c, v in tabs.items()},
           "str_replace pairs whose gutter differs": len(diffs), "extra draw": None}
    pick = None
    if diffs:
        idx = int(np.random.default_rng(sel.SEED).integers(len(diffs))); t, i, j = diffs[idx]; a, b = calls[t][0][i], calls[t][1][j]
        side = lambda run, c, pos: {"arm": run["arm"], "cohort": run["cohort"], "file": run["file"], "sample_id": run["sample_id"], "epoch": run["epoch"], "key": run["key"],  # noqa: E731
                                    "position k": pos, "agent turn": c["agent_turn"], "assistant message index": c["assistant_message_index"], "tool message index": c["tool_message_index"],
                                    "reply bytes": len(c["reply"].encode()), "reply sha256": hashlib.sha256(c["reply"].encode()).hexdigest(),
                                    "gutter": gutter(c["reply"]), "trailing newline": c["reply"].endswith("\n")}
        meta = {"pool": "str_replace tier-2 pairs whose gutter differs", "pool size": len(diffs), "draw index": idx, "main task": t, "environment": e1[t]["environment"],
                "real": side(e1[t], a, i), "simulated": side(sims[t], b, j)}
        rec["extra draw"] = meta
        pick = meta | {"real command": {"function": a["function"], "arguments": a["arguments"]}, "simulated command": {"function": b["function"], "arguments": b["arguments"]},
                       "real reply": a["reply"], "simulated reply": b["reply"], "real reply repr": repr(a["reply"]), "simulated reply repr": repr(b["reply"])}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    if pick:
        out = sel.REPO / "data" / "figure1_pairs"; out.mkdir(parents=True, exist_ok=True)
        with open(dp := out / f"figure1_gutter_pair_{stamp}.json", "x") as fh:
            fh.write(json.dumps({"counts record": f"results/figure1_slip_counts_{stamp}.json", "pair": pick}, indent=1, ensure_ascii=False) + "\n")
        rec["data file"] = {"path": str(dp.relative_to(sel.REPO)), "sha256": hashlib.sha256(dp.read_bytes()).hexdigest()}
    with open(rp := HERE / "results" / f"figure1_slip_counts_{stamp}.json", "x") as fh:
        fh.write(json.dumps(rec, indent=1) + "\n")
    print(json.dumps(rec, indent=1)); print(rp)


if __name__ == "__main__":
    main()
