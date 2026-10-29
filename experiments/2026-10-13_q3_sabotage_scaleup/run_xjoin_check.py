"""Qwen3-32B cross-read label-join check (0d's GO, 10-13): keys only, no fit, no text.

One small Modal CPU container reads, read-only, the cross-read extraction metas on the q3xread acts Volume (main: attack,
sim_attack@september, honest_a, honest_b; run 2: attack) and the claim C cache's row keys (on_rows.L63.json on the q3xread score
Volume). It returns, per kept run (redrawn and rerun attempts skipped, as the cache does): run, task, epoch, arm and each call_end
entry's (turn, call_index, irid, has a stored row), plus a census of the call entries' field names and value types. Locally, the
frozen sabotage labels (three vote files, keys only) are joined on (run, epoch, turn, call_index) and on (world, task, turn,
call_index), and the counts per world and split are written to results/xjoin_check_<stamp>.json.

  uv run --no-project --python 3.12 --with "modal==1.6.1" modal run run_xjoin_check.py --inputs <scaled_inputs dir holding the three vote
      files, split_v1_21e748ba.json and the two ID lists>
"""

import csv
import json
import subprocess
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

APP = "probe-shift_aswin-surya_sub-q3xjoin_check_kill-2026-11-12"
ACTS = "probe-shift_aswin-surya_sub-q3xread_acts_kill-2026-11-13"
SCORE = "probe-shift_aswin-surya_sub-q3xread_score_kill-2026-11-13"
SOURCES = [("2026-10-13_xread_main_v1", "attack", "real"), ("2026-10-13_xread_run2_v1", "attack", "run2"),
           ("2026-10-13_xread_main_v1", "sim_attack@september", "sim"), ("2026-10-13_xread_main_v1", "honest_a", "honest_a"),
           ("2026-10-13_xread_main_v1", "honest_b", "honest_b")]
HERE = Path(__file__).parent

app = modal.App(APP)
acts = modal.Volume.from_name(ACTS, version=2)
score = modal.Volume.from_name(SCORE, version=2)
image = modal.Image.debian_slim(python_version="3.12")


@app.function(image=image, cpu=1, memory=4096, timeout=600, retries=0, volumes={"/acts": acts.read_only(), "/score": score.read_only()})
def keys_job() -> dict:
    runs, census = [], Counter()
    for x, arm, world in SOURCES:
        for f in sorted(Path(f"/acts/{x}/meta/{arm}").glob("*.json")):
            m = json.loads(f.read_text())
            if m.get("redrawn_because") or m.get("attempt_kind") == "rerun":
                continue
            calls = []
            for c in m["calls"]:
                census.update(f"{c.get('kind', 'command')}:{k}:{type(v).__name__}" for k, v in c.items())
                if c.get("kind") == "call_end":
                    calls.append([c.get("turn"), c.get("call_index"), c.get("irid"), c.get("row") is not None])
            runs.append({"world": world, "run": m["run"], "task": m["task"], "epoch": m["epoch"], "calls": calls})
    on = json.loads(Path("/score/xread_claimc_cache_v1/on_rows.L63.json").read_text())
    cache = Counter((r["world"], r["kind"]) for r in on)
    keys = Counter(tuple(sorted(r.keys())) for r in on)
    return {"runs": runs, "census": dict(census), "cache_rows": {f"{w}|{k}": n for (w, k), n in cache.items()}, "cache_row_keys": [list(k) for k in keys]}


@app.local_entrypoint()
def main(inputs: str):
    I = Path(inputs)
    new_votes, split = I / "scaleup_panel_votes_k116.csv", I / "split_v1_21e748ba.json"
    id_lists = f"{I / 'sim_calls_all_2026-10-13_120231.csv'},{I / 'real_run2_calls_all_2026-10-13_122622.csv'}"
    out = keys_job.remote()
    sp = json.loads(Path(split).read_text())["tasks"]
    task_of = {r["call_id"]: r["task"] for f in id_lists.split(",") for r in csv.DictReader(open(f))}
    votes = []
    for path, default in ((I / "sim_votes.csv", "sim"), (I / "real_votes.csv", None), (Path(new_votes), None)):
        for r in csv.DictReader(open(path)):
            votes.append({"world": r.get("world") or default, "task": r.get("task") or task_of[r["call_id"]], "run": r["run_id"], "epoch": int(r["epoch"]),
                          "turn": int(r["turn"]), "call_index": int(r["call_index"]), "label": r["label"], "sc": r["state_changing"] == "1"})
    by_run = {(x["world"], x["run"]): x for x in out["runs"]}
    k4 = {(x["run"], int(x["epoch"]), int(c[0]), int(c[1])): c[3] for x in out["runs"] for c in x["calls"] if c[0] is not None and c[1] is not None}
    kw = {(x["world"], x["task"], int(c[0]), int(c[1])): c[3] for x in out["runs"] for c in x["calls"] if c[0] is not None and c[1] is not None}
    side = lambda t: ("held_out_stage1" if sp[t]["stage1"] else "held_out_new") if sp.get(t, {}).get("side") == "held_out" else (  # noqa: E731
        "train_stage1" if sp.get(t, {}).get("stage1") else "other")
    cnt = Counter()
    for v in votes:
        if not v["sc"] or v["label"] not in ("sabotage", "not_sabotage"):
            continue
        g = f"{v['world']} / {side(v['task'])}"
        a = (v["run"], v["epoch"], v["turn"], v["call_index"])
        b = (v["world"], v["task"], v["turn"], v["call_index"])
        cnt[(g, "eligible_labelled")] += 1
        cnt[(g, "run_in_xread")] += (v["world"], v["run"]) in by_run
        cnt[(g, "join_by_run_key")] += a in k4
        cnt[(g, "join_by_run_key_with_row")] += bool(k4.get(a))
        cnt[(g, "join_by_world_task_key")] += b in kw
        cnt[(g, "join_by_world_task_key_with_row")] += bool(kw.get(b))
    res = {}
    for (g, k), n in sorted(cnt.items()):
        res.setdefault(g, {})[k] = n
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=HERE, capture_output=True, text=True).stdout.strip()
    rep = {"written": stamp, "git": git, "app": APP, "join_counts": res, "xread_runs_by_world": dict(Counter(x["world"] for x in out["runs"])),
           "xread_tasks_by_world": {w: len({x["task"] for x in out["runs"] if x["world"] == w}) for w in ("real", "run2", "sim", "honest_a", "honest_b")},
           "call_entry_field_census": out["census"], "cache_rows_L63": out["cache_rows"], "cache_row_keys": out["cache_row_keys"]}
    dst = HERE / "results" / f"xjoin_check_{stamp}.json"
    with dst.open("x") as fh:
        fh.write(json.dumps(rep, indent=1))
    print(json.dumps({k: rep[k] for k in ("join_counts", "xread_runs_by_world", "xread_tasks_by_world", "cache_rows_L63", "cache_row_keys")}, indent=1),
          "\n", json.dumps(out["census"]), "\n->", dst)
