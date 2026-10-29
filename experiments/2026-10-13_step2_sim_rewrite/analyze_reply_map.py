"""Step 2 pilot (POST HOC, requested by q2 and df on 10-13): which simulated replies equal the real ones, and a
numbers-only reply map for df's "changed replies only" primary. Modal CPU; reads only the Volume; writes no text.

Inputs on the Volume: the staging upload (runs.jsonl, calls.jsonl, sim_results.jsonl), the staged rewritten rows
(`2026-10-13_q2_step2_simrewrite_inputs_v1`) and the extraction's meta files.
Outputs (results/):
  - `reply_map_<stamp>.json`: per key, every tool reply in transcript order (submit excluded, as reader_messages) with
    k, step and action (the indices segment_end_positions gives its output_end: step = agent turn, action = output
    after it, checked against the extraction's meta positions), tool, changed (real sha256 != simulated sha256), both
    sha256s, and the first changed reply. No text.
  - `identical_breakdown_<stamp>.json`: counts of identical vs changed replies by tool and command, real reply length
    (empty, < 50, 50-500, > 500 characters), fixed-message class (empty; text_editor's create, edit and insert
    confirmations; other), whether the same command with the same real reply appears earlier in the run, whether the
    same real reply text appears earlier in the run, and per run.

  PYTHONPATH=src uv run --no-project --python 3.12 --with 'modal==1.6.1' modal run \\
      experiments/2026-10-13_step2_sim_rewrite/analyze_reply_map.py --upload upload_2026-10-13_175452
"""

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import modal

APP = "probe-shift_aswin-surya_sub-q2-step2_replymap_kill-2026-11-13"
ACTS = "probe-shift_aswin-surya_sub-la-activations"
WORK = "2026-10-13_q2_step2_simrewrite_work"
DST = "2026-10-13_q2_step2_simrewrite_inputs_v1"
RUN = "2026-10-13_q2_step2_simrewrite_qwen3-32b_l48-63_v1"
ARM = "real-honest-simrewrite"

app = modal.App(APP)
acts = modal.Volume.from_name(ACTS, version=2)
image = modal.Image.debian_slim(python_version="3.12")
h = lambda s: hashlib.sha256(s.encode()).hexdigest()  # noqa: E731


def filename(key: str) -> str:  # the extractor's file stems
    return key if len(key.encode()) <= 240 else f"{key[:150]}__{hashlib.sha1(key.encode()).hexdigest()[:12]}"


def length_bin(n: int) -> str:
    return "empty" if n == 0 else "<50" if n < 50 else "50-500" if n <= 500 else ">500"


def fixed_class(text: str) -> str:
    t = text.strip()
    if not t:
        return "empty"
    if t.startswith("File created successfully at"):
        return "editor create confirmation"
    if "has been edited" in t:
        return "editor edit confirmation"
    if t.startswith("Text inserted") or "inserted" in t[:80] and t.startswith("The file"):
        return "editor insert confirmation"
    return "other"


@app.function(image=image, volumes={"/acts": acts}, cpu=4, memory=32768, timeout=1800, retries=0)
def analyze(upload: str) -> dict:
    up = Path("/acts") / WORK / upload
    read = lambda p: [json.loads(l) for l in p.read_text().splitlines()]  # noqa: E731
    runs = {r["run"]: r for r in read(up / "runs.jsonl")}
    calls = defaultdict(dict)
    for c in read(up / "calls.jsonl"):
        calls[c["run"]][c["k"]] = c
    sim = {}
    for r in read(up / "sim_results.jsonl"):
        if r["status"] == "ok":
            sim[(r["run"], r["k"])] = r["result"]["reader_output"]
    staged = {it["key"]: it for it in read(Path("/acts") / DST / "inputs" / "shard_0.jsonl")}
    reply_map, cnt, per_run = {}, Counter(), {}
    for i, run in sorted(runs.items()):
        key, n = run["key"], run["n_actions"]
        msgs = staged[key]["messages"]
        step, action, sa = -1, 0, []
        for m in msgs:  # reader_messages' indexing, as segment_end_positions: step = agent turn, action = output after it
            if m["role"] == "assistant":
                step, action = step + 1, 0
            elif m["role"] == "tool":
                sa.append((step, action))
                action += 1
        assert len(sa) == n, f"run {i}: {len(sa)} tool messages vs {n} calls"
        meta = json.loads((Path("/acts") / RUN / "meta" / ARM / f"{filename(key)}.json").read_text())
        meta_sa = {(p["step"], p["action"]) for p in meta["positions"] if p["kind"] == "output_end"}
        assert meta_sa == set(sa), f"run {i}: output_end (step, action) differ from the extraction's meta"
        entries, seen_cmd_out, seen_out = [], set(), set()
        for k in range(n):
            c = calls[i][k]
            real, simo = c["real_output"], sim[(i, k)]
            rs, ss = h(real), h(simo)
            changed = rs != ss
            fn = c["step"]["function"]
            cmd = fn + (f":{c['step']['arguments'].get('command')}" if fn == "text_editor" else "")
            cmd_key = h(fn + json.dumps(c["step"]["arguments"], sort_keys=True))
            entries.append({"k": k, "step": sa[k][0], "action": sa[k][1], "tool": fn, "changed": changed,
                            "real_sha256": rs, "sim_sha256": ss})
            grp = "changed" if changed else "identical"
            cnt[(grp, "total", "")] += 1
            cnt[(grp, "command", cmd if fn in ("bash", "text_editor") else "other")] += 1
            cnt[(grp, "real length", length_bin(len(real)))] += 1
            cnt[(grp, "fixed message", fixed_class(real))] += 1
            cnt[(grp, "same command and real reply earlier in the run", str((cmd_key, rs) in seen_cmd_out))] += 1
            cnt[(grp, "same real reply text earlier in the run", str(rs in seen_out))] += 1
            seen_cmd_out.add((cmd_key, rs))
            seen_out.add(rs)
        first = next((e for e in entries if e["changed"]), None)
        reply_map[key] = {"run": i, "n_replies": n, "n_changed": sum(e["changed"] for e in entries),
                          "first_changed": None if first is None else {x: first[x] for x in ("k", "step", "action")},
                          "replies": entries}
        per_run[i] = {"key": key, "replies": n, "identical": n - reply_map[key]["n_changed"]}
    breakdown = defaultdict(lambda: defaultdict(dict))
    for (grp, dim, val), v in sorted(cnt.items()):
        breakdown[dim][val][grp] = v
    return {"reply_map": reply_map, "breakdown": {d: dict(v) for d, v in breakdown.items()}, "per_run": per_run}


@app.local_entrypoint()
def main(upload: str):
    import subprocess
    from datetime import datetime
    from zoneinfo import ZoneInfo

    out = analyze.remote(upload)
    here = Path(__file__).parent
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=here).stdout.strip()
    prov = {"label": "POST HOC, PRELIMINARY (numbers only, no text)", "upload": upload, "staged": DST, "extraction": RUN,
            "arm": ARM, "git": git, "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "app": APP}
    with open(here / "results" / f"reply_map_{stamp}.json", "x") as fh:
        fh.write(json.dumps(prov | {"keys": out["reply_map"]}, indent=1))
    with open(here / "results" / f"identical_breakdown_{stamp}.json", "x") as fh:
        fh.write(json.dumps(prov | {"breakdown": out["breakdown"], "per_run": out["per_run"]}, indent=1))
    print(json.dumps(out["breakdown"], indent=1))
