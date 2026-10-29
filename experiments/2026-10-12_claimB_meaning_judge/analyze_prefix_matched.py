"""Meaning-judge planning ($0, local, keys and numbers only; v2 counts each task once and saves site sizes; v3 hashes what the
agent saw; v4 adds the n437 honest simulated run (unrepaired September simulator) as a third arm; v5 requires the same turn
grouping through k and saves each site's run keys): where do two real honest runs reply to the same command at
the same prior state? Q1's convention picks e1 and e2 per main task (the first and second valid real-honest run by key;
`src/probe_shift/q1_eval.py`), within splits v4's 437 paired honest tasks. Calls are matched on a hash of function and
arguments, in order (submit excluded). Position k is "prefix-matched" if e1's and e2's calls 0..k are identical; it is
"strict" if, in addition, their results 0..k-1 are byte-identical (same hash). Also counted: whether the two replies at
k are byte-identical (a trivial agreement), and how many of the Oct-6 matched pairs' 100 honest sites (simulated reply
already written on the real history) sit at a strict position against a second real run. No command or output text is
printed or saved: hashes are compared in memory only.

  uv run --no-project --python 3.12 python experiments/2026-10-12_claimB_meaning_judge/analyze_prefix_matched.py
"""

import hashlib
import json
import subprocess
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).parent
REPO = HERE.parents[1]
DATA = Path.home() / "probe-shift-viewer/.run_cache/viewer/eval"
SIMDATA = Path.home() / ".cache/probe-shift/s3"  # the n437 honest simulated runs, cached from S3
DEV = {"auto_workflow", "file_indexing", "model_registry", "slack_clone", "web_scraping"}  # claim B practice
SUBMIT = "submit"
H = lambda x: hashlib.sha256(x.encode()).hexdigest()  # noqa: E731


def text(c):
    return c if isinstance(c, str) else "\n".join(x.get("text") or "" for x in c or [] if x.get("type") == "text")  # probe_shift.transcripts.visible_text


def seen(m):  # what the agent saw: the error message for an errored call (probe_shift.transcripts.tool_result_text, v3)
    return (m["error"].get("message") or text(m.get("content"))) if m.get("error") else text(m.get("content"))


def calls_of(sample):
    res = {m.get("tool_call_id"): (H(seen(m)), len(seen(m))) for m in sample["messages"] if m["role"] == "tool"}
    pre = 0  # characters the judge would read before each reply: system, user and agent text, call arguments, earlier outputs
    out = []
    for m in sample["messages"]:
        if m["role"] in ("system", "user") or (m["role"] == "assistant" and not m.get("tool_calls")):
            pre += len(text(m.get("content") or ""))
        if m["role"] != "assistant" or not m.get("tool_calls"):
            continue
        pre += len(text(m.get("content") or ""))
        pos = 0  # v5: the call's place among this turn's non-submit calls (same places through k = same turn grouping)
        for tc in m["tool_calls"]:
            if tc["function"] == SUBMIT:
                continue
            args = tc.get("arguments") or {}
            fam = f"text_editor:{args.get('command')}" if tc["function"] == "text_editor" else ("bash" if tc["function"].endswith("bash") else "other")
            r = res[tc.get("id")]  # a call without a result fails loudly
            pre += len(json.dumps(args))
            out.append({"cmd": H(tc["function"] + " " + json.dumps(args, sort_keys=True)), "res": r[0], "chars": r[1], "pre": pre,
                        "fam": fam, "single": len(m["tool_calls"]) == 1, "pos": pos})
            pos += 1
            pre += r[1]
    return out


def load_file(job):
    path, wanted = job
    got = {}
    with zipfile.ZipFile(path) as z:
        for n in z.namelist():
            if n.startswith("samples/") and n.endswith(".json"):
                x = json.loads(z.read(n))
                if (x["id"], x["epoch"]) in wanted:
                    got[wanted[(x["id"], x["epoch"])]] = calls_of(x)
    assert len(got) == len(wanted), f"{Path(path).name}: {len(got)} of {len(wanted)}"
    return got


def main():
    ex = json.loads((REPO / "splits/exclusions_v4.json").read_text())
    tasks = set(ex["paired_honest_cells"]["cells"])
    runs = defaultdict(list)
    for r in sorted((r for r in ex["samples"] if r["arm"] == "real-honest" and not r["reason"] and r["main_task"] in tasks), key=lambda r: r["key"]):
        runs[r["main_task"]].append(r)
    pairs = {t: v[:2] for t, v in runs.items() if len(v) >= 2}
    jobs = defaultdict(dict)
    for v in runs.values():
        for r in v:
            jobs[str(DATA / r["cohort"] / r["file"])][(r["sample_id"], r["epoch"])] = r["key"]
    sims = {r["main_task"]: r for r in ex["samples"] if r["arm"] == "sim-honest" and not r["reason"] and r["main_task"] in pairs}  # v4: n437, unrepaired
    for r in sims.values():
        jobs[str(SIMDATA / r["cohort"].removeprefix("s3://") / "logs" / r["file"])][(r["sample_id"], r["epoch"])] = r["key"]
    with ProcessPoolExecutor(14) as pool:
        calls = {k: v for part in pool.map(load_file, jobs.items()) for k, v in part.items()}
    env = {r["main_task"]: r["environment"] for v in runs.values() for r in v}
    c, site_rows = defaultdict(Counter), []
    for t, (r1, r2) in pairs.items():
        a, b = calls[r1["key"]], calls[r2["key"]]
        split = "practice" if env[t] in DEV else "test"
        strict_ok, n_prefix, n_strict, sites = True, 0, 0, []
        sm = calls.get(sims[t]["key"]) if t in sims else None
        sim_ok = sm is not None  # v4: the simulated run also issued the same commands, with byte-identical earlier outputs
        for k in range(min(len(a), len(b))):
            if a[k]["cmd"] != b[k]["cmd"]:
                break
            n_prefix += 1
            strict_ok = strict_ok and (k == 0 or a[k - 1]["res"] == b[k - 1]["res"]) and a[k]["pos"] == b[k]["pos"]
            sim_ok = sim_ok and len(sm) > k and sm[k]["cmd"] == a[k]["cmd"] and (k == 0 or sm[k - 1]["res"] == a[k - 1]["res"]) and sm[k]["pos"] == a[k]["pos"]
            fam = a[k]["fam"] if a[k]["fam"] in ("bash", "text_editor:view") else "other"
            if strict_ok:
                n_strict += 1
                sites.append({"task": t, "e1": r1["key"], "e2": r2["key"], "sim": sims[t]["key"] if t in sims else None, "split": split, "k": k, "family": fam, "prefix_chars": a[k]["pre"], "reply_chars": [a[k]["chars"], b[k]["chars"]],
                              "reply_identical": a[k]["res"] == b[k]["res"], "single_call_turns": a[k]["single"] and b[k]["single"],
                              "sim_matched": sim_ok, "sim_reply_identical_to_e1": sim_ok and sm[k]["res"] == a[k]["res"]})
            for g in (f"{split}|{fam}", f"{split}|any", "all|any"):
                c[g]["prefix_positions"] += 1
                c[g]["prefix_positions_k_ge_1"] += k >= 1
                if strict_ok:
                    c[g]["strict_positions"] += 1
                    c[g]["strict_k_ge_1"] += k >= 1
                    c[g]["strict_k_ge_1_single_call_turn"] += k >= 1 and a[k]["single"] and b[k]["single"]
                    c[g]["strict_k_ge_1_reply_identical"] += k >= 1 and a[k]["res"] == b[k]["res"]
        for g in (f"{split}|any", "all|any"):
            c[g]["task_pairs"] += 1
            c[g]["task_pairs_with_strict_k_ge_1"] += any(x["k"] >= 1 for x in sites)
            c[g]["task_pairs_with_any_strict"] += n_strict >= 1
            c[g]["prefix_len_sum"] += n_prefix
        site_rows.extend(sites)
    oct6 = json.loads(subprocess.run(["git", "show", "3820d9e5:experiments/2026-10-06_matched_step_pairs/results/pairs_2026-10-06_175844.json"],
                                     capture_output=True, text=True, check=True, cwd=REPO).stdout)
    o = Counter()
    for p in oct6["pairs"]:
        if p["arm"] != "honest":
            continue
        o["honest_sites"] += 1
        other = [r for r in runs.get(p["main_task"], []) if r["key"] != p["key"]]
        o["with_a_second_real_run"] += bool(other)
        if p["key"] not in calls or not other:
            continue
        a, k = calls[p["key"]], p["k"]
        if any(len(calls[r["key"]]) > k and all(a[j]["cmd"] == calls[r["key"]][j]["cmd"] for j in range(k + 1))
               and all(a[j]["res"] == calls[r["key"]][j]["res"] for j in range(k)) for r in other):
            o["site_strict_matched_by_a_second_real_run"] += 1
    out = {"label": "Meaning-judge planning: prefix-matched real honest positions (keys and numbers only)", "counts": {g: dict(v) for g, v in sorted(c.items())},
           "oct6_honest_sites": dict(o), "strict_sites": site_rows, "git": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=REPO).stdout.strip(),
           "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    with open(HERE / "results" / f"prefix_matched_{stamp}.json", "x") as fh:
        fh.write(json.dumps(out) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k != "strict_sites"}, indent=1))


if __name__ == "__main__":
    main()
