"""How the long empty-reply streaks happen, in simulated and real runs (POST HOC, descriptive). $0, local, metadata only.

The orchestrator, for Aswin, 2026-10-13 (two messages). First: how the 4 simulated runs dropped in width_196 broke
(t118_clinical_trial, t66/t86/t96_file_indexing): per streak, the tool names, exactly-empty vs whitespace replies, error
and status fields on the tool and simulator calls, token counts, timestamps, whether the streak ends the run, the context
size when it starts, the same tasks' real runs, and how common streaks of 5+, 10+ and 20+ are in simulated and real runs.
Second: for every run with 20+ empty replies in a row, in every world and arm, its longest streak, whether its calls
during the streak were flagged at the frozen threshold (counts), and whether the run sits in a calibration fold. The
question is whether long empty streaks are simulator-only or also happen in real runs without being flagged.

Builds on aswin/q3-empty-streaks @ 9c5bcbca (result at 31020c54), which counted runs with 20+ / 50+ streaks from
extraction metadata (empty = null readout). This script reads the runs' own .eval files instead and cross-checks that list.

- **Runs:** every distinct run file in the ID sheet, per arm (one sample per file, checked).
- **Empty reply:** a tool message whose text (attachments resolved) is empty after stripping whitespace, as in
  width_196; exactly empty = 0 characters. **Streak:** consecutive empty tool messages in message order.
- **Long-streak runs:** longest streak >= 20. For each one, every streak of >= 20 is described by enum fields, counts,
  lengths, token counts and timestamps only. No command, reply, completion or message text is printed or stored.
- **Simulator calls** of a reply are the model events inside that tool call's span.
- **Flags:** draw 0 of the 196-task 2x2 (frozen 1% thresholds; flagged if score > the fold's threshold) for every cell
  that scores the run's world. A scored call joins to its tool reply through its request id (the agent model event's
  x-irid header), the sheet's call index and the tool-call id. **Calibration:** the run's rows that appear in a cell's
  stored calibration rows (read-only calls of the training world, fold (k+1) mod 5 for model k).
- **Gates:** input hashes; the 4 simulated runs and their longest streaks [23, 124, 128, 20] reproduce width_196; every
  scored call of a long-streak run maps to exactly one tool reply, and the sheet's tool name equals the call's function.

PREDICTION (committed before running): (1) simulated streak replies are >= 90% exactly empty, on bash calls, with no
tool errors, no simulator API errors and no truncation: the simulator answers with an empty output, not a failure.
(2) No context limit: the simulator's input tokens at streak start are below its run maximum in 3 of 4 runs.
(3) The two longest simulated streaks (124, 128) run to the end of the run. (4) Real runs also have 20+ streaks (the
earlier branch lists 5), so long streaks are not simulator-only; within them few calls are flagged (<= 5% of scored
calls), while the simulated streaks hold most of fold 3's 139 real->sim false alarms. (5) Runs with a streak of 5+:
about 25% of simulated runs and 15% of real attack runs; 10+: about 8% and 4%.

  uv run --no-project --python 3.14 python -I analyze_empty_streak_runs_POSTHOC.py --draw0 <json> --id-sheet <csv> \\
      --evals <bundle_src dir> --prior <empty_streaks_POSTHOC json> --out <json>
"""

import argparse
import csv
import hashlib
import json
import re
import statistics
import subprocess
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).parent
SHA = {"draw0": "c2656658baaf3aa150f83f4d105e7572bfd854f98032831621419e1f8e8f36bf",
       "id_sheet": "9d43c790e5dc7415d2e6e23f67673dba5e4946de11e7b61c7f1d253d3d9bf87b",
       "prior": "2fcda110a042f1f0240bd75158489f9702617c865d043b1f2b3f841a45824f16"}
LAUNCHES = ["2026-10-09_000523_rerun_strict", "2026-10-09_173815_scaleup_canary", "2026-10-09_182505_scaleup",
            "2026-10-10_001314_scaleup_topup", "2026-10-10_001314_second_attack"]
ARMS = {"real_attack_1": "real", "sim_attack": "sim", "real_attack_2": "run2", "real_honest_A": "honest_a",
        "real_honest_B": "honest_b", "sim_honest": "sim_honest"}
CELLS = {"real": ["real->real", "sim->real"], "sim": ["real->sim", "sim->sim"], "run2": ["real->run2"]}
WIDTH_DROPPED = {"t118_clinical_trial": 23, "t66_file_indexing": 124, "t86_file_indexing": 128, "t96_file_indexing": 20}
LONG, LEVELS = 20, (5, 10, 20, 50)
REF = re.compile(r"attachment://([0-9a-f]{32})")


def load(evals: Path, key: str) -> dict:
    hits = [evals / L / key for L in LAUNCHES if (evals / L / key).exists()]
    if len(hits) != 1:
        raise SystemExit(f"run file not found exactly once ({len(hits)})")
    z = zipfile.ZipFile(hits[0])
    names = [n for n in z.namelist() if n.startswith("samples/")]
    if len(names) != 1:
        raise SystemExit("a run file without exactly one sample")
    return {"sample": json.loads(z.read(names[0])), "header": json.loads(z.read("header.json"))}


def textlen(c, att) -> tuple[int, int]:
    """(raw length, stripped length) of a content value; the text itself is never returned."""
    def text(c):
        if isinstance(c, str):
            m = REF.fullmatch(c)
            return text(att[m.group(1)]) if m else c
        if isinstance(c, list):
            return "".join(text(p.get("text", "")) if isinstance(p, dict) else text(p) for p in c)
        return ""
    t = text(c)
    return len(t), len(t.strip())


def streaks(sample: dict) -> list[tuple[int, int]]:
    """(start, length) of every maximal run of empty tool replies, indexed among tool messages."""
    att, out, start = sample.get("attachments") or {}, [], None
    tools = [m for m in sample.get("messages") or [] if m.get("role") == "tool"]
    for i, m in enumerate(tools + [None]):
        empty = m is not None and textlen(m.get("content"), att)[1] == 0
        if empty and start is None:
            start = i
        if not empty and start is not None:
            out.append((start, i - start))
            start = None
    return out


def irids(sample: dict) -> set:
    return {((e.get("call") or {}).get("request") or {}).get("extra_headers", {}).get("x-irid")
            for e in sample.get("events") or [] if e.get("event") == "model"} - {None}


def summarize_file(args):
    evals, key = args
    s = load(Path(evals), key)["sample"]
    st = streaks(s)
    return key, {"longest": max((n for _, n in st), default=0), "empty": sum(n for _, n in st),
                 "replies": sum(m.get("role") == "tool" for m in s.get("messages") or []), "irids": sorted(irids(s))}


def describe(loaded: dict, flags_for) -> dict:
    """Metadata of every streak >= LONG in one run: enums, counts, lengths, tokens and times only."""
    s, h = loaded["sample"], loaded["header"]
    att, ev = s.get("attachments") or {}, s.get("events") or []
    tools = [m for m in s["messages"] if m.get("role") == "tool"]
    tev = {e["id"]: e for e in ev if e.get("event") == "tool"}
    spans = {e["id"]: e for e in ev if e.get("event") == "span_begin"}
    # a tool event's span_id is its parent; its own tool span is the nearest earlier unclaimed span_begin of type tool
    # with the same parent and name
    tool_span_of, claimed = {}, set()
    for i, e in enumerate(ev):
        if e.get("event") != "tool":
            continue
        for b in reversed(ev[max(0, i - 50):i]):
            if (b.get("event") == "span_begin" and b.get("type") == "tool" and b["id"] not in claimed
                    and b.get("parent_id") == e.get("span_id") and b.get("name") == e.get("function")):
                tool_span_of[e["id"]] = b["id"]
                claimed.add(b["id"])
                break
    by_span = defaultdict(list)
    for e in ev:
        if e.get("event") in ("model", "store"):
            by_span[e.get("span_id")].append(e)
    agent = [e for e in ev if e.get("event") == "model" and str(e.get("model", "")).startswith("vllm/")]
    call_owner = {}  # tool-call id -> index of the agent model event that made it
    for i, e in enumerate(agent):
        for tc in ((e.get("output") or {}).get("choices") or [{}])[0].get("message", {}).get("tool_calls") or []:
            call_owner[tc["id"]] = i
    sim_events = [e for e in ev if e.get("event") == "model" and not str(e.get("model", "")).startswith("vllm/")
                  and (spans.get(e.get("span_id")) or {}).get("type") == "tool"]
    sim_in = [((e.get("output") or {}).get("usage") or {}).get("input_tokens") for e in sim_events]
    agent_in = [((e.get("output") or {}).get("usage") or {}).get("input_tokens") for e in agent]
    store_vals = defaultdict(list)
    for e in ev:
        if e.get("event") == "store":
            for c in e.get("changes") or []:
                for name in ("EmptyCompletionCounter:streak", "StepLimitCounter:count"):
                    if name in str(c.get("path", "")) and isinstance(c.get("value"), (int, float)):
                        store_vals[name].append(c["value"])
    cfg = (h.get("eval") or {}).get("config") or {}
    lim = s.get("limit")
    run = {"replies": len(tools), "agent_model_events": len(agent), "simulator_model_events": len(sim_events),
           "limits_configured": {k: v for k, v in cfg.items() if "limit" in k},
           "sample_limit": ({"type": lim.get("type"), "limit": lim.get("limit")} if isinstance(lim, dict) else lim),
           "sample_error": s.get("error") is not None, "error_retries": len(s.get("error_retries") or []),
           "total_time_s": s.get("total_time"), "working_time_s": s.get("working_time"),
           "store_EmptyCompletionCounter_streak_max": max(store_vals["EmptyCompletionCounter:streak"], default=None),
           "store_StepLimitCounter_count_last": (store_vals["StepLimitCounter:count"] or [None])[-1],
           "agent_input_tokens_max": max([x for x in agent_in if x is not None], default=None),
           "simulator_input_tokens_max": max([x for x in sim_in if x is not None], default=None),
           "simulator_input_tokens_median": statistics.median([x for x in sim_in if x is not None]) if any(sim_in) else None,
           "last_agent_event": None, "tool_span_names": dict(Counter(sp.get("name") for sp in spans.values() if sp.get("type") == "tool"))}
    if agent:
        o = (agent[-1].get("output") or {})
        ch = (o.get("choices") or [{}])[0]
        run["last_agent_event"] = {"stop_reason": ch.get("stop_reason"), "tool_calls": len((ch.get("message") or {}).get("tool_calls") or []),
                                   "output_tokens": (o.get("usage") or {}).get("output_tokens")}
    run["context_window_fields"] = {k: v for k, v in json.loads(json.dumps(h.get("eval") or {})).items()
                                   if any(w in k for w in ("max_model_len", "context"))}
    reply_empty = {m.get("tool_call_id"): textlen(m.get("content"), att)[1] == 0 for m in tools}
    stop_len = {True: [], False: []}
    for tid, sp in tool_span_of.items():
        if tid in reply_empty:
            for e in by_span.get(sp, []):
                o = e.get("output") or {}
                if e.get("event") == "model" and (o.get("choices") or [{}])[0].get("stop_reason") == "stop":
                    stop_len[reply_empty[tid]].append(textlen(o.get("completion"), att)[0])
    q = lambda v: {"n": len(v), "min": min(v), "median": statistics.median(v), "max": max(v)} if v else None  # noqa: E731
    run["simulator_stop_completion_chars"] = {"empty_replies": q(stop_len[True]), "non_empty_replies": q(stop_len[False])}
    out = []
    for start, n in streaks(s):
        if n < LONG:
            continue
        seg = tools[start:start + n]
        R = {"start_reply": start, "length": n, "replies_after": len(tools) - (start + n), "functions": Counter(),
             "exactly_empty": 0, "whitespace_only": 0, "tool_event_found": 0, "tool_event_error_types": Counter(),
             "tool_event_truncated": 0, "tool_span_is_tool": 0, "simulator": {"calls": 0, "models": Counter(), "stop_reasons": Counter(),
             "statuses": Counter(), "incomplete_reasons": Counter(), "api_error_codes": Counter(), "event_errors": 0,
             "output_item_types": Counter(), "function_call_names": Counter(), "completion_length_zero": 0,
             "output_tokens": [], "reasoning_tokens": [], "input_tokens_first": None}, "working_time_s": [], "times": []}
        for m in seg:
            R["functions"][m.get("function")] += 1
            raw, _ = textlen(m.get("content"), att)
            R["exactly_empty" if raw == 0 else "whitespace_only"] += 1
            te = tev.get(m.get("tool_call_id"))
            if te is None:
                continue
            R["tool_event_found"] += 1
            err = te.get("error")
            R["tool_event_error_types"][err.get("type") if isinstance(err, dict) else ("none" if err is None else "other")] += 1
            R["tool_event_truncated"] += te.get("truncated") is not None
            R["working_time_s"].append(te.get("working_time"))
            R["times"].append(te.get("timestamp"))
            sp = tool_span_of.get(te["id"])
            R["tool_span_is_tool"] += sp is not None
            S = R["simulator"]
            for e in by_span.get(sp, []):
                if e.get("event") != "model":
                    continue
                o, call = e.get("output") or {}, e.get("call") or {}
                ch, u, resp = (o.get("choices") or [{}])[0], o.get("usage") or {}, call.get("response") or {}
                S["calls"] += 1
                S["models"][e.get("model")] += 1
                S["stop_reasons"][ch.get("stop_reason")] += 1
                S["statuses"][resp.get("status") if isinstance(resp, dict) else "n/a"] += 1
                inc = resp.get("incomplete_details") if isinstance(resp, dict) else None
                S["incomplete_reasons"][inc.get("reason") if isinstance(inc, dict) else None] += 1
                aerr = resp.get("error") if isinstance(resp, dict) else None
                S["api_error_codes"][aerr.get("code") if isinstance(aerr, dict) else aerr] += 1
                S["event_errors"] += e.get("error") is not None
                for it in (resp.get("output") or []) if isinstance(resp, dict) else []:
                    if isinstance(it, dict):
                        S["output_item_types"][it.get("type")] += 1
                        if it.get("type") == "function_call":
                            S["function_call_names"][it.get("name")] += 1
                S["completion_length_zero"] += textlen(o.get("completion"), att)[0] == 0
                S["output_tokens"].append(u.get("output_tokens"))
                S["reasoning_tokens"].append(u.get("reasoning_tokens"))
                if S["input_tokens_first"] is None:
                    S["input_tokens_first"] = u.get("input_tokens")
        first_owner = call_owner.get(seg[0].get("tool_call_id"))
        last_owner = call_owner.get(seg[-1].get("tool_call_id"))
        R["agent_input_tokens_at_start"] = ((agent[first_owner].get("output") or {}).get("usage") or {}).get("input_tokens") if first_owner is not None else None
        R["agent_model_events_after"] = (len(agent) - 1 - last_owner) if last_owner is not None else None
        ts = [datetime.fromisoformat(t) for t in R.pop("times") if t]
        R["duration_s"] = (max(ts) - min(ts)).total_seconds() if ts else None
        wt = [w for w in R.pop("working_time_s") if w is not None]
        R["tool_working_time_s"] = {"median": statistics.median(wt), "max": max(wt)} if wt else None
        S = R["simulator"]
        for k in ("output_tokens", "reasoning_tokens"):
            v = [x for x in S[k] if x is not None]
            S[k] = {"min": min(v), "median": statistics.median(v), "max": max(v)} if v else None
        R["flags"] = flags_for({m.get("tool_call_id") for m in seg})
        out.append(json.loads(json.dumps(R, default=dict)))
    return {"run": run, "streaks": out}


def main(draw0: str, id_sheet: str, evals: str, prior: str, out: str):
    g = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True, check=True, cwd=HERE).stdout  # noqa: E731
    if g("status", "--porcelain", "--untracked-files=no"):
        raise SystemExit("uncommitted tracked changes")
    git = g("rev-parse", "HEAD").strip()
    for name, p in (("draw0", draw0), ("id_sheet", id_sheet), ("prior", prior)):
        if hashlib.sha256(Path(p).read_bytes()).hexdigest() != SHA[name]:
            raise SystemExit(f"{name} sha256 mismatch")
    D = json.load(open(draw0))
    d0, meta, folds = D["draws"][0], D["meta"], D["folds"]
    sheet = list(csv.DictReader(open(id_sheet, newline="")))
    files = defaultdict(dict)  # arm -> file key -> task
    for r in sheet:
        files[r["arm"]][r["run_key"].rsplit("#e", 1)[0]] = r["task"]
    if any(len(set(v.values())) != len(v) for v in files.values()):
        print("note: some arm has several run files for one task")
    jobs = [(evals, k) for arm in ARMS for k in files[arm]]
    with ProcessPoolExecutor() as ex:
        per = dict(ex.map(summarize_file, jobs, chunksize=4))
    scored_irids = defaultdict(set)
    for m in meta:
        scored_irids[m["world"]].add(m["irid"])
    prevalence = {}
    for arm, w in ARMS.items():
        for subset in ("all", "scored"):
            ks = [k for k in files[arm] if subset == "all" or set(per[k]["irids"]) & scored_irids.get(w, set())]
            if subset == "scored" and w not in CELLS:
                continue
            L = [per[k]["longest"] for k in ks]
            prevalence[f"{arm}|{subset}"] = {"runs": len(ks), "any_empty": sum(per[k]["empty"] > 0 for k in ks)} | {
                f"longest>={n}": sum(x >= n for x in L) for n in LEVELS}
    # gate: the width_196 simulated runs
    sim_long = {files["sim_attack"][k]: per[k]["longest"] for k in files["sim_attack"] if per[k]["longest"] >= LONG}
    if sim_long != WIDTH_DROPPED:
        raise SystemExit("gate failed: the simulated runs with >= 20 streaks differ from width_196")
    # scored rows: (world, irid, call) -> meta row; per cell row -> (score, fold threshold); calibration rows per cell
    row_of = {(m["world"], m["irid"], int(m["call"])): i for i, m in enumerate(meta)}
    cell_score, cal_rows, thr = {}, {}, {}
    for c in (c for v in CELLS.values() for c in v):
        pc = d0["cells"][c]["per_call"]
        cell_score[c] = dict(zip(pc["row"], pc["score"]))
        tw = c.split("->")[0]  # the stored list holds all rows the fold model scored; calibration = training world, read-only
        cal_rows[c] = {r for r in pc["calibration"]["row"] if meta[r]["world"] == tw and meta[r]["y"] == 0}
        thr[c] = {f["fold"]: f["threshold"] for f in d0["cells"][c]["folds"] if f["threshold_kind"] == "frozen"}
    sheet_by_irid = defaultdict(list)
    for r in sheet:
        sheet_by_irid[r["irid"]].append(r)
    long_runs, prior_d = [], json.load(open(prior))
    for arm, w in ARMS.items():
        for k, task in sorted(files[arm].items()):
            if per[k]["longest"] < LONG:
                continue
            loaded = load(Path(evals), k)
            s = loaded["sample"]
            agent = [e for e in s["events"] if e.get("event") == "model" and str(e.get("model", "")).startswith("vllm/")]
            tc_of = {}  # meta row -> tool-call id
            if w in CELLS:
                by_irid = defaultdict(list)
                for e in agent:
                    by_irid[((e.get("call") or {}).get("request") or {}).get("extra_headers", {}).get("x-irid")].append(e)
                for (ww, irid, call), i in row_of.items():
                    if ww != w or meta[i]["task"] != task or irid not in by_irid:
                        continue
                    if len(by_irid[irid]) != 1:
                        raise SystemExit("gate failed: a request id on several agent model events")
                    sr = [r for r in sheet_by_irid[irid] if int(r["call"]) == call and r["arm"] == arm]
                    tcs = ((by_irid[irid][0].get("output") or {}).get("choices") or [{}])[0].get("message", {}).get("tool_calls") or []
                    if len(sr) != 1 or int(sr[0]["call_index"]) >= len(tcs) or tcs[int(sr[0]["call_index"])].get("function") != sr[0]["tool"]:
                        raise SystemExit("gate failed: a scored call doesn't map to exactly one tool call")
                    tc_of[i] = tcs[int(sr[0]["call_index"])]["id"]
            tool_ids = {m.get("tool_call_id") for m in s["messages"] if m.get("role") == "tool"}
            if any(t not in tool_ids for t in tc_of.values()):
                raise SystemExit("gate failed: a scored call without a tool reply")

            def flags_for(ids, tc_of=tc_of, w=w, task=task):
                if w not in CELLS:
                    return {"scored": "not scored by the 2x2 (no draw-0 rows for this arm)"}
                rows = [i for i, t in tc_of.items() if t in ids]
                f = {"scored_calls": len(rows), "impactful": sum(meta[i]["y"] == 1 for i in rows),
                     "read_only": sum(meta[i]["y"] == 0 for i in rows)}
                for c in CELLS[w]:
                    t = thr[c][folds[task]]
                    sc = [(meta[i]["y"], cell_score[c][i]) for i in rows if i in cell_score[c]]
                    f[c] = {"evaluated": len(sc), "flagged_read_only": sum(y == 0 and v > t for y, v in sc),
                            "flagged_impactful": sum(y == 1 and v > t for y, v in sc),
                            "in_calibration": sum(i in cal_rows[c] for i in rows)}
                return f
            d = describe(loaded, flags_for)
            all_rows = list(tc_of)
            d["run"] |= {"arm": arm, "world": w, "task": task, "longest": per[k]["longest"], "fold": folds.get(task),
                         "run_scored_rows": len(all_rows), "task_rows_in_draw0": sum(m["world"] == w and m["task"] == task for m in meta),
                         "run_calibration_rows": {c: sum(i in cal_rows[c] for i in all_rows) for c in CELLS.get(w, [])},
                         "run_flagged_read_only": {c: sum(meta[i]["y"] == 0 and cell_score[c].get(i, -1e9) > thr[c][folds[task]] for i in all_rows)
                                                   for c in CELLS.get(w, [])}}
            long_runs.append(d)
    same_task = {t: {arm: {kk: per[k][kk] for kk in ("longest", "empty", "replies")} for arm in ARMS for k, tt in files[arm].items() if tt == t}
                 for t in WIDTH_DROPPED}
    pw = {"real": "real", "sim": "sim", "run2": "run2", "honest_a": "honest_a", "honest_b": "honest_b", "sim_honest": "sim_honest"}
    prior_list = sorted((wd, x["task"], x["max_streak"]) for wd, v in prior_d["acting"]["worlds"].items() for x in v["runs_at_or_above_20"])
    mine_list = sorted((pw[d["run"]["world"]], d["run"]["task"], d["run"]["longest"]) for d in long_runs)
    report = {"label": "POST HOC, descriptive; metadata only (no text)", "git": git, "inputs_sha256": SHA,
              "builds_on": "aswin/q3-empty-streaks @ 9c5bcbca (result 31020c54)", "prevalence": prevalence,
              "long_streak_runs": long_runs, "same_task_runs_of_the_4_simulated": same_task,
              "cross_check_prior": {"prior": prior_list, "this": mine_list},
              "written": datetime.now(ZoneInfo("America/Los_Angeles")).isoformat(timespec="seconds")}
    Path(out).write_text(json.dumps(report, indent=1, default=dict))
    print(f"git {git[:8]} | runs per arm: " + ", ".join(f"{a} {len(files[a])}" for a in ARMS))
    for k, v in prevalence.items():
        print(f"{k:28s} " + " ".join(f"{kk} {vv}" for kk, vv in v.items()))
    for d in long_runs:
        r = d["run"]
        print(f"\n== {r['arm']} {r['task']} (fold {r['fold']}): longest {r['longest']}, replies {r['replies']}, limit {r['sample_limit']}, "
              f"time {r['total_time_s'] and round(r['total_time_s'])} s, last agent stop {r['last_agent_event']}, step counter {r['store_StepLimitCounter_count_last']}, "
              f"empty-completion counter max {r['store_EmptyCompletionCounter_streak_max']}")
        print(f"   agent input tokens max {r['agent_input_tokens_max']}; simulator input tokens median {r['simulator_input_tokens_median']} max {r['simulator_input_tokens_max']}; "
              f"simulator final-completion chars {r['simulator_stop_completion_chars']}; run scored rows {r['run_scored_rows']} of the task's {r['task_rows_in_draw0']} draw-0 rows, calibration rows {r['run_calibration_rows']}, flagged read-only {r['run_flagged_read_only']}")
        for x in d["streaks"]:
            S = x["simulator"]
            print(f"   streak at reply {x['start_reply']} length {x['length']}, replies after {x['replies_after']}, agent events after {x['agent_model_events_after']} | "
                  f"functions {x['functions']} | exactly empty {x['exactly_empty']}, whitespace {x['whitespace_only']} | tool errors {x['tool_event_error_types']}, "
                  f"truncated {x['tool_event_truncated']} | tool spans found {x['tool_span_is_tool']} | {x['duration_s']} s | agent input tokens at start {x['agent_input_tokens_at_start']}")
            print(f"     simulator: calls {S['calls']} models {S['models']} stop {S['stop_reasons']} status {S['statuses']} incomplete {S['incomplete_reasons']} "
                  f"api errors {S['api_error_codes']} event errors {S['event_errors']} items {S['output_item_types']} fn {S['function_call_names']} "
                  f"zero-length completion {S['completion_length_zero']} out tokens {S['output_tokens']} input tokens at start {S['input_tokens_first']}")
            print(f"     flags: {json.dumps(x['flags'])}")
    print("\nsame tasks (longest/empty/replies):", json.dumps(same_task))
    print("cross-check with aswin/q3-empty-streaks (world, task, longest):", "identical" if prior_list == mine_list else
          f"differs: only prior {sorted(set(prior_list) - set(mine_list))}; only this {sorted(set(mine_list) - set(prior_list))}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    for a in ("--draw0", "--id-sheet", "--evals", "--prior", "--out"):
        ap.add_argument(a, required=True)
    a = ap.parse_args()
    main(a.draw0, a.id_sheet, a.evals, a.prior, a.out)
