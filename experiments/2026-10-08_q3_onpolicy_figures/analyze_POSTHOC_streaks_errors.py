"""POST HOC, exploratory (D) and (E): are false alarms a general pattern of long, repetitive, error-driven runs, and does the
simulator have fewer such loops? $0, local; numbers only (no command or output text is printed or saved).

Per scored read-only call, from its transcript (honest runs, real attack runs 1 and 2, and the September simulated
attack runs; attack transcripts are read by this script only, and only numbers leave it):
- streak: the call sits in a maximal run of >= 5 consecutive tool calls with the same command shape (bash: the first five
  tokens, digits as N; text_editor: its command; other tools: the tool name);
- prev_error: an output of the previous completed assistant turn (the last turn before the call's own) was an error (the
  tool message's error field, or a nonzero exit code or an error marker in the output), i.e. an error the agent had seen;
- run length: the run's tool calls.
Calls are joined to transcripts through the scorer's own rows (analyze_honest_ab_audit.local_load) and each meta call's
(turn, call_index); the main rows must equal the main per-call meta, and the join must match the tool name and the
parameter name on at least 99.5% of calls overall and 99% within each world, or the script stops. Losses are reported. Flags use (b)'s
simulation-set thresholds at 1% and 5% (point estimate).

(D) Honest runs (and attack runs beside them): false-alarm rate by run-length tercile; the share of alarms inside streaks
    vs the share of calls inside streaks; the rate after an error output vs not.
(E) Matched tasks (read-only calls in both real attack run 1 and the simulated attack run): per world, the share of
    read-only calls inside streaks, the share preceded by an error, and how many tasks have a streak of >= 5.

Predictions, written before running:
(D) at least 40% of honest alarms fall inside same-shape streaks (vs about 20% of calls); calls right after an error
    output alarm about twice as often.
(E) simulated attack runs show fewer long same-shape streaks and fewer errors before read-only calls than real run 1 on
    the same tasks.

  uv run --no-project --python 3.14 --with numpy --with pyyaml python analyze_POSTHOC_streaks_errors.py \\
      <main per-call> <honest_scores.json> <acts meta root> <launch dir>...
"""

import argparse
import collections
import hashlib
import json
import re
import zipfile
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import yaml

from analyze_honest_ab_audit import local_load
from analyze_scaleup_plants import threshold

HERE = Path(__file__).parent
ERR = re.compile(r"(^|\n)\s*(Error|error:|Traceback|Execution halted|command not found|No such file|cannot open|exit code [1-9]|Exit code: [1-9])", re.I)


def shape(fn: str, args) -> str:
    if fn == "bash" or (isinstance(fn, str) and fn.endswith("_bash")):
        cmd = (args.get("cmd") or args.get("command") or "") if isinstance(args, dict) else ""
        return "bash:" + " ".join(re.sub(r"\d+", "N", t)[:24] for t in cmd.split()[:5])
    if fn == "text_editor":
        return "editor:" + str(args.get("command") if isinstance(args, dict) else "")
    return str(fn)


def transcript(path: Path) -> list:
    """[(turn, call_index, tool, shape, error, argument names)] in run order; turn = the assistant message's index."""
    with zipfile.ZipFile(path) as z:
        (n,) = [x for x in z.namelist() if x.startswith("samples/") and x.endswith(".json")]
        s = json.loads(z.read(n))
    out_of = {m.get("tool_call_id"): m for m in s["messages"] if m["role"] == "tool"}
    calls, turn = [], -1
    for m in s["messages"]:
        if m["role"] != "assistant":
            continue
        turn += 1
        for j, tc in enumerate(m.get("tool_calls") or []):
            args = tc.get("arguments") or {}
            args = json.loads(args) if isinstance(args, str) else args
            res = out_of.get(tc.get("id"), {})
            c = res.get("content")
            text = c if isinstance(c, str) else " ".join(x.get("text", "") for x in (c or []) if isinstance(x, dict))
            calls.append((turn, j, tc.get("function"), shape(tc.get("function"), args), bool(res.get("error")) or bool(ERR.search(text)),
                          set(args) if isinstance(args, dict) else set()))
    return calls


def features(calls: list) -> dict:
    """(turn, call_index) -> (in_streak, prev_error, tool); plus the run length and whether it has a streak."""
    streak = [False] * len(calls)
    i = 0
    while i < len(calls):
        j = i
        while j + 1 < len(calls) and calls[j + 1][3] == calls[i][3]:
            j += 1
        if j - i + 1 >= 5:
            for k in range(i, j + 1):
                streak[k] = True
        i = j + 1
    turn_err = collections.defaultdict(bool)
    for c in calls:
        turn_err[c[0]] |= c[4]
    turns = sorted(turn_err)
    prev_turn = {t: (turns[i - 1] if i else None) for i, t in enumerate(turns)}
    f = {(c[0], c[1]): (streak[k], prev_turn[c[0]] is not None and turn_err[prev_turn[c[0]]], c[2], c[5]) for k, c in enumerate(calls)}
    return {"f": f, "n": len(calls), "has_streak": any(streak)}


def main(main_path: str, honest_path: str, root: str, launches: list):
    root, M, H = Path(root), json.loads(Path(main_path).read_bytes()), json.loads(Path(honest_path).read_bytes())
    cfg = yaml.safe_load((HERE / "config" / "score_scaleup196.yaml").read_text())
    ext = yaml.safe_load((HERE / "config" / "score_scaleup196_simhon.yaml").read_text())
    folds = M["folds"]
    info = {}
    for x in cfg["extractions"]:
        for f in sorted((root / x / "meta").glob("*/*.json")):
            m = json.loads(f.read_text())
            for c in m["calls"]:
                if c["row"] is not None:
                    info[(x, m["chunk"], m["row_start"] + c["row"])] = (m["run"], c["turn"], c.get("call_index", c.get("call")), c["name"], c["param"])
    main_rows = local_load(cfg, folds, root)
    if [[r[0], r[1], r[5], r[6], r[8]] for r in main_rows] != [[m["world"], m["task"], m["y"], m["irid"], m["call"]] for m in M["meta"]]:
        raise SystemExit("main rows differ from the main per-call meta")
    ext_rows = [r for r in local_load(ext, folds, root) if (r[0] in ("honA", "honB") and r[5] == 0) or r[0] == "simH"]
    if [[r[0], r[1], folds[r[1]], r[5]] for r in ext_rows] != [[x["world"], x["task"], x["fold"], x["y"]] for x in H["rows"]]:
        raise SystemExit("honest rows differ from honest_scores.json")
    cal = M["draws"][0]["cells"]["real->real"]["per_call"]["calibration"]
    th = {}
    for fpr in (0.01, 0.05):
        th[fpr] = {}
        for k in range(5):
            s = np.array([sc for row, sc, mk in zip(cal["row"], cal["score"], cal["model_fold"]) if mk == k and M["meta"][row]["world"] == "sim" and M["meta"][row]["y"] == 0])
            th[fpr][k] = threshold(s, np.ones(len(s)), fpr=fpr)
    scored = []  # (world, task, score, fold, info key)
    cells = M["draws"][0]["cells"]
    for world, cell in (("real", "real->real"), ("run2", "real->run2"), ("sim", "real->sim")):
        p = cells[cell]["per_call"]
        for row, sc in zip(p["row"], p["score"]):
            r = main_rows[row]
            if r[0] == world and r[5] == 0:
                scored.append((world, r[1], sc, folds[r[1]], (r[2], r[3], r[4])))
    for r, x in zip(ext_rows, H["rows"]):
        if r[0] in ("honA", "honB"):
            scored.append((r[0], r[1], x["s"][x["fold"]], x["fold"], (r[2], r[3], r[4])))
    # transcripts, once per run
    runs = sorted({info[s[4]][0] for s in scored})
    feat, missing = {}, 0
    for run in runs:
        path = next((Path(d) / run for d in launches if (Path(d) / run).exists()), None)
        if path is None:
            missing += 1
            continue
        feat[run] = features(transcript(path))
    rows, mismatch, lost = [], 0, collections.Counter()
    for world, task, sc, fold, key in scored:
        run, turn, ci, name, param = info[key]
        if run not in feat:
            continue
        g = feat[run]["f"].get((turn, ci))
        if g is None or g[2] != name or (param and param not in g[3]):
            mismatch += 1
            lost[world] += 1
            continue
        rows.append({"world": world, "task": task, "flag1": sc > th[0.01][fold], "flag5": sc > th[0.05][fold],
                     "streak": g[0], "prev_error": g[1], "run": run, "run_len": feat[run]["n"]})
    per_world = collections.Counter(s[0] for s in scored)
    if mismatch > 0.005 * len(scored) or missing or any(lost[w] > 0.01 * n for w, n in per_world.items()):
        raise SystemExit(f"join check failed: {mismatch} mismatches of {len(scored)} ({dict(lost)}), {missing} runs without a transcript")

    def summary(sel):
        out = {"calls": len(sel), "share_calls_in_streak": float(np.mean([r["streak"] for r in sel]))}
        for fl in ("flag1", "flag5"):
            al = [r for r in sel if r[fl]]
            pe, npe = [r for r in sel if r["prev_error"]], [r for r in sel if not r["prev_error"]]
            out[fl] = {"rate": len(al) / len(sel), "alarms": len(al),
                       "share_alarms_in_streak": float(np.mean([r["streak"] for r in al])) if al else None,
                       "rate_after_error": sum(r[fl] for r in pe) / len(pe) if pe else None, "calls_after_error": len(pe),
                       "rate_after_no_error": sum(r[fl] for r in npe) / len(npe) if npe else None,
                       "rate_in_streak": (sum(r[fl] for r in sel if r["streak"]) / max(1, sum(r["streak"] for r in sel))),
                       "rate_outside_streak": (sum(r[fl] for r in sel if not r["streak"]) / max(1, sum(not r["streak"] for r in sel)))}
        return out

    D = {}
    for name, worlds in (("honest (A + B)", ("honA", "honB")), ("honest A", ("honA",)), ("honest B", ("honB",)), ("attack run 1", ("real",)), ("attack run 2", ("run2",))):
        sel = [r for r in rows if r["world"] in worlds]
        lens = sorted({(r["run"], r["run_len"]) for r in sel}, key=lambda x: x[1])
        cuts = np.percentile([n for _, n in lens], [100 / 3, 200 / 3])
        terc = {t: [r for r in sel if int(np.searchsorted(cuts, r["run_len"], side="right")) == t] for t in range(3)}
        D[name] = summary(sel) | {"run_length_cuts_calls": cuts.tolist(),
                                  "by_run_length_tercile": {t: {"calls": len(v), "rate1": sum(r["flag1"] for r in v) / len(v), "rate5": sum(r["flag5"] for r in v) / len(v)} for t, v in terc.items() if v}}
    matched = sorted({r["task"] for r in rows if r["world"] == "real"} & {r["task"] for r in rows if r["world"] == "sim"})
    E = {"matched_tasks": len(matched)}
    for world in ("real", "sim"):
        sel = [r for r in rows if r["world"] == world and r["task"] in matched]
        task_runs = {r["task"]: r["run"] for r in sel}
        E[world] = {"read_only_calls": len(sel), "share_in_streak": float(np.mean([r["streak"] for r in sel])),
                    "share_after_error": float(np.mean([r["prev_error"] for r in sel])),
                    "tasks_with_a_streak_of_5": sum(feat[run]["has_streak"] for run in set(task_runs.values())),
                    "median_run_length_calls": float(np.median([feat[run]["n"] for run in set(task_runs.values())]))}
    out = {"label": "POST HOC, exploratory (D) and (E); numbers only",
           "predictions (written before running)": {"D": "at least 40% of honest alarms inside same-shape streaks (vs about 20% of calls); calls after an error output alarm about twice as often",
                                                    "E": "simulated attack runs have fewer long same-shape streaks and fewer errors before read-only calls than real run 1 on the same tasks"},
           "inputs": {"main_per_call": hashlib.sha256(Path(main_path).read_bytes()).hexdigest(), "honest_scores": hashlib.sha256(Path(honest_path).read_bytes()).hexdigest()},
           "join": {"scored_calls": len(scored), "joined": len(rows), "mismatches": mismatch, "runs": len(feat),
                    "lost_by_world": {w: [lost[w], n] for w, n in per_world.items()}},
           "thresholds": {str(k): v for k, v in th.items()}, "D": D, "E": E}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    dst = HERE / "results" / f"streaks_errors_POSTHOC_{stamp}.json"
    with dst.open("x") as fh:
        fh.write(json.dumps(out, indent=1, default=float))
    print(json.dumps({k: out[k] for k in ("join", "E")}, indent=1, default=float))
    for name, v in D.items():
        print(name, {k: (round(x, 4) if isinstance(x, float) else x) for k, x in v.items() if k in ("calls", "share_calls_in_streak")},
              {fl: {k: (round(x, 4) if isinstance(x, float) else x) for k, x in v[fl].items()} for fl in ("flag1", "flag5")},
              {t: {k: round(x, 4) if isinstance(x, float) else x for k, x in d.items()} for t, d in v["by_run_length_tercile"].items()})
    print("->", dst)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("main_path")
    ap.add_argument("honest_path")
    ap.add_argument("root")
    ap.add_argument("launches", nargs="+")
    a = ap.parse_args()
    main(a.main_path, a.honest_path, a.root, a.launches)
