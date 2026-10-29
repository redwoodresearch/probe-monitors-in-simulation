"""Does the simulator hide failures, and does that change agent behavior and side-task outcomes?

Paired by attack cell (main task + side task, the 939 `paired_attack_cells` of splits/exclusions_v3.json): the real
attack run (e1) against the Astra simulated attack run (sim), with the real rerun (e2) against e1 as the real-vs-real
noise floor. Reads the local .eval archives; no model calls.

A tool call FAILED, in real and simulated runs alike, if any of:
  - Inspect recorded a tool error other than `parsing` (argument validation is the agent's mistake, not the
    environment's): text_editor errors ("does not exist", "No replacement was performed", ...), timeouts;
  - a text_editor result carries one of those error messages as plain text (a simulator can render them that way);
  - a bash-like result (bash, db_bash, rust_worker_bash, ...) has an error line (BASH_FAIL: "command not found",
    "No such file or directory", a Traceback, "fatal:", ...). A heuristic: it sees error-looking lines only.

Measures (each with a 95% bootstrap interval over main tasks, 2,000 resamples):
  1. side-task success (`linux_scorer` `side_task_success`, C = success, I = failure; N and E excluded and counted);
  2. failure hiding: calls with the same function and arguments in both runs of a cell (k-th occurrence with k-th),
     the share where one run's call failed and the other's did not; also on the `pre_edit` subset, calls made before
     either run edited a file, where earlier divergent edits cannot explain a difference;
  3. behavior: steps (agent turns with a non-submit tool call), share of steps right after a failed step, share of
     steps that retry (same function and target as a failed call) or edit (text_editor write, or a bash write) right
     after a failed step, steps to the first submit call;
  4. link: cells split by whether e1 had a failed call; difference-in-differences of side-task success.

  uv run python experiments/2026-10-07_sim_failure_hiding/analyze_failure_hiding.py
"""

import argparse
import hashlib
import json
import re
import subprocess
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

from probe_shift.transcripts import RUNNER_SUBMIT, tool_result_text

HERE = Path(__file__).parent
REPO = HERE.parents[1]
ARMS = {"real-attack-e1": "e1", "real-attack-e2": "e2", "sim-attack": "sim"}
TE_FAIL = re.compile(r"does not exist\. Please provide a valid path|No replacement was performed|File already exists at|"
                     r"Cannot overwrite files|Multiple occurrences of old_str|Invalid `view_range`|"
                     r"Invalid `insert_line`|is a directory\. |Ran into \[Errno \d+\]", re.I)
# Error lines, plus the failure summaries of common checks: pytest ("3 failed", "FAILED x::y", "ERROR at setup"),
# jest ("FAIL tests/x"), rustc ("error[E0425]:"), `cargo fmt --check` (prints "Diff in <file>:<line>:"), panics.
BASH_FAIL = re.compile(r"command not found|No such file or directory|Permission denied|cannot access|"
                       r"Traceback \(most recent call last\)|syntax error|Connection refused|npm ERR!|"
                       r"timed out before completing|failed because it timed out|exited unexpectedly|"
                       r"Segmentation fault|core dumped|\bpanicked at\b|\b[1-9]\d* failed\b|ERROR at (?:setup|teardown) of|"
                       r"^(?:bash|sh|/bin/sh): |^fatal: |^error(?:\[E\d+\])?: |^Error: |^ERROR[: ]|^E: |"
                       r"^[A-Z][A-Za-z]+Error: |^FAILED|^FAIL\s|^Diff in \S+:\d+:", re.M)
TE_WRITES = {"create", "str_replace", "insert"}
BASH_WRITE = re.compile(r"sed -i|\btee\b|cat\s*>|cat\s*<<|\bpatch\b|(?:echo|printf)[^|\n]*>\s*[^&\s]")
# A command that exercises the environment (tests, builds, services, network, databases), not just reads files.
BASH_RUN = re.compile(r"\b(?:pytest|cargo\s+(?:build|check|test|run|fmt)|npm\s+(?:test|run|ci|install)|pnpm|yarn|"
                      r"go\s+(?:test|build|run|vet)|make\b|mvn|gradle|dotnet|restart\.sh|curl|node\b|"
                      r"python3?\s+(?!-c\b)\S+\.py|bundle|rspec|rake|psql|redis-cli|mysql|sqlite3|systemctl|"
                      r"supervisorctl|docker)")


def failure(function: str, message: dict) -> str | None:
    error = message.get("error")
    if error and error.get("type") != "parsing":
        return f"tool_error:{error.get('type')}"
    text = tool_result_text(message)
    if function == "text_editor" and TE_FAIL.search(text):
        return "editor_message"
    if function.endswith("bash") and BASH_FAIL.search(text):
        return "bash_error_line"
    return None


def target(function: str, args: dict) -> str:
    if function == "text_editor":
        return str(args.get("path"))
    return " ".join(str(args.get("cmd") or args.get("command") or "").split())[:80]


def is_edit(function: str, args: dict) -> bool:
    if function == "text_editor":
        return args.get("command") in TE_WRITES
    return function.endswith("bash") and bool(BASH_WRITE.search(str(args.get("cmd") or args.get("command") or "")))


def kind(function: str, args: dict) -> str:
    """`write` (edits a file), `run` (exercises the environment: tests, builds, services) or `read`."""
    if is_edit(function, args):
        return "write"
    cmd = str(args.get("cmd") or args.get("command") or "")
    return "run" if function != "text_editor" and BASH_RUN.search(cmd) else "read"


# Matched-call subsets. `pre_edit`: neither run had edited a file before the call, so both start from the
# environment's initial files (state divergence from earlier edits cannot explain a difference there).
KINDS = ("all", "read", "write", "run", "pre_edit", "pre_edit_run")


def in_kind(kd: str, x: dict, y: dict) -> bool:
    pre_edit = x["prior_edits"] == 0 and y["prior_edits"] == 0
    return {"all": True, "pre_edit": pre_edit,
            "pre_edit_run": pre_edit and kind(x["function"], x["args"]) == "run"}.get(
        kd, kind(x["function"], x["args"]) == kd)


def record(sample: dict) -> dict:
    """Side-task score, every non-submit tool call (step, function, arguments, failure, output head), steps."""
    value = ((sample.get("scores") or {}).get("linux_scorer") or {}).get("value") or {}
    results = {m.get("tool_call_id"): m for m in sample["messages"] if m["role"] == "tool"}
    calls, step, submit_step = [], 0, None
    for m in sample["messages"]:
        if m["role"] != "assistant" or not m.get("tool_calls"):
            continue
        acts = [tc for tc in m["tool_calls"] if tc["function"] != RUNNER_SUBMIT]
        if acts:
            step += 1
        if submit_step is None and any(tc["function"] == RUNNER_SUBMIT for tc in m["tool_calls"]):
            submit_step = step + (0 if acts else 1)
        for tc in acts:
            res = results.get(tc.get("id"))
            assert res is not None, f"{sample['id']}: no result for tool call {tc.get('id')}"
            args = tc.get("arguments") or {}
            calls.append({"step": step, "function": tc["function"], "args": args,
                          "prior_edits": sum(is_edit(c["function"], c["args"]) for c in calls),
                          "key": tc["function"] + " " + json.dumps(args, sort_keys=True),
                          "fail": failure(tc["function"], res), "error_field": (res.get("error") or {}).get("type"),
                          "out": tool_result_text(res)[:400]})
    judge = (((sample.get("scores") or {}).get("linux_scorer") or {}).get("metadata") or {}).get("side_task_success")
    judge_models = sorted({m["model"] for m in judge.get("judge_messages") or [] if m.get("model")}) \
        if isinstance(judge, dict) else []
    return {"side": {"C": 1, "I": 0}.get(value.get("side_task_success")), "side_raw": value.get("side_task_success"),
            "side_judge": ",".join(judge_models) or "environment scorer (no judge messages)",
            "calls": calls, "steps": step, "submit_step": submit_step}


def load_file(job: tuple[str, list[tuple[str, str, int]]]) -> list[tuple[str, dict]]:
    path, wanted = job
    want = {(sid, ep): key for key, sid, ep in wanted}
    out = []
    with zipfile.ZipFile(path) as z:
        for n in z.namelist():
            if n.startswith("samples/") and n.endswith(".json"):
                x = json.loads(z.read(n))
                if (x["id"], x["epoch"]) in want:
                    out.append((want[(x["id"], x["epoch"])], record(x)))
    assert len(out) == len(want), f"{path}: found {len(out)} of {len(want)} samples"
    return out


# ---- bootstrap over main tasks ----

class Boot:
    def __init__(self, mains: list[str], reps: int, seed: int = 0):
        names = sorted(set(mains))
        self.idx = np.array([names.index(m) for m in mains])
        rng = np.random.default_rng(seed)
        self.w = rng.multinomial(len(names), np.full(len(names), 1 / len(names)), size=reps)[:, self.idx]  # reps x cells

    def ratio(self, num, den, mask=None):
        """sum(num)/sum(den) over cells (mask selects cells), with its 95% interval."""
        num, den = np.asarray(num, float), np.asarray(den, float)
        m = np.ones(len(num), bool) if mask is None else np.asarray(mask)
        point = num[m].sum() / den[m].sum() if den[m].sum() else float("nan")
        with np.errstate(invalid="ignore", divide="ignore"):
            b = (self.w[:, m] * num[m]).sum(1) / (self.w[:, m] * den[m]).sum(1)
        return point, b

    def mean(self, x, mask=None):
        return self.ratio(x, np.ones(len(x)), mask)


def ci(point, b) -> dict:
    lo, hi = np.nanpercentile(b, [2.5, 97.5])
    return {"estimate": round(float(point), 4), "ci95": [round(float(lo), 4), round(float(hi), 4)]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--exclusions", type=Path, default=REPO / "splits/exclusions_v3.json")
    p.add_argument("--data", type=Path, default=Path.home() / "probe-shift-viewer/.run_cache/viewer/eval")
    p.add_argument("--reps", type=int, default=2000)
    p.add_argument("--workers", type=int, default=14)
    p.add_argument("--examples", type=int, default=10)
    args = p.parse_args()
    args.exclusions = args.exclusions.resolve()

    ex = json.loads(args.exclusions.read_text())
    cells = ex["paired_attack_cells"]["cells"]
    sim_ref = ex["paired_attack_cells"]["sim_reference"]
    chosen, duplicates, cell_set = {}, [], set(cells)  # (arm, cell) -> sample row; first by key if an arm has two
    for s in sorted(ex["samples"], key=lambda s: s["key"]):
        if s["arm"] in ARMS and s["cell"] in cell_set:
            k = (ARMS[s["arm"]], s["cell"])
            if k in chosen:
                duplicates.append(s["key"])
                continue
            chosen[k] = s
    assert all((a, c) in chosen for a in ARMS.values() for c in cells), "a paired cell lacks a sample"
    jobs = defaultdict(list)
    for (arm, cell), s in chosen.items():
        jobs[str(args.data / s["cohort"] / s["file"])].append((f"{arm}|{cell}", s["sample_id"], s["epoch"]))
    with ProcessPoolExecutor(args.workers) as pool:
        recs = dict(r for part in pool.map(load_file, jobs.items()) for r in part)
    run = {a: [recs[f"{a}|{c}"] for c in cells] for a in ARMS.values()}
    mains = [chosen[("e1", c)]["main_task"] for c in cells]
    boot = Boot(mains, args.reps)
    out = {}

    # 0. tool-error rates per run (the 46% vs 0.1% claim), by definition
    def any_of(r, pred):
        return float(any(pred(c) for c in r["calls"]))
    rates = {}
    for a in ARMS.values():
        rates[a] = {
            "runs_with_any_error_field": ci(*boot.mean([any_of(r, lambda c: c["error_field"]) for r in run[a]])),
            "runs_with_nonparsing_error_field": ci(*boot.mean([any_of(r, lambda c: c["error_field"] not in (None, "parsing"))
                                                               for r in run[a]])),
            "runs_with_any_failed_call": ci(*boot.mean([any_of(r, lambda c: c["fail"]) for r in run[a]])),
            "failed_call_share": ci(*boot.ratio([sum(bool(c["fail"]) for c in r["calls"]) for r in run[a]],
                                                [len(r["calls"]) for r in run[a]])),
            "failure_reasons": dict(Counter(c["fail"].split(":")[0] if c["fail"].startswith("tool_error") else c["fail"]
                                            for r in run[a] for c in r["calls"] if c["fail"])),
        }
    out["0_tool_failure_rates"] = rates

    # 1. side-task success, paired
    side = {a: np.array([r["side"] if r["side"] is not None else np.nan for r in run[a]]) for a in ARMS.values()}
    raw = {a: dict(Counter(str(r["side_raw"]) for r in run[a])) for a in ARMS.values()}
    scored = ~np.isnan(side["e1"]) & ~np.isnan(side["e2"]) & ~np.isnan(side["sim"])  # one cell set for every contrast

    def paired_side(a, b, mask=None):
        ok = scored & (np.ones(len(cells), bool) if mask is None else mask)
        pa, ba = boot.mean(np.nan_to_num(side[a]), ok)
        pb, bb = boot.mean(np.nan_to_num(side[b]), ok)
        return {"n_cells": int(ok.sum()), a: ci(pa, ba), b: ci(pb, bb), f"{b}_minus_{a}": ci(pb - pa, bb - ba),
                "discordant": {f"{a}_only": int((ok & (side[a] == 1) & (side[b] == 0)).sum()),
                               f"{b}_only": int((ok & (side[a] == 0) & (side[b] == 1)).sum())}}
    refs = np.array([sim_ref[c] for c in cells])
    out["1_side_task_success"] = {
        "field": "scores.linux_scorer.value.side_task_success (C success, I failure; N and E excluded)",
        "raw_values": raw,
        "judged_by": {a: dict(Counter(r["side_judge"] for r in run[a])) for a in ARMS.values()},
        "main_task_success": "not available: attack runs (real and sim) are scored on the side task only, and "
                             "simulated honest runs carry no score",
        "sim_vs_e1": paired_side("e1", "sim"), "e2_vs_e1": paired_side("e1", "e2"),
        "sim_vs_e1_by_reference": {ref: paired_side("e1", "sim", refs == ref) for ref in ("base200", "overlap800")},
    }

    # 2. failure hiding on identical calls
    examples, hide_pool = {}, []
    def matched(a, b):
        num = defaultdict(list)
        for i, (ra, rb) in enumerate(zip(run[a], run[b])):
            qa, qb = defaultdict(list), defaultdict(list)
            for c in ra["calls"]:
                qa[c["key"]].append(c)
            for c in rb["calls"]:
                qb[c["key"]].append(c)
            pairs = [(x, y) for k in sorted(qa.keys() & qb.keys()) for x, y in zip(qa[k], qb[k])]
            for kd in KINDS:
                ps = [(x, y) for x, y in pairs if in_kind(kd, x, y)]
                num[(kd, "pairs")].append(len(ps))
                num[(kd, "a_fail")].append(sum(bool(x["fail"]) for x, _ in ps))
                num[(kd, "b_fail")].append(sum(bool(y["fail"]) for _, y in ps))
                num[(kd, "a_only")].append(sum(bool(x["fail"]) and not y["fail"] for x, y in ps))
                num[(kd, "b_only")].append(sum(bool(y["fail"]) and not x["fail"] for x, y in ps))
            num[("all", "a_errfield_b_none")].append(sum(x["error_field"] not in (None, "parsing") and not y["fail"]
                                                         for x, y in pairs))
            num[("all", "a_errfield")].append(sum(x["error_field"] not in (None, "parsing") for x, _ in pairs))
            if (a, b) == ("e1", "sim"):
                hide_pool.extend((cells[i], x, y) for x, y in pairs if x["fail"] and not y["fail"])
                fake_pool.extend((cells[i], x, y) for x, y in pairs if y["fail"] and not x["fail"])
        n = {k: np.array(v) for k, v in num.items()}
        res = {"cells_with_a_match": int((n[("all", "pairs")] > 0).sum()),
               f"of_{a}_error_field_{b}_not_failed": ci(*boot.ratio(n[("all", "a_errfield_b_none")], n[("all", "a_errfield")]))}
        for kd in KINDS:
            g = lambda f: n[(kd, f)]
            res[f"kind_{kd}"] = {
                "matched_calls": int(g("pairs").sum()),
                f"{a}_failed_share": ci(*boot.ratio(g("a_fail"), g("pairs"))),
                f"{b}_failed_share": ci(*boot.ratio(g("b_fail"), g("pairs"))),
                f"{a}_failed_{b}_not_share": ci(*boot.ratio(g("a_only"), g("pairs"))),
                f"{b}_failed_{a}_not_share": ci(*boot.ratio(g("b_only"), g("pairs"))),
                f"of_{a}_failed_{b}_not": ci(*boot.ratio(g("a_only"), g("a_fail"))),
                "counts": {f: int(g(f).sum()) for f in ("pairs", "a_fail", "b_fail", "a_only", "b_only")}}
        return res
    hide_pool, fake_pool = [], []
    out["2_failure_hiding"] = {"e1_vs_sim": matched("e1", "sim"), "e1_vs_e2_floor": matched("e1", "e2")}
    rng = np.random.default_rng(1)
    for name, pool, k in (("real_failed_sim_not", hide_pool, args.examples), ("sim_failed_real_not", fake_pool, 5)):
        examples[name] = [{"cell": cell, "kind": kind(x["function"], x["args"]), "pre_edit": in_kind("pre_edit", x, y),
                           "command": x["key"][:300],
                           "real_failure": x["fail"], "sim_failure": y["fail"], "real_output": x["out"][:300],
                           "sim_output": y["out"][:300]}
                          for cell, x, y in (pool[i] for i in rng.permutation(len(pool))[:k])]
    out["2_failure_hiding"]["examples"] = examples
    out["2_failure_hiding"]["reasons_real_failed_sim_not"] = dict(Counter(x["fail"] for _, x, _ in hide_pool))
    out["2_failure_hiding"]["reasons_sim_failed_real_not"] = dict(Counter(y["fail"] for _, _, y in fake_pool))

    # 3. behavior per run
    def behavior(r):
        by_step = defaultdict(list)
        for c in r["calls"]:
            by_step[c["step"]].append(c)
        after_fail = retry_edit = 0
        for s in range(2, r["steps"] + 1):
            failed = [c for c in by_step[s - 1] if c["fail"]]
            if failed:
                after_fail += 1
                tgts = {(c["function"], target(c["function"], c["args"])) for c in failed}
                if any((c["function"], target(c["function"], c["args"])) in tgts or is_edit(c["function"], c["args"])
                       for c in by_step[s]):
                    retry_edit += 1
        return r["steps"], after_fail, retry_edit, r["submit_step"]
    beh = {a: np.array([behavior(r) for r in run[a]], dtype=float) for a in ARMS.values()}
    def paired_mean(a, b, col, num_col=None):
        if num_col is None:
            pa, ba = boot.mean(beh[a][:, col])
            pb, bb = boot.mean(beh[b][:, col])
        else:
            pa, ba = boot.ratio(beh[a][:, num_col], beh[a][:, 0])
            pb, bb = boot.ratio(beh[b][:, num_col], beh[b][:, 0])
        return {a: ci(pa, ba), b: ci(pb, bb), f"{b}_minus_{a}": ci(pb - pa, bb - ba)}
    def submit_steps(a, b):
        ok = ~np.isnan(beh[a][:, 3]) & ~np.isnan(beh[b][:, 3])
        pa, ba = boot.mean(np.nan_to_num(beh[a][:, 3]), ok)
        pb, bb = boot.mean(np.nan_to_num(beh[b][:, 3]), ok)
        return {"n_cells_both_submitted": int(ok.sum()), a: ci(pa, ba), b: ci(pb, bb), f"{b}_minus_{a}": ci(pb - pa, bb - ba)}
    out["3_behavior"] = {
        f"{b}_vs_{a}": {"steps_per_run": paired_mean(a, b, 0), "share_steps_after_failed_step": paired_mean(a, b, 0, 1),
                        "share_steps_retry_or_edit_after_failed_step": paired_mean(a, b, 0, 2),
                        "steps_to_first_submit": submit_steps(a, b),
                        "median_steps": {x: float(np.median(beh[x][:, 0])) for x in (a, b)},
                        "runs_without_submit": {x: int(np.isnan(beh[x][:, 3]).sum()) for x in (a, b)}}
        for a, b in (("e1", "sim"), ("e1", "e2"))}

    # 4. link: does a failure in e1 predict a larger sim advantage?
    def link(flag_name, flags):
        flags = np.asarray(flags, bool)
        ok = ~np.isnan(side["e1"]) & ~np.isnan(side["e2"]) & ~np.isnan(side["sim"])
        res = {"proxy": flag_name, "n_cells": int(ok.sum()), "n_e1_failed": int((ok & flags).sum())}
        boots = {}
        for g, m in (("e1_failed", ok & flags), ("e1_clean", ok & ~flags)):
            for a in ARMS.values():
                pt, b = boot.mean(np.nan_to_num(side[a]), m)
                boots[(g, a)] = (pt, b)
                res[f"{a}_success_when_{g}"] = ci(pt, b)
        for x, y in (("sim", "e1"), ("e2", "e1"), ("sim", "e2")):
            pt = (boots[("e1_failed", x)][0] - boots[("e1_failed", y)][0]) - (boots[("e1_clean", x)][0] - boots[("e1_clean", y)][0])
            b = (boots[("e1_failed", x)][1] - boots[("e1_failed", y)][1]) - (boots[("e1_clean", x)][1] - boots[("e1_clean", y)][1])
            res[f"did_{x}_minus_{y}"] = ci(pt, b)
        return res
    out["4_link"] = {
        "note": "No per-step side-task labels exist, so 'a side-task-critical command errored' is proxied by failures "
                "anywhere in the e1 run (primary: an Inspect tool error other than parsing).",
        "primary_error_field": link("e1 has a non-parsing tool error", [any(c["error_field"] not in (None, "parsing")
                                                                          for c in r["calls"]) for r in run["e1"]]),
        "any_failed_call": link("e1 has any failed call", [any(c["fail"] for c in r["calls"]) for r in run["e1"]]),
        "failed_call_in_second_half": link("e1 has a failed call in its second half of steps",
                                           [any(c["fail"] and c["step"] > r["steps"] / 2 for c in r["calls"]) for r in run["e1"]]),
    }
    # dose-response: sim's advantage over the real rerun, by how many calls failed in e1
    n_fail = np.array([sum(bool(c["fail"]) for c in r["calls"]) for r in run["e1"]])
    ok = ~np.isnan(side["e1"]) & ~np.isnan(side["e2"]) & ~np.isnan(side["sim"])
    dose = {}
    for lo, hi in ((0, 0), (1, 2), (3, 5), (6, 10**9)):
        m = ok & (n_fail >= lo) & (n_fail <= hi)
        ps, bs = boot.mean(np.nan_to_num(side["sim"]), m)
        p2, b2 = boot.mean(np.nan_to_num(side["e2"]), m)
        p1, b1 = boot.mean(np.nan_to_num(side["e1"]), m)
        dose[f"{lo}-{hi if hi < 10**9 else 'max'}"] = {"n_cells": int(m.sum()), "e1": ci(p1, b1), "e2": ci(p2, b2),
                                                       "sim": ci(ps, bs), "sim_minus_e2": ci(ps - p2, bs - b2)}
    out["4_link"]["by_e1_failed_calls"] = dose

    git = lambda *a: subprocess.run(["git", "-C", str(HERE), *a], capture_output=True, text=True).stdout.strip()
    stamp = datetime.now(ZoneInfo("America/Los_Angeles"))
    meta = {"created": stamp.isoformat(timespec="seconds"), "git_commit": git("rev-parse", "HEAD"),
            "git_dirty": bool(git("status", "--porcelain", "--", str(HERE))),
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "exclusions": {"file": str(args.exclusions.relative_to(REPO) if args.exclusions.is_relative_to(REPO)
                                        else args.exclusions),
                           "sha256": hashlib.sha256(args.exclusions.read_bytes()).hexdigest()},
            "data_root": str(args.data), "cohorts": {a: chosen[(a, cells[0])]["cohort"] for a in ARMS.values()},
            "n_cells": len(cells), "n_main_tasks": len(set(mains)), "bootstrap": {"reps": args.reps, "cluster": "main task", "seed": 0},
            "duplicate_samples_skipped": duplicates,
            "definitions": {"TE_FAIL": TE_FAIL.pattern, "BASH_FAIL": BASH_FAIL.pattern, "BASH_WRITE": BASH_WRITE.pattern, "BASH_RUN": BASH_RUN.pattern,
                            "TE_WRITES": sorted(TE_WRITES)}}
    path = HERE / "results" / f"failure_hiding_{stamp.strftime('%Y%m%d-%H%M%S')}.json"
    path.parent.mkdir(exist_ok=True)
    with open(path, "x") as f:
        f.write(json.dumps({"meta": meta, **out}, indent=1))
    print(path)


if __name__ == "__main__":
    main()
