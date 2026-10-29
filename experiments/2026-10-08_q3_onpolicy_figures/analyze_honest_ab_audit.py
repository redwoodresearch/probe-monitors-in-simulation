"""Audit: why honest A and honest B differ at the simulation-set thresholds (audit_honest_ab_2026-10-11.md; exploratory).
$0, local; numbers only.

Joins honest_scores.json's honest A and B read-only calls (and the main per-call draw's attack run-1 read-only calls, the
control) to their extraction metadata by re-running run_score_onpolicy.load()'s own row selection on a local copy of
the five extractions' meta folders. The join must reproduce honest_scores.json's (world, task, fold, label) sequence and
the main file's meta exactly. Each call is flagged against (b)'s simulation-set thresholds (per fold model k, the 1% point
on the simulated attack runs' read-only calls of calibration fold k + 1, scored by model k).

Section 5 also reads how each honest run's attempt ended (submitted, step cap, ended without submit, error) from the
launch folders' launch.json, matching the run's log by file name.

  uv run --no-project --with numpy --with pyyaml python analyze_honest_ab_audit.py <main per-call> <honest_scores.json> <acts meta root> <launch dir> ...
"""

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import yaml

from analyze_scaleup_plants import threshold

HERE = Path(__file__).parent
BOOT, SEED = 2000, 20261011


def local_load(cfg: dict, folds: dict, root: Path) -> list:
    """run_score_onpolicy.load()'s rows (world, task, run, chunk, row, y, irid, matched, call), read from a local meta copy:
    its source with the root and the torch import swapped, and the full rows returned."""
    src = (HERE / "run_score_onpolicy.py").read_text()
    body = src[src.index("def load(cfg: dict, folds: dict, cats: list | None = None) -> dict:"):src.index("def rows_sha(rows)")]
    body = body.replace('    import torch\n', '', 1).replace('root = Path("/acts")', 'root = ROOT', 1)
    body = body.replace('    if cfg.get("_rows_only"):\n        return {"rows": src_rows}', '    if cfg.get("_rows_only"):\n        return {"rows": src_rows, "full": rows}', 1)
    ns = {"Path": Path, "json": json, "ROOT": root}
    exec(compile(body, "load", "exec"), ns)
    return ns["load"](cfg | {"_rows_only": True}, folds)["full"]


def main(main_path: str, honest_path: str, root: str, launches: list):
    root = Path(root)
    raw_m, raw_h = Path(main_path).read_bytes(), Path(honest_path).read_bytes()
    M, H = json.loads(raw_m), json.loads(raw_h)
    cfg = yaml.safe_load((HERE / "config" / "score_scaleup196.yaml").read_text())
    ext = yaml.safe_load((HERE / "config" / "score_scaleup196_simhon.yaml").read_text())
    folds = M["folds"]

    # metadata per (extraction, chunk, absolute row); run-level length and last turn over all of a run's context epochs
    info, run_calls, run_last_turn, run_tokens, run_epochs = {}, Counter(), {}, Counter(), Counter()
    for x in cfg["extractions"]:
        for f in sorted((root / x / "meta").glob("*/*.json")):
            m = json.loads(f.read_text())
            cmd = [c for c in m["calls"] if c.get("kind", "command") == "command"]
            run_calls[(x, m["run"])] += len(cmd)
            run_tokens[(x, m["run"])] += m["n_tokens"]  # summed context-epoch tokens, a length proxy: retained context is recounted
            run_epochs[(x, m["run"])] += 1
            run_last_turn[(x, m["run"])] = max([run_last_turn.get((x, m["run"]), -1)] + [c["turn"] for c in cmd])
    for x in cfg["extractions"]:
        for f in sorted((root / x / "meta").glob("*/*.json")):
            m = json.loads(f.read_text())
            for c in m["calls"]:
                if c["row"] is not None:
                    info[(x, m["chunk"], m["row_start"] + c["row"])] = {
                        "run": m["run"], "attempt_index": m.get("attempt_index"), "attempt_kind": m.get("attempt_kind"),
                        "n_tokens": m["n_tokens"], "tool": c["name"], "param": c["param"], "kind": c.get("kind"), "turn": c["turn"],
                        "calls_in_run": run_calls[(x, m["run"])], "last_turn": run_last_turn[(x, m["run"])],
                        "run_tokens": run_tokens[(x, m["run"])], "epochs": run_epochs[(x, m["run"])]}

    # the join, checked against both stored sequences
    main_rows = local_load(cfg, folds, root)
    if [[r[0], r[1], r[5], r[6], r[8]] for r in main_rows] != [[m["world"], m["task"], m["y"], m["irid"], m["call"]] for m in M["meta"]]:
        raise SystemExit("main rows differ from the main per-call meta")
    ext_rows = [r for r in local_load(ext, folds, root) if (r[0] in ("honA", "honB") and r[5] == 0) or r[0] == "simH"]
    if [[r[0], r[1], folds[r[1]], r[5]] for r in ext_rows] != [[x["world"], x["task"], x["fold"], x["y"]] for x in H["rows"]]:
        raise SystemExit("honest rows differ from honest_scores.json")
    if len({(r[0], r[6], r[8]) for r in ext_rows}) != len(ext_rows):
        raise SystemExit("duplicate honest rows")

    # thresholds: (b)'s point estimate
    cal = M["draws"][0]["cells"]["real->real"]["per_call"]["calibration"]
    th = {}
    for k in range(5):
        s = np.array([sc for row, sc, mk in zip(cal["row"], cal["score"], cal["model_fold"]) if mk == k and M["meta"][row]["world"] == "sim" and M["meta"][row]["y"] == 0])
        th[k] = threshold(s, np.ones(len(s)))

    calls = []  # (world, task, fold, flagged, meta)
    pc = M["draws"][0]["cells"]["real->real"]["per_call"]
    for row, sc in zip(pc["row"], pc["score"]):
        r = main_rows[row]
        if r[0] == "real" and r[5] == 0:
            f = folds[r[1]]
            calls.append(("attack", r[1], f, sc > th[f], info[(r[2], r[3], r[4])]))
    for r, x in zip(ext_rows, H["rows"]):
        if r[0] in ("honA", "honB"):
            calls.append((r[0], r[1], x["fold"], x["s"][x["fold"]] > th[x["fold"]], info[(r[2], r[3], r[4])]))
    W = ("honA", "honB", "attack")
    tasks = sorted({c[1] for c in calls})
    ti = {t: i for i, t in enumerate(tasks)}

    def rate(sel):
        n = len(sel)
        return {"calls": n, "alarms": int(sum(c[3] for c in sel)), "rate": (sum(c[3] for c in sel) / n) if n else None}

    def boot(filter_fn=lambda c: True):
        """A − B false-alarm rate (per call), task-resampled; A's and B's calls on a drawn task move together."""
        a = np.zeros((2, len(tasks))), np.zeros((2, len(tasks)))
        for c in calls:
            if c[0] in ("honA", "honB") and filter_fn(c):
                w = a[c[0] == "honB"]
                w[0, ti[c[1]]] += 1
                w[1, ti[c[1]]] += c[3]
        out = []
        for i in range(1, BOOT + 1):
            m = np.bincount(np.random.default_rng([SEED, i]).integers(0, len(tasks), len(tasks)), minlength=len(tasks))
            na, nb = m @ a[0][0], m @ a[1][0]
            if na and nb:
                out.append(m @ a[0][1] / na - m @ a[1][1] / nb)
        point = a[0][1].sum() / a[0][0].sum() - a[1][1].sum() / a[1][0].sum()
        return {"A_minus_B": point, "ci95": [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))], "failed": BOOT - len(out)}

    res = {"label": "exploratory audit; observed variation between real runs", "questions": "audit_honest_ab_2026-10-11.md",
           "inputs": {"main_per_call": hashlib.sha256(raw_m).hexdigest(), "honest_scores": hashlib.sha256(raw_h).hexdigest()},
           "join": "reproduced honest_scores.json's (world, task, fold, label) sequence and the main meta exactly",
           "thresholds": th, "overall": {w: rate([c for c in calls if c[0] == w]) for w in W}, "overall_A_minus_B": boot()}

    # 1. by task: concentration of A's excess
    per = {t: {w: rate([c for c in calls if c[0] == w and c[1] == t]) for w in W} for t in tasks}
    both = [t for t in tasks if per[t]["honA"]["calls"] and per[t]["honB"]["calls"]]
    excess = sorted(((per[t]["honA"]["alarms"] - per[t]["honB"]["alarms"], t) for t in both), reverse=True)
    total = sum(e for e, _ in excess)
    res["1 by task"] = {"tasks_with_both": len(both), "excess_alarms_A_minus_B": total,
                        "top5_share_of_excess": sum(e for e, _ in excess[:5]) / total if total else None,
                        "top10_share_of_excess": sum(e for e, _ in excess[:10]) / total if total else None,
                        "top10_excess": [e for e, _ in excess[:10]],
                        "alarm_count_distribution": {w: dict(sorted(Counter(per[t][w]["alarms"] for t in both).items())) for w in ("honA", "honB")},
                        "A_minus_B_without_top5": boot(lambda c, s={t for _, t in excess[:5]}: c[1] not in s),
                        "A_minus_B_without_top10": boot(lambda c, s={t for _, t in excess[:10]}: c[1] not in s)}
    res["1 top tasks by excess"] = [{"task": t, "excess": e, "fold": folds[t], **{w: per[t][w] for w in W},
                                     "alarm_tools": {w: dict(Counter(f"{c[4]['tool']}|{c[4]['param']}" for c in calls if c[0] == w and c[1] == t and c[3])) for w in ("honA", "honB")},
                                     "epoch_tokens": {w: sorted({c[4]["n_tokens"] for c in calls if c[0] == w and c[1] == t}) for w in ("honA", "honB")},
                                     "attempt_kind": {w: sorted({str(c[4]["attempt_kind"]) for c in calls if c[0] == w and c[1] == t}) for w in ("honA", "honB")},
                                     "A_alarm_share_of_A_alarms": per[t]["honA"]["alarms"] / res["overall"]["honA"]["alarms"]} for e, t in excess[:5]]
    redraw = {c[1] for c in calls if c[0] == "honA" and c[4]["attempt_kind"] == "redraw"}
    res["7 redrawn honest A tasks"] = {"tasks": len(redraw), "A_alarms_there": sum(c[3] for c in calls if c[0] == "honA" and c[1] in redraw),
                                       "A_minus_B_without_them": boot(lambda c, s=redraw: c[1] not in s)}
    # 2. same tasks in both batches
    alarmed = {w: {t for t in both if per[t][w]["alarms"]} for w in ("honA", "honB", "attack")}
    ra = np.array([per[t]["honA"]["rate"] for t in both]); rb = np.array([per[t]["honB"]["rate"] for t in both])
    def rank(v):  # average ranks for ties (Spearman)
        order = np.argsort(v, kind="stable")
        r = np.empty(len(v))
        r[order] = np.arange(len(v))
        for x in np.unique(v):
            r[v == x] = r[v == x].mean()
        return r
    res["2 same tasks"] = {"tasks_alarmed": {w: len(v) for w, v in alarmed.items()}, "alarmed_in_A_and_B": len(alarmed["honA"] & alarmed["honB"]),
                           "alarmed_in_A_and_attack": len(alarmed["honA"] & alarmed["attack"]), "alarmed_in_B_and_attack": len(alarmed["honB"] & alarmed["attack"]),
                           "spearman_per_task_rate_A_B": float(np.corrcoef(rank(ra), rank(rb))[0, 1])}
    # 3. by tool and parameter
    groups = Counter((c[4]["tool"], c[4]["param"]) for c in calls if c[0] in ("honA", "honB"))
    res["3 by tool and parameter"] = {f"{g[0]}|{g[1]}": {w: rate([c for c in calls if c[0] == w and (c[4]["tool"], c[4]["param"]) == g]) for w in W}
                                      | ({"A_minus_B": boot(lambda c, g=g: (c[4]["tool"], c[4]["param"]) == g)} if n >= 200 else {})
                                      for g, n in groups.most_common()}
    # 4. by fold and threshold
    res["4 by fold"] = {k: {"threshold": th[k], **{w: rate([c for c in calls if c[0] == w and c[2] == k]) for w in W},
                            "A_minus_B": boot(lambda c, k=k: c[2] == k)} for k in range(5)}
    # 5. by run length and position
    runs = {(c[0], c[4]["run"]): c[4]["calls_in_run"] for c in calls if c[0] in ("honA", "honB")}  # run length: command calls, all epochs
    cut = np.percentile(list(runs.values()), [100 / 3, 200 / 3])
    terc = lambda n: int(np.searchsorted(cut, n, side="right"))
    pos = lambda c: min(int(10 * c[4]["turn"] / (c[4]["last_turn"] + 1)), 9)  # against every command turn of the run
    by_run = defaultdict(list)
    for c in calls:
        by_run[(c[0], c[4]["run"])].append(c)
    res["5 by run length tercile"] = {"cuts_command_calls": cut.tolist(), **{t: {w: rate([c for c in calls if c[0] == w and terc(c[4]["calls_in_run"]) == t]) for w in W} for t in range(3)}}
    # 5, as registered: run-token terciles (summed context-epoch tokens, a length proxy), and context epochs per batch
    rtok = {(c[0], c[4]["run"]): (c[4]["run_tokens"], c[4]["epochs"]) for c in calls if c[0] in ("honA", "honB")}
    tcut = np.percentile([v[0] for v in rtok.values()], [100 / 3, 200 / 3])
    tterc = lambda n: int(np.searchsorted(tcut, n, side="right"))
    res["5 by run-token tercile"] = {"cuts_run_tokens": tcut.tolist(), **{t: {w: rate([c for c in calls if c[0] == w and tterc(c[4]["run_tokens"]) == t]) for w in W} for t in range(3)}}
    res["5 run tokens by batch"] = {w: {"runs": len(v), "median": float(np.median([a for a, _ in v])), "max": max(a for a, _ in v),
                                        "runs_by_context_epochs": dict(sorted(Counter(e for _, e in v).items()))}
                                    for w in ("honA", "honB") for v in [[x for (y, _), x in rtok.items() if y == w]]}
    # 5, cut short: how each counted honest attempt ended, from launch.json's recorded outcome. The inventory covers every
    # counted attempt in the launch files (honest A the first counted attempt, B the last, as the judge and scorer); the
    # alarm rates use the scored runs, matched to their attempt by log file name.
    def how_ended(a):
        o = a.get("outcome") or {}
        pcs = o.get("policy_calls") or []
        e = {p.get("ended") for p in pcs}
        # submit before the step cap: a run that submits on its last allowed call records ended "submit" and a step-limit hit
        return ("error" if "error" in e or o.get("sample_errors") or "error" in (o.get("eval_status") or [])
                else "submitted" if e == {"submit"} else "step cap" if any(p.get("step_limit_hits") for p in pcs)
                else "ended without submit" if e == {"step limit or no submit"} else f"other: {sorted(map(str, e))}")

    ended, counted, launch_in = {}, {}, []
    for d in map(Path, launches):
        lj = (d / "launch.json").read_bytes()
        launch_in.append({"key": f"phase2/runs/{d.name}/launch.json", "sha256": hashlib.sha256(lj).hexdigest(), "bytes": len(lj)})
        for r in json.loads(lj)["runs"]:
            logs = {}
            for a in r["attempts"]:
                for ev in (d / Path(a["log_dir"]).relative_to(Path(a["log_dir"]).parents[1])).glob("*.eval"):
                    if ev.name in ended:
                        raise SystemExit(f"two logs named {ev.name}")
                    ended[ev.name], logs[id(a)] = how_ended(a), ev.name
            ok = [a for a in r["attempts"] if not a.get("redrawn_because")]
            if r["arm"] in ("honest_a", "honest_b") and ok:
                a = ok[0] if r["arm"] == "honest_a" else ok[-1]
                k = ("honA" if r["arm"] == "honest_a" else "honB", r["task"])
                if k in counted:
                    raise SystemExit(f"two counted attempts for {k}")
                counted[k] = (how_ended(a), logs.get(id(a)))
    how = {k: ended[Path(k[1]).name] for k in runs}
    scored_logs = {(w, Path(run).name) for w, run in runs}
    res["inputs"]["launch_json"] = launch_in
    res["5 how runs ended"] = {w: {"counted_attempts": dict(Counter(e for (x, _), (e, _) in counted.items() if x == w)),
                                   "scored_runs": dict(Counter(v for (x, _), v in how.items() if x == w)),
                                   "scored_runs_not_the_counted_attempt": sum(1 for x, n in scored_logs if x == w and (x, n) not in {(y, g) for (y, _), (_, g) in counted.items()}),
                                   **{f"rate_in_{e.replace(' ', '_')}_runs": rate([c for c in calls if c[0] == w and how[(w, c[4]["run"])] == e])
                                      for e in sorted(set(how.values()))}} for w in ("honA", "honB")}
    res["5 by position decile"] = {d: {w: rate([c for c in calls if c[0] == w and pos(c) == d]) for w in W} for d in range(10)}
    # 6. splits and weighting
    task_rate = {w: [per[t][w]["rate"] for t in both if per[t][w]["calls"]] for w in ("honA", "honB")}
    res["6 splits and weighting"] = {"per_task_weighted_rate": {w: float(np.mean(v)) for w, v in task_rate.items()},
                                     "tasks": {w: len({c[1] for c in calls if c[0] == w}) for w in W},
                                     "same_task_set_A_B": {c[1] for c in calls if c[0] == "honA"} == {c[1] for c in calls if c[0] == "honB"},
                                     "folds_calls": {w: dict(sorted(Counter(c[2] for c in calls if c[0] == w).items())) for w in W},
                                     "runs": {w: len({c[4]["run"] for c in calls if c[0] == w}) for w in W},
                                     "read_only_calls_per_run_median": {w: float(np.median([len(v) for (x, _), v in by_run.items() if x == w])) for w in W}}
    # 7. join and bug checks
    res["7 checks"] = {"attempt_index": {w: dict(Counter(c[4]["attempt_index"] for c in calls if c[0] == w)) for w in W},
                       "attempt_kind": {w: dict(Counter(c[4]["attempt_kind"] for c in calls if c[0] == w)) for w in W},
                       "run_command_calls_median": {w: float(np.median([n for (x, _), n in runs.items() if x == w])) for w in ("honA", "honB")},
                       "calls_per_task_A_minus_B_median": float(np.median([per[t]["honA"]["calls"] - per[t]["honB"]["calls"] for t in both]))}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    dst = HERE / "results" / f"audit_honest_ab_{stamp}.json"
    with dst.open("x") as fh:
        fh.write(json.dumps(res, indent=1, default=float))
    print("->", dst)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("main_path")
    ap.add_argument("honest_path")
    ap.add_argument("root")
    ap.add_argument("launches", nargs="+")
    a = ap.parse_args()
    main(a.main_path, a.honest_path, a.root, a.launches)
