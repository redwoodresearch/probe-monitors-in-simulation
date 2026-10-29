"""Fresh-task honest runs, devbox side (PLAN.md amendment 1): run the full batch's stages back to back, each with
launch_fresh.sh, so the devbox stays busy overnight. Before each stage it applies the committed stop rule and stops (writes
fresh_chain.CHAIN_STOP with the reason and launches nothing more) when a check fails:
  - every earlier launch in fresh_chain.jsonl has its "stage done" line (a launch without one stops the chain: resume by hand);
  - the running API total: smoke, parity and every earlier stage, each re-priced from its final logs with the watcher's own
    run_spend (policy_spend_watch.py), never below the watcher's final reading; an attempt that can't be priced, a final
    reading with a running run, POLICY_HALT, or a watcher SIGTERM stops the chain;
  - every earlier stage finished all its planned runs (launch.json): each is 'done' (its last attempt ended and called the
    policy) or 'no policy call' (every attempt ended before the policy's first call, e.g. a task whose setup fails: $0, an
    errored run the RULE leaves out); a stage with more than 5 'no policy call' runs stops the chain (an infrastructure
    smell). Folders: honest_b/<run id>/ and the launcher's pre-start redraws <run id>_redraw<n>/, one .eval each;
  - errored runs (no policy call, or a done run whose last outcome is not clean, has sample errors or is not 'success',
    list_fresh_runs.py's rule): the chain stops if one main task has errored runs in more than one stage (each task's runs
    sit in one stage, so this guards a changed task file), or if the errored share so far, projected to all 654 runs,
    reaches 26 (80% of the RULE's 5% no-verdict line, about 33 runs) (0d via q2, 10-14 about 02:15);
  - stage k >= 2: projection = running total + (planned runs not yet launched) x the fresh stages' API $ per done run;
    above $1,000 stops the chain (report, no trimming);
  - the stage's config (config/pilot_fresh_honest_s<k>.yaml) holds policy_stop_usd = min($170, $975 - running total);
  - worst-per-run reservation = max($2.614, the costliest priced attempt so far).
After a launch it waits for both the launcher's tmux session (phase2 fh) and the watcher's (canary fhw) to end before
reading anything. Events go to <runs>/fresh_chain.jsonl, for the laptop's transfers. Counts and dollars only.

  $PY chain_fresh.py --first 1 --last 1 --dry-run   (the checks for stage 1, no launch)
  $PY chain_fresh.py --mark-done <k> <run> "<why>"  (after a CHAIN_STOP on a finished stage: the same checks with this code)
  tmux -L chain new -d -s fhc "cd ~/phase2/attack_check_fresh && $PY chain_fresh.py --first 1 --last 7 >> ~/phase2/runs/fresh_chain.log 2>&1"
"""
import argparse
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import yaml

from policy_spend_watch import run_spend  # the watcher's own pricing (the September snapshot's sim.prices)

C = Path("/home/aswin/phase2/attack_check_fresh")
RUNS = Path("/home/aswin/phase2/runs")
CHAIN = RUNS / "fresh_chain.jsonl"
DONE_BEFORE = ["2026-10-13_234837_fresh_smoke", "2026-10-13_235821_fresh_parity"]
CAP, RESERVE, STAGE_STOP, WORST0, TOTAL_RUNS, MAX_NO_CALL, ERR_PROJ_STOP = 1000.0, 25.0, 170.0, 2.614, 654, 5, 26
NO_CALL_OVERRIDES = {5: 10}  # AMENDMENT_STOPS_2026-10-14.md: stage 5's known count, the only allowed override
POLICY = "openai/gpt-5.6-sol"
PT = ZoneInfo("America/Los_Angeles")


def log(rec):
    rec = {"time": datetime.now(PT).isoformat(timespec="seconds")} | rec
    with open(CHAIN, "a") as fh:
        fh.write(json.dumps(rec) + "\n")
    print(json.dumps(rec), flush=True)


def stop(reason, **kw):
    log({"event": "CHAIN_STOP", "reason": reason} | kw)
    (RUNS / "fresh_chain.CHAIN_STOP").write_text(reason + "\n")
    raise SystemExit(reason)


def priced(run, prices):
    """(total $, costliest attempt $) of a finished run folder, re-priced from its final logs."""
    r = RUNS / run
    rd = json.loads((r / "policy_spend.json").read_text())
    if rd.get("running") != 0:
        stop("a final reading still shows a running run", run=run)
    if (r / "POLICY_HALT").exists() or any(json.loads(x).get("action") for x in (r / "policy_spend.jsonl").read_text().splitlines()):
        stop("the launcher halted or the watcher sent SIGTERM", run=run)
    usd = []
    for d in sorted((r / "honest_b").glob("*")):
        try:
            usd.append(run_spend(d, POLICY, prices, policy_calls=True))
        except Exception as e:  # noqa: BLE001 — any unpriceable attempt stops the chain
            stop("an attempt could not be priced from its final log", run=run, attempt=d.name, error=type(e).__name__)
    return max(sum(usd), rd["actual_usd"]), max(usd, default=0.0)


def completed(run, planned, max_no_call=None):
    """(done runs, runs ended without a policy call) of a finished stage, by the rules in the docstring; otherwise stop."""
    lj = json.loads((RUNS / run / "launch.json").read_text())
    runs = {r["task"]: r for r in lj["runs"] if r["arm"] == "honest_b"}
    want = {Path(a["log_dir"]).name for t in planned for a in (runs[t]["attempts"] if t in runs else [])}
    dirs = {d.name: list(d.glob("*.eval")) for d in (RUNS / run / "honest_b").glob("*")}
    if set(dirs) != want or any(len(v) != 1 for v in dirs.values()) or any(
            not (d in planned or (d.rsplit("_redraw", 1)[0] in planned and d.rsplit("_redraw", 1)[1].isdigit())) for d in dirs):
        stop("a stage's folders are not exactly its planned runs' attempts, one .eval each", run=run, planned=len(planned),
             folders=len(dirs), attempts=len(want))
    done = [t for t in planned if t in runs and runs[t].get("state") == "done" and runs[t]["attempts"]
            and "end" in runs[t]["attempts"][-1] and runs[t]["attempts"][-1].get("policy_called") is True]
    no_call = [t for t in planned if t in runs and runs[t].get("state") == "no policy call" and runs[t]["attempts"]
               and all("end" in a and a.get("policy_called") is False for a in runs[t]["attempts"])]
    if set(runs) != set(planned) or len(done) + len(no_call) != len(planned) or lj.get("stopped_by_signal") or lj.get("not_run"):
        stop("the launcher did not finish every planned run", run=run, done=len(done), no_policy_call=len(no_call),
             unplanned=len(set(runs) - set(planned)), stopped_by_signal=bool(lj.get("stopped_by_signal")), not_run=len(lj.get("not_run") or []))
    if len(no_call) > (MAX_NO_CALL if max_no_call is None else max_no_call):
        stop("more than 5 runs of a stage ended before any policy call", run=run, no_policy_call=len(no_call))
    def unclean(r):
        o = r["attempts"][-1].get("outcome") or {}
        return o.get("clean") is not True or bool(o.get("sample_errors")) or o.get("eval_status") != ["success"]
    errored = sorted(set(no_call) | {t for t in done if unclean(runs[t])})
    return len(done), len(no_call), errored


def tmux_alive(sock, name):
    return subprocess.run(["tmux", "-L", sock, "has-session", "-t", name], capture_output=True).returncode == 0


def mark_done(k, run, why, max_no_call=None):
    if max_no_call is not None and NO_CALL_OVERRIDES.get(k) != max_no_call:
        raise SystemExit(f"--no-call-limit {max_no_call} for stage {k} is not an amendment's override {NO_CALL_OVERRIDES}")
    """After a CHAIN_STOP on a finished stage: re-run the stage-done checks on it with this code; if they pass, move the stop
    file aside and write the stage's 'stage done' line, marked as resolved by hand."""
    from sim.prices import load_prices

    stages = json.loads((C / "config/pilot_tasks_fresh_honest.json").read_text())["stages"]
    ev = [json.loads(x) for x in CHAIN.read_text().splitlines()]
    if not (RUNS / "fresh_chain.CHAIN_STOP").exists() or ev[-1].get("event") != "CHAIN_STOP" or ev[-1].get("run") != run:
        raise SystemExit("the last event is not a CHAIN_STOP on this run")
    if not any(e.get("event") == "stage launch" and e.get("run") == run and e.get("stage") == k for e in ev):
        raise SystemExit("no launch of this stage under this run")
    if tmux_alive("phase2", "fh") or tmux_alive("canary", "fhw"):
        raise SystemExit("the launcher or watcher is still running")
    usd, top = priced(run, load_prices())
    n, nc, err = completed(run, stages[str(k)], max_no_call)
    (RUNS / "fresh_chain.CHAIN_STOP").rename(RUNS / f"fresh_chain.CHAIN_STOP.resolved_{datetime.now(PT).strftime('%Y%m%d_%H%M%S')}")
    log({"event": "stage done", "stage": k, "run": run, "done runs": n, "no policy call runs": nc, "errored runs": len(err),
         "usd": round(usd, 3), "costliest run usd": round(top, 3), "resolved by hand after CHAIN_STOP": why}
        | ({"no policy call limit (amendment)": max_no_call} if max_no_call is not None else {}))


def main(first, last, dry):
    from sim.prices import load_prices

    prices = load_prices()
    stages = json.loads((C / "config/pilot_tasks_fresh_honest.json").read_text())["stages"]
    ev = [json.loads(x) for x in CHAIN.read_text().splitlines()] if CHAIN.exists() else []
    launched = [e["run"] for e in ev if e.get("event") == "stage launch"]
    fresh = {e["run"]: e["stage"] for e in ev if e.get("event") == "stage done"}
    override = {e["run"]: e["no policy call limit (amendment)"] for e in ev if e.get("event") == "stage done" and "no policy call limit (amendment)" in e}
    if any(NO_CALL_OVERRIDES.get(fresh[r]) != lim for r, lim in override.items()):
        stop("a recorded no-policy-call override is not one the amendment allows")
    if set(launched) != set(fresh):
        stop("a launch in fresh_chain.jsonl has no 'stage done' line: resolve it by hand", unresolved=sorted(set(launched) - set(fresh)))
    for k in range(first, last + 1):
        if (RUNS / "fresh_chain.CHAIN_STOP").exists():
            raise SystemExit("fresh_chain.CHAIN_STOP exists")
        if k in fresh.values():
            stop("this stage was already launched", stage=k)
        p = {run: priced(run, prices) for run in DONE_BEFORE + list(fresh)}
        total = sum(x[0] for x in p.values())
        fresh_usd = sum(p[run][0] for run in fresh)
        comp = {run: completed(run, stages[str(s)], override.get(run)) for run, s in fresh.items()}
        fresh_n = sum(c[0] for c in comp.values())
        cells = {c["key"]: c["main"] for c in json.loads((C / "config/pilot_tasks_fresh_honest.json").read_text())["cells"]}
        err_stages = {}
        for run, c in comp.items():
            for t in c[2]:
                err_stages.setdefault(cells[t], set()).add(fresh[run])
        if any(len(v) > 1 for v in err_stages.values()):
            stop("one main task has errored runs in more than one stage", tasks=sorted(t for t, v in err_stages.items() if len(v) > 1))
        launched_n = sum(len(stages[str(s)]) for s in fresh.values())
        n_err = sum(len(c[2]) for c in comp.values())
        if launched_n and n_err / launched_n * TOTAL_RUNS >= ERR_PROJ_STOP:
            stop("the errored share, projected to all 654 runs, heads toward the RULE's 5% line", errored=n_err, launched=launched_n,
                 projected=round(n_err / launched_n * TOTAL_RUNS, 1))
        worst = max([WORST0] + [x[1] for x in p.values()])
        remaining = TOTAL_RUNS - sum(len(stages[str(s)]) for s in fresh.values())
        projection = total + remaining * fresh_usd / fresh_n if fresh_n else None
        if projection is not None and projection > CAP:
            stop("projection for all 654 crosses $1,000", total=round(total, 3), projection=round(projection, 2))
        want = min(STAGE_STOP, CAP - RESERVE - total)
        cfg = f"config/pilot_fresh_honest_s{k}.yaml"
        have = float(yaml.safe_load((C / cfg).read_text())["policy_stop_usd"])
        if abs(have - want) > 0.0005:
            stop("the stage config's stop is not the rule's", stage=k, config_stop=have, rule_stop=round(want, 3))
        if dry:  # --dry-run: the checks and numbers for stage `first`, no launch, nothing written
            print(json.dumps({"stage": k, "running total usd": round(total, 3), "per folder usd": {r: round(x[0], 3) for r, x in p.items()},
                              "projection usd": round(projection, 2) if projection else None, "stop usd": want, "worst": round(worst, 3)}))
            return
        name = datetime.now(PT).strftime("%Y-%m-%d_%H%M%S") + f"_fresh_s{k}"
        log({"event": "stage launch", "stage": k, "run": name, "runs": len(stages[str(k)]), "running total usd": round(total, 3),
             "fresh usd per done run so far": round(fresh_usd / fresh_n, 4) if fresh_n else None,
             "errored runs so far": n_err if fresh else 0,
             "projection usd": round(projection, 2) if projection else None, "stop usd": want, "worst per run usd": round(worst, 3)})
        out = subprocess.run(["bash", str(C / "launch_fresh.sh"), name, str(k), cfg, f"{worst:.3f}"], cwd=C, capture_output=True, text=True)
        if out.returncode:
            stop("launch_fresh.sh failed", stage=k, tail=out.stderr[-500:])
        while tmux_alive("phase2", "fh") or tmux_alive("canary", "fhw"):  # the launcher, then the watcher's final reading
            time.sleep(30)
        usd, top = priced(name, prices)
        n, nc, err = completed(name, stages[str(k)])
        fresh[name] = k
        log({"event": "stage done", "stage": k, "run": name, "done runs": n, "no policy call runs": nc, "errored runs": len(err),
             "usd": round(usd, 3), "costliest run usd": round(top, 3)})
    log({"event": "chain done", "stages": [first, last]})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--first", type=int)
    ap.add_argument("--last", type=int)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--mark-done", nargs=3, metavar=("STAGE", "RUN", "WHY"), help="resolve a CHAIN_STOP on a finished stage")
    ap.add_argument("--no-call-limit", type=int, help="with --mark-done only: this stage's no-policy-call limit, set by a dated amendment")
    a = ap.parse_args()
    if a.mark_done and (a.dry_run or a.first is not None or a.last is not None):
        raise SystemExit("--mark-done takes no other option")
    if a.no_call_limit is not None and not a.mark_done:
        raise SystemExit("--no-call-limit goes with --mark-done only")
    if a.mark_done:
        mark_done(int(a.mark_done[0]), a.mark_done[1], a.mark_done[2], a.no_call_limit)
    else:
        main(a.first, a.last, a.dry_run)
