"""Settings-equality gate for the canary (RULE.md, gate 2). The reference is what the purchase fork resolved from the
existing launches' run records (results/sim_settings_2026-10-14_125923.json @ 9e40c801, aswin/q3-q35-simhonest-purchase:
191 sim_attack@september runs, one given-settings hash, one value per metadata field). Its check_sim_settings.py is
reused unchanged (copied beside this file as check_sim_settings_purchase.py); only the reference comes from that JSON,
because the reference launches live on another devbox. Prints keys, values and hashes only.

Exactly one mode:
  --harness <dir>  pre-launch: config/sim.yaml and config/run.yaml hash, and run.yaml's pinned tree, equal a reference
                   launch's.
  --new <run dir>  a watch for the whole launch: every POLL_S it reads launch.json (a partial write is retried, at most
                   10 times in a row). Every attempt of every started simulated run (a replaced one too) must have given
                   settings hashing to the reference's one hash; every done run must have exactly one .eval whose
                   recorded metadata and simulator call configs are the reference's. It ends PASS when --session has
                   ended and the final launch.json shows every started simulated run terminal and checked. Any mismatch
                   or error (setup included) interrupts the launcher before anything else: Ctrl+C, after 5 minutes kill
                   the session and every inspect eval's process group, and confirm (logging and the <run dir>.STOP_REQUEST
                   note are best effort). allin_stop.sh then sees the launcher's pane gone and stops the apps once every
                   launcher of the group has ended; idle servers also exit by themselves after 20 minutes.

Runs ON THE DEVBOX from the harness folder with the September venv:
  $PY gate_settings.py --reference-json <json> --out <json> (--harness . | --new <run dir> --session <launcher session>)
"""

import argparse
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path

import check_sim_settings_purchase as g
import yaml

POLL_S = 120
LIVE = ("running", "waiting", "ready")


def say(*a) -> None:
    """Logging that can never stop a stop (a full disk makes print raise)."""
    try:
        print(*a, flush=True)
    except Exception:  # noqa: BLE001
        pass


def tmux(*a: str) -> int:
    return subprocess.run(["tmux", "-L", "phase2", *a]).returncode


def stop(session: str, run: Path) -> None:
    """Ask for the group stop (allin_stop.sh reads the file), Ctrl+C the launcher, kill its session if it hasn't ended
    within 5 minutes, and confirm."""
    try:  # best effort: the stop below runs whatever happens here
        Path(f"{run}.STOP_REQUEST").write_text(time.strftime("%F %T") + " settings gate\n")
    except OSError as e:
        say(f"could not write the stop request ({e}); stopping directly")
    for _ in range(3):
        if tmux("send-keys", "-t", f"={session}:", "C-c") == 0:
            break
        time.sleep(5)
    for _ in range(30):
        if tmux("has-session", "-t", f"={session}") == 1:
            say(f"launcher session {session} ended: stop confirmed")
            return
        time.sleep(10)
    tmux("kill-session", "-t", f"={session}")
    # run_pilot.py starts each inspect eval in its own session; on the dedicated canary devbox, stop them too
    kill_inspect_groups()
    ended = tmux("has-session", "-t", f"={session}") == 1
    say(f"launcher session {session} {'killed: stop confirmed' if ended else 'STILL LIVE: ALARM'}")


def kill_inspect_groups() -> None:
    """Every inspect eval on this dedicated devbox, by process group (run_pilot.py starts each in its own session): TERM,
    then KILL; the bracket keeps the pattern from matching this command's own arguments."""
    for sig in ("TERM", "KILL"):
        pids = subprocess.run(["pgrep", "-f", "[.]venv/bin/inspect eval"], capture_output=True, text=True).stdout.split()
        for pid in pids:
            subprocess.run(["kill", f"-{sig}", "--", f"-{pid}"])
        time.sleep(15)
    left = subprocess.run(["pgrep", "-f", "[.]venv/bin/inspect eval"], capture_output=True, text=True).stdout.split()
    say(f"inspect evals left: {len(left)}")


def load_reference(path: Path) -> dict:
    ref_all = json.loads(path.read_text())
    ref = ref_all["reference_summary"]
    assert ref_all["pass"] and ref["runs"] == ref["recorded"] > 0 and len(ref["given_hashes"]) == 1, "reference incomplete"
    assert all(len(v) == 1 for v in ref["metadata_values"].values()) and ref["model_configs"], "reference not uniform"
    ref_hash = next(iter(ref["given_hashes"]))
    assert g.canon(ref["given"]) == ref_hash, "the reference's given settings don't hash to its one hash"
    return {"all": ref_all, "given": ref["given"], "hash": ref_hash, "configs": set(ref["model_configs"]),
            "md": {k: set(v) for k, v in ref["metadata_values"].items()}}


def check_row(r: dict, ref: dict) -> list:
    bad = []
    if r["given_sha256"] != ref["hash"]:
        env, renv = r["given"]["env"], ref["given"]["env"]
        bad.append({"given": {k: (env.get(k), renv.get(k)) for k in sorted(set(env) | set(renv)) if env.get(k) != renv.get(k)}
                    | {k: (r["given"][k], ref["given"][k]) for k in ("flags", "tree") if r["given"][k] != ref["given"][k]}})
    if r["state"] == "done" and "recorded" not in r:
        bad.append({"recorded": "a done run without exactly one .eval"})
    for k, v in r.get("recorded", {}).get("md", {}).items():
        if ref["md"].get(k) != {json.dumps(v, sort_keys=True)}:
            bad.append({k: (v, sorted(ref["md"].get(k, [])))})
    for c in r.get("recorded", {}).get("configs", []):
        if json.dumps(c, sort_keys=True) not in ref["configs"]:
            bad.append({"simulator call config not in the reference": c})
    return bad


def harness(args) -> None:
    assert not args.out.exists(), f"{args.out} exists"
    ref = load_reference(args.reference_json)
    h = yaml.safe_load((args.harness / "config" / "run.yaml").read_text())["simulators"]["september"]
    mine = {"inputs_sha256": {p: g.sha((args.harness / p).read_bytes()) for p in g.INPUTS},
            "tree": {"tree": str(Path(h["tree"]).expanduser()), "src_sha256": h["src_sha256"]}}
    equal = [n for n, v in ref["all"]["harness"]["launches"].items()
             if v["inputs_sha256"] == mine["inputs_sha256"] and v["tree"] == mine["tree"]]
    print(f"harness {args.harness.resolve()}: {mine}; equal to reference launches {equal}")
    args.out.write_text(json.dumps({"reference_json": str(args.reference_json), "reference_given_sha256": ref["hash"],
                                    "harness": mine | {"equal_to": equal}, "pass": bool(equal)}, indent=1))
    print(f"SETTINGS GATE {'PASS' if equal else 'FAIL'} -> {args.out}")
    sys.exit(0 if equal else 1)


def watch(args) -> None:
    checked, report = {}, {"reference_json": str(args.reference_json), "new": str(args.new)}
    try:
        assert not args.out.exists(), f"{args.out} exists"
        ref = load_reference(args.reference_json)
        report["reference_given_sha256"] = ref["hash"]
        fails = 0
        while True:
            ended = tmux("has-session", "-t", f"={args.session}") == 1
            try:
                rec = json.loads((args.new / "launch.json").read_text())
                runs = g.runs_of(args.new, ("sim_attack", "sim_honest"))
                fails = 0
            except (FileNotFoundError, json.JSONDecodeError) as e:  # not written yet, or caught mid-write
                fails += 1
                print(f"{time.strftime('%T')} launch.json unreadable ({type(e).__name__}), {fails} in a row", flush=True)
                if ended or fails >= 10:
                    raise
                time.sleep(30)
                continue
            full = {r["key"]: r for r in rec["runs"]}
            for r in runs:
                attempts = full[r["run"]]["attempts"]
                for n in range(1, len(attempts) + 1):  # every attempt's given settings, a replaced one too
                    state = r["state"] if n == len(attempts) else "replaced"
                    k = f"{r['run']}#{n}"
                    if checked.get(k, {}).get("state") == state:
                        continue
                    gv = g.given(rec, {**full[r["run"]], "attempts": attempts[:n]})
                    row = {**r, "state": state, "attempt": n, "given": gv, "given_sha256": g.canon(gv)}
                    if state == "done":
                        g.add_recorded([row])
                    row["mismatches"] = check_row(row, ref)
                    checked[k] = row
                    if row["mismatches"]:
                        print(f"MISMATCH {k}: {json.dumps(row['mismatches'])}", flush=True)
                        raise SystemExit("settings mismatch")
            print(f"{time.strftime('%T')} {len(checked)} simulated attempts checked, "
                  f"{sum(r['state'] == 'done' for r in checked.values())} done with recorded settings: all EQUAL", flush=True)
            if ended:
                live = [r["key"] for r in rec["runs"] if not r["real"] and r["state"] in LIVE]  # every simulated row
                assert not live, f"the launcher ended but its final record shows non-terminal simulated runs: {live}"
                break
            time.sleep(POLL_S)
        assert checked, "no simulated run started"
        report |= {"attempts": list(checked.values()), "pass": True}
        args.out.write_text(json.dumps(report, indent=1, default=str))
    except BaseException as e:  # every failure, setup included, goes through the stop
        report |= {"attempts": list(checked.values()), "pass": False, "error": f"{type(e).__name__}: {e}",
                   "where": traceback.format_exc().splitlines()[-3:]}
        stop(args.session, args.new)  # first: nothing below may keep it from running
        say(f"SETTINGS GATE FAIL: {report['error']}")
        try:
            Path(f"{args.out}.failed.{int(time.time())}").write_text(json.dumps(report, indent=1, default=str))
        except Exception:  # noqa: BLE001
            pass
        sys.exit(1)
    print(f"SETTINGS GATE PASS -> {args.out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference-json", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--harness", type=Path)
    mode.add_argument("--new", type=Path)
    ap.add_argument("--session", help="the launcher's tmux session (required with --new)")
    args = ap.parse_args()
    if args.harness:
        harness(args)
    assert args.session, "--new needs --session"
    watch(args)


if __name__ == "__main__":
    main()
