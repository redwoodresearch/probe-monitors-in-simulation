"""Settings-equality gate (RULE.md, amendment 4): a new launch's simulated runs must have exactly the simulator settings
of the existing on-policy simulated attack runs (the reference: every `sim_attack@september` run of --reference).

Two layers, hashed per run:
- given (launch.json, any started run): the attempt's env_vars minus BY_DESIGN (mode, per-task and per-server keys),
  its inspect flags (--log-dir's value dropped) and the simulator tree (path, src_sha256);
- recorded (.eval, done runs): the sample's simulator metadata (SIM_MD) and every distinct (model, generate config) of
  the simulator's model calls (effort, temperature, seed, max tokens, timeout, retries; long strings as sha256).
Pre-launch, --harness compares the inputs the settings are resolved from (config/sim.yaml and config/run.yaml by sha256,
and run.yaml's pinned tree) with every reference launch's record: they must equal at least one reference launch's (a
launch whose inputs differ is reported; its runs count only through the reference's one given hash). With --new, it
exits 1 unless every new run's given hash is the reference's one given hash and every done new run's recorded settings
are the reference's (one value per metadata field; per model, a config the reference used). It prints keys, values and
hashes only, never transcript text; the reference source (BA_REAL_LOG: the task's own Qwen3.5 honest_a) differs by
design and is recorded per run.

Runs ON THE DEVBOX with the September snapshot's python:
  $PY check_sim_settings.py --reference ~/phase2/runs/<launch> ... --out <json> [--harness <frozen harness dir>]
      --expect-reference 191 [--new ~/phase2/runs/<new launch> ... [--wait-s 1800]]
At a launch's start (--wait-s) and again when it has finished (final validation: every done run's recorded settings);
any nonzero exit is a failure.
"""

import argparse
import hashlib
import json
import re
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import yaml

POLICY = "vllm/Qwen/Qwen3.5-27B"
BY_DESIGN = {"BA_MODE", "BA_CELLS", "PHASE2_CELLS", "BA_MAINS", "LA_ENV", "BA_REAL_LOG", "BA_HONEST_LOG",
             "PHASE2_DIGEST_SET", "VLLM_BASE_URL"}
SIM_MD = ("simulator_model", "simulator_fallback_model", "simulator_refusal_judge_model",
          "simulator_router_judge_model", "orig_trajectory_mode", "orig_trajectory_out_cap", "codebase_exec",
          "exec_source_authority", "session_notes", "self_critique")
INPUTS = ("config/sim.yaml", "config/run.yaml")


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def canon(x) -> str:
    return sha(json.dumps(x, sort_keys=True, default=str).encode())


def short(v):
    return f"sha256:{sha(v.encode())}" if isinstance(v, str) and len(v) > 200 else v


def given(rec: dict, r: dict) -> dict:
    a = r["attempts"][-1]
    assert not [k for k in a["env_vars"] if re.search(r"API_KEY|SECRET|PASSWORD|_TOKEN$", k)], "credential-like key"
    cmd = a["cmd"]
    i = cmd.index("--log-dir")
    t = rec["trees"][r["simulator"]]
    return {"env": {k: v for k, v in a["env_vars"].items() if k not in BY_DESIGN},
            "flags": cmd[i + 2:], "tree": {"tree": t["tree"], "src_sha256": t["src_sha256"]}}


def recorded(path: str) -> dict:
    from inspect_ai.event import ModelEvent
    from inspect_ai.log import read_eval_log

    (s,) = read_eval_log(path, resolve_attachments=True).samples
    md = {k: s.metadata.get(k) for k in SIM_MD}
    md["fork_fraction"] = (s.metadata.get("fork") or {}).get("fraction")
    md["source_tree"] = {k: (s.metadata.get("source_tree") or {}).get(k) for k in ("commit", "dirty")}
    configs = sorted({json.dumps([e.model, {k: short(v) for k, v in e.config.model_dump(exclude_none=True).items()}],
                                 sort_keys=True)
                      for e in s.events if isinstance(e, ModelEvent) and e.model != POLICY})
    return {"md": md, "configs": [json.loads(c) for c in configs]}


def runs_of(launch: Path, arms: tuple) -> list:
    rec = json.loads((launch / "launch.json").read_text())
    out = []
    for r in rec["runs"]:
        if r["real"] or r["arm"] not in arms or r["simulator"] != "september" or not r["attempts"]:
            continue
        row = {"launch": launch.name, "run": r["key"], "arm": r["arm"], "task": r["task"], "state": r["state"],
               "reference_log": Path(r["attempts"][-1]["env_vars"].get("BA_REAL_LOG", "")).name}
        g = given(rec, r)
        row |= {"given": g, "given_sha256": canon(g)}
        d = Path(r["attempts"][-1]["log_dir"])
        evals = sorted((launch / d.parent.name / d.name).glob("*.eval")) if r["state"] == "done" else []
        row["eval"] = str(evals[0]) if len(evals) == 1 else None
        out.append(row)
    return out


def add_recorded(rows: list) -> None:
    todo = [r for r in rows if r["eval"]]
    with ProcessPoolExecutor(max_workers=16) as pool:
        for r, rc in zip(todo, pool.map(recorded, [r["eval"] for r in todo])):
            r["recorded"] = rc


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--reference", nargs="+", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--harness", type=Path, help="the frozen harness folder a new launch resolves its settings from")
    ap.add_argument("--new", nargs="*", type=Path, default=[])
    ap.add_argument("--wait-s", type=int, default=0, help="wait this long for each --new launch's first started run")
    ap.add_argument("--expect-reference", type=int, required=True, help="reference runs, each with its .eval (191)")
    args = ap.parse_args()
    assert not args.out.exists(), f"{args.out} exists"
    ok, report = True, {"reference": [str(p) for p in args.reference], "new": [str(p) for p in args.new]}

    ref = [r for p in args.reference for r in runs_of(p, ("sim_attack",))]
    add_recorded(ref)
    ref_given = Counter(r["given_sha256"] for r in ref)
    ref_md = defaultdict(set)
    for r in ref:
        for k, v in r.get("recorded", {}).get("md", {}).items():
            ref_md[k].add(json.dumps(v, sort_keys=True))
    ref_configs = {json.dumps(c, sort_keys=True) for r in ref for c in r.get("recorded", {}).get("configs", [])}
    report["reference_summary"] = {
        "runs": len(ref), "recorded": sum("recorded" in r for r in ref), "given_hashes": dict(ref_given),
        "given": next(r["given"] for r in ref if r["given_sha256"] == ref_given.most_common(1)[0][0]),
        "metadata_values": {k: sorted(v) for k, v in ref_md.items()},
        "model_configs": sorted(ref_configs)}
    print(f"reference: {len(ref)} sim_attack@september runs, {report['reference_summary']['recorded']} with a "
          f".eval; given-settings hashes {dict(ref_given)}")
    for k, v in sorted(ref_md.items()):
        print(f"  {k}: {sorted(v)}")
    for c in sorted(ref_configs):
        print(f"  simulator call config: {c}")
    if len(ref_given) != 1 or any(len(v) != 1 for v in ref_md.values()):
        ok = False
        print("REFERENCE NOT UNIFORM: the existing runs disagree among themselves")
    recorded_n = report["reference_summary"]["recorded"]
    if len(ref) != args.expect_reference or recorded_n != args.expect_reference:
        ok = False
        print(f"REFERENCE INCOMPLETE: {len(ref)} runs, {recorded_n} with one .eval; expected {args.expect_reference}")

    if args.harness:
        h = yaml.safe_load((args.harness / "config" / "run.yaml").read_text())["simulators"]["september"]
        pinned = {"tree": str(Path(h["tree"]).expanduser()), "src_sha256": h["src_sha256"]}
        files = {p: sha((args.harness / p).read_bytes()) for p in INPUTS}
        report["harness"] = {"inputs_sha256": files, "tree": pinned, "launches": {}}
        print(f"harness {args.harness}: tree {pinned}")
        for p in args.reference:
            rec = json.loads((p / "launch.json").read_text())
            mine = {f: rec["input_sha256"].get(f) for f in INPUTS}
            tree = {k: str(rec["trees"]["september"][k]) for k in ("tree", "src_sha256")}
            same = mine == files and tree == pinned
            report["harness"]["launches"][p.name] = {"inputs_sha256": mine, "tree": tree, "equal": same}
            print(f"  {p.name}: inputs and tree {'EQUAL' if same else 'DIFFER'} "
                  f"{'' if same else {f: (mine[f], files[f]) for f in INPUTS if mine[f] != files[f]}}")
        # A launch with other input files whose runs still resolved to the reference's one hash is reported, not
        # failed: the frozen inputs must equal at least one reference launch's, and the reference must be uniform.
        ok &= any(v["equal"] for v in report["harness"]["launches"].values())

    if args.new:
        deadline = time.time() + args.wait_s
        while True:  # each new launch must have started at least one simulated run
            new = [r for p in args.new for r in runs_of(p, ("sim_attack", "sim_honest"))]
            if {r["launch"] for r in new} == {p.name for p in args.new} or time.time() > deadline:
                break
            time.sleep(30)
        add_recorded(new)
        report["new_runs"] = new
        missing = sorted({p.name for p in args.new} - {r["launch"] for r in new})
        if missing:
            ok = False
            print(f"NEW LAUNCH WITHOUT A STARTED SIMULATED RUN: {missing}")
        for r in new:
            bad = []
            ref_g = report["reference_summary"]["given"]
            if r["given_sha256"] not in ref_given or len(ref_given) != 1:
                env, renv = r["given"]["env"], ref_g["env"]
                diff = {k: (env.get(k), renv.get(k)) for k in sorted(set(env) | set(renv)) if env.get(k) != renv.get(k)}
                diff |= {k: (r["given"][k], ref_g[k]) for k in ("flags", "tree") if r["given"][k] != ref_g[k]}
                bad.append({"given": diff})
            if r["state"] == "done" and not r["eval"]:  # complete evidence: a done run must have exactly one .eval
                bad.append({"recorded": "done without exactly one .eval"})
            for k, v in r.get("recorded", {}).get("md", {}).items():
                if ref_md.get(k) != {json.dumps(v, sort_keys=True)}:
                    bad.append({k: (v, sorted(ref_md.get(k, [])))})
            for c in r.get("recorded", {}).get("configs", []):
                if json.dumps(c, sort_keys=True) not in ref_configs:
                    bad.append({"simulator call config not in the reference": c})
            r["mismatches"] = bad
            print(f"  new {r['launch']} {r['run']}: given {'EQUAL' if not bad else 'see below'}"
                  f"{', recorded checked' if 'recorded' in r else ''}; reference log {r['reference_log']}")
            for b in bad:
                print(f"    MISMATCH {b}")
            ok &= not bad
    report["pass"] = ok
    args.out.write_text(json.dumps(report, indent=1, default=str))
    print(f"SETTINGS GATE {'PASS' if ok else 'FAIL'} -> {args.out}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
