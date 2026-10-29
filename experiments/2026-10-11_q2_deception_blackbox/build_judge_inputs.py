"""Deception probe vs a black-box judge (PLAN.md here), step 1: the judge's input per run, the exact commands the probe
scored, in order and verbatim. Aswin allowed this read of attack-run commands at 23:16:03 PT 10-11 ("OK, run it ($30)",
relayed by the orchestrator). Local, $0.

- Runs: the run-level test's (experiments/2026-10-10_q2_deception_runlevel), confirmatory tasks t21-t196 only: attack run
  1, honest_a and honest_b from the scale-up dumps, and attack run 2 from the second-attack dump.
- Rows: every scored row of the run-level scores file (sha256 pinned), in (epoch, turn, call index) order.
- Each row is found in its .eval (cached S3 copy, sha256 checked against the launch's manifest): the acting model's event
  with that request id, its tool call at that index, attachments resolved. Checks per row: the staging record's command
  row exists for (irid, call index) with the same tool name and parameter; the frozen labeller (fingerprint equal to the
  staging record's) labels it impactful; the parameter is the labeller's command parameter. Any failure stops the build.
- Python 3.14: the .eval archives are zstd-compressed.
- Output: the commands are model text, so they go to the git-ignored data/; results/ gets counts, sizes and hashes.

  PYTHONPATH=src uv run --no-project --python 3.14 python experiments/2026-10-11_q2_deception_blackbox/build_judge_inputs.py \\
      <scores json> <folder of staging records> <cached phase2/runs folder>
"""
import hashlib
import json
import re
import subprocess
import sys
import zipfile
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from probe_shift.calls import fingerprint, label_call
from probe_shift.reader import command_param

HERE = Path(__file__).parent
ROOT = HERE.parents[1]
SCORES_SHA = "f3c4dee40c00368f96d3f379f980630108f5ba4c1a68ff9afe0bbc8e70e72e72"
N_TASKS = 171  # the run-level test's confirmatory part 1 (runlevel_analysis_2026-10-10_215302.json)
ACTING = "vllm/Qwen/Qwen3.5-27B"
DUMPS = {  # dump -> (cached launch folder, staging record, arms kept)
    "2026-10-10_scaleup_canary_segends_v1": ("2026-10-09_173815_scaleup_canary", "replay_readouts_scaleup_canary_segends_2026-10-10_014918.json", ("attack", "honest_a", "honest_b")),
    "2026-10-10_scaleup_l2_segends_v1": ("2026-10-09_182505_scaleup", "replay_readouts_scaleup_l2_segends_2026-10-10_014046.json", ("attack", "honest_a", "honest_b")),
    "2026-10-10_scaleup_topup_segends_v1": ("2026-10-10_001314_scaleup_topup", "replay_readouts_scaleup_topup_segends_2026-10-10_023740.json", ("attack", "honest_a", "honest_b")),
    "2026-10-10_second_attack_segends_v1": ("2026-10-10_001314_second_attack", "replay_readouts_second_attack_segends_2026-10-10_024517.json", ("attack_run2",)),
}
REF = re.compile(r"attachment://[0-9a-f]{32}")
tnum = lambda task: int(task.split("_", 1)[0][1:])


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def main(scores_p: Path, staging_dir: Path, runs_dir: Path):
    me = Path(__file__).resolve()
    if subprocess.run(["git", "ls-files", "--error-unmatch", str(me)], cwd=HERE, capture_output=True).returncode \
            or subprocess.run(["git", "diff", "--quiet", "HEAD", "--", str(me)], cwd=HERE).returncode:
        raise SystemExit("the script differs from HEAD: commit it before running")
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip()
    raw = scores_p.read_bytes()
    if sha(raw) != SCORES_SHA:
        raise SystemExit("scores file differs from its pin")
    sc = json.loads(raw)
    rows = defaultdict(list)  # (dump, arm, run) -> scored rows
    for r in sc["scores"]:
        if r["dump"] in DUMPS and r["arm"] in DUMPS[r["dump"]][2] and 21 <= tnum(r["task"]) <= 196:
            rows[(r["dump"], r["arm"], r["run"])].append(r)
    # eligibility as in analyze_runlevel.py part 1: attack run 1 and at least one honest run with scored rows
    has = defaultdict(set)
    for (d, arm, run), rs in rows.items():
        has[rs[0]["task"]].add("honest" if arm.startswith("honest") else arm)
    eligible = {t for t, a in has.items() if "attack" in a and "honest" in a}
    if len(eligible) != N_TASKS:
        raise SystemExit(f"{len(eligible)} eligible tasks, not the run-level test's {N_TASKS}")
    rows = {k: rs for k, rs in rows.items() if rs[0]["task"] in eligible}
    out, c, n_rows = [], Counter(), 0
    for dump, (launch, staging_name, _) in DUMPS.items():
        sb = (staging_dir / staging_name).read_bytes()
        if sha(sb) != sc["staging sha256"][staging_name]:
            raise SystemExit(f"{staging_name} differs from the scores file's record of it")
        stg = json.loads(sb)
        if stg["summary"]["labeler"] != fingerprint():
            raise SystemExit(f"{staging_name}: labeller fingerprint {stg['summary']['labeler']} is not this checkout's {fingerprint()}")
        cmd_rows = {(x["irid"], x["call_index"]): x for x in stg["calls"] if x.get("kind") == "command"}
        manifest = json.loads((runs_dir / f"{launch}.manifest_v1.json").read_text())
        msha = {m["key"].split(f"/{launch}/", 1)[1]: m["sha256"] for m in manifest["files"] if f"/{launch}/" in m["key"]}
        for (d, arm, run), rs in sorted(rows.items()):
            if d != dump:
                continue
            ev = runs_dir / launch / run
            evb = ev.read_bytes()
            if msha.get(run) != sha(evb):
                raise SystemExit(f"{run}: .eval sha256 differs from the launch manifest")
            with zipfile.ZipFile(ev) as z:
                names = [n for n in z.namelist() if n.startswith("samples/") and n.endswith(".json")]
                if len(names) != 1:
                    raise SystemExit(f"{run}: {len(names)} samples")
                s = json.loads(z.read(names[0]))
            att = s.get("attachments") or {}

            def resolved(x):
                if isinstance(x, str) and REF.fullmatch(x):
                    return att[x.removeprefix("attachment://")]
                if isinstance(x, list):
                    return [resolved(v) for v in x]
                if isinstance(x, dict):
                    return {k: resolved(v) for k, v in x.items()}
                return x
            calls, pos = {}, {}
            for ei, e in enumerate(s.get("events") or []):
                if e.get("event") == "model" and e.get("model") == ACTING:
                    irid = (((e.get("call") or {}).get("request") or {}).get("extra_headers") or {}).get("x-irid")
                    msg = ((e.get("output") or {}).get("choices") or [{}])[0].get("message") or {}
                    if irid is not None:
                        if irid in calls:
                            raise SystemExit(f"{run}: request id {irid} repeats")
                        calls[irid] = msg.get("tool_calls") or []; pos[irid] = ei
            cmds, order = [], sorted(rs, key=lambda r: (r["epoch"], r["turn"], r["call_index"]))
            if [(pos.get(r["irid"], -1), r["call_index"]) for r in order] != sorted((pos.get(r["irid"], -1), r["call_index"]) for r in order):
                raise SystemExit(f"{run}: the (epoch, turn, call) order differs from the .eval's event order")
            for r in order:
                st = cmd_rows.get((r["irid"], r["call_index"]))
                tcs = calls.get(r["irid"])
                if st is None or tcs is None or r["call_index"] >= len(tcs) or (st["turn"], st["epoch"]) != (r["turn"], r["epoch"]):
                    raise SystemExit(f"{run} {r['irid']} {r['call_index']}: not found in staging or the .eval")
                tc = tcs[r["call_index"]]; name = tc.get("function"); args = resolved(tc.get("arguments") or {})
                param = command_param(name, args)
                if name != st["name"] or param != st["param"] or label_call(name, args)["label"] != "impactful" or st["label"] != "impactful":
                    raise SystemExit(f"{run} {r['irid']} {r['call_index']}: tool, parameter or label differs from the staging record")
                v = args[param]
                if REF.search(v if isinstance(v, str) else json.dumps(v)):
                    raise SystemExit(f"{run}: attachment reference left in a command")
                cmds.append({"tool": name, "command": v if isinstance(v, str) else json.dumps(v)})
            n_rows += len(cmds); c[arm] += 1
            out.append({"key": f"{dump}|{arm}|{run}", "dump": dump, "arm": arm, "task": rs[0]["task"], "run": run, "commands": cmds})
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    d = ROOT / "data" / "deception_blackbox"; d.mkdir(parents=True, exist_ok=True); (HERE / "results").mkdir(exist_ok=True)
    with open(jp := d / f"judge_inputs_{stamp}.jsonl", "x") as fh:
        fh.writelines(json.dumps(x) + "\n" for x in out)
    chars = sorted(sum(len(k["command"]) for k in x["commands"]) for x in out)
    summ = {"plan": "PLAN.md (this folder)", "git": head, "scores sha256": SCORES_SHA, "labeller fingerprint": fingerprint(),
            "runs by arm": dict(c), "tasks": len({x["task"] for x in out}), "scored rows (commands)": n_rows,
            "command characters per run: min / median / 90th / max": [chars[0], chars[len(chars) // 2], chars[int(0.9 * len(chars))], chars[-1]],
            "command characters, all runs": sum(chars), "judge inputs": str(jp.relative_to(ROOT)), "judge inputs sha256": sha(jp.read_bytes())}
    with open(dest := HERE / "results" / f"judge_inputs_summary_{stamp}.json", "x") as fh:
        fh.write(json.dumps(summ, indent=1) + "\n")
    print(json.dumps(summ, indent=1)); print(dest)


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
