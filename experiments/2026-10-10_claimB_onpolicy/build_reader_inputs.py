"""Follow-up (c), staging: reader inputs for Qwen3.5-27B's 196-task September scale-up, with the frozen round-one and
round-two rules applied once to the simulated arm (fixed-sim PLAN.md amendment 6; RULES.md). Free, local, no model.

Runs (S3 phase2/runs, read from the local cache; the strict rerun's missing files fetched through transcripts._get):
  strict rerun t01–t20, canary t21–t25, launch 2 t26–t185, top-up t186–t196 (real attack run 1, the September simulated
  attack run, honest_a for the system prompt) and the second-attack launch (real attack run 2). Per (launch, arm, task),
  the canonical run is the last attempt in launch.json whose outcome is eval_status ['success'] with no sample errors and
  one sample; a task covered by two launches takes the one with such an attempt (both: an error, listed).
Rows, in the staged-inputs schema (`key` = <file stem>__<sample id>__<epoch>, cell = main:side as claim B):
  real-attack-r1, real-attack-r2, sim-attack-sept (the simulator's text), sim-attack-sept-fixed (R1, R2, R5 from fixes.py,
  then R8 from fixes_round2.py; not R6, whose "now" window and donors cannot fit runs dated 10-09/10, PLAN.md (c)); only
  tasks complete in all three. Messages: transcripts.reader_messages with the honest system prompt of the same task.
  `tokens`/`truncated`: Qwen3.5-27B's tokenizer (the reader), 32k cut. Rule files are checked against RULES.md's pins;
  the same rules are dry-run over both real arms, per rule: outputs considered and changed, with each changed one's task
  and message index (no text). A real output a rule would change is reported, not hidden (on LA it was 0 for every rule).
Examples (gate (a)'s TF-IDF input): q1's stage_run at the Qwen3-32B tokenizer, as the claim B examples were staged, for
real-attack-r1, real-attack-r2 and sim-attack-sept-fixed. Writes data/<RUN>/{inputs,examples,records,manifest.json}
(git-ignored) and results/build_counts_<stamp>.json (counts only).

  HF_HUB_OFFLINE=1 PYTHONPATH=src uv run --no-project --with transformers==4.56.2 --with jinja2 --with boto3 --with pyyaml \
      python experiments/2026-10-10_claimB_onpolicy/build_reader_inputs.py
"""

import gzip
import hashlib
import io
import json
import sys
import zipfile
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).parent
REPO = HERE.parents[1]
FS = REPO / "experiments" / "2026-10-09_fixed_sim"
sys.path[:0] = [str(FS), str(REPO / "experiments" / "2026-10-06_q1_text_baselines")]
import fixes  # noqa: E402
import fixes_round2  # noqa: E402
from run_stage_examples import raw_outputs, side_task_id, stage_run  # noqa: E402

from probe_shift.models import QWEN3_32B, QWEN35_27B  # noqa: E402
from probe_shift.reader import segment_end_positions  # noqa: E402
from probe_shift.transcripts import CACHE, _get, invalidity, main_task_id, reader_messages, submit_calls, system_prompt  # noqa: E402

RUN, BUCKET, MAX_TOKENS, SHARDS = "2026-10-10_la_september-onpolicy_v1", "individual-aswin-surya-probe-shift-kill-2027-04-06", 32768, 20
RUNS = CACHE / BUCKET / "phase2" / "runs"
LAUNCHES = {"2026-10-09_000523_rerun_strict": ("attack", "sim_attack", "honest_a"), "2026-10-09_173815_scaleup_canary": ("attack", "sim_attack", "honest_a"),
            "2026-10-09_182505_scaleup": ("attack", "sim_attack", "honest_a"), "2026-10-10_001314_scaleup_topup": ("attack", "sim_attack", "honest_a"),
            "2026-10-10_001314_second_attack": ("attack",)}
ROLE = {("2026-10-10_001314_second_attack", "attack"): "r2"}  # every other launch's attack arm is run 1
PINS = {FS / "fixes.py": "b1ba4a4e574e964c2ab2445066bf57c1b36b367959f2edc0e36454b26bd58655",
        REPO / "experiments/2026-10-07_q2_rendering_swap/rerender.py": "45662fb3f4bb8f9e924123f2c841393e705843c6d03b41b039a5784505899a01",
        REPO / "experiments/2026-10-07_q2_rendering_swap/str_replace_tells.py": "320e88a52d38425086c873c669efa1422ee8b555cba404dd7807495474ce2dc5",
        REPO / "experiments/2026-10-06_q1_text_baselines/run_stage_examples.py": "2dd5ded90a282b565c894a202525940f0a777d39eda879c065e645ceae78de63",
        FS / "fixes_round2.py": "381c5fe22a962c8695394904e61745c2242830b4a88fd05a3547c2daff9917cf"}
ROUND1, ROUND2 = ("R1", "R2", "R5"), ["R8"]


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def valid(a: dict) -> bool:
    o = a.get("outcome") or {}
    return o.get("eval_status") == ["success"] and o.get("sample_errors") == [] and o.get("samples") == 1


def canonical(launch: str, arms: tuple[str, ...]) -> dict:
    """{(role, task): (eval key under RUNS's bucket prefix, attempt index, n attempts)} for the launch's chosen arms."""
    meta = json.loads((RUNS / launch / "launch.json").read_bytes() if (RUNS / launch / "launch.json").exists()
                      else _get(BUCKET, f"phase2/runs/{launch}/launch.json"))
    files = json.loads(_get(BUCKET, f"phase2/runs/{launch}.manifest_v1.json"))["files"]
    by_dir = {}
    for f in files:
        if f["key"].endswith(".eval"):
            by_dir.setdefault(f["key"].rsplit("/", 1)[0], []).append(f)
    out = {}
    for r in meta["runs"]:
        if r["arm"] not in arms or (r["arm"] == "sim_attack" and r.get("simulator") != "september"):
            continue
        ok = [(i, a) for i, a in enumerate(r.get("attempts") or []) if valid(a)]
        if r["state"] != "done" or not ok:
            continue
        i, a = ok[-1]
        d = f"phase2/runs/{launch}/" + a["log_dir"].split(f"/{launch}/", 1)[1]
        evals = by_dir.get(d, [])
        if len(evals) != 1:
            raise ValueError(f"{d}: {len(evals)} .eval files in the manifest")
        role = {"sim_attack": "sim", "honest_a": "honest"}.get(r["arm"], ROLE.get((launch, r["arm"]), "r1"))
        out[(role, r["task"])] = (evals[0]["key"], evals[0]["sha256"], i, len(r["attempts"]))
    return out


def sample_of(key: str, want_sha: str) -> tuple[str, dict]:
    body = _get(BUCKET, key)
    if sha(body) != want_sha:
        raise ValueError(f"{key}: sha256 differs from its launch manifest")
    with zipfile.ZipFile(io.BytesIO(body)) as z:
        names = [n for n in z.namelist() if n.startswith("samples/") and n.endswith(".json")]
        if len(names) != 1:
            raise ValueError(f"{key}: {len(names)} samples")
        return key.rsplit("/", 1)[1], json.loads(z.read(names[0]))


def main():
    from transformers import AutoTokenizer

    for p, h in PINS.items():
        if sha(p.read_bytes()) != h:
            raise ValueError(f"{p.relative_to(REPO)}: sha256 differs from RULES.md's pin")
    tok35 = AutoTokenizer.from_pretrained(QWEN35_27B["name"], revision=QWEN35_27B["revision"])
    tok32 = AutoTokenizer.from_pretrained(QWEN3_32B["name"], revision=QWEN3_32B["revision"])
    chosen, source, count = {}, {}, Counter()
    for launch, arms in LAUNCHES.items():
        for (role, task), v in canonical(launch, arms).items():
            if (role, task) in chosen:
                raise ValueError(f"{role} {task}: valid runs in {source[(role, task)]} and {launch}")
            chosen[(role, task)], source[(role, task)] = v, launch
            count[f"{launch}|{role}|chosen_attempt_{'first' if v[2] == 0 else 'later'}"] += 1
    tasks = sorted({t for _, t in chosen}, key=lambda t: int(t[1:].split("_")[0]))
    complete = [t for t in tasks if all((r, t) in chosen for r in ("r1", "r2", "sim", "honest"))]
    count["tasks_seen"], count["tasks_complete"] = len(tasks), len(complete)
    missing = {t: sorted(r for r in ("r1", "r2", "sim", "honest") if (r, t) not in chosen) for t in tasks if t not in complete}
    out_dir = HERE / "data" / RUN
    if out_dir.exists():
        raise SystemExit(f"{out_dir} exists; staging is done once")
    rows, examples, records, dry_changes = [], {"outputs": [], "calls": [], "runs": []}, [], []
    for t in complete:
        _, honest = sample_of(*chosen[("honest", t)][:2])
        system = system_prompt(honest)
        for role, arm in (("r1", "real-attack-r1"), ("r2", "real-attack-r2"), ("sim", "sim-attack-sept")):
            file, s = sample_of(*chosen[(role, t)][:2])
            bad = invalidity(s)
            count[f"{arm}|invalid_{bad}"] += bad is not None
            side, env = side_task_id(s), (s.get("metadata") or {}).get("environment")
            env = env.get("id") if isinstance(env, dict) else env  # the scale-up's metadata holds the environment record
            if main_task_id(s) != main_task_id(honest) or not isinstance(env, str) or not t.endswith(f"_{env}"):
                raise ValueError(f"{t}: {arm}'s main task or environment disagrees with honest_a's or the task name")
            msgs = reader_messages(s, system)
            n_tool = sum(m["role"] == "tool" for m in msgs)
            if n_tool != len(raw_outputs(s)):
                raise ValueError(f"{t} {arm}: {n_tool} reader outputs vs {len(raw_outputs(s))} non-submit tool messages")
            meta = {"key": f"{file.removesuffix('.eval')}__{s['id'].replace('/', '_')}__{s['epoch']}", "arm": arm,
                    "cohort": f"phase2_september_scaleup_{role}", "file": file, "sample_id": s["id"], "epoch": s["epoch"],
                    "environment": env, "main_task": main_task_id(s), "side_task": side,
                    "cell": f"{main_task_id(s)}:{side}", "sim_reference": None, "task": t, "launch": source[(role, t)],
                    "n_submit_calls": submit_calls(s)}
            versions = [(arm, msgs)]
            if role == "sim":
                new, recs = fixes.apply(msgs, ROUND1)
                new, recs2 = fixes_round2.apply(new, ROUND2)
                recs += recs2
                ch = {x["msg"] for x in recs if x["changed"]}
                if any(new[i] != m for i, m in enumerate(msgs) if i not in ch) or any(msgs[i]["role"] != "tool" for i in ch):
                    raise ValueError(f"{t}: an unrecorded or non-tool change")
                records += [{"key": meta["key"], "task": t, "environment": meta["environment"]} | x for x in recs]
                count["sim-attack-sept-fixed|outputs_changed"] += len(ch)
                versions.append(("sim-attack-sept-fixed", new))
            else:  # the same rules over the real arm, one at a time: what each would change
                dpkg = [c for c in fixes.calls_of(msgs) if c["tool"] != "text_editor" and fixes.DPKG_MARKER in str(msgs[c["msg"]]["content"])]
                count[f"{arm}|dpkg_outputs"] += len(dpkg)  # R5 records only what it changes; these are what it looks at
                count[f"{arm}|dpkg_outputs_with_cr"] += sum("\r" in str(msgs[c["msg"]]["content"]) for c in dpkg)
                for name in ROUND1 + tuple(ROUND2):
                    _, recs = (fixes.RULES[name] if name in fixes.RULES else fixes_round2.RULES[name])(msgs)
                    count[f"{arm}|dry_run|{name}|records"] += len(recs)
                    count[f"{arm}|dry_run|{name}|changed"] += sum(x["changed"] for x in recs)
                    dry_changes += [{"task": t, "arm": arm, "rule": name, "msg": x["msg"], "category": x["category"]} for x in recs if x["changed"]]
            for a, m in versions:
                seg = segment_end_positions(tok35, m, MAX_TOKENS)
                rows.append(meta | {"arm": a, "tokens": seg["n_tokens"], "truncated": seg["truncated"], "messages": m})
                if a != "sim-attack-sept":  # q1-style examples for TF-IDF, at Qwen3-32B's tokenizer as claim B's gate (a)
                    _, o, c, run = stage_run(tok32, m, MAX_TOKENS, {k: meta[k] for k in ("key", "file", "sample_id", "epoch", "environment",
                                                                                         "main_task", "cell", "sim_reference")} | {"arm": a})
                    examples["outputs"] += o
                    examples["calls"] += c
                    examples["runs"].append(run | {"environment": meta["environment"]})
                count[f"{a}|runs"] += 1
    keys = [(r["arm"], r["key"]) for r in rows]
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate keys")
    for x in records:
        count[f"rule|{x['rule']}|{x['category']}|{x['changed']}"] += 1
        if x["changed"]:
            count[f"rule|{x['rule']}|changed|{x['environment']}"] += 1
    rows.sort(key=lambda r: (-r["tokens"], r["arm"], r["key"]))  # longest first, dealt round-robin: balanced shards
    files = {f"inputs/shard_{i}.jsonl": "".join(json.dumps(r) + "\n" for r in rows[i::SHARDS]).encode() for i in range(SHARDS)}
    files["examples/outputs.jsonl.gz"] = gzip.compress("".join(json.dumps(x) + "\n" for x in examples["outputs"]).encode(), mtime=0)
    files["examples/calls.jsonl.gz"] = gzip.compress("".join(json.dumps(x) + "\n" for x in examples["calls"]).encode(), mtime=0)
    files["examples/runs.jsonl"] = "".join(json.dumps(x) + "\n" for x in examples["runs"]).encode()
    files["records/records.jsonl.gz"] = gzip.compress("".join(json.dumps(x) + "\n" for x in records).encode(), mtime=0)
    for name, b in files.items():
        (out_dir / name).parent.mkdir(parents=True, exist_ok=True)
        (out_dir / name).write_bytes(b)
    manifest = {"run": RUN, "files": {k: sha(v) for k, v in sorted(files.items())}}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    res = {"run": RUN, "manifest_sha256": sha((out_dir / "manifest.json").read_bytes()), "rules": {"round1": ROUND1, "round2": ROUND2},
           "pins_checked": {str(p.relative_to(REPO)): h for p, h in PINS.items()}, "incomplete_tasks": missing,
           "real_dry_run_changes": dry_changes,
           "task_source": {t: {r: source[(r, t)] for r in ("r1", "r2", "sim", "honest")} for t in complete},
           "counts": dict(sorted(count.items())), "bytes": {k: len(v) for k, v in files.items()}}
    (HERE / "results").mkdir(exist_ok=True)
    with open(HERE / "results" / f"build_counts_{stamp}.json", "x") as fh:
        fh.write(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k not in ("task_source", "bytes")}, indent=1))
    print("->", HERE / "results" / f"build_counts_{stamp}.json")


if __name__ == "__main__":
    main()
