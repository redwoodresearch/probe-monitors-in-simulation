"""Fixed-sim on Modal CPU, reading the probe's reader inputs (Volume `…sub-la-activations`,
`2026-10-06_la_segment-end_qwen3-32b_v1/inputs/`), for the 939 paired attack cells (splits/exclusions_v4.json).

  ::checks  pre-freeze checks, nothing written to the data:
            (a) R1, R2 and R5 run over the REAL runs (e1 and e2) as a dry run: outputs changed per rule, by dev / test;
            (b) R1 on the simulated runs: inserts that gain the "Review the changes…" note, plus 3 seeded dev examples;
            (c) the clock rule (R6) on dev simulated runs: timestamps shifted / outside the window / unparsed, and runs
                where an untouched assistant message or tool call names a timestamp that a shifted output prints;
            (d) R2 and R5 on dev simulated runs: outputs per category, and sign-first diffs R2 left unclassified.
  ::apply   the one application of the frozen rules (RULES.md), after the freeze commit: every sha256 RULES.md lists is
            checked on the laptop and again in each container, and the work tree must be clean for them. Per shard
            (one container each), the simulated runs of the 939 cells get R1, R2, R5 (fixes.apply), then R6 (clock.py,
            donors = real epoch-1 runs of the dev environments, "now" read from the uncorrected outputs as in the
            checks); real rows are copied byte for byte. Writes, never overwriting, to the Volume run
            RUN_OUT: inputs/shard_<i>.jsonl (the reader-input schema; simulated rows get recounted `tokens` and
            `truncated` and a `fixed_sim` record), examples/ (q1's stage_run per output and call, for TF-IDF),
            records/ (one row per output a rule looked at), and manifest.json (sha256 of every file).
  Results come back to results/ as JSON.

  cd <repo worktree> && uv run modal run experiments/2026-10-09_fixed_sim/run_fixed_sim_modal.py::checks
  cd <repo worktree> && uv run modal run experiments/2026-10-09_fixed_sim/run_fixed_sim_modal.py::apply
"""

import gzip
import hashlib
import json
import random
import re
import subprocess
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).parent
# repo paths exist only on the laptop; inside the container this file sits at /root
REPO = HERE.parents[1] if modal.is_local() else Path("/root")
SWAP = REPO / "experiments" / "2026-10-07_q2_rendering_swap"
NAME = "probe-shift_aswin-surya_sub-fixed-sim_kill-2026-11-09"
INPUTS = "2026-10-06_la_segment-end_qwen3-32b_v1"
RUN_OUT = "2026-10-09_la_fixed-sim_v2"  # v1: the first apply, superseded by plan amendment 1 (clock rule v2)
Q1 = REPO / "experiments" / "2026-10-06_q1_text_baselines"
MAX_TOKENS = 32768  # q1's stage.yaml cut, the probe's
# container path of every file RULES.md pins
PINNED = {"experiments/2026-10-09_fixed_sim/fixes.py": "/root/fixed_sim/fixes.py",
          "experiments/2026-10-09_fixed_sim/clock.py": "/root/fixed_sim/clock.py",
          "experiments/2026-10-07_q2_rendering_swap/rerender.py": "/root/2026-10-07_q2_rendering_swap/rerender.py",
          "experiments/2026-10-07_q2_rendering_swap/str_replace_tells.py": "/root/2026-10-07_q2_rendering_swap/str_replace_tells.py",
          "experiments/2026-10-06_q1_text_baselines/run_stage_examples.py": "/root/q1/run_stage_examples.py"}
ARMS = ("real-attack-e1", "real-attack-e2", "sim-attack")

app = modal.App(NAME)
acts = modal.Volume.from_name("probe-shift_aswin-surya_sub-la-activations")
image = (modal.Image.debian_slim(python_version="3.13")
         .add_local_python_source("probe_shift")  # rerender.read_only uses probe_shift.calls.label_call
         .add_local_file(HERE / "fixes.py", "/root/fixed_sim/fixes.py")
         .add_local_file(HERE / "clock.py", "/root/fixed_sim/clock.py")
         .add_local_file(SWAP / "rerender.py", "/root/2026-10-07_q2_rendering_swap/rerender.py")
         .add_local_file(SWAP / "str_replace_tells.py", "/root/2026-10-07_q2_rendering_swap/str_replace_tells.py"))
# the apply step also tokenizes (q1's stage_run, Qwen3-32B's tokenizer at the shared pin); run_stage_examples imports
# probe_shift.transcripts, which imports boto3 and yaml (no AWS call is made)
apply_image = (modal.Image.debian_slim(python_version="3.13")
               .uv_pip_install("transformers==4.56.2", "jinja2", "pyyaml", "boto3")
               .add_local_python_source("probe_shift")
               .add_local_file(HERE / "fixes.py", "/root/fixed_sim/fixes.py")
               .add_local_file(HERE / "clock.py", "/root/fixed_sim/clock.py")
               .add_local_file(SWAP / "rerender.py", "/root/2026-10-07_q2_rendering_swap/rerender.py")
               .add_local_file(SWAP / "str_replace_tells.py", "/root/2026-10-07_q2_rendering_swap/str_replace_tells.py")
               .add_local_file(Q1 / "run_stage_examples.py", "/root/q1/run_stage_examples.py"))


def load(cells: set[str]) -> list[dict]:
    rows = []
    for f in sorted(Path(f"/acts/{INPUTS}/inputs").glob("shard_*.jsonl")):
        rows += [r for r in map(json.loads, f.read_text().splitlines()) if r["arm"] in ARMS and r["cell"] in cells]
    return rows


def text_of_calls(messages: list[dict]) -> str:
    """Everything the policy wrote (assistant text and tool-call arguments): never changed by a rule."""
    return "\n".join(str(m.get("content") or "") + json.dumps(m.get("tool_calls") or [], default=str)
                     for m in messages if m["role"] == "assistant")


@app.function(image=image, volumes={"/acts": acts}, cpu=8, memory=32768, timeout=1800, retries=0)
def checks_job(cells: list[str], dev: list[str]) -> dict:
    sys.path[:0] = ["/root/fixed_sim", "/root/2026-10-07_q2_rendering_swap"]
    import clock
    import fixes

    rows = load(set(cells))
    dev = set(dev)
    out = {"rows": dict(Counter(r["arm"] for r in rows))}
    assert out["rows"] == {a: len(cells) for a in ARMS}, out["rows"]

    # (a) the kept format rules over the REAL runs (dry run)
    real = Counter()
    examples = {}
    for r in (r for r in rows if r["arm"] != "sim-attack"):
        group = "dev" if r["environment"] in dev else "test"
        dpkg_outputs = sum(1 for c in fixes.calls_of(r["messages"]) if c["tool"] != "text_editor"
                           and fixes.DPKG_MARKER in str(r["messages"][c["msg"]]["content"]))
        for name in ("R1", "R2", "R5"):
            new, recs = fixes.RULES[name](r["messages"])
            changed = [x for x in recs if x["changed"]]
            # R1 and R2 record every insert / str_replace they look at; R5 records only changes, so its denominator is
            # every non-editor output holding a dpkg install
            real[(r["arm"], group, name, "outputs_considered")] += dpkg_outputs if name == "R5" else len(recs)
            real[(r["arm"], group, name, "outputs_changed")] += len(changed)
            for x in changed[:1]:
                ex = examples.setdefault(name, [])
                if len(ex) < 5:
                    ex.append({"arm": r["arm"], "group": group, "key": r["key"], "step": x["step"], "category": x.get("category"),
                               "before": r["messages"][x["msg"]]["content"][:600], "after": new[x["msg"]]["content"][:600]})
    out["a_real_dry_run"] = {"|".join(k): v for k, v in sorted(real.items())}
    out["a_real_examples"] = examples

    # (b) R1 on simulated runs: inserts that gain the review note
    note = Counter()
    gained = []
    for r in (r for r in rows if r["arm"] == "sim-attack"):
        new, recs = fixes.r1_insert(r["messages"])
        for x in recs:
            if x["msg"] is None:  # a transcript R1 skipped whole (a non-text tool output)
                note[("skipped_transcripts", x["category"])] += 1
                continue
            before, after = r["messages"][x["msg"]]["content"], new[x["msg"]]["content"]
            note[("inserts", x["category"])] += 1
            g = fixes.rerender.REVIEW in after and fixes.rerender.REVIEW not in before
            note[("gains_note", g)] += 1
            if g and r["environment"] in dev:
                gained.append({"key": r["key"], "environment": r["environment"], "step": x["step"],
                               "category": x["category"], "before": before[:900], "after": after[:900]})
    out["b_insert_note"] = {"|".join(map(str, k)): v for k, v in sorted(note.items())}
    out["b_examples"] = random.Random(0).sample(gained, min(3, len(gained)))

    # (d) R2 and R5 on DEV simulated runs: outputs per category, and diffs holding a sign-first line that R2 did not
    # classify as sign-first (each would be numbered by the swap code; listed for reading)
    cat, odd = Counter(), []
    for r in (r for r in rows if r["arm"] == "sim-attack" and r["environment"] in dev):
        for name in ("R2", "R5"):
            new, recs = fixes.RULES[name](r["messages"])
            for x in recs:
                cat[(name, x["category"], x["changed"])] += 1
                old = r["messages"][x["msg"]]["content"]
                if name == "R2" and x["category"] not in ("sign_first", "numbered") and isinstance(old, str) \
                        and any(l[:1] in "+-" and len(l) > 1 and fixes.SIGN_FIRST_LINE.match(l) for l in old.split("\n")) and len(odd) < 5:
                    odd.append({"key": r["key"], "step": x["step"], "category": x["category"], "before": old[:600],
                                "after": new[x["msg"]]["content"][:600]})
    out["d_dev_sim_categories"] = {"|".join(map(str, k)): v for k, v in sorted(cat.items())}
    out["d_sign_first_unclassified_examples"] = odd

    # (c) the clock rule on DEV simulated runs, donors from DEV real e1 runs
    def outputs(r):
        return [m["content"] for m in r["messages"] if m["role"] == "tool" and isinstance(m["content"], str)]
    donors = {r["cell"]: n for r in rows if r["arm"] == "real-attack-e1" and r["environment"] in dev
              if (n := clock.now_of(outputs(r))) is not None}
    cc, runs, xref = Counter(), Counter(), []
    for r in (r for r in rows if r["arm"] == "sim-attack" and r["environment"] in dev):
        now = clock.now_of(outputs(r))
        if now is None:  # nothing shifts, but its stamps and failures still count
            runs["no_now"] += 1
            for t in outputs(r):
                found = clock.stamps(t)
                cc.update(clock.untouched_counts(t, [(x["start"], x["end"]) for x in found]))
                cc.update(f"no_now:{x['kind']}" for x in found)
            continue
        _, donor_now = clock.donor_for(r["cell"], donors)
        offset = donor_now - now
        runs["shifted_runs"] += 1
        policy = text_of_calls(r["messages"])
        exact = partial = 0
        for t in outputs(r):
            new, c = clock.shift_text(t, offset, now)
            cc.update(c)
            cc["outputs_touched"] += new != t
            for x in clock.stamps(t, now.year):
                if x["kind"] == "FAILED" or abs(x["utc"] - now) > clock.WINDOW:
                    continue
                if x["m"].group(0) in policy:
                    exact += 1
                elif x["dt"].strftime("%H:%M") in policy or x["dt"].strftime("%Y-%m-%d") in policy:
                    partial += 1  # the policy names the stamp's time or date component
        runs["runs_with_policy_reference_exact"] += exact > 0
        runs["runs_with_policy_reference_partial"] += (partial > 0) and not exact
        if (exact or partial) and len(xref) < 5:
            xref.append(r["key"])
    out["c_clock_dev"] = {"runs": dict(runs), "stamps": dict(cc), "donors": len(donors),
                          "runs_with_policy_reference_examples": xref,
                          "sim_outputs_dev": sum(len(outputs(r)) for r in rows if r["arm"] == "sim-attack" and r["environment"] in dev)}
    return out


@app.function(image=image, volumes={"/acts": acts}, cpu=2, memory=8192, timeout=900, retries=0)
def real_r2_job(keys: list[str]) -> list[dict]:
    """Full before / after of the real outputs R2 changes in the dry run, with where they differ."""
    sys.path[:0] = ["/root/fixed_sim", "/root/2026-10-07_q2_rendering_swap"]
    import fixes

    out = []
    for f in sorted(Path(f"/acts/{INPUTS}/inputs").glob("shard_*.jsonl")):
        for r in map(json.loads, f.read_text().splitlines()):
            if r["key"] not in keys:
                continue
            new, recs = fixes.r2_str_replace(r["messages"])
            for x in recs:
                if x["changed"]:
                    b, a = r["messages"][x["msg"]]["content"], new[x["msg"]]["content"]
                    i = next((k for k in range(min(len(a), len(b))) if a[k] != b[k]), min(len(a), len(b)))
                    out.append({"key": r["key"], "step": x["step"], "category": x["category"], "len_before": len(b),
                                "len_after": len(a), "first_difference": i, "before_around": b[max(0, i - 200):i + 200],
                                "after_around": a[max(0, i - 200):i + 200], "before_head": b[:400]})
    return out


@app.local_entrypoint()
def real_r2_details(checks_file: str):
    d = json.loads((HERE / "results" / checks_file).read_text())
    keys = sorted({e["key"] for e in d["a_real_examples"].get("R2", [])})
    print(json.dumps(real_r2_job.remote(keys), indent=1))


@app.local_entrypoint()
def checks():
    ex = json.loads((REPO / "splits" / "exclusions_v4.json").read_text())
    split = json.loads((HERE / "split.json").read_text())
    out = checks_job.remote(ex["paired_attack_cells"]["cells"], split["dev"])
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    (HERE / "results").mkdir(exist_ok=True)
    path = HERE / "results" / f"prefreeze_checks_{stamp}.json"
    with open(path, "x") as f:
        f.write(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if not k.endswith("examples")}, indent=1))
    print(path)


def sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def frozen_hashes() -> dict[str, str]:
    """RULES.md's pinned sha256s ({repo path: sha256}), each checked against the file on disk."""
    pins = {p: h for h, p in re.findall(r"^([0-9a-f]{64})  (\S+)$", (HERE / "RULES.md").read_text(), re.M) if p in PINNED}
    assert set(pins) == set(PINNED), f"RULES.md pins {sorted(pins)}, expected {sorted(PINNED)}"
    for p, h in pins.items():
        assert sha256(REPO / p) == h, f"{p} differs from the frozen version"
    return pins


def outputs_of(messages: list[dict]) -> list[str]:
    return [m["content"] for m in messages if m["role"] == "tool" and isinstance(m["content"], str)]


@app.function(image=apply_image, volumes={"/acts": acts}, cpu=2, memory=8192, timeout=1200, retries=0)
def donors_job(cells: list[str], dev: list[str]) -> dict[str, str]:
    """Each dev real epoch-1 run's "now" (ISO, UTC), by cell; runs without a timestamp are not donors."""
    sys.path[:0] = ["/root/fixed_sim", "/root/2026-10-07_q2_rendering_swap"]
    import clock

    rows = [r for r in load(set(cells)) if r["arm"] == "real-attack-e1" and r["environment"] in set(dev)]
    assert len({r["cell"] for r in rows}) == len(rows), "two dev real e1 runs share a cell"
    return {r["cell"]: n.isoformat() for r in rows if (n := clock.now_of(outputs_of(r["messages"]))) is not None}


@app.function(image=apply_image, volumes={"/acts": acts}, cpu=4, memory=16384, timeout=1200, retries=0)
def apply_shard(shard: int, cells: list[str], donors: dict[str, str], pins: dict[str, str]) -> dict:
    sys.path[:0] = ["/root/fixed_sim", "/root/2026-10-07_q2_rendering_swap", "/root/q1"]
    for p, h in pins.items():
        assert sha256(PINNED[p]) == h, f"container copy of {p} differs from the frozen version"
    import clock
    import fixes
    from run_stage_examples import stage_run
    from transformers import AutoTokenizer

    from probe_shift.models import QWEN3_32B

    tok = AutoTokenizer.from_pretrained(QWEN3_32B["name"], revision=QWEN3_32B["revision"])
    donors = {c: datetime.fromisoformat(n) for c, n in donors.items()}
    cells = set(cells)
    out_dir = Path("/acts") / RUN_OUT
    lines_in = Path(f"/acts/{INPUTS}/inputs/shard_{shard}.jsonl").read_text().splitlines()
    lines_out, outputs, calls, runs, records = [], [], [], [], []
    count = Counter()
    for line in lines_in:
        r = json.loads(line)
        if r["arm"] not in ARMS or r["cell"] not in cells:
            continue
        meta = {k: r[k] for k in ("arm", "key", "file", "sample_id", "epoch", "environment", "main_task", "cell", "sim_reference")}
        if r["arm"] != "sim-attack":  # real: copied byte for byte, staged from the same messages
            seg, o, c, run = stage_run(tok, r["messages"], MAX_TOKENS, meta)
            assert seg["truncated"] == r["truncated"], f"{r['key']}: truncation disagrees with the reader inputs"
            count[(r["arm"], "token_count_matches_inputs")] += seg["n_tokens"] == r["tokens"]
            lines_out.append(line)
        else:
            new, recs = fixes.apply(r["messages"], ("R1", "R2", "R5"))
            now = clock.now_of(outputs_of(r["messages"]))  # from the uncorrected outputs, as the checks read it
            clock_rec = {"now": None, "donor_cell": None, "donor_now": None, "offset_s": None}
            if now is not None:
                donor, donor_now = clock.donor_for(r["cell"], donors)
                offset = donor_now - now
                clock_rec = {"now": now.isoformat(), "donor_cell": donor, "donor_now": donor_now.isoformat(),
                             "offset_s": round(offset.total_seconds())}
                for i, m in enumerate(new):
                    if m["role"] == "tool" and isinstance(m["content"], str):
                        shifted, c6 = clock.shift_text(m["content"], offset, now)
                        count.update({("sim-attack", "R6", k): v for k, v in c6.items() if v})
                        if shifted != m["content"]:
                            new[i] = m | {"content": shifted}
                            recs.append({"msg": i, "rule": "R6", "category": "shifted", "changed": True})
            else:
                count[("sim-attack", "R6", "run_without_now")] += 1
            for x in recs:
                count[("sim-attack", x["rule"], x["category"], x["changed"])] += 1
            changed_msgs = {x["msg"] for x in recs if x["changed"]}
            assert all(new[i] == m for i, m in enumerate(r["messages"]) if i not in changed_msgs), f"{r['key']}: an unrecorded change"
            assert all(m["role"] == "tool" for i, m in enumerate(r["messages"]) if i in changed_msgs), f"{r['key']}: a non-tool message changed"
            seg, o, c, run = stage_run(tok, new, MAX_TOKENS, meta)
            count[("sim-attack", "runs_changed")] += bool(changed_msgs)
            count[("sim-attack", "outputs_changed")] += len(changed_msgs)
            count[("sim-attack", "truncated_before")] += r["truncated"]
            count[("sim-attack", "truncated_after")] += seg["truncated"]
            records += [{"key": r["key"], "cell": r["cell"], "environment": r["environment"]} | x for x in recs]
            lines_out.append(json.dumps(r | {"messages": new, "tokens": seg["n_tokens"], "truncated": seg["truncated"],
                                             "fixed_sim": {"tokens_before": r["tokens"], "truncated_before": r["truncated"],
                                                           "outputs_changed": len(changed_msgs), "clock": clock_rec}}))
        outputs += o
        calls += c
        runs.append(run | {"environment": r["environment"]})
        count[(r["arm"], "runs")] += 1
    (out_dir / "inputs").mkdir(parents=True, exist_ok=True)
    (out_dir / "examples" / "parts").mkdir(parents=True, exist_ok=True)
    (out_dir / "records").mkdir(parents=True, exist_ok=True)
    files = {f"inputs/shard_{shard}.jsonl": "".join(x + "\n" for x in lines_out).encode(),
             f"examples/parts/outputs_{shard}.jsonl.gz": gzip.compress("".join(json.dumps(x) + "\n" for x in outputs).encode(), mtime=0),
             f"examples/parts/calls_{shard}.jsonl.gz": gzip.compress("".join(json.dumps(x) + "\n" for x in calls).encode(), mtime=0),
             f"examples/parts/runs_{shard}.jsonl": "".join(json.dumps(x) + "\n" for x in runs).encode(),
             f"records/records_{shard}.jsonl.gz": gzip.compress("".join(json.dumps(x) + "\n" for x in records).encode(), mtime=0)}
    for name, data in files.items():
        with open(out_dir / name, "xb") as f:  # never overwrites
            f.write(data)
    acts.commit()
    real_in = [l for l in lines_in if (r := json.loads(l))["arm"] in ARMS[:2] and r["cell"] in cells]
    assert real_in == [l for l in lines_out if json.loads(l)["arm"] != "sim-attack"], "real rows were not copied byte for byte"
    return {"shard": shard, "counts": {"|".join(map(str, k)): v for k, v in sorted(count.items())},
            "files": {n: hashlib.sha256(d).hexdigest() for n, d in files.items()}}


@app.function(image=apply_image, volumes={"/acts": acts}, cpu=2, memory=16384, timeout=1200, retries=0)
def merge_job(n_shards: int) -> dict:
    """Joins the per-shard example parts into examples/outputs.jsonl.gz, calls.jsonl.gz and runs.jsonl (shard order),
    then hashes every file of the run into manifest.json."""
    acts.reload()
    d = Path("/acts") / RUN_OUT
    for name, ext in (("outputs", ".jsonl.gz"), ("calls", ".jsonl.gz"), ("runs", ".jsonl")):
        parts = [(d / "examples" / "parts" / f"{name}_{i}{ext}").read_bytes() for i in range(n_shards)]
        data = gzip.compress(b"".join(gzip.decompress(p) for p in parts), mtime=0) if ext.endswith(".gz") else b"".join(parts)
        with open(d / "examples" / f"{name}{ext}", "xb") as f:
            f.write(data)
    files = {str(p.relative_to(d)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(d.rglob("*")) if p.is_file()}
    with open(d / "manifest.json", "x") as f:
        f.write(json.dumps({"run": RUN_OUT, "files": files}, indent=1))
    acts.commit()
    return files


@app.local_entrypoint()
def apply():
    pins = frozen_hashes()
    dirty = subprocess.run(["git", "status", "--porcelain", "--", *PINNED, str(HERE / "RULES.md")], cwd=REPO,
                           capture_output=True, text=True, check=True).stdout
    assert not dirty, f"pinned files have uncommitted changes:\n{dirty}"
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()
    try:
        acts.listdir(RUN_OUT)
    except (FileNotFoundError, modal.exception.NotFoundError):
        pass
    else:
        raise SystemExit(f"{RUN_OUT} already exists on the Volume; the rules are applied once")
    ex = json.loads((REPO / "splits" / "exclusions_v4.json").read_text())
    cells = ex["paired_attack_cells"]["cells"]
    split = json.loads((HERE / "split.json").read_text())
    donors = donors_job.remote(cells, split["dev"])
    n_shards = len([e for e in acts.listdir(f"{INPUTS}/inputs") if e.path.endswith(".jsonl")])
    shards = list(apply_shard.map(range(n_shards), kwargs={"cells": cells, "donors": donors, "pins": pins}))
    total = Counter()
    for sh in shards:
        total.update(sh["counts"])
    runs = {a: total[f"{a}|runs"] for a in ARMS}
    assert runs == {a: len(cells) for a in ARMS}, runs
    files = merge_job.remote(n_shards)
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    out = {"run": RUN_OUT, "git_head": head, "rules_sha256": pins, "donors": len(donors), "counts": dict(sorted(total.items())),
           "shard_files": {n: h for sh in shards for n, h in sh["files"].items()}, "manifest_files": files}
    path = HERE / "results" / f"apply_{stamp}.json"
    with open(path, "x") as f:
        f.write(json.dumps(out, indent=1))
    print(json.dumps(out["counts"], indent=1))
    print(path)


AUDIT_SEED, AUDIT_N = 20261009, 5  # v1's audit; the v2 clock re-audit uses 20261010, 10 outputs (plan amendment 1)


@app.function(image=image, volumes={"/acts": acts}, cpu=2, memory=16384, timeout=900, retries=0)
def audit_job(seed: int = AUDIT_SEED, n: int = AUDIT_N, rules: tuple = ("R1", "R2", "R5", "R6")) -> list[dict]:
    """A seeded sample of `n` changed outputs per rule (all environments), with the text before (the original reader
    inputs) and after (the fixed run), for the hand audit (audit.md)."""
    d = Path("/acts") / RUN_OUT
    recs = [json.loads(l) for f in sorted((d / "records").glob("records_*.jsonl.gz"))
            for l in gzip.decompress(f.read_bytes()).decode().splitlines()]
    changed = [r for r in recs if r["changed"]]
    rules_of = {}
    for r in changed:
        rules_of.setdefault((r["key"], r["msg"]), []).append(r["rule"])
    rng = random.Random(seed)
    picks = []
    for rule in rules:
        pool = sorted((r for r in changed if r["rule"] == rule), key=lambda r: (r["key"], r["msg"]))
        picks += rng.sample(pool, min(n, len(pool)))
    want = {(p["key"], p["msg"]) for p in picks}
    text = {}
    for src, run in (("before", INPUTS), ("after", RUN_OUT)):
        for f in sorted(Path(f"/acts/{run}/inputs").glob("shard_*.jsonl")):
            for r in map(json.loads, f.read_text().splitlines()):
                if r["arm"] == "sim-attack":
                    for k, m in enumerate(r["messages"]):
                        if (r["key"], k) in want:
                            text[(src, r["key"], k)] = m["content"]
    return [p | {"rules_on_this_output": rules_of[(p["key"], p["msg"])], "before": text[("before", p["key"], p["msg"])],
                 "after": text[("after", p["key"], p["msg"])]} for p in picks]


@app.local_entrypoint()
def audit(seed: int = AUDIT_SEED, n: int = AUDIT_N, rules: str = "R1,R2,R5,R6"):
    out = audit_job.remote(seed, n, tuple(rules.split(",")))
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    path = HERE / "results" / f"audit_sample_{stamp}.json"
    with open(path, "x") as f:
        f.write(json.dumps({"seed": seed, "per_rule": n, "rules": rules, "run": RUN_OUT, "samples": out}, indent=1))
    print(path)
