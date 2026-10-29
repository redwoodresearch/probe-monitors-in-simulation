"""Gate row 5 (PLAN.md here): staging. One Modal CPU container. A copy of P4's q2_reply_stage.py
(../2026-10-12_plantslip_transfer, as run at f89f152f) with the whitespace plant in place of plant_both, on the 939 real
e2 runs only, from #36's staged messages (#36's dump's inputs). It stages two runs on the shared activations Volume:
- `2026-10-13_q2_row5_ws_controls_inputs_v1`: 20 seeded unplanted e2 runs (arm `real-attack-e2-control`), for the gate;
- `2026-10-13_q2_row5_ws_planted_inputs_v1`: every e2 run the plant changed (arm `real-attack-e2-planted`).
The plant (PLAN.md's table): W1 (`TRAILING_SPACE.sub("", t)`) on text_editor and bash outputs; W2
(`BLANK_RUNS.sub("\\n\\n", t)`) on bash and other; W3 (append one "\\n" when the output doesn't end in whitespace, empty outputs included) on
text_editor create and insert, and other. They apply in that order to string tool outputs. Harness-truncated and
non-string outputs are skipped and counted (the reader renders each tool message's content verbatim). Only counts and keys
leave the container. Reach: runs whose Qwen3-32B token ids change within the extraction's 32,768-token cut.

  PYTHONPATH=src uv run --no-project --python 3.12 --with "modal==1.6.1" modal run \\
      experiments/2026-10-13_q2_row5_whitespace/stage_whitespace.py --runs <#36 runs.json>
"""
import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).parent
EXP = HERE.parent
FILES = {"rerender.py": EXP / "2026-10-07_q2_rendering_swap", "str_replace_tells.py": EXP / "2026-10-07_q2_rendering_swap",
         "fixes.py": EXP / "2026-10-09_fixed_sim"}  # named, not HERE: in the container HERE is /root
APP = "probe-shift_aswin-surya_sub-q2-row5ws_stage_kill-2026-11-13"
ACTS = "probe-shift_aswin-surya_sub-la-activations"
SOURCE = "2026-10-06_la_segment-end_qwen3-32b_v1"
CONTROLS_RUN, PLANTED_RUN = "2026-10-13_q2_row5_ws_controls_inputs_v1", "2026-10-13_q2_row5_ws_planted_inputs_v1"
ARM, PLANTED_ARM, CONTROL = "real-attack-e2", "real-attack-e2-planted", "real-attack-e2-control"
QWEN3 = ("Qwen/Qwen3-32B", "9216db5781bf21249d130ec9da846c4624c16137")  # probe_shift.models.QWEN3_32B
SEED, N_CONTROLS, N_SHARD_FILES, CUT = "20261013", 20, 8, 32768
PIN = {"runs.json": "76b11486cc353ab1eec390d49ef7449d729d4039d1e85643541d6679d73c3b96"}  # #36's run list
W1_GROUPS, W2_GROUPS = {"text_editor", "bash"}, {"bash", "other"}

app = modal.App(APP)
acts = modal.Volume.from_name(ACTS, version=2)
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("transformers==4.56.2", "jinja2==3.1.6", "huggingface_hub").add_local_python_source("probe_shift")
for _n, _f in FILES.items():
    image = image.add_local_file(_f / _n, f"/root/{_f.name}/{_n}")


def _mods():
    import sys
    for folder in {f for f in FILES.values()}:
        for p in (f"/root/{folder.name}", str(folder)):
            if p not in sys.path:
                sys.path.insert(0, p)
    import fixes  # noqa: E402  (it puts the swap folder on the path)
    return fixes


def group(tool: str) -> str:  # the direction count's grouping (count_whitespace_direction.py:65)
    return "text_editor" if tool == "text_editor" else "bash" if tool == "bash" else "other"


def plant_ws(messages: list[dict], fixes):
    """(planted messages, [(msg, step, action, scope, ops)] changed, Counter of skipped outputs by reason)."""
    from collections import Counter

    from probe_shift.scrub import BLANK_RUNS, TRAILING_SPACE
    out, changed, skipped = [dict(m) for m in messages], [], Counter()
    for c in fixes.calls_of(messages):
        i, g = c["msg"], group(c["tool"])
        cmd = c["args"].get("command") if c["tool"] == "text_editor" else None
        o = messages[i]["content"]
        if not isinstance(o, str):
            skipped["non-string content"] += 1; continue
        if fixes.TRUNCATED in o:
            skipped["harness-truncated"] += 1; continue
        t, ops = o, []
        if g in W1_GROUPS and (t2 := TRAILING_SPACE.sub("", t)) != t:
            t, ops = t2, ops + ["W1"]
        if g in W2_GROUPS and (t2 := BLANK_RUNS.sub("\n\n", t)) != t:
            t, ops = t2, ops + ["W2"]
        if ((g == "text_editor" and cmd in ("create", "insert")) or g == "other") and not t[-1:].isspace():
            skipped["W3 on an empty output (counted, not skipped)"] += t == ""
            t, ops = t + "\n", ops + ["W3"]
        if t != o:
            out[i]["content"] = t
            changed.append((i, c["step"], c["action"], f"text_editor:{cmd}" if g == "text_editor" else g, "+".join(ops)))
    return out, changed, skipped


@app.function(image=image, volumes={"/acts": acts}, cpu=16, memory=32768, timeout=3600, retries=0)
def stage(keys: list[str], controls: list[str]) -> dict:
    try:
        return _stage(keys, controls)
    except Exception as e:  # nothing from the exception (which may quote staged text) crosses the boundary
        import traceback
        tb = traceback.extract_tb(e.__traceback__)[-1]
        raise RuntimeError(f"staging failed: {type(e).__name__} {getattr(e, 'name', '') or ''} at {Path(tb.filename).name}:{tb.lineno}") from None


def _stage(keys, controls):
    from collections import Counter

    from transformers import AutoTokenizer

    from probe_shift.reader import _layout
    fixes = _mods()
    tok = AutoTokenizer.from_pretrained(QWEN3[0], revision=QWEN3[1])
    want = set(keys)
    items = {}
    for shard in sorted((Path("/acts") / SOURCE / "inputs").glob("shard_*.jsonl")):
        for line in shard.read_text().splitlines():
            it = json.loads(line)
            if it["arm"] == ARM and it["key"] in want:
                items[it["key"]] = it
    if set(items) != want or not set(controls) <= want:
        raise ValueError("a run is missing from the staged inputs")
    n, staged_p, staged_c, per_run = Counter(), [], [], []
    for k in sorted(want):
        it = items[k]
        new, changed, skipped = plant_ws(it["messages"], fixes)
        for reason, v in skipped.items():
            n[f"outputs: {reason}" if reason.startswith("W3") else f"outputs skipped: {reason}"] += v
        if not changed:
            n["runs unchanged"] += 1
            continue
        diff = [i for i, (a, b) in enumerate(zip(it["messages"], new)) if a != b]
        assert len(new) == len(it["messages"]) and diff == [c[0] for c in changed]
        assert all({x for x in new[i] if new[i][x] != it["messages"][i][x]} == {"content"} for i in diff)
        ids_new, ids_old = _layout(tok, new)[2]["input_ids"], _layout(tok, it["messages"])[2]["input_ids"]
        reach = list(ids_new[:CUT]) != list(ids_old[:CUT])
        n["runs changed"] += 1; n["outputs changed"] += len(changed); n["runs whose Qwen3-32B token ids change within the 32k cut"] += reach
        for _, _, _, scope, ops in changed:
            for op in ops.split("+"):
                n[f"outputs changed: {op} on {scope}"] += 1
        for op, scope in {(op, s) for *_, scope, ops in changed for op in ops.split("+") for s in {scope, scope.split(":")[0]}}:
            n[f"runs changed: {op} on {scope}"] += 1  # per command, and per group (text_editor pooled)
        staged_p.append(it | {"arm": PLANTED_ARM, "messages": new, "tokens": len(ids_new), "plant": "whitespace", "orig_arm": ARM,
                              "changed": [list(c) for c in changed]})
        per_run.append({"key": k, "outputs changed": len(changed), "reach within cut": bool(reach),
                        "ops": dict(Counter(op for *_, ops in changed for op in ops.split("+")))})
    for k in controls:
        it = items[k]
        staged_c.append(it | {"arm": CONTROL, "tokens": len(_layout(tok, it["messages"])[2]["input_ids"]), "plant": None, "orig_arm": ARM})
        n["control runs"] += 1
    shas = {}
    for run, staged in ((CONTROLS_RUN, staged_c), (PLANTED_RUN, staged_p)):
        d = Path("/acts") / run / "inputs"
        if d.exists() and any(d.iterdir()):
            raise FileExistsError(f"{run} is already staged")
        d.mkdir(parents=True)
        for s in range(N_SHARD_FILES):
            body = "".join(json.dumps(x) + "\n" for x in staged[s::N_SHARD_FILES]).encode()
            (d / f"shard_{s}.jsonl").write_bytes(body)
            shas[f"{run}/shard_{s}.jsonl"] = hashlib.sha256(body).hexdigest()
    acts.commit()
    return {"counts": dict(sorted((k, int(v)) for k, v in n.items())), "per run": per_run, "inputs_sha256": shas,
            "planted": [{"key": x["key"], "orig_arm": x["orig_arm"], "arm": x["arm"]} for x in staged_p],
            "controls": [{"key": x["key"], "orig_arm": x["orig_arm"], "arm": x["arm"]} for x in staged_c],
            "Qwen3-32B tokens staged": {"planted": int(sum(x["tokens"] for x in staged_p)), "controls": int(sum(x["tokens"] for x in staged_c))}}


@app.local_entrypoint()
def main(runs: str):
    import subprocess
    for f in [Path(__file__).resolve(), HERE / "PLAN.md"] + [folder / name for name, folder in FILES.items()]:
        if subprocess.run(["git", "ls-files", "--error-unmatch", str(f)], cwd=HERE, capture_output=True).returncode \
                or subprocess.run(["git", "diff", "--quiet", "HEAD", "--", str(f)], cwd=HERE).returncode:
            raise SystemExit(f"{f.name} differs from HEAD: commit it before running")
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip()
    if hashlib.sha256(Path(runs).read_bytes()).hexdigest() != PIN["runs.json"]:
        raise SystemExit("not #36's runs.json")
    keys = [r["key"] for r in json.loads(Path(runs).read_text()) if r["arm"] == ARM]
    if len(keys) != 939 or len(set(keys)) != 939:
        raise SystemExit("not 939 distinct e2 runs")
    rank = lambda k: hashlib.sha256(f"{SEED}|{k}".encode()).hexdigest()  # noqa: E731
    controls = sorted(keys, key=rank)[:N_CONTROLS]
    res = stage.remote(keys, controls)
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    out = {"plan": "PLAN.md (this folder)", "git": head, "runs.json sha256": PIN["runs.json"], "source": SOURCE,
           "staged runs (Volume probe-shift_aswin-surya_sub-la-activations)": [CONTROLS_RUN, PLANTED_RUN], "control seed": SEED} | res
    (HERE / "results").mkdir(exist_ok=True)
    with open(dest := HERE / "results" / f"stage_ws_{stamp}.json", "x") as fh:
        fh.write(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k not in ("per run", "planted", "controls")}, indent=1)); print(dest)
