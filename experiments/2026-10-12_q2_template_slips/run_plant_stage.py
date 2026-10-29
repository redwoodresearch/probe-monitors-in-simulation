"""Template slips, step 2, staging (PLAN.md addendum 3). One Modal CPU container: plants (b) gutter stripped and (c)
review note dropped, each on its own, into the 939 real e2 runs' staged messages (the inputs Qwen3.5's dump read), and
stages the changed runs plus 8 unplanted controls for the extractor copy. Counts and keys only leave the container.

  PYTHONPATH=src uv run --no-project --python 3.12 --with "modal==1.6.1" modal run \\
      experiments/2026-10-12_q2_template_slips/run_plant_stage.py --runs <Qwen3.5 run_scores_2026-10-12_011944/runs.json>
"""
import hashlib
import json
import re
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).parent
EXP = HERE.parent
FILES = {"rerender.py": EXP / "2026-10-07_q2_rendering_swap", "str_replace_tells.py": EXP / "2026-10-07_q2_rendering_swap",
         "fixes.py": EXP / "2026-10-09_fixed_sim"}
APP = "probe-shift_aswin-surya_sub-q2-tplslips_stage_kill-2026-11-12"
ACTS = "probe-shift_aswin-surya_sub-la-activations"  # read-only: the staged messages Qwen3.5's dump read
INPUTS = "probe-shift_aswin-surya_sub-q2-tplslips_inputs_kill-2026-11-12"
SOURCE, STAGED = "2026-10-06_la_segment-end_qwen3-32b_v1", "2026-10-12_q2_tplslips_plants_inputs_v1"
QWEN35 = ("Qwen/Qwen3.5-27B", "fc05daec18b0a78c049392ed2e771dde82bdf654")
E2, PLANTS, CONTROL = "real-attack-e2", ("e2-gutter-stripped", "e2-note-dropped"), "e2-control"
SEED, N_CONTROLS, N_SHARD_FILES = "20261020", 8, 8
PIN = {"runs.json": "3c028aca246181950e460349067ca27d0fb85bff60bc4c01b518c9b27a234ec1"}  # the Qwen3.5 run's run list
GUTTER = re.compile(r"^ *\d+\|(?:([ +\-])(?: (.*))?)?$")  # fixes.REAL_GUTTER_LINE, with the sign and code captured

app = modal.App(APP)
acts = modal.Volume.from_name(ACTS, version=2).read_only()
inputs = modal.Volume.from_name(INPUTS, version=2, create_if_missing=True)
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("transformers==5.19.0", "jinja2==3.1.6", "huggingface_hub").add_local_python_source("probe_shift")
for _n, _f in FILES.items():
    image = image.add_local_file(_f / _n, f"/root/{_f.name}/{_n}")


def _mods():
    import sys
    for folder in {f for f in FILES.values()}:
        for p in (f"/root/{folder.name}", str(folder)):
            if p not in sys.path:
                sys.path.insert(0, p)
    import fixes  # noqa: E402  (it puts the swap folder on the path)
    import rerender
    return fixes, rerender


def strip_gutter(out: str) -> str:
    """Step 1's (b) pair, as in run_template_slips.py: each gutter line `{n}|{sign} {code}` becomes `{sign}{code}`."""
    head, sep, body = out.partition("+++ edited_version\n")
    lines = []
    for line in body.split("\n"):
        m = GUTTER.match(line)
        lines.append(line if line.startswith("@@") or not m else (m.group(1) or " ") + (m.group(2) or ""))
    return head + sep + "\n".join(lines)


def plant(messages: list[dict], which: str):
    """(planted messages, [(message index, step, action)] changed, in order)."""
    fixes, rerender = _mods()
    out, changed = [dict(m) for m in messages], []
    for c in fixes.calls_of(messages):
        o = messages[c["msg"]]["content"]
        if c["tool"] != "text_editor" or not isinstance(o, str) or fixes.TRUNCATED in o:
            continue
        cmd, new = c["args"].get("command"), None
        if which == "e2-gutter-stripped" and cmd == "str_replace" and rerender.DIFF in o[:300] \
                and fixes.numbered_form(o.partition("+++ edited_version\n")[2]) == "numbered":
            new = strip_gutter(o)
        if which == "e2-note-dropped" and cmd == "insert" and o.rstrip().endswith(rerender.REVIEW):
            new = o.rstrip().rstrip("\n")[: -len(rerender.REVIEW)].rstrip("\n")
        if new is not None and new != o:
            out[c["msg"]]["content"] = new
            changed.append((c["msg"], c["step"], c["action"]))
    return out, changed


@app.function(image=image, volumes={"/acts": acts, "/in": inputs}, cpu=16, memory=32768, timeout=3600, retries=0)
def stage(e2: list[dict], controls: list[str]) -> dict:
    try:
        return _stage(e2, controls)
    except Exception as e:  # nothing from the exception (which may quote staged text) crosses the boundary
        raise RuntimeError(f"staging failed: {type(e).__name__}") from None


def _stage(e2, controls):
    from collections import Counter

    from transformers import AutoTokenizer

    from probe_shift.reader import _layout
    tok = AutoTokenizer.from_pretrained(*QWEN35[:1], revision=QWEN35[1])
    want = {r["key"] for r in e2}
    items = {}
    for shard in sorted((Path("/acts") / SOURCE / "inputs").glob("shard_*.jsonl")):
        for line in shard.read_text().splitlines():
            it = json.loads(line)
            if it["arm"] == E2 and it["key"] in want:
                items[it["key"]] = it
    if set(items) != want:
        raise ValueError("an e2 run is missing from the staged inputs")
    n, per_run, staged = Counter(), [], []

    def tokens(msgs):
        return len(_layout(tok, msgs)[2]["input_ids"])
    for k in sorted(want):
        it = items[k]
        for which in PLANTS:
            new, changed = plant(it["messages"], which)
            if not changed:
                n[(which, "runs unchanged")] += 1
                continue
            # only the changed tool messages differ, and only their content
            diff = [i for i, (a, b) in enumerate(zip(it["messages"], new)) if a != b]
            assert len(new) == len(it["messages"]) and diff == [c[0] for c in changed]
            assert all(set(new[i]) == set(it["messages"][i]) and {x for x in new[i] if new[i][x] != it["messages"][i][x]} == {"content"} for i in diff)
            t_new, t_old = tokens(new), tokens(it["messages"])  # a reader failure aborts staging: every changed run is needed
            tool_idx = [i for i, m in enumerate(it["messages"]) if m["role"] == "tool"]
            n[(which, "runs changed")] += 1; n[(which, "outputs changed")] += len(changed)
            per_run.append({"plant": which, "key": k, "outputs changed": len(changed), "first changed (step, action)": list(changed[0][1:]),
                            "first changed is the first tool output": changed[0][0] == tool_idx[0], "qwen35 tokens": t_new, "qwen35 tokens, original": t_old})
            staged.append(it | {"arm": which, "messages": new, "tokens": t_new, "plant": which, "orig_arm": E2})
    for k in controls:
        it = items[k]
        staged.append(it | {"arm": CONTROL, "tokens": tokens(it["messages"]), "plant": None, "orig_arm": E2})
        n[(CONTROL, "runs")] += 1
    d = Path("/in") / STAGED / "inputs"
    if d.exists() and any(d.iterdir()):
        raise FileExistsError(f"{STAGED} is already staged")
    d.mkdir(parents=True)
    shas = {}
    for s in range(N_SHARD_FILES):
        body = "".join(json.dumps(x) + "\n" for x in staged[s::N_SHARD_FILES]).encode()
        (d / f"shard_{s}.jsonl").write_bytes(body)
        shas[f"shard_{s}.jsonl"] = hashlib.sha256(body).hexdigest()
    inputs.commit()
    return {"counts": [{"arm": a, "count": c, "n": int(v)} for (a, c), v in sorted(n.items())], "per run": per_run, "inputs_sha256": shas,
            "staged transcripts": len(staged), "qwen35 tokens staged": int(sum(x["tokens"] for x in staged))}


@app.local_entrypoint()
def main(runs: str):
    import subprocess
    for f in [Path(__file__).resolve(), HERE / "PLAN.md"] + [folder / name for name, folder in FILES.items()]:
        if subprocess.run(["git", "ls-files", "--error-unmatch", str(f)], cwd=HERE, capture_output=True).returncode \
                or subprocess.run(["git", "diff", "--quiet", "HEAD", "--", str(f)], cwd=HERE).returncode:
            raise SystemExit(f"{f.name} differs from HEAD: commit it before running")
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip()
    if hashlib.sha256(Path(runs).read_bytes()).hexdigest() != PIN["runs.json"]:
        raise SystemExit("not the Qwen3.5 run's runs.json")
    rs = [r for r in json.loads(Path(runs).read_text()) if r["arm"] == E2]
    if len(rs) != 939:
        raise SystemExit(f"{len(rs)} e2 runs, not 939")
    rank = lambda k: hashlib.sha256(f"{SEED}|{k}".encode()).hexdigest()  # noqa: E731
    controls = sorted((r["key"] for r in rs), key=rank)[:N_CONTROLS]
    res = stage.remote([{"key": r["key"], "filename": r["filename"]} for r in rs], controls)
    # the pilot: 4 controls, and per plant 2 runs whose first changed output is not their first output (sha rank)
    pilot = [[CONTROL, k] for k in controls[:4]]
    for which in PLANTS:
        ok = sorted((x["key"] for x in res["per run"] if x["plant"] == which and not x["first changed is the first tool output"]), key=rank)
        pilot += [[which, k] for k in ok[:2]]
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    out = {"plan": "PLAN.md addendum 3 (this folder)", "git": head, "runs.json sha256": PIN["runs.json"], "source": SOURCE,
           "staged": f"{INPUTS}:{STAGED}", "controls": controls, "pilot": pilot} | res
    (HERE / "results").mkdir(exist_ok=True)
    with open(dest := HERE / "results" / f"plant_stage_{stamp}.json", "x") as fh:
        fh.write(json.dumps(out, indent=1) + "\n")
    (HERE / "config").mkdir(exist_ok=True)
    with open(HERE / "config" / "plants_pilot_transcripts.json", "x") as fh:  # the extractor copy's `transcripts:` list
        fh.write(json.dumps({"inputs_run": STAGED, "affected": [p for p in pilot if p[0] != CONTROL], "controls": [p for p in pilot if p[0] == CONTROL],
                             "inputs_sha256": res["inputs_sha256"]}, indent=1) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k != "per run"}, indent=1)); print(dest)
