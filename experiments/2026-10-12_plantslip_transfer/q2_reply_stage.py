"""Planted-slip transfer, q2's reply pairs P4-P6 (RULE.md, Amendment 1): staging. One Modal CPU container. It applies
`plant_both()` (this folder; both slips in one pass) to the 939 real e2 and 939 real e1 runs of #36's cohort, from the
staged messages every reply reader read (#36's dump's inputs), and stages two runs on the shared activations Volume:
- `2026-10-12_q2_plantslip_reply_controls_inputs_v1`: 20 seeded unplanted e2 runs (arm `real-attack-e2-control`), for
  each pair's reproduction gate;
- `2026-10-12_q2_plantslip_reply_planted_inputs_v1`: every e1 and e2 run the plant changed (arms
  `real-attack-e2-planted`, `real-attack-e1-planted`), with the changed (message, step, action, slip) list.
Plant reach is counted per arm: runs and outputs changed, outputs per slip, and, per reader template, the planted
str_replace outputs whose appended trailing newline survives into the rendered prompt (a minimal one-call rendering of
each planted output). Counts and keys only leave the container.

  PYTHONPATH=src uv run --no-project --python 3.12 --with "modal==1.6.1" modal run \\
      experiments/2026-10-12_plantslip_transfer/q2_reply_stage.py --runs <#36 runs.json>
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
         "fixes.py": EXP / "2026-10-09_fixed_sim", "plant_both.py": EXP / "2026-10-12_plantslip_transfer"}  # named, not HERE: in the container HERE is /root
APP = "probe-shift_aswin-surya_sub-q2-plantq2_stage_kill-2026-11-12"
ACTS = "probe-shift_aswin-surya_sub-la-activations"  # the staged messages the readers read; the new staged runs go here too
SOURCE = "2026-10-06_la_segment-end_qwen3-32b_v1"
CONTROLS_RUN, PLANTED_RUN = "2026-10-12_q2_plantslip_reply_controls_inputs_v1", "2026-10-12_q2_plantslip_reply_planted_inputs_v1"
ARMS, CONTROL = {"real-attack-e2": "real-attack-e2-planted", "real-attack-e1": "real-attack-e1-planted"}, "real-attack-e2-control"
TOKENIZERS = {"Qwen3-32B": ("Qwen/Qwen3-32B", "9216db5781bf21249d130ec9da846c4624c16137"),
              "Olmo-3.1-32B": ("allenai/Olmo-3.1-32B-Instruct", "ac0587e4a7744a551c059d8cd17ba220bc940dae"),
              "Qwen3.5-27B": ("Qwen/Qwen3.5-27B", "fc05daec18b0a78c049392ed2e771dde82bdf654")}
SEED, N_CONTROLS, N_SHARD_FILES = "20261012", 20, 8
PIN = {"runs.json": "76b11486cc353ab1eec390d49ef7449d729d4039d1e85643541d6679d73c3b96"}  # #36's run list

app = modal.App(APP)
acts = modal.Volume.from_name(ACTS, version=2)
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
    from plant_both import plant_both
    return fixes, rerender, plant_both


@app.function(image=image, volumes={"/acts": acts}, cpu=16, memory=32768, timeout=3600, retries=0)
def stage(runs: list[dict], controls: list[str]) -> dict:
    try:
        return _stage(runs, controls)
    except Exception as e:  # nothing from the exception (which may quote staged text) crosses the boundary
        import traceback  # the failing frame's file and line, and a missing module's name: never staged text
        tb = traceback.extract_tb(e.__traceback__)[-1]
        raise RuntimeError(f"staging failed: {type(e).__name__} {getattr(e, 'name', '') or ''} at {Path(tb.filename).name}:{tb.lineno}") from None


def _stage(runs, controls):
    from collections import Counter

    from transformers import AutoTokenizer

    from probe_shift.reader import _layout
    fixes, rerender, plant_both = _mods()
    toks, tok_fail = {}, {}
    for name, (m, rv) in TOKENIZERS.items():
        try:
            toks[name] = AutoTokenizer.from_pretrained(m, revision=rv)
        except Exception as e:  # the reach check for that reader is then reported as unavailable
            tok_fail[name] = type(e).__name__
    if "Qwen3-32B" not in toks:
        raise ValueError("the Qwen3-32B tokenizer is needed for the token counts")
    want = {(r["arm"], r["key"]) for r in runs}
    items = {}
    for shard in sorted((Path("/acts") / SOURCE / "inputs").glob("shard_*.jsonl")):
        for line in shard.read_text().splitlines():
            it = json.loads(line)
            if (it["arm"], it["key"]) in want:
                items[(it["arm"], it["key"])] = it
    if set(items) != want:
        raise ValueError("a run is missing from the staged inputs")

    def survives(name, planted):  # does a planted output's final newline change the rendered prompt? (a trimming template: no)
        def render(content):
            msgs = [{"role": "user", "content": "x"},
                    {"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": {"name": "text_editor", "arguments": {"command": "str_replace"}}}]},
                    {"role": "tool", "content": content}]
            return toks[name].apply_chat_template(msgs, tokenize=False)
        return render(planted) != render(planted[:-1]) if planted.endswith("\n") else False
    n, staged_p, staged_c, per_run = Counter(), [], [], []
    for (arm, k) in sorted(want):
        it = items[(arm, k)]
        new, changed = plant_both(it["messages"], fixes, rerender)
        if not changed:
            n[(arm, "runs unchanged")] += 1
            continue
        diff = [i for i, (a, b) in enumerate(zip(it["messages"], new)) if a != b]  # only the changed tool messages' content
        assert len(new) == len(it["messages"]) and diff == [c[0] for c in changed]
        assert all({x for x in new[i] if new[i][x] != it["messages"][i][x]} == {"content"} for i in diff)
        n[(arm, "runs changed")] += 1; n[(arm, "outputs changed")] += len(changed)
        for i, step, action, slip in changed:
            n[(arm, f"outputs changed: {slip}")] += 1
            if slip == "gutter+newline":
                for name in toks:
                    n[(arm, f"gutter+newline outputs whose trailing newline reaches the {name} prompt")] += survives(name, new[i]["content"])
        t = len(_layout(toks["Qwen3-32B"], new)[2]["input_ids"])  # Qwen3-32B tokens: the extractors' shard order
        staged_p.append(it | {"arm": ARMS[arm], "messages": new, "tokens": t, "plant": "both", "orig_arm": arm, "changed": [list(c) for c in changed]})
        per_run.append({"arm": ARMS[arm], "orig_arm": arm, "key": k, "outputs changed": len(changed), "first changed (step, action)": list(changed[0][1:3])})
    for k in controls:
        it = items[("real-attack-e2", k)]
        staged_c.append(it | {"arm": CONTROL, "tokens": len(_layout(toks["Qwen3-32B"], it["messages"])[2]["input_ids"]), "plant": None, "orig_arm": "real-attack-e2"})
        n[(CONTROL, "runs")] += 1
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
    return {"counts": [{"arm": a, "count": c, "n": int(v)} for (a, c), v in sorted(n.items())], "per run": per_run, "inputs_sha256": shas,
            "planted": [{"key": x["key"], "orig_arm": x["orig_arm"], "arm": x["arm"]} for x in staged_p],
            "controls": [{"key": x["key"], "orig_arm": x["orig_arm"], "arm": x["arm"]} for x in staged_c],
            "reach check unavailable (tokenizer failed to load)": tok_fail,
            "Qwen3-32B tokens staged": {"planted": int(sum(x["tokens"] for x in staged_p)), "controls": int(sum(x["tokens"] for x in staged_c))}}


@app.local_entrypoint()
def main(runs: str):
    import subprocess
    for f in [Path(__file__).resolve(), HERE / "RULE.md"] + [folder / name for name, folder in FILES.items()]:
        if subprocess.run(["git", "ls-files", "--error-unmatch", str(f)], cwd=HERE, capture_output=True).returncode \
                or subprocess.run(["git", "diff", "--quiet", "HEAD", "--", str(f)], cwd=HERE).returncode:
            raise SystemExit(f"{f.name} differs from HEAD: commit it before running")
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip()
    if hashlib.sha256(Path(runs).read_bytes()).hexdigest() != PIN["runs.json"]:
        raise SystemExit("not #36's runs.json")
    rs = [r for r in json.loads(Path(runs).read_text()) if r["arm"] in ARMS]
    if sorted(sum(r["arm"] == a for r in rs) for a in ARMS) != [939, 939]:
        raise SystemExit("not 939 e1 and 939 e2 runs")
    rank = lambda k: hashlib.sha256(f"{SEED}|{k}".encode()).hexdigest()  # noqa: E731
    controls = sorted((r["key"] for r in rs if r["arm"] == "real-attack-e2"), key=rank)[:N_CONTROLS]
    res = stage.remote([{"arm": r["arm"], "key": r["key"]} for r in rs], controls)
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    out = {"plan": "RULE.md, Amendment 1 (this folder)", "git": head, "runs.json sha256": PIN["runs.json"], "source": SOURCE,
           "staged runs (Volume probe-shift_aswin-surya_sub-la-activations)": [CONTROLS_RUN, PLANTED_RUN], "control seed": SEED} | res
    (HERE / "results").mkdir(exist_ok=True)
    with open(dest := HERE / "results" / f"q2_reply_stage_{stamp}.json", "x") as fh:
        fh.write(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k not in ("per run", "planted", "controls")}, indent=1)); print(dest)
