"""Q4, RULE.md Amendment 8 test 4 (POST HOC; Modal CPU; counts only, no text anywhere): the tool-output type mix and the
simulator tells fix-full repairs, in the simulated-attack and honest runs, Sol's vs Qwen3.5's own, within the j = 21 window
(the first 21 tool outputs of a run, in order, as `fixes.calls_of` pairs them) and over the whole run.
- Reader messages, original text (no scrub): Sol's from the v4 staging (`2026-10-06_la_segment-end-v4_qwen3-32b_v1/inputs`,
  #36's sim-attack and real-honest runs, by key); own runs' from their .eval files through stage_q4b.py's loader and
  `transcripts.reader_messages` (set main: sim_attack@september, honest_a).
- Output type: text_editor create / str_replace / insert / view / other command; bash; other tool.
- Tell: an output the fixed-sim rules change (`fixes.apply(messages, ("R1", "R2", "R5"))`: R1 the insert format, R2 the
  str_replace gutter and trailing newline, R5 dpkg's lines). Visible tell: the change survives the reader's rendering,
  which strips every message's content (`reader.render`), i.e. original.strip() != fixed.strip().
Reported: per run means and totals by dataset and arm. Whitelisted keys only.

  PYTHONPATH=src uv run --no-project --python 3.12 --with modal==1.6.1 modal run --detach \\
      experiments/2026-10-13_q4_why_weaker_POSTHOC/run_q4d_tells.py
"""

import json
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1] if len(HERE.parents) > 1 else HERE
APP = "probe-shift_aswin-surya_sub-q4own_tells_kill-2026-11-13"
OWN, LA, HF = "probe-shift_aswin-surya_sub-q4own_evals_kill-2026-11-13", "probe-shift_aswin-surya_sub-la-activations", "probe-shift_aswin-surya_hf-cache"
SOL_STAGED, R36 = "2026-10-06_la_segment-end-v4_qwen3-32b_v1", "experiments/2026-10-06_q3_attack_detector/results/run_scores_2026-10-06_181548/runs.json"
WINDOW, RULES = 21, ("R1", "R2", "R5")
TYPES = ("te_create", "te_str_replace", "te_insert", "te_view", "te_other", "bash", "other")
LABEL_FILES = ["experiments/2026-10-08_q3_onpolicy_figures/results/phase2_attempt_labels_2026-10-09_000523_rerun_strict_selfreport_2026-10-10_111809.jsonl",
               "experiments/2026-10-08_q3_onpolicy_figures/results/phase2_attempt_labels_2026-10-09_173815_scaleup_canary_selfreport_2026-10-10_111817.jsonl",
               "experiments/2026-10-08_q3_onpolicy_figures/results/phase2_attempt_labels_2026-10-10_001314_scaleup_topup_selfreport_2026-10-10_111821.jsonl",
               "experiments/2026-10-08_q3_onpolicy_figures/results/phase2_attempt_labels_2026-10-10_001314_second_attack_2026-10-10_111840.jsonl"]
LABEL_S3 = Path("/private/tmp/claude-501/-Users-aswinsurya-probe-distribution-shift-from-sims/1faf3281-1269-46cf-883f-277091c9aa65/scratchpad/crossread_labels"
                "/phase2_attempt_labels_2026-10-09_182505_scaleup_selfreport_2026-10-10_111838.jsonl")
MODS = ["experiments/2026-10-09_fixed_sim/fixes.py", "experiments/2026-10-12_plantslip_transfer/plant_both.py",
        "experiments/2026-10-07_q2_rendering_swap/rerender.py", "experiments/2026-10-07_q2_rendering_swap/str_replace_tells.py",
        "experiments/2026-10-07_phase2_attack_check/config/pilot_tasks_amendment9_all196.json"]
app = modal.App(APP)
image = (modal.Image.debian_slim(python_version="3.13").uv_pip_install("transformers==5.19.0", "numpy", "jinja2", "zstandard", "boto3", "pyyaml")
         .env({"HF_HOME": "/hf", "HF_HUB_OFFLINE": "1"}).add_local_python_source("probe_shift")
         .add_local_file(HERE / "stage_q4b.py", "/root/repo/q4/stage_q4b.py").add_local_file(HERE / "evalzip.py", "/root/repo/q4/evalzip.py")
         .add_local_file(HERE / "results" / "manifest_q4b_eval_upload_v1.json", "/root/repo/manifest_q4b_eval_upload_v1.json")
         .add_local_file(LABEL_S3, f"/root/repo/labels/{LABEL_S3.name}").add_local_file(REPO / R36, "/root/runs36.json"))
for m in MODS + LABEL_FILES:
    image = image.add_local_file(REPO / m, f"/root/repo/{m}")
VOLS = {"/ev": modal.Volume.from_name(OWN, version=2), "/acts": modal.Volume.from_name(LA, version=2).read_only(), "/hf": modal.Volume.from_name(HF).read_only()}


def _fixes():
    import sys

    sys.path.insert(0, "/root/repo/experiments/2026-10-09_fixed_sim")
    import fixes

    return fixes


def kind_of(tool, args):
    if tool == "text_editor":
        c = (args or {}).get("command")
        return f"te_{c}" if c in ("create", "str_replace", "insert", "view") else "te_other"
    return "bash" if tool == "bash" else "other"


def count_run(msgs):
    """Per run: output types and tells in the first WINDOW outputs and over the whole run. Counts only."""
    from collections import Counter

    F = _fixes()
    outs = F.calls_of(msgs)
    new, recs = F.apply(msgs, RULES)
    changed = {}
    for r in recs:
        if r.get("changed") and r.get("msg") is not None:
            changed.setdefault(r["msg"], set()).add(r["rule"])
    res = {}
    for scope, sel in (("window", outs[:WINDOW]), ("whole", outs)):
        c = Counter(outputs=len(sel))
        for o in sel:
            k = kind_of(o["tool"], o["args"])
            c[f"type|{k}"] += 1
            rules = changed.get(o["msg"], set())
            if rules:
                old, nw = msgs[o["msg"]]["content"], new[o["msg"]]["content"]
                vis = isinstance(old, str) and isinstance(nw, str) and old.strip() != nw.strip()
                c["tell"] += 1
                c["tell_visible"] += vis
                c[f"tell|{k}"] += 1
                c[f"tell_visible|{k}"] += vis
                for r in sorted(rules):
                    c[f"tell_rule|{r}"] += 1
                    c[f"tell_rule_visible|{r}"] += vis
        res[scope] = dict(c)
    return res


@app.function(image=image, cpu=2, memory=8192, timeout=1800, retries=0, volumes=VOLS)
def sol_shard(i: int) -> dict:
    keys = {(r["arm"], r["key"]) for r in json.loads(Path("/root/runs36.json").read_text()) if r["arm"] in ("sim-attack", "real-honest")}
    out, failed = [], 0
    for line in Path("/acts", SOL_STAGED, "inputs", f"shard_{i}.jsonl").open():
        r = json.loads(line)
        if (r["arm"], r["key"]) not in keys:
            continue
        try:
            out.append({"arm": r["arm"], **count_run(r["messages"])})
        except Exception as e:  # counts only: the error type, never text
            failed += 1
            out.append({"arm": r["arm"], "error": type(e).__name__})
    return {"runs": out, "failed": failed}


@app.function(image=image, cpu=2, memory=8192, timeout=1800, retries=0, volumes=VOLS)
def own_part(k: int, n: int) -> dict:
    import sys

    sys.path.insert(0, "/root/repo/q4")
    import stage_q4b as S

    counted = S.counted_attempts("main")
    todo = sorted(x for x in counted if x[0] in ("sim_attack@september", "honest_a") and counted[x]["launch"] in S.MAIN)[k::n]
    out, failed = [], 0
    for arm_dir, task in todo:
        try:
            s = S.load(counted[(arm_dir, task)]["eval"])
            msgs = S.transcripts.reader_messages(s, S.transcripts.system_prompt(s))
            out.append({"arm": arm_dir, **count_run(msgs)})
        except Exception as e:
            failed += 1
            out.append({"arm": arm_dir, "error": type(e).__name__})
    return {"runs": out, "failed": failed}


def summarize(runs):
    from collections import Counter, defaultdict

    by = defaultdict(list)
    for r in runs:
        by[r["arm"]].append(r)
    out = {}
    for arm, rs in by.items():
        ok = [r for r in rs if "error" not in r]
        o = {"runs": len(rs), "errors": len(rs) - len(ok), "error_types": dict(Counter(r["error"] for r in rs if "error" in r))}
        for scope in ("window", "whole"):
            tot = Counter()
            for r in ok:
                tot.update(r[scope])
            o[scope] = {"per_run_mean": {k: round(v / max(len(ok), 1), 4) for k, v in sorted(tot.items())}, "totals": dict(sorted(tot.items()))}
        out[arm] = o
    return out


@app.local_entrypoint()
def main():
    g = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True, check=True, cwd=REPO).stdout.strip()  # noqa: E731
    if g("status", "--porcelain", "--untracked-files=no"):
        raise SystemExit("refusing to run from a dirty tree")
    head = g("rev-parse", "HEAD")  # read once, before any spawn
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    own = own_part.starmap([(k, 16) for k in range(16)])
    sol = list(sol_shard.map(range(64)))
    own = list(own)
    res = {"label": "POST HOC (RULE.md Amendment 8 test 4); counts only", "git": head, "stamp": stamp, "window": WINDOW, "rules": RULES,
           "sol": summarize([r for x in sol for r in x["runs"]]), "own": summarize([r for x in own for r in x["runs"]])}
    vol = modal.Volume.from_name(OWN, version=2)
    import io

    with vol.batch_upload(force=True) as b:
        b.put_file(io.BytesIO(json.dumps(res, indent=1).encode()), f"q4d_work/tells_{stamp}.json")
    dest = HERE / "results" / f"q4d_tells_POSTHOC_{stamp}.json"
    dest.open("x").write(json.dumps(res, indent=1))
    print("git", head)
    for ds in ("sol", "own"):
        for arm, o in res[ds].items():
            w = o["window"]["per_run_mean"]
            print(f"{ds:3s} {arm:22s} runs {o['runs']} errors {o['errors']} | window per run: outputs {w.get('outputs', 0):.2f} tell {w.get('tell', 0):.3f} "
                  f"visible {w.get('tell_visible', 0):.3f} | " + " ".join(f"{t} {w.get('type|' + t, 0):.2f}/{w.get('tell_visible|' + t, 0):.3f}" for t in TYPES)
                  + " | rules " + " ".join(f"{r} {w.get('tell_rule|' + r, 0):.3f}/{w.get('tell_rule_visible|' + r, 0):.3f}" for r in RULES))
    print(dest.name)
