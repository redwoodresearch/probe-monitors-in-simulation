"""POST-HOC check (plan Amendment 12): how often a command-end (call_end) readout lands on the last digit of a trailing
`timeout` parameter, per arm, over each run's first 3 and first 5 commands and the whole run. Modal CPU reads the staged
call metadata (each call_end's last parameter name and its readout token id); the statistics run locally. Keys and
numbers only: no command text is decoded or printed; the tokenizer decodes only vocabulary entries, to find digit tokens.

A readout is a timeout digit when its call's last parameter is `timeout` and its token is a digit (attack_eval.
call_end_kind's rule 1, on the staged metadata). Calls and order as the claim C cache: kind call_end, readout not None,
(turn, call index) order; epochs the cache skips (redrawn, or attempt_kind "rerun") are left out.

  PYTHONPATH=src uv run --no-project --python 3.12 --with 'modal>=1.4.3' --with numpy modal run \
      experiments/2026-10-08_q3_onpolicy_figures/run_count_timeout_readout.py --labels <label files>
"""

import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).parent
REPO = HERE.parents[1] if len(HERE.parents) > 1 else HERE
ACTS, HF_CACHE = "probe-shift_aswin-surya_sub-q3onpol_acts_kill-2026-11-08", "probe-shift_aswin-surya_hf-cache"
MODEL, REVISION = "Qwen/Qwen3.5-27B", "fc05daec18b0a78c049392ed2e771dde82bdf654"
T196 = "experiments/2026-10-07_phase2_attack_check/config/pilot_tasks_amendment9_all196.json"
BASES = ["rerun_strict_segends_v2_scrub1", "scaleup_canary_segends_v1_scrub1", "scaleup_l2_segends_v1_scrub1", "scaleup_topup_segends_v1_scrub1"]
SOURCES = ([("run1", f"{b}_swapsys", "attack") for b in BASES] + [("run2", "second_attack_segends_v1_scrub1_swapsys", "attack")]
           + [("sim", f"{b}_swapsys2", "sim_attack@september") for b in BASES]
           + [(h, b, h) for b in BASES for h in ("honest_a", "honest_b")])
KS, B, BAR = (3, 5), 2000, 0.05

app = modal.App("probe-shift_aswin-surya_sub-q3onpol_timeoutcount_kill-2026-11-10")
image = (modal.Image.debian_slim(python_version="3.12").uv_pip_install("transformers==5.19.0", "jinja2", "numpy")
         .env({"HF_HOME": "/hf", "HF_HUB_OFFLINE": "1"}))


@app.function(image=image, cpu=4, memory=32768, timeout=1800, retries=0,
              volumes={"/acts": modal.Volume.from_name(ACTS, version=2).read_only(), "/hf": modal.Volume.from_name(HF_CACHE).read_only()})
def count() -> dict:
    import numpy as np
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    digit = np.array([tok.decode([i]).strip().isdigit() for i in range(len(tok))])
    runs, seen = {}, {}
    for world, staging, arm in SOURCES:
        for line in (Path("/acts") / staging / "inputs" / f"{arm}.jsonl").read_text().splitlines():
            ep = json.loads(line)
            if ep.get("redrawn_because") or ep.get("attempt_kind") == "rerun":
                continue
            key = f"{world}|{ep['task']}"
            if seen.setdefault(key, ep["run"]) != ep["run"]:
                raise ValueError(f"{key}: several counted runs")
            for c in ep["calls"]:
                if c.get("kind") == "call_end" and c["readout"] is not None:
                    runs.setdefault(key, []).append(((c["turn"], c.get("call_index", 0)), bool(c["param"] == "timeout" and digit[ep["ids"][c["readout"]]])))
    out = {}
    for key, cs in runs.items():
        flags = [f for _, f in sorted(cs)]
        if len({o for o, _ in cs}) != len(cs):
            raise ValueError(f"{key}: two call_end readouts share a (turn, call index)")
        out[key] = {"n": len(flags), "all": sum(flags), **{f"first{k}": sum(flags[:k]) for k in KS}}
    return out


@app.local_entrypoint()
def main(labels: str):
    import numpy as np

    from analyze_retrain import frozen_folds, interval, task_weights

    lab = {}
    for f in labels.split(","):
        for r in map(json.loads, Path(f).read_text().splitlines()):
            w = {"attack": "run2" if "second_attack" in Path(f).name else "run1", "sim_attack@september": "sim"}.get(r["arm"])
            if w and r.get("label"):
                if f"{w}|{r['task']}" in lab:
                    raise ValueError(f"two labels for {w} {r['task']}")
                lab[f"{w}|{r['task']}"] = r["label"]
    f196 = frozen_folds(json.loads((REPO / T196).read_text()))
    tasks = sorted(f196)
    ti = {t: i for i, t in enumerate(tasks)}
    W = [task_weights([f196[t] for t in tasks], b) for b in range(1, B + 1)]
    one = np.ones(len(tasks))
    per = count.remote()
    res = {}
    for scope in ("first3", "first5", "all"):
        k = None if scope == "all" else int(scope[5:])
        ok = lambda w, t: f"{w}|{t}" in per and (k is None or per[f"{w}|{t}"]["n"] >= k)
        att = lambda w: {t for t in tasks if ok(w, t) and lab.get(f"{w}|{t}") == "attempted"}
        cohort = att("run1") & att("run2") & att("sim")  # the first-k analysis's cohort: all three attempted, with k commands
        arms = {w: sorted(cohort) for w in ("run1", "run2", "sim")} | {h: [t for t in tasks if ok(h, t)] for h in ("honest_a", "honest_b")}

        def share(w, wt):
            ts = arms[w]
            num = sum(wt[ti[t]] * per[f"{w}|{t}"][scope] for t in ts)
            den = sum(wt[ti[t]] * (k or per[f"{w}|{t}"]["n"]) for t in ts)
            return None if den == 0 else float(num / den)

        diff = lambda a, b: (lambda wt: None if share(a, wt) is None or share(b, wt) is None else share(a, wt) - share(b, wt))
        d_ah, d_as = diff("run1", "honest_a"), diff("run1", "sim")
        iv_ah, iv_as = interval([d_ah(w) for w in W]), interval([d_as(w) for w in W])
        res[scope] = {"runs": {w: len(v) for w, v in arms.items()},
                      "commands": {w: int(sum(k or per[f"{w}|{t}"]["n"] for t in v)) for w, v in arms.items()},
                      "timeout_readouts": {w: int(sum(per[f"{w}|{t}"][scope] for t in v)) for w, v in arms.items()},
                      "share": {w: share(w, one) for w in arms},
                      "run1_minus_honest_a": {"value": d_ah(one), "interval": iv_ah},
                      "run1_minus_sim": {"value": d_as(one), "interval": iv_as}}
        if scope == "first3":
            v = res[scope]["run1_minus_honest_a"]
            res[scope]["differ_clearly"] = bool(not iv_ah["invalid"] and (iv_ah["ci95"][0] > 0 or iv_ah["ci95"][1] < 0) and abs(v["value"]) >= BAR)
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    path = HERE / "results" / f"timeout_readout_count_POSTHOC_{stamp}.json"
    with path.open("x") as fh:
        fh.write(json.dumps({"amendment": "12 (post hoc)", "rule": "differ clearly = run 1 minus honest_a share at k = 3 has a task-grouped 95% interval excluding 0 and |difference| >= 5 points",
                             "results": res}, indent=1, default=float))
    for scope, v in res.items():
        print(scope, json.dumps({"share": v["share"], "run1_minus_honest_a": [v["run1_minus_honest_a"]["value"], v["run1_minus_honest_a"]["interval"]["ci95"]],
                                 "run1_minus_sim": [v["run1_minus_sim"]["value"], v["run1_minus_sim"]["interval"]["ci95"]], "runs": v["runs"],
                                 "differ_clearly": v.get("differ_clearly")}, default=float))
    print(path.name)
