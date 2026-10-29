"""0d for Aswin (10-16 ~12:2x): do run limits or models differ across task sets? For each set (S_old, S_new, S_newmains, S_paper493)
x world (run1, run2, honest_a, honest_b, sim, sim_honest), read the .eval HEADER ONLY (no samples) of 3 runs picked by sha256(key + seed)
from the INPUTS_FULL-pinned inventories, on Modal CPU. Only the log header is deserialized (header_only=True: no samples); inside the
container, the returned output is restricted to: Inspect's eval config limits (numbers), the policy model id, numeric generate-config
limits, and numeric values (or model-id strings under 'model' keys) from task_args / metadata / plan steps whose key is a short identifier
naming a step, limit, timeout, token, model or retry setting. Every emitted key and string passes a full-match, length-bounded grammar.
  modal run run_eval_limits_check.py --inputs-full <path> --commit <sha> --out results/eval_limits_<ts>.json
"""
import hashlib
import json
import re
import subprocess

import modal

APP = "probe-shift_aswin-surya_sub-sp493_evallimits_kill-2026-10-23"
VOLS = {"chain": "probe-shift_aswin-surya_sub-q3su311_acts_kill-2026-11-14",
        "q4own": "probe-shift_aswin-surya_sub-q4own_evals_kill-2026-11-13"}
WORLDS = ("run1", "run2", "honest_a", "honest_b", "sim", "sim_honest")
SEED = "eval-limits-v1"
KEEP = re.compile(r"(?i)(step|limit|timeout|token|model|retries|max_)")
MODEL_ID = re.compile(r"[A-Za-z0-9_.-]{1,40}/[A-Za-z0-9_.:/-]{1,80}")  # fullmatch only, e.g. vllm/Qwen/Qwen3.5-27B, openai/gpt-6-astra
NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_.-]{0,60}")  # fullmatch only (Astra r1 #2: no trailing newline, bounded length)
SETS = ("S_old", "S_new", "S_newmains", "S_paper493")
image = modal.Image.debian_slim(python_version="3.12").pip_install("inspect_ai==0.3.251", "zstandard==0.25.0")
app = modal.App(APP, image=image)
vols = {k: modal.Volume.from_name(v, version=2) for k, v in VOLS.items()}


def build(inputs_full, commit):
    full = json.loads(subprocess.check_output(["git", "show", f"{commit}:{inputs_full}"]))
    pick = {}
    for name in ("S_new", "S_old", "S_paper493", "S_paper493_w23"):  # S_old's inventory also holds S_newmains (split by origin)
        p = full["inventories"][name]
        for r in json.loads(subprocess.check_output(["git", "show", f"{p['commit']}:{p['path']}"]))["runs"]:
            s = r.get("origin") or r.get("set") or name
            if name == "S_old" and s == "S_old":  # inventory_b_fix_v1 holds S_old AND S_newmains with no set/origin field: split by task id
                s = "S_newmains" if r["task"].startswith("nm") else "S_old"
            s = "S_paper493" if str(s).startswith("S_paper493") else s
            if r["world"] in WORLDS:
                pick.setdefault((s, r["world"]), []).append(r["key"])
    extra = sorted(f"{s}|{w}" for s, w in pick if s not in SETS)
    if extra:
        raise SystemExit(f"unexpected sets in the inventories: {extra}")
    rows = []
    for (s, w), ks in sorted(pick.items()):
        paths = sorted({re.sub(r"#e\d+$", "", k) for k in ks}, key=lambda p: hashlib.sha256((p + SEED).encode()).hexdigest())  # distinct files (Astra r1 #3)
        rows += [{"set": s, "world": w, "key": p, "path": p} for p in paths[:3]]
    short = {f"{s}|{w}": len({r["path"] for r in rows if (r["set"], r["world"]) == (s, w)}) for s in SETS for w in WORLDS}
    if any(v != 3 for v in short.values()):
        raise SystemExit(f"cells without 3 distinct .eval files: {[k for k, v in short.items() if v != 3]}")
    return {"inputs_full": f"{inputs_full} @ {commit}", "rows": rows, "n_by_set_world": {f"{s}|{w}": len(ks) for (s, w), ks in sorted(pick.items())}}


@app.function(volumes={f"/{k}": v.read_only() for k, v in vols.items()}, cpu=2, memory=4096, timeout=1800, retries=0)
def header(chunk: list) -> list:
    import os

    from inspect_ai.log import read_eval_log

    def scal(d):
        out, names = {}, []
        for k, v in (d or {}).items():  # emitted keys must pass NAME too (Astra r1 #1); rejected keys are dropped, only counted
            if not (isinstance(k, str) and NAME.fullmatch(k)):
                names.append("<non-identifier key>")
                continue
            names.append(k)
            if not KEEP.search(k):
                continue
            if isinstance(v, (int, float, bool)):
                out[k] = v
            elif isinstance(v, str) and "model" in k.lower() and MODEL_ID.fullmatch(v):
                out[k] = v
        return out, sorted(names)

    res = []
    for r in chunk:
        rec = {k: r[k] for k in ("set", "world", "key")}
        hit = next((v for v in VOLS if os.path.isfile(f"/{v}/evals_v1/{r['path']}")), None)
        if hit is None:
            res.append(rec | {"missing": True})
            continue
        try:
            log = read_eval_log(f"/{hit}/evals_v1/{r['path']}", header_only=True)
            e = log.eval
            cfg = e.config
            gc = e.model_generate_config.model_dump(exclude_none=True) if e.model_generate_config else {}
            ta, ta_names = scal(e.task_args)
            md, md_names = scal(e.metadata)
            steps = [{"solver": s.solver if isinstance(s.solver, str) and NAME.fullmatch(s.solver) else "<non-identifier>",
                      **dict(zip(("params", "param_names"), scal(s.params)))}
                     for s in (log.plan.steps if log.plan else [])]
            res.append(rec | {"volume": hit, "status": log.status, "model": e.model if isinstance(e.model, str) and MODEL_ID.fullmatch(e.model) else "<non-model-id>",
                              "limits": {x: getattr(cfg, x, None) for x in ("time_limit", "message_limit", "token_limit", "working_limit", "epochs")},
                              "generate_config": {k: v for k, v in gc.items() if KEEP.search(k) and isinstance(v, (int, float, bool))},
                              "task_args": ta, "task_arg_names": ta_names, "metadata": md, "metadata_names": md_names, "plan": steps,
                              "n_model_roles": len(e.model_roles or {}) if hasattr(e, "model_roles") else None})  # names not returned (Astra r1 #1)
        except Exception as x:
            res.append(rec | {"volume": hit, "read_error": type(x).__name__})
    return res


@app.function(cpu=1, memory=1024, timeout=1800, retries=0)
def drive(rows: list) -> list:
    return [x for part in header.map([rows[i:i + 12] for i in range(0, len(rows), 12)]) for x in part]


@app.local_entrypoint()
def main(inputs_full: str, commit: str, out: str):
    f = open(out, "x")
    spec = build(inputs_full, commit)
    res = drive.remote(spec["rows"])  # the one remote call
    json.dump({"what": __doc__.strip().splitlines()[0], "app": APP, "volumes": VOLS, "seed": SEED} | {k: v for k, v in spec.items() if k != "rows"}
              | {"rows": res}, f, indent=1)
    f.close()
    ok = {f"{s}|{w}": sum(1 for x in res if (x["set"], x["world"]) == (s, w) and "limits" in x) for s in SETS for w in WORLDS}
    for x in res:
        print(x["set"], x["world"], x.get("missing") or x.get("read_error") or (x["model"], x["limits"], x["generate_config"], x["task_args"]))
    print("headers read per cell:", ok)
    if any(v != 3 for v in ok.values()):  # Astra r1 #4: fail after writing the record when any cell lacks 3 successful header reads
        raise SystemExit(1)
