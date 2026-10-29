"""POST HOC (PLAN_POSTHOC.md in this folder): per-output L63 scores for B-dev's runs from its stored activations, with
no extraction. One H100, as B-dev's score() ran: B-dev's own refit() (imported unchanged), then every matched output
scored. Gate: each run's mean reproduces B-dev's stored L63 before / after / planted_shift within 1e-6. Records hold
keys and numbers only; no text leaves the container.

  modal deploy experiments/2026-10-11_q3_q11_remainder/run_q11_rescore_modal.py
  PYTHONPATH=src uv run --no-project --python 3.12 --with 'modal>=1.4.3' --with numpy python \\
      experiments/2026-10-11_q3_q11_remainder/run_q11_rescore_modal.py --records-out <path outside the repo>
"""

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).parent
BDEV = HERE.parent / "2026-10-11_claimB_slips_evened_probe"
sys.path.insert(0, str(BDEV) if modal.is_local() else "/root")
import run_slips_evened_modal as B  # noqa: E402

APP = "probe-shift_aswin-surya_sub-q3-q11rem_rescore_kill-2026-11-11"
OUT, STORED = "claimB/q11_remainder_v1", f"{B.OUT}/result_bdd71ed1.json"
B_SCRIPT = "0abee4a4d8730c0d3330e4702d179dfd8d3a11a228279e18e84fc4baa1d4cd06"
B_DEPS = "33d539e9a646c5adfeb3b8069c708c3f696f1ac794e09c26a281448bf0c6a65d"
TOL = 1e-6

app = modal.App(APP)
image = B.image.add_local_file(BDEV / "run_slips_evened_modal.py", "/root/run_slips_evened_modal.py")


def kind_of(tool, cmd) -> str:
    if tool == "text_editor":
        return cmd if cmd in ("view", "create", "str_replace", "insert") else "editor_other"
    if tool == "bash":
        return "shell"
    return "other"


def step_action(msgs) -> dict:
    """(step, action) -> tool-message index, counted as probe_shift.reader._layout counts: step = agent turn, action = tool
    message since that turn."""
    out, step, action = {}, -1, 0
    for i, m in enumerate(msgs):
        if m["role"] == "assistant":
            step, action = step + 1, 0
        elif m["role"] == "tool":
            out[(step, action)] = i
            action += 1
    return out


@app.function(image=image, gpu="H100!", cpu=8, memory=65536, volumes={"/acts": B.P.vol}, timeout=20 * 60, retries=0)
def rescore(env_to_fold: dict) -> dict:
    import numpy as np
    import torch

    from probe_shift.q1_eval import step_table

    M = B.M
    B.P.vol.reload()
    if (B.SCRIPT_SHA, B.DEPS_SHA) != (B_SCRIPT, B_DEPS):
        raise ValueError("B-dev's code here differs from the code that ran")
    base = Path("/acts") / B.OUT
    stored = {(x["cell"], x["arm"]): x for x in json.loads((Path("/acts") / STORED).read_text())["per_run"]}
    pruns = json.loads((Path("/acts") / M.PACK / "runs.L63.json").read_text())
    t = step_table(pruns, "output_end")
    arm_of = np.array([pruns[i]["arm"] for i in t["run"]])
    efold = np.array([env_to_fold[pruns[i]["environment"]] for i in t["run"]])
    XP = {63: np.load(Path("/acts") / M.PACK / "L63_output_end.npy")}
    probes, fits = {}, {}
    for e in B.DEV:
        probes[e], fits[e] = B.refit(e, 63, env_to_fold, arm_of, efold, XP)

    def s(e, X):
        mean, std, W, b = probes[e]
        return ((((torch.from_numpy(np.asarray(X)).cuda().double() - mean) / std * M.C**0.5) @ W + b)[:, 0]).cpu().numpy()

    runs = B.dev_runs()
    key_of = {pruns[i]["key"]: i for i in range(len(pruns))}
    rows_of = {}
    for k in range(len(t["run"])):
        rows_of.setdefault(int(t["run"][k]), {})[(int(t["step"][k]), int(t["action"][k]))] = k
    files = {}
    for f in (base / "runs").glob("*.npz"):
        z = np.load(f)
        m = json.loads(str(z["meta"]))
        if (m["script_sha256"], m.get("deps_sha256")) != (B_SCRIPT, B_DEPS):
            raise ValueError(f"{f.name} was made by other code")
        files[(m["cell"], m["arm"], m["version"])] = (m, z)
    recs, worst, tool_mismatch, checked, raw_tools = [], {"before": 0.0, "after": 0.0, "planted_shift": 0.0}, 0, 0, {}
    nonfinite, planted_compared = 0, 0

    def note(name, mine, theirs):
        nonlocal nonfinite
        d = abs(mine - theirs) if theirs is not None else float("nan")
        if not np.isfinite(d):
            nonfinite += 1
        else:
            worst[name] = max(worst[name], d)
    for (cell, arm), r in sorted(runs.items()):
        e = r["environment"]
        pr = rows_of[key_of[r["key"]]]
        ev, pl = files.get((cell, arm, "evened")), files.get((cell, arm, "planted"))
        rng = {tuple(x) for x in ev[0]["range"]} if ev else set()
        b_steps = sorted(pr)
        a_steps = [tuple(x) for x in ev[0]["steps"]] if ev else b_steps
        common = sorted((set(b_steps) & set(a_steps)) - rng)
        if not common:
            continue
        msgs = r["messages"]
        sa, calls = step_action(msgs), B.calls_of(msgs)
        ev_msgs = B.evened(msgs)[0]
        pl_msgs = B.planted(ev_msgs)[0] if pl else None
        before = s(e, XP[63][[pr[x] for x in common]])
        ai = {x: k for k, x in enumerate(a_steps)}
        after = s(e, ev[1]["L63"][[ai[x] for x in common]]) if ev else before
        planted = {}
        if pl:
            p_steps = [tuple(x) for x in pl[0]["steps"]]
            pi = {x: k for k, x in enumerate(p_steps)}
            pc = sorted(set(common) & set(p_steps))
            planted = dict(zip(pc, s(e, pl[1]["L63"][[pi[x] for x in pc]])))
        st = stored[(cell, arm)]
        note("before", float(before.mean()), st["L63|before"])
        note("after", float(after.mean()), st["L63|after"])
        if pl:
            a_of = dict(zip(common, after))
            shift = float(np.mean([planted[x] for x in planted]) - np.mean([a_of[x] for x in planted])) if planted else float("nan")
            note("planted_shift", shift, st.get("L63|planted_shift"))
            planted_compared += 1
        for k, x in enumerate(common):
            i = sa[x]
            tool, cmd = calls.get(i, (msgs[i].get("tool_name"), None))
            checked += 1
            tool_mismatch += tool != t["tool"][pr[x]]
            raw_tools[tool] = raw_tools.get(tool, 0) + 1
            touched = isinstance(msgs[i]["content"], str) and ev_msgs[i]["content"] != msgs[i]["content"]
            edited = pl_msgs is not None and pl_msgs[i]["content"] != ev_msgs[i]["content"]
            recs.append({"cell": cell, "arm": arm, "environment": e, "main_task": r["main_task"], "step": x[0], "action": x[1],
                         "tool": tool, "kind": kind_of(tool, cmd), "touched": bool(touched), "plant_edited": bool(edited),
                         "before": float(before[k]), "after": float(after[k]),
                         "planted": float(planted[x]) if x in planted else None})
    planted_stored = sum(x.get("L63|planted_shift") is not None for x in stored.values())
    scored = len({(x["cell"], x["arm"]) for x in recs})
    gate = {"max_abs_diff": worst, "tol": TOL, "nonfinite_comparisons": nonfinite, "tool_mismatches": int(tool_mismatch),
            "outputs_checked": checked, "runs_scored": scored, "runs_stored": len(stored), "planted_compared": planted_compared,
            "planted_stored": planted_stored,
            "passed": bool(max(worst.values()) <= TOL and nonfinite == 0 and tool_mismatch == 0 and scored == len(stored)
                           and planted_compared == planted_stored == 100)}
    res = {"gate": gate, "fits": fits, "raw_tool_counts": raw_tools, "b_script_sha256": B_SCRIPT, "b_deps_sha256": B_DEPS}
    if not gate["passed"]:
        return res | {"records": None}
    body = json.dumps(recs).encode()
    (Path("/acts") / OUT).mkdir(parents=True, exist_ok=True)
    (Path("/acts") / OUT / "records.json").write_bytes(body)
    B.P.vol.commit()
    return res | {"records": recs, "records_sha256": hashlib.sha256(body).hexdigest(), "volume_path": f"{OUT}/records.json"}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--records-out", required=True)
    args = ap.parse_args()
    folds = json.loads((B.M.REPO / "splits" / "folds_v2.json").read_text())["environment_folds"]["environment_to_fold"]
    call = modal.Function.from_name(APP, "rescore").spawn(folds)
    print("spawned", call.object_id, flush=True)
    res = call.get()
    recs = res.pop("records")
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    if recs is not None:
        Path(args.records_out).write_text(json.dumps(recs))
        res["records_local_copy_sha256"] = hashlib.sha256(Path(args.records_out).read_bytes()).hexdigest()
        res["records_n"] = len(recs)
    with open(HERE / "results" / f"q11_rescore_gate_POSTHOC_{stamp}.json", "x") as fh:
        fh.write(json.dumps(res, indent=1))
    print(json.dumps(res["gate"], indent=1))
