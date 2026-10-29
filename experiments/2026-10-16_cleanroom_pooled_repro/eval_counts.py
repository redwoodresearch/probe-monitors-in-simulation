"""Step, tool-call and limit counts for the five arms of cell t325_model_registry (0d's request, 10-17; a1). Counts, enums and numbers
only: no message text, no argument, no error message leaves the job.

The five runs are taken from the pinned inventory (inventory_v3_2026-10-15.json @ b3f8dea7b, sha256 19306ffa…; INPUTS_FULL
inventories.S_new): run1, run2, honest_b, sim (attack) and sim_honest of t325_model_registry. Each .eval (a zip) is read from the "chain"
Volume at evals_v1/<launch>/<eval_rel> and its sha256 is checked against the inventory. Per run (revision 2, after review round 1):
  - header: eval status, and the numeric limits in eval.config (message / token / time / working limits) and numeric task_args;
  - per sample: messages by role (allowlisted role names), assistant turns, tool calls, Inspect's `limit` (allowlisted type, numbers),
    the policy's own stop (store.policy_ended_by: policy_timeout / out_of_context / other; the step cap and policy timeout are
    enforced by the policy, not by Inspect), whether an error is set (boolean only), total_time and working_time;
  - per model (from the sample's model events): completed calls (no error, at least one choice) and failed calls; stop reasons
    max_tokens and model_length (context full) counted apart; max output tokens per call; calls at or above the 32,768-token per-turn
    cap; the largest logged max_tokens; and step_limit_reached = completed calls >= 200 (meaningful for the policy model).

  CR_CFG=<folder holding the inventory> uv run --no-project --python 3.12 --with modal==1.6.1 modal run eval_counts.py
"""

import hashlib
import io
import json
import os
import time
import zipfile
from pathlib import Path

import modal

CHAIN = "probe-shift_aswin-surya_sub-q3su311_acts_kill-2026-11-14"
INV = "inventory_v3_2026-10-15.json"
INV_SHA = "19306ffa7c42ab35a62715c1a18822a43c6273492e5ede9b2f62cd9e428ad9a3"
CELL = "t325_model_registry"
WORLDS = ["run1", "run2", "honest_b", "sim", "sim_honest"]
CAP = 32768
STEP_CAP = 200  # BA_MAX_STEPS; enforced by the policy, so read from completed policy calls and store.policy_ended_by
ROLES = {"system", "user", "assistant", "tool"}
LIMIT_TYPES = {"context", "time", "working", "message", "token", "operator", "custom", "cost"}
ENDED = {"policy_timeout", "out_of_context"}

app = modal.App("probe-shift_aswin-surya_sub-repro_cleanroom_kill-2026-11-16")
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("zstandard==0.23.0")
if modal.is_local():  # .eval members are zstd (zip method 93): the project's evalzip shim registers it through zstandard
    image = (image.add_local_file(str(Path(os.environ["CR_CFG"]) / INV), f"/cfg/{INV}")
             .add_local_file(str(Path(__file__).resolve().parents[1] / "2026-10-14_onpolicy_sim" / "evalzip.py"), "/root/evalzip.py"))


def num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


@app.function(image=image, cpu=1, memory=8192, timeout=900, retries=0,
              volumes={"/c": modal.Volume.from_name(CHAIN, version=2).read_only()})
def counts() -> dict:
    import sys

    sys.path.insert(0, "/root")
    import evalzip  # noqa: F401  (registers zip method 93, zstd)

    b = Path("/cfg", INV).read_bytes()
    assert hashlib.sha256(b).hexdigest() == INV_SHA, "inventory sha"
    runs = [r for r in json.loads(b)["runs"] if r["task"] == CELL and r["world"] in WORLDS]
    assert sorted(r["world"] for r in runs) == sorted(WORLDS), [r["world"] for r in runs]
    out = {}
    for r in sorted(runs, key=lambda r: WORLDS.index(r["world"])):
        data = Path("/c/evals_v1", r["launch"], r["eval_rel"]).read_bytes()  # the chain's upload layout (chain.py:586 @ 5cbd3354c)
        assert hashlib.sha256(data).hexdigest() == r["eval_sha256"], f"{r['world']}: .eval sha"
        res = {"launch": r["launch"], "arm": r["arm"], "tokens_before": r.get("tokens_before"), "samples": []}
        with zipfile.ZipFile(io.BytesIO(data)) as z:  # the bytes whose sha was checked
            names = z.namelist()
            assert "header.json" in names, f"{r['world']}: no header.json"
            ev = json.loads(z.read("header.json")).get("eval", {})
            res["status"] = json.loads(z.read("header.json")).get("status")
            res["config_limits"] = {k: v for k, v in (ev.get("config") or {}).items() if num(v) and "limit" in k}
            res["task_args_numeric"] = {k: v for k, v in (ev.get("task_args") or {}).items() if num(v)}
            sample_files = sorted(n for n in names if n.startswith("samples/") and n.endswith(".json"))
            assert sample_files, f"{r['world']}: no samples/*.json"
            for i, n in enumerate(sample_files):
                s = json.loads(z.read(n))
                assert isinstance(s, dict) and "messages" in s, f"{r['world']}: sample file {i} is not one sample"
                roles = {}
                for m in s.get("messages") or []:
                    k = m.get("role") if m.get("role") in ROLES else "other"
                    roles[k] = roles.get(k, 0) + 1
                asst = [m for m in s.get("messages") or [] if m.get("role") == "assistant"]
                lim = s.get("limit")
                ended = (s.get("store") or {}).get("policy_ended_by")
                per_model = {}
                for e in s.get("events") or []:
                    if e.get("event") != "model":
                        continue
                    o = e.get("output") or {}
                    chs = o.get("choices") or []
                    ok = e.get("error") is None and len(chs) >= 1  # a completed call (a step, for the policy model)
                    sr = (chs[0].get("stop_reason") if chs else None) or o.get("stop_reason")
                    ot = (o.get("usage") or {}).get("output_tokens")
                    mt = (e.get("config") or {}).get("max_tokens")
                    d = per_model.setdefault(str(e.get("model")), {"completed": 0, "failed": 0, "stop_max_tokens": 0,
                                                                   "stop_model_length": 0, "max_output_tokens": 0,
                                                                   "calls_at_or_above_cap": 0, "max_config_max_tokens": None})
                    d["completed" if ok else "failed"] += 1
                    d["stop_max_tokens"] += int(sr == "max_tokens")
                    d["stop_model_length"] += int(sr == "model_length")
                    if num(ot):
                        d["max_output_tokens"] = max(d["max_output_tokens"], int(ot))
                        d["calls_at_or_above_cap"] += int(ot >= CAP)
                    if num(mt):
                        d["max_config_max_tokens"] = max(d["max_config_max_tokens"] or 0, int(mt))
                for d in per_model.values():
                    d["step_limit_reached"] = d["completed"] >= STEP_CAP  # meaningful for the policy model only
                res["samples"].append({
                    "index": i, "messages_by_role": roles, "assistant_turns": len(asst),
                    "tool_calls": sum(len(m.get("tool_calls") or []) for m in asst),
                    "inspect_limit": ({"type": lim.get("type") if lim.get("type") in LIMIT_TYPES else "other",
                                       **{k: v for k, v in lim.items() if num(v)}} if isinstance(lim, dict) else None),
                    "policy_ended_by": None if ended is None else (ended if ended in ENDED else "other"),
                    "error_set": s.get("error") is not None,
                    "total_time": s.get("total_time") if num(s.get("total_time")) else None,
                    "working_time": s.get("working_time") if num(s.get("working_time")) else None,
                    "per_model": per_model})
        out[r["world"]] = res
    return out


@app.local_entrypoint()
def main():
    stamp = time.strftime("%Y-%m-%d_%H%M%S")
    res = counts.remote()
    path = Path(__file__).parent / "results" / f"eval_counts_t325_{stamp}.json"
    with path.open("x") as fh:
        fh.write(json.dumps(res, indent=1))
    print("wrote", path)
