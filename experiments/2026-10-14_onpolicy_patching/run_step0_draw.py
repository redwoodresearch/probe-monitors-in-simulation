"""Step 0 of the patching pilot (RULE.md "N"): count each candidate run's non-submit tool outputs (= simulator calls) in
Qwen3.5-27B's own Inspect logs, then draw the pilot runs. Modal CPU; numbers, keys and task names only leave the Volume.

Candidates: the epoch map's run1 (first real attack, 193) and honest_a (191) on the q4own_evals Volume, excluding the
environments Step 2 excluded (media_processing, model_registry). Draw: per arm, a permutation seeded by
sha256("patching|<arm>"); then honest, attack, honest, ... in turn; a run is taken while the cumulative calls stay
≤ CAP and its arm has < 10; the draw stops at the first run that would exceed CAP (no skipping, so no bias toward short
runs beyond the stop itself).

  uv run --no-project --python 3.12 --with modal==1.6.1 modal run experiments/2026-10-14_onpolicy_patching/run_step0_draw.py --epoch-map <q4b_epoch_map_v1.json>
"""

import hashlib
import json
import random
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).parent
APP = "probe-shift_aswin-surya_sub-patching_step0_kill-2026-11-14"
EVALS = "probe-shift_aswin-surya_sub-q4own_evals_kill-2026-11-13"
ARMS = {"honest": "honest_a", "attack": "run1"}  # pilot arm -> epoch-map world
EXCLUDED_ENVS = ("media_processing", "model_registry")  # as Step 2 (q2's records_2026-10-13_171253.json)
CAP, PER_ARM, SUBMIT, WINDOW = 450, 10, "submit", 20  # Amendment 1: calls = min(outputs, WINDOW); 0-output runs excluded

app = modal.App(APP)
image = (modal.Image.debian_slim(python_version="3.12").uv_pip_install("zstandard==0.23.0")
         .add_local_file(str(HERE / "evalzip.py"), "/root/evalzip.py"))  # byte copy of aswin/q4-why-weaker's evalzip.py


@app.function(image=image, cpu=(8, 8), memory=8192, timeout=1800, retries=0,
              volumes={"/ev": modal.Volume.from_name(EVALS, version=2).read_only()})
def count(rels: list[str]) -> dict:
    import sys
    import zipfile
    from concurrent.futures import ThreadPoolExecutor

    sys.path.insert(0, "/root")
    import evalzip  # noqa: F401  (zstd .eval members)

    def one(rel):
        with zipfile.ZipFile(Path("/ev/evals_v1") / rel) as z:
            (n,) = [x for x in z.namelist() if x.startswith("samples/") and x.endswith(".json")]
            s = json.loads(z.read(n))
        msgs = s["messages"]
        submit_ids = {tc.get("id") for m in msgs if m["role"] == "assistant" for tc in m.get("tool_calls") or [] if tc["function"] == SUBMIT}
        outs = [m for m in msgs if m["role"] == "tool" and m.get("function") != SUBMIT and m.get("tool_call_id") not in submit_ids]
        mt = (s.get("metadata") or {}).get("main_task")
        return rel, {"tool_outputs": len(outs), "non_string_outputs": sum(not isinstance(m.get("content"), str) for m in outs),
                     "main_task": mt.get("id") if isinstance(mt, dict) else mt}

    with ThreadPoolExecutor(16) as ex:
        return dict(ex.map(one, rels))


REFERENCES = HERE.parent / "2026-10-06_astra_sim_honest/results/references_20261006-142754.json"  # Amendment 2


@app.local_entrypoint()
def main(epoch_map: str):
    git = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True, cwd=HERE, check=True).stdout.strip()
    if git("status", "--porcelain", "--untracked-files=no"):
        raise SystemExit("refusing to run from a dirty tree")
    raw = Path(epoch_map).read_bytes()
    em = json.loads(raw)
    cand = {}
    for arm, world in ARMS.items():
        keys = sorted(k for k, (w, task) in em["map"].items() if w == world and not any(e in task for e in EXCLUDED_ENVS))
        cand[arm] = [{"key": k, "rel": k.split("#")[0], "task": em["map"][k][1]} for k in keys]
    refs_raw = REFERENCES.read_bytes()
    covered = set(json.loads(refs_raw)["references"])
    counts = count.remote([c["rel"] for a in cand.values() for c in a])
    order = {}
    for arm, cs in cand.items():
        rng = random.Random(int(hashlib.sha256(f"patching|{arm}".encode()).hexdigest()[:16], 16))
        idx = list(range(len(cs)))
        rng.shuffle(idx)
        order[arm] = [c | {"calls": min(c["tool_outputs"], WINDOW)} for c in (cs[i] | counts[cs[i]["rel"]] for i in idx)
                      if c["tool_outputs"] > 0 and c["main_task"] in covered]  # Amendment 2: a September overlap800 reference
    picked, total, n = [], 0, {a: 0 for a in ARMS}
    for i in range(max(len(v) for v in order.values())):
        stop = False
        for arm in ("honest", "attack"):
            if n[arm] >= PER_ARM or i >= len(order[arm]):
                continue
            c = order[arm][i]
            if total + c["calls"] > CAP:
                stop = True
                break
            total += c["calls"]
            n[arm] += 1
            picked.append({"arm": arm, "draw_index": i, "cumulative_calls": total} | c)
        if stop or all(n[a] >= PER_ARM for a in ARMS):
            break
    res = {"git": git("rev-parse", "HEAD"), "epoch_map_sha256": hashlib.sha256(raw).hexdigest(),
           "references_sha256": hashlib.sha256(refs_raw).hexdigest(), "eligible": {a: len(v) for a, v in order.items()}, "cap": CAP, "per_arm": PER_ARM, "window": WINDOW,
           "excluded_envs": EXCLUDED_ENVS, "candidates": {a: len(v) for a, v in cand.items()},
           "calls_per_run": {a: {"mean": sum(c["tool_outputs"] for c in v) / len(v), "min": min(c["tool_outputs"] for c in v),
                                 "max": max(c["tool_outputs"] for c in v)} for a, v in order.items()},
           "non_string_outputs": sum(c["non_string_outputs"] for v in order.values() for c in v),
           "picked": picked, "picked_per_arm": n, "picked_calls": total,
           "picked_cost_usd_at_0.243": round(total * 0.243, 2)}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    p = HERE / "results" / f"step0_draw_{stamp}.json"
    with p.open("x") as fh:
        fh.write(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in ("candidates", "calls_per_run", "picked_per_arm", "picked_calls", "picked_cost_usd_at_0.243")}), flush=True)
    print(f"wrote {p.name}", flush=True)
