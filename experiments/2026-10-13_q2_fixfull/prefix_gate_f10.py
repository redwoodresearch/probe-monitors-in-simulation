"""F10 prefix gate (PLAN_F10.md Amendment 8a), POST HOC: F10 changes token positions after a run's first changed message, so
its rows are vouched for by the rows BEFORE that message, which must equal full + F9's rows for the same run. Modal CPU;
counts only (no text, no activations leave the container).

  - Qwen3-32B: run 2026-10-13_q2_fxavg_qwen3-32b_l7_v1, arm sim-<side>-fx-fullF9F10 vs sim-<side>-fx-fullF9 of the same run,
    every kind with a row (segment ends and both means), all 7 layers.
  - Seed-OSS-36B: run 2026-10-13_f10_seedcall_seed-oss-36b_v1, arm sim-<side>-fx-fullF9F10 vs full + F9's output_end rows:
    the F9 run (2026-10-13_fixfull_f9_seed-oss-36b_v1, arm sim-<side>-fx-fullF9) for the F9 mapping's override keys, else the
    fix-full run (2026-10-13_fixfull_seed-oss-36b_v1, arm sim-<side>-fx-full); all 7 layers.
  - Prefix: an output position whose tool message comes before the run's first changed message (stage record), or a call
    position whose output message is at or before it (the call precedes its own output). Positions are matched by (kind,
    step, action) and must have the same token index (and span, for means).
  - Positive control: rows after the first change (same (kind, step, action) in both runs) are counted with their bit-exact
    share; F10 must change some of them, or the comparison is not reading different data.
  Gate per model: every prefix position matched with the same token, rows compared > 0, min cosine >= 0.999 (bit-exact counts
  reported by GPU pair; the two runs' rows may come from different shards, GPUs or sequence lengths), and post-change rows not
  all bit-exact.

  PYTHONPATH=src uv run --no-project --python 3.12 --with 'modal==1.6.1' modal run --detach experiments/2026-10-13_q2_fixfull/prefix_gate_f10.py
"""
import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).parent
REPO = HERE.parents[1] if modal.is_local() else Path("/root")
APP = "probe-shift_aswin-surya_sub-q2-f10_prefix_kill-2026-11-13"
F10_INPUTS = "2026-10-13_q2_f10_inputs_v1"
QWEN = "2026-10-13_q2_fxavg_qwen3-32b_l7_v1"
SEED_F10 = "2026-10-13_f10_seedcall_seed-oss-36b_v1"
SEED_FULL, SEED_F9 = "2026-10-13_fixfull_seed-oss-36b_v1", "2026-10-13_fixfull_f9_seed-oss-36b_v1"
SEED_MAP = HERE / "results" / "f9_mapping_seed_2026-10-13_154954.json"
LAYERS = (16, 24, 32, 40, 48, 56, 63)
CALL_KINDS = ("pre_call", "call_end", "call_marker", "call_end_mean")

app = modal.App(APP)
la = modal.Volume.from_name("probe-shift_aswin-surya_sub-la-activations", version=2).read_only()
g3 = modal.Volume.from_name("probe-shift_aswin-surya_sub-q2-g3_acts_kill-2026-11-12", version=2).read_only()
image = (modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy")
         .add_local_file(REPO / "experiments/2026-10-07_q2_rendering_swap/rerender.py", "/root/swap/rerender.py")
         .add_local_file(REPO / "experiments/2026-10-07_q2_rendering_swap/str_replace_tells.py", "/root/swap/str_replace_tells.py")
         .add_local_file(REPO / "experiments/2026-10-09_fixed_sim/fixes.py", "/root/fsim/fixes.py"))


def filename(key: str) -> str:
    return key if len(key.encode()) <= 240 else f"{key[:150]}__{hashlib.sha1(key.encode()).hexdigest()[:12]}"


@app.function(image=image, volumes={"/la": la, "/g3": g3}, cpu=8, memory=32768, timeout=1800, retries=0)
def gate(overrides: dict) -> dict:
    try:
        return _gate(overrides)
    except Exception as e:  # noqa: BLE001 — type and location only
        import traceback
        tb = traceback.extract_tb(e.__traceback__)[-1]
        raise RuntimeError(f"gate failed: {type(e).__name__} at {Path(tb.filename).name}:{tb.lineno}") from None


def _gate(overrides: dict) -> dict:
    import sys
    from collections import defaultdict

    import numpy as np
    sys.path[:0] = ["/root/swap", "/root/fsim"]
    import fixes

    rec = json.loads((Path("/la") / F10_INPUTS / "stage_record.json").read_text())
    first = {(r["arm"], r["key"]): r["first_changed_message"] for r in rec["runs"]}
    msg_of = {}
    for p in sorted((Path("/la") / F10_INPUTS / "inputs").glob("shard_*.jsonl")):
        for line in p.read_text().splitlines():
            it = json.loads(line)
            msg_of[(it["arm"], it["key"])] = {(x["step"], x["action"]): x["msg"] for x in fixes.calls_of(it["messages"])}
    if set(msg_of) != set(first):
        raise ValueError("the staged runs differ from the stage record's")

    gpus = {}

    def gpu(root, run, chunk):
        k = (root, run, int(chunk.split("_")[0].removeprefix("shard")))
        if k not in gpus:
            f = Path(root) / run / "stats" / f"shard_{k[2]}.json"
            gpus[k] = json.loads(f.read_text()).get("gpu") if f.exists() else "unknown"
        return gpus[k]

    def compare(model, pairs, kinds):
        """pairs: (arm, key, (root, run, arm) of F10, (root, run, arm) of the base)."""
        out = defaultdict(lambda: {"runs": 0, "rows": 0, "bit-exact rows": 0, "max abs diff": 0.0, "min cosine": 1.0})
        tot = {"runs": 0, "runs with no prefix row": 0, "prefix positions": 0, "prefix positions missing in the base": 0,
               "prefix positions with a different token or span": 0, "post-change rows compared": 0, "post-change rows bit-exact": 0}
        for arm, key, (r1, run1, a1), (r2, run2, a2) in pairs:
            stem = filename(key)
            m1 = json.loads((Path(r1) / run1 / "meta" / a1 / f"{stem}.json").read_text())
            m2 = json.loads((Path(r2) / run2 / "meta" / a2 / f"{stem}.json").read_text())
            fc, mo = first[(arm, key)], msg_of[(arm, key)]
            base = {(p["kind"], p.get("step"), p.get("action")): p for p in m2["positions"] if p.get("row") is not None and p["kind"] in kinds}
            x1 = {l: np.load(Path(r1) / run1 / "by_layer" / f"L{l}" / f"{m1['chunk']}.npy", mmap_mode="r") for l in LAYERS}
            x2 = {l: np.load(Path(r2) / run2 / "by_layer" / f"L{l}" / f"{m2['chunk']}.npy", mmap_mode="r") for l in LAYERS}
            pair = f"{gpu(r1, run1, m1['chunk'])} vs {gpu(r2, run2, m2['chunk'])}"
            tot["runs"] += 1
            n_prefix = 0
            for p in m1["positions"]:
                if p.get("row") is None or p["kind"] not in kinds:
                    continue
                msg = mo.get((p.get("step"), p.get("action")))
                if msg is None:
                    continue
                prefix = msg <= fc if p["kind"] in CALL_KINDS else msg < fc
                q = base.get((p["kind"], p.get("step"), p.get("action")))
                if prefix:
                    tot["prefix positions"] += 1
                    if q is None:
                        tot["prefix positions missing in the base"] += 1
                        continue
                    if (p["token"], p.get("span")) != (q["token"], q.get("span")):
                        tot["prefix positions with a different token or span"] += 1
                        continue
                n_prefix += prefix
                if q is None:
                    continue
                exact, cmin, dmax = True, 1.0, 0.0
                for l in LAYERS:
                    u = np.asarray(x1[l][m1["row_start"] + p["row"]], dtype=np.float32)
                    v = np.asarray(x2[l][m2["row_start"] + q["row"]], dtype=np.float32)
                    cmin = min(cmin, float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-12)))
                    dmax = max(dmax, float(np.abs(u - v).max()))
                    exact &= bool(np.array_equal(u, v))
                if prefix:
                    o = out[pair]
                    o["rows"] += 1; o["bit-exact rows"] += exact
                    o["max abs diff"] = max(o["max abs diff"], dmax); o["min cosine"] = min(o["min cosine"], cmin)
                else:
                    tot["post-change rows compared"] += 1; tot["post-change rows bit-exact"] += exact
            tot["runs with no prefix row"] += n_prefix == 0
            out[pair]["runs"] += 1
        rows = sum(o["rows"] for o in out.values())
        ok = (rows > 0 and tot["prefix positions missing in the base"] == 0 and tot["prefix positions with a different token or span"] == 0
              and all(o["min cosine"] >= 0.999 for o in out.values() if o["rows"])
              and tot["post-change rows compared"] > tot["post-change rows bit-exact"])
        return {"model": model, "totals": tot, "by GPU pair (F10 vs base)": dict(out), "gate": "pass" if ok else "FAIL"}

    sides = (("attack", "sim-attack-fx-fullF9F10"), ("honest", "sim-honest-fx-fullF9F10"))
    keys = {a: sorted(k for (b, k) in first if b == a) for _, a in sides}
    qwen_pairs = [(a, k, ("/la", QWEN, a), ("/la", QWEN, a.replace("F9F10", "F9"))) for _, a in sides for k in keys[a]]
    seed_pairs = [(a, k, ("/g3", SEED_F10, a),
                   ("/g3", SEED_F9, a.replace("F9F10", "F9")) if k in overrides[side] else ("/g3", SEED_FULL, a.replace("fullF9F10", "full")))
                  for side, a in sides for k in keys[a]]
    q_kinds = ("pre_call", "call_end", "call_marker", "output_end", "output_marker", "call_end_mean", "output_end_mean")
    return {"qwen3-32b": compare("Qwen3-32B", qwen_pairs, q_kinds), "seed-oss-36b": compare("Seed-OSS-36B", seed_pairs, ("output_end",)),
            "runs by arm": {a: len(v) for a, v in keys.items()},
            "seed base: F9 override runs": sum(1 for side, a in sides for k in keys[a] if k in overrides[side])}


@app.local_entrypoint()
def main():
    if subprocess.run(["git", "diff", "--quiet", "HEAD", "--", str(Path(__file__))], cwd=HERE).returncode:
        raise SystemExit("commit this script before running")
    smap = json.loads(SEED_MAP.read_text())["arms"]
    overrides = {"attack": smap["full + F9, attack"]["override keys"], "honest": smap["full + F9, honest"]["override keys"]}
    out = gate.remote(overrides)
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip()
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    with open(p := HERE / "results" / f"prefix_gate_f10_{stamp}.json", "x") as fh:
        fh.write(json.dumps({"POST HOC": True, "plan": "PLAN_F10.md Amendment 8a", "git": head,
                             "inputs sha256": {SEED_MAP.name: hashlib.sha256(SEED_MAP.read_bytes()).hexdigest()}} | out, indent=1) + "\n")
    print(p)
    for m in ("qwen3-32b", "seed-oss-36b"):
        print(m, out[m]["gate"], json.dumps(out[m]["totals"]), json.dumps(out[m]["by GPU pair (F10 vs base)"]))
