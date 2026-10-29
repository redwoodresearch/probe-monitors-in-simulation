"""Cell (a) transfer, Amendment 4's checks (POST HOC; one Modal CPU container; counts and differences only).

1. Row gate (item 4): per variant, every L63 output_end row built from the replay dumps as run_xfer_layers.py builds them
   (counted epochs; output_end calls with a row) is matched to the cache row (on_rows.L63.json / on_L63.npy) with the same
   (world, task, turn, call_index); returned: matched, dump-only, cache-only, duplicate keys, max |difference|.
2. Truncation counts (Amendments 2 and 3): per control (unswapped level 0, unswapped level 1, swapped level 1) and run,
   the reader-kept output_end count (eval-file identity against the replay's counted metas, as run_replay_matched.py)
   and the replay cache's output count.

  PYTHONPATH=src uv run --no-project --python 3.12 --with "modal==1.6.1" --with numpy modal run \\
      experiments/2026-10-13_cell_a_transfer_POSTHOC/run_xfer_checks.py
"""

import json
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1] if len(HERE.parents) > 1 else HERE
APP = "probe-shift_aswin-surya_sub-onpol-reply_xferchecks_kill-2026-11-13"
Q35, ACTS, SCORE = ("probe-shift_aswin-surya_sub-qwen35_acts_kill-2026-11-06", "probe-shift_aswin-surya_sub-q3onpol_acts_kill-2026-11-08",
                    "probe-shift_aswin-surya_sub-q3onpol_score_kill-2026-11-08")
T196 = "experiments/2026-10-07_phase2_attack_check/config/pilot_tasks_amendment9_all196.json"
D0 = ["2026-10-09_rerun_strict_segends_v1", "2026-10-10_scaleup_canary_segends_v1", "2026-10-10_scaleup_l2_segends_v1", "2026-10-10_scaleup_topup_segends_v1"]
R2 = "2026-10-10_second_attack_segends_v1"
SRC = {"unswapped_L0": {"run1": [(d, "attack") for d in D0], "run2": [(R2, "attack")], "sim": [(d, "sim_attack@september") for d in D0],
                        "honest_a": [(d, "honest_a") for d in D0], "honest_b": [(d, "honest_b") for d in D0]},
       "unswapped_L1": {"run1": [(d + "_scrub1", "attack") for d in D0], "run2": [(R2 + "_scrub1", "attack")], "sim": [(d + "_scrub1", "sim_attack@september") for d in D0],
                        "honest_a": [(d + "_scrub1", "honest_a") for d in D0], "honest_b": [(d + "_scrub1", "honest_b") for d in D0]},
       "swapped_L1": {"run1": [(d + "_scrub1_swapsys", "attack") for d in D0], "run2": [(R2 + "_scrub1_swapsys", "attack")],
                      "sim": [(d + "_scrub1_swapsys2", "sim_attack@september") for d in D0],
                      "honest_a": [(d + "_scrub1", "honest_a") for d in D0], "honest_b": [(d + "_scrub1", "honest_b") for d in D0]}}
CACHE = {"unswapped_L0": "retrain_cache_v1/level0", "unswapped_L1": "retrain_cache_v1/level1", "swapped_L1": "claimc_cache_v1"}
CONTROL_LEVEL = {"unswapped_L0": "0", "unswapped_L1": "1", "swapped_L1": "1"}
OLD = "2026-10-10_la_september-onpolicy-l{L}_qwen35-27b_alllayers_v1"
NEW = "2026-10-13_la_september-onpolicy-honest-l{L}_qwen35-27b_alllayers_v1"
READER = {"run1": ("old", "real-attack-r1"), "run2": ("old", "real-attack-r2"), "sim": ("old", "sim-attack-sept-fixed"),
          "honest_a": ("new", "real-honest-a"), "honest_b": ("new", "real-honest-b")}
app = modal.App(APP)
q35, acts, score_vol = (modal.Volume.from_name(n, version=2).read_only() for n in (Q35, ACTS, SCORE))
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy").add_local_file(REPO / T196, f"/root/{T196}")


@app.function(image=image, cpu=8, memory=65536, timeout=2400, retries=0, volumes={"/q35": q35, "/acts": acts, "/score": score_vol})
def checks() -> dict:
    import numpy as np

    t196 = json.loads((Path("/root") / T196).read_text())
    key_of = {(c["env"], c["main"]): c["key"] for c in t196["cells"]}
    out = {"row_gate": {}, "truncation": {}}
    for v, src in SRC.items():
        counted, dump = {}, {}
        for world, sources in src.items():
            for d, arm in sources:
                for fm in sorted((Path("/acts") / d / "meta" / arm).glob("*.json")):
                    m = json.loads(fm.read_text())
                    if m.get("redrawn_because") or m.get("attempt_kind") == "rerun":
                        continue
                    counted[(world, m["task"])] = Path(m["run"]).name
                    for c in m["calls"]:
                        if c.get("kind") == "output_end" and c.get("row") is not None:
                            dump.setdefault((world, m["task"], int(c["turn"]), int(c.get("call_index", 0))), []).append((d, m["chunk"], m["row_start"] + c["row"]))
        cd = Path("/score") / CACHE[v]
        rows = json.loads((cd / "on_rows.L63.json").read_text())
        A = np.load(cd / "on_L63.npy", mmap_mode="r")
        cache = {}
        for i, r in enumerate(rows):
            if r["kind"] == "output_end" and r["world"] in src:
                cache.setdefault((r["world"], r["task"], int(r["order"][0]), int(r["order"][1])), []).append(i)
        both = sorted(set(dump) & set(cache))
        mx, by = 0.0, {}
        for k in both:
            d, ch, rr = dump[k][0]
            by.setdefault((d, ch), []).append((rr, cache[k][0]))
        for (d, ch), pairs in by.items():
            B = np.load(Path("/acts") / d / "by_layer" / "L63" / f"{ch}.npy", mmap_mode="r")
            x = np.asarray(B[[p for p, _ in pairs]], np.float64)
            y = np.asarray(A[[q for _, q in pairs]], np.float64)
            mx = max(mx, float(np.abs(x - y).max()))
        out["row_gate"][v] = {"matched": len(both), "dump_only": len(set(dump) - set(cache)), "cache_only": len(set(cache) - set(dump)),
                              "dump_duplicate_keys": sum(len(x) > 1 for x in dump.values()), "cache_duplicate_keys": sum(len(x) > 1 for x in cache.values()),
                              "max_abs_diff": mx}
        L = CONTROL_LEVEL[v]
        kept = {}
        for world, (which, arm) in READER.items():
            root = Path("/q35") / (OLD if which == "old" else NEW).format(L=L)
            for f in sorted((root / "meta" / arm).glob("*.json")):
                m = json.loads(f.read_text())
                t = key_of.get((m["environment"], m["main_task"]))
                if t is None or counted.get((world, t)) != Path(m["file"]).name:
                    continue
                kept[f"{world}|{t}"] = sum(p["kind"] == "output_end" and p["kept"] and p["row"] is not None for p in m["positions"])
        n_rep = {}
        for (w, t, _, _), ix in cache.items():
            n_rep[f"{w}|{t}"] = n_rep.get(f"{w}|{t}", 0) + len(ix)
        out["truncation"][v] = {k: [kept[k], n_rep.get(k, 0)] for k in sorted(kept)}
    return out


@app.local_entrypoint()
def main():
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip()
    if subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True, cwd=HERE).stdout:
        raise SystemExit("uncommitted tracked changes")
    res = checks.remote()
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    dest = HERE / "results" / f"xfer_checks_POSTHOC_{stamp}.json"
    dest.open("x").write(json.dumps({"label": "POST HOC (RULE.md Amendment 4): row gate and truncation counts", "git": git} | res))
    print(json.dumps(res["row_gate"]))
    for v, t in res["truncation"].items():
        print(v, "runs", len(t), "reader_more", sum(a > b for a, b in t.values()), "reader_zero", sum(a == 0 for a, _ in t.values()))
    print(dest.name)
