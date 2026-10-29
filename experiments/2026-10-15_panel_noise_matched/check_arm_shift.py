"""Why rows move between two pooled-record versions: per origin, arm and point, the draw-0 catch and the draw-0 pool thresholds/sizes, old vs new
(git show, sha256-checked; small JSON reads only). Writes results/armshift_<stamp>.json.

  uv run --no-project python experiments/2026-10-15_panel_noise_matched/check_arm_shift.py
"""

import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent
D = "experiments/2026-10-15_fpr_sweep_507_prep/results"
PAIRS = {"C": ((f"2892fa260:{D}/pooled455_C_2026-10-15_151852.json", "edfef938d3520cb736ae7bed970e82b6e1371dfbf37f5621afdf2156fa317146"),
               (f"672b1dd1b:{D}/pooled455v2_C_2026-10-15_160734.json", "63871423fd878f62c3fc1275b1ff1f87e3e7135d16cbb391475c24666d102ad0")),
         "R": ((f"dafee6914:{D}/pooled455_R_2026-10-15_155043.json", "97a6831d4d4a803509574b5f889df324f4c73a8a18d6c0d3fd20bdc9c4bd8ecf"),
               (f"672b1dd1b:{D}/pooled455v2_R_2026-10-15_160736.json", "3f36aab2c8820fd2812f9e8031dcd2c1627aeaf897e93b27aa16c6f5d8fdcc18"))}
ARMS = ("sim@S|primary", "run1@B", "run2@B", "run2@A", "run1@B2")


def load(spec, sha):
    b = subprocess.run(["git", "show", spec], capture_output=True, check=True, cwd=HERE).stdout
    assert hashlib.sha256(b).hexdigest() == sha, spec
    return json.loads(b)


out, summ = {"rows": {}}, {}
for o_, n_ in PAIRS.values():
    A, B = load(*o_), load(*n_)
    for lb in A["rows"]:
        for org in ("S_old", "S_new"):
            for f in ("0.01", "0.05"):
                ca, cb = A["rows"][lb][org]["points"]["joint"][f]["catch"], B["rows"][lb][org]["points"]["joint"][f]["catch"]
                for arm in ARMS:
                    if arm not in ca:
                        continue
                    d = 100 * (cb[arm]["point"] - ca[arm]["point"])
                    out["rows"].setdefault(lb, {}).setdefault(org, {}).setdefault(f, {})[arm] = round(d, 4)
                    k = (org, f, arm)
                    summ.setdefault(k, {"rows": 0, "changed": 0, "max_abs_points": 0.0})
                    summ[k]["rows"] += 1
                    summ[k]["changed"] += abs(d) > 1e-9
                    summ[k]["max_abs_points"] = max(summ[k]["max_abs_points"], abs(d))
                pa, pb = A["rows"][lb][org].get("pools_at_draw0", {}).get(f, {}), B["rows"][lb][org].get("pools_at_draw0", {}).get(f, {})
                out["rows"][lb][org][f]["pools_draw0_changed"] = sorted(p for p in set(pa) | set(pb) if json.dumps(pa.get(p), sort_keys=True) != json.dumps(pb.get(p), sort_keys=True))
out["summary"] = [{"origin": k[0], "f": k[1], "arm": k[2], **v} for k, v in sorted(summ.items())]
out["n_per_batch_S_new"] = {ro: {"old": load(*o_)["rows"]["Sol (a) " + ro]["S_new"].get("n_per_batch"), "new": load(*n_)["rows"]["Sol (a) " + ro]["S_new"].get("n_per_batch")}
                            for ro, (o_, n_) in PAIRS.items()}
stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
dest = HERE / "results" / f"armshift_{stamp}.json"
dest.open("xb").write(json.dumps({"label": "POST HOC; from raw data, by script; draw-0 catch differences new - old in points", "pairs": PAIRS, **out}, indent=1).encode())
for r in out["summary"]:
    print(f"{r['origin']:6s} {r['f']:5s} {r['arm']:14s} changed {r['changed']:2d}/{r['rows']:2d}  max |d| {r['max_abs_points']:.2f}")
print(json.dumps(out["n_per_batch_S_new"])[:600])
print(dest.name)
