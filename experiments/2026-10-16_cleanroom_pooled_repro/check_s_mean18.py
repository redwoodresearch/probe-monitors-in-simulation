"""Addendum 18d confirming check (09's ruling, 2026-10-20): read-only, booleans and counts only, no recompute.

(1) For each of the 144 failing S units_per_run_mean rows of the A18 compute's comparison file (output Volume), does the record's
    mean equal mine without the structural run's 0? ref x (runs + zero) == mine x (runs + zero + 1) within b = 1e-9·max(1, |rhs|),
    where runs = the row's S runs (mine; its verdict row is bucket "0"), zero + 1 = mine of the S #sum row (zero-unit catalogue runs +
    the 1 structural run; STRUCT_OK: 1 per label on S).
(2) For every C label x pooled origin row at level 0.01 in A's record: round_to_nearest_5(n_units / runs) ==
    round_to_nearest_5(n_units / (runs + 1)) on pools_at_draw0["0.01"].S (would nm014's 0, RULE l.3642, move the printed range?).
    A superset of rc1a 16.units_per_run.{trained, public}.S's member rows, reported per label.

  CR_CFG=<pinned input folder> uv run --no-project --python 3.12 --with modal==1.6.1 modal run check_s_mean18.py
"""

import gzip
import json
import os
import time
from pathlib import Path

import modal

OUT = "probe-shift_aswin-surya_sub-repro_cleanroom_kill-2026-11-16"
CMP = "cleanroom_comparison_A18_2026-10-20_040452.json"
CMP_SHA = "584c9d8c9564df728aaa692bc03c696446100bd1e68ba17406d64f24571a61d7"
REC = "routec_A_alltasks_CRD_FULL_r2_allorigins_pooledworld_scored_e12_2026-10-19_180327.json.gz"
COMBO = "pooled_all_origins"

app = modal.App(OUT)
OUT_VOL = modal.Volume.from_name(OUT, version=2)
image = modal.Image.debian_slim(python_version="3.12")
if modal.is_local():
    image = image.add_local_file(str(Path(os.environ["CR_CFG"]) / REC), f"/cfg/{REC}")


@app.function(image=image, cpu=1, memory=4096, timeout=600, retries=0, volumes={"/o": OUT_VOL})
def check() -> dict:
    import hashlib

    raw = Path("/o", CMP).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == CMP_SHA, "comparison file sha"
    cmp = json.loads(raw)["comparison"]
    by = {tuple(r["path"]): r for r in cmp}
    mean = [r for r in cmp if r["verdict"] and r["family"] == "pool_units_per_run_mean" and r["path"][5] == "S" and not r["match_flip_rule"]]
    assert len(mean) == 144, len(mean)
    ok1 = []
    for r in mean:
        b0 = tuple(r["path"][:6])
        runs = by[b0 + ("runs",)]["mine"]
        zs = by[b0 + ("no_unit_runs_counted_non_alarming", "#sum")]["mine"]
        assert by[b0 + ("runs",)]["delta_bucket"] == "0" and isinstance(runs, int) and isinstance(zs, int) and zs >= 1
        lhs, rhs = r["ref"] * (runs + zs - 1), r["mine"] * (runs + zs)
        ok1.append(abs(lhs - rhs) <= 1e-9 * max(1.0, abs(rhs)))
    rec = json.loads(gzip.decompress(Path("/cfg", REC).read_bytes()))

    def r5(x):
        return 5 * round(x / 5)

    per_label = {}
    for lab, rows in rec["rows"].items():
        if not lab.endswith(" C"):
            continue
        res = []
        for o, r_ in rows.items():
            if f"[pooled: {COMBO}]" not in o:
                continue
            s = r_["pools_at_draw0"]["0.01"]["S"]
            n, runs = s["n_units"], s["runs"]
            res.append(r5(n / runs) == r5(n / (runs + 1)))
        per_label[lab] = {"origin_rows": len(res), "all_equal": all(res), "n_false": sum(not x for x in res)}
    return {"check1": {"rows": len(ok1), "n_true": sum(ok1), "n_false": len(ok1) - sum(ok1)},
            "check2": {"labels": len(per_label), "all_equal": all(v["all_equal"] for v in per_label.values()), "per_label": per_label}}


@app.local_entrypoint()
def main():
    out = check.remote()
    path = Path(__file__).parent / "results" / f"check_s_mean18_{time.strftime('%Y-%m-%d_%H%M%S')}.json"
    path.write_text(json.dumps(out, indent=1, sort_keys=True))
    print(json.dumps({"check1": out["check1"], "check2_all_equal": out["check2"]["all_equal"]}))
    print("wrote", path)
