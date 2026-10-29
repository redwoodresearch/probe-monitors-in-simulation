"""Readability of the strict-alarm (low FPR) points on ALL tasks, counts only (2e 10-15 ~19:30). Reads the FINAL sweep records' coded readable
flags (run_peraction311.py @ 9e18a3971 line 770: a pool is readable at level f iff k >= 1 and >= 10 units above T at draw 0; ties at T can make
units_above < k, listed under k_differs_from_units_above) and their pool unit counts. No threshold, catch or gap value is read out or written.
- per origin: readable iff every pool it uses (B, A, S; S_old also B2) is readable (line 822/891);
- all-tasks point with per-origin lines: readable iff S_old, S_new and S_newmains are all readable;
- S_old sim-honest shortfall: units needed so floor(f * n) >= 10, minus today's n; runs = shortfall / S_old's mean units per run (PROJECTION);
- projections +28 runs (pilot 20 + fill-in 8, routed now) and +42 (adds g4 14), by S_old's mean units per sim-honest run (PROJECTION);
- runs needed: more S_old sim-honest runs for S_old alone to be readable, and for every origin; "not reachable" where a real pool is short;
- extraction cost per run from the simh_t196fill SPEND row (an estimate; generation cost is not in the ledgers and is not included);
- shared-pool deviation (UNREGISTERED, never a substitute): B, A and S each pooled over the origins present (S_newmains only where not withheld),
  readable iff floor(f * N) >= 10 for each pooled role, in two versions: (i) S_old's B-alone yardstick pool (B2) KEPT as its own constraint,
  (ii) B2 REPLACED by the pooled B (a further deviation). The records' own shared-pool label covers S_new u S_newmains only; this adds S_old.
  python3 experiments/2026-10-15_readability_alltasks/count_readability.py
"""
import hashlib
import json
import math
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent
C = "d62cfe39e"
REC = {"C": ("experiments/2026-10-15_fpr_sweep_507_prep/results/sweep_final_C_2026-10-15_160740.json", "e828f365f95078c4275c5a56fd146fe9efbd4e3061b7c2fdb7e30fd8ad7e4f64"),
       "R": ("experiments/2026-10-15_fpr_sweep_507_prep/results/sweep_final_R_2026-10-15_160743.json", "66d4a2a77d7f6d45d9fe6ce958c84f2cd434039e3523be3c870f8705ba772011")}
ORIG = ("S_old", "S_new", "S_newmains")
EXTRACT = (0.20, "SPEND.md row 439 @ 7bdac8d1b (origin/aswin/q3-overnight-chain): simh_t196fill, 8 epochs at ~$0.20 per epoch (estimate before launch; actual pending)")
ADD = {"+28 (pilot 20 + fill-in 8; routed now)": 28, "+42 (adds g4 14)": 42}


def git(*a):
    return subprocess.run(["git", *a], capture_output=True, check=True, cwd=HERE).stdout


def ok(f, n):
    return math.floor(f * n + 1e-9) >= 10


def need(f):
    n = math.ceil(10 / f - 1e-6)
    while not ok(f, n):
        n += 1
    return n


def main():
    out = {"label": "PRELIMINARY: readability of all-tasks strict-alarm points; counts only (flags and unit counts); projections labelled",
           "inputs": {}, "rows": {}}
    md = ["| readout | probe | level | readable origins (coded) | all-tasks (all 3) | S_old S units / needed | S_old binding pools | +28 S_old | +42 S_old | "
          "more S_old runs: S_old readable (PROJECTION) | more S_old runs: every origin readable (PROJECTION) | extraction cost of those runs (~$0.20/run; "
          "generation not included) | shared-pool deviation (UNREGISTERED; B, A, S pooled; S_old B-alone kept) | same, S_old B-alone replaced by pooled B (further deviation) |",
          "|" + "---|" * 14]
    for ro, (p, want) in REC.items():
        b = git("show", f"{C}:{p}")
        assert hashlib.sha256(b).hexdigest() == want, p
        out["inputs"][ro] = f"{p} @ {C} (sha256 {want})"
        d = json.loads(b)
        for lab, row in d["rows"].items():
            o_ = {o: row[o] for o in ORIG}
            wh = [o for o, x in o_.items() if "withheld" in x]
            levels = list(o_["S_old"]["readable"])
            res = []
            for lv in levels:
                f = float(lv)
                flags = {o: (None if o in wh else x["readable"][lv]) for o, x in o_.items()}
                for o, x in o_.items():
                    if o in wh:
                        continue
                    for nm, pz in x["pools_at_draw0"][lv].items():
                        assert flags[o][nm] == (pz["k"] >= 1 and pz["units_above"] >= 10), (lab, o, lv, nm)
                        assert flags[o][nm] == ok(f, pz["n_units"]), ("the count rule disagrees with a coded flag", lab, o, lv, nm)
                        if pz["units_above"] != pz["k"]:
                            out.setdefault("k_differs_from_units_above (ties at T; flags unchanged)", []).append(
                                f"{ro} | {lab} | {o} | {lv} | {nm}: k {pz['k']}, units_above {pz['units_above']}")
                rd = {o: (None if fl is None else all(fl.values())) for o, fl in flags.items()}
                so = o_["S_old"]["pools_at_draw0"][lv]
                nS, mS = so["S"]["n_units"], so["S"]["units_per_run_mean"]
                other_old = [nm for nm, v in flags["S_old"].items() if nm != "S" and not v]
                proj = {k_: ok(f, nS + a * mS) and not other_old for k_, a in ADD.items()}
                short = max(0, need(f) - nS)
                others_block = [f"{o}:{nm}" for o in ("S_new", "S_newmains") if rd[o] is not None for nm, v in flags[o].items() if not v]
                runs_old = "not reachable: S_old real pools short (" + ", ".join(other_old) + ")" if other_old else math.ceil(short / mS)
                if rd["S_newmains"] is None:
                    runs_all = "n/a (S_newmains withheld)"
                elif other_old or others_block:
                    runs_all = "not reachable by S_old sim-honest runs: " + ", ".join([f"S_old:{x}" for x in other_old] + others_block)
                else:
                    runs_all = math.ceil(short / mS)
                pooled = {}
                for nm in ("B", "A", "S"):
                    pooled[nm] = sum(o_[o]["pools_at_draw0"][lv][nm]["n_units"] for o in ORIG if o not in wh)
                b2 = o_["S_old"]["pools_at_draw0"][lv]["B2"]["n_units"]
                shared_rep = all(ok(f, n) for n in pooled.values())
                shared = shared_rep and ok(f, b2)
                allt = None if wh else all(rd.values())
                res.append({"level": f, "readable_by_origin": rd, "pool_flags": flags, "all_tasks_readable": allt if not wh else f"n/a ({', '.join(wh)} withheld)",
                            "S_old_sim_units": nS, "S_old_sim_runs": so["S"]["runs"], "S_old_mean_units_per_run": mS, "units_needed_floor_f_n_ge_10": need(f),
                            "S_old_sim_shortfall_units": short, "S_old_non_sim_pools_unreadable": other_old,
                            "PROJECTION_S_old_readable": proj, "PROJECTION_more_S_old_runs_for_S_old": runs_old, "PROJECTION_more_S_old_runs_for_every_origin": runs_all,
                            "extraction_cost_usd_of_runs_for_every_origin": round(runs_all * EXTRACT[0], 2) if isinstance(runs_all, int) else None,
                            "shared_pool_deviation_UNREGISTERED": {"pooled_units": pooled, "S_old_B2_units": b2, "readable_B2_kept": shared,
                                                                   "readable_B2_replaced_by_pooled_B": shared_rep}})
                md.append(f"| {ro} | {lab} | {f:.3g} | {', '.join(o for o, v in rd.items() if v) or 'none'}{' (' + ', '.join(wh) + ' withheld)' if wh else ''} | "
                          f"{'n/a' if wh else 'yes' if allt else 'no'} | {nS} / {need(f)} | {', '.join(other_old) or '-'} | {'yes' if proj[list(ADD)[0]] else 'no'} | "
                          f"{'yes' if proj[list(ADD)[1]] else 'no'} | {runs_old} | {runs_all} | "
                          f"{'$%.2f' % (runs_all * EXTRACT[0]) if isinstance(runs_all, int) else '-'} | {'yes' if shared else 'no'} | {'yes' if shared_rep else 'no'} |")
            out["rows"][f"{lab}"] = {"withheld": wh, "levels": res}
    out["cost"] = {"extraction_per_run_usd": EXTRACT[0], "source": EXTRACT[1], "generation": "not in the SPEND ledgers searched (q3-overnight-chain, fpr-sweep-507-prep, main, "
                   "convergence-audit-09); ask d9 for the sim-honest generation rate"}
    out["meta"] = {"git": git("rev-parse", "HEAD").decode().strip(), "stamp_pt": datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")}
    dest = HERE / "results" / f"readability_alltasks_{out['meta']['stamp_pt']}"
    dest.parent.mkdir(exist_ok=True)
    Path(f"{dest}.json").open("x").write(json.dumps(out, indent=1) + "\n")
    Path(f"{dest}.md").open("x").write("PRELIMINARY (counts only; projections labelled; shared-pool column is an unregistered deviation)\n\n" + "\n".join(md) + "\n")
    print(dest)


if __name__ == "__main__":
    main()
