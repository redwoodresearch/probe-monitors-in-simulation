"""Quote (no launch) for one interleaved run3 + sim2 launch of il_old's selection-half cells (d9 11:3x; 09: 97 keys).
Inputs, all committed: config/il_keys_selection.json, config/tasks_il_old_all.json, results/il_no_reference_cells_2026-10-15.json,
results/sim_spend_reads_2026-10-15_0849/*.jsonl (launch.json spend per IL launch), the rollout log (oldc start/end).
Rates: sims at the old family's completion rate (oldc $/sim run, done); run3 on one 6 x H200 set at $5.30/server-hour;
extraction an estimate (stated). Cap rule (completions, K = 10): cap = max(cells x $20.5, cells x price + 10 x $75).
Duration scales oldc's wall time by waves = cells / initial admitted cells (cap / $75).
  python quote_sel97.py --out <json>
"""
import argparse, glob, json, math
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).parent
H200, SERVERS, RATE, RESERVE, K, BUF = 5.30, 6, 20.5, 75.0, 10, 1.3


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--earliest-start", required=True)
    ap.add_argument("--start-basis", required=True); ap.add_argument("--census", required=True); a = ap.parse_args()
    sel = json.load(open(HERE / "config/il_keys_selection.json"))
    t = {c["key"]: c for c in json.load(open(HERE / "config/tasks_il_old_all.json"))["cells"]}
    noref = json.load(open(HERE / "results/il_no_reference_cells_2026-10-15.json"))
    nr = {r["key"] for lj in noref["launches"] for r in lj["rows"]}
    assert all(k in t for k in sel), "a selection key is not an il_old (311 harness) cell"
    reads = {}
    for f in glob.glob(str(HERE / "results/sim_spend_reads_2026-10-15_0849/*.jsonl")):
        for line in open(f):
            r = json.loads(line)
            if "error" not in r:
                reads[Path(r["run_dir"]).name] = r
    oc = reads["2026-10-15_il_oldc"]
    price = oc["spend_usd"] / oc["n_sim_done"]
    log = [json.loads(l) for l in open(HERE / "results/il_rollout_decisions_2026-10-15.jsonl")]
    st = next(r for r in log if r["action"] == "start" and "il_oldc" in str(r["launch"]))["time_pt"]
    en = next(r for r in log if r["action"] == "ended" and "il_oldc" in str(r["launch"]))["time_pt"]
    oc_h = (datetime.fromisoformat(en) - datetime.fromisoformat(st)).total_seconds() / 3600
    oc_cells, oc_cap = 62, 1656
    n = len(sel)
    cap = math.ceil(max(n * RATE, n * price + K * RESERVE))
    admit0 = int(cap // RESERVE)
    waves_oc, waves = oc_cells / (oc_cap // RESERVE), n / admit0
    hours = oc_h * waves / waves_oc
    sims_e, sims_h = n * price, cap
    gpu_h_e = SERVERS * (hours + 0.5)            # + ~0.5 h startup and stop by procedure
    gpu_e, gpu_h = gpu_h_e * H200, SERVERS * 10 * H200  # hard: the set's 10 h Modal timeout
    ext_e, ext_h = 0.30 * 2 * n, 1.00 * 2 * n    # estimate per run (expected / hard); the chain books extraction from its il pool
    exp = sims_e + gpu_e + ext_e
    out = {"what": "QUOTE ONLY (no launch): one interleaved run3 + sim2 launch of il_old's selection-half cells", "cells": n,
           "cells_by_key": sel, "origin": "2026-10-14_il_old (all il_old cells; 311 harness; references_from ~/phase2/il_refindex_old)",
           "no_pinned_reference": sorted(set(sel) & nr),
           "rates": {"sim_usd_per_run": round(price, 2), "sim_rate_source": f"2026-10-15_il_oldc: ${oc['spend_usd']} / {oc['n_sim_done']} done sims (results/sim_spend_reads_2026-10-15_0849)",
                     "h200_usd_per_server_hour": H200, "servers": SERVERS, "extraction_usd_per_run": "0.30 expected / 1.00 hard (estimate)"},
           "cap": {"rule": "max(cells x $20.5, cells x price + 10 x $75)", "cap_usd": cap, "admitted_at_start": admit0},
           "duration_h": round(hours, 2), "duration_basis": f"oldc {oc_cells} cells, cap ${oc_cap}, {oc_h:.2f} h; scaled by waves (cells / initial admitted)",
           "cost_usd": {"sims": {"expected": round(sims_e), "hard": sims_h}, "gpu_run3": {"expected": round(gpu_e), "hard": round(gpu_h)},
                        "extraction": {"expected": round(ext_e), "hard": round(ext_h)},
                        "total": {"expected": round(exp), "expected_plus_30pct": round(exp * BUF), "hard": round(sims_h + gpu_h + ext_h)},
                        "modal_share": {"expected": round(gpu_e + ext_e), "hard": round(gpu_h + ext_h)}}}
    from datetime import timedelta
    start = datetime.fromisoformat(a.earliest_start)
    out["schedule"] = {"box": "xbox1 i-04331f272d6778df0 (311 harness; il_old's own box; references il_refindex_old 79acbfc1... and all approach files present: precheck PASS)",
                       "earliest_start_pt": a.earliest_start, "basis": a.start_basis,
                       "end_pt_expected": (start + timedelta(hours=hours)).isoformat(timespec="minutes"),
                       "end_pt_plus_30pct": (start + timedelta(hours=hours * BUF)).isoformat(timespec="minutes"),
                       "xbox1_poweroff_pt": "2026-10-15T18:30-07:00 (Aswin, 11:15 relay)",
                       "fits_poweroff": start + timedelta(hours=hours * BUF + 0.5) <= datetime.fromisoformat("2026-10-15T18:30-07:00"),
                       "latest_start_for_poweroff_pt": (datetime.fromisoformat("2026-10-15T18:30-07:00") - timedelta(hours=hours * BUF + 0.5)).isoformat(timespec="minutes"),
                       "servers": "one 6 x H200 set (servers_il.yaml), started ~25 min before the start"}
    out["census"] = {"latest": a.census, "launch_adds_at_start": admit0, "rule": "paired IL never pauses: start only when the census projects <= ~110 simulated running for the whole run with 0 RateLimitError in the latest bins, counting nm360 sim-honest's planned growth; with 27 added at start that needs <= 83 running elsewhere"}
    Path(a.out).write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: out[k] for k in ("cells", "no_pinned_reference", "rates", "cap", "duration_h", "cost_usd")}))


if __name__ == "__main__":
    main()
