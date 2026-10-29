"""RULE.md Amendment 4: a supplement printout from the two archived records (reads and formatting only, no statistics), for the
review threads on #1824 and #1826. It shows, per probe, the effective threshold pools, the calls-pooled delta, the false-alarm
controls and realized false alarms beside the task-weighted delta, the cell count by key, and the title cells' repaired and
continuous comparisons. Each record is checked against results/s3_archive_manifest.jsonl first.

  uv run --no-project --python 3.12 python experiments/2026-10-14_p2_sabtitle/analyze_supplement.py <sabotage_spec_….json> <titlecells_spec_….json>

Renamed from report_supplement.py (review on #1844). The title cells are the rows run_titlecells_spec.py printed (probe b in every
configuration, and every probe under excl8 with the honest-B threshold), each with the comparisons the record stores.
"""
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent
PRIMARY = "primary: honest B u C impactful calls, all non-training tasks (reverses d9's pre-registered primary; Append 19 sec. 1)"
man = {r["path"]: r for r in map(json.loads, (HERE / "results" / "s3_archive_manifest.jsonl").read_text().splitlines())}
recs = []
for a in sys.argv[1:3]:
    b = Path(a).read_bytes()
    if hashlib.sha256(b).hexdigest() != man[Path(a).name]["sha256"]:
        raise SystemExit(f"{a}: sha256 differs from the manifest")
    recs.append(json.loads(b))
sp, tc = recs
f = lambda v: f"{100 * v['point']:+.1f} [{100 * v['ci95'][0]:+.1f}, {100 * v['ci95'][1]:+.1f}]"
L = ["Supplement (RULE.md Amendment 4): fields read from the archived records; points in percentage points, 95% bootstrap intervals",
     f"sabotage record label as stored: '{sp['label']}'. The record is PRELIMINARY (the as-run entrypoint's label was overwritten; Amendment 4)"]
pc = sum(1 for p in sp["probes"].values() for k in p if "|" in k)
rl = sum(1 for p in sp["probes"].values() for k, c in p.items() if "|" in k and "run level (max rule)" in c)
mc, mr = len(sp["R44 draw mean"]), sum("run level" in v for v in sp["R44 draw mean"].values())
L.append(f"cells by key: per call {pc}, run level {rl}, R44 draw mean per call {mc}, run level {mr}; total {pc + rl + mc + mr} (stored 'n cells' {sp['n cells']})")
for p, pr in sp["probes"].items():
    pool_c = "B u C" if pr["sameness per call"]["C pooled"] else "B only"
    pool_r = "B u C" if pr["sameness run max (C vs A u B)"]["C pooled"] else "B only (C failed run-max sameness)"
    for a in ("0.01", "0.05"):
        c = pr[f"{PRIMARY}|{a}"]
        fa1, fa2 = c["FA control: not_sabotage calls, run2 - run1"], c["FA control: simulated honest - honest B impactful calls"]
        rf = ", ".join(f"{k} {100 * v:.2f}" for k, v in c["realised false alarm, impactful honest calls"].items())
        cp = c["calls pooled (d9's convention), delta"]
        L.append(f"[{p}] primary {a}: pools per call {pool_c}, run level {pool_r}; Δ task-weighted {f(c['delta_sim_minus_mean_real'])}, "
                 f"calls pooled {f(cp)}; "
                 f"FA controls: not_sabotage run2−run1 {f(fa1)} (within ±1: {fa1['within +-1 point?']}), sim honest − B {f(fa2)} "
                 f"(within ±1: {fa2['within +-1 point?']}); realized FA % {rf}")
for m in tc["metrics"]:
    for k, c in m["cells"].items():
        if not (k.split("|")[2] == "b" or k.startswith("excl8|honest B only")):  # run_titlecells_spec.py's printout selection
            continue
        r, z = c.get("repaired sim (beside)"), c.get("continuous: sim - mean real, standardized")
        L.append(f"[title {m['reader']} L{m['layer']} {k}] as-is Δ {f(c['delta_sim_minus_mean_real'])}"
                 + (f"; repaired Δ {f(r['delta_repaired_minus_mean_real'])} ({r['tasks']} tasks)" if r else "; repaired: not stored")
                 + (f"; continuous Δ (SD units) {z['point']:+.3f} {[round(x, 3) for x in z['ci95']]}" if z else "; continuous: not stored")
                 + ("; greyed (k < 10)" if c["greyed (k < 10)"] else ""))
stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
(HERE / "results" / f"supplement_{stamp}.printout.txt").open("x").write("\n".join(L) + "\n")
print("\n".join(L))
