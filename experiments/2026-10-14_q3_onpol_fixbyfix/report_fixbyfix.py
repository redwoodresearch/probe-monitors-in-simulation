"""Print the fix-by-fix table from a result JSON (score_fixbyfix.py's output): per layer, rate, calibration and probe, each
variant's simulated catch, gap (simulated - real, run 1), change vs as-is and share of the as-is gap closed, in points.
  python3 -I report_fixbyfix.py results/fixbyfix_<reader>_<stamp>.json [--layers 48] [--probes a,b]"""
import json
import sys


def pc(x, k=1):
    return f"{x['point'] * 100:+.{k}f} [{x['ci95'][0] * 100:+.{k}f}, {x['ci95'][1] * 100:+.{k}f}]"


def main(path, layers=(48, 63), probes=("a", "b")):
    d = json.load(open(path))
    print(f"{d['reader']}  stamp {d['stamp']}  git {d['git'][:10]}  ({d['label']})")
    for m in d["metrics"]:
        if m["layer"] not in layers:
            continue
        print(f"== L{m['layer']}  gates: probe-b rows reproduced {m['gate: on-policy probe-b rows reproduced exactly']}; coverage "
              f"{m['gate: coverage']['copies']} copies, no-ops {m['gate: coverage']['fixes that change no run (verified no-ops)']}; rescoring max diff "
              f"{m['gate: frozen rescoring of #36 runs']['max |difference|']:.1e}")
        for key, r in m["results"].items():
            cal, alpha, probe = key.split("|")
            if probe not in probes:
                continue
            flag = " [as-is gap CI contains 0: shares unstable]" if r["as-is gap interval contains 0"] else ""
            k0 = " [k = 0, descriptive only]" if r["k = 0 (descriptive only)"] else ""
            print(f"-- {cal}, {float(alpha) * 100:.0f}%, probe {probe}: real catch (run 1) {r['real catch, run 1 (attempted)']['point'] * 100:.1f}{flag}{k0}")
            for v, row in r["rows"].items():
                ch = row.get("change vs as-is"); sh = row.get("share of the as-is gap closed")
                print(f"   {v:13s} n={row['simulated runs re-read']:3d} sim {row['simulated catch']['point'] * 100:5.1f}  gap {pc(row['gap = simulated - real (run 1)'])}"
                      + (f"  change {pc(ch)}" if ch else "") + (f"  share {sh['point']:+.2f} [{sh['ci95'][0]:+.2f}, {sh['ci95'][1]:+.2f}]" if sh else "")
                      + f"  sim-honest FA {row['simulated honest false alarms at this cutoff']['point'] * 100:.1f}")
            sk = [k for k in r if k.startswith("sum of single-fix changes")][0]
            if r[sk]:
                print(f"   {sk}: {pc(r[sk])}  vs all together ({r['all together']}) {pc(r['rows']['all together']['change vs as-is'])}")


if __name__ == "__main__":
    a = sys.argv[1:]
    layers = tuple(int(x) for x in a[a.index("--layers") + 1].split(",")) if "--layers" in a else (48, 63)
    probes = tuple(a[a.index("--probes") + 1].split(",")) if "--probes" in a else ("a", "b")
    main(a[0], layers, probes)
