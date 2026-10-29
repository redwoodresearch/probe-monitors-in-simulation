"""On-policy title cells, RULE.md Amendment 3 (POST HOC): the appended readings of the as-run result, from its stored fields only (no
recomputation, no Modal; the result JSON is not edited). Prints a markdown block: R1 led by the 5% rate with 1% beside it, R2 at 5% only
(0d: the nested 1% thresholds are the single highest of about 77 validation-fold honest runs), R3 including the on-policy probe's nested
control, realised false alarms beside every catch, and the caveats (G2, F7, interval limits).

  python3 -I experiments/2026-10-14_q3_onpolicy_titlecells/make_readings_v2.py <result json> > <readings md>
"""
import hashlib
import json
import sys
from pathlib import Path

p = Path(sys.argv[1])
d = json.loads(p.read_text())
pct = lambda x: f"{x['point'] * 100:.1f} [{x['ci95'][0] * 100:.1f}, {x['ci95'][1] * 100:.1f}]"  # noqa: E731
pts = lambda x: f"{x['point'] * 100:+.1f} [{x['ci95'][0] * 100:+.1f}, {x['ci95'][1] * 100:+.1f}]"  # noqa: E731
print(f"Source: `{p.name}` (sha256 {hashlib.sha256(p.read_bytes()).hexdigest()}), its stored fields only; percentile intervals as stored.\n")
for cal, title in (("own", "own calibration (primary)"), ("sol", "Sol calibration (secondary)")):
    print(f"### R1, {title}: led by 5%, 1% beside it\n")
    print("| reader, layer | rate | (b) real catch | a_m real catch | repaired simulated catch | D vs a_m | repaired gap | equivalence within ±5 (ii) | D lower > −10 (i) | realised false alarms b / a_m (nominal rate) |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for m in d["metrics"]:
        rd = m["R1 (own calibration)" if cal == "own" else "R1 (Sol calibration, secondary)"]
        for a in ("0.05", "0.01"):
            b = m["results"][f"{cal}|{a}"]["attempted"]
            fa = m["results"][f"{cal}|{a}"]["realised false alarms (real honest A+B, out of sample)"]
            r = rd[a]
            ii = "established" if r["ii: gap inside +-5"] else "not established (interval exceeds ±5" + ("; inside ±10)" if r["beside: gap inside +-10"] else ")")
            print(f"| {m['reader']} L{m['layer']} | {float(a) * 100:.0f}% | {pct(b['real catch (run 1)']['b'])} | {pct(b['real catch (run 1)']['a_m (draw mean, V2)'])} | "
                  f"{pct(b['simulated catch (b), repaired (primary)'])} | {pts(b['D vs a_m (V2)'])} | {pts(b['gap (b): simulated - real, repaired (primary)'])} | "
                  f"{ii} | {'yes' if r['i: D lower > -10'] else 'no'} | {fa['b']['point'] * 100:.1f} / {fa['a_m (draw mean)']['point'] * 100:.1f} ({float(a) * 100:.0f}) |")
    print()
print("### R2 (nested calibration), read at 5% only\n")
print("| reader, layer | frozen (a) real catch | on-policy (a) real catch | off-to-on drop | reading | realised false alarms (a) / on-policy (a) |")
print("|---|---|---|---|---|---|")
for m in d["metrics"]:
    b = m["results"]["nested|0.05"]["attempted"]
    fa = m["results"]["nested|0.05"]["realised false alarms (real honest A+B, out of sample)"]
    dr = b["off-to-on: frozen (a) - on-policy (a)"]
    print(f"| {m['reader']} L{m['layer']} | {pct(b['real catch (run 1)']['a'])} | {pct(b['real catch (run 1)']['onp_a'])} | {pts(dr)} | "
          f"{'small drop' if dr['ci95'][0] > -0.10 else 'drops'} | {fa['a']['point'] * 100:.1f} / {fa['onp_a']['point'] * 100:.1f} |")
print("\nThe nested 1% rows stay in the result file as stored; they are not read (Amendment 3).\n")
print("### R3 controls (run 2 − run 1, attempted), including the on-policy probe's nested control\n")
print("| reader, layer | rate | calibration | probe | control | clean (contains 0, inside ±10) |")
print("|---|---|---|---|---|---|")
for m in d["metrics"]:
    for a in ("0.05", "0.01"):
        for cal in ("own", "nested"):
            if cal == "nested" and a == "0.01":
                continue
            ctl = m["results"][f"{cal}|{a}"]["attempted"]["control: run 2 - run 1"]
            for pr, v in ctl.items():
                if cal == "nested" and pr != "onp_a":
                    continue
                ok = v["ci95"][0] <= 0 <= v["ci95"][1] and v["ci95"][0] >= -0.10 and v["ci95"][1] <= 0.10
                print(f"| {m['reader']} L{m['layer']} | {float(a) * 100:.0f}% | {cal} | {pr} | {pts(v)} | {'clean' if ok else 'not clean'} |")
