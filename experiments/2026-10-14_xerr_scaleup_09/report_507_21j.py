"""09's 507 report under Append 21j (accepted @ 579ebeb17): reads the released FINAL record and prints, per readout x alpha,
the fixed-prominence rows, real and sim catch beside each gap, the 21j between-origin verdicts with the numeric definition
of "materially different", and the old pool-if-match row. Pure read and format of committed values (no metric is
computed here; verdicts compare stored intervals with the registered ±5 bounds). PRELIMINARY.

  python3 experiments/2026-10-14_xerr_scaleup_09/report_507_21j.py > results/report_507_21j_<stamp>.md
"""

import hashlib
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REC = HERE / "results" / "xerr_scaleup_final_2026-10-15_013255.json"
SHA = "7dceb8bb25752aa3"
X = 5.0


def f(v, nd=1):
    return "n/a" if v is None or (isinstance(v, float) and not math.isfinite(v)) else f"{v:.{nd}f}"


def ci(c, nd=1):
    return "n/a" if not c or None in c else f"[{f(c[0], nd)}, {f(c[1], nd)}]"


def verdict(c90, c95):
    if not c90 or not c95 or None in c90 or None in c95:
        return "decision unavailable"
    if c90[0] > -X and c90[1] < X:
        return "equivalent"
    if c95[0] > X or c95[1] < -X:
        return "materially different"
    return "inconclusive at this n"


def main():
    raw = REC.read_bytes()
    if not hashlib.sha256(raw).hexdigest().startswith(SHA):
        sys.exit("record sha mismatch")
    r = json.loads(raw)["result"]
    out = print
    out("# 507 X_err report under Append 21j (PRELIMINARY)\n")
    out(f"Record: {REC.name} @ 3ce6ea55e (sha256 {SHA}…); released @ c901ff2e3; first read 09:44:23 PDT (READ_LOG_507.md).")
    out("Contrast: sim − mean(run1, run2) (registered primary). Units: percentage points of catch. Positive = simulation overstates catch.")
    out(f'21j "materially different": the 95% interval of D = Δ_S_new − Δ_S_old lies entirely outside ±{X:g} points; "equivalent": the 90% interval strictly inside ±{X:g}.\n')
    for key in ("R|0.05", "R|0.01", "C|0.05", "C|0.01"):
        p = r["results"][key]["primary"]
        rd, a = key.split("|")
        out(f"\n## Readout {'reply (mean)' if rd == 'R' else 'command (max)'}, FPR {float(a) * 100:g}%\n")
        el = p.get("eligible_v6_frozen") or []
        out(f"Frozen eligible members: {', '.join(el) if el else 'none'}.")
        sm = p.get("sim_minus_mean_real") or {}
        rx = p.get("reexecution_run2_minus_run1") or {}
        if len(el) < 2:
            out("Family X_err/U unavailable (fewer than 2 eligible members); per-probe rows below.\n")
        out("\n**Fixed rows (21j §2: mixture first, S_old beside it, then S_new).**\n")
        out("| row | mains | cells | X_err | U (approx 95%) | bias (family mean Δ) [95%] | rerun yardstick run2−run1: X_err / U |")
        out("|---|---|---|---|---|---|---|")
        for name, lab in (("combined", "mixture S_old ∪ S_new"), ("S_old", "S_old (T196)"), ("S_new", "S_new (+311)")):
            b = sm.get(name) or {}
            pl = b.get("pooled") or {}
            rb = (rx.get(name) or {}).get("pooled") or {}
            out(f"| {lab} | {b.get('mains', 'n/a')} | {b.get('cells', 'n/a')} | {f(pl.get('X_err'))} | {f(pl.get('U_approx_upper_95'))} | {f(pl.get('bias_mean'))} {ci(pl.get('bias_ci95'))} | {f(rb.get('X_err'))} / {f(rb.get('U_approx_upper_95'))} |")
        out("\nWeighted-RMS companion (21j §2b(i)): X_err_strat and U_strat pending (computed in the job revision from per-draw plug-ins, not here).")
        out("\n**Per probe: real and sim catch beside the gap.**\n")
        out("| origin | probe | catch run1 | catch run2 | catch mean real | catch sim | Δ = sim − mean real [95%] |")
        out("|---|---|---|---|---|---|---|")
        cp = p.get("catch_percent") or {}
        for name in ("combined", "S_old", "S_new"):
            c = cp.get(name) or {}
            pp = ((sm.get(name) or {}).get("pooled") or {}).get("per_probe") or {}
            for pr in el:
                d = pp.get(pr) or {}
                out(f"| {name} | {pr} | {f((c.get('catch_run1') or {}).get(pr))} | {f((c.get('catch_run2') or {}).get(pr))} | {f((c.get('catch_mean_real') or {}).get(pr))} | {f((c.get('catch_sim') or {}).get(pr))} | {f(d.get('delta'))} {ci(d.get('ci95'))} |")
        t = (p.get("pooling_tests_tost5") or {}).get("S_new-S_old") or {}
        out("\n**21j between-origin verdicts (D = Δ_S_new − Δ_S_old; never change prominence).**\n")
        out("| probe | D | 90% | 95% | verdict |")
        out("|---|---|---|---|---|")
        tp = t.get("per_probe") or {}
        for pr in el:
            d = tp.get(pr) or {}
            out(f"| {pr} | {f(d.get('diff'))} | {ci(d.get('ci90'))} | {ci(d.get('ci95'))} | {verdict(d.get('ci90'), d.get('ci95'))} |")
        out(f"\nDraws complete: {t.get('complete_draws', 'n/a')} of 10,000. MDD_0: pending (needs SE_diff from draws; job revision).")
        out("\n**Old pool-if-match rule (TOST ±5, all members), for comparison only.**\n")
        if not t or not t.get("defined"):
            out("Old-rule decision unavailable (gate not defined).")
        else:
            passes = {pr: bool((tp.get(pr) or {}).get("within_5")) for pr in el}
            gate = all(passes.values()) and bool(passes)
            out(f"Per member TOST pass: {', '.join(f'{k} {chr(10003) if v else chr(10007)}' for k, v in passes.items())}. Gate: {'PASS' if gate else 'FAIL'}.")
            out(f"Implied presentation under the old rule: {'the combined (pooled) row would lead' if gate else 'strata differ (equivalence not established); stratum rows would lead'}.")
            out(f"Differs from the 21j presentation shown above (mixture first): {'no' if gate else 'yes'}.")
            out(f"Record's own labels (computed under the old rule): pooled_label \"{t.get('pooled_label')}\"; headline \"{p.get('headline')}\".")
        st = p.get("strategy_pooling_tests_tost5") or {}
        out("\n**Strategy strata (Append 18, unchanged by 21j).**\n")
        out("| row | mains | X_err | U |")
        out("|---|---|---|---|")
        for name in ("S_old|with_strategy", "S_old|without_strategy", "S_new|with_strategy", "S_new|without_strategy"):
            b = sm.get(name) or {}
            pl = b.get("pooled") or {}
            out(f"| {name} | {b.get('mains', 'n/a')} | {f(pl.get('X_err'))} | {f(pl.get('U_approx_upper_95'))} |")
        for o in ("S_old", "S_new"):
            if o in st:
                out(f"\nStrategy pooling test within {o}: {st[o].get('pooled_label', 'n/a')}.")
    out("\n---\nCaveats: probe readings are on the reader layout (no offered-tool block; donor honest system prompt on attack runs; no reasoning); served-context sensitivity per Appends 28/33. U is approximate, conditional on fixed calibration and eligibility; coverage not guaranteed. S_new's sim ran in a separate follow-up launch (not launch-matched). Command readout tokens differ by probe (Amendment 9). PRELIMINARY until 4f's harness re-verifies this report's values against the record.")


if __name__ == "__main__":
    main()
