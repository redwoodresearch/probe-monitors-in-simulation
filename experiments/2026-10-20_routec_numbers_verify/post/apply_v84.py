"""Swap route C's readout strings into the draft (v83 prep -> v84).

Usage: python apply_v84.py DRAFT_IN.md READOUT.json DRAFT_OUT.md REPORT.md
- READOUT.json is the slot readout's output: {"slots": {key: string}, ...}. Strings are placed as given (formatting only;
  nothing is computed except picking the largest |point| for two "largest shift" sentences, which is flagged for review).
- Each rule applies only if every key it needs is present (so drop 1 and drop 2 can be applied separately).
- Every anchor must occur exactly once in the input, or the run stops.
- Sentences whose wording depends on a sign, a ranking or whether an interval includes zero are FLAGGED for hand review.
"""
import json, re, sys

MINUS = "−"
din, rin, dout, rep = sys.argv[1:5]
text = open(din, encoding="utf-8").read()
S = json.load(open(rin, encoding="utf-8"))["slots"]
log, flags, skipped = [], [], []

NUM = re.compile(r"^([−+]?[\d,]*\.?\d+)(%?)(?:\s*\((.+?) to (.+?)\))?$")


def parse(key):
    s = S[key].strip()
    m = NUM.match(s)
    if not m:
        return {"raw": s}  # ranges and other text slots are placed as given
    return {"raw": s, "pt": m.group(1), "pct": m.group(2), "lo": m.group(3), "hi": m.group(4)}


def val(p):  # sign of a formatted point, for direction words only
    return float(p.replace(MINUS, "-").replace("+", "").replace(",", ""))


def absv(p):
    return p.lstrip("+").lstrip(MINUS)


def ci(p):
    return f"{p['lo']} to {p['hi']}"


def ci_has_zero(p):
    return val(p["lo"]) <= 0 <= val(p["hi"])


def need(*keys):
    miss = [k for k in keys if k not in S]
    return miss


def sub(rule, old, new, keys, old_sign=None, check=None):
    """Replace `old` (must occur once) with `new` (a function of parsed slots)."""
    global text
    miss = need(*keys)
    if miss:
        skipped.append((rule, miss))
        return
    n = text.count(old)
    if n != 1:
        raise SystemExit(f"STOP {rule}: anchor occurs {n} times: {old[:90]!r}")
    P = {k: parse(k) for k in keys}
    text = text.replace(old, new(P))
    log.append(rule)
    if check:
        for msg in check(P):
            flags.append((rule, msg))


def lowhigh(p):
    return "lower" if val(p["pt"]) < 0 else "higher"


def fewmore(p):
    return "fewer" if val(p["pt"]) < 0 else "more"


def sign_flag(name, p, was):
    now = "−" if val(p["pt"]) < 0 else "+"
    return [f"{name}: sign changed from {was} to {now}; reread the sentence"] if now != was else []


def zero_flag(name, p, was_includes):
    now = ci_has_zero(p)
    return [f"{name}: interval {'now includes' if now else 'now excludes'} zero (was {'includes' if was_includes else 'excludes'}); reread wording"] if now != was_includes else []


K = lambda n, grp, ro, lv, f: f"{n:02d}.coheadline.{grp}|{ro}@{lv}.{f}"
S6C, Y6C = K(1, "headline_six_full", "C", "0.01", "S"), K(1, "headline_six_full", "C", "0.01", "Y")
S4C, Y4C = K(2, "auroc06_subset", "C", "0.01", "S"), K(2, "auroc06_subset", "C", "0.01", "Y")
S6R, Y6R = K(3, "headline_six_full", "R", "0.01", "S"), K(3, "headline_six_full", "R", "0.01", "Y")
S4R, Y4R = K(4, "auroc06_subset", "R", "0.01", "S"), K(4, "auroc06_subset", "R", "0.01", "Y")
S6C5 = K(5, "headline_six_full", "C", "0.05", "S")
E6C, E6R, E4C = "06.excess.headline_six_full|C", "07.excess.headline_six_full|R", "08.excess.four_probe|C"
LAB = {"on": "own_all_rs_uncut_L48", "off": "sol_a_uncut", "dec": "FIBS|uncut", "harm": "harmfulness|uncut",
       "ref": "refusal|uncut", "hs": "high-stakes|uncut"}
NAME = {"on": "on-policy probe", "off": "off-policy probe", "dec": "deception probe", "harm": "harmfulness direction",
        "ref": "refusal direction", "hs": "high-stakes probe"}
ROW = {"on": "On-policy probe", "off": "Off-policy probe", "dec": "Deception probe", "harm": "Harmfulness direction",
       "ref": "Refusal direction", "hs": "High-stakes probe"}
D = lambda p, ro: f"10.delta_full.{LAB[p]} {ro}"
Yp = lambda p, ro: f"11.run2_minus_run1_at_tauB.{LAB[p]} {ro}"
C = lambda p, ro, arm: f"09.catch.{LAB[p]} {ro}.{arm}"

# ---------------- main text ----------------
sub("M1 headline", "simulated catch was 1.0 points lower than real catch (95% interval −3.1 to +1.0), and real run 2 was 0.2 points lower than real run 1",
    lambda P: f"simulated catch was {absv(P[S6C]['pt'])} points {lowhigh(P[S6C])} than real catch (95% interval {ci(P[S6C])}), and real run 2 was {absv(P[Y6C]['pt'])} points {lowhigh(P[Y6C])} than real run 1",
    [S6C, Y6C], check=lambda P: sign_flag("six-probe S", P[S6C], "−") + sign_flag("six-probe Y", P[Y6C], "−") + zero_flag("six-probe S", P[S6C], True))
sub("M2 four", "Over those four, simulated catch was 1.4 points lower than real catch, and real run 2 was 0.6 points higher than real run 1.",
    lambda P: f"Over those four, simulated catch was {absv(P[S4C]['pt'])} points {lowhigh(P[S4C])} than real catch, and real run 2 was {absv(P[Y4C]['pt'])} points {lowhigh(P[Y4C])} than real run 1.",
    [S4C, Y4C], check=lambda P: sign_flag("four-probe S", P[S4C], "−") + sign_flag("four-probe Y", P[Y4C], "+"))
sub("M2 fn", "the 95% intervals are −3.7 to +0.8 points for simulated minus real catch and −0.9 to +2.2 for real run 2 minus real run 1.",
    lambda P: f"the 95% intervals are {ci(P[S4C])} points for simulated minus real catch and {ci(P[Y4C])} for real run 2 minus real run 1.",
    [S4C, Y4C], check=lambda P: zero_flag("four-probe S", P[S4C], True))
sub("M3 tool", "Their simulated catch was 1.3 points lower than real catch, while real run 2 caught 0.9 points more than real run 1",
    lambda P: f"Their simulated catch was {absv(P[S6R]['pt'])} points {lowhigh(P[S6R])} than real catch, while real run 2 caught {absv(P[Y6R]['pt'])} points {fewmore(P[Y6R])} than real run 1",
    [S6R, Y6R], check=lambda P: sign_flag("tool S", P[S6R], "−") + sign_flag("tool Y", P[Y6R], "+"))
sub("M3 fn", "On tool outputs, the 95% intervals are −3.9 to +1.1 points for simulated minus real catch and −0.6 to +2.3 for real run 2 minus real run 1.",
    lambda P: f"On tool outputs, the 95% intervals are {ci(P[S6R])} points for simulated minus real catch and {ci(P[Y6R])} for real run 2 minus real run 1.",
    [S6R, Y6R], check=lambda P: zero_flag("tool S", P[S6R], True))
sub("M4 halfwidth main", "Single probes are less certain, with intervals of up to ±6.0 points.",
    lambda P: f"Single probes are less certain, with intervals of up to ±{absv(P['10.halfwidth_C.max_all']['pt'])} points.",
    ["10.halfwidth_C.max_all"])
for rid, anchor in (("M5 share main", "so at that level 26% to 42% of honest runs raise at least one alarm, depending on the probe."),
                    ("M5 share appA", "At 1 alarm per 100 honest commands, 26% to 42% of honest runs raise at least one alarm, depending on the probe and setting.")):
    sub(rid, anchor, lambda P, a=anchor: a.replace("26% to 42%", S["15.alarm_share.range"].strip()), ["15.alarm_share.range"])
sub("M6 units", "The probes score about 65 to 75 commands per honest run",
    lambda P: f"The probes score about {S['16.units_per_run.range'].strip().removeprefix('about ')} commands per honest run", ["16.units_per_run.range"])
sub("M7 skip share", "That skips about one in seven of the calls the published probes score (13% to 14%).",
    lambda P: f"That skips about one in seven of the calls the published probes score ({S['17.readout_share.range'].strip()}).",
    ["17.readout_share.range"], check=lambda P: ["readout share: confirm 'about one in seven' still fits " + S["17.readout_share.range"]])
sub("M8 pools", "The real threshold is set on 1,906 real honest runs and the simulated threshold on 848 simulated ones",
    lambda P: f"The real threshold is set on {int(S['14.after_runs.sol_a_uncut C.B'].strip().replace(',', '')):,} real honest runs and the simulated threshold on {int(S['14.after_runs.sol_a_uncut C.S'].strip().replace(',', '')):,} simulated ones",
    ["14.after_runs.sol_a_uncut C.B", "14.after_runs.sol_a_uncut C.S"],
    check=lambda P: ["pool sizes: confirm 'about two and about one per main task' still fits"])

# ---------------- tables 5 and 8 ----------------
def table_rows(ro):
    rows = []
    for p in ("on", "off", "dec", "harm", "ref", "hs"):
        keys = [C(p, ro, "run1"), C(p, ro, "run2"), C(p, ro, "sim"), D(p, ro), Yp(p, ro)]
        if need(*keys):
            return None, need(*keys)
        r1, r2, sm = (parse(k)["pt"] for k in keys[:3])  # point only; tables show no catch intervals
        rows.append(f"| {ROW[p]} | {r1} | {r2} | {sm} | {S[keys[3]].strip()} | {S[keys[4]].strip()} |")
    return rows, []


for ro, first in (("C", "| On-policy probe | 99.8 | 99.8 | 99.5 |"), ("R", "| On-policy probe | 99.8 | 99.7 | 99.4 |")):
    rows, miss = table_rows(ro)
    if rows is None:
        skipped.append((f"T {ro} probe rows", miss))
        continue
    lines = text.split("\n")
    idx = [i for i, l in enumerate(lines) if l.startswith(first)]
    if len(idx) != 1:
        raise SystemExit(f"STOP table {ro}: first row found {len(idx)} times")
    i = idx[0]
    for j, r in enumerate(rows):
        if not lines[i + j].startswith("| " + r.split(" | ")[0][2:]):
            raise SystemExit(f"STOP table {ro}: row order differs at {lines[i + j][:40]!r}")
        lines[i + j] = r
    text = "\n".join(lines)
    log.append(f"T {ro} probe rows")
    flags.append((f"T {ro}", "average rows (six, four) are NOT swapped by this script: they need run 1 / run 2 / sim group means; fill from 09.catch_avg + S/Y slots by hand"))

# ---------------- appendix C ----------------
sub("C1 halfwidth", "with intervals of ±4.2 to ±6.0 points.",
    lambda P: f"with intervals of ±{absv(P['10.halfwidth_C.min_excl_on_policy']['pt'])} to ±{absv(P['10.halfwidth_C.max_excl_on_policy']['pt'])} points.",
    ["10.halfwidth_C.min_excl_on_policy", "10.halfwidth_C.max_excl_on_policy"])
sub("C2 on-policy", "which excludes zero (−0.35, 95% interval −1.36 to −0.02).",
    lambda P: f"which excludes zero ({P[D('on','C')]['pt']}, 95% interval {ci(P[D('on','C')])}).",
    [D("on", "C")], check=lambda P: zero_flag("on-policy C", P[D("on", "C")], False))
sub("C3 off-policy", "The off-policy probe caught 5.0 points fewer attacks in simulation than on real machines (95% interval −10.9 to +1.1).",
    lambda P: f"The off-policy probe caught {absv(P[D('off','C')]['pt'])} points {fewmore(P[D('off','C')])} attacks in simulation than on real machines (95% interval {ci(P[D('off','C')])}).",
    [D("off", "C")], check=lambda P: sign_flag("off-policy C", P[D("off", "C")], "−") + zero_flag("off-policy C", P[D("off", "C")], True)
    + ([f"off-policy run2−run1 is {S[Yp('off','C')]}: check 'differed by less than a point'"] if Yp("off", "C") in S and abs(val(parse(Yp('off', 'C'))['pt'])) >= 1 else []))


def next_largest(P):
    others = [(abs(val(P[D(p, 'C')]["pt"])), p) for p in ("on", "dec", "harm", "ref", "hs")]
    v, p = max(others)
    return f"the next-largest shift was {absv(P[D(p, 'C')]['pt'])} points, for the {NAME[p]}."


sub("C4 next-largest", "the next-largest shift was 2.3 points, for the harmfulness direction.", next_largest,
    [D(p, "C") for p in ("on", "dec", "harm", "ref", "hs")], check=lambda P: ["RANK: next-largest probe chosen by |point|; check the off-policy probe is still the largest"])
sub("C5 1in20", "−1.2 points averaged over the six probes (95% interval −2.4 to −0.2).",
    lambda P: f"{P[S6C5]['pt']} points averaged over the six probes (95% interval {ci(P[S6C5])}).",
    [S6C5], check=lambda P: sign_flag("1-in-20 S", P[S6C5], "−") + zero_flag("1-in-20 S (sentence says 'measurably below', also App C opening)", P[S6C5], False))

# ---------------- appendix D ----------------
sub("D1 gap six", "the simulation gap was 0.9 points larger than the rerun gap (95% interval −0.4 to +2.8).",
    lambda P: f"the simulation gap was {absv(P[E6C]['pt'])} points {'larger' if val(P[E6C]['pt']) >= 0 else 'smaller'} than the rerun gap (95% interval {ci(P[E6C])}).",
    [E6C], check=lambda P: sign_flag("excess six C", P[E6C], "+"))
sub("D2 gap four", "the difference was +1.7 points (95% interval −0.5 to +3.6) (Figure 5).",
    lambda P: f"the difference was {P[E4C]['pt']} points (95% interval {ci(P[E4C])}) (Figure 5).", [E4C])
flags.append(("D3 extra probe", "App D extra-probe sentence (c_asis_uncut, 13.delta_full / 13.real_line): REWRITE BY HAND; pooled signs may differ from per-set"))

# ---------------- appendix F ----------------
sub("F1 tool six", "simulated catch minus real catch was −1.3 points (95% interval −3.9 to +1.1), against +0.9 for real run 2 minus real run 1 (Figure 6). On commands, where a blocking monitor would read, the same two numbers are −1.0 points (95% interval −3.1 to +1.0) and −0.2.",
    lambda P: f"simulated catch minus real catch was {P[S6R]['pt']} points (95% interval {ci(P[S6R])}), against {P[Y6R]['pt']} for real run 2 minus real run 1 (Figure 6). On commands, where a blocking monitor would read, the same two numbers are {P[S6C]['pt']} points (95% interval {ci(P[S6C])}) and {P[Y6C]['pt']}.",
    [S6R, Y6R, S6C, Y6C])


def two_largest(P):
    tops = sorted(((abs(val(P[D(p, 'R')]["pt"])), p) for p in LAB), reverse=True)[:2]
    (_, a), (_, b) = tops
    inc = all(ci_has_zero(P[D(x, 'R')]) for x in (a, b))
    return (f"The two largest shifts were the {NAME[a]}, {P[D(a,'R')]['pt']} points (95% interval {ci(P[D(a,'R')])}), and the {NAME[b]}, "
            f"{P[D(b,'R')]['pt']} points (95% interval {ci(P[D(b,'R')])}). {'Both intervals include zero' if inc else 'NOT BOTH INTERVALS INCLUDE ZERO — REWRITE'}, and Table 8 gives all six probes.")


sub("F2 two largest", "The two largest shifts were the refusal direction, −3.7 points (95% interval −8.7 to +2.9), and the deception probe, −2.5 points (95% interval −8.6 to +2.4). Both intervals include zero, and Table 8 gives all six probes.",
    two_largest, [D(p, "R") for p in LAB], check=lambda P: ["RANK: two largest tool-output shifts chosen by |point|; reread"])
sub("F3 tool gap", "the six-probe simulation gap on tool outputs exceeds the rerun gap by +0.7 points (95% interval −0.5 to +2.9), against +0.9 on commands.",
    lambda P: f"the six-probe simulation gap on tool outputs exceeds the rerun gap by {P[E6R]['pt']} points (95% interval {ci(P[E6R])}), against {P[E6C]['pt']} on commands.",
    [E6R, E6C], check=lambda P: sign_flag("excess six R ('exceeds')", P[E6R], "+"))


# ---------------- drop-2 guard (09): visible drop-2 text must carry the marker until slot 19 arrives ----------------
DROP2 = [("At 1 in 1,000, the simulated threshold rests on", "19.")]
for anchor, prefix in DROP2:
    have = any(k.startswith(prefix) for k in S)
    if not have:
        for line in text.split("\n"):
            if anchor in line and "DROP2-MARKER" not in line:
                raise SystemExit(f"STOP drop-2 guard: visible text {anchor!r} has no DROP2-MARKER and slots {prefix}* are absent")
        flags.append(("drop-2", f"{anchor!r} still at the 171-pool value (marked); fill at drop 2"))
exec(open(__file__.replace("apply_v84.py", "appendix_rules.py"), encoding="utf-8").read())
open(dout, "w", encoding="utf-8").write(text)
with open(rep, "w", encoding="utf-8") as f:
    f.write(f"# v84 swap report\n\nApplied: {len(log)} rules\n" + "".join(f"- {r}\n" for r in log))
    f.write(f"\nSkipped (keys missing): {len(skipped)}\n" + "".join(f"- {r}: {', '.join(m[:4])}{' …' if len(m) > 4 else ''}\n" for r, m in skipped))
    f.write(f"\nFLAGS for hand review: {len(flags)}\n" + "".join(f"- {r}: {m}\n" for r, m in flags))
print(f"applied {len(log)}, skipped {len(skipped)}, flags {len(flags)}")
