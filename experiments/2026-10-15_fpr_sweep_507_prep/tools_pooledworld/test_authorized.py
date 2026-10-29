"""Tests make_authorized.py and the runner's pw_auth_overlap_check on the SAME synthetic manifests / catalogues (fake keys only), per label x
combo maps that differ, a world / key mismatch, plus both on the held attempt-3 catalogue (keys only; given 4 arguments), the primary rule
against the runner's own matcher pattern, 09's Δ_rr label verbatim, and the per-row Δ_rr overlap and the reducer's member agreement.
Usage (the runner import needs the repo env): <repo>/.venv/bin/python test_authorized.py [<held manifest> <held catalogue> <manifest ref> <catalogue ref>]"""
import copy
import hashlib
import json
import re
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).parent
sys.path[:0] = [str(HERE), str(HERE.parent)]
import make_authorized as M  # noqa: E402
import run_peraction311 as R  # noqa: E402

D = Path(tempfile.mkdtemp())
ok = []
REF = "xfer/fpr507prep/records/results/x_2026-10-16_000000"
HC = [f"honest_c|t{i:03d}_main" for i in range(192)]


def world(k):
    return k.split("|", 1)[0]


def case(over, runs, over2=None):
    """over: {key: roles} in two label x combos (over2, when given, in the second); runs: the catalogue. Returns (make_authorized STOPs?, runner STOPs?)."""
    man = {"per_label_combo": {f"p{j} | pooled_all_origins": {"cross_pool_overlaps": {"roles_by_key": over if j == 0 or over2 is None else over2}}
                               for j in range(2)}}
    mb, cb = json.dumps(man).encode(), json.dumps({"runs": runs}).encode()
    (D / "m.json").write_bytes(mb)
    (D / "c.json").write_bytes(cb)
    res = []
    for f in (lambda: M.main(D / "m.json", D / "c.json", f"{REF}_manifest.json@{hashlib.sha256(mb).hexdigest()}",
                             f"{REF}_catalogue.json@{hashlib.sha256(cb).hexdigest()}", D / "out.json"),
              lambda: R.pw_auth_overlap_check(man["per_label_combo"], runs)):
        try:
            f()
            res.append(False)
        except SystemExit:
            res.append(True)
    return tuple(res)


good_runs = {k: {"world": "honest_c", "sets": ["S_old"], "roles": ["A", "B"]} for k in HC} | {
    "honest_b|t000_main": {"world": "honest_b", "sets": ["S_old"], "roles": ["B"]}}
good_over = {k: ["A", "B"] for k in HC}
ok.append(("the 192 S_old honest_c runs in A and B pass both", case(good_over, good_runs) == (False, False)))
r = copy.deepcopy(good_runs)
r[HC[5]]["sets"] = ["S_new", "S_old"]
ok.append(("an honest_c run also admitted by S_new STOPs both (B1)", case(good_over, r) == (True, True)))
r = copy.deepcopy(good_runs)
r[HC[5]]["sets"] = ["S_new"]
ok.append(("an honest_c run from S_new only STOPs both (B1)", case(good_over, r) == (True, True)))
ok.append(("an extra overlap (honest_b in A and B) STOPs both", case(good_over | {"honest_b|t000_main": ["A", "B"]}, good_runs) == (True, True)))
ok.append(("an honest_c run in A, B and S STOPs both", case(good_over | {HC[0]: ["A", "B", "S"]}, good_runs) == (True, True)))
ok.append(("an honest_c run missing from the overlaps STOPs both", case({k: v for k, v in good_over.items() if k != HC[0]}, good_runs) == (True, True)))
r = {k: v for k, v in good_runs.items() if k != HC[0]}
ok.append(("191 honest_c runs STOP both", case({k: v for k, v in good_over.items() if k != HC[0]}, r) == (True, True)))
ok.append(("no overlaps at all STOP both", case({}, good_runs) == (True, True)))
ok.append(("Astra r9 #1: one label x combo missing an honest_c overlap STOPs both (no union across combos)",
           case(good_over, good_runs, {k: v for k, v in good_over.items() if k != HC[0]}) == (True, True)))
ok.append(("Astra r9 #1: one label x combo with no overlaps STOPs both", case(good_over, good_runs, {}) == (True, True)))
r = {("honest_b|t191_main" if k == HC[-1] else k): v for k, v in good_runs.items()}
o = {("honest_b|t191_main" if k == HC[-1] else k): v for k, v in good_over.items()}
ok.append(("Astra r9 #2: a key whose catalogued world is not its key's world token STOPs both", case(o, r) == (True, True)))
case(good_over, good_runs)  # rewrite the passing case's files for the B2 and citation checks
mb, cb = (D / "m.json").read_bytes(), (D / "c.json").read_bytes()
mr, cr = f"{REF}_manifest.json@{hashlib.sha256(mb).hexdigest()}", f"{REF}_catalogue.json@{hashlib.sha256(cb).hexdigest()}"


def stops(f):
    try:
        f()
        return False
    except SystemExit:
        return True


ok.append(("a catalogue reference not beside the manifest STOPs (B2)", stops(lambda: M.main(D / "m.json", D / "c.json", mr, cr.replace("x_2026", "y_2026"), D / "o.json"))))
ok.append(("a catalogue copy with another sha256 STOPs (B2)", stops(lambda: M.main(D / "m.json", D / "c.json", mr, cr[:-1] + ("0" if cr[-1] != "0" else "1"), D / "o.json"))))
ents = M.main(D / "m.json", D / "c.json", mr, cr, D / "o.json")
rx = re.search(r'RX_RULE_ = r"([^"]+)"', (HERE.parent / "run_peraction311.py").read_text()).group(1)  # the runner's own matcher pattern
ok.append(("the primary rule fully matches the runner's matcher and is READ_LOG @ 4ca9769a4:936",
           all(re.fullmatch(rx, e["rule"]) for e in ents) and ents[0]["rule"] == "READ_LOG @ 4ca9769a4:936"))
ok.append(("the 395-407 range is in the note; a range never matches the runner's matcher", "RULE @ efdfcc8b2:395-407" in ents[0]["note"]
           and not re.fullmatch(rx, "RULE @ efdfcc8b2:395-407")))
keep = list(M.CITES)
M.CITES[0] = ("RULE.md", "efdfcc8b2", 395, 410, "## Append 6")  # past the file's 409 lines
ok.append(("a range end past the file STOPs", stops(lambda: M.main(D / "m.json", D / "c.json", mr, cr, D / "o.json"))))
M.CITES[0] = ("RULE.md", "efdfcc8b2", 396, 407, "## Append 6")
ok.append(("a line without the quoted words STOPs", stops(lambda: M.main(D / "m.json", D / "c.json", mr, cr, D / "o.json"))))
M.CITES[0] = ("RULE.md", "0000000aa", 395, 407, "## Append 6")
ok.append(("an unknown commit STOPs", stops(lambda: M.main(D / "m.json", D / "c.json", mr, cr, D / "o.json"))))
M.CITES[:] = keep
ok.append(("09's pooled Δ_rr label is verbatim", R.PW_DRR_LABEL == "Pooled: τ_A and τ_B share S_old's honest C (192 runs; per set the two pools "
           "were disjoint), so part of their threshold noise is shared and this real-vs-real yardstick may be slightly narrow."))
# Claude r9 NB 5: the per-row overlap (pw_drr_overlap) and the reducer's member agreement (pw_drr_members)
A_ = [f"honest_a|t{i:03d}_main" for i in range(300)] + HC[:190]  # H_A: 300 honest_a + 190 honest_c with units
B_ = [f"honest_b|t{i:03d}_main" for i in range(400)] + HC  # H_B: 400 honest_b + all 192 honest_c with units
d = R.pw_drr_overlap(A_, B_, HC[190:], [], "t")  # honest_c 190, 191 have zero units in H_A (admitted there as zero-unit runs)
ok.append(("pw_drr_overlap: admitted basis 192 shared / 492 H_A / 592 H_B; with units 190 / 490 / 592; both bases named",
           (d["shared_runs"], d["H_A_runs"], d["H_B_runs"]) == (192, 492, 592) and d["share_of_H_A"] == 192 / 492 and d["share_of_H_B"] == 192 / 592
           and (d["with_units"]["shared_runs"], d["with_units"]["H_A_runs"], d["with_units"]["H_B_runs"]) == (190, 490, 592)
           and d["with_units"]["share_of_H_A"] == 190 / 490 and d["with_units"]["share_of_H_B"] == 190 / 592  # Astra r9c optional 1
           and "admitted" in d["basis"] and "with units" in d["with_units"]["basis"]))
ok.append(("pw_drr_overlap: 191 shared runs STOP", stops(lambda: R.pw_drr_overlap(A_, B_[:-1], HC[190:191], [], "t"))))
ok.append(("pw_drr_overlap: a shared run outside honest_c STOPs, with exactly 192 shared (the world check alone; Astra r9c optional 2)",
           len((set(A_ + ["honest_b|t000_main"]) | set(HC[190:191])) & set(B_)) == 192
           and stops(lambda: R.pw_drr_overlap(A_ + ["honest_b|t000_main"], B_, HC[190:191], [], "t"))))
d2 = json.loads(json.dumps(d))  # a stored row read back
ok.append(("pw_drr_members: identical member dicts (one read back from JSON) pass", R.pw_drr_members([d, d2, d], "t") == d))
ok.append(("pw_drr_members: one differing member STOPs", stops(lambda: R.pw_drr_members([d, d | {"H_A_runs": 491}], "t"))))
ok.append(("pw_drr_members: no members STOPs", stops(lambda: R.pw_drr_members([], "t"))))
if len(sys.argv) == 5:  # the held attempt-3 manifest and catalogue (keys only)
    hm, hcat, hmr, hcr = sys.argv[1:5]
    e2 = M.main(hm, hcat, hmr, hcr, D / "held.json")
    chk = R.pw_auth_overlap_check(json.loads(Path(hm).read_bytes())["per_label_combo"], json.loads(Path(hcat).read_bytes())["runs"])
    ok.append(("held attempt-3 catalogue: both pass with 192 S_old honest_c keys", len(e2) == 192 and chk["overlap_runs"] == 192))
for n, v in ok:
    print("PASS" if v else "FAIL", n)
print("ALL PASS" if all(v for _, v in ok) else "FAILURES")
sys.exit(0 if all(v for _, v in ok) else 1)
