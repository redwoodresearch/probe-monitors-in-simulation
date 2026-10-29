"""Append-only label NOTE for own_all_rs_L48's verify record (09 via 4f: every record carries amendment 3 / 3a's labels). Binds the verify record
by path + sha256 and restates its gate facts from the committed record; also checks, by key, the fold list's r36-basis mains against Sol #36's
runs.json in git (@ dea0a4e79), including the 7 r36 mains outside own_all v1's universe that the independent recheck could not check.
Keys, counts and shas only; no Modal; no score.

  python3 -I experiments/2026-10-16_own_all_rs/make_verify_label_note.py
"""

import hashlib
import json
import subprocess
from pathlib import Path

FAM = "experiments/2026-10-14_p2_family_expansion/results"
VERIFY = f"{FAM}/own_all_rs_verify_2026-10-16_004620.json"
FOLDS = "experiments/2026-10-16_own_all_rs/results/own_all_rs_folds_2026-10-16.json"
RUNS36 = ("dea0a4e79", "experiments/2026-10-06_q3_attack_detector/results/run_scores_2026-10-06_175649/runs.json", "76b11486cc353ab1eec390d49ef7449d729d4039d1e85643541d6679d73c3b96")
V1 = (f"{FAM}/own_all_inputs_2026-10-15.json", "26f9d8d1fe08dd2ef430a1bef8af50f336da47234fe364e1a94879bcb54491d1")
LABELS = {
    "section_2_and_3a_a_disclosure": "headline on-policy probe chosen by the author after own_all v1's values were seen; launch-disjoint sensitivity (LOO) beside "
                                     "(disclosure NOTE: experiments/2026-10-14_priority1_spec_09/READ_LOG.md @ 300b4008b, 0d's decision log)",
    "3a_d_launch_wording": "trained on other mains' run1 and honest_a across the included sets; The v1 diagnostic (own_all v1, three older sets) did not detect the "
                           "specified differential launch pattern; it does not establish negligible launch effects (e.g. own_all's S_old reply catch DiD at 1% spans "
                           "about −0.0 to +10.7 points, @ c6f0b432d)",
    "3a_f_training_sets": "trained on the three earlier sets; S_paper493 scored, not trained on",
    "3a_c_scoring_assert": "0 fold −1 units: every scoring job STOPs on a fold −1 row or a scored main outside the pinned fold list @ ec56720ad",
}


def git(*a, b=False):
    r = subprocess.run(["git", *a], capture_output=True, check=True)
    return r.stdout if b else r.stdout.decode().strip()


def main():
    vb = git("show", f"HEAD:{VERIFY}", b=True)
    v = json.loads(vb)
    if v.get("pass") is not True:
        raise SystemExit("the verify record did not pass")
    fm1 = {k: c["rows_fold_minus1"] for k, c in v["rows"]["counts"].items()}
    if any(fm1.values()):
        raise SystemExit("the verify record has fold -1 rows")
    fl = json.loads(git("show", f"HEAD:{FOLDS}", b=True))
    rb = git("show", f"{RUNS36[0]}:{RUNS36[1]}", b=True)
    if hashlib.sha256(rb).hexdigest() != RUNS36[2] or fl["pins"]["runs36_sha256"] != RUNS36[2]:
        raise SystemExit("runs.json in git differs from the fold list's runs36 pin")
    r36 = {}
    for r in json.loads(rb):
        r36.setdefault(r["main_task"], set()).add(r["fold"])
    v1b = git("show", f"HEAD:{V1[0]}", b=True)
    if hashlib.sha256(v1b).hexdigest() != V1[1]:
        raise SystemExit("own_all v1 inputs differ from the pin")
    uni = set(json.loads(v1b)["folds"]["by_main"])
    r36_basis = {m: f for m, (f, b) in fl["by_main"].items() if b == "r36"}
    bad = sorted(m for m, f in r36_basis.items() if r36.get(m) != {f})
    outside = sorted(m for m in r36_basis if m not in uni)
    env_hash_with_r36 = sorted(m for m, (f, b) in fl["by_main"].items() if b == "env-hash" and m in r36)
    if bad or env_hash_with_r36:
        raise SystemExit(f"STOP: r36-basis folds differ from runs.json for {len(bad)} mains, or {len(env_hash_with_r36)} env-hash mains have an r36 fold")
    out = {"label": "own_all_rs_L48 verify record: label NOTE (append-only; 09 via 4f, before any own_all_rs value is used); keys, counts and shas only",
           "binds": {"path": VERIFY, "commit": git("log", "-1", "--format=%H", "--", VERIFY), "sha256": hashlib.sha256(vb).hexdigest()},
           "labels": LABELS,
           "verify_facts_from_the_record": {"pass": v["pass"], "rows": {k: c["rows"] for k, c in v["rows"]["counts"].items()}, "mismatch": {k: c["mismatch"] for k, c in v["rows"]["counts"].items()},
                                            "rows_fold_minus1": fm1, "negative_failed_closed": v["negative_e2e"]["failed_closed"], "score_records": v["score_records"]},
           "r36_basis_check": {"runs36": {"commit": RUNS36[0], "path": RUNS36[1], "sha256": RUNS36[2]}, "r36_basis_mains": len(r36_basis), "fold_equal_to_runs36": len(r36_basis) - len(bad),
                               "env_hash_mains_with_an_r36_fold": 0, "r36_mains_outside_own_all_v1_universe": outside,
                               "note": "the independent recheck's 7 unchecked r36 mains are in this list; every r36-basis main's fold equals runs.json's by key"}}
    p = Path(FAM) / "own_all_rs_verify_2026-10-16_004620_LABEL_NOTE.json"
    with p.open("x") as fh:
        json.dump(out, fh, indent=1, ensure_ascii=False)
    print(p, json.dumps(out["r36_basis_check"] | {"runs36": RUNS36[0]}))


if __name__ == "__main__":
    main()
