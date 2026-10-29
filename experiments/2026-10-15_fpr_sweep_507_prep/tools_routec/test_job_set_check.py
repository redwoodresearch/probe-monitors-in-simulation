"""Synthetic checks of job_set_check (09's S1a item 3 job sets, 10-18). No Modal call, no real spec, no git."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import job_set_check as J  # noqa: E402


def main():
    s0 = {"jobs": [("L C", {"origin": "S_old"}, {}), ("L R", {"origin": "S_old"}, {}), ("L C", {"origin": "S_new"}, {})]}
    js = [(f"{b} {r}", {"origin": o}, {}) for b in ("Sol (a)", "FIBS") for r in "CR" for o in ("S_old", "S_new")]
    drop, f, g = J.job_set(s0, s0 | {"jobs": s0["jobs"][:2]}), J.job_factors({"jobs": js}), J.job_factors({"jobs": js[1:]})
    cases = {
        "a spec against itself: equal": J.job_set(s0, s0)["equal"],
        "a dropped job: listed by key on its side, not equal": not drop["equal"] and drop["only_a"] == ["L C | S_new"]
        and (drop["n_a"], drop["n_b"], drop["n_only_a"], drop["n_only_b"]) == (3, 2, 1, 0),
        "a repeated key: not equal (multiset)": not J.job_set(s0, s0 | {"jobs": s0["jobs"] + s0["jobs"][:1]})["equal"],
        "the key is the row label (full_as) and _row_impl's origin tag": J.job_parts(("x C", {"origin": "S_newmains", "no_s4": True, "full_as": "y C"}, {}))
        == ("y C", "S_newmains without S4") and J.job_parts(("x R", {}, {})) == ("x R", "S_new"),
        "a 2 x 2 x 2 product is full; one missing cell is not": (f["n_jobs"], f["n_label_bases"], f["n_readouts"], f["n_origin_tags"], f["product"],
                                                                  f["full_product"]) == (8, 2, 2, 2, 8, True) and not g["full_product"],
    }
    for name, ok in cases.items():
        print(("PASS " if ok else "FAIL ") + name)
    assert all(cases.values()), "job_set_check tests failed"
    print(f"{len(cases)} checks pass")


if __name__ == "__main__":
    main()
