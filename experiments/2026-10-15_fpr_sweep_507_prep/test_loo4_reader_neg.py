"""By-key negative tests of the LOO-4 S_paper493 reader (rows_any, loo4_pending): S_paper493 units go to model o3 only (a bare read on another
model STOPs); LOO-3 refuses S_paper493; inside LOO-4's S_paper493 row every unit (attack and pool) must be on o3 and no unit's main in o3's
training, else the row is pending (09: (a), else (c)).   python test_loo4_reader_neg.py   (no Modal call)"""

import numpy as np

import run_peraction311 as R

P3 = {"label": "own_all_LOO_L48 C", "family": "own_all_loo", "folds": 1}
P4 = P3 | {"loo_folds": R.OWN_ALL_LOO4_FOLD}


def z(folds):
    n = len(folds)
    return {"key": np.array([f"sim|t{600 + i}_tiktok" for i in range(n)]), "fold": np.array(folds), "score": np.zeros(n)}


def row_case(folds, read_as):
    """1 when the read ends the row as LOO4 pending (Loo4Pending, converted by row() to LOO4_PENDING), 0 when it passes."""
    q = P4 | {"loo_row_origin": "S_paper493", "_loo4_viol": []}
    try:
        R.rows_any(z(folds), q, read_as)
    except R.Loo4Pending as e:
        return int(bool(e.args[0]["non_o3_reads"]))
    return 0


def stops(fn):
    try:
        fn()
    except SystemExit:
        return True
    return False


cases = {
    "LOO-4 S_paper493 all on o3: passes": not stops(lambda: R.rows_any(z([3, 3, 3]), P4, "S_paper493")),
    "LOO-4 S_paper493 unit on o1 (key sim|t601_tiktok): STOPs": stops(lambda: R.rows_any(z([3, 1, 3]), P4, "S_paper493")),
    "LOO-4 S_paper493 unit on o0: STOPs": stops(lambda: R.rows_any(z([0]), P4, "S_paper493")),
    "LOO-3 on S_paper493: STOPs (refuses)": stops(lambda: R.rows_any(z([3]), P3, "S_paper493")),
    "LOO-4 S_new rows on o1: passes (unchanged routing)": not stops(lambda: R.rows_any(z([1, 1]), P4, "S_new")),
    "LOO-4 S_paper493 ROW, S_new pool file on o1: the row ends as LOO4 pending (no job STOP)": row_case([1], "S_new") == 1,
    "LOO-4 S_paper493 ROW, attack file with one unit on o2: the row ends as LOO4 pending": row_case([3, 2], "S_paper493") == 1,
    "LOO-4 S_paper493 ROW, pool file rescored on o3: passes": row_case([3], "S_new") == 0,
    "loo4_pending: all on o3, every main dropped by o3: None (row computes)": R.loo4_pending({"_loo4_viol": []}, ["m1", "m2"], {"m1", "m2"}) is None,
    "loo4_pending: a pool main in o3's training: pending": R.loo4_pending({"_loo4_viol": []}, ["m1", "m9"], {"m1"})["n_mains_in_o3_training_or_unknown"] == 1,
    "loo4_pending: an unknown main (None): pending": R.loo4_pending({"_loo4_viol": []}, [None], {"m1"}) is not None,
    "loo4_pending: a non-o3 read: pending": R.loo4_pending({"_loo4_viol": [{"keys": ["honest_b|t04"]}]}, ["m1"], {"m1"}) is not None,
    "the pending print is 09's exact text": R.LOO4_PENDING == "S_paper493: LOO4 pending (o3 pool scores / reader awaiting review)",
    "LOO-3 S_old on o0: passes (byte-identical path)": not stops(lambda: R.rows_any(z([0, 0]), P3, "S_old")),
}
for k, v in cases.items():
    print(("PASS " if v else "FAIL ") + k)
if not all(cases.values()):
    raise SystemExit(1)
