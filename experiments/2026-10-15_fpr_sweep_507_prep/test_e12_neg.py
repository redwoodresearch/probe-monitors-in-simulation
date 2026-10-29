"""Lean negative tests for the e12 key joins (CUT 03:4x: redraw tasks): the cell by key, never the raw task string.  python test_e12_neg.py"""
import sys
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.modules.setdefault("modal", mock.MagicMock())
import make_e12_block as B  # noqa: E402
import run_peraction311 as R  # noqa: E402

assert B.task_cell("t347_media_processing_redraw1") == "t347_media_processing" == R.e12_cell("t347_media_processing_redraw1")
assert B.task_cell("t01_media_processing_e2b") == "t01_media_processing" == B.task_cell("nm091_file_indexing_redraw2").replace("nm091_file_indexing", "t01_media_processing")
assert B.task_cell("t01_x_e2b_redraw1") == "t01_x" == R.e12_cell("t01_x_redraw2_e2a")  # combined tags, either order (Codex r3)
assert B.task_cell("t01_x_redrawA") == "t01_x_redrawA"  # not a redraw suffix: kept, so a join on it fails closed
assert R.e12_rename(["honest_b|t321_tiktok_redraw1", "honest_b|t322_x"], "honest_b", "honest_b_e2") == ["honest_b_e2|t321_tiktok", "honest_b_e2|t322_x"]
for bad in (["honest_b|t321_tiktok", "honest_b|t321_tiktok_redraw1"], ["honest_b|t01_x", "honest_b|t01_x_e2b_redraw1"], ["honest_a|t321_tiktok"]):  # two raws on one cell; another world
    try:
        R.e12_rename(bad, "honest_b", "honest_b_e2")
        raise AssertionError(f"no STOP for {bad}")
    except SystemExit:
        pass
ids = {"honest_b_e2|t01_x": ("L/honest_b/t01_x_e2b/a.eval#e0", "s2")}
assert R.e12_identity(ids, {"L/honest_b/t01_x_e2b/a.eval#e0"}, {"s1"}) == 1
for inv, e1 in (({"other#e0"}, {"s1"}), ({"L/honest_b/t01_x_e2b/a.eval#e0"}, {"s2"})):  # not an epoch-2 inventory run; an epoch-1 eval sha
    try:
        R.e12_identity(ids, inv, e1)
        raise AssertionError(f"no STOP for {inv} {e1}")
    except SystemExit:
        pass
ks = {"H_B": "a", "H_B2": None, "H_A": "x"}
assert R.e12_pool_key_check({"FIBS C": {"S_new": {"calibration_e12_key_sets": ks}}, "own_all_LOO_L48 C": {"S_new": {"calibration_e12_key_sets": ks}},
                             "own_all_rs_L48 C": {"S_new": {"calibration_e12_key_sets": ks}}})["H_B_H_B2_equal_across_probes"]
for bad in ({"H_B": "b", "H_B2": None, "H_A": "x"}, {"H_B": "a", "H_B2": None, "H_A": "y"}):  # LOO's H_B differs; LOO's H_A differs from own_all_rs's
    try:
        R.e12_pool_key_check({"own_all_rs_L48 C": {"S_new": {"calibration_e12_key_sets": ks}}, "own_all_LOO_L48 C": {"S_new": {"calibration_e12_key_sets": bad}}})
        raise AssertionError(f"no STOP for {bad}")
    except SystemExit:
        pass
blk = {"coverage": {"headline_families": ["sol"]}, "caveat": "c", "fill_in": {}, "folders": {},
       "identity_sets": {"inventory_ep2_keys": ["a#e0"], "epoch1_eval_sha256": []},
       "pools": {o: {"H_B": {"e2": [{"key_e2": "honest_b_e2|t1", "replicates": "honest_b|t1", "inventory_key": "a#e0", "eval_sha256": "h"}]}} for o in ("S_old", "S_newmains")}
       | {"S_new": {"H_B": {"e2": []}, "H_A": {"g1": {"e2": [{"key_e2": "honest_a_e2|t2", "replicates": "honest_a|t2", "inventory_key": "b#e0", "eval_sha256": "h2"}]}},
                    "H_A_group_of_family": {"sol": "g1", "_rule": "x"}}}}
assert set(R.e12_cfg(blk)["ids"]) == {"honest_b_e2|t1", "honest_a_e2|t2"}  # nested S_new H_A groups are covered
blk["pools"]["S_newmains"]["H_B"]["e2"][0]["eval_sha256"] = "other"  # one key, two physical identities
try:
    R.e12_cfg(blk)
    raise AssertionError("no STOP for two identities")
except SystemExit:
    pass
print("test_e12_neg: 15 cases pass")
