# synthetic checks of check_nm172_provenance.py's pure helpers (no Volume, no real data)
import copy, importlib.util, sys
spec = importlib.util.spec_from_file_location("prov", sys.argv[1]); P = importlib.util.module_from_spec(spec); spec.loader.exec_module(P)
R, S = {"rk"}, {"sk"}
def m(mount, meta, key, n_tok, n_rows=10, calls=8):
    return {"mount": mount, "root": meta.split("/")[0], "meta": meta, "key": key, "n_tokens": n_tok, "n_rows": n_rows, "calls_len": calls}
am = [m("/aU", "rU/meta/a/1", "rk", 556), m("/aU", "sU/meta/a/2", "sk", 556), m("/a32", "r3/meta/a/3", "rk", 556), m("/a32", "s3/meta/a/4", "sk", 515)]
def row(n_ids, h, calls=10):
    return {"other_fields": {"ids": {"len": n_ids, "sha256": h}, "calls": {"len": calls, "sha256": "c"}}}
own = {"rU/meta/a/1": row(556, "R"), "sU/meta/a/2": row(515, "S"), "r3/meta/a/3": row(556, "R"), "s3/meta/a/4": row(515, "S")}
ctl = {"/aU:rU": True, "/aU:sU": True, "/a32:r3": True, "/a32:s3": True}
def cp(close_real, close_sim, exact_u=True):
    return [{"a": "rU/meta/a/1", "b": "sU/meta/a/2", "exact": exact_u, "close": exact_u},
            {"a": "rU/meta/a/1", "b": "r3/meta/a/3", "exact": False, "close": close_real}, {"a": "sU/meta/a/2", "b": "r3/meta/a/3", "exact": False, "close": close_real},
            {"a": "rU/meta/a/1", "b": "s3/meta/a/4", "exact": False, "close": close_sim}, {"a": "s3/meta/a/4", "b": "sU/meta/a/2", "exact": False, "close": close_sim},
            {"a": "r3/meta/a/3", "b": "s3/meta/a/4", "exact": False, "close": False}]
sl, cl, src = P.source_rule(am, cp(True, False), R, S, own, ctl); u = src["uncut"]
assert sl == ["uncut"] and u["source"] == "real" and u["own_counts_match_own_staged_row"] == {"real": True, "sim": True}, u
assert u["staged_ids_equal_across_lines"]["real"]["equal"] is True and u["staged_ids_equal_across_lines"]["sim"]["equal"] is True
w = u["within_line_counts_match (evidence only)"]  # both shared metas count like the real staged row (556), the sim meta not like its own (515)
assert w["sk @ sU/meta/a/2"] == {"real": True, "sim": False} and w["rk @ rU/meta/a/1"] == {"real": True, "sim": False}, w
assert P.source_rule(am, cp(False, True), R, S, own, ctl)[2]["uncut"]["source"] == "sim"
assert P.source_rule(am, cp(True, True), R, S, own, ctl)[2]["uncut"]["source"] == "undetermined"
assert P.source_rule(am, cp(True, False), R, S, own, ctl | {"/a32:s3": False})[2]["uncut"]["source"] == "undetermined"  # a control fails
am2 = copy.deepcopy(am); am2[3]["n_tokens"] = 556  # a swap in the other line: the sim's 32k meta counts like the real row, not its own
assert P.source_rule(am2, cp(True, False), R, S, own, ctl)[2]["uncut"]["source"] == "undetermined"
own2 = dict(own, **{"r3/meta/a/3": row(556, "R32")})  # the real's 32k staged ids differ from the uncut ones
assert P.source_rule(am, cp(True, False), R, S, own2, ctl)[2]["uncut"]["staged_ids_equal_across_lines"]["real"] == {"equal": False, "note": "close not expected (different rendered inputs)"}
assert P.source_rule(am, cp(True, False, exact_u=False), R, S, own, ctl)[:3:2] == ([], {})
sl, cl, src = P.source_rule(am, [{"a": "rU/meta/a/1", "b": "s3/meta/a/4", "exact": True, "close": True}], R, S, own, ctl); assert sl == [] and cl == [("uncut", "32k")]
assert P.counts_match(am[0], None) is None and P.counts_match(am[0], {"other_fields": {}}) is False
ids, other = P.summ({"key": "L/a/nm172_shared_dotfiles/f.eval#e0", "task": "nm172_shared_dotfiles", "text": "free text here", "ids": [1, 2, 3],
                     "redrawn_because": "a reason with spaces", "n_rows": 5, "calls": [{"x": 1}]})
assert set(ids) == {"key", "task", "n_rows"} and set(other) == {"text", "ids", "redrawn_because", "calls"} and other["ids"]["len"] == 3
assert all(set(v) == {"type", "len", "sha256"} for v in other.values())
mm = {"app": "x", "counts": {"fibs": {"rows": 3}}, "gate_honest_b": {"c": {"max_abs_diff": 1}}, "dump": {"run": "r_extract_v1", "staged": [{"path": "s_v1/inputs/a.jsonl", "sha": "ab"}],
      "gate_tol": 1, "volume": "v", "s4_orphans": ["2026/attack/x.eval"]}, "files": {"fibs__command.npz": "cd"}, "note": "words with spaces"}
pv = {k: v for k, v in P.id_leaves(mm).items() if P.PROVP.search(k)}
assert pv == {"app": "x", "dump.run": "r_extract_v1", "dump.staged[0].path": "s_v1/inputs/a.jsonl", "dump.staged[0].sha": "ab", "dump.volume": "v",
              "files.fibs__command.npz": "cd"}, pv
for c, ok in [("nm172_shared_dotfiles", 1), ("nm172_shared_dotfiles_r2", 1), ("nm172_shared_dotfiles_e2a", 1), ("nm172_shared_dotfiles_redraw2", 1),
              ("nm1720_x", 0), ("nm172_shared_dotfiles_x", 0), ("nm173_shared_dotfiles", 0)]:
    assert bool(P.CELLRX.match(c)) == bool(ok), c
assert P.EP2RX.search("/conv/cutfull/ep2-nm/p2/x") and P.EP2RX.search("/conv/cutfull_ep2/x") and not P.EP2RX.search("/conv/cutfull/nm360-s1_m1/p2/x")
assert P.TAGS.sub("", "nm172_shared_dotfiles_e2a") == "nm172_shared_dotfiles"
import numpy as np
assert P.unit_bytes([np.array([0.0, np.nan])]) == P.unit_bytes([np.array([-0.0, -np.nan])]) and P.unit_bytes([np.array([1.0])]) != P.unit_bytes([np.array([1.0 + 1e-15])])
print("synthetic checks: all pass")
# CUT R2 (a): an unreadable pair touching the involved metas, or any unreadable nm172 meta, leaves the source undetermined
cpe = cp(True, False) + [{"a": "sU/meta/a/2", "b": "s3/meta/a/4", "error": "FileNotFoundError"}]
assert P.source_rule(am, cpe, R, S, own, ctl)[2]["uncut"]["source"] == "undetermined"
assert P.source_rule(am, cp(True, False), R, S, own, ctl, 1)[2]["uncut"]["source"] == "undetermined"
cpo = cp(True, False) + [{"a": "x/meta/a/9", "b": "y/meta/a/8", "error": "OSError"}]  # an error on metas not involved does not touch the rule
am_o = am + [m("/aU", "x/meta/a/9", "zk", 1), m("/aU", "y/meta/a/8", "zk2", 1)]
assert P.source_rule(am_o, cpo, R, S, own, ctl)[2]["uncut"]["source"] == "real"
# (b): unreadable inputs come first in the reading
src_real = P.source_rule(am, cp(True, False), R, S, own, ctl)[2]
assert P.abc_of(src_real, ["f"], 0, 0, 0).startswith("A: the shared rows are the real run")
assert P.abc_of(src_real, ["f"], 1, 0, 0).startswith("undetermined (unreadable inputs: 1 score files")
assert P.abc_of({}, [], 0, 0, 2).startswith("undetermined") and P.abc_of({}, [], 0, 0, 0).startswith("no shared units")
assert P.abc_of({"uncut": {"source": "undetermined"}}, ["f"], 0, 0, 0).startswith("B:")
# (c): labels a lost hit or an unreadable nm172 meta may name are not clean
labs = ["honest_a_e2|nm172_shared_dotfiles", "honest_a|nm172_shared_dotfiles", "simh_nm360|nm172_shared_dotfiles", "run1|nm172_shared_dotfiles"]
lk = {"simh_nm360|nm172_shared_dotfiles": {"L/sim_honest@september/nm172_shared_dotfiles/f.eval#e0"}}
assert P.unclean_labels(labs, lk, [{"lost_hit_keys": ["honest_a|nm172_shared_dotfiles"]}], []) == {"honest_a_e2|nm172_shared_dotfiles", "honest_a|nm172_shared_dotfiles"}
assert P.unclean_labels(labs, lk, [], [{"names_nm172": True, "keys_in_bytes": ["L/sim_honest@september/nm172_shared_dotfiles/f.eval#e0"]}]) == {"simh_nm360|nm172_shared_dotfiles"}
assert P.unclean_labels(labs, lk, [], [{"names_nm172": True, "keys_in_bytes": []}]) == set(labs)
assert P.unclean_labels(labs, lk, [], [{"names_nm172": False, "keys_in_bytes": []}]) == set()
assert [x.decode() for x in P.EVKEY.findall(b'{"key": "2026-10-15_simh_nm_s1/sim_honest@september/nm172_shared_dotfiles/2026-10-15T19-11-26-00-00_la-x_kq.eval#e0", bad')] == \
    ["2026-10-15_simh_nm_s1/sim_honest@september/nm172_shared_dotfiles/2026-10-15T19-11-26-00-00_la-x_kq.eval#e0"]
print("CUT R2 synthetic checks: all pass")
