"""Synthetic tests of 09's route C in full_override and inputs_full (10-17): the S_old H_sim allowlist. No Modal call; the only git reads are the
base full_override (704122edf) for the regression cases and the approved counted-keys pin (344829af6) for the inputs_full checks.

- s_old_simh: without an allowlist the result is the registered list, unchanged, and the registered checks STOP as before. The ONE listed root
  (the pilot) is admitted, tagged S_old-only with its admitted keys. An unlisted root still STOPs, and so does a listed root with no uncut entry
  or that is not an inventory sim-honest root.
- The pin (Astra R1 / R2): absent -> None; present -> exactly the approved allowlist; null, another root, pin or a second entry STOP.
- The keys (Astra R3): the pinned file's 20 simh_pilot|<task> keys; 21, a duplicate, another world or stratum STOP. Row time: an entry's run keys
  must equal its admitted keys exactly (namespace included).
- B1 (a): only S_old variants carry the tagged entry (full_jobs, addendum_jobs); S_paper493's list is the registered one; the reader guard STOPs a
  tagged entry on any other origin.
- apply_full / full_jobs integration (Astra optional): with the key absent the output equals the base code's (704122edf); with the pilot listed,
  S_old's H_sim gains exactly the tagged pilot entry and every other block is unchanged.
- inputs_full (B3): n/a on inside_32k_equal is accepted ONLY for the allowlisted S_old sim-honest root; the allowlist checks.

  uv run python experiments/2026-10-15_fpr_sweep_507_prep/test_s_old_simh_extra.py
"""

import copy
import json
import pickle
import subprocess
import types
from pathlib import Path

import full_override as FO
import inputs_full as IFU

HERE = Path(__file__).resolve().parent
B1, B2, PL = "simh_q4f_v1_full_b1__sim_honest", "simh_q4f_v1_full_b2__sim_honest", next(iter(FO.S_OLD_EXTRA))
KEYS = [f"simh_pilot|t{i}_task" for i in range(21, 41)]
CUT = [{"batch": "b1", "root": B1}, {"batch": "b2", "root": B2}]
E = {r: {"batch": r.split("__")[0], "root": r, "path": f"cutfull/x/{r}.npz", "sha256": "0" * 64} for r in (B1, B2, PL, "simh_other__sim_honest")}
ALLOW = {PL: KEYS}


def stops(f, *a, **k):
    try:
        f(*a, **k)
    except SystemExit:
        return True
    return False


def run(entries, inv, extra=None):
    return FO.s_old_simh(CUT, [E[r] for r in entries], inv, extra, "Sol (a) C")


def run_zu(entries, inv, extra, zu):
    return FO.s_old_simh(CUT, [E[r] for r in entries], inv, extra, "Sol (a) C", zu)


def base_fo():
    """full_override at 704122edf, as a module (its imports resolve here)."""
    src = subprocess.run(["git", "show", "704122edf:./full_override.py"], capture_output=True, check=True, cwd=HERE).stdout.decode()
    m = types.ModuleType("fo_704122edf")
    exec(compile(src, "fo_704122edf", "exec"), m.__dict__)
    return m


def fixture(pilot):
    fn = "p2/x.npz"
    cfg = {"probes": [{"label": "Sol (a) C", "new": fn, "old": {"kind": "fam", "path": "old.npz", "sha256": "1" * 64}}], "relaunch_files": {}}
    cfg_old = dict(cfg, origin="S_old", simh_old={"Sol (a) C": [dict(E[B1], path="cut1.npz"), dict(E[B2], path="cut2.npz")]})
    simh = [dict(E[r], gate={"x": 1}) for r in (B1, B2) + ((PL,) if pilot else ())]
    ovs = {"S_old": {"arms": {"realF": {fn: {"path": "u_real.npz", "sha256": "2" * 64}}}, "simh": {fn: simh}}}
    roles = {"realF": "real", B1: "simh", B2: "simh"} | ({PL: "simh"} if pilot else {})
    return cfg, cfg_old, ovs, ["Sol (a) C"], roles


reg, add = run([B1, B2], [B1, B2])
reg_x, add_x = run([B1, B2, PL], [B1, B2, PL], ALLOW)
tagged = E[PL] | {FO.S_OLD_ONLY: True, "admitted": KEYS, "zero_unit_ok": []}
cases = {
    "no allowlist: the registered list, unchanged (byte path)": reg == [E[B1], E[B2]] and add == [],
    "no allowlist: an extra root in the override and inventory STOPs (the registered check, as before)": stops(run, [B1, B2, PL], [B1, B2, PL]),
    "no allowlist: an extra root only in the override STOPs": stops(run, [B1, B2, PL], [B1, B2]),
    "POSITIVE: the listed pilot root is admitted after the registered roots, tagged S_old-only with its keys": reg_x == [E[B1], E[B2]] and add_x == [tagged],
    "NEGATIVE: an UNLISTED extra root still STOPs while the pilot is allowlisted": stops(run, [B1, B2, PL, "simh_other__sim_honest"],
                                                                                     [B1, B2, PL, "simh_other__sim_honest"], ALLOW),
    "listed root with no uncut entry: STOPs": stops(run, [B1, B2], [B1, B2, PL], ALLOW),
    "listed root that is not an inventory sim-honest root: STOPs": stops(run, [B1, B2, PL], [B1, B2], ALLOW),
    "listed root with two uncut entries: STOPs": stops(run, [B1, B2, PL, PL], [B1, B2, PL], ALLOW),
    "a registered root missing while the pilot is listed: STOPs": stops(run, [B1, PL], [B1, B2, PL], ALLOW),
    "listed root without admitted keys: STOPs": stops(run, [B1, B2, PL], [B1, B2, PL], {PL: []}),
    "empty 32k list: STOPs (as before)": stops(FO.s_old_simh, [], [E[B1]], [B1], None, "x"),
    "(d) an override entry already carrying the S_old-only tag: STOPs": stops(FO.s_old_simh, CUT, [E[B1], E[B2] | {FO.S_OLD_ONLY: True}], [B1, B2], None, "x"),
    "(d) a 32k entry carrying the tag: STOPs": stops(FO.s_old_simh, CUT[:1] + [CUT[1] | {FO.S_OLD_ONLY: True}], [E[B1], E[B2]], [B1, B2], None, "x"),
}

# the pin (Astra R1 / R2)
cases |= {
    "pin absent: None": FO.s_old_extra_pin({}) is None,
    "pin present and approved: the allowlist": FO.s_old_extra_pin({"s_old_simh_extra_roots": copy.deepcopy(FO.S_OLD_EXTRA)}) == FO.S_OLD_EXTRA,
    "pin null: STOPs": stops(FO.s_old_extra_pin, {"s_old_simh_extra_roots": None}),
    "pin empty: STOPs": stops(FO.s_old_extra_pin, {"s_old_simh_extra_roots": {}}),
    "pin on another root: STOPs": stops(FO.s_old_extra_pin, {"s_old_simh_extra_roots": {B1: FO.S_OLD_EXTRA[PL]}}),
    "pin at another file with its own sha (Astra R1's example): STOPs": stops(FO.s_old_extra_pin, {"s_old_simh_extra_roots": {
        PL: {"commit": "ea0d2fa3c", "path": "experiments/2026-10-15_fpr_sweep_507_prep/full_override.py", "sha256": "91b0cad8" + "0" * 56}}}),
    "pin with a second entry: STOPs": stops(FO.s_old_extra_pin, {"s_old_simh_extra_roots": FO.S_OLD_EXTRA | {B1: FO.S_OLD_EXTRA[PL]}}),
}

# the keys (Astra R3)
MAN = {"n": 20, "keys": [{"world": "simh_pilot", "task": f"t{i}_task", "stratum": "S_old"} for i in range(21, 41)]}
real_ = json.loads(subprocess.run(["git", "show", f"{FO.S_OLD_EXTRA[PL]['commit']}:{FO.S_OLD_EXTRA[PL]['path']}"], capture_output=True, check=True,
                                  cwd=HERE).stdout)
cases |= {
    "keys: the synthetic 20, sorted, world|task": FO.s_old_extra_keys(MAN) == sorted(KEYS),
    "keys: the approved pinned file gives 20 simh_pilot keys": len(FO.s_old_extra_keys(real_)) == 20 and all(
        k.startswith("simh_pilot|") for k in FO.s_old_extra_keys(real_)),
    "keys: a 21st key STOPs": stops(FO.s_old_extra_keys, MAN | {"n": 21, "keys": MAN["keys"] + [{"world": "simh_pilot", "task": "t41_x", "stratum": "S_old"}]}),
    "keys: a duplicate STOPs": stops(FO.s_old_extra_keys, MAN | {"keys": MAN["keys"][:19] + [MAN["keys"][0]]}),
    "keys: another world STOPs": stops(FO.s_old_extra_keys, MAN | {"keys": MAN["keys"][:19] + [{"world": "simh_full", "task": "t40_task", "stratum": "S_old"}]}),
    "keys: another stratum STOPs": stops(FO.s_old_extra_keys, MAN | {"keys": MAN["keys"][:19] + [{"world": "simh_pilot", "task": "t40_task", "stratum": "S_new"}]}),
    "row: run keys == admitted: passes, counts recorded, no zero-unit run": FO.s_old_extra_check(tagged, KEYS * 3, "L")["n"] == 20
    and FO.s_old_extra_check(tagged, KEYS, "L")["zero_unit"] == [],
    "row: an extra run key STOPs (Astra R3's NOT_A_PILOT_TASK)": stops(FO.s_old_extra_check, tagged, KEYS + ["simh_pilot|NOT_A_PILOT_TASK"], "L"),
    "row: an admitted key without rows is a zero-unit member, listed, not a STOP (09, 10-17)": FO.s_old_extra_check(tagged, KEYS[1:], "L")["zero_unit"]
    == [KEYS[0]],
    "row (R3): a zero-unit key NOT on the pinned pre-flight list is unlisted (the row withholds)": FO.s_old_extra_check(tagged, KEYS[1:], "L")[
        "zero_unit_unlisted"] == [KEYS[0]],
    "row (R3): a zero-unit key on the pinned list is admitted": FO.s_old_extra_check(tagged | {"zero_unit_ok": [KEYS[0]]}, KEYS[1:], "L")[
        "zero_unit_unlisted"] == [],
    "s_old_simh (R3): each tagged entry carries its own file's pinned zero-unit list, by sha": run_zu([B1, B2, PL], [B1, B2, PL], ALLOW,
                                                                                                  {E[PL]["sha256"]: [KEYS[3]], "f" * 64: [KEYS[4]]})[1][0]["zero_unit_ok"] == [KEYS[3]],
    "row: another namespace STOPs": stops(FO.s_old_extra_check, tagged, [k.replace("simh_pilot|", "simh_full|") for k in KEYS], "L"),
    "row guard: a tagged entry on an S_paper493 row STOPs": stops(FO.s_old_only_guard, [E[B1], tagged], "S_paper493", "L"),
    "row guard: a tagged entry on an S_old row passes": not stops(FO.s_old_only_guard, [E[B1], tagged], "S_old", "L"),
}

# apply_full / full_jobs integration (Astra optional): the base code's output when absent; exactly one tagged entry more when listed
old = base_fo()
a_new = FO.apply_full(*fixture(False))
a_old = old.apply_full(*fixture(False))
b_new = FO.apply_full(*fixture(True), extra_roots={PL: KEYS})
c_a, co_a, ap_a, al_a = a_new
c_b, co_b, ap_b, al_b = b_new
pl_file = ("conv", E[PL]["path"], E[PL]["sha256"])
cases |= {
    "apply_full, key absent: equal to 704122edf's output": pickle.dumps(a_new) == pickle.dumps(a_old),
    "apply_full, pilot listed: S_old H_sim = the registered entries + the tagged pilot entry": co_b["simh_old"]["Sol (a) C"] == co_a["simh_old"]["Sol (a) C"]
    + [{k: tagged[k] for k in ("batch", "root", "path", "sha256", FO.S_OLD_ONLY, "admitted", "zero_unit_ok")}],
    "apply_full, pilot listed: every other block unchanged (cfg; cfg_old but H_sim; allowed = + the pilot file only)":
        {k: v for k, v in c_b.items() if k != "full_allowed"} == {k: v for k, v in c_a.items() if k != "full_allowed"}
        and {k: v for k, v in co_b.items() if k not in ("simh_old", "full_allowed")} == {k: v for k, v in co_a.items() if k not in ("simh_old", "full_allowed")}
        and set(map(tuple, al_b)) - set(map(tuple, al_a)) == {pl_file} and set(map(tuple, al_a)) <= set(map(tuple, al_b)),
    "apply_full, pilot listed: the applied entry names the extra root": [x.get("simh_extra_roots") for x in ap_b if "simh_extra_roots" in x] != [],
}
JOBS = [("Sol (a) C", {"origin": "S_old", "simh_old": {}}, None), ("Sol (a) C", {"origin": "S_paper493", "simh_old": {}}, None),
        ("Sol (a) C", {"origin": "S_paper493", "simh_old": {}, "cohort": "registered"}, None), ("Sol (a) C", {"origin": "S_new"}, None)]
j_old = old.full_jobs(JOBS, (c_a, co_a), {}, al_a)
j_new_a = FO.full_jobs(JOBS, (c_a, co_a), {}, al_a)
j_new_b = FO.full_jobs(JOBS, (c_b, co_b), {}, al_b)
nm_ = next(iter(FO.ADDENDUM))
JOBS_ADD = [(FO.ADDENDUM_OF[nm_] + " C", v, b) for _, v, b in JOBS]
cases |= {
    "full_jobs, key absent: equal to 704122edf's (shared S_old list object kept)": pickle.dumps(j_new_a) == pickle.dumps(j_old)
    and j_new_a[0][1]["simh_old"] is j_new_a[1][1]["simh_old"],
    "full_jobs, pilot listed: the S_old job carries the tagged entry": any(e.get(FO.S_OLD_ONLY) for e in j_new_b[0][1]["simh_old"]["Sol (a) C"]),
    "full_jobs, pilot listed: both S_paper493 jobs carry exactly the registered list (B1 (a))":
        all(j_new_b[i][1]["simh_old"] == co_a["simh_old"] for i in (1, 2)),
    "full_jobs, pilot listed: a job without simh_old gets none": "simh_old" not in j_new_b[3][1],
    "addendum_jobs, pilot listed: S_old tagged, S_paper493 registered": (lambda j: any(e.get(FO.S_OLD_ONLY) for e in j[0][1]["simh_old"]["Sol (a) C"])
                                                                         and j[1][1]["simh_old"] == co_a["simh_old"])(
        FO.addendum_jobs(JOBS_ADD, (c_b, co_b), al_b, nm_)),
    "addendum_jobs, key absent: equal to 704122edf's": pickle.dumps(FO.addendum_jobs(JOBS_ADD, (c_a, co_a), al_a, nm_))
    == pickle.dumps(old.addendum_jobs(JOBS_ADD, (c_a, co_a), al_a, nm_)),
}

# inputs_full (B3): n/a only for the allowlisted root; the allowlist checks
G = {"commit": "c0ffee0", "path": "gates.json", "sha256": "a" * 64}
OK = {"keys_bound_by_task": True, "runs_without_rows_ok": True, "n_tokens_eq_full_len": True}
CM = {"S_old": {"C": {"old": {"p2/x.npz": {}}, "simh": {f"{B1}|p2/x.npz": {}, f"{B2}|p2/x.npz": {}}}}}
ROLES = {B1: "simh", B2: "simh", PL: "simh"}


def ep(root, extra):
    e = {"path": "cutfull/x.npz", "sha256": "b" * 64, "staged_shas": ["s" * 64], "runs_without_rows": [], "batch": "x", "root": root,
         "gate": G | {"file": f"{root}|p2/x.npz"}}
    gate = {"files": {f"{root}|p2/x.npz": OK | {"sha256": "b" * 64, "inside_32k_equal": IFU.NA}}}
    return IFU.entry_problems("S_old", "C", {"arms": {}, "simh": {"p2/x.npz": [e]}}, G, gate, set(), None, cm=CM, roles_old=ROLES, extra_roots=extra)[0]


def git_get(e, what):
    b = subprocess.run(["git", "show", f"{e['commit']}:{e['path']}"], capture_output=True, check=True, cwd=HERE).stdout
    return b if IFU.IF.sha(b) == e["sha256"] else None


cases |= {
    "inputs_full: n/a on the allowlisted pilot root is accepted": ep(PL, FO.S_OLD_EXTRA) == [],
    "inputs_full: n/a on the pilot root WITHOUT the allowlist is a problem": ep(PL, {}) != [],
    "inputs_full: n/a on a registered root (b1) stays a problem with the allowlist": ep(B1, FO.S_OLD_EXTRA) != [],
    "inputs_full: key absent: no problem, no allowlist": IFU.extra_roots_problems({}, git_get) == ([], None),
    "inputs_full: the approved allowlist with the pilot as a sim-honest folder: no problem":
        IFU.extra_roots_problems({"s_old_simh_extra_roots": FO.S_OLD_EXTRA, "roles_old": ROLES}, git_get) == ([], FO.S_OLD_EXTRA),
    "inputs_full: null is a problem": IFU.extra_roots_problems({"s_old_simh_extra_roots": None, "roles_old": ROLES}, git_get)[0] != [],
    "inputs_full: the pilot roled REAL (world not sim_honest) is a problem":
        IFU.extra_roots_problems({"s_old_simh_extra_roots": FO.S_OLD_EXTRA, "roles_old": ROLES | {PL: "real"}}, git_get)[0] != [],
    "inputs_full: the pilot missing from roles_old is a problem":
        IFU.extra_roots_problems({"s_old_simh_extra_roots": FO.S_OLD_EXTRA, "roles_old": {B1: "simh", B2: "simh"}}, git_get)[0] != [],
}

ZREC = {"what": FO.ROUTEC_PREFLIGHT_WHAT, "status": "CLEAR", "spec": {"sha256": "a" * 64}, "zero_unit_by_file": {"f" * 64: KEYS[:1], "e" * 64: []}}
cases |= {
    "pinned record: a CLEAR pre-flight naming its spec: the list and the spec sha": FO.routec_zero_unit_record(ZREC, KEYS) == (ZREC["zero_unit_by_file"], "a" * 64),
    "pinned record: status DEFECT STOPs": stops(FO.routec_zero_unit_record, ZREC | {"status": "DEFECT"}, KEYS),
    "pinned record: another 'what' (any JSON with the field) STOPs": stops(FO.routec_zero_unit_record, ZREC | {"what": "something else"}, KEYS),
    "pinned record: no spec sha STOPs": stops(FO.routec_zero_unit_record, ZREC | {"spec": {}}, KEYS),
    "pinned record: a key outside the admitted 20 STOPs": stops(FO.routec_zero_unit_record, ZREC | {"zero_unit_by_file": {"f" * 64: ["simh_pilot|x"]}}, KEYS),
    "pinned record: a file key that is not a sha256 STOPs": stops(FO.routec_zero_unit_record, ZREC | {"zero_unit_by_file": {"pilot.npz": []}}, KEYS),
}


def row_order():
    """(a) 507-prep: in _row, the guard runs before the H_sim loop; inside it the B2 check precedes K.append; xk_ is subtracted in the
    S_paper493 cross-check; the post-filter record follows the primary H_sim pool (PS) and precedes the pooled export's return."""
    import ast

    tree = ast.parse((HERE / "run_peraction311.py").read_text())
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_row")
    src = lambda n: ast.unparse(n)  # noqa: E731
    body = fn.body
    i_guard = next(i for i, s in enumerate(body) if "FO.s_old_only_guard(ents_, orig, label)" in src(s))
    i_loop = next(i for i, s in enumerate(body) if isinstance(s, ast.For) and src(s.iter) == "ents_")
    loop = body[i_loop]
    i_b2 = next(i for i, s in enumerate(loop.body) if isinstance(s, ast.If) and "FO.s_old_extra_check" in src(s))
    i_app = next(i for i, s in enumerate(loop.body) if "K.append(k)" in src(s))
    xsub = any(isinstance(n, ast.Assign) and "H_sim_run_keys_for_sp493" in src(n.targets[0]) and "- xk_" in src(n.value) for n in ast.walk(fn))
    i_ps = next(i for i, s in enumerate(body) if isinstance(s, ast.Assign) and src(s.targets[0]) == "PS")
    i_post = next(i for i, s in enumerate(body) if isinstance(s, ast.If) and src(s.test) == "xk_" and "post_filter_keys" in src(s))
    i_exp = next(i for i, s in enumerate(body) if isinstance(s, ast.If) and src(s.test) == "cfg.get('pw_export')" and "np.savez" in src(s))
    xz2 = sum(1 for n in ast.walk(fn) if isinstance(n, ast.Assign) and src(n.targets[0]).strip("()") in ("hcells", "hac_, hsc_")
              and "for r in xz_" in src(n.value)) == 2  # Claude r1 R2: the zero-unit pilot cells counted as S_paper493's, at both lines
    wh_ = any(isinstance(s, ast.If) and "zero_unit_unlisted" in src(s.test) and isinstance(s.body[0], ast.Return) for s in ast.walk(loop))  # R3
    return i_guard < i_loop and i_b2 < i_app and xsub and i_ps < i_post < i_exp and xz2 and wh_


cases["(a) _row call order: guard before the loop, B2 before K.append, xk_ in the cross-check, post-filter record between PS and the export; "
      "R2 zero-unit cells at both lines; R3 withhold in the loop"] = row_order()
for name, ok in cases.items():
    print(("PASS " if ok else "FAIL ") + name)
assert all(cases.values()), "route C tests failed"
print(f"{len(cases)} checks pass")
