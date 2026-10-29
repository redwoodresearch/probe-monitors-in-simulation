"""Synthetic checks of full_allowed_check's clauses 1-4 (09, 10-18), one positive control per clause. No Modal call, no real spec, override
or record; full_override is imported from the 507-prep folder (its own _walk_entries / combine)."""

import ast
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import full_allowed_check as K  # noqa: E402
import full_override as FO  # noqa: E402

P = K.PILOT


def stops(fn, *a):
    try:
        fn(*a)
    except SystemExit as e:
        return str(e).startswith("STOP")
    return False


def calls_helper():
    tree = ast.parse((HERE / "full_allowed_check.py").read_text())
    fns = {n.name: n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    called = lambda f: any(isinstance(c, ast.Call) and getattr(c.func, "id", None) == "stop_if_inputs_differ" for c in ast.walk(fns[f]))  # noqa: E731
    return called("main") and called("check_remote")


def main():
    old_e = {"root": "r1", "batch": "b1", "path": "s/old.npz", "sha256": "1" * 64}
    pil_e = {"root": P, "batch": "p", "path": "s/pilot.npz", "sha256": "2" * 64}
    arm = {"path": "a/real.npz", "sha256": "3" * 64}
    ov_a = FO.combine([{"arms": {"fold": {"f.npz": arm}}, "simh": {"f.npz": [old_e]}}])
    ov_b = FO.combine([{"arms": {"fold": {"f.npz": arm}}, "simh": {"f.npz": [old_e, pil_e]}}])
    roles = {"fold": "real", "r1": "simh"}
    ta, tb = K.s_old_entries(FO, ov_a, roles), K.s_old_entries(FO, ov_b, roles | {P: "simh"})
    base = sorted(ta)
    pilot = {K.triple(pil_e)}
    job = lambda fa, extra=None: ("L C", {"origin": "S_old", "full_as": "L|uncut C", "full_allowed": fa,  # noqa: E731
                                          "probes": [{"label": "L C", "new": "f.npz"}],
                                          "simh_old": {"L C": [dict(old_e)] + ([dict(pil_e, s_old_only=True)] if extra is None else extra)}}, {})
    a = {"jobs": [job(base, [])]}
    b = {"jobs": [job(sorted(tb))]}
    b_bad1 = {"jobs": [job(sorted(tb) + [("conv", "x.npz", "9" * 64)])]}
    b_bad2 = {"jobs": [job(sorted(tb), [dict(old_e, s_old_only=True)])]}
    frozen = {"_path": "config/F.json", "overrides": {"S_old|C": {"commit": "c", "path": "o", "sha256": "s"}}}
    meta = {"full_length": {"inputs_full": {"path": "config/F.json", "sha256": "f" * 64}, "overrides": frozen["overrides"]}}
    for n, (base_, commit, sha) in K.EXPECT3.items():
        meta[n] = {"path": f"results/{base_}", "sha256": sha} | ({"commit": commit + "0" * 31} if commit else {})
    b3 = {"meta": meta}
    b3_bad = {"meta": meta | {"waitlist": dict(meta["waitlist"], sha256="0" * 64)}}
    pin = lambda c: {"commit": c * 9, "path": f"o_{c}.json", "sha256": c * 64}  # noqa: E731
    oa = {"S_new|C": pin("1"), "S_old|C": pin("2"), "S_old|R": pin("3")}
    ob = oa | {"S_old|C": pin("4"), "S_old|R": pin("5")}
    ra = {"f1": "real", "f2": "simh"}
    rb = ra | {P: "simh"}
    fa, fb = {"overrides": oa, "roles_old": ra, "status": "x"}, {"overrides": ob, "roles_old": rb, "status": "y"}
    d = {"only_a": [], "only_b": ["jobs/0/1/full_allowed/7", "jobs/0/1/simh_old/L C/2", K.M + "copies.py"],
         "changed": ["jobs/0/1/full_allowed/3/1", "meta/waitlist/sha256", "meta/full_length/allowed_files"]}
    cases = {
        "s_old_entries: the pilot entry is the whole expected delta": tb - ta == pilot and not ta - tb,
        "clause 1: the observed delta equals the expected one": K.clause1(a, b, tb - ta, ta - tb, pilot)["pass"],
        "clause 1, positive control: one extra allowed triple fails it": not K.clause1(a, b_bad1, tb - ta, ta - tb, pilot)["pass"],
        "clause 2: the tagged entry is the label's pilot-root entry": K.clause2(b, ov_b)["pass"],
        "clause 2, positive control: a tagged non-pilot entry fails it": not K.clause2(b_bad2, ov_b)["pass"],
        "clause 3: the pins equal FROZEN and the three named files": K.clause3(b3, frozen, "f" * 64)["pass"],
        "clause 3, positive control: another waitlist sha fails it": not K.clause3(b3_bad, frozen, "f" * 64)["pass"],
        "clause 4: every path is in a clause or the allowed list": K.clause4(d)["pass"] and K.clause4(d)["by_clause"] == {"1": 2, "2": 1, "3": 1, "4": 2},
        "clause 4, positive control: an unlisted meta path and an only_a path fail it":
            K.clause4(d | {"only_a": ["meta/x"], "changed": d["changed"] + ["meta/new_thing"]})["n_other"] == 2,
        "inputs: only S_old|C, S_old|R and the pilot folder in roles_old differ: no difference reported": K.inputs_delta(fa, fb) == [],
        "inputs, positive control (overrides): another set's pin differs": K.inputs_delta(fa, fb | {"overrides": ob | {"S_new|C": pin("9")}}) == ["overrides S_new|C"],
        "inputs, positive control (overrides): a key on one side only": len(K.inputs_delta(fa, fb | {"overrides": ob | {"S_old|D": pin("8")}})) == 1,
        "inputs, positive control (roles_old): another folder's role differs": K.inputs_delta(fa, fb | {"roles_old": rb | {"f1": "relaunch"}}) == ["roles_old f1"],
        "inputs, positive control (roles_old): the pilot folder not simh": len(K.inputs_delta(fa, fb | {"roles_old": rb | {P: "real"}})) == 1,
        "inputs, positive control (another roles_* block): it differs": K.inputs_delta(fa | {"roles_new": {"x": "real"}}, fb | {"roles_new": {"x": "simh"}}) == ["roles_new"],
        "inputs, positive control (Ma): a third S_old| override key, present on both sides and different, STOPs":
            K.inputs_delta(fa | {"overrides": oa | {"S_old|X": pin("6")}}, fb | {"overrides": ob | {"S_old|X": pin("7")}}) == ["overrides S_old|X"]
            and stops(K.stop_if_inputs_differ, fa | {"overrides": oa | {"S_old|X": pin("6")}}, fb | {"overrides": ob | {"S_old|X": pin("7")}}),
        "inputs, positive control (Mc): the shared helper RAISES a STOP on a difference and returns [] on none":
            stops(K.stop_if_inputs_differ, fa, fb | {"roles_old": rb | {"f1": "relaunch"}}) and K.stop_if_inputs_differ(fa, fb) == [],
        "inputs (Mc): both the laptop path (main) and the container path (check_remote) call the helper": calls_helper(),
    }
    for name, ok in cases.items():
        print(("PASS " if ok else "FAIL ") + name)
    assert all(cases.values()), "full_allowed_check tests failed"
    print(f"{len(cases)} checks pass")


if __name__ == "__main__":
    main()
