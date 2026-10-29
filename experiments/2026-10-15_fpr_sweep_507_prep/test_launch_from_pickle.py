"""Tests of --launch-from-pickle (2e option A, 10-17): no Modal, no upload, no real spec, nothing unpickled.

- FO.launch_pickle_checks STOPs before any upload on a pickle sha that is not the named one or not the passing verdict's side b, a failing
  verdict, a dirty tracked tree, side b not built at HEAD, or no .json record; it names the Volume spec path as dispatch() does, from the
  verdict's side-b metadata (R2: no pickle is opened on the laptop).
- FO.missing_inputs lists the shipped inputs the Volume lacks (R1: a missing blob STOPs before the spec upload and the spawn).
- By syntax tree: launch_from() recomputes the gate, then runs the checks, then the inputs listing, then the upload, then the spawn; neither
  launch_from nor launch_pickle_checks calls pickle.loads; and the runner minus the new sites (two parameters, the first main() branch,
  launch_from) equals the runner at BASE, statement for statement.

  uv run python experiments/2026-10-15_fpr_sweep_507_prep/test_launch_from_pickle.py [--base cad177099]
"""

import argparse
import ast
import subprocess
from pathlib import Path

import full_override as FO

HERE = Path(__file__).resolve().parent
RP = "run_peraction311.py"
PARAMS = ("launch_from_pickle", "launch_diff")


def stops(f, *a):
    try:
        f(*a)
    except SystemExit:
        return True
    return False


def strip_launch(tree):
    """Remove the launch-from-pickle sites, asserting each one's shape; -> the number removed."""
    mod = tree.body
    lf = [s for s in mod if isinstance(s, ast.FunctionDef) and s.name == "launch_from"]
    assert len(lf) == 1
    mod.remove(lf[0])
    m = next(s for s in mod if isinstance(s, ast.FunctionDef) and s.name == "main")
    for nm in PARAMS:
        names = [a.arg for a in m.args.args]
        i, off = names.index(nm), len(names) - len(m.args.defaults)
        assert ast.unparse(m.args.defaults[i - off]) == "''" and ast.unparse(m.args.args[i].annotation) == "str"
        del m.args.args[i], m.args.defaults[i - off]
    br = m.body[0]
    assert isinstance(br, ast.If) and ast.unparse(br.test) == "launch_from_pickle or launch_diff" and len(br.body) == 1 and not br.orelse
    assert ast.unparse(br.body[0]) == "return launch_from(launch_from_pickle, launch_diff, dict(locals()))"  # first in main(): locals = the flags
    m.body.remove(br)
    left = [ast.unparse(x) for x in ast.walk(tree) if isinstance(x, ast.Name) and x.id in PARAMS + ("launch_from",)]
    assert not left, left[:3]
    return 4


OP1N = ".spawn(spec_path, lh_)"  # O-P1 (10-17): the one intended launch-site change since the launch commit; mapped back for the equality
OP1 = ".spawn(spec_path)"
KO_TEST = "(d_.get('b_meta') or {}).get('keys_only') is True"  # the census branch (Aswin A1, 10-17): the second intended change
KO_BODY = ["pin_, verdict_ = (None, GSD.gate_keys_only(d_))"]


def unfork_keys_only(src):
    """launch_from's keys-only If (its body exactly KO_BODY) replaced by its else branch, the scored path; then O-P1 mapped back. -> source
    whose launch sites must equal the launch commit's in full (so the scored path is that commit's, statement for statement)."""
    t = ast.parse(src)
    lf = next(s for s in t.body if isinstance(s, ast.FunctionDef) and s.name == "launch_from")
    i = [j for j, s in enumerate(lf.body) if isinstance(s, ast.If) and ast.unparse(s.test) == KO_TEST]
    assert len(i) == 1 and [ast.unparse(x) for x in lf.body[i[0]].body] == KO_BODY, "the keys-only branch"
    lf.body[i[0]:i[0] + 1] = lf.body[i[0]].orelse
    return OP1.join(ast.unparse(t).split(OP1N, 1))


def launch_sites(src):
    """The launch sites' full syntax trees (launch_from, main's first branch, the two parameters)."""
    t = ast.parse(src)
    m = next(s for s in t.body if isinstance(s, ast.FunctionDef) and s.name == "main")
    off = len(m.args.args) - len(m.args.defaults)  # Astra launch r2: a parameter's DEFAULT is part of the site (ast.arg holds none)
    return [ast.dump(s) for s in t.body if isinstance(s, ast.FunctionDef) and s.name == "launch_from"] + [ast.dump(m.body[0])] + [
        ast.dump(a) + " = " + ast.dump(m.args.defaults[i - off]) for i, a in enumerate(m.args.args) if a.arg in PARAMS]


def calls_in(fn):
    return {ast.unparse(c.func) for c in ast.walk(fn) if isinstance(c, ast.Call)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="cad177099")
    ap.add_argument("--launch-commit", default="adfd8e76f")  # where the launch sites took their reviewed form; later runner edits elsewhere are fine
    a = ap.parse_args()
    sh = "5" * 64
    V = {"pass": True, "b_sha256": sh, "b_meta": {"git": "h" * 40, "dest": "xfer/fpr507prep/records/results/alltasks_X_2026-10-17_020000.json", "pooled": True}}
    src = (HERE / RP).read_text()
    tree = ast.parse(src)
    lf = next(s for s in tree.body if isinstance(s, ast.FunctionDef) and s.name == "launch_from")
    body_ = lf.body[1:]  # after the docstring

    def first(pred):
        return min([i for i, s in enumerate(body_) if pred(s)], default=None)

    order = [first(lambda s: "GSD.gate(" in ast.unparse(s)), first(lambda s: "FO.launch_pickle_checks(" in ast.unparse(s)),
             first(lambda s: "vol_out.listdir" in ast.unparse(s)), first(lambda s: "FO.missing_inputs(" in ast.unparse(s)),
             first(lambda s: isinstance(s, ast.If) and ast.unparse(s.test) == "miss_" and isinstance(s.body[0], ast.Raise)),  # the STOP itself (Astra opt.)
             first(lambda s: isinstance(s, ast.With) and "vol_out.batch_upload" in ast.unparse(s)), first(lambda s: ".spawn(" in ast.unparse(s))]
    fo_tree = ast.parse((HERE / "full_override.py").read_text())
    lpc = next(s for s in fo_tree.body if isinstance(s, ast.FunctionDef) and s.name == "launch_pickle_checks")
    show = lambda c: subprocess.run(["git", "show", f"{c}:./{RP}"], capture_output=True, check=True, cwd=HERE).stdout.decode()  # noqa: E731
    lc_src = show(a.launch_commit)
    new = ast.parse(lc_src)  # the no-flag equality at the commit that added the sites (a later edit elsewhere in the runner cannot break it)
    n_sites = strip_launch(new)
    old = ast.parse(show(a.base))
    cases = {
        "passes: the drivers/ path as dispatch() names it, and the pooled flag, from the verdict's side b": FO.launch_pickle_checks(sh, sh, V, "h" * 40, "")
        == ("xfer/fpr507prep/drivers/alltasks_X_2026-10-17_020000.pkl", True),
        "a cutfull record keeps its cutfull/ prefix": FO.launch_pickle_checks(sh, sh, V | {"b_meta": V["b_meta"] | {"dest": "cutfull/xfer/fpr507prep/records/r/g1.json"}},
                                                                              "h" * 40, "")[0] == "cutfull/xfer/fpr507prep/drivers/g1.pkl",
        "STOP: the pickle's sha is not the named one": stops(FO.launch_pickle_checks, sh, "0" * 64, V, "h" * 40, ""),
        "STOP: the pickle is not the verdict's side b": stops(FO.launch_pickle_checks, sh, sh, V | {"b_sha256": "0" * 64}, "h" * 40, ""),
        "STOP: a failing verdict": stops(FO.launch_pickle_checks, sh, sh, V | {"pass": False}, "h" * 40, ""),
        "STOP: no verdict": stops(FO.launch_pickle_checks, sh, sh, None, "h" * 40, ""),
        "STOP: a dirty tracked tree": stops(FO.launch_pickle_checks, sh, sh, V, "h" * 40, " M experiments/x.py\n"),
        "STOP: side b not built at HEAD": stops(FO.launch_pickle_checks, sh, sh, V, "g" * 40, ""),
        "STOP: no .json record": stops(FO.launch_pickle_checks, sh, sh, V | {"b_meta": V["b_meta"] | {"dest": "x"}}, "h" * 40, ""),
        "R1: a shipped input missing from the Volume is listed (the launcher STOPs before the upload)": FO.missing_inputs(
            {"t196": {"path": f"xfer/fpr507prep/inputs/by_sha/{'a' * 64}", "sha256": "a" * 64}, "v6": {"path": f"xfer/fpr507prep/inputs/by_sha/{'b' * 64}", "sha256": "b" * 64}},
            {"a" * 64}) == ["v6"],
        "R1: a shipped input at another path is listed too": FO.missing_inputs({"t196": {"path": "elsewhere", "sha256": "a" * 64}}, {"a" * 64}) == ["t196"],
        "R1: all present: none": FO.missing_inputs({"t196": {"path": f"xfer/fpr507prep/inputs/by_sha/{'a' * 64}", "sha256": "a" * 64}}, {"a" * 64}) == [],
        "flags_set: only non-default flags other than the launch pair": FO.flags_set(
            {"readouts": "C", "dryrun": False, "remote_driver": True, "launch_from_pickle": "p@x", "launch_diff": "g@y"},
            {"readouts": "C", "dryrun": False, "remote_driver": False, "launch_from_pickle": "", "launch_diff": ""}, PARAMS) == ["remote_driver"],
        "launch_from order: gate recomputed < checks < inputs listing < missing-input list < the STOP on it < upload < spawn": None not in order and order == sorted(order)
        and len(set(order)) == len(order),
        "R2: no pickle.loads in launch_from or FO.launch_pickle_checks (nothing unpickled on the laptop)": not ({"pickle.loads", "loads", "pickle.load", "load"}
                                                                                                         & (calls_in(lf) | calls_in(lpc)))
        and not [x for f_ in (lf, lpc) for x in ast.walk(f_) if isinstance(x, (ast.Import, ast.ImportFrom)) and "pickle" in ast.unparse(x)],
        "the gate is recomputed from the current code, never read from a stored verdict": "GSD.gate(d_, json.loads(pb_), pin_)" in ast.unparse(lf),
        f"no-flag path: {a.launch_commit}'s runner minus the {n_sites} launch sites equals {a.base}'s, statement for statement": ast.dump(new) == ast.dump(old),
        f"the working tree's launch sites equal {a.launch_commit}'s in full, except O-P1's spawn sha and the census keys-only branch (10-17)": (
            launch_sites(unfork_keys_only(src)) == launch_sites(lc_src) and src.count(OP1N) == 1),
        "positive control: a change to the scored path inside the keys-only If's else branch is caught": launch_sites(unfork_keys_only(src.replace(
            '        if C.sha_b(pb_) != pin_["sha256"]:', '        if False:', 1))) != launch_sites(lc_src),
        "O-P1 (10-17): launch_from's spawn passes lh_, the pinned sha launch_pickle_checks matched to the blob, as drive()'s 2nd argument": (
            src.count(OP1N) == 1 and launch_sites(src) != launch_sites(lc_src)),
        "positive control: a non-empty launch_from_pickle DEFAULT in the working tree is caught (Astra launch r2)": launch_sites(unfork_keys_only(src.replace(
            'launch_from_pickle: str = ""', 'launch_from_pickle: str = "p@x"', 1))) != launch_sites(lc_src),
        "positive control: an upload added before the checks in the working tree's launch_from is caught": launch_sites(unfork_keys_only(src.replace(
            "    git_ = lambda *a: subprocess.run", "    vol_out.listdir('/x')\n    git_ = lambda *a: subprocess.run", 1))) != launch_sites(lc_src),
        "spec_only stays main()'s last parameter": [x.arg for x in next(s for s in tree.body if isinstance(s, ast.FunctionDef) and s.name == "main").args.args][-1]
        == "spec_only",
    }
    for name, ok_ in cases.items():
        print(("PASS " if ok_ else "FAIL ") + name)
    assert all(cases.values()), "launch-from-pickle tests failed"
    print(f"{len(cases)} checks pass")


if __name__ == "__main__":
    main()
