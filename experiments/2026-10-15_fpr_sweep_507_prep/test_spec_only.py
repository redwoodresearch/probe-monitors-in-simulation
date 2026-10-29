"""Tests of --spec-only and spec_diff (09's controls via 507-prep, 10-17). No Modal call, no real spec.

SAME BUILDER, BY CODE LINE (and byte-identical behaviour with no flag): run_peraction311.py's syntax tree with every spec-only site removed
(each one guarded by a false spec_only / SPEC_ONLY["path"] when the flag is absent) equals the tree at BASE, statement for statement. Each site is
asserted to be one of the five allowed shapes; the dispatch() branch sits after the spec bytes are built and right before the upload and spawn.
spec_diff on synthetic specs: NULL (equal specs, and specs differing only in ignored fields or the run stamp, give no difference), POSITIVE
(plant() gives exactly the planted path, and only in one of two jobs sharing a cfg), and single changes give exactly one path.

  uv run python experiments/2026-10-15_fpr_sweep_507_prep/test_spec_only.py [--base 704122edf | none]
  (--base none on a branch that merged this mode on top of other runner changes: shapes only; the equality was shown at 704122edf.
   --base <pre-merge head> --src <merge commit>: the merge's runner minus the sites equals the pre-merge runner, and no spec-only line
   changed from the merge to the working tree; e.g. route C: --base 762772c71 --src 94b6ec670)
"""

import argparse
import ast
import copy
import subprocess
from pathlib import Path

import numpy as np

import spec_diff as SD

HERE = Path(__file__).resolve().parent
RP = "run_peraction311.py"
G0, G1, SH = "704122edf950d994ad5cef709dcaac69c24c1b1b", "b1e92a06d" + "0" * 31, "704122edf" + "e" * 55  # HEADs; a sha256 starting like G0
RPM, FOM = (f"experiments/2026-10-15_fpr_sweep_507_prep/{f}" for f in (RP, "full_override.py"))


def is_name(n, s):
    return isinstance(n, ast.Name) and n.id == s


def is_spec_path(n):  # SPEC_ONLY["path"]
    return isinstance(n, ast.Subscript) and is_name(n.value, "SPEC_ONLY") and isinstance(n.slice, ast.Constant) and n.slice.value == "path"


def site_dumps(src):
    """Every spec-only site's full syntax tree (branch bodies included), in order, from a source text."""
    tree = ast.parse(src)
    dumps = [ast.dump(s) for s in tree.body if isinstance(s, ast.Assign) and any(is_name(x, "SPEC_ONLY") for x in s.targets)]
    for n in ast.walk(tree):
        if isinstance(n, ast.If) and (is_spec_path(n.test) or is_name(n.test, "spec_only") or (
                isinstance(n.test, ast.BoolOp) and n.test.values and is_name(n.test.values[0], "spec_only"))):
            dumps.append(ast.dump(n))
        elif isinstance(n, ast.Assign) and any(is_spec_path(x) for x in n.targets):
            dumps.append(ast.dump(n))
        elif isinstance(n, ast.If) and isinstance(n.test, ast.UnaryOp) and isinstance(n.test.op, ast.Not) and is_name(n.test.operand, "spec_only"):
            dumps.append(ast.dump(n))
        elif isinstance(n, ast.IfExp) and is_name(n.test, "spec_only"):
            dumps.append(ast.dump(n))
        elif isinstance(n, ast.arg) and n.arg == "spec_only":
            dumps.append(ast.dump(n))
    return dumps


def strip(tree, sites):
    """Remove the spec-only sites from the tree (asserting each one's shape); -> the tree. sites collects what was removed."""
    mod = tree.body
    for s in [s for s in mod if isinstance(s, (ast.Assign, ast.AnnAssign, ast.AugAssign)) and any(
            is_name(t, "SPEC_ONLY") for t in (s.targets if isinstance(s, ast.Assign) else [s.target]))]:
        assert isinstance(s, ast.Assign) and len(s.targets) == 1 and ast.unparse(s.value) == "{'path': ''}", \
            "module: SPEC_ONLY = {'path': ''} exactly (one target, the inert initializer; Astra r1 R3)"
        mod.remove(s), sites.append("module SPEC_ONLY")
    fn = {s.name: s for s in mod if isinstance(s, ast.FunctionDef)}
    d = fn["dispatch"].body
    ifs = [i for i, s in enumerate(d) if isinstance(s, ast.If) and is_spec_path(s.test)]
    assert len(ifs) == 1, "dispatch: one spec-only branch"
    i = ifs[0]
    br = d[i]
    up = d[i + 1]
    assert isinstance(up, ast.With) and "vol_out.batch_upload" in ast.unparse(up.items[0].context_expr), "dispatch: the branch is right before the upload"
    assert any(isinstance(s, ast.Assign) and is_name(s.targets[0], "blob") and "pickle.dumps" in ast.unparse(s.value) for s in d[:i]), \
        "dispatch: the branch is after the spec bytes are built"
    assert isinstance(br.body[-1], ast.Return) and not br.orelse, "dispatch: the branch returns"
    calls = {ast.unparse(c.func) for c in ast.walk(br) if isinstance(c, ast.Call)}
    assert calls <= {"open", "fh_.write", "print", "json.dumps", "C.sha_b", "round", "len"}, f"dispatch branch calls {calls}"
    del d[i]
    sites.append("dispatch branch")
    m = fn["main"]
    a_ = [k for k, x in enumerate(m.args.args) if x.arg == "spec_only"]
    assert len(a_) == 1 and a_[0] == len(m.args.args) - 1 and ast.unparse(m.args.defaults[-1]) == "''" and is_name(m.args.args[-1].annotation, "str"), \
        "main: spec_only is the last arg, annotated str, default ''"
    m.args.args.pop(), m.args.defaults.pop()
    sites.append("main arg")

    def walk_body(body):
        out = []
        for s in body:
            if isinstance(s, ast.If) and isinstance(s.test, ast.BoolOp) and isinstance(s.test.op, ast.And) and is_name(s.test.values[0], "spec_only"):
                assert len(s.body) == 1 and isinstance(s.body[0], ast.Raise) and not s.orelse, "main: the spec-only validation only raises"
                calls_ = sorted(ast.unparse(c) for c in ast.walk(s) if isinstance(c, ast.Call))  # Claude r1: no other call in the condition
                assert calls_ == sorted(["Path(spec_only).exists()", "Path(spec_only)", ast.unparse(s.body[0].exc)]) \
                    and isinstance(s.body[0].exc.args[0], ast.Constant), "main: the validation calls only Path(spec_only).exists() and raises a text"
                sites.append("main validation")
                continue
            if isinstance(s, (ast.Assign, ast.AnnAssign, ast.AugAssign)) and any(
                    is_spec_path(t) or is_name(t, "SPEC_ONLY") for t in (s.targets if isinstance(s, ast.Assign) else [s.target])):
                assert isinstance(s, ast.Assign) and len(s.targets) == 1 and is_spec_path(s.targets[0]) and is_name(s.value, "spec_only"), \
                    "main: SPEC_ONLY['path'] = spec_only exactly (one target; Astra r1 R3)"
                sites.append("main set")
                continue
            if isinstance(s, ast.If) and isinstance(s.test, ast.UnaryOp) and isinstance(s.test.op, ast.Not) and is_name(s.test.operand, "spec_only"):
                assert ast.unparse(s.body) == "print(json.dumps({'read_started_pt': stamp, 'rows': labels}), flush=True)" and not s.orelse, \
                    "main: the not-spec-only branch only prints the read_started line (2e 10-17: one stdout JSON object on --spec-only)"
                out += s.body  # with no flag the line prints, as before
                sites.append("main read_started")
                continue
            if isinstance(s, ast.If) and is_name(s.test, "spec_only"):
                assert ast.unparse(s.body) == "todo_ = {}", "main: the shipping branch only skips the listing and sending"
                out += walk_body(s.orelse)
                sites.append("main shipping")
                continue
            for f in ("body", "orelse", "finalbody"):
                if isinstance(getattr(s, f, None), list) and getattr(s, f) and isinstance(getattr(s, f)[0], ast.stmt):
                    setattr(s, f, walk_body(getattr(s, f)))
            if isinstance(s, ast.Try):
                for h in s.handlers:
                    h.body = walk_body(h.body)
            out.append(s)
        return out

    m.body = walk_body(m.body)

    class IfExpOut(ast.NodeTransformer):
        def visit_IfExp(self, n):
            self.generic_visit(n)
            if is_name(n.test, "spec_only"):
                assert isinstance(n.body, ast.Constant), "main: the spec-only arm of sent_now is a constant"  # Claude r1
                sites.append("main sent_now")
                return n.orelse
            return n

    tree = IfExpOut().visit(tree)
    left = [ast.unparse(n) for n in ast.walk(tree) if is_name(n, "spec_only") or is_name(n, "SPEC_ONLY")]
    assert not left, f"spec-only references outside the allowed sites: {left[:3]}"
    return tree


SRC = {"commit": None}  # --src <commit>: the runner as committed there (a merge that added the spec-only mode), not the working tree


def runner_src():
    if SRC["commit"]:
        return subprocess.run(["git", "show", f"{SRC['commit']}:./{RP}"], capture_output=True, check=True, cwd=HERE).stdout.decode()
    return (HERE / RP).read_text()


def builder_check(base, src=None):
    """-> (sites, tree equal to base's). base "none" (a branch whose runner also carries other changes, e.g. route C after the merge): the site
    shapes and no stray reference are still asserted; the equality is not computed (None)."""
    new = ast.parse(src if src is not None else runner_src())
    if base == "none":
        sites = []
        strip(new, sites)
        return sites, None
    old = ast.parse(subprocess.run(["git", "show", f"{base}:./{RP}"], capture_output=True, check=True, cwd=HERE).stdout.decode())
    sites = []
    strip(new, sites)
    return sites, ast.dump(new) == ast.dump(old)


def tamper_fails(base, old, new):
    """Positive control of the builder check: a tampered copy of the source must fail it (an assert, or trees that differ)."""
    src = runner_src()
    assert src.count(old) == 1, old
    try:
        return builder_check(base, src.replace(old, new))[1] is False
    except AssertionError:
        return True


def spec(stamp="2026-10-16_211500", git=None):
    git = git or G0
    shared = {"origin": "S_old", "e12": {"fill_S_old": [{"root": "r1", "admitted": ["w|a", "w|b"]}]}, "simh_old": {"L C": [{"root": "r0"}]},
              "honest_b_relaunch": {"pin": {"commit": git, "path": "m.json"}}}
    return {"mode": "alltasks", "label": "x", "dest": f"xfer/fpr507prep/records/results/alltasks_{stamp}.json",
            "jobs": [("L C", shared, {"_ref": {}}), ("L R", shared, {"_ref": {}}), ("L C", {"origin": "S_new"}, {"_ref": {}})],
            "meta": {"git": git, "stamp": stamp, "hbr": {"pin": {"commit": git, "path": "x.json", "sha256": SH}}, "fold_drops": {"commit": git[:9]}, "read_started_pt": stamp, "script_sha256": "1", "shipped_inputs_sent_now": ["t196"],
                     "pooled_world": {"modules_sha256": {RPM: "m", FOM: "f"}, "levels": [0.01]}, "W": np.arange(6.0).reshape(2, 3), "nan": float("nan")},
            "IN": {"k": [1, 2]}, "cfg": {"F_list": [0.01], "G": 1}}


MIN0 = {"mode": "alltasks", "dest": "records/x_2026-10-16_211500.json", "jobs": [], "meta": {"git": G0, "stamp": "2026-10-16_211500"}, "IN": {}, "cfg": {}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="704122edf")
    ap.add_argument("--src", default="")  # e.g. --base 762772c71 --src 94b6ec670: the merge's runner minus the sites == the pre-merge runner
    a = ap.parse_args()
    SRC["commit"] = a.src or None
    sites, same = builder_check(a.base)
    want = ["module SPEC_ONLY", "dispatch branch", "main arg", "main validation", "main set", "main shipping", "main sent_now", "main read_started"]
    s0 = spec()
    s1 = spec("2026-10-17_013000", G1)
    s1["meta"] |= {"script_sha256": "2", "shipped_inputs_sent_now": "spec-only: not sent", "pooled_world": {"modules_sha256": {RPM: "n", FOM: "f"}, "levels": [0.01]}}
    d1 = SD.diff(s0, s1)
    pl, pp = SD.plant(s0)

    def one(path, f):
        s = copy.deepcopy(s0)
        f(s)
        d = SD.diff(s0, s)
        return d["n_differences"] == 1 and path in d["only_a"] + d["only_b"] + d["changed"]

    cases = {
        f"same builder by code line: the tree minus the spec-only sites equals {a.base}'s": None if a.base == "none" else same,  # None: SKIP (Astra opt.)
        "the spec-only sites are exactly the eight allowed ones": sorted(sites) == sorted(want),
        **({f"--src {a.src}: the working tree's spec-only sites (full trees, branch bodies included) equal --src's":
            site_dumps((HERE / RP).read_text()) == site_dumps(runner_src()),
            "--src, positive control: an upload added inside the working tree's dispatch spec-only branch is caught":
            site_dumps((HERE / RP).read_text().replace("            fh_.write(blob)\n", "            fh_.write(blob)\n            vol_out.batch_upload()\n", 1))
            != site_dumps(runner_src())} if a.src else {}),
        **({f"no spec-only line changed from --src {a.src} to the working tree": not [
            ln_ for ln_ in subprocess.run(["git", "diff", a.src, "--", RP], capture_output=True, check=True, cwd=HERE).stdout.decode().splitlines()
            if ln_[:1] in "+-" and not ln_.startswith(("+++", "---")) and ("spec_only" in ln_ or "SPEC_ONLY" in ln_)]} if a.src else {}),
        "builder check, positive control: a changed line elsewhere fails it": None if a.base == "none" else tamper_fails(a.base, "else drive).spawn(spec_path)",
                                                                                          "else drive).spawn(spec_path + '')"),
        "builder check, positive control: an extra statement in the spec-only shipping branch fails it":
            tamper_fails(a.base, "    if spec_only:  # --spec-only: the same content-addressed refs; nothing is listed or sent\n        todo_ = {}\n",  # route C runner port: the
                         "    if spec_only:  # --spec-only: the same content-addressed refs; nothing is listed or sent\n        todo_ = {}\n        meta['x'] = 1\n"),  # kappa dry run adds a 2nd todo_ = {}
        "builder check, positive control: a call hidden in the validation condition fails it":
            tamper_fails(a.base, "or Path(spec_only).exists()):", "or Path(spec_only).exists() or vol_out.listdir('/x')):"),
        "builder check, positive control: an extra statement in the not-spec-only read_started branch fails it":
            tamper_fails(a.base, "        print(json.dumps({\"read_started_pt\": stamp, \"rows\": labels}), flush=True)\n",
                         "        print(json.dumps({\"read_started_pt\": stamp, \"rows\": labels}), flush=True)\n        meta['x'] = 1\n"),
        "builder check, positive control: a call in the spec-only arm of sent_now fails it":
            tamper_fails(a.base, '"spec-only: not sent" if spec_only', 'vol_out.listdir("/x") if spec_only'),
        "builder check, positive control: a call in the SPEC_ONLY initializer fails it (Astra r1 R3)":
            tamper_fails(a.base, 'SPEC_ONLY = {"path": ""}', 'SPEC_ONLY = {"path": vol_out.listdir("/x")}'),
        "builder check, positive control: a second target on the SPEC_ONLY set fails it (Astra r1 R3)":
            tamper_fails(a.base, '    SPEC_ONLY["path"] = spec_only\n', '    SPEC_ONLY["path"] = remote_driver = spec_only\n'),
        "builder check, positive control: another annotation on spec_only fails it":
            tamper_fails(a.base, '         spec_only: str = ""):', '         spec_only: object = ""):'),
        "builder check, positive control: a spec-only branch that also uploads fails it":
            tamper_fails(a.base, "            fh_.write(blob)\n", "            fh_.write(blob)\n            vol_out.batch_upload()\n"),
        "NULL: a spec against a deep copy: no difference": SD.diff(s0, copy.deepcopy(s0))["n_differences"] == 0,
        "NULL: only ignored fields and the run stamp differ: no difference, the ignored ones listed": d1["n_differences"] == 0
        and sorted(d1["ignored_differing"]) == sorted(["meta/git", "meta/stamp", "meta/read_started_pt", "meta/script_sha256",
                                                       "meta/shipped_inputs_sent_now", "meta/pooled_world/modules_sha256/" + RPM])
        and sorted(d1["normalized_differing"]) == sorted(["dest", "jobs/0/1/honest_b_relaunch/pin/commit", "meta/hbr/pin/commit", "meta/fold_drops/commit"]),
        "a sha256 that merely starts with HEAD's prefix is NOT read as <git>": one("meta/hbr/pin/sha256", lambda s: s["meta"]["hbr"]["pin"].__setitem__(
            "sha256", "704122edf" + "f" * 55)),
        "HEAD in a field not named commit is NOT read as <git> (Astra r1 opt. (a))": (lambda d: d["n_differences"] == 1 and d["changed"] == ["meta/note"])(
            SD.diff(s0 | {"meta": s0["meta"] | {"note": G0}}, s1 | {"meta": s1["meta"] | {"note": G1}})),
        "a pin commit that is not HEAD is NOT read as <git>": one("meta/hbr/pin/commit", lambda s: s["meta"]["hbr"]["pin"].__setitem__("commit", "deadbeef1")),
        "R1: an ignored field present only on b: ignored, not a difference": (lambda d: d["n_differences"] == 0 and d["ignored_differing"] == ["meta/_submit_unix"])(
            SD.diff(s0, s0 | {"meta": s0["meta"] | {"_submit_unix": 123.0}})),
        "R1: an ignored field present only on a: ignored, not a difference": (lambda d: d["n_differences"] == 0 and d["ignored_differing"] == ["meta/_submit_unix"])(
            SD.diff(s0 | {"meta": s0["meta"] | {"_submit_unix": 123.0}}, s0)),
        "R1: a NON-ignored field present on one side only is a difference": SD.diff(s0, s0 | {"meta": s0["meta"] | {"new_field": 1}})["only_b"] == ["meta/new_field"],
        "R2: a meta.git-only change (no HEAD pins): ignored, not normalized": (lambda d: d["n_differences"] == 0 and d["ignored_differing"] == ["meta/git"]
                                                                               and d["normalized_differing"] == [])(SD.diff(MIN0, MIN0 | {"meta": MIN0["meta"] | {"git": G1}})),
        "R2: a stamp-only change: meta.stamp ignored, dest normalized": (lambda d: d["n_differences"] == 0 and d["ignored_differing"] == ["meta/stamp"]
                                                                         and d["normalized_differing"] == ["dest"])(
            SD.diff(MIN0, MIN0 | {"meta": MIN0["meta"] | {"stamp": "2026-10-17_013000"}, "dest": "records/x_2026-10-17_013000.json"})),
        "pins_at: route C's zero-unit pin as {at, path, sha256} only (no other field leaves); none when absent": SD.pins_at(
            {"meta": {"full_length": {"s_old_simh_extra_roots": {"R": {"zero_unit_list": {"path": "p.json", "sha256": "9" * 64, "n_keys": 3}}}}}}) == [
            {"at": "meta/full_length/s_old_simh_extra_roots/R/zero_unit_list", "path": "p.json", "sha256": "9" * 64}] and SD.pins_at(s0) == [],
        "key order: not a difference, listed": (lambda d: d["n_differences"] == 0 and d["order_differing"] == ["cfg"])(
            SD.diff(s0, s0 | {"cfg": {"G": s0["cfg"]["G"], "F_list": s0["cfg"]["F_list"]}})),
        "POSITIVE: plant() gives exactly the planted path, in one of the two jobs sharing a cfg":
            {k: v for k, v in SD.diff(s0, pl).items() if k != "order_differing"} == {"only_a": [], "only_b": [pp], "changed": [], "ignored_differing": [],
                                                                                         "normalized_differing": [], "n_differences": 1}
            and pp == "jobs/0/1/e12/fill_S_old/0/admitted/2" and s0["jobs"][0][1]["e12"]["fill_S_old"][0]["admitted"] == ["w|a", "w|b"],
        "a non-ignored meta field changed: one path": one("meta/pooled_world/levels/0", lambda s: s["meta"]["pooled_world"]["levels"].__setitem__(0, 0.05)),
        "another module's sha changed (full_override): one path, not ignored":
            one("meta/pooled_world/modules_sha256/" + FOM, lambda s: s["meta"]["pooled_world"]["modules_sha256"].__setitem__(FOM, "g")),
        "a numpy array changed: one path": one("meta/W", lambda s: s["meta"]["W"].__setitem__((0, 0), 9.0)),
        "a dtype change alone: one path": one("meta/W", lambda s: s["meta"].__setitem__("W", s["meta"]["W"].astype(np.float32))),
        "int vs float: one path": one("IN/k/0", lambda s: s["IN"]["k"].__setitem__(0, 1.0)),
        "a key added to one job: one path": one("jobs/2/1/simh_old", lambda s: s["jobs"][2][1].__setitem__("simh_old", {})),
        "another stamp inside a pinned path is NOT normalized": one("dest", lambda s: s.__setitem__("dest", s["dest"].replace("211500", "211501"))),
        "NaN equals NaN": SD.diff(s0, copy.deepcopy(s0) | {"meta": copy.deepcopy(s0["meta"]) | {"nan": float("nan")}})["n_differences"] == 0,
        "plant with no S_old e12 job: STOPs": False,
        "VOL is run_common455's solavg Volume": SD.VOL == ast.literal_eval(next(n.value for n in ast.parse((HERE / "run_common455.py").read_text()).body
                                                                          if isinstance(n, ast.Assign) and is_name(n.targets[0], "VOLS")))["solavg"],
    }
    try:
        SD.plant(s0 | {"jobs": [s0["jobs"][2]]})
    except SystemExit:
        cases["plant with no S_old e12 job: STOPs"] = True
    for name, ok in cases.items():
        print(("SKIP " if ok is None else "PASS " if ok else "FAIL ") + name)
    assert all(v is not False for v in cases.values()), "spec-only tests failed"
    n_skip = sum(v is None for v in cases.values())  # --base none: the tree equality and its control are not run, and say so
    print(f"{len(cases) - n_skip} checks pass" + (f", {n_skip} SKIPPED (no base: shapes only)" if n_skip else ""))


if __name__ == "__main__":
    main()
