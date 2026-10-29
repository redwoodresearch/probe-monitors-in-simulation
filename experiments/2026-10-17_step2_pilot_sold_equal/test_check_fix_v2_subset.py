"""Synthetic test of check_fix_v2_subset.py. Fabricated inventories, counted keys and INPUTS_FULL in a throwaway git repo; no real record
is read. Each case asserts its exact outcome.

Positive cases:
- all rows identical after re-serializing: PASS, rows_reserialized_identical 3 and rows_canonical_only 0 in the record, and the written
  pins equal the inputs.
Negative cases, each STOPping with its exact reason and writing nothing:
- one fix_v1 row with only its key order changed: STOP to 09 with both counts and that row's key (2e's ruling: canonical-only must be 0);
- two fix_v1 rows swapped in fix_v2, and a pilot row between fix_v1's rows (fix_v1 is not the ordered prefix);
- a repeated member name in one fix_v2 row (raw JSON; json.loads alone would keep the last value);
- a changed value; a changed type (1 vs "1"); a fix_v1 row missing;
- an extra non-pilot row; a missing pilot row; a duplicate key; a fix_v1 row K changed in the fix_v1 part plus K's original copy among
  the added rows (one exact reason: the duplicate key, before any key map);
- a wrong world; a wrong .eval sha;
- S_newmains not fix_v1; INPUTS_FULL without the pins;
- a wrong INPUTS_FULL sha256; an inventory missing from git.
Direct-pin mode (--fix-v2 / --fix-v1, 09 via 2e, 10-17): a PASS with input_mode naming the mode and both pins; the positive controls (a
planted extra row, a changed row) STOP; --fix-v1 not fix_v1, --fix-v2 alone, --fix-v1 with --inputs-full, a pin without =<sha256>, and
--inputs-full with --fix-v2 (argparse) each STOP, writing nothing.

The checker's bytes are this branch's HEAD blob (--worktree: the working-tree file, to test before a commit).

  uv run --no-project --python 3.12 python experiments/2026-10-17_step2_pilot_sold_equal/test_check_fix_v2_subset.py [--tmp <dir>] [--worktree]
"""
import argparse
import copy
import hashlib
import importlib.util
import json
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REAL = HERE.parents[1]
REL = "experiments/2026-10-17_step2_pilot_sold_equal"
sha = lambda b: hashlib.sha256(b).hexdigest()  # noqa: E731
KEYS = [f"2026-10-14_140858_simh_pilot/sim_honest@september/t{21 + i}_x/run{i}.eval#e0" for i in range(20)]


def git(repo, *a):
    return subprocess.run(["git", "-C", str(repo), *a], capture_output=True, text=True, check=True).stdout.strip()


def main(tmp, worktree=False):
    repo = Path(tempfile.mkdtemp(prefix="fixv2_check_", dir=tmp))
    git(repo, "init", "-q")
    for k, v in (("user.email", "test@example.invalid"), ("user.name", "test"), ("commit.gpgsign", "false")):
        git(repo, "config", k, v)
    (repo / REL).mkdir(parents=True)
    (repo / REL / "check_fix_v2_subset.py").write_bytes((HERE / "check_fix_v2_subset.py").read_bytes() if worktree else
                                                        subprocess.run(["git", "-C", str(REAL), "show", f"HEAD:{REL}/check_fix_v2_subset.py"],
                                                                       capture_output=True, check=True).stdout)
    v1 = {"runs": [{"key": f"other/run{i}.eval#e0", "world": "honest_c", "full_len": 5 + i, "uncut_planned": {"folder": "q4f_c_cutfull_v1__honest_b"}} for i in range(3)]}
    pilot = [{"key": k, "world": "sim_honest", "eval_sha256": sha(k.encode()), "full_len": 100 + i} for i, k in enumerate(KEYS)]
    counted = {"n": 20, "keys": [{"volume_path": "simh_evals_v1/" + k.removesuffix("#e0"), "eval_sha256": sha(k.encode())} for k in KEYS]}
    v2_good = {"status": "COMPLETE", "runs": copy.deepcopy(v1["runs"]) + pilot}
    r0 = v1["runs"][0]
    variants = {  # name -> (fix_v2 document, the exact STOP reason or None for a pass)
        "reordered keys": ({"runs": [dict(reversed(list(r0.items())))] + v2_good["runs"][1:]}, None),
        "a changed value": ({"runs": [dict(r0, full_len=99)] + v2_good["runs"][1:]}, "a fix_v1 row is not content-identical (canonical JSON) in fix_v2"),
        "a changed type (1 vs \"1\")": ({"runs": [dict(r0, full_len=str(r0["full_len"]))] + v2_good["runs"][1:]},
                                         "a fix_v1 row is not content-identical (canonical JSON) in fix_v2"),
        "a fix_v1 row missing": ({"runs": v2_good["runs"][1:]}, "a fix_v1 row is missing from fix_v2"),
        "an extra non-pilot row": ({"runs": v2_good["runs"] + [{"key": "extra/run.eval#e0", "world": "sim_honest"}]}, "fix_v2 minus fix_v1 is not exactly the 20 counted pilot keys"),
        "a pilot row missing": ({"runs": v2_good["runs"][:-1]}, "fix_v2 minus fix_v1 is not exactly the 20 counted pilot keys"),
        "a duplicate key": ({"runs": v2_good["runs"] + [pilot[0]]}, "a key appears twice in fix_v1 or fix_v2"),
        "a wrong world": ({"runs": v2_good["runs"][:-1] + [dict(pilot[-1], world="simh_pilot")]}, "a pilot row's inventory world is not sim_honest"),
        "a wrong .eval sha": ({"runs": v2_good["runs"][:-1] + [dict(pilot[-1], eval_sha256="0" * 64)]}, "a pilot row's eval_sha256 differs from its counted key's"),
        "two fix_v1 rows swapped": ({"runs": [v1["runs"][1], v1["runs"][0], v1["runs"][2]] + pilot}, "fix_v1's keys are not the ordered prefix of fix_v2's"),
        "K changed, K's original among the added rows": ({"runs": [dict(r0, full_len=99)] + v1["runs"][1:] + pilot[:19] + [r0]},
                                                          "a key appears twice in fix_v1 or fix_v2"),
        "a pilot row between fix_v1's rows": ({"runs": v1["runs"][:1] + pilot[:1] + v1["runs"][1:] + pilot[1:]}, "fix_v1's keys are not the ordered prefix of fix_v2's"),
    }
    files = {"fab/fix_v1.json": v1, "fab/counted.json": counted, "fab/fix_v2_good.json": v2_good} | {f"fab/fix_v2_{i}.json": d for i, (d, _) in enumerate(variants.values())}
    for p, obj in files.items():
        (repo / p).parent.mkdir(parents=True, exist_ok=True)
        (repo / p).write_text(json.dumps(obj, indent=1))
    good_txt = (repo / "fab/fix_v2_good.json").read_text()  # a repeated member name, written as raw JSON text (json.dumps cannot)
    dup_txt = good_txt.replace('"full_len": 5,', '"full_len": 5, "full_len": 6,', 1)
    assert dup_txt != good_txt
    (repo / "fab/fix_v2_dup_member.json").write_text(dup_txt)
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "fabricated")
    c = git(repo, "rev-parse", "HEAD")[:12]
    pin = lambda p: {"commit": c, "path": p, "sha256": sha((repo / p).read_bytes())}  # noqa: E731
    spec = importlib.util.spec_from_file_location("checker", repo / REL / "check_fix_v2_subset.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    m.FIX_V1, m.COUNTED = (c, "fab/fix_v1.json", pin("fab/fix_v1.json")["sha256"]), (c, "fab/counted.json", pin("fab/counted.json")["sha256"])
    out = repo / REL / "results"
    listing = lambda: sorted(out.glob("*")) if out.exists() else []  # noqa: E731

    def run(name, doc, inputs_sha=None):  # -> (STOP message or None, the written record or None)
        p = f"fab/inputs_full_{hashlib.sha256(name.encode()).hexdigest()[:16]}.json"  # a content hash, never Python's hash()
        (repo / p).write_text(json.dumps(doc, indent=1))
        git(repo, "add", "-A")
        git(repo, "commit", "-q", "-m", f"inputs_full {name}")
        before = listing()
        time.sleep(1.1)  # the record's name has a seconds stamp (open "x")
        try:
            m.main(["--inputs-full", f"{p}@{git(repo, 'rev-parse', 'HEAD')[:12]}={inputs_sha or sha((repo / p).read_bytes())}"])
            e = None
        except SystemExit as x:
            e = str(x)
        new = [q for q in listing() if q not in before]
        return e, (json.loads(new[0].read_text()) if len(new) == 1 else None), len(new), p

    inv = lambda s_old, s_new="fab/fix_v1.json": {"inventories": {"S_old": pin(s_old), "S_newmains": pin(s_new)}}  # noqa: E731
    res = {}
    e, rec, n, p = run("good", inv("fab/fix_v2_good.json"))
    res["all rows identical after re-serializing: PASS, rows_reserialized_identical 3, rows_canonical_only 0; the written pins equal the inputs"] = \
        e is None and n == 1 and rec["rows_reserialized_identical"] == 3 and rec["rows_canonical_only"] == 0 and "reorder_explained" not in rec \
        and {k: rec["fix_v2"][k] for k in ("commit", "path", "sha256")} == pin("fab/fix_v2_good.json") \
        and {k: rec["fix_v1"][k] for k in ("commit", "path", "sha256")} == pin("fab/fix_v1.json") \
        and rec["inputs_full"] == {"path": p, "commit": git(repo, "log", "-1", "--format=%H", "--", p), "sha256": sha((repo / p).read_bytes())}
    reorder_stop = ("STOP (to 09): 1 fix_v1 rows are only canonical-identical in fix_v2 (key order differs; rows_reserialized_identical 2, rows_canonical_only 1); "
                    f"keys {[r0['key']]}; nothing written")
    for i, (name, (_, reason)) in enumerate(variants.items()):
        if name == "reordered keys":
            e, rec, n, _ = run(name, inv(f"fab/fix_v2_{i}.json"))
            res[f"NEGATIVE one row with only its key order changed: STOP '{reorder_stop}', nothing written"] = e == reorder_stop and n == 0
            if e != reorder_stop:
                print("   got:", e)
            continue
        e, rec, n, _ = run(name, inv(f"fab/fix_v2_{i}.json"))
        want = f"STOP: {[reason]}; nothing written"
        res[f"NEGATIVE {name}: STOP '{reason}', nothing written"] = e == want and n == 0
        if e != want:
            print("   got:", e)
    e, rec, n, _ = run("a repeated member name", inv("fab/fix_v2_dup_member.json"))
    res["NEGATIVE a repeated member name in a fix_v2 row: exact STOP, nothing written"] = \
        e == "STOP: fix_v1 or fix_v2: duplicate member 'full_len'; nothing written" and n == 0
    if e != "STOP: fix_v1 or fix_v2: duplicate member 'full_len'; nothing written":
        print("   got:", e)
    e, rec, n, _ = run("S_newmains not fix_v1", inv("fab/fix_v2_good.json", "fab/fix_v2_good.json"))
    res["NEGATIVE INPUTS_FULL's S_newmains is not fix_v1: exact STOP, nothing written"] = \
        e == "STOP: INPUTS_FULL's inventories.S_newmains is not inventory_b_fix_v1; nothing written" and n == 0
    e, rec, n, _ = run("no inventories", {"other": 1})
    res["NEGATIVE INPUTS_FULL without the inventories pins: exact STOP, nothing written"] = \
        e == "STOP: INPUTS_FULL lacks inventories.S_old or inventories.S_newmains as {commit, path, sha256}; nothing written" and n == 0
    e, rec, n, p = run("wrong sha", inv("fab/fix_v2_good.json"), inputs_sha="0" * 64)
    res["NEGATIVE a wrong INPUTS_FULL sha256: exact STOP, nothing written"] = \
        e == f"STOP: {p} @ {git(repo, 'rev-parse', 'HEAD')[:12]} differs from its pinned sha256; nothing written" and n == 0
    e, rec, n, _ = run("missing object", {"inventories": {"S_old": {"commit": c, "path": "fab/not_there.json", "sha256": "0" * 64}, "S_newmains": pin("fab/fix_v1.json")}})
    res["NEGATIVE an inventory missing from git: exact STOP (reasoned, no crash), nothing written"] = \
        e == f"STOP: fab/not_there.json @ {c} is not in git; nothing written" and n == 0
    # direct-pin mode (09 via 2e, 10-17): --fix-v2 / --fix-v1 instead of --inputs-full; the same checks and record, plus input_mode
    arg = lambda p: f"{p}@{c}={pin(p)['sha256']}"  # noqa: E731

    def run_direct(argv):  # -> (STOP message or None, the written record or None, the number of files written)
        before = listing()
        time.sleep(1.1)
        try:
            m.main(argv)
            e = None
        except SystemExit as x:
            e = str(x)
        new = [q for q in listing() if q not in before]
        return e, (json.loads(new[0].read_text()) if len(new) == 1 else None), len(new)

    vi = {name: i for i, name in enumerate(variants)}
    e, rec, n = run_direct(["--fix-v2", arg("fab/fix_v2_good.json"), "--fix-v1", arg("fab/fix_v1.json")])
    res["direct pins: PASS with the same counts; input_mode names the mode and both pins; inputs_full is null"] = \
        e is None and n == 1 and rec["rows_reserialized_identical"] == 3 and rec["rows_canonical_only"] == 0 and rec["inputs_full"] is None \
        and rec["input_mode"] == {"mode": "direct pins", "fix_v2": pin("fab/fix_v2_good.json"), "fix_v1": pin("fab/fix_v1.json")} \
        and {k: rec["fix_v2"][k] for k in ("commit", "path", "sha256")} == pin("fab/fix_v2_good.json") and rec["fix_v2"]["from"] == "--fix-v2" \
        and {k: rec["fix_v1"][k] for k in ("commit", "path", "sha256")} == pin("fab/fix_v1.json") and rec["fix_v1"]["from"] == "--fix-v1"
    for name in ("an extra non-pilot row", "a changed value"):  # the positive controls: a planted extra row, a changed row
        e, rec, n = run_direct(["--fix-v2", arg(f"fab/fix_v2_{vi[name]}.json"), "--fix-v1", arg("fab/fix_v1.json")])
        want = f"STOP: {[variants[name][1]]}; nothing written"
        res[f"NEGATIVE direct pins, {name}: STOP '{variants[name][1]}', nothing written"] = e == want and n == 0
        if e != want:
            print("   got:", e)
    for name, argv, want in (
            ("--fix-v1 is not fix_v1", ["--fix-v2", arg("fab/fix_v2_good.json"), "--fix-v1", arg("fab/fix_v2_good.json")],
             "STOP: --fix-v1 is not inventory_b_fix_v1; nothing written"),
            ("--fix-v2 without --fix-v1", ["--fix-v2", arg("fab/fix_v2_good.json")],
             "STOP: --fix-v2 and --fix-v1 go together, and never with --inputs-full; nothing written"),
            ("--fix-v1 with --inputs-full", ["--inputs-full", "x@y=z", "--fix-v1", arg("fab/fix_v1.json")],
             "STOP: --fix-v2 and --fix-v1 go together, and never with --inputs-full; nothing written"),
            ("a pin without =<sha256>", ["--fix-v2", f"fab/fix_v2_good.json@{c}", "--fix-v1", arg("fab/fix_v1.json")],
             "STOP: --fix-v2 or --fix-v1 is not <path>@<commit>=<sha256>; nothing written"),
            ("--inputs-full with --fix-v2 (argparse: mutually exclusive)", ["--inputs-full", "x@y=z", "--fix-v2", arg("fab/fix_v2_good.json"),
                                                                            "--fix-v1", arg("fab/fix_v1.json")], "2")):
        e, rec, n = run_direct(argv)
        res[f"NEGATIVE direct pins, {name}: exact STOP, nothing written"] = e == want and n == 0
        if e != want:
            print("   got:", e)
    for k, v in res.items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}")
    print("ALL PASS" if all(res.values()) else "FAILED")
    shutil.rmtree(repo)
    if not all(res.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tmp", default=None)
    ap.add_argument("--worktree", action="store_true", help="test the working-tree checker instead of the HEAD blob")
    x = ap.parse_args()
    main(x.tmp, x.worktree)
