"""Synthetic END-TO-END test of the pilot-only sold_equal copies, with changes (5) and (6) and 09's (6) binding. Fabricated records only,
in a throwaway git repo whose "origin/aswin/q3-cut-option" ref is local; no Modal, no real record read.

What it does:
- build_cutfull_message.py --sold builds the pilot message from a fabricated attestation, stage, extraction, format and step-0 record,
  the counted keys, and a fix_v2-like inventory. That inventory is bound through its own fields to a pilot-rows file and a fix_v1 ⊂ fix_v2
  check record.
- check_sold_equality.py then passes on that one record.
Each negative asserts its exact STOP reason and that nothing was written:
- (5): the pinned gate record with passes false, or with passes "true" (a string: the pilot takes JSON true only); the pinned record's
  run keys != the counted keys; a record other than the pinned one;
- Astra R1: a pilot attestation that cites the legacy compare (a passing record at the legacy pin) is still gated as the pilot;
- strict booleans: a format record with passes "true" (a string);
- git failures in the pinned reads (show()): a pin to a path not in git, and a pin to a commit not in git, each a reasoned STOP. Other git
  calls in the copy can still raise CalledProcessError (a traceback; still fail-closed);
- 2e's sold_equal REQUIRED set: a check record with rows_canonical_only 1, or rows_reserialized_identical != n_fix_v1; a fix_v1 row
  whose members are reordered, or whose 7 became 7.0 (json.dumps, not dict equality); two fix_v1 rows swapped (not the ordered
  prefix); a repeated member name in the inventory (raw JSON); a pilot-rows file differing from the inventory's 20 added rows in one
  row, or in their order; an attestation whose pilot identity disagrees (pilot folder with b1's original folder, with another
  original_target, or with none);
- 2e's follow-up: a fix_v1 row K changed in the fix_v1 part plus K's original copy among the 20 added rows (a duplicate key that
  last-wins would hide); a check record whose rows_canonical_only is false or 0.0, whose n_fix_v1, n_added or counted_keys sha is wrong;
- (6): each binding field wrong (pilot_rows_file path, its sha256, status, rows_added, supersedes, original_folder, uncut_folder); the
  pins unset; the check record not passing, or naming another fix_v2; a fix_v1 row missing; a fix_v1 row different;
- the one-tuple rule: a build-copy pin literal different from the check copy's.
The constants that name real pins are set in-process to the fabricated files; the copies' bytes are this branch's HEAD blobs
(--worktree: the working-tree files, to test before a commit).

  uv run --no-project --python 3.12 python experiments/2026-10-17_step2_pilot_sold_equal/test_build_sold_e2e.py [--tmp <dir>] [--worktree]
"""
import argparse
import copy
import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REAL = HERE.parents[1]
REL = "experiments/2026-10-17_step2_pilot_sold_equal"
CH = "experiments/2026-10-14_q3_overnight_chain/results"
ATT = f"{CH}/cutfull/attest/attestations_sold__simh_q4f_v1_pilot_cutfull.json"
ROUT = f"{CH}/cutfull/routing/sold__simh_q4f_v1_pilot_cutfull.json"
PF, PO, SS, ROOT, ARM, TOK = ("simh_q4f_v1_pilot_cutfull_v1__sim_honest", "simh_q4f_v1_pilot__sim_honest", "simh_pilot_cutfull_v1",
                              "simh_pilot_cutfull_extract_v1", "sim_honest@september", "sim_honest")
VOL = "probe-shift_aswin-surya_sub-q3cutfull_acts_kill-2026-11-14"
U = hashlib.sha256(b"fabricated staged file").hexdigest()
KEYS = sorted(f"2026-10-14_140858_simh_pilot/{ARM}/t{21 + i}_x/run{i}.eval#e0" for i in range(20))
sha = lambda b: hashlib.sha256(b).hexdigest()  # noqa: E731
GATE_STOP = 'adapter gates failed: {"step0_gate": {"pass": false}}'
Q1_STOP = "STOP (5): the attestation's pilot identity disagrees: cutfull.folders, original_folders and original_target must all name the pilot, or none"


def git(repo, *a):
    return subprocess.run(["git", "-C", str(repo), *a], capture_output=True, text=True, check=True).stdout.strip()


def put(repo, rel, obj):
    p = repo / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(obj if isinstance(obj, bytes) else json.dumps(obj, indent=1).encode())
    return rel


def commit(repo, msg):
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", msg)
    c = git(repo, "rev-parse", "HEAD")
    git(repo, "update-ref", "refs/remotes/origin/aswin/q3-cut-option", c)  # every commit is "on origin" (the copies' on_origin) and on BRANCH
    return c


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def stops(fn):
    try:
        fn()
    except SystemExit as e:
        return str(e) or "exit"
    return None


# (a2, 10-17, 09 via 2e): the copies' inventory pins are now FILLED with the real fix_v2 / pilot-rows tuples. The fabricated cases below
# need them unset, so the copies are written into the throwaway repo with exactly these literals reverted to None (nothing else changes).
FIXV2 = ("98575cc3471a224c87401111fbb8caee28ff5323", "experiments/2026-10-15_cut_sensitivity/results/inventory_b_fix_v2_2026-10-17_081456.json",
         "f8dfc018d44c5a5bd373f88d9f7974dd9774c25a325d3ceceb385fcb9c82719f")
PROWS = ("98575cc3471a224c87401111fbb8caee28ff5323", "experiments/2026-10-15_cut_sensitivity/results/inventory_b_pilot_rows_v1_2026-10-17_081456.json",
         "854ff211aaa61a3b24ca17b3104c910a377ea11808e06680f4b99e4ae1f9bf45")
CHK = ("26c12ab14b5d33b57e383d9774c39ae865f7b25b", "experiments/2026-10-17_step2_pilot_sold_equal/results/inventory_b_fix_v2_subset_check_2026-10-17_115418.json",
       "dbbd63f66827321f56407ade32b1c2e3304dd30c13884dada3853b0a9f727583")
FILLED = {"build_cutfull_message.py": {"INVENTORY_B": FIXV2, "INVENTORY_B_RECORD": PROWS, "INVENTORY_B_CHECK": CHK}, "check_sold_equality.py": {"INVENTORY": FIXV2}}


def unpin(f, b):
    """The copy f's bytes with its FILLED pin literals set back to None; each must be there exactly once (else the test fails loudly)."""
    t = b.decode()
    for n, v in FILLED[f].items():
        lit = f"\n{n} = {repr(v).replace(chr(39), chr(34))}  #"
        assert t.count(lit) == 1, (f, n)
        t = t.replace(lit, f"\n{n} = None  #")
    return t.encode()


def main(tmp, worktree=False):
    repo = Path(tempfile.mkdtemp(prefix="sold_e2e_", dir=tmp))
    git(repo, "init", "-q")
    for k, v in (("user.email", "test@example.invalid"), ("user.name", "test"), ("commit.gpgsign", "false")):
        git(repo, "config", k, v)
    for f in ("build_cutfull_message.py", "check_sold_equality.py"):
        put(repo, f"{REL}/{f}", unpin(f, (HERE / f).read_bytes() if worktree else
            subprocess.run(["git", "-C", str(REAL), "show", f"HEAD:{REL}/{f}"], capture_output=True, check=True).stdout))
    # fabricated inputs (names and paths as the copies expect)
    step0 = put(repo, f"{CH}/cutfull_step0_compare_2026-10-17_020344.json", {"passes": True, "runs": [{"key": k} for k in KEYS]})
    step0_false = put(repo, "fab/step0_passes_false.json", {"passes": False, "runs": [{"key": k} for k in KEYS]})
    step0_missing = put(repo, "fab/step0_missing.json", {"passes": True, "runs": [{"key": k} for k in KEYS[1:]]})
    step0_str = put(repo, "fab/step0_passes_string.json", {"passes": "true", "runs": [{"key": k} for k in KEYS]})
    legacy = put(repo, f"{CH}/cutfull_step0_compare_2026-10-15_214206.json", {"passes": True, "runs": [{"key": k} for k in KEYS]})  # at the legacy STEP0 path
    counted = put(repo, "fab/counted_keys.json", {"keys": [{"volume_path": "simh_evals_v1/" + k.removesuffix("#e0")} for k in KEYS]})
    v1_rows = [{"key": f"other/run{i}.eval#e0", "uncut_planned": {"folder": "q4f_c_cutfull_v1__honest_b"}, "world": "honest_c", "full_len": 7} for i in range(3)]
    fix_v1 = put(repo, "fab/inventory_b_fix_v1.json", {"runs": v1_rows})
    pilot_rows = [{"key": k, "task": f"t{21 + i}_x", "world": "sim_honest", "arm": ARM, "original_folder": PO, "original_folder_registered": True,
                   "full_len": 1000 + i, "tokens_before": 1000 + i, "registered_source": {"eval_sha256": sha(k.encode())}, "uncut_planned": {"folder": PF}}
                  for i, k in enumerate(KEYS)]
    rows_file = put(repo, "fab/inventory_b_pilot_rows_v1_2026-10-17_000000.json", {"runs": pilot_rows})
    rows_dupk = put(repo, "fab/inventory_b_pilot_rows_v1_2026-10-17_000003.json", {"runs": pilot_rows[:19] + [v1_rows[0]]})  # the suffix of the K case
    rows_bad = {"one row differs": put(repo, "fab/inventory_b_pilot_rows_v1_2026-10-17_000001.json", {"runs": pilot_rows[:-1] + [dict(pilot_rows[-1], full_len=1)]}),
                "the rows in another order": put(repo, "fab/inventory_b_pilot_rows_v1_2026-10-17_000002.json", {"runs": pilot_rows[1:] + pilot_rows[:1]})}
    good = {"status": "COMPLETE", "supersedes": {"sha256": sha((repo / fix_v1).read_bytes())},
            "fix_v2": {"pilot_rows_file": {"path": Path(rows_file).name, "sha256": sha((repo / rows_file).read_bytes())}, "rows_added": 20,
                       "original_folder": PO, "uncut_folder": PF}, "runs": v1_rows + pilot_rows}
    variants = {"good": good}  # name -> inventory document; each gets its own check record naming its sha
    for why, mut in (("fix_v2.pilot_rows_file is not the pinned pilot-rows file", lambda d: d["fix_v2"]["pilot_rows_file"].update(path="other.json")),
                     ("fix_v2.pilot_rows_file is not the pinned pilot-rows file (sha256)", lambda d: d["fix_v2"]["pilot_rows_file"].update(sha256="0" * 64)),
                     ("status is not COMPLETE", lambda d: d.update(status="STOP")), ("fix_v2.rows_added is not 20", lambda d: d["fix_v2"].update(rows_added=19)),
                     ("supersedes.sha256 is not inventory_b_fix_v1's", lambda d: d.update(supersedes={"sha256": "0" * 64})),
                     ("fix_v2.original_folder is not the pilot's", lambda d: d["fix_v2"].update(original_folder="simh_q4f_v1_full_b1__sim_honest")),
                     ("fix_v2.uncut_folder is not the pilot's", lambda d: d["fix_v2"].update(uncut_folder="simh_pilot_cutfull_v1__sim_honest")),
                     ("fix_v1 row missing", lambda d: d.update(runs=d["runs"][1:])),
                     ("fix_v1 row differs", lambda d: d.update(runs=[dict(d["runs"][0], full_len=8)] + d["runs"][1:])),
                     ("fix_v1 row with its members reordered", lambda d: d.update(runs=[dict(reversed(list(d["runs"][0].items())))] + d["runs"][1:])),
                     ("fix_v1 row with 7 as 7.0", lambda d: d.update(runs=[dict(d["runs"][0], full_len=7.0)] + d["runs"][1:])),
                     ("fix_v1 rows swapped", lambda d: d.update(runs=[d["runs"][1], d["runs"][0]] + d["runs"][2:])),
                     *[(f"pilot rows file: {k}", (lambda d, p=p: d["fix_v2"].update(pilot_rows_file={"path": Path(p).name, "sha256": sha((repo / p).read_bytes())})))
                       for k, p in rows_bad.items()],
                     ("K changed in the fix_v1 part, K's original among the added rows", lambda d: d.update(
                         runs=[dict(d["runs"][0], full_len=99)] + d["runs"][1:3] + pilot_rows[:19] + [d["runs"][0]],
                         fix_v2=dict(d["fix_v2"], pilot_rows_file={"path": Path(rows_dupk).name, "sha256": sha((repo / rows_dupk).read_bytes())})))):
        d = copy.deepcopy(good)
        mut(d)
        variants[why] = d
    inv_files = {n: put(repo, f"fab/inventory_b_fix_v2_{i}.json", d) for i, (n, d) in enumerate(variants.items())}
    good_txt = (repo / inv_files["good"]).read_text()
    dup_txt = good_txt.replace('"world": "honest_c",', '"world": "honest_c", "world": "honest_c",', 1)  # raw JSON: json.dumps cannot repeat a member
    assert dup_txt != good_txt
    inv_files["a repeated member name"] = put(repo, "fab/inventory_b_fix_v2_dup_member.json", dup_txt.encode())
    checks = {n: put(repo, f"fab/fix_v2_subset_check_{i}.json", {"passes": True, "fix_v1": {"sha256": sha((repo / fix_v1).read_bytes())},
                                                                  "fix_v2": {"sha256": sha((repo / p).read_bytes())}, "n_fix_v1": 3, "n_added": 20,
                                                                  "counted_keys": {"commit": "x", "path": counted, "sha256": sha((repo / counted).read_bytes())},
                                                                  "rows_reserialized_identical": 3, "rows_canonical_only": 0}) for i, (n, p) in enumerate(inv_files.items())}
    check_false = put(repo, "fab/fix_v2_subset_check_false.json", json.loads((repo / checks["good"]).read_text()) | {"passes": False})
    check_other = put(repo, "fab/fix_v2_subset_check_other.json", json.loads((repo / checks["good"]).read_text()) | {"fix_v2": {"sha256": "0" * 64}})
    check_other_v1 = put(repo, "fab/fix_v2_subset_check_other_v1.json", json.loads((repo / checks["good"]).read_text()) | {"fix_v1": {"sha256": "0" * 64}})
    check_canon1 = put(repo, "fab/fix_v2_subset_check_canonical_only_1.json", json.loads((repo / checks["good"]).read_text())
                       | {"rows_reserialized_identical": 2, "rows_canonical_only": 1, "reorder_explained": "make_inventory_b_fix_v2.py@67e04b0c5:L163"})
    check_reser2 = put(repo, "fab/fix_v2_subset_check_reserialized_2.json", json.loads((repo / checks["good"]).read_text()) | {"rows_reserialized_identical": 2})
    gck = json.loads((repo / checks["good"]).read_text())
    check_bind = {why: put(repo, f"fab/fix_v2_subset_check_bind_{i}.json", gck | ch) for i, (why, ch) in enumerate((
        ("rows_canonical_only is false", {"rows_canonical_only": False}), ("rows_canonical_only is 0.0", {"rows_canonical_only": 0.0}),
        ("n_fix_v1 is 2 (with rows_reserialized_identical 2)", {"n_fix_v1": 2, "rows_reserialized_identical": 2}), ("n_added is 19", {"n_added": 19}),
        ("n_added is 20.0", {"n_added": 20.0}), ("counted_keys.sha256 is another file's", {"counted_keys": dict(gck["counted_keys"], sha256="0" * 64)})))}
    stage = put(repo, f"{CH}/chain_stage_simh_pilot_cutfull_2026-10-17_010154.json",
                {"git": "0" * 40, "simh": {"out": SS, "assembled": {"epochs": {ARM: 20}, "staged_inputs_sha256": {f"{ARM}.jsonl": U}},
                                           "c_gate": {"manifest_keys": 20, "staged": 20, "excluded": {}}}})
    put(repo, f"{CH}/extract_chain_{ROOT}_a1_2026-10-17_014605.json",
        {"config": {"run": ROOT, "inputs": SS, "inputs_sha256": {f"{ARM}.jsonl": U}, "model": "Qwen/Qwen3.5-27B", "cut": "full", "shards": 1},
         "volume": VOL, "shards": [{"shard": 0, "gpu_name": "NVIDIA H200"}]})
    fmt = put(repo, f"{CH}/format_full_simh_pilot_cutfull_sim_honest_2026-10-17_010534.json",
              {"passes": True, "failing_keys": [], "staged_sha": [U], "epochs_checked": 20, "arm": ARM})
    fmt_str = put(repo, "fab/format_passes_string.json", {"passes": "true", "failing_keys": [], "staged_sha": [U], "epochs_checked": 20, "arm": ARM})
    c0 = commit(repo, "fabricated inputs")
    att = {"revision": 1, "failing": {}, "label": "(b) descriptive", "template_class": {TOK: {k: "fixed, render unchanged" for k in KEYS}},
           "cutfull": {"volume": VOL, "staged_set": SS, "extraction_root": ROOT, "folders": {TOK: PF}, "stage_record": f"{stage} @ {c0[:12]}"},
           "original_folders": {TOK: PO}, "original_target": "simh_q4f_v1_pilot", "units": {TOK: {"epochs": 20}}, "format_check": {TOK: f"{fmt} @ {c0[:12]}"},
           "repro_gate": f"{step0} @ {c0[:12]} (the pilot's own per-root step-0 gate)"}
    put(repo, ATT, att)
    c1 = commit(repo, "fabricated attestation")
    b = load(repo / REL / "build_cutfull_message.py", "build_copy")
    pin = lambda p: (c0[:12], p, sha((repo / p).read_bytes()))  # noqa: E731
    GOOD = {"PILOT_STEP0": pin(step0), "COUNTED": pin(counted), "INVENTORY_B": pin(inv_files["good"]), "INVENTORY_B_V1": pin(fix_v1),
            "INVENTORY_B_RECORD": pin(rows_file), "INVENTORY_B_CHECK": pin(checks["good"])}
    out = repo / REL / "results" / "cutfull_b_msgs"
    sys.argv = ["build_cutfull_message.py", "--record", f"{c1}:{ATT}", "--sold", "--scope", "b", "--out-dir", str(out)]
    res = {}

    def neg(name, reason, **consts):
        for k, v in (GOOD | consts).items():
            setattr(b, k, v)
        before = sorted(out.glob("*")) if out.exists() else []
        e = stops(b.main)
        res[f"NEGATIVE {name}: STOP '{reason}', nothing written"] = e == reason and (sorted(out.glob("*")) if out.exists() else []) == before
        if e != reason:
            print(f"   {name}: got:", e)

    # (5) the pinned gate record says passes false: the same path and commit, its own bytes (the file at the gate path is swapped below)
    neg("(5) a record other than the pinned one (sha256)", "step0_gate: sha256 differs from its pin", PILOT_STEP0=(c0[:12], step0, "0" * 64))
    neg("git failure: a pin to a path not in git", f"inventory_pilot_rows: fab/not_there.json @ {c0[:12]} is not in git",
        INVENTORY_B_RECORD=(c0[:12], "fab/not_there.json", "0" * 64))
    neg("git failure: a pin to a commit not in git", f"inventory_subset_check: {checks['good']} @ 0123456789ab is not in git",
        INVENTORY_B_CHECK=("0123456789ab", checks["good"], "0" * 64))
    neg("(6) the pilot-rows pin unset", "STOP (6): the pilot-rows file (INVENTORY_B_RECORD) is not pinned", INVENTORY_B_RECORD=None)
    neg("(6) the check-record pin unset", "STOP (6): the fix_v1 ⊂ fix_v2 check record (INVENTORY_B_CHECK) is not pinned", INVENTORY_B_CHECK=None)
    for why in [w for w in variants if w != "good"]:
        differs = "STOP (6): a fix_v1 row differs in the pilot inventory"
        rows7 = "STOP (6): the pilot-rows file is not exactly the pilot inventory's 20 added rows, in order"
        reason = {"fix_v1 row missing": "STOP (6): a fix_v1 row is missing from the pilot inventory", "fix_v1 row differs": differs,
                  "fix_v1 row with its members reordered": differs, "fix_v1 row with 7 as 7.0": differs,
                  "fix_v1 rows swapped": "STOP (6): fix_v1's keys are not the ordered prefix of the pilot inventory's",
                  "pilot rows file: one row differs": rows7, "pilot rows file: the rows in another order": rows7,
                  "K changed in the fix_v1 part, K's original among the added rows": "STOP (6): a key appears twice in the pilot inventory",
                  }.get(why, f"STOP (6): the pilot inventory's {why.removesuffix(' (sha256)')}")
        rec_pin = pin(rows_bad[why.removeprefix("pilot rows file: ")]) if why.startswith("pilot rows file: ") else \
            pin(rows_dupk) if why.startswith("K changed") else GOOD["INVENTORY_B_RECORD"]
        neg(f"(6) {why}", reason, INVENTORY_B=pin(inv_files[why]), INVENTORY_B_CHECK=pin(checks[why]), INVENTORY_B_RECORD=rec_pin)
    neg("(6) a repeated member name in the inventory (raw JSON)", "STOP (6): inventory: duplicate member 'world'",
        INVENTORY_B=pin(inv_files["a repeated member name"]), INVENTORY_B_CHECK=pin(checks["a repeated member name"]))
    neg("(6) the check record has rows_canonical_only 1 (a 7b060e888-era --reorder-explained record)",
        "STOP (6): the fix_v1 ⊂ fix_v2 check record does not pass on these two inventories", INVENTORY_B_CHECK=pin(check_canon1))
    neg("(6) the check record has rows_reserialized_identical != n_fix_v1", "STOP (6): the fix_v1 ⊂ fix_v2 check record does not pass on these two inventories",
        INVENTORY_B_CHECK=pin(check_reser2))
    for why, cp in check_bind.items():
        neg(f"(6) the check record's {why}", "STOP (6): the fix_v1 ⊂ fix_v2 check record does not pass on these two inventories", INVENTORY_B_CHECK=pin(cp))
    neg("(6) the check record does not pass", "STOP (6): the fix_v1 ⊂ fix_v2 check record does not pass on these two inventories", INVENTORY_B_CHECK=pin(check_false))
    neg("(6) the check record names another fix_v2", "STOP (6): the fix_v1 ⊂ fix_v2 check record does not pass on these two inventories",
        INVENTORY_B_CHECK=pin(check_other))
    neg("(6) the check record names another fix_v1", "STOP (6): the fix_v1 ⊂ fix_v2 check record does not pass on these two inventories",
        INVENTORY_B_CHECK=pin(check_other_v1))
    for k, v in GOOD.items():
        setattr(b, k, v)
    e = stops(b.main)
    res["build passes on the fabricated pilot records"] = e is None
    msg = json.loads((out / "msg_simh_q4f_v1_pilot_cutfull_r1.json").read_text())
    ad = json.loads(next(out.glob("adapter_simh_q4f_v1_pilot_cutfull_r1_*.json")).read_text())
    ep = msg["epochs"]
    res["message: 20 epochs, world sim_honest, folder = the pilot folder, original as is, tokens_before = full_len"] = \
        len(ep) == 20 and {x["world"] for x in ep} == {"sim_honest"} and {x["folder"] for x in ep} == {PF} and {x["original_folder"] for x in ep} == {PO} \
        and all(x["tokens_before"] == 1000 + KEYS.index(x["key"]) for x in ep)
    res["adapter: every gate passes; the step-0, counted-keys, inventory, pilot-rows, check-record and fix_v1 pins recorded"] = \
        all(g["pass"] for g in ad["gate"].values()) and {"step0_gate", "counted_keys", "inventory", "inventory_pilot_rows", "inventory_subset_check",
                                                         "inventory_fix_v1"} <= set(ad["pins"])
    # (5) at the pinned path and pinned by their own sha: the gate record with passes false, then with one run key missing
    good_step0 = (repo / step0).read_bytes()
    for name, src in (("the pinned gate record says passes false", step0_false), ("the pinned gate record says passes \"true\" (a string)", step0_str),
                      ("the pinned record's run keys != the counted keys", step0_missing)):
        (repo / step0).write_bytes((repo / src).read_bytes())
        cm = commit(repo, f"fabricated: {name}")
        put(repo, ATT, dict(att, repro_gate=f"{step0} @ {cm[:12]} (the pilot's own per-root step-0 gate)"))
        cx = commit(repo, f"fabricated attestation citing: {name}")
        sys.argv[2] = f"{cx}:{ATT}"
        before = sorted(out.glob("*"))
        b.PILOT_STEP0 = (cm[:12], step0, sha((repo / step0).read_bytes()))
        e = stops(b.main)
        res[f"NEGATIVE (5) {name}: STOP '{GATE_STOP}', nothing written"] = e == GATE_STOP and sorted(out.glob("*")) == before
        if e != GATE_STOP:
            print("   got:", e)
    (repo / step0).write_bytes(good_step0)
    b.PILOT_STEP0 = GOOD["PILOT_STEP0"]
    # Astra R1: the gate is chosen by the attestation's pilot identity, never by the record it cites. A pilot attestation citing the
    # legacy compare (a passing record, at the legacy pin) must still take the pilot gate and STOP.
    b.STEP0 = (c0[:12], legacy)
    for name, att_x, want in (("Astra R1: a pilot attestation cites the legacy compare (passing, at the legacy pin)",
                               dict(att, repro_gate=f"{legacy} @ {c0[:12]} (the legacy compare)"), GATE_STOP),
                              ("strict booleans: the format record says passes \"true\" (a string)", dict(att, format_check={TOK: f"{fmt_str} @ {c0[:12]}"}),
                               'adapter gates failed: {"format": {"label": "format_check passed (d9 @ %s)", "pass": false}}' % c0[:9]),
                              *[(f"Q1: the pilot folder with {k}", v, Q1_STOP) for k, v in (
                                  ("b1's original folder", dict(att, original_folders={TOK: "simh_q4f_v1_full_b1__sim_honest"})),
                                  ("another original_target", dict(att, original_target="simh_q4f_v1_full_b1")),
                                  ("no original_target", {k_: v_ for k_, v_ in att.items() if k_ != "original_target"}))]):
        put(repo, ATT, att_x)
        cx = commit(repo, f"fabricated attestation: {name}")
        sys.argv[2] = f"{cx}:{ATT}"
        before = sorted(out.glob("*"))
        e = stops(b.main)
        res[f"NEGATIVE {name}: STOP '{want}', nothing written"] = e == want and sorted(out.glob("*")) == before
        if e != want:
            print("   got:", e)
    # the check on the built message (the attestation restored to the bytes the message was built from)
    put(repo, ATT, att)
    put(repo, ROUT, {"batch": "sold", "staged_set": SS, "extraction_root": ROOT, "folders": {TOK: PF}, "original_folders": {TOK: PO},
                     "attestation": {"path": f"{ATT} @ {c1[:12]}", "sha256": sha((repo / ATT).read_bytes()), "revision": 1},
                     "arms": [{"meta_arm": ARM, "keys": KEYS, "epochs": 20, "staged_sha": U}], "revision": 1})
    c3 = commit(repo, "fabricated routing record + the built message")
    c = load(repo / REL / "check_sold_equality.py", "check_copy")
    c.INVENTORY, c.COUNTED = GOOD["INVENTORY_B"], GOOD["COUNTED"]
    sys.argv = ["check_sold_equality.py", "--records-commit", c3]
    e_chk = stops(c.main)
    res["check passes on the built pilot message (exactly one adapter, the 20 counted keys)"] = e_chk is None
    if e_chk:
        print("  check stopped:", e_chk[:300])
    bsrc = (repo / REL / "build_cutfull_message.py").read_text()
    (repo / REL / "build_cutfull_message.py").write_text(bsrc.replace("INVENTORY_B = None", "INVENTORY_B = ('x', 'y', 'z')", 1))
    time.sleep(1.1)
    e = stops(c.main)
    res["NEGATIVE the build copy's inventory pin literal differs from the check copy's: STOP"] = e == "STOP: the build and check copies do not pin the same pilot inventory"
    for k, v in res.items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}")
    print("ALL PASS" if all(res.values()) else "FAILED")
    shutil.rmtree(repo)
    if not all(res.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tmp", default=None)
    ap.add_argument("--worktree", action="store_true", help="test the working-tree copies instead of the HEAD blobs")
    x = ap.parse_args()
    main(x.tmp, x.worktree)
