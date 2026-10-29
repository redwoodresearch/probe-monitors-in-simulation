"""Synthetic test of the pilot-only sold_equal copies. Fabricated keys and shas only, in a throwaway git repo; no Modal, no real
record read. What it checks:
- An unset pilot inventory pin STOPs, in both copies; a wrong sha256 STOPs. Every STOP is matched by its exact reason.
- The one-tuple rule: identical literals that are not None or a (commit, path, sha256) tuple of strings STOP (a 2-tuple, a list).
- Git failures in the check copy: the inventory, or the counted keys, pinned to a path not in git STOP with a reason (no traceback).
  Other git calls in the copies can still raise CalledProcessError (a traceback; still fail-closed).
- A wrong folder fails build_sold's original_folders gate. The gate's own expression is taken from the copy by ast.
- The check copy passes on exactly one pilot record, and STOPs on 0 records or 2. Each negative asserts the note's FULL mismatch list.
- The adapter's inventory pin binds on the pilot path only: a pilot adapter with no pin, or another pin, STOPs; an S_old adapter with no
  pin, or another inventory's pin (b1's real adapter pins inventory_b_v2 @ a0bf963a7), passes that gate.
- sold_world: the pilot's original gives "sim_honest". Every other name gives what step3's original gives: b1 is unchanged, and a near-miss
  pilot name is None.

The copies' bytes are this branch's HEAD blobs (--worktree: the working-tree files, to test before a commit).

  uv run --no-project --python 3.12 python experiments/2026-10-17_step2_pilot_sold_equal/test_sold_equal_copies.py [--tmp <dir>] [--worktree]
"""
import argparse
import ast
import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REAL = HERE.parents[1]
REL = "experiments/2026-10-17_step2_pilot_sold_equal"
ORIG = ("b681a880d", "experiments/2026-10-15_step3_retrain/build_cutfull_message.py")  # step3's original, for sold_world's other names
DIR = "experiments/2026-10-14_q3_overnight_chain/results/cutfull/routing"
ATT = "experiments/2026-10-14_q3_overnight_chain/results/cutfull/attest/attestations_sold__simh_q4f_v1_pilot_cutfull.json"
PF, PO, SS, ROOT, ARM, U = ("simh_q4f_v1_pilot_cutfull_v1__sim_honest", "simh_q4f_v1_pilot__sim_honest", "simh_pilot_cutfull_v1",
                            "simh_pilot_cutfull_extract_v1", "sim_honest@september", hashlib.sha256(b"staged").hexdigest())
KEYS = [f"2026-10-14_140858_simh_pilot/{ARM}/t{21 + i}_x/run{i}.eval#e0" for i in range(20)]


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


def git(repo, *a):
    return subprocess.run(["git", "-C", str(repo), *a], capture_output=True, text=True, check=True).stdout.strip()


def put(repo, rel, obj):
    p = repo / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=1))
    return rel


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


def main(tmp, worktree=False):
    repo = Path(tempfile.mkdtemp(prefix="sold_equal_test_", dir=tmp))
    git(repo, "init", "-q")
    for k, v in (("user.email", "test@example.invalid"), ("user.name", "test"), ("commit.gpgsign", "false")):
        git(repo, "config", k, v)
    real_src = {}
    for f in ("build_cutfull_message.py", "check_sold_equality.py"):
        (repo / REL).mkdir(parents=True, exist_ok=True)
        real_src[f] = (HERE / f).read_bytes() if worktree else \
            subprocess.run(["git", "-C", str(REAL), "show", f"HEAD:{REL}/{f}"], capture_output=True, check=True).stdout
        (repo / REL / f).write_bytes(unpin(f, real_src[f]))
    att_bytes = json.dumps({"fabricated": True}).encode()
    (repo / ATT).parent.mkdir(parents=True, exist_ok=True)
    (repo / ATT).write_bytes(att_bytes)
    rec = {"batch": "sold", "staged_set": SS, "extraction_root": ROOT, "folders": {"sim_honest": PF}, "original_folders": {"sim_honest": PO},
           "attestation": {"path": f"{ATT} @ 0123456789ab", "sha256": hashlib.sha256(att_bytes).hexdigest(), "revision": 1},
           "arms": [{"meta_arm": ARM, "keys": KEYS, "epochs": len(KEYS), "staged_sha": U}], "revision": 1}
    put(repo, f"{DIR}/sold__simh_q4f_v1_pilot_cutfull.json", rec)
    B1O, B1KEYS = "simh_q4f_v1_full_b1__sim_honest", [f"2026-10-14_160932_simh_cont/{ARM}/t19{i}_x/run{i}.eval#e0" for i in range(3)]  # an S_old (b1-like) record
    ATT_B1 = ATT.replace("simh_q4f_v1_pilot_cutfull", "simh_q4f_v1_full_b1_cutfull_r2")
    put(repo, f"{DIR}/sold__simh_q4f_v1_full_b1_cutfull_r2.json", rec | {"original_folders": {"sim_honest": B1O}, "arms": [{"meta_arm": ARM, "keys": B1KEYS, "epochs": 3, "staged_sha": U}],
                                                                       "attestation": dict(rec["attestation"], path=f"{ATT_B1} @ 0123456789ab")})
    inv = put(repo, "fab/inventory_pilot.json", {"runs": [{"key": k, "original_folder": PO, "original_folder_registered": True, "world": "sim_honest"} for k in KEYS]
                                                 + [{"key": k, "original_folder": B1O, "original_folder_registered": True, "world": "sim_honest"} for k in B1KEYS]})
    msg = put(repo, f"{REL}/results/cutfull_b_msgs/msg_simh_q4f_v1_pilot_cutfull_r1.json",
              {"batch": "sold", "provenance": [{"staged_set": SS, "run": ROOT}],
               "epochs": [{"key": k, "arm": ARM, "staged_sha256": U, "folder": PF, "original_folder": PO, "world": "sim_honest", "tokens_before": 100} for k in KEYS]})
    msg19 = put(repo, f"{REL}/results/cutfull_b_msgs/msg_pilot_19keys.json", json.loads((repo / msg).read_text()) | {"epochs": json.loads((repo / msg).read_text())["epochs"][1:]})
    msg_b1 = put(repo, f"{REL}/results/cutfull_b_msgs/msg_simh_q4f_v1_full_b1_cutfull_r2.json",
                 {"batch": "sold", "provenance": [{"staged_set": SS, "run": ROOT}],
                  "epochs": [{"key": k, "arm": ARM, "staged_sha256": U, "folder": PF, "original_folder": B1O, "world": "sim_honest", "tokens_before": 9} for k in B1KEYS]})
    counted = put(repo, "fab/counted_keys.json", {"keys": [{"volume_path": "simh_evals_v1/" + k.removesuffix("#e0")} for k in KEYS]})
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "fabricated records")
    rc = git(repo, "rev-parse", "HEAD")
    pin = (rc[:12], inv, hashlib.sha256((repo / inv).read_bytes()).hexdigest())
    inv_pin = {"path": inv, "sha256": pin[2]}
    adapter = {"pins": {"routing_record": {"path": ATT, "sha256": hashlib.sha256(att_bytes).hexdigest()}, "inventory": inv_pin}, "message_path": msg}
    adapter_b1 = {"pins": {"routing_record": {"path": ATT_B1, "sha256": hashlib.sha256(att_bytes).hexdigest()}, "inventory": inv_pin}, "message_path": msg_b1}
    adir = repo / REL / "results" / "cutfull_b_msgs"
    res = {}

    # 0. (a2) the REAL copies' pins: build INVENTORY_B and check INVENTORY are the same fix_v2 tuple (the one-tuple rule), build
    # INVENTORY_B_RECORD is the pilot-rows file, and INVENTORY_B_CHECK is run 1's committed record (26c12ab14)
    names = {"build_cutfull_message.py": ("INVENTORY_B", "INVENTORY_B_RECORD", "INVENTORY_B_CHECK"), "check_sold_equality.py": ("INVENTORY",)}
    lits = {f: {x.targets[0].id: ast.literal_eval(x.value) for x in ast.parse(real_src[f].decode()).body
                if isinstance(x, ast.Assign) and getattr(x.targets[0], "id", "") in names[f]} for f in real_src}
    res["(a2) the real copies pin INVENTORY_B = INVENTORY = fix_v2, INVENTORY_B_RECORD = the pilot rows, INVENTORY_B_CHECK = run 1's record"] = \
        lits == {"build_cutfull_message.py": {"INVENTORY_B": FIXV2, "INVENTORY_B_RECORD": PROWS, "INVENTORY_B_CHECK": CHK},
                 "check_sold_equality.py": {"INVENTORY": FIXV2}}

    # 1. an unset pin STOPs, in both copies
    b = load(repo / REL / "build_cutfull_message.py", "build_copy")
    sys.argv = ["x", "--record", f"{rc}:{ATT}", "--sold", "--scope", "b", "--out-dir", str(repo / "out")]
    res["build: unset inventory pin STOPs '(b): inventory_b_v2 is not pinned yet (4f relays it)'"] = \
        stops(b.main) == "(b): inventory_b_v2 is not pinned yet (4f relays it)" and b.INVENTORY_B is None
    c = load(repo / REL / "check_sold_equality.py", "check_copy")
    sys.argv = ["x", "--records-commit", rc]
    res["check: unset inventory pin STOPs 'STOP: the pilot inventory pin is unset'"] = stops(c.main) == "STOP: the pilot inventory pin is unset" and c.INVENTORY is None
    c.INVENTORY = (pin[0], pin[1], "0" * 64)
    res["check: a wrong inventory sha256 STOPs 'STOP: the pilot inventory differs from its pin (sha256)'"] = \
        stops(c.main) == "STOP: the pilot inventory differs from its pin (sha256)"
    c.INVENTORY = (pin[0], "fab/not_there.json", pin[2])
    res["check, git failure: the inventory pinned to a path not in git STOPs 'STOP: the pilot inventory is not in git at its pin'"] = \
        stops(c.main) == "STOP: the pilot inventory is not in git at its pin"
    lit_stop = "STOP: the build and check copies do not pin the same pilot inventory"
    srcs = {f: (repo / REL / f).read_text() for f in ("build_cutfull_message.py", "check_sold_equality.py")}
    for name, lit in (("a 2-tuple", '("a", "b")'), ("a list", '["a", "b", "c"]'), ("a tuple with a non-string", '("a", "b", 3)')):
        for f, n in (("build_cutfull_message.py", "INVENTORY_B = None"), ("check_sold_equality.py", "INVENTORY = None")):
            (repo / REL / f).write_text(srcs[f].replace(n, n.replace("None", lit), 1))
        res[f"check, one-tuple rule: identical literals that are {name} STOP '{lit_stop}'"] = stops(c.main) == lit_stop
    for f, t in srcs.items():
        (repo / REL / f).write_text(t)

    # 2. a wrong folder: build_sold's original_folders gate, its own expression (ast, from the copy)
    src = (repo / REL / "build_cutfull_message.py").read_text()
    f = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "build_sold")
    node = [n for n in ast.walk(f) if isinstance(n, ast.Assign) and 'gate["original_folders"]' in ast.get_source_segment(src, n)]
    gate = compile(ast.Expression(node[0].value), "build_sold", "eval")
    env = lambda fo: vars(b) | {"folder": {ARM: fo}, "metas": [ARM], "tok": {ARM: "sim_honest"}, "of": {"sim_honest": PO}, "cf": {"staged_set": SS}}  # noqa: E731
    res["build: the pilot folder passes the folder gate"] = len(node) == 1 and eval(gate, env(PF))["pass"] is True
    res["build: a wrong folder (derived from the staged set) fails it"] = eval(gate, env(f"{SS}__sim_honest"))["pass"] is False
    res["build: b1's folder fails it (pilot only)"] = eval(gate, env("simh_q4f_v1_full_b1_cutfull_v1__sim_honest"))["pass"] is False

    # 3. the check (09 (i): the widened glob with an EXACT-SET filter on both paths): each case asserts its exact reason in the note
    c.INVENTORY, c.COUNTED = pin, (rc[:12], counted, hashlib.sha256((repo / counted).read_bytes()).hexdigest())
    PR, BR = "sold__simh_q4f_v1_pilot_cutfull.json", "sold__simh_q4f_v1_full_b1_cutfull_r2.json"

    def run(adapters):  # [(file name, adapter)] -> (STOP message or None, the note's mismatches)
        for p in adir.glob("adapter_*.json"):
            p.unlink()
        for fname, ad in adapters:
            (adir / fname).write_text(json.dumps(ad))
        time.sleep(1.1)  # the note's name has a seconds stamp (open "x")
        e = stops(c.main)
        notes = sorted((repo / REL / "results").glob("cutfull_sold_equality_note_*.json"))
        return e, (json.loads(notes[-1].read_text())["mismatches"] if notes else None)

    P1, P2 = "adapter_simh_q4f_v1_pilot_cutfull_r1_2026-10-17_031500.json", "adapter_simh_q4f_v1_pilot_cutfull_r1_2026-10-17_031501.json"
    e, mm = run([(P1, adapter)])
    res["check, pilot path: exactly one adapter with the 20 counted keys passes"] = e is None and mm == []
    good_counted = c.COUNTED
    c.COUNTED = (rc[:12], "fab/not_there.json", good_counted[2])
    e, mm = run([(P1, adapter)])
    res["check, git failure: the counted keys pinned to a path not in git STOP 'STOP: the counted pilot keys are not in git at their pin'"] = \
        e == "STOP: the counted pilot keys are not in git at their pin"
    c.COUNTED = good_counted
    o2o = "the adapter records do not map one-to-one onto d9's standard S_old records: "  # the check copy's one-to-one message
    CHECK_STOP = "STOP: a mismatch between the --sold read and d9's standard S_old records"  # the check copy's exit (check_sold_equality.py L131)
    for name, ads, want in (("0 adapters", [], [f"{PR}: 0 adapters (exactly one required)", f"{o2o}[]"]),
                            ("2 adapters (a duplicate)", [(P1, adapter), (P2, adapter)],
                             [f"{PR}: 2 adapters (exactly one required)", f"{o2o}{[f'{DIR}/{PR}', f'{DIR}/{PR}']}"]),
                            ("the adapter's epochs miss one counted key", [(P1, dict(adapter, message_path=msg19))],
                             [f"{PR}: sim_honest: keys", f"{PR}: the adapter's epochs are not exactly the 20 counted pilot run keys"]),
                            ("an extra adapter for another record", [(P1, adapter), ("adapter_simh_q4f_v1_full_b1_cutfull_r2_2026-10-16_000000.json", adapter_b1)],
                             [f"{BR}: A25: original as is", f"{BR}: an adapter for a record outside REQUIRED", f"{o2o}{[f'{DIR}/{BR}', f'{DIR}/{PR}']}"]),
                            ("the adapter's inventory pin is another inventory", [(P1, dict(adapter, pins=dict(adapter["pins"], inventory={"path": inv, "sha256": "0" * 64})))],
                             [f"{PR}: the adapter's inventory pin is not this check's INVENTORY"]),
                            ("the adapter has no inventory pin", [(P1, dict(adapter, pins={k: v for k, v in adapter["pins"].items() if k != "inventory"}))],
                             [f"{PR}: the adapter's inventory pin is not this check's INVENTORY"]),
                            ("an extra adapter of another kind (not sold)", [(P1, adapter), ("adapter_other_kind_r1_2026-10-17_000000.json",
                             dict(adapter, pins=dict(adapter["pins"], routing_record={"path": f"{DIR}/sold__other_cutfull_attest.json", "sha256": "0" * 64})))],
                             ["adapter_other_kind_r1_2026-10-17_000000.json: not an S_old/pilot (sold) adapter"]),
                            ("an adapter whose routing record is absent at the records commit", [(P1, adapter), ("adapter_ghost_r1_2026-10-17_000000.json",
                             dict(adapter, pins=dict(adapter["pins"], routing_record={"path": ATT.replace("simh_q4f_v1_pilot_cutfull", "ghost_cutfull"), "sha256": "0" * 64})))],
                             [f"sold__ghost_cutfull.json: no routing record {DIR}/sold__ghost_cutfull.json at the records commit",
                              "sold__ghost_cutfull.json: an adapter for a record outside REQUIRED"])):
        e, mm = run(ads)
        res[f"check, pilot path, NEGATIVE {name}: STOP '{CHECK_STOP}' with exactly {want}"] = e == CHECK_STOP and mm == want
        if mm != want:
            print("   got:", mm)
    bad_folder = dict(adapter, message_path=put(repo, f"{REL}/results/cutfull_b_msgs/msg_wrong_folder.json",
                                                json.loads((repo / msg).read_text()) | {"epochs": [e | {"folder": f"{SS}__sim_honest"} for e in json.loads((repo / msg).read_text())["epochs"]]}))
    e, mm = run([(P1, bad_folder)])
    res["check, pilot path, NEGATIVE a message with the wrong folder: STOP with exactly ['<record>: sim_honest: folder']"] = e == CHECK_STOP and mm == [f"{PR}: sim_honest: folder"]
    c.REQUIRED = {BR: B1O}  # the S_old path (as the b1 control runs it): its adapter set = what the 10-16 glob matched
    S1, S2 = "adapter_simh_q4f_v1_full_b1_cutfull_r2_2026-10-16_023636.json", "adapter_simh_q4f_v1_full_b1_cutfull_r2_2026-10-16_023637.json"
    e, mm = run([(S1, adapter_b1)])
    res["check, S_old path: exactly its one 10-16 adapter passes"] = e is None and mm == []
    if mm:
        print("   S_old path mismatches:", mm)
    for name, ad_ in (("with no inventory pin", dict(adapter_b1, pins={k: v for k, v in adapter_b1["pins"].items() if k != "inventory"})),
                      ("pinning another inventory (as b1's real adapter pins inventory_b_v2)",
                       dict(adapter_b1, pins=dict(adapter_b1["pins"], inventory={"path": "experiments/2026-10-15_cut_sensitivity/results/inventory_b_v2.json", "sha256": "0" * 64})))):
        e, mm = run([(S1, ad_)])
        res[f"check, S_old path: its one 10-16 adapter {name} passes (the pin binds on the pilot path only)"] = e is None and mm == []
        if mm:
            print("   got:", mm)
    for name, ads, want in (("0 adapters", [], [f"{BR}: 0 adapters (exactly one required)", f"{o2o}[]"]),
                            ("2 adapters (an extra)", [(S1, adapter_b1), (S2, adapter_b1)],
                             [f"{BR}: 2 adapters (exactly one required)", f"{o2o}{[f'{DIR}/{BR}', f'{DIR}/{BR}']}"]),
                            ("an adapter outside the 10-16 set", [("adapter_simh_q4f_v1_full_b1_cutfull_r2_2026-10-17_000000.json", adapter_b1)],
                             [f"{BR}: the adapter is not in the S_old 10-16 adapter set"])):
        e, mm = run(ads)
        res[f"check, S_old path, NEGATIVE {name}: STOP '{CHECK_STOP}' with exactly {want}"] = e == CHECK_STOP and mm == want
        if mm != want:
            print("   got:", mm)

    # 4. sold_world: the pilot's original -> sim_honest; every other name as step3's original
    o = {"re": __import__("re")}
    exec(compile(ast.Module([n for n in ast.parse(subprocess.run(["git", "-C", str(REAL), "show", f"{ORIG[0]}:{ORIG[1]}"], capture_output=True, text=True,
                                                                  check=True).stdout).body
                             if isinstance(n, ast.Assign) and {getattr(t, "id", "") for t in n.targets} & {"SOLD_RE", "SIMH_Q4F_RE", "SOLD_WORLD"}
                             or isinstance(n, ast.FunctionDef) and n.name == "sold_world"], []), "orig", "exec"), o)
    names = ["simh_q4f_v1_full_b1__sim_honest", "simh_q4f_v1_full_b2__sim_honest", "q4f_c_v1__honest_b", "rh1e4_stage_v1__honest_b", "q4f_main_v1__attack",
             "simh_q4f_v1_pilot2__sim_honest", "simh_q4f_v1_pilot__sim_honestx", "xsimh_q4f_v1_pilot__sim_honest", "simh_pilot__sim_honest"]
    res["sold_world: the pilot's original -> sim_honest"] = b.sold_world(PO) == "sim_honest"
    res["sold_world: every other name as step3's original (b1 unchanged, near-misses None)"] = all(b.sold_world(n) == o["sold_world"](n) for n in names) \
        and b.sold_world("simh_q4f_v1_full_b1__sim_honest") == "sim_honest" and b.sold_world("simh_q4f_v1_pilot2__sim_honest") is None
    for k, v in res.items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}")
    print("ALL PASS" if all(res.values()) else "FAILED")
    __import__("shutil").rmtree(repo)
    if not all(res.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tmp", default=None)
    ap.add_argument("--worktree", action="store_true", help="test the working-tree copies instead of the HEAD blobs")
    x = ap.parse_args()
    main(x.tmp, x.worktree)
