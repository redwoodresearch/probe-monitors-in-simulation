"""Synthetic test of make_pilot_records.py: fabricated records only (keys, shas and counts made up here) in a throwaway git repo; no Modal,
no real record read. The builder's three phases run there, committing between them as in use; then each scorer's pilot checks run on the
outputs. The check code is the scorer's own, taken from its blob at a pinned commit of this repo's object store (ast: whole functions, or
the test of the one If / the value of the one Assign that holds an anchor string), so a changed scorer line changes what runs. Negative
cases (a wrong sha, a missing key, a wrong folder name, another Volume stage record) must write nothing. A publish that fails at
fsync or at os.link must STOP with its exact message and leave neither the output nor a temp file; an existing output, or a name another writer creates during
the publish, must STOP and stay untouched (write-once). d9's routing
writer is copied from its pin (8efb04c58) and checked against its ORIGINAL sha256 0184ffaa...; only the pin's commit is rebound to
the throwaway repo, never its digest. Phase 3 (d9's router, run unchanged): a leftover router temp STOPs before the router runs; an
empty, a truncated, or a revision-2 published record STOPs after it as "published but invalid".

  uv run --no-project --python 3.12 python experiments/2026-10-14_q3_overnight_chain/cutfull/test_make_pilot_records.py [--tmp <dir>]
"""
import argparse
import ast
import hashlib
import importlib.util
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
REAL = HERE.parents[2]
REL = "experiments/2026-10-14_q3_overnight_chain"
# scorer files pinned by git blob id (harness-only commits on the scorer branches leave these bytes unchanged); the commit is where the blob was read
SCORERS = {"sol": ("c7434549e21e85bd4cd4b6d1facf2e0bdb83340e", "9cea48891", "experiments/2026-10-14_q35_sol_avg_POSTHOC/run_sold_cutfull_score.py"),
           "p2": ("f255cde51181e21811f9334bd634e156a3b1944f", "d26947c96", "experiments/2026-10-14_p2_simh_scores/run_simh_scores.py"),
           "own_all": ("a6208d60d2f9a4cb8a97b9407ca8ed8e1a68195c", "0bf04ca10", "experiments/2026-10-14_p2_family_expansion/run_own_all.py"),  # loo
           "own_all_simt": ("1efae1b971af0152749983d6c9d1950aec5ab045", "10a7bdf2b", "experiments/2026-10-14_p2_family_expansion/run_own_all.py"),
           "own_all_loo3": ("4e587046c00b4b4a4eec282b56e26a0f793dbc8e", "40f9142a8", "experiments/2026-10-14_p2_family_expansion/run_own_all.py"),
           "famexp": ("920e84281d1c5c654eb2447277e7258ce60ca562", "e46920fc3", "experiments/2026-10-14_p2_family_expansion/run_famexp_cutfull.py"),
           "famexp_cutfull": ("905900061a232aa44805863d7ea9bcedef1f54cd", "e46920fc3", "experiments/2026-10-14_p2_family_expansion/cutfull.py")}
ROUTER_ORIGINAL = ("experiments/2026-10-14_q3_overnight_chain/cutfull/make_sold_routing.py", "8efb04c58",
                   "0184ffaa1c0ab61e42dcb6a6206b95eae39616081c51b67d9cf1cddadbd59c40")  # d9's routing writer, as pinned in the builder
sha = lambda b: hashlib.sha256(b).hexdigest()  # noqa: E731
fake = lambda s: sha(s.encode())  # noqa: E731
N, ARM, AF, TOK = 20, "sim_honest@september", "sim_honest@september.jsonl", "sim_honest"
LAUNCH = "2026-10-14_140858_simh_pilot"
RELS = [f"{ARM}/t{21 + i}_task{i}/run{i}.eval" for i in range(N)]
KEYS = sorted(f"{LAUNCH}/{r}#e0" for r in RELS)
TASK = {f"{LAUNCH}/{r}#e0": f"t{21 + i}_task{i}" for i, r in enumerate(RELS)}
EVAL = {k: fake(f"eval {k}") for k in KEYS}
REG_SHA, U_SHA, VSTAGE = fake("registered staged file"), fake("full-length staged file"), fake("volume stage record")
CHANGED = [KEYS[-1]]


def git(repo, *a, text=True):
    return subprocess.run(["git", "-C", str(repo), *a], capture_output=True, text=text, check=True).stdout


def commit(repo, msg):
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", msg)
    return git(repo, "rev-parse", "HEAD").strip()


def put(repo, rel, obj):
    p = repo / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(obj if isinstance(obj, bytes) else json.dumps(obj, indent=1).encode())
    return rel


def outputs(repo):
    return sorted(str(p) for p in (repo / REL / "results/cutfull").rglob("*") if p.is_file())


def refuses(repo, fn, why, want=None):
    """fn must STOP and write nothing; with want (a regex), the STOP message must fullmatch it."""
    before = outputs(repo)
    try:
        fn()
    except SystemExit as e:
        assert outputs(repo) == before and not git(repo, "status", "--porcelain").strip(), f"{why}: wrote something"
        assert want is None or re.fullmatch(want, str(e)), f"{why}: STOP message {str(e)!r} is not {want!r}"
        print(f"  NEGATIVE ok ({why}): {str(e)[:110]}")
        return
    raise AssertionError(f"{why}: did not refuse")


def load(name, names, ns):
    """Exec the scorer's top-level functions / assignments named in `names` (file order) into ns."""
    bl, c, p = SCORERS[name]
    src = git(REAL, "cat-file", "blob", bl)
    tree, got = ast.parse(src), set()
    for n in tree.body:
        tn = {n.name} if isinstance(n, ast.FunctionDef) else {x.id for t in n.targets for x in (t.elts if isinstance(t, ast.Tuple) else [t])
                                                               if isinstance(x, ast.Name)} if isinstance(n, ast.Assign) else set()
        if tn & names:
            exec(compile(ast.Module([n], []), f"{p}@{c} (blob {bl[:12]})", "exec"), ns)
            got |= tn & names
    assert got == names, (name, names - got)
    return src, tree


def piece(src, tree, func, anchor, kind="if"):
    """The test of the one If (or the value of the one Assign) in `func` whose source holds `anchor` -> (code, line)."""
    f = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == func)
    hits = [n for n in ast.walk(f) if (kind == "if" and isinstance(n, ast.If) and anchor in ast.get_source_segment(src, n.test))
            or (kind == "assign" and isinstance(n, ast.Assign) and anchor in ast.get_source_segment(src, n))]
    assert len(hits) == 1, (func, anchor, len(hits))
    return compile(ast.Expression(hits[0].test if kind == "if" else hits[0].value), f"{func}:L{hits[0].lineno}", "eval"), hits[0].lineno


def build(base):
    """A throwaway repo with the builder, d9's routing builder and fabricated committed inputs at the pinned paths."""
    repo = Path(tempfile.mkdtemp(prefix="pilot_records_test_", dir=base))
    git(repo, "init", "-q")
    for k, v in (("user.email", "test@example.invalid"), ("user.name", "test"), ("commit.gpgsign", "false")):
        git(repo, "config", k, v)
    put(repo, f"{REL}/cutfull/make_pilot_records.py", (HERE / "make_pilot_records.py").read_bytes())
    rb = git(REAL, "show", f"{ROUTER_ORIGINAL[1]}:{ROUTER_ORIGINAL[0]}", text=False)  # R2: d9's bytes at the pin, checked against the ORIGINAL digest
    assert sha(rb) == ROUTER_ORIGINAL[2], "d9's routing writer at 8efb04c58 is not the original 0184ffaa..."
    assert sha((REAL / ROUTER_ORIGINAL[0]).read_bytes()) == ROUTER_ORIGINAL[2], "this repo's working-tree make_sold_routing.py is not 0184ffaa..."
    put(repo, ROUTER_ORIGINAL[0], rb)
    S = "experiments/2026-10-14_q3_q35_simhonest_purchase"
    put(repo, f"{S}/run_simh_stage_c.py", b"# fabricated stager\n")
    put(repo, f"{S}/stage_simh.py", b"# fabricated stager module\n")
    g0 = commit(repo, "fabricated stager")
    spec = importlib.util.spec_from_file_location(f"mpr_{repo.name}", repo / REL / "cutfull/make_pilot_records.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    P = m.P
    assert m.ROUTER == ROUTER_ORIGINAL, "the builder's ROUTER pin is not d9's original (path, 8efb04c58, 0184ffaa...)"
    cg = {"manifest_keys": N, "staged": N, "excluded": {}, "render_changed": CHANGED}
    run = lambda k, **kw: {"key": k, "arm": ARM, "passes": True, "checks": {"binding": {"cut": True, "uncut": True, "cut_file_sha": REG_SHA, "uncut_file_sha": U_SHA},  # noqa: E731
                                                                            "pilot_cut_stage_record": {"files": ["stage_2026-10-14_165444.json"], "sha256": VSTAGE}}} | kw
    f = {"COUNTED": {"n": N, "keys": [{"task": TASK[k], "launch": LAUNCH, "eval_sha256": EVAL[k], "volume_path": "simh_evals_v1/" + k.removesuffix("#e0"),
                                       "world": "simh_pilot"} for k in KEYS]},
         "REG": {"git": g0, "simh": {"out": P["original_staged_set"], "assembled": {"epochs": {ARM: N}, "staged_inputs_sha256": {AF: REG_SHA}}, "c_gate": cg}},
         "REG_X": {"config": {"run": P["registered_root"], "inputs": P["original_staged_set"], "inputs_sha256": {AF: REG_SHA}, "test": False}},
         "STAGE": {"git": "0" * 40, "simh": {"out": P["staged_set"], "assembled": {"epochs": {ARM: N}, "staged_inputs_sha256": {AF: U_SHA},
                                                                                     "format_gate": {ARM: {"passes": True}}}, "c_gate": cg}},
         "IDENT": {"roots": {"head": P["staged_set"]}, "arm": ARM, "files": {"head": U_SHA}, "stage_records": {"head": ["stage_2026-10-17_010154.json"]}, "fail": [],
                   "keys": {k: {"row_equal": True} for k in KEYS}, "n_keys_identical": N, "identical": True, "pinned_keys": KEYS},
         "FMT": {"target": P["cutfull_target"], "arm": ARM, "staged_sha": [U_SHA], "epochs_checked": N, "failing_keys": [], "rows": [{"key": k} for k in KEYS], "passes": True},
         "XREC": {"config": {"run": P["root"], "inputs": P["staged_set"], "inputs_sha256": {AF: U_SHA}, "cut": "full", "attempt": 1, "test": False},
                  "volume": m.VOLUME, "attempt": 1, "shards": [{"shard": 0, "gpu_name": "NVIDIA H200", "chunks": [{"chunk": "shard000_a1_chunk000", "rows": 9}]}]}}
    extra = {"step0": {"passes": True, "n_runs": N, "n_pass": N, "runs": [run(k) for k in KEYS]},
             "step0_missing": {"passes": True, "n_runs": N - 1, "n_pass": N - 1, "runs": [run(k) for k in KEYS[1:]]},
             "metas": {"roots": [{"root": P["root"], "uncut_set": P["staged_set"], "epochs": N, "stage_record": "stage_2026-10-17_010154.json", "failing": {},
                                  "arms": {TOK: {"meta_arm": ARM, "epochs": N, "units": 300, "units_past_32768": 7, "staged_sha256": U_SHA, "keys": KEYS}}}]}}
    for n, obj in f.items():
        put(repo, getattr(m, n)[0], obj)
    put(repo, f"{REL}/results/cutfull/routing/sold__simh_q4f_v1_full_b1_cutfull_r2.json", {"batch": "sold", "staged_set": "simh_q4f_v1_full_b1_cutfull_v1_try2",
        "extraction_root": "simh_q4f_v1_full_b1_cutfull_extract_v1_try2", "arms": [{"meta_arm": ARM}], "revision": 2})  # another root's active record (S_old b1)
    paths = {n: put(repo, f"{REL}/results/fab_{n}.json", obj) for n, obj in extra.items()}
    cA = commit(repo, "fabricated inputs")
    for n in f:
        setattr(m, n, (getattr(m, n)[0], cA[:9], sha((repo / getattr(m, n)[0]).read_bytes())))
    m.ROUTER = (m.ROUTER[0], cA[:9], ROUTER_ORIGINAL[2])  # R2: rebind only the commit; the digest stays the original
    pins = {n: f"{p}@{cA[:12]}={sha((repo / p).read_bytes())}" for n, p in paths.items()}
    return repo, m, cA, pins


def main(base):
    repo, m, cA, pins = build(base)
    P, R = m.P, lambda p: repo / p  # noqa: E731
    print("throwaway repo", repo)
    good_reg = m.REG
    m.REG = (good_reg[0], good_reg[1], "0" * 64)
    refuses(repo, lambda: m.main(["provenance", "--cut-stage-sha", VSTAGE]), "wrong sha: the registered record's pin")
    m.REG = good_reg
    for step in ("fsync", "link"):  # R1: a publish failing at fsync or at os.link leaves neither the output nor a temp file
        real = getattr(m.os, step)
        def boom(*a, _step=step):
            raise OSError(f"simulated {_step} failure")
        setattr(m.os, step, boom)
        try:
            refuses(repo, lambda: m.main(["provenance", "--cut-stage-sha", VSTAGE]), f"publish fails at os.{step}: no output, no temp file",
                    rf"stager_provenance_simh_pilot_[0-9]{{4}}-[0-9]{{2}}-[0-9]{{2}}\.json: publish failed \(OSError: simulated {step} failure\); nothing written")
        finally:
            setattr(m.os, step, real)
        assert not [q for q in (repo / REL / "results").rglob("*") if q.name.endswith(".tmp")], f"os.{step} failure left a temp file"
    wo, real_link = repo.parent / f"{repo.name}_write_once", m.os.link  # R1: another writer creates the name between the check and the link
    wo.mkdir()
    def racing(src, dst):
        Path(dst).write_text("another writer")
        return real_link(src, dst)
    m.os.link = racing
    try:
        m.write_once(wo / "x.json", {"a": 1})
        raise AssertionError("write_once overwrote a name created during the publish")
    except SystemExit as e:
        assert str(e) == "x.json: exists; nothing written" and (wo / "x.json").read_text() == "another writer" and [q.name for q in wo.iterdir()] == ["x.json"], e
        print(f"  NEGATIVE ok (a name created between the exists() check and os.link: not overwritten, no temp file): {e}")
    finally:
        m.os.link = real_link
        shutil.rmtree(wo)
    m.main(["provenance", "--cut-stage-sha", VSTAGE])
    prov = next(Path(p) for p in outputs(repo) if "stager_provenance_simh_pilot_" in p).relative_to(repo)
    cP = commit(repo, "provenance")
    refuses(repo, lambda: m.main(["provenance", "--cut-stage-sha", VSTAGE]), "an existing output: write-once")
    att_args = lambda **kw: ["attest", "--provenance", f"{cP[:9]}:{prov}", "--step0", kw.get("step0", pins["step0"]), "--attest-metas", pins["metas"],  # noqa: E731
                             "--cut-stage-sha", kw.get("vstage", VSTAGE)]
    refuses(repo, lambda: m.main(att_args(step0=pins["step0"][:-64] + "f" * 64)), "wrong sha: the step-0 pin")
    refuses(repo, lambda: m.main(att_args(step0=pins["step0_missing"])), "missing key: the step-0 record lacks one staged key")
    refuses(repo, lambda: m.main(att_args(vstage=fake("another stage record"))), "another Volume stage record than the provenance's")
    P["folder"] = "simh_pilot_cutfull_v1__sim_honest"
    refuses(repo, lambda: m.main(att_args()), "wrong folder name: the old provisional planned folder")
    P["folder"] = "simh_q4f_v1_pilot_cutfull_v1__sim_honest"
    m.main(att_args())
    a_path = f"{REL}/results/cutfull/attest/attestations_sold__{P['original_target']}_cutfull.json"
    cAt = commit(repo, "attestation")
    router, before = R(m.ROUTER[0]), outputs(repo)  # 507-prep R1: a mutated working-tree copy of d9's routing writer STOPs, nothing written
    good_router = router.read_bytes()
    router.write_bytes(good_router + b"# mutated\n")
    try:
        m.main(["routing"])
        raise AssertionError("a mutated make_sold_routing.py did not STOP")
    except SystemExit as e:
        router.write_bytes(good_router)
        assert outputs(repo) == before and not git(repo, "status", "--porcelain").strip(), "the mutated-router case wrote something"
        print(f"  NEGATIVE ok (mutated working-tree make_sold_routing.py): {str(e)[:110]}")
    # phase 3 runs d9's router unchanged, which neither closes nor fsyncs its temp before linking: a leftover temp STOPs before it runs,
    # and an empty, truncated or wrong published record STOPs after it (the test then moves the bad record aside, as an operator would)
    r_path = f"{REL}/results/cutfull/routing/sold__{P['original_target']}_cutfull.json"
    stale = R(f"{REL}/results/cutfull/routing/.sold__{P['original_target']}_cutfull.json.tmp")
    stale.write_bytes(b"")
    before, st0 = outputs(repo), git(repo, "status", "--porcelain")  # the stale temp itself is untracked: compare against it, not "clean"
    want = (f"leftover router temp file(s) ['{stale.name}'] in {REL}/results/cutfull/routing from a failed earlier publish: move aside "
            "(results are append-only); nothing written")
    try:
        m.main(["routing"])
        raise AssertionError("a leftover router temp file did not STOP")
    except SystemExit as e:
        assert str(e) == want and outputs(repo) == before and git(repo, "status", "--porcelain") == st0, ("a leftover router temp file", str(e))
        print(f"  NEGATIVE ok (a leftover router temp file): {e}")
    stale.unlink()
    real_dumps = json.dumps
    for why, fake_dumps in (("an empty published record (json.dumps -> '' in the router)", lambda o, **kw: ""),
                            ("a truncated published record", lambda o, **kw: real_dumps(o, **kw)[: len(real_dumps(o, **kw)) // 2]),
                            ("a published record with revision 2", lambda o, **kw: real_dumps(dict(o, revision=2) if isinstance(o, dict) and "batch" in o else o, **kw))):
        json.dumps = fake_dumps
        try:
            m.main(["routing"])
            raise AssertionError(f"{why}: did not STOP")
        except SystemExit as e:
            json.dumps = real_dumps
            assert str(e) == f"{r_path}: published but invalid (move aside; results are append-only); STOP", (why, str(e))
            assert not [q for q in R(f"{REL}/results/cutfull/routing").iterdir() if q.name.endswith(".tmp")], f"{why}: a temp file is left"
            print(f"  NEGATIVE ok ({why}): {e}")
        finally:
            json.dumps = real_dumps
        R(r_path).unlink()  # move aside (test only)
    m.main(["routing"])
    cR = commit(repo, "routing")
    att, rec = json.loads(R(a_path).read_bytes()), json.loads(R(r_path).read_bytes())
    assert att["failing"] == {} and att["revision"] == 1 and rec["revision"] == 1 and rec["attestation"]["sha256"] == sha(R(a_path).read_bytes())
    # the other forks' inputs, in the forms the scorers read (synthetic): the pilot inventory, T196 cells, the 32k Sol pack record, r36
    inv_rows = [{"key": k, "original_folder": P["original_folder"], "original_folder_registered": True, "arm": ARM, "world": "sim_honest", "task": TASK[k],
                 "cut": {"extraction_root": P["registered_root"], "staged_set": P["original_staged_set"]}, "tokens_before": 1000 + i, "full_len": 1000 + i,
                 "eval_sha256": EVAL[k], "uncut_planned": {"folder": P["folder"], "staged_set": P["staged_set"], "extraction_root": P["root"], "volume": m.VOLUME,
                 "staged_sha256": U_SHA, "routing": f"{a_path} @ {cAt[:12]}", "revision": 1, "staged_inputs_file": AF}} for i, k in enumerate(KEYS)]
    inv = put(repo, "fab/inventory_pilot.json", {"runs": inv_rows})
    t196 = put(repo, "fab/t196.json", {"cells": [{"key": TASK[k], "main": f"main_{TASK[k]}"} for k in KEYS]})
    solrec = put(repo, "fab/simh_score_pilot.json", {"dump": {"extract_run": P["registered_root"], "staged_run": P["original_staged_set"]}, "stage": "done",
                                                     "outputs": {"path": "probe-shift_aswin-surya_sub-convaudit_work_kill-2026-11-14:/simh/pilot/sol/", "manifest_sha256": fake("pack")}})
    cI = commit(repo, "other forks' inputs (synthetic)")
    git(repo, "update-ref", "refs/remotes/origin/aswin/q3-cut-option", cI)
    r36 = repo.parent / f"{repo.name}_r36.json"
    r36.write_text(json.dumps([{"main_task": f"main_{TASK[k]}", "fold": i % 5} for i, k in enumerate(KEYS)]))
    pin = lambda p, c: (p, c[:12], sha(git(repo, "show", f"{c}:{p}", text=False)))  # noqa: E731
    sh = lambda c, p: git(repo, "show", f"{c}:{p}", text=False)  # noqa: E731
    att_raw, rb = sh(cAt, a_path), sh(cR, r_path)
    S0 = m.cli_pin(pins["step0"])  # (path, commit, sha256)

    # Sol: admit() end to end (inventory, attestation, provenance, step-0 gate, staging identity, extraction attestation and records, keys gate) + routing_equal
    ns = {"json": json, "hashlib": hashlib, "re": re, "subprocess": subprocess, "Path": Path, "itertools": __import__("itertools")}
    load("sol", {"Q_REL", "CF_BRANCH", "CHAIN", "ADIR", "RDIR", "CFVOL", "OUTVOL", "MOUNT", "INVB", "PROV", "CF_GATE", "CF_KEYS", "CORRB", "Q4F_STAGE", "T196", "R36_PIN",
                 "Q4F_C_EXTRACT", "RH_COUNTED", "SOL_REC", "SETS", "PILOT", "git", "show", "full", "on_branch", "pinned", "pilot_pin", "cite", "ls", "highest", "admit",
                 "pilot_want", "pilot_keys_gate", "routing_equal", "active_roots", "_strings"}, ns)
    ns["HERE"] = repo
    ns["PILOT"] |= {"inventory": (inv, cI[:12], sha(sh(cI, inv))), "counted_keys": (m.COUNTED[0], cA[:12], m.COUNTED[2]),
                    "provenance": (str(prov), cP[:9], sha(sh(cP, str(prov)))), "attestation": (a_path, cAt[:12], sha(att_raw)), "routing": (r_path, cR[:12], sha(rb)),
                    "step0_gate": S0, "staging_identity": (m.IDENT[0], cA[:12], m.IDENT[2])}
    ns["T196"], ns["SOL_REC"], ns["R36_PIN"] = pin(t196, cI), {"simh_pilot": pin(solrec, cI)}, sha(r36.read_bytes())[:20]
    assert att["authority"]["provenance"] == f"{ns['PILOT']['provenance'][1]}:{ns['PILOT']['provenance'][0]}"
    plan = ns["admit"](P["folder"], str(r36), f"{r_path}@{cR[:12]}")
    assert plan["routing"]["passes"] and len(plan["epochs"]) == N and {e["key"].split("|")[0] for e in plan["epochs"]} == {"simh_pilot"}
    print(f"  Sol {SCORERS['sol'][1]}: admit() + routing_equal() PASS on the pilot ({len(plan['epochs'])} epochs, template {plan['cite']['template_class_counts']})")

    # p2: the inline-attestation reader, the routing-record checks (the `bad` list), template classes, and the A48a step-0 / identity checks
    ns = {"re": re, "json": json, "Path": Path}
    src, tree = load("p2", {"PILOT_WORLD", "PILOT_STAGED", "PILOT_ORIG", "PILOT_PINS", "CUTFULL_RECORDS", "CUTFULL_INPUT_VOL", "CUTFULL_STEP0", "SOLD_RECORDS",
                            "TEMPLATE_CLASSES", "A25_ORIG", "_attest_source", "_pilot_pin", "_target_of", "_pilot_ref", "_sold_inline"}, ns)
    def commit_of(c):  # as p2's main: the full commit a 9-40 hex ref names, or None
        r_ = subprocess.run(["git", "-C", str(repo), "rev-parse", "--verify", "--quiet", f"{c}^{{commit}}"], capture_output=True, text=True, check=False)
        return r_.stdout.strip() if re.fullmatch(r"[0-9a-f]{9,40}", c or "") and r_.returncode == 0 else None

    ns["PILOT_PINS"] |= {"routing": (cR[:12], r_path, sha(rb)), "attestation": (cAt[:12], a_path, sha(att_raw)), "step0_gate": (S0[1], S0[0], S0[2]),
                         "staging_identity": (cA[:12], m.IDENT[0], m.IDENT[2]), "keys": (cA[:12], m.COUNTED[0], m.COUNTED[2])}
    sset, rt = P["staged_set"], f"{ns['CUTFULL_RECORDS']}/routing"
    listing = set(git(repo, "ls-tree", "-r", "--full-tree", "--name-only", cI, "--", ns["CUTFULL_RECORDS"]).splitlines())
    target = ns["_target_of"](sset)
    v = ns | {"listing": listing, "rt": rt, "target": target}
    hits = eval(piece(src, tree, "main", "re.escape(target)}(?:_r([0-9]+))?", "assign")[0], v)
    batch, k, rp = max(hits, key=lambda h: h[1])
    att_p2 = f"{ns['CUTFULL_RECORDS']}/attest/attestations_{batch}__{target}.json"
    assert ns["_attest_source"](batch, rp, sha(rb), False) == "inline"
    pj, ab, aj, sold_bad = ns["_sold_inline"](json.loads(rb), rp, att_p2, sh, commit_of, sha)
    rec2, mine = json.loads(rb), [{"meta_arm": ARM, "path": f"{sset}/inputs/{AF}", "sha": U_SHA, "run": P["root"], "folder": f"/cutfull/sold/p2/{P['folder']}"}]
    v |= {"rec": rec2, "k": k, "pj": pj, "aj": aj, "batch": batch, "cutfull_batch": "sold", "sset": sset, "pilot_rec": True, "commit_of": commit_of, "rp": rp, "att": att_p2,
          "ab": ab, "sha": sha, "toks": [a["meta_arm"].split("@")[0] for a in rec2["arms"]], "arms": {a["meta_arm"]: a for a in rec2["arms"]}, "mine": mine}
    c1, l1 = piece(src, tree, "main", 'att != _pilot_pin("attestation")[1]')
    c2, l2 = piece(src, tree, "main", '("one step-0 compare"', "assign")
    tc = aj["template_class"][TOK]
    v |= {"planned": {k_: 0 for k_ in KEYS}, "tc": tc, "extra_tc": set(tc) - set(KEYS), "s4_orph": set(), "il_out": {}}
    c3, l3 = piece(src, tree, "main", "any(v not in TEMPLATE_CLASSES for v in tc.values())")
    g_, id_ = json.loads(sh(cA, S0[0])), json.loads(sh(cA, m.IDENT[0]))
    kr_ = json.loads(sh(cA, m.COUNTED[0]))["keys"]
    v |= {"g_": g_, "id_": id_, "gr_": g_["runs"], "kr_": kr_, "inv_map": {k_: 0 for k_ in KEYS}, "pilot_att": aj,
          "want_k": {str(r_.get("volume_path")).removeprefix("simh_evals_v1/") + "#e0" for r_ in kr_}}
    c4, l4 = piece(src, tree, "main", 'g_.get("passes") is not True')
    c5, l5 = piece(src, tree, "main", 'id_.get("identical") is not True')
    o = rec2["original_folders"][TOK]
    assert not sold_bad and not eval(c1, v) and eval(c2, v) == [] and not eval(c3, v) and not eval(c4, v) and not eval(c5, v) and o in ns["A25_ORIG"] \
        and o == inv_rows[0]["original_folder"], (sold_bad, eval(c2, v))
    print(f"  p2 {SCORERS['p2'][1]}: _sold_inline, L{l1} pilot pins, L{l2} routing checks, L{l3} template classes, L{l4} step-0 gate, L{l5} staging identity PASS"
          " (sold_equal: its pilot-only copies are in experiments/2026-10-17_step2_pilot_sold_equal, not exercised here)")

    # own_all: _cut_jobs_b on the pilot inventory: the S_old attestation path, _pilot_check and the pilot positive control; simt carries loo's code
    srcs = {x: git(REAL, "cat-file", "blob", SCORERS[x][0]) for x in ("own_all", "own_all_simt", "own_all_loo3")}
    seg = lambda s, fn: ast.get_source_segment(s, next(n for n in ast.parse(s).body if isinstance(n, ast.FunctionDef) and n.name == fn))  # noqa: E731
    assert all(seg(srcs["own_all_simt"], fn) == seg(srcs["own_all"], fn) for fn in ("_pilot_check", "_cut_jobs_b", "_simh_world", "_pilot_positive_control"))
    for x in ("own_all", "own_all_loo3"):
        ns = {"json": json, "hashlib": hashlib, "re": re, "subprocess": subprocess, "Path": Path, "cf": SimpleNamespace(VOLUME=m.VOLUME, RULE="(synthetic)")}
        load(x, {"CUT_ROUTING", "SOLD_ATT", "INV_WORLD", "PILOT_WORLD", "PILOT_FOLDER", "PILOT_ORIGINAL", "PILOT_ROUTING", "PILOT_ATTEST", "PILOT_INVENTORY",
                 "PILOT_KEYS", "PILOT_STEP0_GATE", "SIMH_WORLD_TABLE", "SIMH_FOLDER_WORLD", "PILOT_TARGET", "_simh_world", "_pilot_check",
                 "_pilot_positive_control", "_cut_jobs_b"}, ns)
        ns |= {"REPO": repo, "PILOT_ROUTING": (cR[:12], r_path, sha(rb)), "PILOT_ATTEST": (cAt[:12], a_path, sha(att_raw)), "PILOT_INVENTORY": (cI[:12], inv, sha(sh(cI, inv))),
               "PILOT_KEYS": (cA[:12], m.COUNTED[0], m.COUNTED[2]), "PILOT_STEP0_GATE": (S0[1], S0[0], S0[2])}
        jobs, skipped = ns["_cut_jobs_b"](lambda *a: git(repo, *a), cI, ns["PILOT_INVENTORY"], ("only=simh_q4f_v1_pilot",))
        assert len(jobs) == 1 and jobs[0][2] == "simh" and len(jobs[0][0]["epochs"]) == N and {e["world"] for e in jobs[0][0]["epochs"]} == {"simh_pilot"}, skipped
        print(f"  {x} {SCORERS[x][1]}{' (= simt ' + SCORERS['own_all_simt'][1] + ')' if x == 'own_all' else ''}: _cut_jobs_b + _pilot_check + _pilot_positive_control "
              f"PASS; skipped {skipped}")

    # famexp: main()'s routed-folder line, _pilot_routing (the pinned pilot record takes PILOT_FOLDER), the planned-folder skip, the S_old attestation
    # reads, _pilot_check (with its step-0 gate) and the pilot positive control, with main()'s condition that calls it
    ns = {"json": json, "hashlib": hashlib, "re": re}
    src, tree = load("famexp", {"PILOT_WORLD", "PILOT_FOLDER", "PILOT_ORIGINAL", "PILOT_ROUTING", "PILOT_ATTEST", "PILOT_INVENTORY", "PILOT_KEYS", "PILOT_STEP0_GATE",
                                "PILOT_STEM", "S_OLD_ORIG_LABEL", "_pilot_check", "_pilot_routing", "_pilot_positive_control"}, ns)
    load("famexp_cutfull", {"_strip"}, ns)
    ns |= {"PILOT_ROUTING": (cR[:12], r_path, sha(rb)), "PILOT_ATTEST": (cAt[:12], a_path, sha(att_raw)), "PILOT_INVENTORY": (cI[:12], inv, sha(sh(cI, inv))),
           "PILOT_KEYS": (cA[:12], m.COUNTED[0], m.COUNTED[2]), "PILOT_STEP0_GATE": (S0[1], S0[0], S0[2])}
    rec3 = json.loads(rb)
    arms3 = [dict(a, token=a["meta_arm"].split("@", 1)[0]) for a in rec3["arms"]]
    der, ld = piece(src, tree, "main", 'a["folder"], a["original_folder"] = ', "assign")
    for a in arms3:
        a["folder"], a["original_folder"] = eval(der, ns | {"cf": SimpleNamespace(_strip=ns["_strip"]), "rec": rec3, "a": a})
    derived = arms3[0]["folder"]
    sel = ns["_pilot_routing"](sh, r_path, rb, rec3, arms3)
    for a in arms3 if sel else []:
        a["folder"] = ns["PILOT_FOLDER"]
    planned = {r_["uncut_planned"]["folder"] for r_ in inv_rows}
    cskip, lskip = piece(src, tree, "main", '{a["folder"] for a in arms} <= planned')  # True = the root is skipped
    ap, ac = rec3["attestation"]["path"].split(" @ ")
    atb = sh(ac, ap)
    fx = {r_["key"]: {"folder": r_["uncut_planned"]["folder"], "world": r_["world"], "task": r_["task"], "tokens_before": r_["full_len"],
                      "label": ns["S_OLD_ORIG_LABEL"][r_["original_folder"]]} for r_ in inv_rows}
    ev = ns["_pilot_check"](sh, r_path, rb, ap, atb, ns["PILOT_INVENTORY"], fx, inv_rows)
    v = ns | {"hashlib": hashlib, "atb": atb, "at": json.loads(atb), "rec": rec3, "k": 1, "ac": ac, "routing_commit": cI, "p": r_path, "cf": SimpleNamespace(VOLUME=m.VOLUME),
              "git": lambda *a: git(repo, *a, text=False), "arms": arms3, "planned": planned, "pilot_selected": sel, "only": "", "refuse": ""}
    c5, l5 = piece(src, tree, "main", '"--contains", routing_commit)')
    c6, l6 = piece(src, tree, "main", '"--contains", ac)')
    c7, l7 = piece(src, tree, "main", 'rec["attestation"].get("revision") != k')
    c8, l8 = piece(src, tree, "main", 'at.get("failing") != {} or at.get("revision") != k')
    c9, l9 = piece(src, tree, "main", 'rec.get("volume") != cf.VOLUME')
    cpc, lpc = piece(src, tree, "main", "pilot_selected or ((PILOT_STEM in only.split")
    assert sel and not eval(cskip, v) and len(ev) == N and not any(eval(c, v) for c in (c5, c6, c7, c8, c9)) and eval(cpc, v)
    ns["_pilot_positive_control"](sh, [{"inventory": fx}])
    print(f"  famexp {SCORERS['famexp'][1]}: L{ld} derives {derived!r}; _pilot_routing selects the pinned record -> {arms3[0]['folder']!r}; L{lskip} in the planned folders; "
          f"L{l5}/L{l6} on the branch, L{l7} sha/revision/originals, L{l8} failing/revision, L{l9} volume/cut/revision; _pilot_check (step-0 gate); "
          f"L{lpc} calls _pilot_positive_control: PASS")
    shutil.rmtree(repo)
    r36.unlink()
    print("ALL PASS")


if __name__ == "__main__":
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--tmp", default=None, help="parent folder of the throwaway repo (default: the system temp folder)")
    main(ap_.parse_args().tmp)
