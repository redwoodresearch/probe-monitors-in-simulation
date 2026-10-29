"""F2 (0d, 10-17): inputs_full.py --copies-root. Inside the records wrapper (RECSTEP_EXC set) the fpr507 copies' root comes ONLY from
the flag: `verify --deep` without it STOPs at the first copy read, even with $PROBE_SHIFT_COPIES set; with it the read goes through. On
the laptop (no RECSTEP_EXC) the env fallback is unchanged. The flag is set before build or verify runs (a bad dir STOPs first).

The deep chain is the real one: inputs_full.main -> verify(deep) -> inputs_final_v2.verify -> inputs_final.verify -> copy_path ->
copies.check -> copies.root(). Only the loaders that read git pins or records are stubbed (inputs_full.load, require_pin_shas,
schema_problems, key_problems, pending_items; inputs_final_v2.load_pin; inputs_final.git_show, file_maps), with a synthetic copies index
and root. No record, no Modal. Each case runs in its own `python -I` child (fresh module state and environment).

  uv run --no-project --python 3.12 python -I test_inputs_full_copies_root.py
"""

import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sha = lambda b: hashlib.sha256(b).hexdigest()  # noqa: E731

R = "simh311_q9_t001_test__sim_honest"  # a synthetic root name
REL = f"v2man/{R}__p2.json"
P2B = b'{"folder": "x/simh311_q9_t001_test__sim_honest"}'
MAPB = json.dumps({"roots": [{"arm": R}]}).encode()

DRIVER = r'''
import json, sys
from pathlib import Path
cfg = json.loads(sys.argv[1])
sys.path.insert(0, cfg["here"])
import copies as CP
CP.INDEX_PATH, CP.INDEX_SHA = cfg["index_path"], cfg["index_sha"]
import inputs_full as F
import inputs_final as IF
import inputs_final_v2 as IF2
base1, base2 = cfg["base1"], cfg["base2"]
F.require_pin_shas = lambda m: None
F.schema_problems = F.key_problems = lambda m: []
F.pending_items = lambda m: m.get("pending")
F.load = lambda e, what: json.dumps(base2).encode()
IF2.load_pin = lambda e, what: base1
IF.git_show = lambda c, p: cfg["mapb"].encode()
IF.file_maps = lambda c: {}
LOG = []
orig = CP._load
def spy(rel):
    ent = {"rel": rel}
    LOG.append(ent)
    r = orig(rel)
    ent["problem"] = r[1]
    return r
CP._load = spy
sys.argv = ["inputs_full.py"] + cfg["argv"]
try:
    F.main()
    out = {"outcome": "returned", "msg": ""}
except SystemExit as e:
    s = str(e.code) if not isinstance(e.code, str) else e.code
    out = {"outcome": "copies_stop" if s.startswith("copies:") else "exit", "msg": s[:200]}
except Exception as e:
    out = {"outcome": "exc:" + type(e).__name__, "msg": str(e)[:200]}
root = CP._ROOT.get("root")
print("DRIVER " + json.dumps(out | {"loads": LOG, "root": str(root) if root else None}))
'''


def setup():
    d = Path(tempfile.mkdtemp(prefix="ifull_cr_"))
    root, bad = d / "root", d / "bad_root"
    for r_, b_ in ((root, P2B), (bad, P2B + b" ")):
        (r_ / "v2man").mkdir(parents=True)
        (r_ / REL).write_bytes(b_)
    idx = json.dumps({"files": {REL: sha(P2B)}}).encode()
    (d / "index.json").write_bytes(idx)
    keys = ["schema", "status", "cutoff_pt", "freeze_pt", "label", "sets", "ledger", "epoch2_honest", "nm360_fold_extension", "gapfill_sim2",
            "sel97", "rms", "extensions", "pools"]
    sets = {"S_new": {"file_maps": {"commit": "f" * 40}, "sol_mappings": {"bt": {"commit": "c" * 40, "file": "mapx", "sha256": sha(MAPB)}},
                      "roots": {"bt": [R]}, "manifest_copies": {R: {"p2": sha(P2B)}}}}
    v1 = {k: {} for k in keys} | {"schema": "inputs_final/v1", "status": "TEST", "sets": sets, "extensions": {}}
    base2 = {k: v1[k] for k in keys} | {"schema": "inputs_final/v2", "extensions": {"S_paper493": "not in this build; second refresh"},
                                        "refresh2": {"base": {"commit": "a" * 40, "path": "x.json", "sha256": "b" * 64}, "registered_fixes": [],
                                                     "pins": {"path": None}, "test": True}}
    m = {"schema": "inputs_full/v1", "status": "TEST", "base": {"commit": "a" * 40, "path": "base.json", "sha256": "b" * 64}, "pending": [],
         "lead_readout": "C", "scope": {}, "fits": {}, "inventories": {}}
    (d / "m.json").write_text(json.dumps(m))
    return d, root, bad, {"here": str(HERE), "index_path": str(d / "index.json"), "index_sha": sha(idx), "base1": v1, "base2": base2,
                          "mapb": MAPB.decode()}


def run(cfg, argv, wrapper, env_root):
    env = {k: v for k, v in os.environ.items() if k not in ("RECSTEP_EXC", "PROBE_SHIFT_COPIES", "PYTHONPATH")}
    if wrapper:
        env["RECSTEP_EXC"] = str(Path(cfg["index_path"]).parent / "exc.json")
    if env_root:
        env["PROBE_SHIFT_COPIES"] = env_root
    p = subprocess.run([sys.executable, "-I", "-c", DRIVER, json.dumps(cfg | {"argv": argv})], capture_output=True, text=True, env=env)
    line = [x for x in p.stdout.splitlines() if x.startswith("DRIVER ")]
    if len(line) != 1:
        return {"outcome": "no_driver_line", "msg": p.stderr[-300:], "loads": None, "root": None}
    return json.loads(line[0][len("DRIVER "):])


d, root, bad, cfg = setup()
F_ = str(d / "m.json")
W_MSG = "copies: inside the records wrapper the root comes only from"
cases = {}

r = run(cfg, ["verify", F_, "--deep"], wrapper=True, env_root=str(root))
cases["wrapper, verify --deep, NO flag (env set): STOP at the first copy read, the env is not used"] = (
    r["outcome"] == "copies_stop" and r["msg"].startswith(W_MSG) and r["loads"] == [{"rel": REL}] and r["root"] is None, r)

r = run(cfg, ["verify", F_, "--deep", "--copies-root", str(root)], wrapper=True, env_root=None)
cases["wrapper, verify --deep, --copies-root: the deep chain reads the copy through the flag's root (sha ok), no copies STOP"] = (
    r["outcome"] != "copies_stop" and r["loads"][:1] == [{"rel": REL, "problem": None}] and r["root"] == str(root), r)

r = run(cfg, ["verify", F_, "--deep", "--copies-root", str(bad)], wrapper=True, env_root=None)
cases["control: --copies-root at a root holding a changed copy: the read is hashed and differs (the copy is really read there)"] = (
    r["outcome"] != "copies_stop" and r["loads"][:1] == [{"rel": REL, "problem": f"copies: {REL} sha256 differs from the index"}], r)

r = run(cfg, ["verify", F_, "--deep"], wrapper=False, env_root=str(root))
cases["laptop (no RECSTEP_EXC), verify --deep, no flag: the env root as before"] = (
    r["outcome"] != "copies_stop" and r["loads"][:1] == [{"rel": REL, "problem": None}] and r["root"] is None, r)

r = run(cfg, ["verify", F_, "--deep", "--copies-root", str(root / "nope")], wrapper=True, env_root=str(root))
cases["verify --copies-root not a directory: STOP before verify (no copy read)"] = (
    r["outcome"] == "copies_stop" and "is not a directory" in r["msg"] and r["loads"] == [], r)

r = run(cfg, ["build", "--pins", "config/none.json", "--test", "--copies-root", str(root / "nope")], wrapper=True, env_root=None)
cases["build --copies-root not a directory: STOP before build (the flag is wired on build too)"] = (
    r["outcome"] == "copies_stop" and "is not a directory" in r["msg"] and r["loads"] == [], r)

r = run(cfg, ["verify", F_], wrapper=True, env_root=None)
cases["wrapper, verify WITHOUT --deep, no flag: this fixture's shallow verify reads no copy (the deep branch is the copy read)"] = (
    r["outcome"] != "copies_stop" and r["loads"] == [], r)

bad_ = [k for k, (ok, _) in cases.items() if not ok]
for k, (ok, r) in cases.items():
    print(("PASS " if ok else "FAIL ") + k + ("" if ok else f"  -> {json.dumps(r)[:400]}"))
print(f"{len(cases) - len(bad_)} of {len(cases)} checks pass")
sys.exit(1 if bad_ else 0)
