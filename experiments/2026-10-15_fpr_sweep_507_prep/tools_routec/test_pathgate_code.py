"""Test of PATHGATE by code (fs_sites.py + pathgate_code.py), on a throwaway git repo: the runner's 11 modules and make_waitlist.py copied
byte for byte from the census head, the table under test, and fabricated configs (no real config or record is read). Cases, each
with its exact outcome:
- the census side a STOP as a positive control: a spec-only step whose @out folder holds no tracked file, with the records wrapper as it
  ran (ed31995): BAD on rp L5347 only; with the passed F-out2 wrapper (70a13a24): OK;
- D's first STOP as a positive control: inputs_b.json's local.labels as an absolute laptop path: BAD on rp L5895 only;
- a63478's R-1 as a positive control: a FROZEN --inputs-full file pinning waitlist, exclusions and census, with the waitlist missing at
  H: BAD on make_waitlist L107 (the child process's read, reached through inputs_full L435); all three present and matching: OK; a
  TEST file with pins pending: OK (the child never runs);
- a flag naming a file not tracked at H, and a flag whose "@sha256" differs from the file's (O4): BAD on rp L5611;
- an --inputs key naming a file not tracked at H: BAD on all 10 of its rows;
- O3: a spec with launch_from_pickle set, or another script: BAD; O5: a config git pin on a commit that is neither an ancestor of H nor
  in the spec's commits: BAD, and OK once the spec lists it;
- O1: a module with an import outside the stdlib, the analysed modules, modal and numpy: BAD (unexpected import);
- the committed table is current at the census head (fs_sites.py on the repository given, code only).

  uv run --no-project --python 3.12 python experiments/2026-10-15_fpr_sweep_507_prep/tools_routec/test_pathgate_code.py [--repo <checkout>]
  (--repo: the repository that holds the census head and the two wrapper blobs; default: this file's checkout; O6)
"""
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
E = "experiments/2026-10-15_fpr_sweep_507_prep"
H0 = "e1bcf52938100be27d97af6c81cc78456fffcc32"  # the census head the table was made at
OLD_WRAP, NEW_WRAP = "ed31995269bc78b7fc14c37623516e5c05a3e7ef", "70a13a249e1dfc9eff29dcb18e17f2624e48f679"
WRAP = "experiments/2026-10-17_records_step_container/run_records_step_in_container.py"
MODS = ["run_peraction311", "run_fprsweep_507", "run_common455", "run_common710", "cutfull_decisive", "full_override", "il_arms",
        "inputs_full", "inputs_final", "inputs_final_v2", "copies", "make_waitlist"]
TABLE = f"{E}/tools_routec/pathgate_code_table_2026-10-17.json"
RP = f"{E}/run_peraction311.py"
UNIT = "experiments/2026-10-14_fpr_sweep_unit/config/inputs_b.json"
PFX = "/abs/scratchpad/fpr507"


def git(repo, *a, text=True):
    return subprocess.run(["git", "-C", str(repo), *a], capture_output=True, text=text, check=True).stdout


def put(repo, path, data):
    p = repo / path
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data if isinstance(data, bytes) else data.encode())


def sha(b):
    return hashlib.sha256(b if isinstance(b, bytes) else b.encode()).hexdigest()


def gate(repo, c):
    r = subprocess.run([sys.executable, "-I", str(HERE / "pathgate_code.py"), "check", "--commit", c, "--spec", "steps/spec.json", "--table", TABLE,
                        "--allowlist", f"{E}/tools_routec/allow.json"], cwd=repo, capture_output=True, text=True)
    return r.returncode, r.stdout, [x.split()[1] for x in r.stdout.splitlines() if x.startswith("BAD ")]


def main(real):
    real_bytes = lambda spec: subprocess.run(["git", "-C", str(real), "cat-file", "-p", spec], capture_output=True, check=True).stdout  # noqa: E731
    res = {}
    tmp = Path(tempfile.mkdtemp(prefix="pathgate_code_test_"))
    repo = tmp / "repo"
    repo.mkdir()
    git(repo, "init", "-q")
    for k, v in (("user.email", "test@example.invalid"), ("user.name", "test"), ("commit.gpgsign", "false")):
        git(repo, "config", k, v)
    for m in MODS:
        put(repo, f"{E}/{m}.py", real_bytes(f"{H0}:{E}/{m}.py"))
    put(repo, TABLE, (HERE / "pathgate_code_table_2026-10-17.json").read_bytes())  # the table under test
    put(repo, WRAP, real_bytes(OLD_WRAP))
    for p in ("config/common455.json", "results/common_run455_2026-10-15_101451.json", "results/owna507_2026-10-15_031024.json",
              "results/owna507_2026-10-15_031320.json", "config/fpr507_copies_index_2026-10-17.json", "config/membership710.json",
              "config/pins.json", "config/m0.json", "config/E12.json", "results/wl.json", "results/ex.json", "results/cen.json"):
        put(repo, f"{E}/{p}", "{}")
    for p in ("data/t196.json", "data/labels.json", "data/v6.json"):
        put(repo, p, "{}")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "base")
    base = git(repo, "rev-parse", "HEAD").strip()
    br = git(repo, "rev-parse", "--abbrev-ref", "HEAD").strip()
    git(repo, "checkout", "-q", "-b", "side")  # a commit that is not an ancestor of the heads below (O5)
    put(repo, "data/side.json", "{}")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "side")
    side = git(repo, "rev-parse", "HEAD").strip()
    git(repo, "checkout", "-q", br)
    IN = {"sets": {"S_newmains": {"membership710": {"path": "config/membership710.json"}}},
          "refresh2": {"pins": {"path": "config/pins.json"}, "m0": {"path": "config/m0.json"}}}
    gitpin = {"git": f"{base}:data/labels.json"}

    def commit(labels=gitpin, inputs=IN, flags=None, wrap=None, ifull=None, commits=(), script=RP, name="c"):
        local = {"t196": {"git": "HEAD:data/t196.json"}, "labels": labels, "v6": {"git": f"{base}:data/v6.json"}}
        ub, ib = json.dumps({"local": local}).encode(), json.dumps(inputs).encode()
        put(repo, UNIT, ub)
        put(repo, f"{E}/config/IN.json", ib)
        allow = [{"path": UNIT, "sha256": sha(ub), "class": "K"}, {"path": f"{E}/config/IN.json", "sha256": sha(ib), "class": "K"}]
        fl = {"alltasks": True, "keys_only": True, "remote_driver": True, "inputs": f"config/IN.json@{sha(ib)}", "copies_root": PFX,
              "spec_only": f"@out:{E}/results/routec_census_specs/census_a_x.pkl"}
        if ifull is not None:
            fb = json.dumps(ifull).encode()
            put(repo, f"{E}/config/IFULL.json", fb)
            allow.append({"path": f"{E}/config/IFULL.json", "sha256": sha(fb), "class": "K"})
            fl["inputs_full"] = f"config/IFULL.json@{sha(fb)}"
        put(repo, f"{E}/tools_routec/allow.json", json.dumps({"files": allow}))
        put(repo, "steps/spec.json", json.dumps({"script": script, "entry": {"attr": "main", "flags": fl | (flags or {})},
                                                 "staging": {"prefix": PFX}, "commits": list(commits)}))
        if wrap:
            put(repo, WRAP, real_bytes(wrap))
        git(repo, "add", "-A")
        git(repo, "commit", "-q", "-m", name)
        return git(repo, "rev-parse", "HEAD").strip()

    rc, out, bads = gate(repo, commit(name="census as it ran"))
    res["positive control (census side a): wrapper ed31995 and an untracked @out folder: BAD on rp L5347 only"] = \
        rc == 1 and bads == ["run_peraction311@71633875:L5347"]
    rc, out, bads = gate(repo, commit(wrap=NEW_WRAP, name="F-out2 wrapper"))
    res["the same spec with the passed F-out2 wrapper (70a13a24): OK"] = rc == 0 and out.startswith("PATHGATE_CODE_OK") \
        and "wrapper_makes_out_parent true" in out
    rc, out, bads = gate(repo, commit({"path": "/Users/someone/scratchpad/slip_pins/claimc_labels_reduced.json"}, name="D as it ran"))
    res["positive control (D's first STOP): local.labels as a laptop path: BAD on rp L5895 only"] = rc == 1 and bads == ["run_peraction311@71633875:L5895"]
    pin = lambda p: {"commit": base, "path": f"{E}/{p}", "sha256": sha("{}")}  # noqa: E731
    rc, out, bads = gate(repo, commit(ifull={"status": "FROZEN x", "waitlist": pin("results/wl_missing.json"), "exclusions": pin("results/ex.json"),
                                             "census": pin("results/cen.json")}, name="frozen, waitlist missing"))
    res["positive control (a63478's R-1): a FROZEN --inputs-full whose waitlist is missing at H: BAD on make_waitlist L107 only"] = \
        rc == 1 and bads == ["make_waitlist@ea10d184:L107"]
    rc, out, bads = gate(repo, commit(ifull={"status": "FROZEN x", "waitlist": pin("results/wl.json"), "exclusions": pin("results/ex.json"),
                                             "census": pin("results/cen.json")}, name="frozen, all present"))
    res["R-1: the same with all three pins present at H with their sha256: OK"] = rc == 0 and out.startswith("PATHGATE_CODE_OK")
    rc, out, bads = gate(repo, commit(ifull={"status": "TEST", "waitlist": "pending", "exclusions": pin("results/ex.json"), "census": "pending"},
                                      name="test, pending"))
    res["R-1: a TEST --inputs-full with waitlist and census pending: the child never runs, so OK"] = rc == 0
    rc, out, bads = gate(repo, commit(flags={"e12": "config/E12_untracked.json@" + "0" * 64}, name="untracked flag"))
    res["a flag naming a file not tracked at H: BAD on rp L5611 only"] = rc == 1 and bads == ["run_peraction311@71633875:L5611"]
    rc, out, bads = gate(repo, commit(flags={"e12": "config/E12.json@" + "0" * 64}, name="flag sha"))
    res["O4: a flag whose @sha256 differs from its file at H: BAD on rp L5611 only"] = rc == 1 and bads == ["run_peraction311@71633875:L5611"] \
        and "sha256 at H differs" in out
    rc, out, bads = gate(repo, commit(inputs={"sets": {"S_newmains": {"membership710": {"path": "config/missing.json"}}},
                                              "refresh2": {"pins": {"path": "config/pins.json"}}}, name="untracked inputs key"))
    res["an --inputs key naming a file not tracked at H: BAD on each of its 10 table rows (rp L6017, inputs_final x9)"] = rc == 1 \
        and len(bads) == 10 and "run_peraction311@71633875:L6017" in bads
    rc, out, bads = gate(repo, commit(flags={"launch_from_pickle": "x.pkl@" + "0" * 64}, name="launch flag"))
    res["O3: a spec with launch_from_pickle set: BAD (spec)"] = rc == 1 and "BAD spec flag launch_from_pickle set" in out
    rc, out, bads = gate(repo, commit(script=f"{E}/inputs_full.py", name="other script"))
    res["O3: a spec whose script is not run_peraction311.py: BAD (spec)"] = rc == 1 and "BAD spec not a records-wrapper step" in out
    rc, out, bads = gate(repo, commit({"git": f"{side}:data/side.json"}, name="side pin"))
    res["O5: a config git pin on a commit neither an ancestor of H nor in the spec's commits: BAD on rp L5895 only"] = rc == 1 \
        and bads == ["run_peraction311@71633875:L5895"] and "not in the bundle" in out
    rc, out, bads = gate(repo, commit({"git": f"{side}:data/side.json"}, commits=[side], name="side pin listed"))
    res["O5: the same pin with its commit in the spec's commits: OK"] = rc == 0
    put(repo, f"{E}/copies.py", (repo / E / "copies.py").read_bytes() + b"\n\ndef _x():\n    import yaml  # noqa\n\n\n_x()\n")
    rc, out, bads = gate(repo, commit(name="unexpected import"))
    res["O1: an import outside the stdlib, the analysed modules, modal and numpy: BAD (unexpected import), and the stale table"] = rc == 1 \
        and "BAD unexpected import" in out and "import yaml" in out and "table module blobs differ" in out
    # the committed table is current at the census head: fs_sites.py on the given repository (code only)
    sys.path.insert(0, str(HERE))
    import pathgate_code as PC

    tb = json.loads((HERE / "pathgate_code_table_2026-10-17.json").read_text())
    en = json.loads(subprocess.run([sys.executable, "-I", str(HERE / "fs_sites.py"), H0], cwd=real, capture_output=True, text=True, check=True).stdout)
    res["the committed table is current at the census head (module blobs and targets, the in-step sites exactly the annotated ones, no "
        "unexpected import, the F-out2 wrapper)"] = en["modules"] == tb["modules"] and en["targets"] == tb["targets"] == ["make_waitlist"] \
        and {PC.key_of(s) for s in en["sites"] if s["in_step"]} == {r["key"] for r in tb["in_step"]} and tb["made_at"] == H0 \
        and en["imports_unexpected"] == [] and tb["out_of_step"]["n"] == sum(1 for s in en["sites"] if not s["in_step"]) \
        and tb["wrapper_makes_out_parent"] == [NEW_WRAP]
    for k, v in res.items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}")
    print("ALL PASS" if all(res.values()) else "FAILED")
    shutil.rmtree(tmp, ignore_errors=True)
    if not all(res.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=str(HERE.parents[2]), help="the repository holding the census head and the wrapper blobs (O6)")
    main(Path(ap.parse_args().repo))
