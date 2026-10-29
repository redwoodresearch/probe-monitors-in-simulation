"""ONE cutfull batch launch (Amendment 44): uv run --no-project --python 3.12 python -I launch_cutfull.py <chain batch> <records commit> [dry]
Every routing record of the batch (highest revision per target) at <records commit> on aswin/q3-cut-option goes into ONE scorer launch,
so the batch's folders are written together (the once-per-batch sibling check never sees a half-written sibling). Inputs are copied
from git by commit into ./<batch>/; the scorer re-reads and re-checks the routing records itself.
2e (10-17): the scorer runs under the named modal 1.6.1 binary (pilot_pipe.M; refused unless it is that version), never `uv run ... modal`.
0d (10-17, Option C, PILOT only): the pilot's stdout and stderr go through pilot_pipe.run_piped, a pipe with no intermediate file straight
into one exclusive upload to the shared raw-text Volume (p2/<run id>/); the launcher prints only that relay and never tails a log. Other
batches keep their log file and retry_launch.sh. The scorer's local printout.txt and manifest are unchanged (no scorer change)."""
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path

BATCH, RC = sys.argv[1], sys.argv[2]
DRY = len(sys.argv) > 3 and sys.argv[3] == "dry"
HERE = Path(__file__).resolve().parent
_pp = importlib.util.spec_from_file_location("pilot_pipe", HERE / "pilot_pipe.py")  # by path: -I keeps this folder off sys.path
pilot_pipe = importlib.util.module_from_spec(_pp)
_pp.loader.exec_module(pilot_pipe)
M = pilot_pipe.M
SH = HERE.parent.parent / "sh"
RT = "experiments/2026-10-14_q3_overnight_chain/results/cutfull/routing"
STEP0 = ("e6896f33b825", "experiments/2026-10-14_q3_overnight_chain/results/cutfull_step0_compare_2026-10-15_214206.json")
RULE09 = ("77a770eec", "experiments/2026-10-14_priority1_spec_09/RULE.md")
git = lambda *a: subprocess.run(["git", *a], capture_output=True, check=True, cwd=SH).stdout  # noqa: E731
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()  # noqa: E731
W = HERE / BATCH
W.mkdir(exist_ok=True)


def fetch(commit, path):
    dest = W / Path(path).name
    b = git("show", f"{commit}:{path}")
    if dest.exists() and dest.read_bytes() != b:
        raise SystemExit(f"{dest.name}: a different copy is already here")
    dest.write_bytes(b)
    return dest


def ref(s):  # "path @ commit" (an optional " (sha256 ...)" before the @)
    p, _, c = str(s).rpartition(" @ ")
    return c.strip(), p.split(" (sha256")[0].strip()


names = git("ls-tree", "-r", "--full-tree", "--name-only", RC, "--", RT).decode().splitlines()
best = {}
for n in names:
    m = re.fullmatch(rf"{re.escape(RT)}/{re.escape(BATCH)}__(.+_cutfull)(?:_r([0-9]+))?\.json", n)
    if m:
        k = int(m.group(2) or 1)
        if k > best.get(m.group(1), (0, ""))[0]:
            best[m.group(1)] = (k, n)
if not best:
    raise SystemExit(f"no routing records for batch {BATCH} at {RC}")
HELD = {t for t in os.environ.get("HELD_TARGETS", "").split(",") if t}  # targets left out of this batch launch (e.g. held until re-attested)
recs = {t: json.loads(git("show", f"{RC}:{n}")) for t, (k, n) in sorted(best.items()) if t not in HELD}
if HELD:
    print(f"{BATCH}: held targets left out: {sorted(HELD & set(best))}", flush=True)
SOLD = BATCH == "sold"  # A47b: the six sold__ records carry no pool / manifest / stage / extraction fields; one target per launch
SOLD_SIMH = {"simh_q4f_v1_pilot_cutfull": ("sold_pilot", "simh_q4f_v1_pilot_cutfull_v1__sim_honest")}  # A48 (09 via 2e, 10-17): the pilot's sold target, by exact
# name, runs in simh mode as its own world; A48d (2e rename): its folder is named here, since its staged set (simh_pilot_cutfull_v1) does not carry it
if SOLD and len(recs) != 1:
    sys.exit(f"sold: launch one target at a time (HELD_TARGETS), not {sorted(recs)}")
if SOLD:
    (r0,) = recs.values()
    ia_p, _, ia_c = r0["attestation"]["path"].partition(" @ ")
    aj0 = json.loads(git("show", f"{ia_c.strip()}:{ia_p}"))  # the scorer re-checks this payload by the record's sha256
    sx = [n for n in git("ls-tree", "-r", "--full-tree", "--name-only", RC, "--", "experiments/2026-10-14_q3_overnight_chain/results").decode().splitlines()
          if re.fullmatch(rf"experiments/2026-10-14_q3_overnight_chain/results/extract_chain_{re.escape(r0['extraction_root'])}_a[0-9]+_[0-9_-]+\.json", n)]
    if len(sx) != 1:
        sys.exit(f"sold: {len(sx)} extraction records for {r0['extraction_root']} at {RC}")
    r0 |= {"pool": "sold_a25" if next(iter(recs)).startswith("simh_q4f_v1_full_b") else SOLD_SIMH.get(next(iter(recs)), ("sold",))[0], "extraction_record": [{"commit": RC, "path": sx[0]}],
           "stage_record": aj0["cutfull"]["stage_record"], "manifest": None}
pools = {r["pool"] for r in recs.values()}
SIMH_WORLD = {"simh311": "simh_311", "simh_nm360": "simh_nm360", "simh_t196fill": "t196fill", "sp493_g4_simh": "sp493_t196", "sp493_g5_simh": "sp493_new",
              "sold_a25": "simh_full", "sold_pilot": "simh_pilot"}  # A47; A47b: the A25 packs' world (SIMH_PAIRS /simh/full_b{1,2}/p2); A48: the pilot's (/simh/pilot/p2)
mode = "simh" if pools <= set(SIMH_WORLD) else "scaleup" if not pools & set(SIMH_WORLD) else sys.exit(f"mixed pools {pools}")
staged, ex, man, stg, n_ep, pins = [], [], [], [], 0, {}
for t, r in recs.items():
    for a in r["arms"]:
        staged.append(f"{r['extraction_root']}={a['staged_inputs_file']}={a['staged_sha']}={a['meta_arm']}")
        n_ep += a["epochs"]
    ex += [fetch(e["commit"], e["path"]) for e in r["extraction_record"]]
    if r["manifest"] is not None:  # A47b: sold has none (entries from inventory_b_fix_v1 by key)
        man.append(fetch(*ref(r["manifest"])))
    stg.append(fetch(*ref(r["stage_record"])))
    if not str(r.get("labels") or "").startswith("not applicable") and r.get("labels"):
        c_, p_ = ref(r["labels"])
        if c_ and re.fullmatch(r"[0-9a-f]{7,40}", c_):
            pins[f"labels_{t}@{c_}"] = fetch(c_, p_)
for x in (y for y in os.environ.get("EXTRA_PINS", "").split(",") if y):  # e.g. the registered root's _rlr admission pins (name=path)
    k_, v_ = x.split("=", 1)
    pins[k_] = Path(v_)
pins[f"append41_09@{RULE09[0]}"] = fetch(*RULE09)
pins[f"cutfull_step0_gate@{STEP0[0]}"] = fetch(*STEP0)
ex, man, stg = sorted(set(ex)), sorted(set(man)), sorted(set(stg))
cmd = [M, "run", "--detach",  # 2e: the named modal 1.6.1 binary (its Python has no numpy; the scorer imports numpy only inside container functions)
       "experiments/2026-10-14_p2_simh_scores/run_simh_scores.py", "--manifest-b", str(SH.parent / "correctedb_manifest_v1.json"), "--mode", mode,
       "--cutfull", "--cutfull-batch", BATCH, "--cutfull-records", RC, "--staged", ",".join(staged), "--n-epochs", str(n_ep),
       "--dump-manifest", ",".join(map(str, ex)), "--dump-manifest-sha", ",".join(sha(x) for x in ex),
       "--stage-records", ",".join(map(str, stg)), "--stage-records-sha", ",".join(sha(x) for x in stg), "--records-commit", RC,
       "--pins", ",".join(f"{k}={v}={sha(v)}" for k, v in pins.items()), "--gate-tol", "1e-6", "--gate-flips"] + (["--dry-run"] if DRY else [])
if (mode == "scaleup" and not SOLD) or pools & {"sp493_g4_simh", "sp493_g5_simh"}:  # A47: sp493 sim-honest roots pass it too (sample_error admission, A45d)
    cmd += ["--keys-manifest", ",".join(map(str, man)), "--keys-manifest-sha", ",".join(sha(x) for x in man)]
if mode == "scaleup":
    wm = ["sp493_sim=sim"] if any(re.fullmatch(r"sp493_g[0-9]+_sim(?:_[xy])?", str(p_)) for p_ in pools) else []  # as at 32k (launch_sp493.py)
    wm += [x for x in os.environ.get("WORLD_MAP", "").split(",") if x]
    cmd += ["--world-map", ",".join(wm)] if wm else []
else:
    ((t1, r),) = recs.items()
    (a,) = r["arms"]
    world = SIMH_WORLD[r["pool"]]
    cmd += ["--run", r["extraction_root"], "--out-dir", f"/cutfull/{BATCH}/p2/{SOLD_SIMH[t1][1] if t1 in SOLD_SIMH else re.sub(r'_try[0-9]+$', '', a['staged_inputs_file'].split('/')[0]) + '__' + a['meta_arm'].split('@')[0]}",
            "--world", world, "--world-map", f"sim_honest={world}"]
print(f"{BATCH}: {mode}; targets {[f'{t} r{best[t][0]}' for t in recs]}; {len(staged)} staged arm(s); {n_ep} epochs; records {RC}; {'DRY RUN' if DRY else 'WRITE'}", flush=True)
env = os.environ | {"SIMH_VOL": "probe-shift_aswin-surya_sub-q3cutfull_acts_kill-2026-11-14", "P2_PROJ_CPU": "2", "PYTHONPATH": "src",
                    "P2_ARM_MEM_MB": "32768", "P2_ARM_TIMEOUT_S": "2400", "P2_PROJ_TIMEOUT_S": "600", "P2_ORCH_TIMEOUT_S": "3000"}
if not pilot_pipe.modal_ok(M):
    sys.exit(f"STOP: {M} is not an executable modal 1.6.1 binary")
if pools == {"sold_pilot"}:  # 0d Option C (PILOT): stdout+stderr piped with no file to the raw-text Volume; one launch, no retry loop, no log tail
    rel = pilot_pipe.run_piped(cmd, SH, env, *pilot_pipe.pilot_names(BATCH, DRY, os.environ.get("LOG_TAG", "")), uploader=pilot_pipe.UPLOADER,
                               timeout_s=pilot_pipe.PILOT_TIMEOUT_S)
    print(json.dumps(rel), flush=True)
    if rel.get("timed_out"):  # Astra (astra_p2pilot4): a timeout never proves that nothing was uploaded
        sys.exit(f"STOP: the pilot launch timed out after {pilot_pipe.PILOT_TIMEOUT_S} s; upload state UNKNOWN: check `{M} volume ls {rel['volume']} "
                 f"{rel['prefix']}`; check modal app list and stop the detached app")
    sys.exit(rel["scorer_exit_code"] if rel["uploader_exit_code"] == 0 and "upload_error" not in rel else 1)
log = W / f"launch_{'dry' if DRY else 'write'}{os.environ.get('LOG_TAG', '')}"  # A47: LOG_TAG keeps a batch's second launch (e.g. IL run3 after sim2) apart
if DRY:  # a dry run's verdict is not "written": run once, no retry loop
    with open(f"{log}_try1.log", "w") as fh:
        sys.exit(subprocess.run(cmd, cwd=SH, env=env, stdout=fh, stderr=subprocess.STDOUT).returncode)
sys.exit(subprocess.run([str(HERE.parent / "retry_launch.sh"), str(log), *cmd], cwd=SH, env=env).returncode)
