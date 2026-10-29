"""Route C §0.3 (10-17): INPUTS_FULL's `dryrun_counts` sidecar from a records-wrapper dry-run step (run_peraction311.py main with
dryrun=True on a TEST INPUTS_FULL, reducer dryrun_plan, verify_only). Keys, paths and counts only.

Laptop-parse class (the audit's three classes): keys-only manifest / config. It reads only the wrapper client's own log lines
(`bundle sha256 … run_id …`, `reduced {…} git_version …`, `RECSTEP_OK`; the reduced object is the dryrun_plan reducer's keys-and-counts
view, every value-typed field already refused in the container) and the step spec at the cited commit (a config). No record is read.

It writes, exclusive-create, the 110447 sidecar's keys (what, app, client_log_sha256, command, git, inputs_full, plan, ran_pt) plus
refused_paths (the JSON paths the reducer refused, never their values) and run_id. Fails closed (STOP, nothing written) unless the log
holds exactly one RECSTEP_OK, one bundle line, one reduced line and one spec line naming THIS spec path, its sha256 at the commit and the
commit (O-S3: the client prints it after preflight); the reduced view passed with a plan whose dryrun is true, whose top-level keys equal
the spec's expect_plan_keys (O-S1) and whose full_length.inputs_full.test is true (O-S2); every string and key of the plan and its
refused paths passes the wrapper's string rule again, with the wrapper as committed at the commit (O2-4) and the spec's expect_strings (R-S1 / O-S3, re-applied before writing); the spec at
the commit is a dryrun_plan step on run_peraction311.py with dryrun true on a TEST INPUTS_FULL; and the app id is an `ap-` id. inputs_full.py
reads no field of this sidecar (verify L866-867 checks only that the pin resolves at its sha256).

  uv run --no-project --python 3.12 python -I make_dryrun_sidecar.py --log <client run.log> --spec <step spec repo path> \\
      --commit <C, 40 hex> --app <ap-…> --out <repo path results/dryrun_counts_FULL_<stamp>.json>
"""

import argparse
import hashlib
import importlib.util
import json
import re
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent
RUNNER = "experiments/2026-10-15_fpr_sweep_507_prep/run_peraction311.py"
WHAT = ("full-length dry-run plan (run_peraction311 --dryrun inside the records wrapper; no remote compute by the runner; keys, paths and "
        "counts only; value-typed fields refused by the wrapper's dryrun_plan reducer): INPUTS_FULL's dryrun_counts sidecar")
BUNDLE = re.compile(r"bundle sha256 [0-9a-f]{64} commits [0-9]+ bytes [0-9]+ run_id ([0-9a-f]{32})")
SPECLINE = re.compile(r"spec (\S+) sha256 ([0-9a-f]{64}) commit ([0-9a-f]{40})")
WRAPPER = "experiments/2026-10-17_records_step_container/run_records_step_in_container.py"


def load_wrapper(src: bytes, tmp: Path):
    """O2-4: the wrapper's string rule as committed at C (main() passes `git show C:WRAPPER`), never the working tree. Loading it runs its
    module scope: stdlib imports, and its Modal app objects only when modal is importable (no Modal call either way)."""
    f = tmp / "recstep_wrapper_at_c.py"
    f.write_bytes(src)
    sp = importlib.util.spec_from_file_location("recstep_wrapper_at_c", f)
    m = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(m)
    return m


def build(log_bytes: bytes, spec_bytes: bytes, spec_path: str, commit: str, app: str, W) -> dict:
    if not re.fullmatch(r"[0-9a-f]{40}", commit) or not re.fullmatch(r"ap-[A-Za-z0-9]{10,40}", app):
        raise SystemExit("STOP: the commit must be 40 hex and the app id an ap- id")
    lines = log_bytes.decode().splitlines()
    oks = [x for x in lines if x == "RECSTEP_OK"]
    bl = [m for m in (BUNDLE.fullmatch(x) for x in lines) if m]
    rl = [x for x in lines if x.startswith("reduced {") and " git_version " in x]
    sl = [m.groups() for m in (SPECLINE.fullmatch(x) for x in lines) if m]
    if len(oks) != 1 or len(bl) != 1 or len(rl) != 1 or len(sl) != 1:
        raise SystemExit(f"STOP: the log must hold exactly one RECSTEP_OK, bundle, reduced and spec line ({len(oks)}, {len(bl)}, {len(rl)}, {len(sl)})")
    if sl[0] != (spec_path, hashlib.sha256(spec_bytes).hexdigest(), commit):
        raise SystemExit("STOP: the log's spec line does not name this spec path, its sha256 at the commit, and the commit (O-S3)")
    red = json.loads(rl[0][len("reduced "):].rsplit(" git_version ", 1)[0])
    if not (isinstance(red, dict) and red.get("pass") is True and isinstance(red.get("plan"), dict) and red["plan"].get("dryrun") is True
            and isinstance(red.get("refused_paths"), list)):
        raise SystemExit("STOP: the reduced view did not pass, or carries no dry-run plan")
    spec = json.loads(spec_bytes)
    fl = ((spec.get("entry") or {}).get("flags")) or {}
    if spec.get("script") != RUNNER or spec.get("reducer") != "dryrun_plan" or spec.get("verify_only") is not True or fl.get("dryrun") is not True \
            or fl.get("spec_only") or not str(fl.get("inputs_full", "")).startswith("config/INPUTS_FULL_TEST_") \
            or not isinstance(spec.get("expect_strings"), list) or not isinstance(spec.get("expect_plan_keys"), list):
        raise SystemExit("STOP: the spec is not a dryrun_plan step of the runner on a TEST INPUTS_FULL with pinned strings and plan keys")
    plan = red["plan"]
    if sorted(plan) != spec["expect_plan_keys"] or (((plan.get("full_length") or {}).get("inputs_full")) or {}).get("test") is not True:
        raise SystemExit("STOP: the plan's keys are not the spec's expect_plan_keys, or full_length.inputs_full.test is not true (O-S1 / O-S2)")
    try:  # R-S1, re-applied before anything is written: every string and key passes the rule (pinned, or a plain identifier)
        W._no_float({"plan": plan, "refused_paths": red["refused_paths"]}, values=True, pinned=frozenset(spec["expect_strings"]))
    except W.Stop as e:
        raise SystemExit(f"STOP: the reduced plan fails the string rule ({e})")
    return {"what": WHAT, "app": app, "client_log_sha256": hashlib.sha256(log_bytes).hexdigest(),
            "command": (f"records wrapper client --spec {spec_path}@{hashlib.sha256(spec_bytes).hexdigest()} --commit {commit} "
                        f"(entry {RUNNER}:main.info.raw_f, flags {json.dumps(fl, sort_keys=True)})"),
            "git": commit, "inputs_full": fl["inputs_full"], "plan": red["plan"], "refused_paths": red["refused_paths"],
            "run_id": bl[0].group(1), "ran_pt": datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d %H:%M:%S PT")}


def main():
    ap = argparse.ArgumentParser()
    for k in ("--log", "--spec", "--commit", "--app", "--out"):
        ap.add_argument(k, required=True)
    a = ap.parse_args()
    if not re.fullmatch(r"[0-9a-f]{40}", a.commit) or not re.fullmatch(r"ap-[A-Za-z0-9]{10,40}", a.app):  # O3-3: before ANY git call
        raise SystemExit("STOP: --commit must be 40 hex and --app an ap- id (checked before any git call or wrapper load)")
    repo = Path(subprocess.run(["git", "-C", str(HERE), "rev-parse", "--show-toplevel"], capture_output=True, text=True, check=True).stdout.strip())
    out = repo / a.out
    if out.exists() or out.is_symlink():
        raise SystemExit(f"STOP: {a.out} exists")
    spec_bytes = subprocess.run(["git", "-C", str(repo), "show", f"{a.commit}:{a.spec}"], capture_output=True, check=True).stdout
    with tempfile.TemporaryDirectory(prefix="dryrun_sidecar_") as td:
        W = load_wrapper(subprocess.run(["git", "-C", str(repo), "show", f"{a.commit}:{WRAPPER}"], capture_output=True, check=True).stdout, Path(td))
        side = build(Path(a.log).read_bytes(), spec_bytes, a.spec, a.commit, a.app, W)
    out.parent.mkdir(parents=True, exist_ok=True)
    b = json.dumps(side, indent=1).encode()
    with out.open("xb") as fh:
        fh.write(b)
    print(json.dumps({"written": a.out, "sha256": hashlib.sha256(b).hexdigest(), "n_refused": len(side["refused_paths"])}))


if __name__ == "__main__":
    main()
