"""Run as-run scoring and table scripts inside one Modal CPU container instead of on the laptop (Aswin, 10-13 17:24 PT via
0d: "i told u to run nothing locally and run it all on modal"). The scripts run byte-identical; only their surroundings
are provided in the container:
  - the repo files they read (this folder, #36's runs files and metrics, G3's folder), shipped as they are at HEAD;
  - #36's run-score files (scores_L48/63.npz) from the work Volume, sha256-checked, under a scores dir;
  - a `git` shim that replays answers the laptop verified first: `rev-parse HEAD` (the pushed, clean HEAD), `status` and
    `diff --quiet` (clean), `ls-files --error-unmatch` (every shipped file is tracked), and `show <rev>:<path>` from blobs
    the laptop extracted with real git;
  - for scripts whose local entrypoint calls a Modal function (`score.remote`), the module's own `_score` runs in this
    container instead, with the same Volumes at the same mount points.
Returned: only the new files under results/ (numbers and keys) and stdout; nothing is written to any Volume.

  PYTHONPATH=src uv run --no-project --python 3.12 --with 'modal==1.6.1' modal run --detach \\
      experiments/2026-10-13_q2_fixfull/run_in_modal.py --job <jobs/<name>.json>
"""
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path

import modal

HERE = Path(__file__).parent
REPO = HERE.parents[1] if modal.is_local() else Path("/repo")
APP = "probe-shift_aswin-surya_sub-q2-f10_runner_kill-2026-11-13"
SHIP = ["experiments/2026-10-13_q2_fixfull", "experiments/2026-10-13_g3_new_reader",
        "experiments/2026-10-06_q3_attack_detector/results/run_scores_2026-10-06_181548",
        "experiments/2026-10-06_q3_attack_detector/results/sim_honest_scores_2026-10-07_124717",
        "experiments/2026-10-06_q3_attack_detector/results/metrics_2026-10-06_181548__2026-10-06_182739.json",
        "experiments/2026-10-12_q3_edit_matched_fa", "experiments/2026-10-12_q3_edit_honest_fa", "src/probe_shift"]
S36 = {"scores_L48.npz": "f6cf939e20134218cc2f0f7d9ecc1bdaf21aadfdb05b921d7361551017e70fc4",
       "scores_L63.npz": "60f8e0d4b7b45baf46c6cf96f61677c7a0e966a3698a99a913763a4d763e6f10"}
SHIM = r'''#!/usr/bin/env python3
import os, sys
a = [x for x in sys.argv[1:]]
if a[:2] == ["rev-parse", "HEAD"]:
    print(os.environ["SHIM_HEAD"]); sys.exit(0)
if a[:1] in (["status"], ["diff"]):
    sys.exit(0)
if a[:1] == ["ls-files"]:
    print(a[-1]); sys.exit(0)
if a[:1] == ["show"] and len(a) == 2:
    p = os.path.join("/gitblobs", a[1].replace("/", "__"))
    if os.path.exists(p):
        sys.stdout.buffer.write(open(p, "rb").read()); sys.exit(0)
    sys.stderr.write("shim: no blob for " + a[1] + "\n"); sys.exit(128)
sys.stderr.write("shim: unsupported git " + " ".join(a) + "\n"); sys.exit(2)
'''

app = modal.App(APP)
la = modal.Volume.from_name("probe-shift_aswin-surya_sub-la-activations", version=2).read_only()
q3work = modal.Volume.from_name("probe-shift_aswin-surya_sub-q3attack_work_kill-2026-11-06", version=2).read_only()
g3acts = modal.Volume.from_name("probe-shift_aswin-surya_sub-q2-g3_acts_kill-2026-11-12", version=2).read_only()
g3work = modal.Volume.from_name("probe-shift_aswin-surya_sub-q2-g3_work_kill-2026-11-12", version=2).read_only()


def build_image():
    img = modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy==2.3.3", "modal==1.6.1")
    if modal.is_local():
        for rel in SHIP:
            p = REPO / rel
            img = img.add_local_dir(p, f"/repo/{rel}", copy=True) if p.is_dir() else img.add_local_file(p, f"/repo/{rel}", copy=True)
        blobs = Path(tempfile.mkdtemp(prefix="gitblobs_"))
        for spec in json.loads((HERE / "jobs" / "git_show_specs.json").read_text()):
            (blobs / spec.replace("/", "__")).write_bytes(subprocess.run(["git", "show", spec], cwd=REPO, capture_output=True, check=True).stdout)
        shim = Path(tempfile.mkdtemp(prefix="gitshim_")) / "git"
        shim.write_text(SHIM)
        extra = REPO / ".f10_inputs"  # untracked copies of Volume files and gitignored inputs; every script checks their sha256
        if extra.is_dir():
            img = img.add_local_dir(extra, "/inputs", copy=True)
        img = img.add_local_dir(blobs, "/gitblobs", copy=True).add_local_file(shim, "/usr/local/bin/git", copy=True).run_commands("chmod +x /usr/local/bin/git")
    return img.env({"PYTHONPATH": "/repo/src"})


image = build_image()


def _run(job: dict, head: str) -> dict:
    import importlib.util
    import io
    import os
    import sys
    import time
    import types
    from contextlib import redirect_stdout
    os.environ["SHIM_HEAD"] = head
    s36 = Path("/tmp/scores36")
    s36.mkdir(exist_ok=True)
    src = Path("/repo/experiments/2026-10-06_q3_attack_detector/results/run_scores_2026-10-06_181548")
    (s36 / "runs.json").write_bytes((src / "runs.json").read_bytes())
    for n, want in S36.items():
        q3 = Path("/q3work") if Path("/q3work").exists() else Path("/work")  # the Qwen3-32B function mounts it at /work
        b = (q3 / "runs/2026-10-06_181548" / n).read_bytes()
        if hashlib.sha256(b).hexdigest() != want:
            raise ValueError(f"{n} is not the pinned #36 score file")
        (s36 / n).write_bytes(b)
    res = Path("/repo/experiments/2026-10-13_q2_fixfull/results")
    before = {p.name for p in res.iterdir()}
    args = [str(x).replace("{S36}", str(s36)) for x in job["args"]]
    t0 = time.time()
    os.chdir("/repo")
    exit_note = None
    if job["kind"] == "cli":
        r = subprocess.run([sys.executable, f"/repo/{job['script']}", *args], cwd="/repo", capture_output=True, text=True)
        out = r.stdout
        if r.returncode:
            exit_note = f"exited {r.returncode}; stderr tail (script messages, no transcript text): {r.stderr[-1500:]}"
    else:  # a Modal app's local entrypoint, run here; its score.remote becomes the module's _score in this container
        spec = importlib.util.spec_from_file_location(Path(job["script"]).stem, f"/repo/{job['script']}")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
        mod.score = types.SimpleNamespace(remote=mod._score)
        kw = dict(a.split("=", 1) for a in args)
        buf = io.StringIO()
        try:
            with redirect_stdout(buf):
                mod.main(**kw)
        except SystemExit as e:  # the script's own refusal or failed gate: its files and message still come back
            exit_note = f"SystemExit: {str(e)[:1500]}"
        out = buf.getvalue()
    import base64
    new = {p.name: base64.b64encode(p.read_bytes()).decode() for p in res.iterdir() if p.name not in before and p.is_file()}
    return {"files": new, "stdout": out[-8000:], "seconds": round(time.time() - t0, 1), "exit": exit_note}


def _safe(job, head):
    try:
        return _run(job, head)
    except Exception as e:  # noqa: BLE001 — scores and keys only flow here; the message is the script's own (no transcript text)
        import traceback
        tb = traceback.extract_tb(e.__traceback__)[-1]
        raise RuntimeError(f"{type(e).__name__} at {Path(tb.filename).name}:{tb.lineno}: {str(e)[:2000]}") from None


@app.function(image=image, cpu=16, memory=65536, timeout=3600, retries=0, volumes={"/acts": la, "/work": q3work})
def run_qwen(job: dict, head: str) -> dict:
    return _safe(job, head)


@app.function(image=image, cpu=16, memory=65536, timeout=3600, retries=0, volumes={"/acts": g3acts, "/work": g3work, "/q3work": q3work})
def run_seed(job: dict, head: str) -> dict:
    return _safe(job, head)


@app.local_entrypoint()
def main(job: str):
    if subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=REPO, capture_output=True, text=True).stdout.strip():
        raise SystemExit("commit first: the container's git shim replays a clean HEAD")
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()
    branch = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()
    remote = subprocess.run(["git", "ls-remote", "origin", f"refs/heads/{branch}"], cwd=REPO, capture_output=True, text=True, check=True).stdout.split()
    if not remote or remote[0] != head:
        raise SystemExit("push first: HEAD must be on the remote branch")
    for rel in SHIP:  # every shipped file must be tracked (or gitignored) and unchanged
        if subprocess.run(["git", "diff", "--quiet", "HEAD", "--", rel], cwd=REPO).returncode:
            raise SystemExit(f"{rel} differs from HEAD")
    jobs = json.loads(Path(job).read_text())
    for j in jobs:
        fn = run_seed if j.get("volumes") == "seed" else run_qwen
        out = fn.remote(j, head)
        import base64
        for name, b64 in out["files"].items():
            with open(HERE / "results" / name, "xb") as fh:  # results/ is append-only
                fh.write(base64.b64decode(b64))
            print(HERE / "results" / name)
        print(f"[{j['script']}] {out['seconds']} s exit: {out['exit']}\n{out['stdout']}")
        if out["exit"] and j.get("stop_on_failure", True):
            raise SystemExit(f"{j['script']}: {out['exit']}")
