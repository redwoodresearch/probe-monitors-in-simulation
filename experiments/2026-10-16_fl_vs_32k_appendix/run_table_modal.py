"""Run the 32k vs full-length table generator in ONE small Modal CPU container. The Stage A records and their per-draw npz stay on the
solavg Volume (mounted read-only); the laptop never holds them. Needs main's go before any run (spend rule); expected cost about $0.02
(2 CPU; most of the time is hashing the two npz files).

The scripts shipped are the COMMITTED versions at HEAD (git show); the plan and the resolved training-length file are sent as bytes; the
side inputs (draws attestation, estimator selection, PANEL records) come from their own sources (sol: read in the container, or git: a
committed JSON) (the
plan names every input with its sha256 and the generator re-checks them). Record sources in the plan must be "sol:<path on the solavg
Volume>". Outputs: the table JSON (and the figure only with --figure, i.e. only if the author asks), written to the output Volume
(outputs/<stamp>/<run id>/, exclusive) and to results/ here (exclusive create). Only fl_vs_32k_* outputs are returned.

  cd ~/probe-distribution-shift-from-sims && uv run modal run <this file> --plan <plan.json> --stamp _final [--figure]
"""

import datetime
import hashlib
import json
import subprocess
import sys
import uuid
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).resolve().parent
REL = "experiments/2026-10-16_fl_vs_32k_appendix"
app = modal.App("probe-shift_aswin-surya_sub-flrender_fl-vs-32k_kill-2026-11-15")
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy==2.3.3", "matplotlib==3.10.6")
SOL_NAME = "probe-shift_aswin-surya_sub-q35solavg_kill-2026-11-13"
SOL = modal.Volume.from_name(SOL_NAME, version=2)  # Stage A records and npz (read only here)
OUT = modal.Volume.from_name("probe-shift_aswin-surya_sub-flrender_fl-vs-32k_kill-2026-11-15", create_if_missing=True)
SCRIPTS = ("flio.py", "make_fl_vs_32k_table.py", "plot_fl_vs_32k.py")
RESERVED = {"plan.json"}


def put(path, b):
    with open(path, "xb") as f:  # exclusive: no input replaces another
        f.write(b)


@app.function(image=image, cpu=2, memory=8192, timeout=1800, retries=0, volumes={"/sol": SOL.read_only(), "/out": OUT})
def run(scripts: dict, files: dict, sol_records: dict, stamp: str, git: str, run_id: str, figure: bool) -> dict:
    import tempfile

    root = Path(tempfile.mkdtemp())
    res = root / "results"
    res.mkdir()
    for n, b in scripts.items():
        put(root / n, b)
    for n, b in files.items():
        put(res / n, b)
    for n, p in sol_records.items():
        put(res / n, (Path("/sol") / p).read_bytes())
    before = {p.name for p in res.iterdir()}
    logs = []
    r = subprocess.run([sys.executable, "make_fl_vs_32k_table.py", "--plan", str(res / "plan.json"), "--stamp", stamp, "--res-dir", str(res),
                        "--npz-root", "/sol", "--npz-volume", SOL_NAME, "--git", git], cwd=root, capture_output=True, text=True, check=False)
    logs.append({"script": "make_fl_vs_32k_table.py", "rc": r.returncode, "stdout": r.stdout[-3000:], "stderr": r.stderr[-6000:]})
    tables = sorted(p for p in res.iterdir() if p.name.startswith(f"fl_vs_32k_table{stamp}_") and p.name not in before)
    if figure and r.returncode == 0 and len(tables) == 1:
        sha = hashlib.sha256(tables[0].read_bytes()).hexdigest()
        r = subprocess.run([sys.executable, "plot_fl_vs_32k.py", "--table", tables[0].name, sha, "--stamp", stamp, "--res-dir", str(res), "--git", git],
                           cwd=root, capture_output=True, text=True, check=False)
        logs.append({"script": "plot_fl_vs_32k.py", "rc": r.returncode, "stdout": r.stdout[-3000:], "stderr": r.stderr[-6000:]})
    outs = {p.name: p.read_bytes() for p in res.iterdir() if p.name not in before and p.name.startswith(("fl_vs_32k_table", "fl_vs_32k_fig"))}
    dest = Path("/out/outputs") / stamp.lstrip("_") / run_id
    dest.mkdir(parents=True, exist_ok=False)
    for n, b in outs.items():
        put(dest / n, b)
    put(dest / "_logs.json", json.dumps(logs, indent=1).encode())
    OUT.commit()
    return {"outputs": outs, "logs": logs, "volume_dir": str(dest).removeprefix("/out/")}


def git(*a):
    return subprocess.run(["git", *a], capture_output=True, check=True, cwd=HERE).stdout


@app.local_entrypoint()
def main(plan: str, stamp: str, figure: bool = False):
    pb = Path(plan).read_bytes()
    pl = json.loads(pb)
    head = git("rev-parse", "HEAD").decode().strip()
    dirty = git("status", "--porcelain", "--", *SCRIPTS).decode().strip()
    assert not dirty, f"commit the scripts first:\n{dirty}"
    scripts = {n: git("show", f"HEAD:{REL}/{n}") for n in SCRIPTS}
    lname = Path(pl["lengths"]["file"]).name
    side = {"draws_attestation": pl["draws_attestation"], "estimator_selection": pl["estimator_selection"],
            **{f"excess_authority.{w}": e for w, e in (pl.get("excess_authority") or {}).items()}}
    rnames = [Path(r["file"]).name for r in pl["records"].values()]
    names = [lname, *rnames, *(Path(e["file"]).name for e in side.values())]  # checked BEFORE any dict is built: a duplicate never drops an input
    if len(set(names)) != len(names) or RESERVED & set(names) or set(SCRIPTS) & set(names):
        sys.exit(f"input names must be distinct and not reserved: {names}")
    files = {"plan.json": pb, lname: (HERE / "results" / lname).read_bytes()}
    sol = {}
    for w, r in pl["records"].items():
        if not r["source"].startswith("sol:"):
            sys.exit(f"{w}: record source must be sol:<path on {SOL_NAME}> (records never pass through the laptop), got {r['source']!r}")
        sol[Path(r["file"]).name] = r["source"][4:].lstrip("/")
    for what, e in side.items():  # small side JSONs (attestation, selection, PANEL records): sol: read in the container, git: a committed file
        nm, src = Path(e["file"]).name, e.get("source", "")
        if src.startswith("sol:"):
            sol[nm] = src[4:].lstrip("/")
        elif src.startswith("git:"):
            _, cm, path = src.split(":", 2)
            files[nm] = git("show", f"{cm}:{path}")
        else:
            sys.exit(f"{what}: source must be sol:<path> or git:<commit>:<path>, got {src!r}")
    run_id = datetime.datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%H%M%S") + "_" + uuid.uuid4().hex[:8]
    out = run.remote(scripts, files, sol, stamp, head, run_id, figure)
    for n, b in sorted(out["outputs"].items()):
        with open(HERE / "results" / n, "xb") as f:  # exclusive create: never overwrite an earlier output
            f.write(b)
        print("wrote", HERE / "results" / n, hashlib.sha256(b).hexdigest())
    log = {"plan": {"path": str(Path(plan).resolve()), "sha256": hashlib.sha256(pb).hexdigest()}, "script_commit": head, "app": app.name,
           "volume_dir": out["volume_dir"], "logs": out["logs"], "outputs": {n: hashlib.sha256(b).hexdigest() for n, b in out["outputs"].items()}}
    with open(HERE / "results" / f"run_log{stamp}_{run_id}.json", "x") as f:
        f.write(json.dumps(log, indent=1))
    for lg in out["logs"]:
        print(lg["script"], "rc", lg["rc"], lg["stdout"][-400:], lg["stderr"][-1500:])
    if any(lg["rc"] for lg in out["logs"]):
        sys.exit("failed (see the run log)")
