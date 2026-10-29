"""Render every final-build figure in ONE Modal CPU container (render only; no recomputation). The laptop never ships record bytes in the call.

Record sources in the plan:
  "sol:<path>"            read inside the container from the solavg Volume (Stage A writes its records to xfer/fpr507prep/records/...): no upload
  "git:<commit>:<path>"   or a local path: uploaded ONCE to the render Volume, content-addressed (records/by_sha/<sha256>), skipped if present
The scripts shipped are the COMMITTED versions at HEAD (git show), so each figure records the commit that drew it; each script re-checks its
records' sha256. Outputs and a log are written to the render Volume (outputs/<stamp>/, exclusive per stamp) and, if the client is still
connected, also to results/ locally (exclusive create). With "detach": true the call is spawned and the laptop may disconnect; fetch later:

  cd ~/probe-distribution-shift-from-sims && uv run modal run --detach <this file> --plan <plan.json>        # run
  cd ~/probe-distribution-shift-from-sims && uv run modal run <this file> --plan <plan.json> --mode fetch    # copy outputs/<stamp>/ to results/
plan.json: {"stamp": "_final", "detach": true, "comparator_overrides": {}, "draw_extra": [], "title_banner": "", "build": {}, "records": {"<basename>": "<source>"},
            "figures": [{"script": "plot_alltasks_bars.py", "args": ["--record", "<basename>", "<sha256>", "--commit", "<c>", "--stamp", "_final"]}]}
comparator_overrides (render-time config, e.g. {"own_a": "delta_full"}) win over the records' meta.primary_comparator; default none.
"""

import datetime
import hashlib
import io
import json
import subprocess
import sys
import uuid
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
REL = "experiments/2026-10-15_fpr_sweep_507_prep"
app = modal.App("probe-shift_aswin-surya_sub-routec-figs_render_kill-2026-11-14")  # route C's own description; the OUT Volume keeps its name
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy==2.3.3", "matplotlib==3.10.6")
SOL = modal.Volume.from_name("probe-shift_aswin-surya_sub-q35solavg_kill-2026-11-13", version=2)  # Stage A's records (read only here)
OUT = modal.Volume.from_name("probe-shift_aswin-surya_sub-render_final-figs_kill-2026-11-13", create_if_missing=True)


def _draw(scripts: dict, records: dict, figures: list, git: str, env: dict) -> tuple:
    """Run the figure scripts in one fresh folder (records read from the Volumes); returns (logs, new output files)."""
    import os
    import tempfile

    root = Path(tempfile.mkdtemp()) / "exp"
    (root / "results").mkdir(parents=True)
    for n, b in scripts.items():
        (root / n).write_bytes(b)
    for n, src in records.items():
        (root / "results" / n).write_bytes((Path("/" + src["vol"]) / src["path"]).read_bytes())
    before = {p.name for p in (root / "results").iterdir()}
    logs = []
    for fg in figures:
        r = subprocess.run([sys.executable, fg["script"], *fg["args"], "--git", git], cwd=root, capture_output=True, text=True,
                           env=os.environ | env)
        logs.append({"script": fg["script"], "args": fg["args"], "rc": r.returncode, "stdout": r.stdout[-3000:], "stderr": r.stderr[-6000:]})
        if r.returncode and env.get("FIG_KEEP_GOING") != "1":
            break  # fail loudly: later figures are not drawn (plan keep_going: draw the rest, still exit non-zero listing every failure)
    return logs, {p.name: p.read_bytes() for p in (root / "results").iterdir() if p.name not in before}


@app.function(image=image, cpu=2, memory=4096, timeout=900, retries=0, volumes={"/sol": SOL, "/out": OUT})
def render_one(scripts: dict, records: dict, fg: dict, git: str, env: dict) -> dict:
    """One figure in its own container (plan field parallel: the render driver fans these out inside Modal)."""
    SOL.reload()
    OUT.reload()
    logs, outs = _draw(scripts, records, [fg], git, env)
    return {"log": logs[0], "outs": outs}


@app.function(image=image, cpu=2, memory=4096, timeout=1800, retries=0, volumes={"/sol": SOL, "/out": OUT})
def render(scripts: dict, records: dict, figures: list, git: str, stamp: str, env: dict, run_id: str, identity: dict) -> dict:
    """records: basename -> {"vol": "sol"|"out", "path": <path on that volume>}. Outputs go to outputs/<stamp>/<run_id>/ (unique per call, so
    concurrent containers never share a folder; Codex r10) with _identity.json naming the plan sha and script commit. Plan field parallel
    (Aswin via 2e, 10-16): this one spawned driver fans out one render_one container per figure inside Modal (drop-safe) and is the only
    writer of the Volume folder; otherwise the figures run here in order."""
    SOL.reload()
    OUT.reload()
    dest = Path("/out/outputs") / stamp.lstrip("_") / run_id
    dest.mkdir(parents=True, exist_ok=False)
    if env.get("FIG_PARALLEL") == "1":
        res = list(render_one.starmap([(scripts, records, fg, git, env) for fg in figures], return_exceptions=True))
        logs, outs = [], {}
        for fg, r_ in zip(figures, res):  # in plan order; a crashed container is a failed figure, never a silent gap
            if isinstance(r_, BaseException):
                logs.append({"script": fg["script"], "args": fg["args"], "rc": -1, "stdout": "", "stderr": f"render_one failed: {r_!r}"[-6000:]})
                continue
            logs.append(r_["log"])
            clash = set(outs) & set(r_["outs"])
            if clash:
                logs.append({"script": fg["script"], "args": fg["args"], "rc": -2, "stdout": "", "stderr": f"output names written twice: {sorted(clash)}"})
            outs |= {n: b for n, b in r_["outs"].items() if n not in clash}
    else:
        logs, outs = _draw(scripts, records, figures, git, env)
    for n, b in outs.items():
        with open(dest / n, "xb") as f:
            f.write(b)
    with open(dest / "_logs.json", "x") as f:
        f.write(json.dumps(logs, indent=1))
    with open(dest / "_identity.json", "x") as f:
        f.write(json.dumps(identity | {"run_id": run_id, "outputs_sha256": {n: hashlib.sha256(b).hexdigest() for n, b in outs.items()}}, indent=1))
    OUT.commit()
    return {"outputs": outs, "logs": logs, "volume_dir": f"outputs/{stamp.lstrip('_')}/{run_id}"}


def git(*a):
    return subprocess.run(["git", *a], capture_output=True, check=True, cwd=HERE).stdout


def write_local(pl, head, records_meta, logs, outs, t0, how):
    log = {"plan": pl, "script_commit": head, "app": app.name, "started_pt": t0, "records": records_meta, "logs": logs, "outputs": {}, "how": how}
    for n, b in sorted(outs.items()):
        with open(RES / n, "xb") as f:  # exclusive create: never overwrite an earlier render
            f.write(b)
        log["outputs"][n] = hashlib.sha256(b).hexdigest()
        print("wrote", RES / n)
    stamp = datetime.datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%H%M%S")
    with open(RES / f"render_log{pl['stamp']}_{stamp}.json", "x") as f:
        f.write(json.dumps(log, indent=1))
    bad = [lg for lg in logs if lg["rc"]]
    for lg in logs:
        print(lg["script"], "rc", lg["rc"])
    if bad or len(logs) != len(pl["figures"]):
        sys.exit(f"render failed: {[lg['script'] for lg in bad] if bad else 'incomplete'}\n" + "\n".join(f"[{lg['script']} {lg['args'][lg['args'].index('--name') + 1] if '--name' in lg['args'] else ''}] "
                                                                                                    + lg["stderr"][-600:] for lg in bad))  # Codex r51


@app.local_entrypoint()
def main(plan: str, mode: str = "run", run_id: str = ""):
    plan_bytes = Path(plan).read_bytes()
    if b"TO_FILL" in plan_bytes or b"<FILL" in plan_bytes or b'"<' in plan_bytes:  # a template is never rendered (checked before the parse)
        sys.exit(f"{plan} still has TO_FILL, <FILL or <...> placeholders")
    pl = json.loads(plan_bytes)
    if isinstance(pl.get("estimator"), dict) and pl["estimator"].get("record") not in pl.get("records", {}):  # Codex ports r8: ship it
        sys.exit(f"{plan}: the estimator's selection record {pl['estimator'].get('record')!r} is not among the plan's records")
    plan_sha = hashlib.sha256(plan_bytes).hexdigest()
    t0 = datetime.datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d %H:%M:%S %Z")
    if mode == "fetch":  # copy a detached render's outputs to results/, only if they were produced by THIS plan (Codex r10)
        assert run_id, "--run-id is required for fetch (printed by the detached run)"
        d = f"outputs/{pl['stamp'].lstrip('_')}/{run_id}"
        files = {e.path.rsplit("/", 1)[-1]: e.path for e in OUT.listdir(d)}
        ident = json.loads(b"".join(OUT.read_file(files.pop("_identity.json"))))
        if ident.get("plan_sha256") != plan_sha:
            sys.exit(f"{d} was produced by plan {ident.get('plan_sha256')}, not this plan ({plan_sha}): refusing to fetch")
        logs = json.loads(b"".join(OUT.read_file(files.pop("_logs.json"))))
        outs = {n: b"".join(OUT.read_file(p)) for n, p in files.items()}
        bad = {n for n, b in outs.items() if hashlib.sha256(b).hexdigest() != ident["outputs_sha256"].get(n)} | (set(ident["outputs_sha256"]) - set(outs))
        if bad:
            sys.exit(f"fetched outputs differ from the run's identity record: {sorted(bad)}")
        write_local(pl, ident.get("script_commit"), ident.get("records"), logs, outs, t0, f"fetched from {OUT.name}:{d}")
        return
    head = git("rev-parse", "HEAD").decode().strip()
    names = {"figio.py", "postfig.py"} | {f["script"] for f in pl["figures"]}
    dirty = git("status", "--porcelain", "--", *names).decode().strip()
    assert not dirty, f"commit the scripts before rendering:\n{dirty}"
    scripts = {n: git("show", f"HEAD:{REL}/{n}") for n in names}
    records, meta, upload = {}, {}, {}
    for base, src in pl["records"].items():
        if src.startswith("sol:"):
            records[base] = {"vol": "sol", "path": src[4:].lstrip("/")}
            meta[base] = {"source": src}
            continue
        if src.startswith("git:"):
            _, cm, path = src.split(":", 2)
            b = git("show", f"{cm}:{path}")
        else:
            b = Path(src).expanduser().read_bytes()
        sha = hashlib.sha256(b).hexdigest()
        records[base] = {"vol": "out", "path": f"records/by_sha/{sha}"}
        meta[base] = {"source": src, "sha256": sha, "volume_path": records[base]["path"]}
        upload[sha] = b
    if upload:  # content-addressed: upload each distinct record once; later renders reuse it
        try:
            have = {e.path.rsplit("/", 1)[-1] for e in OUT.listdir("records/by_sha")}
        except modal.exception.NotFoundError:  # first upload: the folder does not exist yet
            have = set()
        todo = {s: b for s, b in upload.items() if s not in have}
        if todo:
            with OUT.batch_upload() as bu:
                for s, b in todo.items():
                    bu.put_file(io.BytesIO(b), f"/records/by_sha/{s}")
        print(f"records: {len(upload)} content-addressed, {len(todo)} uploaded, {len(upload) - len(todo)} already on {OUT.name}")
    env = {"FIG_COMPARATOR_OVERRIDES": json.dumps(pl.get("comparator_overrides") or {}), "FIG_DRAW_EXTRA": json.dumps(pl.get("draw_extra") or []),
           "FIG_UNION_NOTE": pl.get("union_note") or "",  # e.g. "independent check pending" for the shared-pool Sol (a)/c_asis addendum
           "FIG_TITLE_BANNER": pl.get("title_banner") or "",  # e.g. tonight's DRAFT stamp (0d 10-16); every figure title carries it
           "FIG_BUILD": json.dumps(pl.get("build") or {}),
           "FIG_KEEP_GOING": "1" if pl.get("keep_going") else "",  # speed (2e 10-16): one failing figure does not block the rest
           "FIG_PARALLEL": "1" if pl.get("parallel") else "",  # Aswin via 2e (10-16): one container per figure, fanned out by the driver
           "FIG_COMPARATOR": pl.get("comparator") or "",  # run_mean (09's lead) | run1, for the post pairs, sweep and signed figures
           "FIG_ESTIMATOR": (json.dumps(pl["estimator"]) if isinstance(pl.get("estimator"), dict) else pl.get("estimator") or "")}  # A | B, the run-mean estimator (09 @ 3aaf425ef); the excess is always (A)  # the build profile (full-length build: probe keys, training lengths, length label)
    run_id = datetime.datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%H%M%S") + "_" + uuid.uuid4().hex[:8]
    identity = {"plan_sha256": plan_sha, "plan_file": str(Path(plan).resolve()), "script_commit": head, "records": meta, "started_pt": t0}
    if pl.get("detach"):
        fc = render.spawn(scripts, records, pl["figures"], head, pl["stamp"], env, run_id, identity)
        print(json.dumps({"render_call": fc.object_id, "run_id": run_id, "script_commit": head, "volume_dir": f"outputs/{pl['stamp'].lstrip('_')}/{run_id}",
                          "fetch": f"uv run modal run {Path(__file__).resolve()} --plan {Path(plan).resolve()} --mode fetch --run-id {run_id}"}))
        return
    res = render.remote(scripts, records, pl["figures"], head, pl["stamp"], env, run_id, identity)
    write_local(pl, head, meta, res["logs"], res["outputs"], t0, f"returned by the call (also on {OUT.name}:{res['volume_dir']})")
