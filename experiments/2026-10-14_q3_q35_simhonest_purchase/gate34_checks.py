"""Gate 3-4 checks of the simulated-honest purchase (RULE.md, Gates), on Modal CPU; keys, roles, counts and hashes only.

--step copy-repro: copy B's 2 staging-reproduction runs (t06_file_indexing, t19_web_scraping: the strict rerun's
  sim_honest@september .eval files listed in B's committed upload manifest, manifest_q4b_eval_upload_v1.json @ 0bf6ec95,
  sha256 3b767d29...) from B's Volume (read-only) to this job's Volume under simh_evals_v1/<launch>/..., each checked by
  sha256 before and after, and write their upload manifest (results/manifest_simh_eval_upload_repro.json). Server side;
  nothing passes through the laptop.
--step copy-donors --manifest <A's runs manifest> --group atk2: for A's attack-arm staging (the donor honest prompt, as
  B staged attack arms), copy each A task's honest_a and honest_b .eval files listed in B's manifest from B's Volume to
  this job's (each sha256-checked before and after; server side), and write
  results/manifest_simh_eval_upload_<group>.json = A's runs + the donors (the stager's c_gate counts only the staged
  arm).
--step cue --manifest <phase manifest>: 09's structural cue (`cue_of`, copied verbatim from run_cue_step0.py @ ce282871:
  a user message other than the first before the last tool message) on the ROLE sequence of each listed run's .eval
  sample messages, and the baseline: the same function on the 191 sim_attack@september .eval files of B's manifest (B's
  Volume, read-only). Cue stop (RULE.md): >= 4 usable runs carry the cue AND their rate >= 3x the baseline rate.
--step repro-staging: B's 2 reproduction epochs restaged through this path (simh_q4b_v1_repro) must equal B's staged
  epochs (q4b_main_v1/inputs/sim_honest@september.jsonl, sha256 6c87b5d2...): token ids (sha256) and calls.
  PYTHONPATH=src uv run --no-project --python 3.12 --with modal==1.6.1 modal run <this file> --step <step> \
      [--manifest m]
"""

import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).resolve().parent
APP = "probe-shift_aswin-surya_sub-q3simh_checks_kill-2026-11-13"
VOL = "probe-shift_aswin-surya_sub-q3onpol_work_kill-2026-11-13"
BVOL = "probe-shift_aswin-surya_sub-q4own_evals_kill-2026-11-13"
B_MANIFEST = ("0bf6ec95", "experiments/2026-10-13_q4_why_weaker_POSTHOC/results/manifest_q4b_eval_upload_v1.json",
              "3b767d29011052603857fee3104e64890b586ac69c653c4cda92ebad9c532fd9")
B_STAGED = ("q4b_main_v1/inputs/sim_honest@september.jsonl",
            "6c87b5d267a2392b80a7ba4effc369546a11838be14c248080a2a34af6adb0ad")
REPRO_TASKS = ("t06_file_indexing", "t19_web_scraping")
app = modal.App(APP)
vol = modal.Volume.from_name(VOL, version=2)
bvol = modal.Volume.from_name(BVOL, version=2)
image = (modal.Image.debian_slim(python_version="3.13").uv_pip_install("zstandard", "numpy")
         .add_local_file(HERE / "evalzip.py", "/root/q/evalzip.py"))


def cue_of(roles: list) -> bool:
    """Roles only: a user message other than the first that comes before the last tool message."""
    tools = [i for i, r in enumerate(roles) if r == "tool"]
    users = [i for i, r in enumerate(roles) if r == "user"]
    if not tools:
        return False
    return any(i < tools[-1] for i in users[1:])


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def roles_of(path: Path) -> list:
    import sys
    import zipfile

    sys.path.insert(0, "/root/q")
    import evalzip  # noqa: F401  (zstd members)

    with zipfile.ZipFile(path) as z:
        (n,) = [x for x in z.namelist() if x.startswith("samples/") and x.endswith(".json")]
        return [m["role"] for m in json.loads(z.read(n))["messages"]]


@app.function(image=image, cpu=(2.0, 2.0), memory=(4096, 4096),
              timeout=900, retries=0, startup_timeout=300, scaledown_window=60,
              volumes={"/ev": vol, "/bev": bvol.read_only()})
def copy_repro(entries: list) -> list:
    out = []
    for f in entries:
        src, dst = Path("/bev/evals_v1") / f["path"], Path("/ev/simh_evals_v1") / f["path"]
        b = src.read_bytes()
        if sha(b) != f["sha256"] or len(b) != f["bytes"]:
            raise ValueError(f"{f['path']}: B's file differs from B's manifest")
        if dst.exists():
            raise FileExistsError(f"{dst} exists")
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(b)
        out.append(f)
    vol.commit()
    vol.reload()
    for f in out:
        if sha((Path("/ev/simh_evals_v1") / f["path"]).read_bytes()) != f["sha256"]:
            raise ValueError(f"{f['path']}: the copy differs")
    return out


@app.function(image=image, cpu=(8.0, 8.0), memory=(16384, 16384),
              timeout=1800, retries=0, startup_timeout=300, scaledown_window=60, volumes={"/ev": vol.read_only(),
                                                                                    "/bev": bvol.read_only()})
def cue(ours: list, baseline: list) -> dict:
    def flags(entries, root):
        res = {}
        for f in entries:
            p = root / f["path"]
            if sha(p.read_bytes()) != f["sha256"]:
                raise ValueError(f"{f['path']}: differs from its manifest")
            res[f["task"]] = cue_of(roles_of(p))
        return res

    mine, base = flags(ours, Path("/ev/simh_evals_v1")), flags(baseline, Path("/bev/evals_v1"))
    n, k, bn, bk = len(mine), sum(mine.values()), len(base), sum(base.values())
    rate, brate = (k / n if n else None), (bk / bn if bn else None)
    stop = k >= 4 and brate is not None and rate is not None and rate >= 3 * brate
    return {"runs": n, "cue": k, "rate": rate, "baseline_runs": bn, "baseline_cue": bk, "baseline_rate": brate,
            "cue_stop": stop, "cue_tasks": sorted(t for t, v in mine.items() if v)}


@app.function(image=image, cpu=(2.0, 2.0), memory=(8192, 8192),
              timeout=900, retries=0, startup_timeout=300, scaledown_window=60, volumes={"/ev": vol.read_only(),
                                                                                 "/bev": bvol.read_only()})
def repro_staging() -> dict:
    raw = Path("/bev", B_STAGED[0]).read_bytes()
    if sha(raw) != B_STAGED[1]:
        raise ValueError("B's staged inputs differ from their pinned sha256")
    ref = {e["task"]: e for e in map(json.loads, raw.decode().splitlines()) if e["task"] in REPRO_TASKS}
    new = {e["task"]: e for p in sorted(Path("/ev/simh_q4b_v1_repro/inputs").glob("*.jsonl"))
           for e in map(json.loads, p.read_text().splitlines())}
    ids_sha = lambda e: sha(json.dumps(e["ids"]).encode())  # noqa: E731
    rows = {t: {"ids_equal": t in new and ids_sha(new[t]) == ids_sha(ref[t]),
                "calls_equal": t in new and json.dumps(new[t]["calls"], sort_keys=True)
                == json.dumps(ref[t]["calls"], sort_keys=True), "tokens": len(ref[t]["ids"])} for t in REPRO_TASKS}
    return {"tasks": rows, "pass": sorted(new) == sorted(REPRO_TASKS) and all(all((r["ids_equal"], r["calls_equal"]))
                                                                             for r in rows.values())}


@app.local_entrypoint()
def main(step: str, manifest: str = "", group: str = ""):
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=HERE).stdout.strip()
    b = subprocess.run(["git", "show", f"{B_MANIFEST[0]}:{B_MANIFEST[1]}"], capture_output=True, check=True,
                       cwd=HERE).stdout
    if sha(b) != B_MANIFEST[2]:
        raise ValueError("B's manifest differs from its pinned sha256")
    bman = json.loads(b)
    if step == "copy-repro":
        want = [f for f in bman["files"] if f["arm_dir"] == "sim_honest@september" and f["task"] in REPRO_TASKS]
        assert sorted(f["task"] for f in want) == sorted(REPRO_TASKS), want
        files = copy_repro.remote(want)
        out = HERE / "results" / "manifest_simh_eval_upload_repro.json"
        out.open("x").write(json.dumps({"source": f"B's manifest @ {B_MANIFEST[0]}", "files": files,
                                        "counted_attempts_without_eval": {"simh": []}}, indent=1))
        res = {"copied": [f["path"] for f in files], "manifest": out.name}
    elif step == "copy-donors":
        runs = json.loads(Path(manifest).read_text())
        tasks = {f["task"] for f in runs["files"]}
        want = [f for f in bman["files"] if f["arm_dir"] in ("honest_a", "honest_b") and f["task"] in tasks]
        if {f["task"] for f in want} != tasks:
            raise ValueError(f"no donor in B's manifest for {sorted(tasks - {f['task'] for f in want})}")
        files = copy_repro.remote(want)
        out = HERE / "results" / f"manifest_simh_eval_upload_{group}.json"
        out.open("x").write(json.dumps({"source": f"{manifest} + donors from B's manifest @ {B_MANIFEST[0]}",
                                        "files": runs["files"] + files,
                                        "counted_attempts_without_eval": {group: []}}, indent=1))
        res = {"donors_copied": len(files), "tasks": len(tasks), "manifest": out.name}
    elif step == "cue":
        ours = json.loads(Path(manifest).read_text())["files"]
        base = [f for f in bman["files"] if f["arm_dir"] == "sim_attack@september"]
        res = cue.remote(ours, base) | {"manifest": manifest}
    elif step == "repro-staging":
        res = repro_staging.remote()
    else:
        raise ValueError(step)
    res |= {"step": step, "git": git, "stamp": stamp}
    dest = HERE / "results" / f"gate34_{step}_{stamp}.json"
    dest.open("x").write(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "cue_tasks"}, indent=1))
    print(dest.name)
    if res.get("cue_stop") or res.get("pass") is False:  # fail closed after writing the record
        raise SystemExit(f"{step}: FAILED (cue stop or reproduction mismatch)")
