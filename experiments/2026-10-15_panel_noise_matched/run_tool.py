"""Run one whitelisted PANEL script that reads RESULT VALUES on Modal CPU, not on the laptop (0d's final rule, 10-16 via 2e). The local entrypoint sends the
committed script and exactly the files it reads (by script and argument position; record files must sit under results/) to one Modal function, which runs
it in a copy of the repo layout and returns the files it created; the local side writes them under results/ (write_once: exclusive, identical bytes kept,
no symlink escape). The return value, or the failure, is also saved to the Volume (detach-safe; fetch_panel_output.py re-materialises it by app id).

  PYTHONPATH=src modal run --detach experiments/2026-10-15_panel_noise_matched/run_tool.py --script report_coheadline.py --args "<repo-relative paths and args>"
"""

import hashlib
import json
import shlex
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).parent
DIR = "experiments/2026-10-15_panel_noise_matched"
RES = f"{DIR}/results"
VOL = "probe-shift_aswin-surya_sub-q35solavg_kill-2026-11-13"
APP = "probe-shift_aswin-surya_sub-panelnm_kill-2026-11-14"
SPEC = {"report_coheadline.py": {"file_args": [0], "sidecars": False}, "report_plant_rows.py": {"file_args": [0], "sidecars": False},
        "report_pscmp.py": {"file_args": [0], "sidecars": False},
        "check_regression.py": {"file_args": [0, 1], "sidecars": True}}  # which argv positions are records the script reads, and whether it reads <stem>.png / _captions.json

app = modal.App(APP)
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy==2.3.3").add_local_python_source("probe_shift")
SOL = modal.Volume.from_name(VOL, version=2)


@app.function(image=image, cpu=(2, 2), memory=(4096, 4096), timeout=600, retries=0, scaledown_window=2, volumes={"/sol": SOL})
def tool(script: str, args: list, files: dict, ctx: dict) -> dict:
    """Run DIR/<script> with args from a temporary repo root holding exactly the sent files; return the files it created, its exit code and full output.
    Any failure (staging, a timeout, collection) is saved to the Volume with whatever output exists, then re-raised."""
    import sys
    import tempfile
    import traceback

    from probe_shift.panel_out import save_out

    cfg = {"script": script, "args": args}
    cp = None
    try:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / RES).mkdir(parents=True, exist_ok=True)
            for rel, b in files.items():
                p = root / rel
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(b)
            before = {p for p in root.rglob("*") if p.is_file()}
            cp = subprocess.run([sys.executable, str(root / DIR / script), *args], cwd=root, capture_output=True, text=True, timeout=540, check=False)
            new = {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file() and p not in before}
    except (Exception, SystemExit) as e_:
        src_ = cp if cp is not None else e_  # the finished run's output when the failure came after it (collection, cleanup), else a timeout's partial output
        part_ = {k: getattr(src_, k) for k in ("stdout", "stderr") if isinstance(getattr(src_, k, None), (str, bytes))}
        part_ |= {"returncode": cp.returncode} if cp is not None else {}
        save_out(SOL, "/sol", ctx["out_name"], {"context": ctx, "cfg": cfg, "exception": traceback.format_exc(), **part_})
        raise
    out = {"returncode": cp.returncode, "stdout": cp.stdout, "stderr": cp.stderr, "files": new}
    return out | {"volume_copy": save_out(SOL, "/sol", ctx["out_name"], {"context": ctx, "cfg": cfg, "out": out})}


def safe_rel(root, rel, under):
    """A repo-relative path with no absolute part, no '..' and no symlink anywhere on it, lying under <under>; returns the absolute path or STOPs."""
    q = Path(rel)
    if q.is_absolute() or ".." in q.parts or not str(q).startswith(under + "/"):
        raise SystemExit(f"STOP: {rel} must be repo-relative, without '..', under {under}/")
    p, cur = root / q, root
    for part in q.parts:
        cur = cur / part
        if cur.is_symlink():
            raise SystemExit(f"STOP: {rel} passes through a symlink ({cur})")
    return p


def write_under(root, rel, data):
    """Create root/rel (rel under RES) through directory file descriptors opened with O_NOFOLLOW from the repo root, so no symlink can redirect a directory
    or the file between check and write (Codex runner r2): exclusive create; an existing file with the SAME bytes is kept, different bytes STOP."""
    import os

    parts = Path(rel).parts
    fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
    try:
        for i, name in enumerate(parts[:-1]):
            if i >= len(Path(RES).parts):  # below results/: create missing directories
                try:
                    os.mkdir(name, 0o755, dir_fd=fd)
                except FileExistsError:
                    pass
            nfd = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd)
            fd = nfd
        try:
            ffd = os.open(parts[-1], os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o644, dir_fd=fd)
        except FileExistsError:
            ffd = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW, dir_fd=fd)
            with os.fdopen(ffd, "rb") as fh:
                if fh.read() != data:
                    raise SystemExit(f"STOP: {rel} exists with different content (never overwritten)") from None
            print(f"{rel}: already present with identical bytes (kept)")
            return
        with os.fdopen(ffd, "wb") as fh:
            fh.write(data)
    finally:
        os.close(fd)


def write_outputs(cfg_name, cfg, out, ctx):
    """The created files into results/ (validated relative paths, then written through no-follow directory descriptors), then the script's output; a
    non-zero exit STOPs after it."""
    root = Path(subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output=True, check=True, text=True, cwd=HERE).stdout.strip()).resolve()
    for rel, b in sorted(out["files"].items()):
        safe_rel(root, rel, RES)  # the path's shape (relative, no '..', under results/)
        write_under(root, rel, b)
        print(f"wrote {rel} (sha256 {hashlib.sha256(b).hexdigest()})")
    print(out["stdout"])
    if out["returncode"]:
        print(out["stderr"])
        raise SystemExit(f"STOP: {cfg['script']} exited {out['returncode']}")


@app.local_entrypoint()
def main(script: str, args: str = ""):
    git = lambda *a: subprocess.run(["git", *a], capture_output=True, check=True, cwd=HERE).stdout
    if git("status", "--porcelain", "--untracked-files=no").strip():
        raise SystemExit("uncommitted tracked changes")
    if script not in SPEC:
        raise SystemExit(f"STOP: {script} is not whitelisted ({sorted(SPEC)})")
    for f_ in (Path(__file__).name, script):  # the runner and the script are committed files (Codex runner r1)
        if subprocess.run(["git", "ls-files", "--error-unmatch", f_], capture_output=True, cwd=HERE, check=False).returncode:
            raise SystemExit(f"STOP: {f_} is not a tracked file")
    root = Path(git("rev-parse", "--show-toplevel").decode().strip()).resolve()
    argv = shlex.split(args)
    sp = SPEC[script]
    if len(argv) <= max(sp["file_args"]):
        raise SystemExit(f"STOP: {script} needs record arguments at positions {sp['file_args']}")
    files = {f"{DIR}/{script}": (HERE / script).read_bytes()}
    for i in sp["file_args"]:  # exactly the files the script reads, each under results/ (reports write beside their input)
        p = safe_rel(root, argv[i], RES)
        side = [p.with_suffix(".png"), p.with_name(p.stem + "_captions.json")] if sp["sidecars"] else []
        for q in [p] + side:
            if not q.is_file():
                raise SystemExit(f"STOP: {q.relative_to(root)} is missing")
            safe_rel(root, str(q.relative_to(root)), RES)
            files[str(q.relative_to(root))] = q.read_bytes()
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    if not app.app_id:
        raise SystemExit("STOP: no app id at launch (the Volume copy needs one)")
    name = Path(script).stem
    ctx = {"kind": "tool", "cfg_name": name, "stamp": stamp, "git": git("rev-parse", "HEAD").decode().strip(), "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
           "tool_script_sha256": hashlib.sha256(files[f"{DIR}/{script}"]).hexdigest(), "app_id": app.app_id, "out_name": f"tool_{name}_{stamp}_{app.app_id}"}
    print(json.dumps({"app_id": ctx["app_id"], "volume_copy_name": ctx["out_name"], "files_sent": sorted(files)}))
    out = tool.remote(script, argv, files, ctx)
    print(json.dumps({"app_id": ctx["app_id"], "volume_copy": out.get("volume_copy")}))
    write_outputs(name, {"script": script, "args": argv}, {k: v for k, v in out.items() if k != "volume_copy"}, ctx)
