"""Hashes and metadata only (no file contents) of the September simulator tree a box ran, and of the codebase folders the
LinuxArena simulator serves (environment.codebase_path = <settings>/linuxarena/<env>/codebase, runner/arenas/linuxarena.py
via control-tower's settings dir), for q2/df (d9, 2026-10-15). Read-only; run from stdin so it writes nothing on the box:
  python3 - <label> <tree> <base tree> <settings dir>[,<settings dir>...] < sim_tree_manifest.py
Per file: relative path, sha256, size, mode (octal, lstat). src_sha256 exactly as run_attack_check.src_sha256. Per codebase
folder: the reference contract's contents digest exactly as runner/arenas/reference_contract._tree_digest (skips .git,
__pycache__, node_modules and .compose-sim-fs-*; hashes modes, sizes, bytes, symlink targets), the same digest with every
.compose-* file also skipped, and the generated .compose-* files present (name, sha256, size, mode)."""
import hashlib, json, os, stat, sys
from pathlib import Path

label, tree, base, settings_dirs = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4].split(",")
ENVS = ["auto_workflow", "clinical_trial", "file_indexing", "media_processing", "model_registry", "shared_dotfiles",
        "slack_clone", "tiktok", "user_tracking", "web_scraping"]
SKIP = {".git", "__pycache__", "node_modules"}


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def src_sha256(t):
    h = hashlib.sha256()
    for p in sorted(q for q in (t / "src").rglob("*") if q.is_file() and "__pycache__" not in q.parts):
        h.update(str(p.relative_to(t)).encode() + b"\0" + hashlib.sha256(p.read_bytes()).digest())
    return h.hexdigest()


def manifest(root, skip_parts=()):
    rows = []
    for p in sorted(root.rglob("*"), key=lambda q: q.relative_to(root).as_posix()):
        rel = p.relative_to(root)
        if any(part in skip_parts for part in rel.parts) or not (p.is_file() or p.is_symlink()):
            continue
        m = stat.S_IMODE(p.lstat().st_mode)
        rows.append([rel.as_posix(), "link:" + os.readlink(p) if p.is_symlink() else sha(p), p.lstat().st_size, f"{m:o}"])
    return rows


def tree_digest(root, skip_compose_all):
    digest = hashlib.sha256()
    digest.update("root\0environment/0\0".encode())
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root)
        if any(part in SKIP or part.startswith(".compose-sim-fs-") or (skip_compose_all and part.startswith(".compose-"))
               for part in relative.parts):
            continue
        mode = stat.S_IMODE(path.lstat().st_mode)
        name = relative.as_posix()
        if path.is_symlink():
            digest.update(f"link\0{name}\0{mode:o}\0{os.readlink(path)}\0".encode())
        elif path.is_file():
            digest.update(f"file\0{name}\0{mode:o}\0{path.stat().st_size}\0".encode())
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            digest.update(b"\0")
        elif path.is_dir():
            digest.update(f"dir\0{name}\0{mode:o}\0".encode())
    return digest.hexdigest()


out = {"box": label, "tree": str(tree), "src_sha256": src_sha256(tree), "base_tree": str(base),
       "base_src_sha256": src_sha256(base) if (base / "src").is_dir() else None,
       "src_manifest": manifest(tree / "src", ("__pycache__",)),
       "attack_policy_paths": [r for r in manifest(tree / "src", ("__pycache__",))
                               if any(w in r[0].lower() for w in ("attack", "policy", "attacker", "red"))],
       "codebases": {}}
for sd in settings_dirs:
    sdp = Path(os.path.expanduser(sd))
    for env in ENVS:
        root = sdp / "linuxarena" / env / "codebase"
        if not root.is_dir():
            out["codebases"][f"{sd}|{env}"] = None
            continue
        git = sdp / "linuxarena" / env / ".git"
        head = None
        try:
            ref = (git / "HEAD").read_text().strip()
            head = (git / ref.split(" ", 1)[1]).read_text().strip() if ref.startswith("ref:") else ref
        except Exception:  # noqa: BLE001 - recorded as unknown
            head = None
        out["codebases"][f"{sd}|{env}"] = {
            "path": str(root), "env_repo_head": head,
            "contract_contents_sha256": tree_digest(root, False),
            "contents_sha256_without_any_compose_files": tree_digest(root, True),
            "compose_files": [r for r in manifest(root, tuple(SKIP)) if r[0].split("/")[-1].startswith(".compose-")],
            "manifest_skip_git_pycache_node_modules": manifest(root, tuple(SKIP))}
print(json.dumps(out))
