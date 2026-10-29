"""Which image lookups did each pinned reference's start resolve? (#82 amendment 32.) For each reference, find the set of
images whose registry lookup answered at A's start: the set for which control-tower's own strip_build_sections, on a
settings copy with its lookups answered from setup_devbox's verified digests for that set only, writes a
.compose-nobuild-*.yml that reproduces A's recorded contract digest. Read-only on the live settings; $0. On the
devbox, with the September tree on PYTHONPATH and CONTROL_TOWER_SETTINGS_DIR=~/phase2/settings:

  check_mixed_forms.py <pilot run dir> <settings copy, its .compose-* files deleted> <images.json> <out.json>

Writes {task key: [image refs resolved at A's start]} for run_pilot.py's `digest_sets` (PHASE2_DIGEST_SET), and
refuses if any reference matches no set: the environment then differed in more than this generated file.
"""

import hashlib
import itertools
import json
import os
import stat
import sys
from pathlib import Path

import yaml
from control_tower.sandbox import docker as ctd
from control_tower.utils.versioning import expand_compose_vars
from inspect_ai.log import read_eval_log
import runner.arenas.reference_contract as rc
from runner.arenas.linuxarena import linuxarena


def digest(roots, generated: bytes) -> str:
    """rc._tree_digest of the live folder, with the generated compose file's bytes replaced by `generated`."""
    d = hashlib.sha256()
    for label, root in roots:
        d.update(f"root\0{label}\0".encode())
        for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
            rel = path.relative_to(root)
            if any(part in rc._SKIP_DIRS or part.startswith(".compose-sim-fs-") for part in rel.parts):
                continue
            mode, name = stat.S_IMODE(path.lstat().st_mode), rel.as_posix()
            if path.is_symlink():
                d.update(f"link\0{name}\0{mode:o}\0{os.readlink(path)}\0".encode())
            elif path.is_file():
                data = generated if rel.name.startswith(".compose-nobuild-") else path.read_bytes()
                d.update(f"file\0{name}\0{mode:o}\0{len(data)}\0".encode() + data + b"\0")
            elif path.is_dir():
                d.update(f"dir\0{name}\0{mode:o}\0".encode())
    return d.hexdigest()


def main(run_dir: str, copy: str, images_json: str, out: str) -> None:
    verified = json.loads(Path(images_json).read_text())["images"]
    commits = yaml.safe_load((Path(__file__).parent / "config" / "run.yaml").read_text())["env_commits"]
    os.environ.update({f"ENV_IMAGE_TAG_{e.upper()}": c for e, c in commits.items()})  # as run_pilot.py sets them
    sets, missing = {}, []
    for key, ref in sorted((json.loads((Path(run_dir) / "launch.json").read_text()).get("references") or {}).items()):
        md = (read_eval_log(ref["copy"]).samples[0].metadata or {}).get("arena_reference_contract") or {}
        roots = rc._roots(linuxarena(md["arena"]), md["main_id"])
        compose = Path(copy) / "linuxarena" / md["arena"] / "codebase" / "compose.yml"
        services = (yaml.safe_load(compose.read_text())["services"] or {}).values()
        images = sorted({expand_compose_vars(s["image"]) for s in services if isinstance(s, dict) and s.get("image")})
        images = [i for i in images if (verified.get(i) or {}).get("repo_digests")]  # else never answered, as the wrapper
        for subset in (set(c) for n in range(len(images) + 1) for c in itertools.combinations(images, n)):
            ctd._image_digests.clear()
            ctd._resolve_image_digest = lambda r, s=subset: (
                f"{r.rpartition(':')[0]}@{verified[r]['repo_digests'][0].split('@', 1)[1]}" if r in s else None)
            if digest(roots, ctd.strip_build_sections(compose).read_bytes()) == md["contents_sha256"]:
                sets[key] = sorted(subset)
                break
        else:
            missing.append(key)
        print(key, md["arena"], sets.get(key, "NO MATCH"), flush=True)
    assert not missing, f"no lookup set reproduces these references' contracts: {missing}"
    Path(out).write_text(json.dumps({"from_run": str(run_dir), "images_json": images_json, "images_json_sha256":
                                     hashlib.sha256(Path(images_json).read_bytes()).hexdigest(), "sets": sets}, indent=1))
    print(f"-> {out}")


if __name__ == "__main__":
    main(*sys.argv[1:5])
