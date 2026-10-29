"""Can the September environment digests be reproduced from the v2-tag env checkouts?

Every September sample records `arena_reference_contract.contents_sha256`: a hash of the env's `codebase/` folder,
file names, permission modes and bytes (sce `runner.arenas.reference_contract`; `.git`, `__pycache__`,
`node_modules` and `.compose-sim-fs-*` are skipped). No env repo commit reproduces it, yet the September file lists
(the simulator's codebase index, which hides only `/.compose-` files) equal the v2-tag checkouts exactly, and real
`cat` reads match their bytes. So the difference must be hidden `.compose-*` files, permission modes, or both.

Hidden file: with `SKIP_DOCKER_BUILD=1` (the fleet sets it), control-tower's `strip_build_sections` writes
`.compose-nobuild-<md5(compose path)[:8]>.yml` beside `compose.yml`: build sections removed, images pinned to registry
digests (env images by the env repo's HEAD commit tag). The real honest runs' sandbox config names exactly such a file.
Modes: a checkout made under umask 002 has group-writable files and folders (664/775), not 644/755.

For each env and September machine path, this regenerates the hidden file with control-tower's own code (digests from
Docker Hub's tag API, which matches `docker buildx imagetools` and avoids its anonymous rate limit; push dates
recorded, since a tag re-pushed after 2026-09-17 has a different digest now) and hashes each combination of
{hidden file or not} x {umask 022 or 002}. Writes results/contract_reproduction_<stamp>.json.

With --write, for each env whose fleet digest is reproduced, writes that file into the local env folder (so the local
folder is byte-identical to the September fleet's) and confirms the sims code's own digest now equals September's.

Run inside the September snapshot (no model calls; without --write it only reads):
  cd ~/simulated-control-evals-sept && CONTROL_TOWER_SETTINGS_DIR=$PWD/.settings SKIP_DOCKER_BUILD=1 uv run python \\
    ~/probe-shift-sims/experiments/2026-10-06_astra_sim_honest/run_reproduce_contracts.py --references <references_*.json>
"""

import argparse
import hashlib
import json
import os
import shutil
import stat
import subprocess
import tempfile
import urllib.request
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import yaml
from control_tower.sandbox import docker

from runner.arenas.linuxarena import linuxarena
from runner.arenas.reference_contract import _tree_digest, reference_contract

HERE = Path(__file__).parent
CONFIG = json.loads((HERE / "config" / "astra_honest.json").read_text())
MACHINES = {  # where each September cohort's env folders lived (sample metadata: codebase_root, sandbox config)
    "overlap800": "/opt/linuxbench/.settings/linuxarena/{env}/codebase/compose.yml",
    "base200": "/home/jeffrey/linuxarena-astra-n200/.settings/linuxarena/{env}/codebase/compose.yml",
}
PUSHED: dict[str, str | None] = {}
BUILDX = docker._resolve_image_digest


def hub_digest(image_ref: str) -> str | None:
    """`repo@sha256:...` for a Docker Hub `repo:tag`, as `docker buildx imagetools inspect` reports it, else None."""
    if "@sha256:" in image_ref:
        return image_ref
    name, _, tag = image_ref.rpartition(":")
    if not name or "/" in tag:
        return None
    if "." in name.split("/")[0]:  # another registry (ghcr.io, quay.io): control-tower's own buildx lookup
        digest = BUILDX(image_ref)
        PUSHED[image_ref] = "resolved by buildx" if digest else "buildx could not resolve; left as tag"
        return digest
    repo = name if "/" in name else f"library/{name}"
    try:
        with urllib.request.urlopen(f"https://hub.docker.com/v2/repositories/{repo}/tags/{tag}", timeout=60) as r:
            body = json.load(r)
    except urllib.error.HTTPError as e:
        PUSHED[image_ref] = f"HTTP {e.code}"
        return None
    if not body.get("digest") and len(body.get("images") or []) == 1:  # single-platform image: its manifest digest
        PUSHED[image_ref] = body.get("tag_last_pushed")
        return f"{name}@{body['images'][0]['digest']}"
    if not body.get("digest"):  # the Hub API omits it for some tags; ask the registry as control-tower does
        digest = BUILDX(image_ref)
        PUSHED[image_ref] = f"{body.get('tag_last_pushed')} (digest via buildx: {'ok' if digest else 'failed'})"
        return digest
    PUSHED[image_ref] = body.get("tag_last_pushed")
    return f"{name}@{body['digest']}"


def nobuild_text(compose: Path) -> str:
    """What `strip_build_sections` would write for this checkout, captured instead of written."""
    captured = {}
    write, resolve = docker.write_temp_compose, docker._resolve_image_digest
    docker.write_temp_compose = lambda path, tag, data: captured.setdefault("text", yaml.dump(data, default_flow_style=False))
    docker._resolve_image_digest = hub_digest
    try:
        docker.strip_build_sections(compose)
    finally:
        docker.write_temp_compose, docker._resolve_image_digest = write, resolve
    return captured["text"]


def set_modes(root: Path, group_writable: bool) -> None:
    """Permissions as a checkout made under umask 022 (group_writable False) or 002 (True) has them, whatever the
    umask of the machine that made this one: group and other write cleared, then group write added back for 002."""
    for path in [root, *root.rglob("*")]:
        if path.is_symlink() or ".git" in path.relative_to(root).parts:
            continue
        mode = stat.S_IMODE(path.lstat().st_mode) & ~0o022
        path.chmod(mode | 0o020 if group_writable else mode)


def digest_with(codebase: Path, extra: dict[str, str], group_writable: bool) -> str:
    with tempfile.TemporaryDirectory() as tmp:
        copy = Path(tmp) / "codebase"
        # The bare checkout: generated `.compose-*` files already in the local folder (a previous --write, a sim
        # run's scratch compose) are left out, so every variant starts from the same tree.
        shutil.copytree(codebase, copy, symlinks=True, ignore=shutil.ignore_patterns(".git", ".compose-*"))
        for name, text in extra.items():
            (copy / name).write_text(text)
        set_modes(copy, group_writable)
        return _tree_digest((("environment/0", copy),))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--references", type=Path, required=True, help="results/references_*.json (September contracts)")
    p.add_argument("--write", action="store_true",
                   help="write the regenerated file into each local env folder whose fleet digest it reproduces")
    args = p.parse_args()
    selection = json.loads(args.references.read_text())
    sept = selection["september_contracts"]  # env -> cohort -> digests
    selected = {}  # env -> the September digests of the references actually chosen
    for r in selection["references"].values():
        selected.setdefault(r["environment"], set()).add(r["sept_contract"]["contents_sha256"])
    assert os.environ.get("SKIP_DOCKER_BUILD") == "1", "set SKIP_DOCKER_BUILD=1, as the fleet did"

    rows = []
    for env in sorted(sept):
        codebase = Path(linuxarena(env).env().codebase_path)
        head = subprocess.run(["git", "-C", str(codebase), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        text = nobuild_text(codebase / "compose.yml")
        images = [s.get("image") for s in (yaml.safe_load(text).get("services") or {}).values()]
        row = {"environment": env, "env_commit": head, "images": images,
               "image_last_pushed": {k: v for k, v in PUSHED.items() if env in k or "/" not in k.split(":")[0]},
               "september": sept[env], "variants": {}}
        for machine, template in MACHINES.items():
            name = f".compose-nobuild-{hashlib.md5(template.format(env=env).encode()).hexdigest()[:8]}.yml"
            for hidden in (False, True):
                for group_writable in (False, True):
                    got = digest_with(codebase, {name: text} if hidden else {}, group_writable)
                    key = f"{machine}{'+' + name if hidden else ''}{' umask002' if group_writable else ' umask022'}"
                    row["variants"][key] = {"digest": got, "matches": [c for c, h in sept[env].items() if got in h]}
        fleet = f".compose-nobuild-{hashlib.md5(MACHINES['overlap800'].format(env=env).encode()).hexdigest()[:8]}.yml"
        reproduced = "overlap800" in row["variants"][f"overlap800+{fleet} umask022"]["matches"]
        if args.write and reproduced:
            # Make the local folder byte-identical to the September fleet's (its file, umask-022 permissions), then
            # confirm with the sims code itself.
            (codebase / fleet).write_text(text)
            set_modes(codebase, group_writable=False)
            local = reference_contract(linuxarena(env), "_")["contents_sha256"]
            # Every chosen reference of this env must carry exactly this digest, or its build will be refused.
            assert selected[env] == {local}, f"{env}: local digest {local[:12]}, chosen references carry {sorted(selected[env])}"
            row["written"] = str(codebase / fleet)
        stamped = env in CONFIG["september_digest_not_reproducible"]
        if args.write and stamped:
            # Its references carry the local digest of a umask-022 checkout (stamped on the laptop). Give this folder
            # the same permissions so that digest does not depend on the machine's umask.
            set_modes(codebase, group_writable=False)
            row["normalized_for_stamped_digest"] = reference_contract(linuxarena(env), "_")["contents_sha256"]
        rows.append(row)
        hits = {k: v["matches"] for k, v in row["variants"].items() if v["matches"]}
        print(f"{env}: {hits or 'no variant matches'}{' | wrote ' + fleet if row.get('written') else ''}", flush=True)
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y%m%d-%H%M%S")
    out = HERE / "results" / f"contract_reproduction_{stamp}.json"
    with open(out, "x") as f:  # results/ is append-only
        f.write(json.dumps(rows, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
