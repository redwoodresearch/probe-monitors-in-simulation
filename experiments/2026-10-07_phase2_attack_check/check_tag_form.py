"""Check fix (B) before a launch, on the devbox, with no_digest_lookup/ first on PATH (no model calls, no containers):

  cd ~/phase2/sims-sept-policy-limits-v2 && PATH=~/phase2/<launch copy>/no_digest_lookup:$PATH \\
      CONTROL_TOWER_SETTINGS_DIR=~/phase2/settings PYTHONPATH=$PWD/src \\
      ~/sims/simulated-control-evals-sept/.venv/bin/python ~/phase2/<launch copy>/check_tag_form.py <images.json> <env>...

Per environment: the digest lookup is refused; two rewrites in separate processes (control-tower caches lookups per
process) give byte-identical tag-form `.compose-nobuild-*.yml` (no `@sha256:` image); every image the rewrite names
resolves locally to the image ID setup_devbox verified (images.json); and whether the compose bind-mounts host paths
into a service (what the agent could see of the codebase folder). Exits nonzero on any failure.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

REWRITE = ("from pathlib import Path; import sys; from control_tower.sandbox.docker import strip_build_sections; "
           "print(strip_build_sections(Path(sys.argv[1])))")


def main(images_json: str, envs: list[str]) -> int:
    from control_tower.utils.versioning import expand_compose_vars  # the launch's own ${VAR:-default} expansion

    verified = json.loads(Path(images_json).read_text())["images"]
    commits = yaml.safe_load((Path(__file__).parent / "config" / "run.yaml").read_text())["env_commits"]
    os.environ.update({f"ENV_IMAGE_TAG_{e.upper()}": c for e, c in commits.items()})  # as run_pilot sets them
    lookup = subprocess.run(["docker", "buildx", "imagetools", "inspect", "alpine:3"], capture_output=True, text=True)
    bad = [] if lookup.returncode != 0 else ["the digest lookup is NOT refused: is no_digest_lookup/ first on PATH?"]
    settings = Path(os.environ["CONTROL_TOWER_SETTINGS_DIR"]) / "linuxarena"
    for env in envs:
        compose = settings / env / "codebase" / "compose.yml"
        outputs = []
        for _ in range(2):  # separate processes: no shared lookup cache
            path = subprocess.run([sys.executable, "-c", REWRITE, str(compose)], capture_output=True, text=True,
                                  check=True).stdout.strip().splitlines()[-1]
            outputs.append(Path(path).read_bytes())
        data = yaml.safe_load(outputs[0])
        images = [s.get("image", "") for s in (data.get("services") or {}).values() if isinstance(s, dict)]
        mounts = [v for s in (data.get("services") or {}).values() if isinstance(s, dict) for v in s.get("volumes", [])
                  if isinstance(v, str) and v.startswith((".", "/"))]
        resolved = {}
        for image in images:
            ref = expand_compose_vars(image)
            got = subprocess.run(["docker", "image", "inspect", "--format", "{{.Id}}", ref], capture_output=True,
                                 text=True)
            resolved[ref] = got.stdout.strip() if got.returncode == 0 else None
        ok_bytes, ok_form = outputs[0] == outputs[1], not any("@sha256:" in i for i in images)
        ids = {r: i for r, i in resolved.items() if i}
        unverified = [r for r, i in ids.items() if verified.get(r, {}).get("id") != i]  # its own recorded ID
        missing = [r for r, i in resolved.items() if not i]
        print(json.dumps({"env": env, "identical_rewrites": ok_bytes, "tag_form": ok_form, "images": resolved,
                          "not_in_images_json": unverified, "not_local": missing, "host_mounts": mounts}))
        if not (ok_bytes and ok_form) or unverified or missing:
            bad.append(env)
    print("FAILED: " + ", ".join(bad) if bad else "all checks passed")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2:]))
