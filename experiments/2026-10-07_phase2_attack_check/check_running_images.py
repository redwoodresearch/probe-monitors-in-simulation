"""After each follow-up phase, on the devbox: every container the phase created ran a verified image (#82 amendment
32), so replaying lookups never changed which images ran. Read-only. Start the event log before the phase's launch and
stop it after the launcher ends:

  docker events --filter type=container --filter type=image --filter event=create --filter event=pull \
    --filter event=load --filter event=import --filter event=tag --filter event=untag --filter event=delete \
    --format '{{json .}}' >> <run dir>/docker_events.jsonl &
  python3 check_running_images.py <images.json> <run dir>/docker_events.jsonl [<ref>=<image ID>...]

Checks: (1) every image reference in images.json, and every extra ref=ID (control-tower's locally built control-tower-
sim-scratch:latest, sha256:ba1bed0c7e0e… on 10-08), resolves locally to its verified ID now; (2) the log has no image
pull, load or import, and no tag onto an unverified image ID (every image event is listed); (3) every container created
during the phase used one of those references (or a verified ID); (4) every live container runs a verified image ID.
Exits 1 on any failure, or when the log holds no container at all (then it verified nothing).
"""

import json
import subprocess
import sys

images = json.load(open(sys.argv[1]))["images"]
extra = dict(arg.split("=", 1) for arg in sys.argv[3:])
want = {ref: v["id"] for ref, v in images.items()} | extra  # tag -> verified ID
refs = set(want) | {d for v in images.values() for d in v.get("repo_digests") or []}
ids = set(want.values())


def local_id(ref: str) -> str:
    out = subprocess.run(["docker", "image", "inspect", ref, "--format", "{{.Id}}"], capture_output=True, text=True)
    return out.stdout.strip()


events = [json.loads(line) for line in open(sys.argv[2]) if line.strip()]
created = [e["Actor"]["Attributes"].get("image", "") for e in events if e.get("Type") == "container"]
store = [f"{e.get('Action')} {e['Actor'].get('ID')}" for e in events if e.get("Type") == "image"]
live = subprocess.run(["docker", "ps", "-q"], capture_output=True, text=True, check=True).stdout.split()
live_ids = subprocess.run(["docker", "inspect", "--format", "{{.Image}}", *live], capture_output=True, text=True,
                          check=True).stdout.split() if live else []
report = {
    "refs_changed": sorted(ref for ref, i in want.items() if local_id(ref) != i),
    "image_events": store,
    # a pull, load or import, or any tag onto an unverified image ID, could put an unverified image behind a verified ref
    "store_changed": [s for s in store if s.split()[0] in ("pull", "load", "import")
                      or (s.split()[0] == "tag" and s.split()[1] not in ids)],
    "containers_created": len(created),
    "created_unverified": sorted({ref for ref in created if ref not in refs and ref not in ids}),
    "live_containers": len(live_ids),
    "live_unverified": sorted({i for i in live_ids if i not in ids}),
}
print(json.dumps(report))
bad = report["refs_changed"] or report["store_changed"] or report["created_unverified"] or report["live_unverified"]
sys.exit(1 if bad or not created else 0)
