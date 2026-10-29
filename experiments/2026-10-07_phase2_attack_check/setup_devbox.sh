#!/usr/bin/env bash
# Prepare the devbox for the attack check. Runs ON THE DEVBOX, from ~/phase2 (where devbox.sh rsyncs this folder):
#   bash setup_devbox.sh
# 1. Our own copy of the LinuxArena settings at the September commits (config/run.yaml), without the generated
#    .compose-* files. The sims batch's folders stay untouched: a real run writes .compose-nobuild-*.yml into its settings
#    folder, which would change the environment digest the sims batch's references were checked against.
# 2. Every image the five environments need, pulled once at the pinned tags, so runs start from the local cache and a
#    Docker Hub rate limit can only cost a warning (control-tower falls back to cached images when a pull fails).
# 3. MinIO, rebuilt locally (minio/build.sh). A cached copy is reused only if its image ID is the one recorded in
#    ~/phase2/verified_images.tsv right after a build whose sha256 and signature checks passed; otherwise it is rebuilt.
# 4. The internet simulator at the pinned tag. 5. `nginx:alpine`, which control-tower's internet simulator starts with
#    a bare `docker run` outside any compose file (its proxy), so a missing image is fatal. It is pulled by the digest
#    pinned in config/run.yaml (what Docker Hub served for the tag on 2026-10-07, from AWS's public mirror of Docker's
#    official images, which costs no Docker Hub quota) and tagged locally; a cached `nginx:alpine` is reused only if it
#    carries that digest.
# 6. The sims code the runs use, one tree per simulator (config/run.yaml, simulators): a copy of each base checkout
#    plus the policy-limits patch, so the bases (the sims batches' trees) stay untouched. Built once; a copy whose src/
#    doesn't hash to its src_sha256 stops the setup (move it aside by hand) rather than being overwritten.
# Other images already in the local cache are not pulled again (their tags are commit or version pins): every pull
# spends Docker Hub's 100-per-hour anonymous quota, and a launch's own digest lookups spend the rest (the first launch
# failed on exactly this).
# Writes results/devbox_setup_<stamp>.json: env commits, every image's id and repo digests, the MinIO build record.
set -euo pipefail
cd "$(dirname "$0")"
SETTINGS=~/phase2/settings
STAMP=$(TZ=America/Los_Angeles date +%Y-%m-%d_%H%M%S)
OUT=results/devbox_setup_$STAMP
mkdir -p "$OUT"
SIMS=~/sims/simulated-control-evals-sept  # the September base: its settings folder and Python run this setup
PY="$SIMS/.venv/bin/python"  # the snapshot's uv-managed interpreter, for every helper below (never the host's)
cfg() { "$PY" -c "import json, yaml; print(json.dumps(yaml.safe_load(open('config/run.yaml'))$1))"; }
str_cfg() { cfg "$1" | tr -d '"'; }
fail() { echo "$1" >&2; exit 1; }
[[ $(str_cfg "['simulators']['september']['base_tree']") == "~/sims/simulated-control-evals-sept" ]] ||
  fail "SIMS is not the september base_tree"
src_hash() {  # run_attack_check.py's own hash of a tree's src/
  "$PY" -c "import sys, pathlib, run_attack_check as r; print(r.src_sha256(pathlib.Path(sys.argv[1])))" "$1"
}
for sim in $(cfg "['simulators']" | "$PY" -c 'import json,sys; print(" ".join(json.load(sys.stdin)))'); do
  BASE=$(str_cfg "['simulators']['$sim']['base_tree']"); BASE="${BASE/#\~/$HOME}"
  TREE=$(str_cfg "['simulators']['$sim']['tree']"); TREE="${TREE/#\~/$HOME}"
  [[ $(src_hash "$BASE") == $(str_cfg "['simulators']['$sim']['base_src_sha256']") ]] ||
    fail "$sim: $BASE is not the pinned base"
  if [[ ! -d $TREE ]]; then
    rsync -a --exclude .venv --exclude .git --exclude .settings --exclude __pycache__ "$BASE/" "$TREE/"
    patch -d "$TREE" -p1 --forward < config/sept_snapshot_policy_limits.patch
    for extra in $(cfg "['simulators']['$sim'].get('extra_patches', [])" | "$PY" -c 'import json,sys; print(" ".join(json.load(sys.stdin)))'); do
      patch -d "$TREE" -p1 --forward < "$extra"
    done
  fi
  [[ $(src_hash "$TREE") == $(str_cfg "['simulators']['$sim']['src_sha256']") ]] ||
    fail "$sim: $TREE/src does not hash to src_sha256"
done

if [[ ! -d $SETTINGS ]]; then
  cp -a "$SIMS/.settings" "$SETTINGS"
  find "$SETTINGS/linuxarena" -maxdepth 3 -name '.compose-*' -delete
fi
ENVS=$(cfg "['envs']" | "$PY" -c 'import json,sys; print(" ".join(json.load(sys.stdin)))')
for env in $ENVS; do
  want=$(cfg "['env_commits']['$env']" | tr -d '"')
  have=$(git -C "$SETTINGS/linuxarena/$env" rev-parse HEAD)
  [[ $have == "$want" ]] || { echo "$env is at $have, config pins $want" >&2; exit 1; }
  # A real run writes .compose-* files into its settings folder; they are run output, not a change to the settings.
  dirty=$(git -C "$SETTINGS/linuxarena/$env" status --porcelain -- . ':(exclude,glob)**/.compose-*')
  [[ -z $dirty ]] || { echo "$env settings folder has changes: $dirty" >&2; exit 1; }
done

IMAGES=()
for env in $ENVS; do
  key=ENV_IMAGE_TAG_$(echo "$env" | tr a-z A-Z)
  commit=$(cfg "['env_commits']['$env']" | tr -d '"')
  while read -r image; do IMAGES+=("$image"); done < <(
    env "$key=$commit" docker compose -f "$SETTINGS/linuxarena/$env/codebase/compose.yml" config --images)
done
IS=$(cfg "['internet_simulator_image']" | tr -d '"')
MINIO=quay.io/minio/minio:RELEASE.2025-09-07T16-13-09Z
cached() { docker image inspect "$1" >/dev/null 2>&1; }
for image in $(printf '%s\n' "${IMAGES[@]}" "$IS" | sort -u); do
  [[ $image == "$MINIO" ]] && continue
  cached "$image" && continue
  echo "pull $image"
  docker pull -q "$image" >/dev/null
done
VERIFIED=~/phase2/verified_images.tsv  # "<tag><TAB><image id>", appended right after an image's checks pass
image_id() { docker image inspect "$1" --format '{{.Id}}' 2>/dev/null; }
if ! { id=$(image_id "$MINIO") && grep -qxF "$MINIO	$id" "$VERIFIED" 2>/dev/null; }; then
  PY="$PY" bash minio/build.sh "$OUT"  # fails unless the binary's sha256 and minisign signature check out
  printf '%s\t%s\n' "$MINIO" "$(image_id "$MINIO")" >> "$VERIFIED"
fi
NGINX=nginx:alpine
NGINX_SRC=$(cfg "['nginx_proxy_image']" | tr -d '"')
if ! docker image inspect $NGINX --format '{{json .RepoDigests}}' 2>/dev/null | grep -qF "${NGINX_SRC#*@}"; then
  docker pull -q "$NGINX_SRC" >/dev/null  # by digest, so the content is the pinned image or the pull fails
  docker tag "$NGINX_SRC" $NGINX
fi
IMAGES+=("$NGINX" "$MINIO")

"$PY" - "$OUT/images.json" "$SETTINGS" $ENVS -- $(printf '%s\n' "${IMAGES[@]}" "$IS" | sort -u) <<'PY'
import json, subprocess, sys
out, settings, *rest = sys.argv[1:]
split = rest.index("--")
envs, images = rest[:split], rest[split + 1:]
def inspect(image):
    r = subprocess.run(["docker", "image", "inspect", image, "--format", "{{json .}}"], capture_output=True, text=True)
    assert r.returncode == 0, f"{image} is not in the local cache: {r.stderr}"
    d = json.loads(r.stdout)
    return {"id": d["Id"], "repo_digests": d.get("RepoDigests", []), "created": d.get("Created")}
commits = {e: subprocess.run(["git", "-C", f"{settings}/linuxarena/{e}", "rev-parse", "HEAD"], capture_output=True,
                             text=True, check=True).stdout.strip() for e in envs}
json.dump({"settings_dir": settings, "env_commits": commits, "images": {i: inspect(i) for i in images}},
          open(out, "w"), indent=1)
print(f"-> {out} ({len(images)} images, all in the local cache)")
PY
