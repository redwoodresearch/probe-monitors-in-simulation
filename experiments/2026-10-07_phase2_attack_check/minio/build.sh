#!/usr/bin/env bash
# Build quay.io/minio/minio:RELEASE.2025-09-07T16-13-09Z locally (see Dockerfile for why). Runs ON THE DEVBOX:
#   PY=<python> bash minio/build.sh <out-dir>   # writes <out-dir>/minio_build.json (every input's sha256, the image id)
# PY is the interpreter for the provenance writer: setup_devbox.sh passes the sims snapshot's uv-managed one.
# Fails unless the release binary matches the release's sha256 AND its minisign signature under MinIO's public key.
set -euo pipefail
R=RELEASE.2025-09-07T16-13-09Z
MINIO_PUBKEY=RWTx5Zr1tiHQLwG9keckT0c45M3AGeHD6IvimQHpyRywVWGbP1aVSGav  # MinIO's release key (Dockerfile.release)
CURL_RELEASE=v8.11.0  # the newest static-curl release when MinIO built its image (2025-09-07); MinIO used "latest"
UBI=registry.access.redhat.com/ubi9/ubi-micro
OUT=$(realpath "${1:?out dir}")
HERE=$(cd "$(dirname "$0")" && pwd)
PY=${PY:?set PY to the uv-managed interpreter, as setup_devbox.sh does}
W=$(mktemp -d)
trap 'rm -rf "$W"' EXIT
cd "$W"

gh=https://github.com/minio/minio/releases/download/$R
curl -fsSL -o minio "$gh/minio.linux-amd64.$R"
curl -fsSL -o minio.sha256sum "$gh/minio.linux-amd64.$R.sha256sum"
curl -fsSL -o minio.minisig "$gh/minio.linux-amd64.$R.minisig"
echo "$(cut -d' ' -f1 minio.sha256sum)  minio" | sha256sum -c -
command -v minisign >/dev/null || sudo apt-get install -y -qq minisign >/dev/null
minisign -Vm minio -x minio.minisig -P "$MINIO_PUBKEY"
chmod +x minio
curl -fsSL -o docker-entrypoint.sh "https://raw.githubusercontent.com/minio/minio/$R/dockerscripts/docker-entrypoint.sh"
chmod +x docker-entrypoint.sh
curl -fsSL -o curl "https://github.com/moparisthebest/static-curl/releases/download/$CURL_RELEASE/curl-amd64"
chmod +x curl
cp /etc/ssl/certs/ca-certificates.crt .
cp "$HERE/Dockerfile" .

docker pull -q "$UBI:latest" >/dev/null
UBI_DIGEST=$(docker image inspect "$UBI:latest" --format '{{index .RepoDigests 0}}' | cut -d@ -f2)
docker build -q --build-arg UBI_MICRO_DIGEST="$UBI_DIGEST" -t "quay.io/minio/minio:$R" . >/dev/null
version=$(docker run --rm "quay.io/minio/minio:$R" --version | head -1)
[[ "$version" == *"$R"* ]] || { echo "built image reports '$version', not $R" >&2; exit 1; }

mkdir -p "$OUT"
"$PY" - "$OUT/minio_build.json" "$R" "$UBI_DIGEST" "$CURL_RELEASE" "$version" <<'PY'
import hashlib, json, subprocess, sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
out, release, ubi, curl_release, version = sys.argv[1:]
sha = {f: hashlib.sha256(Path(f).read_bytes()).hexdigest()
       for f in ("minio", "docker-entrypoint.sh", "curl", "ca-certificates.crt", "Dockerfile")}
image = subprocess.run(["docker", "image", "inspect", f"quay.io/minio/minio:{release}", "--format", "{{.Id}}"],
                       capture_output=True, text=True, check=True).stdout.strip()
Path(out).write_text(json.dumps({"built": datetime.now(ZoneInfo("America/Los_Angeles")).isoformat(timespec="seconds"),
                                 "tag": f"quay.io/minio/minio:{release}", "image_id": image, "reports": version,
                                 "ubi_micro_digest": ubi, "static_curl_release": curl_release,
                                 "checks": ["sha256 vs release .sha256sum", "minisign vs MinIO public key"],
                                 "sha256": sha}, indent=1))
print(f"-> {out}")
PY
