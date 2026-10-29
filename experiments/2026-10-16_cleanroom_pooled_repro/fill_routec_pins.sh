#!/usr/bin/env bash
# Fill config/routec_pins.json from 2e's fill-sheet pins (a1, 2026-10-18). Local git + jq + shasum only; prints shas, never contents.
# Copies the AUTH file and pooled A (.json.gz) out of git by commit into CR_CFG under their committed basenames, checks every sha
# (pooled A: the .gz file's sha is recorded, the decompressed sha must equal 2e's), writes the pins, and fails if any TBD is left.
#
#   CR_CFG=<pinned input folder> ./fill_routec_pins.sh AUTH_COMMIT AUTH_PATH AUTH_SHA256 A_COMMIT A_PATH A_DECOMP_SHA256 \
#                                                      CAT_PATH CAT_SHA256 MAN_PATH MAN_SHA256
set -euo pipefail
[ $# -eq 10 ] || { echo "need 10 args (see header)"; exit 2; }
AUTH_COMMIT=$1 AUTH_PATH=$2 AUTH_SHA=$3 A_COMMIT=$4 A_PATH=$5 A_DECOMP=$6 CAT_PATH=$7 CAT_SHA=$8 MAN_PATH=$9 MAN_SHA=${10}
HERE=$(cd "$(dirname "$0")" && pwd)
CFG="${ROUTEC_CFG:-$HERE/config/routec_pins.json}"  # override only for a dry test on a copy
: "${CR_CFG:?set CR_CFG}"
git -C "$HERE" fetch -q origin

pull() {  # commit path -> CR_CFG/basename; refuse to overwrite a different file
  local out="$CR_CFG/$(basename "$2")" tmp
  tmp=$(mktemp "$CR_CFG/.pull.XXXXXX")
  git -C "$HERE" cat-file -e "$1:$2" || { echo "not in git: $1:$2"; rm -f "$tmp"; exit 1; }
  git -C "$HERE" show "$1:$2" > "$tmp"
  if [ -f "$out" ] && ! cmp -s "$tmp" "$out"; then echo "exists with different content: $out"; rm -f "$tmp"; exit 1; fi
  mv "$tmp" "$out"
  shasum -a 256 "$out" | cut -d' ' -f1
}

auth_sha=$(pull "$AUTH_COMMIT" "$AUTH_PATH")
[ "$auth_sha" = "$AUTH_SHA" ] || { echo "AUTH sha $auth_sha != fill sheet $AUTH_SHA"; exit 1; }
a_sha=$(pull "$A_COMMIT" "$A_PATH")
a_dec=$(gzip -dc "$CR_CFG/$(basename "$A_PATH")" | shasum -a 256 | cut -d' ' -f1)
[ "$a_dec" = "$A_DECOMP" ] || { echo "pooled A decompressed sha $a_dec != fill sheet $A_DECOMP"; exit 1; }

tmp=$(mktemp "$HERE/config/.fill.XXXXXX")
jq --arg an "$(basename "$AUTH_PATH")" --arg ac "$AUTH_COMMIT" --arg ap "$AUTH_PATH" --arg as "$auth_sha" \
   --arg pn "$(basename "$A_PATH")" --arg pc "$A_COMMIT" --arg pp "$A_PATH" --arg ps "$a_sha" --arg pd "$A_DECOMP" \
   --arg cp "$CAT_PATH" --arg cs "$CAT_SHA" --arg mp "$MAN_PATH" --arg ms "$MAN_SHA" '
  .authorized_overlaps = {name: $an} | .pooled_A = {name: $pn, decompressed_sha256: $pd}
  | .catalogue = {path: $cp, sha256: $cs} | .manifest = {path: $mp, sha256: $ms}
  | .files = ([.files[] | select((.[2] | split("/") | last) != $an and (.[2] | split("/") | last) != $pn)] + [[$as, $ac, $ap], [$ps, $pc, $pp]])
' "$CFG" > "$tmp"
if jq -e 'del(.what) | tostring | test("TBD")' "$tmp" > /dev/null; then echo "TBD left in config"; rm -f "$tmp"; exit 1; fi
mv "$tmp" "$CFG"
echo "AUTH $(basename "$AUTH_PATH") sha256 $auth_sha"
echo "pooled A $(basename "$A_PATH") gz sha256 $a_sha, decompressed $a_dec"
echo "config sha256 $(shasum -a 256 "$CFG" | cut -d' ' -f1)"
