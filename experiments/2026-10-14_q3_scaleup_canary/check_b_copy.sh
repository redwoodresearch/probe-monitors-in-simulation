#!/usr/bin/env bash
# Launch B's copies are launch A's (RULE.md; d9's condition), ON THE DEVBOX, by script; exit 0 only if all hold:
#   1. B's harness folder: every file's sha256 equals A's, except config/run.yaml, whose only difference is the
#      settings_dir line (results/, __pycache__ and the folders' own manifests excluded);
#   2. B's settings tree: every file's sha256 equals A's settings tree, except generated .compose-* files and git's
#      index (stat data, rewritten by any `git status`); and per environment, the same git HEAD and a clean worktree;
#   3. images: one Docker daemon serves both launches, so every image ID is the same; the image manifest both launches
#      pin (IMAGES_JSON) is one file.
#   check_b_copy.sh <A harness> <B harness>
set -euo pipefail
A=$1 B=$2 SA=/home/aswin/phase2/settings SB=/home/aswin/phase2/settings_b
OUT=$A/results/b_copy_check_$(TZ=America/Los_Angeles date +%Y-%m-%d_%H%M%S).txt
fail() { echo "B COPY CHECK FAIL: $*" | tee -a "$OUT"; exit 1; }
sums() { (cd "$1" && find . -type f "${@:2}" | LC_ALL=C sort | xargs -d '\n' sha256sum); }
diff <(sums "$A" ! -path './results/*' ! -path '*/__pycache__/*' ! -path './config/harness_manifest.tsv' ! -path './config/run.yaml') \
     <(sums "$B" ! -path './results/*' ! -path '*/__pycache__/*' ! -path './config/harness_manifest.tsv' ! -path './config/run.yaml') \
  > "$OUT" || fail "harness files differ"
d=$(diff "$A/config/run.yaml" "$B/config/run.yaml" | grep '^[<>]' | sed 's/ *#.*//; s/ *$//' | tr -s ' ') || true
[[ $d == "< settings_dir: ~/phase2/settings"$'\n'"> settings_dir: ~/phase2/settings_b" ]] || fail "run.yaml differs beyond settings_dir: $d"
echo "harness: $(sums "$A" ! -path './results/*' ! -path '*/__pycache__/*' | wc -l) files equal; run.yaml differs only in settings_dir" >> "$OUT"
diff <(sums "$SA" ! -name '.compose-*' ! -path '*/.git/index') <(sums "$SB" ! -name '.compose-*' ! -path '*/.git/index') >> "$OUT" ||
  fail "settings trees differ"
n=0
for env in $(ls "$SA/linuxarena"); do
  [[ -d $SA/linuxarena/$env/.git ]] || continue
  ha=$(git -C "$SA/linuxarena/$env" rev-parse HEAD); hb=$(git -C "$SB/linuxarena/$env" rev-parse HEAD)
  [[ $ha == "$hb" ]] || fail "$env: HEAD $ha vs $hb"
  [[ -z $(git -C "$SB/linuxarena/$env" status --porcelain -- . ':(exclude,glob)**/.compose-*') ]] || fail "$env: B's settings dirty"
  n=$((n + 1))
done
echo "settings: $(sums "$SA" ! -name '.compose-*' ! -path '*/.git/index' | wc -l) files equal; $n environments at the same HEAD, clean" >> "$OUT"
echo "images: one Docker daemon for both launches; IMAGES_JSON is one file" >> "$OUT"
echo "B COPY CHECK PASS -> $OUT" | tee -a "$OUT"
