# Read-only, ON a box: ilc_box.sh's freeze check (same find exclusions), dry; prints FREEZE_OK <n> or FREEZE_DIFF <n> with the diff count.
SRC=/home/aswin/phase2/attack_check_311; MANIFEST=$SRC/config/harness_manifest_fu.tsv; T=$(mktemp -d)
(cd "$SRC" && set -o pipefail && find . -type f ! -path './results/*' ! -path '*/__pycache__/*' ! -path './attempt/results/*' \
  ! -path './config/harness_manifest*.tsv' ! -path './config/pilot_il_*' ! -path './config/tasks_il_*' \
  ! -path './config/pilot_ilrr_*' ! -path './config/tasks_ilrr_*' ! -path './config/ilrr_*' \
  ! -path './config/pilot_ilc_*' ! -path './config/tasks_ilc_*' ! -path './config/ilc_*' \
  ! -path './config/pilot_ilrr2_*' ! -path './config/pilot_ilrr3_*' ! -path './config/tasks_ilrr3_*' ! -path './config/ilrr3_*' -print0 \
  | LC_ALL=C sort -z | xargs -0 sha256sum > "$T/now.tsv")
grep -v '  ./config/harness_manifest' "$MANIFEST" | LC_ALL=C sort > "$T/frozen.tsv"; LC_ALL=C sort "$T/now.tsv" > "$T/now_s.tsv"
n=$(wc -l < "$T/now_s.tsv"); d=$(diff "$T/frozen.tsv" "$T/now_s.tsv" | grep -c '^[<>]')
if [[ $d == 0 ]]; then echo "FREEZE_OK $n manifest_sha256 $(sha256sum < $MANIFEST | cut -c1-16)"; else echo "FREEZE_DIFF $n diff_lines $d"; diff "$T/frozen.tsv" "$T/now_s.tsv" | grep '^[<>]' | awk '{print $1, $3}' | head -5; fi; rm -rf "$T"
