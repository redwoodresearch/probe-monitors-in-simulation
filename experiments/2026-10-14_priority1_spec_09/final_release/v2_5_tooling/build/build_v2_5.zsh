#!/bin/zsh -f
# build_v2_5.zsh: build a v2.5 release-record DRAFT from the committed v2.4 record by key (jq + git + shasum; no Python). Prints shas and counts only.
#   zsh -f build_v2_5.zsh <resolved rulings json> <out.json>
# Optional, by environment: F78ON=1 turns on the figs78 branch (i). It is accepted only with 09's verification line (GATES L2430 @ ae77708bf,
# "figs78 results commit 854f901b6 verified") present and the CONFIRMED recheck descriptor f78_recheck.json (the relayed constants); the
# record is read from git at 854f901b6 and sha-checked (09 via 2e). Without F78ON the build is branch (ii).
set -e; setopt pipefail; export LC_ALL=C
H=${0:A:h}; D=${H:h}
REPO=/Users/aswinsurya/probe-distribution-shift-from-sims
RES=${1:A}; OUT=${2:A}
V24=$D/inputs/v2.4_record_6a46541d5.json; P18=$D/inputs/provenance_final18_58bf4a3ac.jsonl; P19=$D/inputs/provenance_final19_X_c20699a5d.jsonl
RUL=$H/rulings_v2_5.json; RV=$H/rv_recheck.json
G() { git -C $REPO "$@"; }
[[ -z ${F78:-} && -z ${F78REF:-} && -z ${F78VER:-} ]] || { echo "STOP use F78ON=1 for the figs78 branch (i); the v2.4 F78 inputs are not used here"; exit 2; }
[[ -z ${F78ON:-} || ${F78ON} == 1 ]] || { echo "STOP F78ON must be 1 or unset"; exit 2; }
[[ $(shasum -a 256 $V24 | cut -c1-64) == f14031d65dfe090f53d9e28045a200cff4b875bb41e60f8f7ebdaeba989c2c91 ]] || { echo "STOP v2.4 record sha"; exit 2; }
[[ $(G cat-file blob 6a46541d5:experiments/2026-10-14_priority1_spec_09/final_release/routec_FINAL_release_v2.4_2026-10-21.json | shasum -a 256 | cut -c1-64) == $(shasum -a 256 $V24 | cut -c1-64) ]] || { echo "STOP v2.4 != commit 6a46541d5"; exit 2; }
[[ $(shasum -a 256 $P18 | cut -c1-64) == 13d29fb90cac3e3e* ]] || { echo "STOP v18 sha"; exit 2; }
[[ $(shasum -a 256 $P19 | cut -c1-64) == a8bf8fced33aa6dc91be4a26f5cc862a0115af1c47b8acf1c80ca8f912757fda ]] || { echo "STOP v19 sha"; exit 2; }
[[ $(G cat-file blob c20699a5d:experiments/2026-10-20_routec_numbers_verify/post/provenance_v84_final19_X.jsonl | shasum -a 256 | cut -c1-64) == $(shasum -a 256 $P19 | cut -c1-64) ]] || { echo "STOP v19 != commit c20699a5d"; exit 2; }
# A2 (09): v19 == v18 on every row except v84_sha256; rows 119-121 differ in exactly {anchor, printed, v84_sha256} (field names only)
jq -n -e --slurpfile a $P18 --slurpfile b $P19 '($a | map({(.row | tostring): .}) | add) as $A | ($b | map({(.row | tostring): .}) | add) as $B
  | ([$a[].row] | sort) == ([$b[].row] | sort) and ($a | length) == 251
  and all(range(1; 252); tostring as $n | ([($A[$n] | keys[]), ($B[$n] | keys[])] | unique | map(select(. as $k | $A[$n][$k] != $B[$n][$k])))
        == (if . >= 119 and . <= 121 then ["anchor", "printed", "v84_sha256"] else ["v84_sha256"] end))' > /dev/null || { echo "STOP A2: v19 vs v18 field differences"; exit 2; }
# the RV recheck descriptor: the relayed one (constants), its report sha, its verifier blob sha and parent, its GATES log line
jq -e '(keys == ["file", "gates", "label", "of", "sha256", "verdict"]) and .label == "from agent summary" and .verdict == "CONFIRMED"
       and .sha256 == "3f7770c15a2f44ec216d9a2a929c0117241a45d7ab239505e836b78d4706ed1b" and .of.commit == "afe11c8096efb4acfdf9eca1efa94959c432b41a"
       and .of.parent == "f11453af93dee8a32bd7e4befb90d8a590341033" and .of.sha256 == "d59138c43422e2e50766611e1781737075bc9aaf9d10af1211a30c7deef7e618"
       and .of.path == "experiments/2026-10-20_routec_numbers_verify/results/numbers_verify_2026-10-21_025253.json" and .gates.must == "RECHECK_RV_afe11c809"' $RV > /dev/null \
  || { echo "STOP RV descriptor is not the relayed recheck"; exit 2; }
[[ $(shasum -a 256 "$(jq -r '.file' $RV)" | cut -c1-64) == $(jq -r '.sha256' $RV) ]] || { echo "STOP RV report sha"; exit 2; }
[[ $(G cat-file blob "$(jq -r '.of.commit' $RV):$(jq -r '.of.path' $RV)" | shasum -a 256 | cut -c1-64) == $(jq -r '.of.sha256' $RV) ]] || { echo "STOP RV verifier result sha"; exit 2; }
[[ $(G rev-parse "$(jq -r '.of.commit' $RV)^") == $(jq -r '.of.parent' $RV) ]] || { echo "STOP RV verifier commit parent"; exit 2; }
[[ $(G cat-file blob "$(jq -r '.gates.commit' $RV):experiments/2026-10-14_priority1_spec_09/GATES_routec_2026-10-17.md" | grep -cF -- "$(jq -r '.gates.must' $RV)") == 1 ]] || { echo "STOP RV GATES log line"; exit 2; }
origins() { G branch -r --contains $1 | tr -s ' \t' '\n' | awk '/^origin\//' | sort | jq -R . | jq -s -c . ; }
mkdir -p $H/_work; WORK=$(mktemp -d "$H/_work/build.XXXXXX")
jq -r '.anchors[] | .anchor | "\(.commit)\t\(.file)\t\(.line)\t\(.line_sha256)"' $RES | sort -u | while IFS=$'\t' read -r c f ln lsha; do
  [[ $(G cat-file blob "$c:$f" | sed -n "${ln}p" | tr -d '\n' | shasum -a 256 | cut -c1-64) == $lsha ]] || { echo "STOP an anchor's line sha"; exit 3; }
  G cat-file blob "$c:$f" | sed -n "${ln}p" | tr -d '\n' | jq -R -s -c --arg k "$c:$f:$ln" '{k: $k, v: .}'
done > $WORK/lines.jsonl
V24C=$(G rev-parse 6a46541d5); V19C=$(G rev-parse c20699a5d)
# figs78 branch (i), only with F78ON=1: 09's verified line, the CONFIRMED recheck (constants), the record from git at 854f901b6 (sha-checked)
F78META=null; F78ARG=(--slurpfile F78 /dev/null)
if [[ ${F78ON:-} == 1 ]]; then
  FR=$H/f78_recheck.json; GPF=experiments/2026-10-14_priority1_spec_09/GATES_routec_2026-10-17.md
  jq -e '(keys == ["file", "gates", "label", "of", "sha256", "verdict"]) and .label == "from agent summary" and .verdict == "CONFIRMED"
         and .sha256 == "a0492000d52c225dffbe3a79e9fbb591346a5cbf64fdc2c01d276ad2be239835" and .of.commit == "854f901b606cd8b44f30f87c263b441e7101a11f"
         and .of.path == "experiments/2026-10-20_routec_rerun_v24figs78/results/figs78_map_2026-10-21_033233.json"
         and .of.sha256 == "ee65f33f9a7856e1be6a6c9043345019785e0c395e1ccd0e0ac8e7f9677e35d1" and .gates.commit == "ae77708bf8fa980ec83acc0c456b89a365b95066"' $FR > /dev/null \
    || { echo "STOP the figs78 recheck descriptor is not the relayed one"; exit 2; }
  [[ $(shasum -a 256 "$(jq -r '.file' $FR)" | cut -c1-64) == $(jq -r '.sha256' $FR) && $(grep -c 'RECHECK_FIGS78 CONFIRMED' "$(jq -r '.file' $FR)") -ge 1 ]] || { echo "STOP the figs78 recheck report sha / verdict"; exit 2; }
  [[ $(G rev-parse "$(jq -r '.of.commit' $FR)^") == $(jq -r '.of.parent' $FR) ]] || { echo "STOP the figs78 results commit's parent"; exit 2; }
  [[ $(G cat-file blob "$(jq -r '.gates.commit' $FR):$GPF" | grep -cF -- "$(jq -r '.gates.verified_must' $FR)") == 1 ]] || { echo "STOP 09's figs78 verified line is not on GATES"; exit 2; }
  G cat-file blob "$(jq -r '.of.commit' $FR):$(jq -r '.of.path' $FR)" > $WORK/f78_record.json
  [[ $(shasum -a 256 $WORK/f78_record.json | cut -c1-64) == $(jq -r '.of.sha256' $FR) ]] || { echo "STOP the figs78 record sha"; exit 2; }
  F78C=$(jq -r '.of.commit' $FR)
  F78META=$(jq -n -c --slurpfile fr $FR --argjson o "$(origins $F78C)" '{record: {path: $fr[0].of.path, commit: $fr[0].of.commit, sha256: $fr[0].of.sha256, origin_branches: $o}, recheck: $fr[0]}')
  F78ARG=(--slurpfile F78 $WORK/f78_record.json)
fi
META=$(jq -n -c --arg date "$(TZ=America/Los_Angeles date '+%Y-%m-%d %H:%M:%S %Z')" \
  --arg bsha $(shasum -a 256 $H/build_v2_5.jq | cut -c1-64) --arg wsha $(shasum -a 256 $H/build_v2_5.zsh | cut -c1-64) \
  --arg rsha $(shasum -a 256 $RUL | cut -c1-64) --arg ressha $(shasum -a 256 $RES | cut -c1-64) \
  --arg v24c $V24C --argjson v24o "$(origins $V24C)" --arg v19c $V19C --argjson v19o "$(origins $V19C)" --slurpfile rv $RV --argjson f78 "$F78META" \
  '{date_pt: $date, repo_path: "experiments/2026-10-14_priority1_spec_09/final_release/routec_FINAL_release_v2.5_2026-10-21.json",
    builder: {script_sha256: $bsha, wrapper_sha256: $wsha, rulings_sha256: $rsha, resolved_sha256: $ressha,
              note: "v2.5 = v2.4 + the v2.5 rulings, by key (build_v2_5.jq); laptop jq + git + shasum, no Python"},
    v24: {kind: "release_record", path: "experiments/2026-10-14_priority1_spec_09/final_release/routec_FINAL_release_v2.4_2026-10-21.json", commit: $v24c,
          sha256: "f14031d65dfe090f53d9e28045a200cff4b875bb41e60f8f7ebdaeba989c2c91", origin_branches: $v24o},
    v19: {kind: "writing_provenance", path: "experiments/2026-10-20_routec_numbers_verify/post/provenance_v84_final19_X.jsonl", commit: $v19c,
          sha256: "a8bf8fced33aa6dc91be4a26f5cc862a0115af1c47b8acf1c80ca8f912757fda", origin_branches: $v19o},
    rv_recheck: $rv[0], figs78: $f78}')
jq -n -f $H/build_v2_5.jq --slurpfile V24 $V24 --slurpfile P18 $P18 --slurpfile P19 $P19 --slurpfile RUL $RUL --slurpfile RES $RES --slurpfile LN $WORK/lines.jsonl \
   "${F78ARG[@]}" --argjson META "$META" > $OUT 2> $WORK/build.err || { echo "STOP build_v2_5.jq failed (stderr kept in $WORK/build.err; not printed)"; exit 3; }
print -r -- "BUILT $OUT sha256 $(shasum -a 256 $OUT | cut -c1-64) figs78 branch $([[ ${F78ON:-} == 1 ]] && print i || print ii)"
jq -r '"ROWS \(.printed_items.all_rows | length) FINAL \([.printed_items.all_rows[] | select(.status | startswith("FINAL"))] | length) PRELIMINARY \([.printed_items.all_rows[] | select(.status | startswith("PRELIMINARY"))] | length) NOT_ROUTE_C \([.printed_items.all_rows[] | select(.status | startswith("not route C"))] | length)"' $OUT
