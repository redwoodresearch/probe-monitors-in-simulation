#!/bin/zsh
# G7b FOLD-MODEL launch wrapper (2e 10-16 ~21:3x): LAUNCH ONLY. The spend guard is a refresh2 watch_set instance on the app id (its own ids file,
# a $3 cap for shapes + main, deadline launch + 2 h), started by refresh2 as soon as the app id is relayed; this script has NO guard loop.
#   run_g7b_foldmodel.sh shapes <stage 1|2> <inputs sha256> <reviewed HEAD>   the keys / shapes read of every pinned file the fold labels read
#   run_g7b_foldmodel.sh check  <stage 1|2> <inputs sha256> <reviewed HEAD>   the catalogue check with --fold-labels (8 public + the stage's labels)
#   run_g7b_foldmodel.sh value  1|2 <inputs sha256> <reviewed HEAD> A|B|R37 [k/n]   the VALUE reproduction on a route C pooled record (A: git a936a5d97,
#                                                                   decompressed sha256 54afad04...; GATES Amendment 18, 10-20; the 171-pool A was
#                                                                   d8fcb945f @ 2981a831..., 2e / 09 22:3x-23:2x) at draw 0 and 20 seeded bootstrap draws
#                                                                   (seed 20261016; the record's per-draw npz), the six headline families x C / R, plus
#                                                                   the two source-score plant calls
# Stage 1 = sol_a_uncut, own_all_rs_uncut_L48; Stage 2 = c_asis_uncut, own_all_rs25_uncut_L48, own_all_rs10_uncut_L48 (literal lists below)
# It refuses an uncommitted checker or wrapper, a HEAD other than the reviewed one, and a checker blob other than the reviewed sha256 (CHK); a
# bounded billing read (120 s, pipefail, a positive row count required) must succeed before the launch; the app id is printed with the time as soon
# as the detached run logs it; after the client exits, a second bounded billing read must succeed; a non-zero client exit is the script's exit.
# NOTE, known limitations (Astra round 2 items 4-8 on a6d60fc9b / cd24be1c7): with the guard removed there is no stop latch, no time-bounded app
# list / app stop, no app discovery when the client dies before logging an id, and no INT / TERM / EXIT traps: the watch_set instance is the guard
# for all of these (2e / 507-prep 21:3x). No test seams: the modal binary, paths and pins are fixed in this file (the stub tests ran on edited
# copies in the scratchpad).
set -e
MODE=$1; STAGE=$2; ISHA=$3; HEADX=$4; REC=${5:-}; PART=${6:-}
[ "$MODE" = shapes ] || [ "$MODE" = check ] || [ "$MODE" = value ] || { echo "usage: run_g7b_foldmodel.sh shapes|check|value 1|2 <inputs sha256> <reviewed HEAD> [A|B|R37 [k/n] for value]"; exit 4; }
[ "$MODE" != value ] || [ "$STAGE" = 1 ] || [ "$STAGE" = 2 ] || { echo "STOP: the value reproduction covers one stage's families (1: the six headline probes; 2: the sensitivity labels)"; exit 4; }
# B's level split (A22, 10-20): value B takes k/n (every n-th level from the k-th; each part its own app and committed result); value A takes it only for
# the A-split equivalence positive control (09 / 2e: 1/2 and 2/2 against the committed unsplit A); R37 takes none
[ -z "$PART" ] || { { [ "${MODE}:${REC}" = value:B ] || [ "${MODE}:${REC}" = value:A ]; } && [[ "$PART" =~ ^[1-9]/[1-9]$ ]] && (( ${PART%/*} <= ${PART#*/} )); } || { echo "STOP: a level part (k/n, k <= n) is for value B (or the A-split control) only"; exit 4; }
# route C (A22, 10-20): the value branch names its record explicitly; each record's commit, path and decompressed sha256 (the checker reads the per-draw
# npz from the record itself and asserts it against PERDRAW_BY_RECORD)
case "${MODE}:${REC}" in
  value:A) RCOMMIT=a936a5d9762763f9feed351aebdeeec2d8114f21; RP_=routec_A_alltasks_CRD_FULL_r2_allorigins_pooledworld_scored_e12_2026-10-19_180327.json.gz
           RS=54afad04422afd288db9b77043713cfdb708adf5c0eaa74499ea3e5dd600082a ;;
  value:B) RCOMMIT=f4e21b0f76acde02d1ae4e7408028d6eaeb559d8; RP_=routec_B_alltasks_CRD_FULL_r2_allorigins_sweep15_pooledworld_scored_e12_2026-10-19_180652.json.gz
           RS=ef5c765716ee338a1e8d207a820eecd5c083ba1d51262ff72361d6d254bf6ad7 ;;
  value:R37) RCOMMIT=55a2fb20e1206b37ffe85656901bcf4aa70edfde; RP_=routec_R37_alltasks_CRD_FULL_r2_allorigins_rule37_pooledworld_scored_e12_2026-10-19_181100.json.gz
           RS=8e6635e7413eb0202e469e3af84af70e72b801de9fc0c9d6b7505848a21807e4 ;;
  value:*) echo "STOP: value needs its record: A, B or R37"; exit 4 ;;
  *:) ;;
  *) echo "STOP: a record argument is for value only"; exit 4 ;;
esac
case "$STAGE" in 1) FLS=sol_a_uncut,own_all_rs_uncut_L48 ;; 2) FLS=c_asis_uncut,own_all_rs25_uncut_L48,own_all_rs10_uncut_L48 ;;
  *) echo "STOP: stage must be 1 or 2"; exit 4 ;; esac
[[ "$ISHA" =~ ^[0-9a-f]{64}$ ]] || { echo "STOP: pass the listing's inputs sha256"; exit 4; }
[[ "$HEADX" =~ ^[0-9a-f]{7,40}$ ]] || { echo "STOP: pass the reviewed HEAD"; exit 4; }
M=/Users/aswinsurya/probe-distribution-shift-from-sims/.venv/bin/modal
SP=/private/tmp/claude-501/-Users-aswinsurya-probe-distribution-shift-from-sims/a683364c-3a60-44cd-bb15-e08d87dd4f9a/scratchpad/g7bfold
W=/private/tmp/claude-501/-Users-aswinsurya-probe-distribution-shift-from-sims/a683364c-3a60-44cd-bb15-e08d87dd4f9a/scratchpad/g7b_rc/wt4  # route C Stage 2 package (branch
# aswin/g7b-foldmodel-routec-stage2, off 3a2fc4822); logs stay in $SP
S=$(date -u -v-3d +%Y-%m-%d); E=$(date -u -v+2d +%Y-%m-%d)
bill() {  # a bounded (<= 7 day) billing read, at most 120 s; prints the row count, or BAD unless the CLI and jq both succeed with a positive count
  setopt localoptions pipefail
  local r
  r=$(perl -e 'alarm 120; exec @ARGV' $M billing report --start $S --end $E --resolution h --json 2>/dev/null \
    | jq -r 'if type=="array" and length>0 then length else "BAD" end' 2>/dev/null) || r=BAD  # pipefail: a non-zero modal or jq exit is BAD
  if [[ "$r" =~ ^[1-9][0-9]*$ ]]; then echo $r; else echo BAD; fi  # an empty read (jq on empty input prints nothing, exit 0) is BAD too
}
cd $W
D=experiments/2026-10-15_e2e_sim_pipeline
F=$D/check_stagea_alltasks.py
CHK=011721ce53730bab7c637f37d989cc4fb131627c00f1c07efa4162ade4bdb1ec  # the checker blob (507-prep NB4: asserted, not only printed); route C Stage 2 POST HOC control fix (09 (a) / (c)), 10-20, was ad99f254...
# Astra r3 #2: every git call must EXIT 0 (a missing or broken .git STOPs; an empty output is never "clean" or "equal"), and both HEADs are 40-hex
GS=$(git status --porcelain --untracked-files=all -- $F $D/run_g7b_foldmodel.sh) || { echo "STOP: git status failed"; exit 3; }
[ -z "$GS" ] || { echo "STOP: uncommitted checker or wrapper"; exit 3; }
H0=$(git rev-parse HEAD) || { echo "STOP: git rev-parse HEAD failed"; exit 3; }
H1=$(git rev-parse --verify --quiet "$HEADX^{commit}") || { echo "STOP: the reviewed HEAD $HEADX does not resolve"; exit 3; }
[[ "$H0" =~ ^[0-9a-f]{40}$ ]] && [[ "$H1" =~ ^[0-9a-f]{40}$ ]] && [ "$H0" = "$H1" ] || { echo "STOP: HEAD ${H0:-?} is not the reviewed $HEADX (${H1:-?})"; exit 3; }
[ "$(shasum -a 256 $F | cut -c1-64)" = "$CHK" ] || { echo "STOP: the checker is not the reviewed blob ($CHK)"; exit 3; }
echo "checker sha256 $CHK (asserted); HEAD $H0 (asserted); mode $MODE; stage $STAGE ($FLS)${REC:+; record $REC ($RS)}${PART:+; level part $PART}"
NFS=$D/results/check_stagea_inputs_alltasks_CRD_FULL_r2_allorigins_e12_2026-10-16_113538_pwcatalogue_2026-10-16_200659.json  # the G7b run's class lists
STR=$(jq -r '.nofilter_source.structural_class_prefixes | join(",")' $NFS); NST=$(jq -r '.nofilter_source.nonstructural_class_prefixes | join(",")' $NFS)
b0=$(bill) || b0=BAD; [[ "$b0" =~ ^[0-9]+$ ]] || { echo "STOP: billing read failed ($S..$E)"; exit 2; }  # a count, never just "not BAD"
echo "billing before: $b0 rows ($S..$E)"
STAMP=$(date +%Y-%m-%d_%H%M%S)
LOG=$SP/run_g7b_foldmodel_${MODE}${REC:+_$REC}${PART:+_part${PART/\//of}}_s${STAGE}_$STAMP.log
EXTRA=(); if [ "$MODE" = shapes ]; then EXTRA=(--shapes); fi
NFA=a05ea8959:experiments/2026-10-15_fpr_sweep_507_prep/results/nofilter_exclusions_2026-10-16_012305.json:f154cad6df7e5018485617ab5a22644fbb03549e3b315355b81312764bbcb65f
if [ "$MODE" = value ]; then  # a route C pooled scored record ($REC: its own meta pins the manifest, overlaps, per-set companion and per-draw npz)
  PYTHONPATH=src $M run --detach $F --commit $RCOMMIT \
    --record experiments/2026-10-15_fpr_sweep_507_prep/results/routec_records/$RP_ \
    --record-sha $RS --nofilter $NFA --structural "$STR" --nonstructural "$NST" \
    --fold-labels $FLS --launch --inputs-sha $ISHA ${PART:+--level-part} ${PART:+$PART} > $LOG 2>&1 &
else
  # route C Stage 1 (10-20): the rcPS rerun per-set record (git b5290895f .json.gz, decompressed 87319480...) with --pw-catalogue on manifest 5ed5407b at
  # the manifest's own code commit 4feb6694b (its meta.git, which the catalogue rules require: chk l.2212 / l.3652): the S1 that stands (585ec2169).
  # A rerun at f2d47f19f (ruling A) STOPped at shapes on that rule (10a030831); option B covers the 2-file tree_sha difference instead (A20)
  PYTHONPATH=src $M run --detach $F --commit b5290895f \
    --record experiments/2026-10-15_fpr_sweep_507_prep/results/alltasks_CRD_FULL_r2_allorigins_e12_2026-10-19_132210.json.gz \
    --record-sha 873194801ac2ab3e1bb79a523b5ca6af5db641ac6e458ae5592cb78e90584347 --nofilter $NFA --structural "$STR" --nonstructural "$NST" \
    --pw-catalogue "4feb6694b|xfer/fpr507prep/records/results/alltasks_CRD_FULL_r2_allorigins_pooledworld_catalogue_e12_2026-10-18_112818_manifest.json@5ed5407b66728c211769a275ae95f6995b29c71ffd9aff53b906ae766272f03a|pooled_all_origins,paper_cells" \
    --fold-labels $FLS --launch --inputs-sha $ISHA $EXTRA > $LOG 2>&1 &
fi
CLIENT=$!
echo "launched $(date '+%Y-%m-%d %H:%M:%S %Z'); client pid $CLIENT; log $LOG"
app=""
appid() {  # the first Modal app id in the log, anchored (507-prep NB-c): "ap-" plus >= 8 alphanumerics, not inside a longer token
  perl -ne 'if (/(?<![A-Za-z0-9_-])(ap-[A-Za-z0-9]{8,})(?![A-Za-z0-9_-])/) { print "$1\n"; exit }' $LOG 2>/dev/null || true
}
while [ -z "$app" ] && kill -0 $CLIENT 2>/dev/null; do  # print the app id the moment the run logs it (relayed at once for the watch_set instance)
  app=$(appid)
  [ -n "$app" ] || sleep 2
done
[ -n "$app" ] || app=$(appid)
echo "APP ID ${app:-NONE (the client exited first; find it by name in modal app list)} at $(date '+%H:%M:%S %Z')"
set +e
wait $CLIENT
RC=$?
set -e
grep -v '^[│├└]' $LOG | grep '"shapes"\|"all_equal"\|"n_compared"\|"n_mismatches"\|"pass"\|"domain_missing"\|"fired"\|files_with_findings\|pooled_with_findings\|manifest_findings\|ap-\|Error\|Traceback\|STOP' | cut -c1-4000
echo "exit $RC; log $LOG; $(date '+%H:%M:%S %Z')"
b1=$(bill) || b1=BAD; echo "billing after: $b1 rows"
[[ "$b1" =~ ^[0-9]+$ ]] || { echo "STOP: the billing read after the run failed ($S..$E)"; [ $RC -ne 0 ] || exit 2; }  # 507-prep NB4: b1 validated
[ $RC -eq 0 ] || exit $RC
