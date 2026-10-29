#!/bin/zsh -f
# Route C file-edit share recount (rcfe) launcher: gates, then the one Modal run. Run it by path from the worktree, at the pushed head:
#   zsh -f experiments/2026-10-20_routec_fileedit_share/launch_routec_fileedit_share.zsh <40-hex HEAD> [gate-only]
# "gate-only" runs every check and the config extraction, then stops before Modal (`modal profile current` is a local read).
# Launch only after the review PASS, 09's key check of PRESTATED v2, the SPEND row, refresh3's ARMED line and 0d's GO.
# It never reads or runs NOTE text; the ten config files and their shas are written here and must also appear in PRESTATED v2.

H=$1
MODE=${2:-run}
W=/Users/aswinsurya/probe-shift-routec-fileedit-share-wt
B=aswin/routec-fileedit-share
E=experiments/2026-10-20_routec_fileedit_share
P=$E/PRESTATED_OUTCOME_2026-10-20_v2.md
P_SHA=6ca76d2471075ba7a18f162b794b6ed4e18c40300b8975b752777f59dc907604
S=$E/run_routec_fileedit_share.py
S_SHA=1afd66ac8ce44c205e808a020cbcb1e5b6153cd2b40a648fbe374aff1e420f6a
L=/Users/aswinsurya/probe-shift-rcfe-logs-2026-10-20
UVM=(uv run --no-project --python 3.12 --with modal==1.6.1 modal)
setopt noclobber

[[ "$H" =~ ^[0-9a-f]{40}$ ]] || { print "STOP: usage: launch_routec_fileedit_share.zsh <40-hex HEAD> [gate-only]"; exit 1; }
[[ "$MODE" == run || "$MODE" == gate-only ]] || { print "STOP: mode must be run or gate-only"; exit 1; }
[[ "${0:A}" == "$W/$E/launch_routec_fileedit_share.zsh" ]] || { print "STOP: run this launcher by its worktree path"; exit 1; }
cd "$W" || { print "STOP: no worktree"; exit 1; }
[[ "$(git symbolic-ref --short HEAD 2>/dev/null)" == "$B" ]] || { print "STOP: the worktree is not attached to $B"; exit 1; }
[[ "$(git rev-parse HEAD)" == "$H" ]] || { print "STOP: HEAD != $H"; exit 1; }
GS=$(git status --porcelain --untracked-files=all) || { print "STOP: git status failed"; exit 1; }
[[ -z "$GS" ]] || { print "STOP: the worktree is not clean"; exit 1; }
[[ "$(git rev-parse "HEAD:$E/launch_routec_fileedit_share.zsh")" == "$(git hash-object "$E/launch_routec_fileedit_share.zsh")" ]] \
  || { print "STOP: this launcher differs from its committed blob"; exit 1; }
[[ "$(git ls-remote origin "refs/heads/$B" | cut -f1)" == "$H" ]] || { print "STOP: $H is not origin's $B"; exit 1; }
[[ "$(git show "HEAD:$P" | shasum -a 256 | cut -d' ' -f1)" == "$P_SHA" ]] || { print "STOP: PRESTATED v2 is not 6ca76d24..."; exit 1; }
[[ "$(shasum -a 256 "$S" | cut -d' ' -f1)" == "$S_SHA" ]] || { print "STOP: the script is not 1afd66ac..."; exit 1; }

# ---- the ten CR_CFG files: commit, path, sha256 (PRESTATED's input table)
CFGS=(
  1a842a856 experiments/2026-10-15_fpr_sweep_507_prep/config/INPUTS_FULL_2026-10-17_233212.json 1e1868586fe8a786f6e1094b9d5b62e362ec0e2ce241e9b0103d6584af351126
  a6bde80a2 experiments/2026-10-15_fpr_sweep_507_prep/config/INPUTS_FINAL_V2_FROZEN_2026-10-16_044120.json 0e065958b206dbceba8afeb3332ec68b898481da520fe3a3654ab2b6913eaadd
  22d7089f5 experiments/2026-10-15_fpr_sweep_507_prep/config/E12_BLOCK_2026-10-16_092322.json 683d4de1040f489a768a44fe5ad53f787d35d8ca22117b897a42a75880e1f4dc
  6414c492b experiments/2026-10-15_fpr_sweep_507_prep/results/exclusions_2026-10-16_102159.json 67a608c25a84e77cb6beb2c70d8c91177e2274d2c96e166469060462d4c9223d
  38b42bf2d experiments/2026-10-15_fpr_sweep_507_prep/config/cutfull/override_S_new_C_2026-10-16_104513_gated.json 908a27ad834a280fb56e9cbe7747bb90a3a3aa726098afe345a9fdc4c91bbadb
  5df9a43ed experiments/2026-10-15_fpr_sweep_507_prep/config/cutfull/override_S_old_C_2026-10-17_151056_gated.json f8ee4ddc603dd9e38f0fd650055bcc9d324df903bbfd5a51809ca6b59ffceafd
  64e9ab71a experiments/2026-10-15_fpr_sweep_507_prep/config/cutfull/override_S_newmains_C_2026-10-16_053303_gated.json 09b766a09208cce60b0882bce7fd538582b9d06d27b30f8ae5a9dda6759c3c3f
  4d2f45e73 experiments/2026-10-15_fpr_sweep_507_prep/config/cutfull/override_S_paper493_C_2026-10-16_101944_gated.json 1d3208ddc36ce08baf0349bef3d16a3fdf79cd5f8756df80bbc70ca1191b02d0
  93ee9afe6 experiments/2026-10-15_fpr_sweep_507_prep/config/cutfull/override_pools_ep2_C_2026-10-16_053038_gated.json 734d2ccb0b4207a7a897b2b6b91776aa35cb8e29e029f2976b4c1f7147968e83
  f5a2504dc experiments/2026-10-15_fpr_sweep_507_prep/config/cutfull/override_S_new25f_C_2026-10-16_045304_gated.json 6ebc9d007c99ad5f48bbef7ef60ab31002d5bc30d461f4d70ac0386986b43326
)
(( ${#CFGS} == 30 )) || { print "STOP: the config table is not 10 rows"; exit 1; }
PTXT=$(git show "HEAD:$P") || { print "STOP: cannot read PRESTATED at HEAD"; exit 1; }
CFG=$(mktemp -d /Users/aswinsurya/probe-shift-rcfe-cfg-2026-10-20-XXXXXX) || { print "STOP: mktemp failed"; exit 1; }
for c p s in $CFGS; do
  [[ "$PTXT" == *"| $c | $s |"* ]] || { print "STOP: PRESTATED's table has no row with $c and $s"; exit 1; }
  f="$CFG/${p:t}"
  [[ -e "$f" ]] && { print "STOP: $f exists"; exit 1; }
  git show "$c:$p" > "$f" || { print "STOP: git show $c:$p failed"; exit 1; }
  [[ "$(shasum -a 256 "$f" | cut -d' ' -f1)" == "$s" ]] || { print "STOP: sha mismatch for $p @ $c"; exit 1; }
done
[[ "$(ls -A "$CFG" | wc -l | tr -d ' ')" == 10 ]] || { print "STOP: $CFG does not hold exactly 10 files"; exit 1; }
chmod a-w "$CFG"/* "$CFG" || { print "STOP: chmod failed"; exit 1; }

[[ "$("${UVM[@]}" profile current < /dev/null 2>/dev/null | tail -1)" == "redwood-default" ]] \
  || { print "STOP: modal profile is not redwood-default"; exit 1; }
mkdir -p "$L" || { print "STOP: no log dir"; exit 1; }
print "RCFE_GATE_OK $H cfg=$CFG"
[[ "$MODE" == gate-only ]] && exit 0

# ---- the run (one Modal CPU container; the record is written to $E/results/ by the script's local entrypoint)
ST=$(TZ=America/Los_Angeles date +%Y-%m-%d_%H%M%S)
LOG="$L/run_$ST.log"
cd "$W/$E" || { print "STOP: no experiment folder"; exit 1; }
CR_CFG="$CFG" uv run --no-project --python 3.12 --with modal==1.6.1 modal run run_routec_fileedit_share.py < /dev/null > "$LOG" 2>&1
RC=$?
print "rc=$RC log=$LOG"
grep -E '^wrote ' "$LOG"
grep -o 'ap-[A-Za-z0-9]*' "$LOG" | head -1
exit $RC
