#!/bin/zsh -f
# a1's route C clean-room fill sheet (2e, 10-18): four lines generated from the chain's markers and pinned records, never retyped.
# Keys only: paths, sha256s, commits, world tokens; no scored record, catalogue or manifest is opened (the committed A record is only
# hashed). Writers (each marker is written last and exclusively by its block): CPL's CAT ok.txt and reply.json (NOTE_routec_cat_auth_ps_2026-10-17.md,
# CPL; the reply's keys only), AUTHC's committed.txt (same NOTE), S7L's launched.txt, S7R's ok.txt, S7SC's sidecar.txt and S7RC's
# record_commit.txt (NOTE_routec_s7_2026-10-17.md). Three records are jq-parsed for keys (a8953dad's R-2, 10-18), each sha-checked first:
# the FROZEN INPUTS_FULL (writer inputs_full.py @ 093bc2bb; .s_old_simh_extra_roots and .inventories.S_old), the counted-keys file
# (writer make_counted_keys.py, blob 3a9296f4, @ 344829af6; .keys[].world) and the S_old inventory (writer make_inventory_b_fix_v2.py
# @ 080ac97b; .runs[] .original_folder / .world); plus full_override.py's S_OLD_EXTRA_WORLD (L114 at 71633875's era; matched by its
# exact assignment line).
#
#   zsh -f tools_routec/a1_fill_sheet.zsh <log dir L> <STAMP_L> <STAMP_A> <STAMP7> <FRZP> <FRZS>     (from the repo, any directory)
set -e -u -o pipefail
L=$1 SL=$2 SA=$3 S7=$4 FRZP=$5 FRZS=$6
E=experiments/2026-10-15_fpr_sweep_507_prep
cd "$(git rev-parse --show-toplevel)"
die() { print -u2 "STOP: $*"; exit 1; }
hx64='^[0-9a-f]{64}$' hx40='^[0-9a-f]{40}$'
sha() { shasum -a 256 | cut -c1-64; }

# 1. rcCAT's catalogue and manifest: CPL's ok.txt, equal to its reply.json's keys
read -r CP CS MP MS CA < "$L/CAT_$SL/ok.txt"
[[ $CS =~ $hx64 && $MS =~ $hx64 ]] || die "CAT ok.txt sha256s"
R=$(jq -er '.pooled_world_catalogue | "\(.catalogue.path) \(.catalogue.sha256) \(.manifest.path) \(.manifest.sha256)"' "$L/CAT_$SL/reply.json")
[[ $R == "$CP $CS $MP $MS" ]] || die "CAT ok.txt differs from its reply.json"
print -r -- "catalogue $CP@$CS manifest $MP@$MS (rcCAT app $CA; $L/CAT_$SL/ok.txt)"

# 2. the authorized-overlaps file: AUTHC's committed.txt, its blob re-hashed at that commit
read -r AP AS AC < "$L/AUTH_$SA/committed.txt"
[[ $AS =~ $hx64 && $AC =~ $hx40 ]] || die "AUTH committed.txt"
[[ $(git show "$AC:$E/$AP" | sha) == $AS ]] || die "AUTH file's sha256 at its commit"
print -r -- "auth ${AP:t} commit $AC sha256 $AS ($L/AUTH_$SA/committed.txt)"

# 3. the pooled A record: S7R's ok.txt (RS = the reply's sha256 of the record), S7L's launched.txt (the head A ran at), S7SC's sidecar
read -r RP RS NP NS AA < "$L/S7_A_$S7/ok.txt"
read -r LA LF LPK LPS LH < "$L/S7_A_$S7/launched.txt"
[[ $RS =~ $hx64 && $LH =~ $hx40 && $LA == $AA ]] || die "A's S7R ok.txt / S7L launched.txt"
SC=none
if [[ -e $L/S7_A_$S7/sidecar.txt ]]; then read -r SCP SCS SCC SCH < "$L/S7_A_$S7/sidecar.txt"; [[ $SCC =~ $hx40 ]] && git merge-base --is-ancestor "$LH" "$SCH" || die "A's sidecar.txt"; SC=$SCC; fi
RC=none  # S7RC's record_commit.txt (NOTE_routec_s7, S7RC): <commit> <path.json.gz> <decompressed sha256>, re-checked here
if [[ -e $L/S7_A_$S7/record_commit.txt ]]; then
  read -r RCC RCF RCS < "$L/S7_A_$S7/record_commit.txt"; [[ $RCC =~ $hx40 && $RCS == $RS ]] || die "A's record_commit.txt"
  [[ $(git show "$RCC:$RCF" | gzip -dc | sha) == $RS ]] || die "A's committed record"; RC="$RCC:$RCF@$RCS"
fi
print -r -- "pooled_A ${RP:t} launch_commit $LH sidecar_commit $SC record $RC decompressed_sha256 $RS (app $AA; $L/S7_A_$S7/ok.txt)"

# 4. the pilot root's world (the run keys') against its files' world (the S_old inventory rows of its folder)
fb() { git show "$LH:$E/$FRZP"; }  # each read straight from git: hashes see the exact bytes
[[ $(fb | sha) == $FRZS ]] || die "FROZEN INPUTS_FULL sha256 at A's launch head"
ROOT=$(fb | jq -er '.s_old_simh_extra_roots | keys | if length == 1 then .[0] else error("not one root") end')
read -r KC KP KS <<<"$(fb | jq -er --arg r "$ROOT" '.s_old_simh_extra_roots[$r] | "\(.commit) \(.path) \(.sha256)"')"
[[ $(git show "$KC:$KP" | sha) == $KS ]] || die "counted-keys file sha256"
KW=$(git show "$KC:$KP" | jq -er '[.keys[].world] | unique | if length == 1 then .[0] else error("not one world") end')
RL=$(git show "$LH:$E/full_override.py" | grep -E -x 'S_OLD_EXTRA_WORLD, S_OLD_EXTRA_N = "[A-Za-z0-9_]+", [0-9]+')
RW=${${RL#*\"}%%\"*}
[[ $RW == $KW ]] || die "full_override's S_OLD_EXTRA_WORLD differs from the counted keys' world"
read -r IC IP IS <<<"$(fb | jq -er '.inventories.S_old | "\(.commit) \(.path) \(.sha256)"')"
[[ $(git show "$IC:$IP" | sha) == $IS ]] || die "S_old inventory sha256"
FW=$(git show "$IC:$IP" | jq -er --arg r "$ROOT" '[.runs[] | select(.original_folder == $r) | .world] | unique | if length == 1 then .[0] else error("not one world") end')
if [[ $RW == $FW ]]; then
  print -r -- "pilot_root $ROOT world $RW: SAME as its files' world (no fill)"
else
  print -r -- "pilot_root $ROOT world $RW: DIFFERS from its files' world $FW (fill $RW; run keys from $KP@${KS[1,12]}, files from the S_old inventory @ ${IS[1,12]})"
fi
