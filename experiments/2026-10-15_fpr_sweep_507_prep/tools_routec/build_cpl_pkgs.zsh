#!/bin/zsh -f
# Build the CPL (rcPS / rcCAT), PCD16, CPLR and PCJN (PS) run files once 09's S1a-launch marker exists. Blocks come from the NOTEs at the reviewed commit
# NC (the CPL-cut commit); the launch head H (HS1, WDX) is set in the GO lines. VARSCP is filled ONLY by tools_routec/fill_varscp.py
# (a8953dad's R-1: regex per value first, one substitution per placeholder, strict diff against the reviewed block, zsh -n, filled sha
# printed); no value is printed into a run file by this script. One block per run file.
# Usage: zsh -f build_cpl_pkgs.zsh <outdir> <NC> <H> <VARSCP reviewed sha256> <STAMP_L> <S1L_C> <S1L_P> <S1L_S>
set -e
W=/Users/aswinsurya/probe-shift-routec-launch-wt
cd $W
O=$1; NC=$2; H=$3; VS=$4; SL=$5; LC=$6; LP=$7; LS=$8
[ ! -e "$O" ] && mkdir -p "$O"
[[ $NC =~ '^[0-9a-f]{40}$' && $H =~ '^[0-9a-f]{40}$' && $VS =~ '^[0-9a-f]{64}$' ]]
git merge-base --is-ancestor "$H" "$NC"
E=experiments/2026-10-15_fpr_sweep_507_prep; R=experiments/2026-10-17_records_step_container
T=$(git rev-parse "$H^{tree}")
N2=$E/NOTE_routec_s2verify_2026-10-17.md; NP=$E/NOTE_routec_pscat_prep_2026-10-17.md; NCA=$E/NOTE_routec_cat_auth_ps_2026-10-17.md
git show "$NC:$N2" > $O/s2verify_at_NC.md; git show "$NC:$NP" > $O/pscat_at_NC.md; git show "$NC:$NCA" > $O/catauth_at_NC.md
blk(){ awk -v id=$2 '$0 == "    # END " id {f=0} f; $0 == "    # BEGIN " id {f=1}' $1; }
WB=$(git rev-parse "$H:$R/run_records_step_in_container_sfx.py")
GO=$O/go_assignments.sh
{
  printf "H='%s'; T='%s'; TG='%s'\n" "$H" "$T" "$T"
  printf 'WRAP=$WRAP_SFX; WRAP_BLOB=%s\n' "$WB"
} > $GO
V0=$O/vars_pc.sh
{ blk $O/s2verify_at_NC.md VARS2; cat $GO; blk $O/pscat_at_NC.md VARSPC; blk $O/pscat_at_NC.md VARSPC2; } > $V0
for t in PS CAT; do
  ( cd $E && uv run --offline --no-project --python 3.12 python -I tools_routec/fill_varscp.py --commit "$NC" --block-sha "$VS" \
      --out "$O/varscp_${t:l}.sh" STAMP_L="$SL" TAG="$t" S1L_C="$LC" S1L_P="$LP" S1L_S="$LS" )
  { cat $V0; cat $O/varscp_${t:l}.sh; blk $O/catauth_at_NC.md CPL; } > $O/cpl_${t:l}_run.sh
  { cat $V0; cat $O/varscp_${t:l}.sh; printf 'JOB=%s\n' ${t:l}; blk $O/pscat_at_NC.md PCD16; } > $O/pcd16_${t:l}_run.sh
  { cat $V0; cat $O/varscp_${t:l}.sh; blk $O/catauth_at_NC.md CPLR; } > $O/cplr_${t:l}_run.sh
  if [ $t = PS ]; then { cat $V0; cat $O/varscp_ps.sh; blk $O/pscat_at_NC.md PCJN; } > $O/pcjn_ps_run.sh; fi  # 09's R-5 (a), after PCD16 ps
done
for f in $O/*_run.sh; do zsh -f -n $f; done
for x in VARSCP CPL CPLR; do printf '%s %s\n' $x $(blk $O/catauth_at_NC.md $x | shasum -a 256 | cut -c1-16); done
for x in VARSPC VARSPC2 PCD16 PCJN; do printf '%s %s\n' $x $(blk $O/pscat_at_NC.md $x | shasum -a 256 | cut -c1-16); done
printf 'NC %s H %s T %s\n' $NC $H $T
shasum -a 256 $O/varscp_*.sh $O/*_run.sh
