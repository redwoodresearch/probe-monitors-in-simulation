#!/usr/bin/env bash
# 09's 27(b) identity check (relayed by 0d, 10-16): is 4f's all-sets detector record (attack_all_2026-10-16_042630.json @ 59acf625e,
# computed 04:26 from 8ea4a6a5d, BEFORE the 11:02 freeze) built on inputs identical to the FROZEN INPUTS_FULL (d6923a75...)? Keys and
# sha256s only; git + jq (no Python on the laptop). Prints one line per check and a final IDENTICAL / DIFFERENT. Run from anywhere:
#   bash check_frozen_identity.sh
set -euo pipefail
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
T=$(mktemp -d)
trap 'rm -rf "$T"' EXIT
F4=experiments/2026-10-16_cmd_sim_detector_uncut_all
P=experiments/2026-10-15_fpr_sweep_507_prep
CS=experiments/2026-10-15_cut_sensitivity/results
show() { git show "$1:$2" > "$3"; echo "$(shasum -a 256 "$3" | cut -c1-64)"; }
fail=0
chk() { if [ "$2" = "$3" ]; then echo "SAME  $1"; else echo "DIFF  $1 ($2 vs $3)"; fail=1; fi; }

echo "record: $(show 59acf625e $F4/results/attack_all_2026-10-16_042630.json $T/rec.json)  $F4/results/attack_all_2026-10-16_042630.json @ 59acf625e"
echo "frozen: $(show 927fabb0d $P/config/INPUTS_FULL_2026-10-16_110447.json $T/full.json)  $P/config/INPUTS_FULL_2026-10-16_110447.json @ 927fabb0d"
chk "INPUTS_FULL sha256 is the frozen d6923a75" "$(shasum -a 256 $T/full.json | cut -c1-8)" "d6923a75"
spec() { jq -r ".$1 | \"\(.commit) \(.path) \(.sha256)\"" $T/full.json; }
read -r c p s < <(spec exclusions); show "$c" "$p" $T/coh_frozen.json > /dev/null
chk "frozen cohort file sha256 = its INPUTS_FULL pin" "$(shasum -a 256 $T/coh_frozen.json | cut -c1-64)" "$s"
show adbc791b6 $P/results/exclusions_2026-10-16_033635.json $T/coh_rec.json > /dev/null

echo "== 1. run keys per cell x arm x fold"
for set in S_old S_new S_newmains S_paper493; do
  for w in run1 run2 sim; do
    a=$(jq -cj --arg s $set --arg w $w '[.run_set | to_entries[] | select(.value == $s and (.key | startswith($w + "|"))) | .key | sub("^[^|]*\\|"; "")] | sort' $T/rec.json | shasum -a 256 | cut -c1-16)
    b=$(jq -cj --arg s $set '.final_cohort_keys[$s].cells | sort' $T/coh_frozen.json | shasum -a 256 | cut -c1-16)
    chk "cells $set|$w: record vs frozen cohort" "$a" "$b"
  done
done
for k in S_old S_newmains S_paper493 S_paper493_w23 S_new; do
  read -r c p s < <(jq -r --arg k $k '.inventories[$k] | "\(.commit) \(.path) \(.sha256)"' $T/full.json)
  show "$c" "$p" "$T/inv_$k.json" > /dev/null
  chk "frozen inventory $k sha256 = its INPUTS_FULL pin" "$(shasum -a 256 $T/inv_$k.json | cut -c1-64)" "$s"
done
for cfg in S_old S_newmains S_paper493_w1 S_paper493_w23; do
  case $cfg in S_old | S_newmains) k=$cfg ;; S_paper493_w1) k=S_paper493 ;; S_paper493_w23) k=S_paper493_w23 ;; esac
  chk "record config $cfg read the frozen inventory (sha256)" "$(git show 59acf625e:$F4/config/inputs_all_${cfg}_v1.json | jq -r '.inventory.sha256')" "$(jq -r --arg k $k '.inventories[$k].sha256' $T/full.json)"
done
git show b8a39f329:$CS/inventory_v3_2026-10-15.json > $T/v3_rec.json
chk "S_new: inventory_v3 the record derived from (b8a39f329) = the frozen copy (sha256)" "$(shasum -a 256 $T/v3_rec.json | cut -c1-64)" "$(shasum -a 256 $T/inv_S_new.json | cut -c1-64)"
jq -r '.runs[] | "\(.key)\t\(.world)"' $T/inv_S_old.json $T/inv_S_paper493.json $T/inv_S_paper493_w23.json $T/inv_S_new.json | sort -u > $T/frozen_keys.tsv
jq -r '.run_keys | to_entries[] | "\(.value)\t\(.key | split("|")[0])"' $T/rec.json | sort > $T/rec_keys.tsv
chk "every record run key (3,939) in a frozen-pinned inventory with the same world" "$(comm -12 $T/rec_keys.tsv $T/frozen_keys.tsv | wc -l | tr -d ' ')" "$(wc -l < $T/rec_keys.tsv | tr -d ' ')"
chk "record cohort file (033635) = frozen cohort (102159), all 4 sets by key" "$(jq -cj '[.final_cohort_keys[] | .cells | sort]' $T/coh_rec.json | shasum -a 256 | cut -c1-16)" "$(jq -cj '[.final_cohort_keys[] | .cells | sort]' $T/coh_frozen.json | shasum -a 256 | cut -c1-16)"

echo "== 2. source file sha256s"
jq -r '.runs[] | select(.uncut_planned.staged_sha256 != null) | "\(.uncut_planned.staged_set)/inputs/\(.arm).jsonl\t\(.uncut_planned.staged_sha256)"' $T/inv_S_old.json $T/inv_S_paper493.json $T/inv_S_paper493_w23.json | sort -u > $T/frozen_staged.tsv
jq -r '.skip.staged_sha256 | to_entries[] | "\(.key)\t\(.value)"' $T/rec.json | sort > $T/rec_staged.tsv
n_b=$(cut -f1 $T/frozen_staged.tsv | sort -u | comm -12 - <(cut -f1 $T/rec_staged.tsv) | wc -l | tr -d ' ')
chk "staged-input sha256, the $n_b record files with a frozen (b-contract) row" "$(comm -12 $T/rec_staged.tsv $T/frozen_staged.tsv | wc -l | tr -d ' ')" "$n_b"
read -r c p s < <(jq -r '.overrides["S_new|R"] | "\(.commit) \(.path) \(.sha256)"' $T/full.json); show "$c" "$p" $T/ov_snew.json > /dev/null
jq -r '[.arms[] | .[]? | objects | .staged_shas[]?] | unique | .[]' $T/ov_snew.json | sort -u > $T/ov_snew_shas.txt
grep '^su311' $T/rec_staged.tsv | cut -f2 | sort -u > $T/rec_snew_shas.txt
chk "staged-input sha256, the $(wc -l < $T/rec_snew_shas.txt | tr -d ' ') S_new record files, in the frozen S_new|R override's staged_shas" "$(comm -12 $T/rec_snew_shas.txt $T/ov_snew_shas.txt | wc -l | tr -d ' ')" "$(wc -l < $T/rec_snew_shas.txt | tr -d ' ')"
chk "staged files: b-contract + S_new = all record files" "$(( n_b + $(wc -l < $T/rec_snew_shas.txt) ))" "$(wc -l < $T/rec_staged.tsv | tr -d ' ')"
jq -r '.runs[] | select(.uncut_planned.extraction_root != null) | "\(.uncut_planned.staged_set)\t\(.uncut_planned.extraction_root)"' $T/inv_S_old.json $T/inv_S_paper493.json $T/inv_S_paper493_w23.json | sort -u > $T/frozen_roots.tsv
jq -r '.inputs.configs[] | .provenance.roots | to_entries[] | "\(.value.staged_set)\t\(.key)"' $T/rec.json | grep -v '^su311' | sort -u > $T/rec_roots.tsv
chk "extraction root per staged set (non-S_new) equals the frozen inventory's" "$(comm -12 $T/rec_roots.tsv $T/frozen_roots.tsv | wc -l | tr -d ' ')" "$(wc -l < $T/rec_roots.tsv | tr -d ' ')"

echo "== 3. the fold map"
spec_f=$(jq -r '.inputs.folds | "\(.commit) \(.path) \(.sha256)"' $T/rec.json); read -r c p s <<< "$spec_f"; show "$c" "$p" $T/folds.json > /dev/null
chk "fold map file sha256 = the record's pin (own_all_rs @ ec56720ad; INPUTS_FULL pins no detector fold map)" "$(shasum -a 256 $T/folds.json | cut -c1-64)" "$s"
chk "runs whose record fold differs from the fold map by main" "$(jq -n --slurpfile r $T/rec.json --slurpfile f $T/folds.json '[$r[0].run_folds | to_entries[] | select($f[0].by_main[$r[0].run_main[.key]][0] != .value)] | length')" "0"

echo "== 4. null seeds and permutation draws"
chk "null seed (base.SEED) and draws" "$(jq -c '[.settings.seed, .settings.n_null]' $T/rec.json)" "[20261014,100]"
chk "perm sha256 identical in all 4 cells (c)" "$(jq -r '[.cells[].problems.c_simatk_vs_run1.perm_sha256] | unique | length' $T/rec.json)" "1"
chk "perm sha256 identical in all 4 cells (d)" "$(jq -r '[.cells[].problems.d_run2_vs_run1.perm_sha256] | unique | length' $T/rec.json)" "1"
echo "      (draws are base.within_task_perms(main of each run, labels, 100, seed 20261014) over the run list; the run list, mains and labels are identical by checks 1 and 3)"

echo "== 5. recipe"
chk "C, layers, readouts, primary" "$(jq -c '[.settings.C, .settings.layers, ([.cells[].readout] | unique), .settings.primary]' $T/rec.json)" '[0.01,[48,63],["command","reply"],{"command":63,"reply":48}]'
chk "iteration cap: tp.fit called without max_iter (default 5000) at the record's code" "$(git show 8ea4a6a5d:$F4/run_attack_all.py | grep -c 'tp.fit(Xtr, Ytr, C0, allow_uncertified=True, return_certificate=True)$')" "1"
chk "torch_probe.fit default max_iter" "$(git show 8ea4a6a5d:src/probe_shift/torch_probe.py | grep -c '    max_iter: int = 5000,')" "1"

echo "== 6. detector code"
chk "record's code commit" "$(jq -r '.git' $T/rec.json)" "8ea4a6a5d88fae66a6498e0f81250c433bdd8aaf"
chk "code diff 8ea4a6a5d..59acf625e (detector files + src/probe_shift), changed lines" "$(git diff 8ea4a6a5d 59acf625e -- $F4/run_attack_all.py $F4/run_simh_control.py $F4/run_controls_a1.py $F4/uncut_inputs.py $F4/uncut_inputs_all.py src/probe_shift/ | grep -c '^[+-][^+-]' || true)" "0"
chk "refit3's copies = 59acf625e (run_attack_all, run_controls_a1, uncut_inputs, uncut_inputs_all)" "$(for f in run_attack_all.py run_controls_a1.py uncut_inputs.py uncut_inputs_all.py; do git show 59acf625e:$F4/$f | shasum -a 256 | cut -c1-16; done | tr '\n' ' ')" "$(for f in run_attack_all.py run_controls_a1.py uncut_inputs.py uncut_inputs_all.py; do shasum -a 256 < experiments/2026-10-16_attack_refit3/$f | cut -c1-16; done | tr '\n' ' ')"
chk "src/probe_shift: 8ea4a6a5d vs this checkout's HEAD, changed lines" "$(git diff 8ea4a6a5d HEAD -- src/probe_shift/ | grep -c '^[+-][^+-]' || true)" "0"

[ $fail = 0 ] && echo "RESULT: IDENTICAL on every check" || echo "RESULT: DIFFERENT (see DIFF lines)"
