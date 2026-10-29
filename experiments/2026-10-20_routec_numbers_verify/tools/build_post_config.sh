#!/usr/bin/env bash
# The "post" artifact entry of config/artifacts_routec_2026-10-20.json (the v84 pass), built from paths, commits and shas only: every
# input pinned {commit, path, sha256} (commit = the one that last touched the file on this branch; 09's and 0d's files at their own
# commits), and the records the post's provenance and deferred rows cite, read from writing's files by jq (record / record_sha256 /
# {commit, path, sha256} fields only; no value is read or printed). Prints counts only.
#   bash tools/build_post_config.sh <out.json>    (from the verifier folder; writes a new file, never in place)
set -euo pipefail
OUT=$1
[[ -e $OUT ]] && { echo "STOP: $OUT exists"; exit 1; }
CFG=config/artifacts_routec_2026-10-20.json
REPO=$(git rev-parse --show-toplevel)
HERE=experiments/2026-10-20_routec_numbers_verify
pin() {  # pin <path relative to the verifier folder>
  local c s
  c=$(git log -1 --format=%H -- "$1")
  s=$(git show "$c:$HERE/$1" | shasum -a 256 | cut -d' ' -f1)
  jq -n --arg c "$c" --arg p "$HERE/$1" --arg s "$s" '{commit: $c, path: $p, sha256: $s}'
}
ext() {  # ext <commit> <repo path>
  local s
  s=$(git -C "$REPO" show "$1:$2" | shasum -a 256 | cut -d' ' -f1)
  jq -n --arg c "$1" --arg p "$2" --arg s "$s" '{commit: $c, path: $p, sha256: $s}'
}
PROV=post/provenance_v84_final17_0141.jsonl
DEF=post/deferred_v84_final8_0141.jsonl
LB29="not route C; checked against cited record by key (record unsigned); U restricted to the headline's common sets"
UNOTE="U restricted to the headline's common sets (09, 10-20): 09 per-launch tally by key, S_new 5 + S_newmains 42 = 47 common of 50 total; S_old honest_a 3 outside (READ_LOG 10-20 ~05:2x)"
ART_SHAS=$(jq -c '[.artifacts[] | .artifact.sha256? // empty]' $CFG)
# provenance records that are not artifacts of this run: "path@commit[ (note)]" + record_sha256
PREC=$(jq -s -c --argjson art "$ART_SHAS" '[.[] | select((.record_sha256 as $s | $art | index($s)) | not)
  | {commit: (.record | split("@")[1] | split(" ")[0]), path: (.record | split("@")[0]), sha256: .record_sha256}] | unique' $PROV)
DREC=$(jq -s -c '[.[] | .record // [] | .[] | {commit, path, sha256}] | unique' $DEF)
RECS=$(jq -n -c --argjson a "$PREC" --argjson b "$DREC" '($a + $b) | unique_by(.sha256)')
# every record must resolve at its commit to its sha (git bytes; gz decompressed as the container does)
n_bad=0
while IFS=$'\t' read -r c p s; do
  if [[ $p == *.gz ]]; then h=$(git -C "$REPO" show "$c:$p" | gunzip | shasum -a 256 | cut -d' ' -f1); else h=$(git -C "$REPO" show "$c:$p" | shasum -a 256 | cut -d' ' -f1); fi
  [[ $h == "$s" ]] || n_bad=$((n_bad + 1))
done < <(jq -r '.[] | [.commit, .path, .sha256] | @tsv' <<<"$RECS")
[[ $n_bad -eq 0 ]] || { echo "STOP: $n_bad cited records do not hash to their sha at their commit"; exit 1; }
# not route C (record unsigned; 09, 10-20): the fit records, the App I counts, the attack-side Sol share
ARATE=$(git -C "$REPO" show 73cd0f2be:experiments/2026-10-15_fpr_sweep_507_prep/results/appendix_a_rates_2026-10-16_121443.json | shasum -a 256 | cut -d' ' -f1)
[[ $(jq -r --arg s "$ARATE" '[.[] | select(.sha256 == $s)] | length' <<<"$RECS") == 1 ]] || { echo "STOP: the bounds-check record is not among the shipped records"; exit 1; }
PRE_H=$(jq -s -c '[.[] | select(.record | test("/honest_all_routec_[^/]*@596bc767e")) | .record_sha256] | unique' $PROV)  # 09: honest_all_routec @ 596bc767e, PRELIMINARY (signed: label override)
PRE_R=$(jq -s -c '[.[] | select(.record | test("/cunits_by_tool_[^/]*@a6d37c024")) | .record_sha256] | unique' $PROV)  # 09 (10-20): the App A recount @ a6d37c024, PRELIMINARY, unsigned: NO label override (R7-2)
[[ $(jq length <<<"$PRE_H") == 1 && $(jq length <<<"$PRE_R") == 1 ]] || { echo "STOP: not exactly one honest and one recount PRELIMINARY record"; exit 1; }
PRE=$(jq -n -c --argjson a "$PRE_H" --argjson b "$PRE_R" '($a + $b) | unique')
NRC=$(jq -s -c '[.[] | select(.record | test("/(read_fit_|sola_uncut_reselect_fit_|appI_real_counts_|attack_sol_share_)[^/]*@")) | .record_sha256] | unique' $PROV)
POST=$(jq -n \
  --argjson art "$(pin post/draft_v84_frozen_0141.md)" \
  --argjson prov "$(pin $PROV)" --argjson def "$(pin $DEF)" \
  --argjson cons "$(pin post/constants_v84_final9_0141.jsonl)" --argjson csrc "$(pin post/constants_sources_final5_0141_030913.jsonl)" \
  --argjson hyp "$(pin config/v84_hypothetical_expressions_c5_pinned.json)" --argjson tol "$(pin config/v84_predicate_tolerances_final17.json)" \
  --argjson din "$(pin post/draft_v84.md.in)" --argjson merged "$(pin post/slots_routeC_rc1a_rc2_merged.json)" \
  --argjson swap "$(pin post/draft_v84.swaponly_drop1a.md)" --argjson diff "$(pin post/hand_edits_swaponly_to_before_drop1b.diff)" \
  --argjson lit "$(pin config/v84_nonresult_literals.json)" --argjson cadd "$(pin post/constants_sources_v9_additions_0141.jsonl)" \
  --argjson pat "$(pin config/v84_nonresult_patterns.json)" --argjson hl "$(pin config/v84_header_labels_0141.json)" \
  --argjson lab "$(pin post/draft_v84_frozen_0240.md)" --argjson lab2 "$(pin post/draft_v84_frozen_0256.md)" --argjson d1b "$(pin post/slots_routeC_drop1b_only.json)" --argjson apply "$(pin post/apply_v84.py)" --argjson rules "$(pin post/appendix_rules.py)" \
  --argjson acc "$(ext ed8c941b3 experiments/2026-10-14_priority1_spec_09/v84_accepted_deferred_reasons_2026-10-20.json)" \
  --argjson cm "$(ext 9c61ba0a4 experiments/2026-10-20_routec_simpool_numbers/code_map_routec_unchanged_regions_2026-10-20.json)" \
  --argjson nv5 "$(ext 4fea20156aaf90086eb54cb8959642c62de43225 $HERE/results/numbers_verify_2026-10-20_041311.json)" \
  --argjson recs "$RECS" --argjson nrc "$NRC" --argjson pre "$PRE" --argjson pre_h "$PRE_H" --argjson pre_r "$PRE_R" --arg arate "$ARATE" --arg lb29 "$LB29" --arg unote "$UNOTE" \
  '{kind: "post", requires: ["tables34", "rc1a", "rc2", "rc1b"], artifact: $art,
    inputs: {provenance: $prov, deferred: $def, constants: $cons, constants_sources: $csrc, hyp_exprs: $hyp, tolerances: $tol,
             draft_in: $din, merged_slots: $merged, swaponly: $swap, hand_diff: $diff, drop1b_slots: $d1b, apply: $apply, rules: $rules,
             accepted_reasons: $acc, code_map: $cm, label_update: $lab, label_update_0256: $lab2, patterns: $pat, nonresult_literals: $lit, constants_sources_additions: $cadd, header_labels: $hl,
             numv5_result: $nv5},
    records: $recs, not_route_C: $nrc,
    preliminary: $pre,
    preliminary_labels: ($pre_h | map({(.): "route C, out of v1 scope (PRELIMINARY); checked against a signed record by key (09 S2 marker S2_2026-10-20_015417 @ 114ca1583)"}) | add),
    code: {"run_examples_v81.py": "f191a2477226f868633ec2c5f6092ebb85d48fea", "run_tables_v81.py": "88589b507a8439d4abb85be40a1d60f4590fd34d"},
    nonresult_approved: {sha256: "0783e56590743b34adade726b879edb49be0be4d0c59339a188c187a3d50bfbd", count: 313,
                         predecessor: "6e92bca28c087452076a85dec14c56e8e118c00109ce7bfc5a78220719c01fb4",
                         approved: "09, relayed by 2e (10-20 05:0x): APPROVED by the diff against 6e92bca2 (+96 / -3) @ cdacb143a; numv7 reports its list and a diff"},
    nonresult_previous: {commit: "cdacb143a", path: "experiments/2026-10-20_routec_numbers_verify/results/nonresult_list_2026-10-20_045432.txt",
                         sha256: "0783e56590743b34adade726b879edb49be0be4d0c59339a188c187a3d50bfbd"},
    not_printed_v84_approved: {sha256: "cfa546c030619c95fa63d06bfb80e6726d86c3e67a56d9ae7b0e73afd569b573", count: 101,
                               scope: "v84 pass (09: the adapter inputs approved, logged @ d852b6be6 on aswin/convergence-audit-09)"},
    orphan_notes: [],
    deferred_record_checks: [{deferred_row: 29, form: "bounds_pct_1dp", record_sha256: $arate,
                              record: "experiments/2026-10-15_fpr_sweep_507_prep/results/appendix_a_rates_2026-10-16_121443.json@73cd0f2be",
                              keys: {failed: ["headline_common_sets", "X_real", "failed"], launched: ["headline_common_sets", "X_real", "launched"],
                                     unseen: ["real_lower_bound", "unseen_attempts_at_least"]},
                              u_common: 47, u_total: 50, u_note: $unote, label: $lb29,
                              approved: "09, relayed by 2e (10-20 05:2x): U_common = 47 a config design constant; the record U must equal the total 50; on PASS the label above"}],
    row_scope_notes: {"81": "verifies the printed claim for the −8, C, 0.01 cell only (RULE item 47)"},
    residual_notes: {"30154": "App A: route C record exists (a6d37c024), not yet mapped", "30226": "App A: route C record exists (a6d37c024), not yet mapped",
                     "8541": "orphan", "11136": "orphan"},
    residual_notes_by: "09, relayed by 2e (10-20 05:3x): residual orphans stay tonight, named by position only; other items take their kind note",
    approvals: {patterns: {sha256: $pat.sha256, count: 6, approved: "09, relayed by 2e (10-20 03:2x): {b2ed5c568692fc6a..., 6} @ e825bc9e6: APPROVED"},
               provenance: {sha256: $prov.sha256, count: 251, approved: "09, relayed by 2e (10-20 05:4x): provenance v17 {220d25d2..., 251} APPROVED by diff (v16 rows 110/125/232 re-mapped; rows 250-251 App A on the recount @ a6d37c024, PRELIMINARY); v15: op 25 and component_string APPROVED fail-closed"},
               constants: {sha256: $cons.sha256, count: 112, approved: "09, relayed by 2e (10-20 05:4x): constants v9 {02751eb2..., 112} APPROVED by diff (rows 20 and 27 anchors widened), with the sources additions v9 {eb03888f..., 5} (rows 108-112) and final5 (rows 1-107)"},
               constants_sources_additions: {sha256: $cadd.sha256, count: 5, approved: "09, relayed by 2e (10-20 04:1x): APPROVED; the v8 additions file 0dc7753f is VOID"},
               deferred: {sha256: $def.sha256, count: 29, approved: "09, relayed by 2e (10-20 05:2x): deferred v8 {9f3260a3..., 29} APPROVED by diff (row 29 note only)"},
               preliminary: {records: $pre, approved: "09, relayed by 2e (10-20 04:0x): honest_all_routec at 596bc767e APPROVED as route C, out of v1 scope (PRELIMINARY), checked against the record by key; its 4 tokens stay PRELIMINARY. 09 (10-20 05:4x): the App A recount cunits_by_tool @ a6d37c024 (072efe87...) PRELIMINARY, unsigned, NO label override (R7-2); its label (route C producer-script recomputation; not independent; PRELIMINARY) is in provenance rows 250-251 and the numv9 NOTE"},
               nonresult_literals: {sha256: $lit.sha256, count: 5, approved: "09, relayed by 2e (10-20 04:4x): {d7c388e1..., 5} APPROVED at ae78b0afd"},
               numv5_result: {sha256: $nv5.sha256, approved: "09, relayed by 2e (10-20 04:5x): pinned by path@sha (6e6c7fa0... @ 4fea20156); positions only, against the Tables 3/4 blocks of this post"},
               constants_sources: {sha256: $csrc.sha256, count: 107, approved: "09 (10-20): sources final5 stays pinned on constants v6"},
               tolerances: {sha256: $tol.sha256, count: 251, approved: "table v17 built mechanically by tools/build_tolerance_table.sh from provenance v17 (same mask as v15, which 09 APPROVED 05:1x with the mask diff); v17 approval by {sha, rows} PENDING 09 at commit time"},
               apply: {sha256: $apply.sha256, approved: "09 G1, relayed by 2e (10-20): APPROVED as copied at 215fc56e8; used ONLY to locate anchors and to classify swapped vs hand-edited spans, never as a source of values"},
               rules: {sha256: $rules.sha256, approved: "09 G1, relayed by 2e (10-20): APPROVED as copied at 215fc56e8; same use as apply_v84.py"},
               code_map: {sha256: $cm.sha256, approved: "09 G1, relayed by 2e (10-20): APPROVED @ 9c61ba0a4, keys, shas and line refs only, 7 regions; the evidence for the 26 not-route-C unchanged deferred rows (the 10-18 code map reason accepted as this 10-20 keys version)"},
               label_update: {sha256: $lab.sha256, approved: "09 G1, relayed by 2e (10-20): APPROVED because it is checked equal to 0141 with hidden comments stripped, with a STOP otherwise"}},
    note: "The v84 pass: every input approved by 09 by sha (log @ d852b6be6); nonresult_approved awaits the list this pass generates."}')
jq --indent 2 --argjson post "$POST" '.artifacts.post = $post' $CFG > "$OUT"
jq -r '.artifacts.post | "inputs \(.inputs|length)  records \(.records|length)  not_route_C \(.not_route_C|length)"' "$OUT"
