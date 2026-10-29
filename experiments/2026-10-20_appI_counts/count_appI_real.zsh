#!/usr/bin/env zsh
# App I, real side: errored-run counts for honest pools and attack runs, from committed records.
#
# Keys and counts only. Each input is extracted with `git show <commit>:<path>` and pinned by sha256
# (STOP on mismatch). jq computes integer counts; nothing else is printed. The output JSON records
# the inputs, the jq programs, the counts, the consistency checks and git HEAD.
#
# Method reproduced: the 2026-10-16 four-set recheck (fourset_recheck.md claims 1, 2 and 4; fourset.md
# tables (i)-(ii), real rows only, and gap 1). Sources per count:
#   honest  (R): .real.failed_runs rows + .real.S_old_honest_a_apart named failures; environment from
#                the task folder (key path part 3, `^(t|nm)N_` and `_redrawN` stripped).
#   attack  (A): run1|/run2| rows of S_old, S_new, S_newmains; errored = .sample_error not null.
#   attack  S_paper493: X rows (set S_paper493, not the honest reference) whose .class names an Error,
#                plus D target rows with a sample error; denominators = B launched_universe run1/run2.
#   honest  epoch-2 attempts with no usable model call (P): .epoch2_candidates[].status_rows_removed[]
#                whose .class matches "usable 0" (the launcher's own "policy calls: usable 0"). These are
#                outside R by construction; their overlap with R's 75 failed runs is counted by (launch, task).
#
# Usage: zsh experiments/2026-10-20_appI_counts/count_appI_real.zsh   (from any directory in the repo)
set -euo pipefail

HERE=${0:A:h}
REPO=$(git -C "$HERE" rev-parse --show-toplevel)
SCRIPT_REL=${${0:A}#$REPO/}
STAMP=$(TZ=America/Los_Angeles date +%Y-%m-%d_%H%M%S)
OUT="$HERE/results/appI_real_counts_${STAMP}.json"
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

# name  commit  path  sha256
PINS=(
  R 73cd0f2bee480b06ef7a0c0cbbad7f6ad8b30ea8
    experiments/2026-10-15_fpr_sweep_507_prep/results/appendix_a_rates_2026-10-16_121443.json
    3c034ec9c74cc6989936a84e15ae07124cc93d94d0fe6fcb4bc6a4e958fd6e64
  A 48a90b1d285f590094f201bd13922331aae46f19
    experiments/2026-10-15_fpr_sweep_507_prep/results/attack_sample_errors_2026-10-16_011901.json
    758c2fa7405135aaa190b65aa51a3261279065c11ee7d486dc04b20dbb1f610d
  X 6414c492b663ee128b07d8ec045735ca41f0635f
    experiments/2026-10-15_fpr_sweep_507_prep/results/exclusions_2026-10-16_102159.json
    67a608c25a84e77cb6beb2c70d8c91177e2274d2c96e166469060462d4c9223d
  D 3c5b64b38c12d024d4bd60bdd51e2bb7ede9a9c8
    experiments/2026-10-15_q3_sp493/results/empty_epoch_summary_2026-10-16_100825.json
    e5612f3754aa5bc434cf951c30c1e7dc82c662e6ca4b0b989aab4a4b3e0ac853
  B a6bde80a2f648477c9df5ab5d8d693671db7ac37
    experiments/2026-10-15_fpr_sweep_507_prep/config/INPUTS_FINAL_V2_FROZEN_2026-10-16_044120.json
    0e065958b206dbceba8afeb3332ec68b898481da520fe3a3654ab2b6913eaadd
  P 5b19c08d8ccf461e7c5d5da4024cc7280e149084
    experiments/2026-10-15_epoch2_poolkeys/results/epoch2_poolkeys_2026-10-15_214512.json
    4c1ef89dd7fdf0a8b21153ede2901f595eb9aefc4879e810de1b1f522cc7eca1
)

INPUTS='[]'
for name commit relpath sha in $PINS; do
  git -C "$REPO" show "$commit:$relpath" > "$TMP/$name.json"
  got=$(shasum -a 256 "$TMP/$name.json" | cut -d' ' -f1)
  if [[ $got != $sha ]]; then
    print -u2 "STOP: sha256 mismatch for input $name ($commit:$relpath)"
    exit 1
  fi
  INPUTS=$(jq -c --arg n "$name" --arg c "$commit" --arg p "$relpath" --arg s "$sha" \
    '. + [{name: $n, commit: $c, path: $p, sha256: $s}]' <<< "$INPUTS")
done

# Honest pools, real (input R).
JQ_HONEST=$(cat <<'JQ'
def env: split("/")[2] | sub("^(t|nm)[0-9]+_"; "") | sub("_redraw[0-9]+$"; "");
def epce: "EmptyPolicyCompletionError";
def n(f): [.[] | select(f)] | length;
def envs: ("model_registry", "slack_clone", "tiktok");
def sets: ("S_old", "S_new", "S_newmains", "S_paper493", "ep2");
[ (.real.failed_runs | to_entries[] | .key as $sa | .value[]
    | {set: ($sa | split("|")[0]), arm: ($sa | split("|")[1]), class, category, key, env: (.key | env)}),
  (.real.S_old_honest_a_apart.named_failures_by_pool_rule[] | .keys[]
    | {set: "S_old", arm: "honest_a_apart", class: epce, category: null, key: ., env: (. | env)}) ] as $rows
| ($rows | map(select(.class == epce))) as $e
| ($e | map(select(.arm != "honest_a_apart"))) as $e_lead
| ([.real.by_set_arm[].launched] | add) as $launched_arms
| {
  failed_runs_rows: ([.real.failed_runs[][]] | length),
  failed_runs_rows_by_set_arm_sum: ([.real.by_set_arm[].failed] | add),
  S_old_honest_a_apart_rows: ([.real.S_old_honest_a_apart.named_failures_by_pool_rule[].keys[]] | length),
  failed_total: ($rows | length),
  failed_unique_run_keys: ($rows | map(.key) | unique | length),
  failed_env_agrees_with_basename: ($rows | n(. as $r | $r.key | split("/") | last
      | contains("la-" + ($r.env | gsub("_"; "-")) + "-real-honest"))),
  failed_set_outside_named_sets: ($rows | n(.set as $s | [sets] | index($s) | not)),
  launched_by_set_arm_sum: $launched_arms,
  launched_S_old_honest_a_apart: .real.S_old_honest_a_apart.launched,
  launched_total: ($launched_arms + .real.S_old_honest_a_apart.launched),
  epce_total: ($e | length),
  epce_by_env: ([envs as $v | {key: $v, value: ($e | n(.env == $v))}] | from_entries),
  epce_env_outside_named_envs: ($e | n(.env as $v | [envs] | index($v) | not)),
  epce_without_S_old_honest_a: ($e_lead | length),
  epce_without_S_old_honest_a_by_env: ([envs as $v | {key: $v, value: ($e_lead | n(.env == $v))}] | from_entries),
  epce_by_set: ([sets as $s | {key: $s, value: ($e | n(.set == $s))}] | from_entries),
  epce_S_old_honest_a_apart: ($e | n(.arm == "honest_a_apart")),
  epce_by_set_env: ([sets as $s | {key: $s, value: ([envs as $v | {key: $v, value: ($e | n(.set == $s and .env == $v))}] | from_entries)}] | from_entries),
  epce_by_category_sum_failed_runs_arms: ([.real.by_set_arm[].by_category["mid-run model error (EmptyPolicyCompletionError)"]] | add),
  non_epce_total: ($rows | n(.class != epce)),
  non_epce_setup_0_model_events: ($rows | n(.class != epce and .category == "setup (0 model events)")),
  non_epce_unclassified_model_events_gt0: ($rows | n(.class != epce and .category == "other error (unclassified, model events > 0)")),
  two_set_S_old_S_new_epce: ($e | n(.set == "S_old" or .set == "S_new")),
  two_set_S_old_S_new_epce_by_env: ([envs as $v | {key: $v, value: ($e | n((.set == "S_old" or .set == "S_new") and .env == $v))}] | from_entries),
  lower_bound_redraws_without_earlier_attempt: (.real_lower_bound.gap_keys | length),
  lower_bound_unseen_attempts_at_least: .real_lower_bound.unseen_attempts_at_least
}
JQ
)

# Attack runs, real, S_old / S_new / S_newmains (input A).
JQ_ATTACK_A=$(cat <<'JQ'
def epce: "EmptyPolicyCompletionError";
def n(f): [.[] | select(f)] | length;
def runs($o; $w): .rows | map(select(.origin == $o and (.key | startswith($w + "|"))));
def tally: {runs: length, errored: n(.sample_error != null), epce: n(.sample_error == epce),
            operation_error: n(.sample_error == "OperationError"), retry_error: n(.sample_error == "RetryError"),
            other_error: n(.sample_error as $s | $s != null and ([epce, "OperationError", "RetryError"] | index($s) | not))};
(.rows | map(select(.key | test("^(run1|run2)\\|")))) as $real
| {
  real_run_rows: ($real | length),
  has_error_disagrees_with_sample_error: (.rows | n((.has_error == true) != (.sample_error != null))),
  by_set: ([("S_old", "S_new", "S_newmains") as $o | {key: $o, value:
             ([("run1", "run2") as $w | {key: $w, value: (runs($o; $w) | tally)}] | from_entries)}] | from_entries),
  three_sets: ($real | map(select(.origin as $o | ["S_old", "S_new", "S_newmains"] | index($o))) | tally),
  two_set_S_old_S_new: ($real | map(select(.origin == "S_old" or .origin == "S_new")) | tally)
}
JQ
)

# Attack runs, real, S_paper493 (inputs X, D, B via --slurpfile).
JQ_ATTACK_P493=$(cat <<'JQ'
def epce: "EmptyPolicyCompletionError";
def n(f): [.[] | select(f)] | length;
def tid: capture("(?<![0-9A-Za-z])(?<t>t[0-9]+)_").t;
$B[0].extensions.S_paper493.launched_universe as $U
| [ $X[0].rows[] | select(.set == "S_paper493" and .stage != "honest reference") | . as $row
    | ("run1", "run2") as $w | select($row.failed_arm | index($w))
    | {arm: $w, tid: ($row.key | tid), errored: ($row.class | test("Error")), epce: ($row.class | test(epce))} ] as $xr
| [ $D[0].rows[] | select(.kind == "target" and .set == "S_paper493" and (.world == "run1" or .world == "run2"))
    | {arm: .world, tid: (.task | tid), errored: (.sample_error != null), epce: (.sample_error == epce)} ] as $dr
| ([$xr[], $dr[]] | map(select(.errored)) | unique_by([.arm, .tid])) as $err
| def in_u: . as $r | [$U[$r.arm][] | select(startswith($r.tid + "_"))] | length == 1;
{
  launched: {run1: ($U.run1 | length), run2: ($U.run2 | length)},
  x_rows: ($xr | length), x_rows_errored: ($xr | n(.errored)),
  x_rows_not_in_launched_universe: ($xr | n(in_u | not)),
  d_target_rows: ($dr | length), d_target_rows_errored: ($dr | n(.errored)),
  d_target_rows_without_sample_error: ($dr | n(.errored | not)),
  errored_x_d_overlap: ([$xr[] | select(.errored) | [.arm, .tid]] - ([$xr[] | select(.errored) | [.arm, .tid]] - [$dr[] | select(.errored) | [.arm, .tid]]) | length),
  errored_in_launched_universe: ($err | n(in_u)),
  run1: {runs: ($U.run1 | length), errored: ($err | n(.arm == "run1")), epce: ($err | n(.arm == "run1" and .epce))},
  run2: {runs: ($U.run2 | length), errored: ($err | n(.arm == "run2")), epce: ($err | n(.arm == "run2" and .epce))},
  errored: ($err | length), epce: ($err | n(.epce))
}
JQ
)

# Honest epoch-2 attempts the launcher counted as unusable ("usable 0"), real (input P; R via --slurpfile).
JQ_EP2_USABLE0=$(cat <<'JQ'
def n(f): [.[] | select(f)] | length;
def env: split("|")[1] | sub("^(t|nm)[0-9]+_"; "") | sub("_redraw[0-9]+$"; "");
def envs: ("model_registry", "slack_clone", "web_scraping", "shared_dotfiles", "media_processing", "tiktok", "auto_workflow");
def launches: {ep2_311: "ep2_311($|[^b])", ep2_311b: "ep2_311b", ep2_nm: "ep2_nm", ep2_t196: "ep2_t196"};
[ .epoch2_candidates | to_entries[] | .key as $g | .value.status_rows_removed[]
  | select(.class | test("usable 0")) | {group: $g, launch_id, key, eval_sha256, env: (.key | env)} ] as $u
| ($R[0] | [(.real.failed_runs[][] | .key), (.real.S_old_honest_a_apart.named_failures_by_pool_rule[].keys[])]
   | map(split("/") | [.[0], .[2]])) as $r_launch_task
| {
  total: ($u | length),
  unique_launch_key: ($u | map([.launch_id, .key]) | unique | length),
  unique_eval_sha256: ($u | map(.eval_sha256) | unique | length),
  by_group: ([.epoch2_candidates | keys[] as $g | {key: $g, value: ($u | n(.group == $g))}] | from_entries),
  by_launch: (launches | map_values(. as $re | $u | n(.launch_id | test($re)))),
  rows_matching_not_exactly_one_launch: ($u | n(.launch_id as $l | [launches[] | select(. as $re | $l | test($re))] | length != 1)),
  by_arm: {honest_a: ($u | n(.key | startswith("honest_a|"))), honest_b: ($u | n(.key | startswith("honest_b|")))},
  model_registry: ($u | n(.env == "model_registry")),
  slack_clone: ($u | n(.env == "slack_clone")),
  other_env: ($u | n(.env != "model_registry" and .env != "slack_clone")),
  other_env_by_env: ([envs as $v | select($v != "model_registry" and $v != "slack_clone")
                      | {key: $v, value: ($u | n(.env == $v))}] | from_entries),
  env_outside_named_envs: ($u | n(.env as $v | [envs] | index($v) | not)),
  overlap_with_R_failed_launch_task: ($u | n([.launch_id, (.key | split("|")[1])] as $lt | $r_launch_task | index([$lt]) != null)),
  overlap_control_launch_in_R_failed: ($u | n(.launch_id as $l | [$r_launch_task[][0]] | index($l) != null)),
  overlap_control_task_in_R_failed: ($u | n((.key | split("|")[1]) as $t | [$r_launch_task[][1]] | index($t) != null))
}
JQ
)

# Combine, check, and assemble the record.
JQ_COMBINE=$(cat <<'JQ'
($a.three_sets.runs + $p.run1.runs + $p.run2.runs) as $runs
| {
  honest_real: $h,
  honest_real_epoch2_usable0: $u,
  attack_real: {
    S_old_S_new_S_newmains: $a,
    S_paper493: $p,
    total: {runs: $runs,
            errored: ($a.three_sets.errored + $p.errored),
            epce: ($a.three_sets.epce + $p.epce),
            operation_error: $a.three_sets.operation_error,
            retry_error: $a.three_sets.retry_error}
  }
}
| . as $c
| {counts: $c, checks: {
    honest_run_keys_unique: ($h.failed_unique_run_keys == $h.failed_total),
    honest_failed_runs_rows_match_by_set_arm: ($h.failed_runs_rows == $h.failed_runs_rows_by_set_arm_sum),
    honest_sets_all_named: ($h.failed_set_outside_named_sets == 0),
    honest_epce_envs_all_named: ($h.epce_env_outside_named_envs == 0),
    honest_epce_matches_by_category: (($h.epce_total - $h.epce_S_old_honest_a_apart) == $h.epce_by_category_sum_failed_runs_arms),
    attack_has_error_agrees: ($a.has_error_disagrees_with_sample_error == 0),
    attack_real_rows_all_in_three_sets: ($a.real_run_rows == $a.three_sets.runs),
    attack_p493_errored_in_universe: ($p.errored_in_launched_universe == $p.errored),
    attack_p493_no_x_d_overlap: ($p.errored_x_d_overlap == 0),
    ep2_usable0_unique: ($u.unique_launch_key == $u.total and $u.unique_eval_sha256 == $u.total),
    ep2_usable0_groups_sum: (([$u.by_group[]] | add) == $u.total),
    ep2_usable0_one_launch_per_row: ($u.rows_matching_not_exactly_one_launch == 0),
    ep2_usable0_arms_sum: (($u.by_arm.honest_a + $u.by_arm.honest_b) == $u.total),
    ep2_usable0_envs_all_named: ($u.env_outside_named_envs == 0),
    ep2_usable0_no_overlap_with_R: ($u.overlap_with_R_failed_launch_task == 0)
  }}
JQ
)

H=$(jq -c "$JQ_HONEST" "$TMP/R.json")
AA=$(jq -c "$JQ_ATTACK_A" "$TMP/A.json")
P=$(jq -n -c --slurpfile X "$TMP/X.json" --slurpfile D "$TMP/D.json" --slurpfile B "$TMP/B.json" "$JQ_ATTACK_P493")
U=$(jq -c --slurpfile R "$TMP/R.json" "$JQ_EP2_USABLE0" "$TMP/P.json")
C=$(jq -n -c --argjson h "$H" --argjson a "$AA" --argjson p "$P" --argjson u "$U" "$JQ_COMBINE")

if ! jq -e '.checks | all' <<< "$C" > /dev/null; then
  print -u2 "STOP: failed checks: $(jq -r '.checks | to_entries | map(select(.value | not) | .key) | join(", ")' <<< "$C")"
  exit 1
fi

HEAD=$(git -C "$REPO" rev-parse HEAD)
SCRIPT_SHA=$(shasum -a 256 "${0:A}" | cut -d' ' -f1)
SCRIPT_BLOB=$(git -C "$REPO" hash-object "${0:A}")
HEAD_BLOB=$(git -C "$REPO" rev-parse -q --verify "HEAD:$SCRIPT_REL" || print none)

jq -n \
  --arg stamp "$STAMP" --arg head "$HEAD" --arg script "$SCRIPT_REL" --arg ssha "$SCRIPT_SHA" \
  --argjson committed "$([[ $SCRIPT_BLOB == $HEAD_BLOB ]] && print true || print false)" \
  --arg jqv "$(jq --version)" --argjson inputs "$INPUTS" --argjson c "$C" \
  --arg jh "$JQ_HONEST" --arg ja "$JQ_ATTACK_A" --arg jp "$JQ_ATTACK_P493" --arg ju "$JQ_EP2_USABLE0" \
  --arg jc "$JQ_COMBINE" \
  '{label: "App I real-side errored-run counts (honest pools and attack runs); keys and counts only",
    stamp_pt: $stamp, git_head: $head,
    script: {path: $script, sha256: $ssha, matches_git_head: $committed},
    jq_version: $jqv, inputs: $inputs,
    jq: {honest_real_R: $jh, attack_real_A: $ja, attack_real_S_paper493_XDB: $jp,
         honest_real_epoch2_usable0_P: $ju, combine: $jc},
    counts: $c.counts, checks: $c.checks}' > "$TMP/out.json"

mkdir -p "${OUT:h}"
cp -n "$TMP/out.json" "$OUT"
jq -r '.counts | paths(scalars) as $p | "\($p | join("."))\t\(getpath($p))"' "$OUT"
print -u2 "wrote ${OUT#$REPO/}"
