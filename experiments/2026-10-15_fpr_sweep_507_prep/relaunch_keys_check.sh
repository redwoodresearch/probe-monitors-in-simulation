#!/bin/zsh
# Keys-only check of a full-length Stage A record's honest_b_relaunch rows (09's asserts (2)/(3) on relaunch_alias): per S_old row x probe, the
# renamed relaunch rows and the pool-keys-vs-958-members check the row recorded. Prints counts and booleans only (no score). Usage: $0 <record.json>
R="$1"
jq -r '
  .rows | to_entries[] | .key as $l | .value | to_entries[]
  | select(.key | startswith("S_old"))
  | .key as $o | .value as $r
  | [$l, $o,
     ($r.honest_b_relaunch_key_alias.renamed_rows // "no_alias_field"),
     ($r.honest_b_relaunch.pools_check.pool_runs // "na"),
     ($r.honest_b_relaunch.pools_check.members // "na"),
     ($r.honest_b_relaunch.pools_check.pool_keys_equal_members // "na"),
     (($r.honest_b_relaunch.pools_check.members_without_units // []) | length),
     (($r.honest_b_relaunch.pools_check.rows_outside_membership_listed_extras // []) | length),
     ($r.honest_b_relaunch.withheld // $r.honest_b_relaunch.unavailable // $r.withheld // "ok")]
  | @tsv' "$R" | sort | awk -F'\t' 'BEGIN{print "label\torigin\trenamed_rows\tpool_runs\tmembers\tkeys_equal\tmembers_wo_units\toutside_listed\tstatus"} {print}'
echo "--- summary"
jq -r '[.rows[] | to_entries[] | select(.key | startswith("S_old")) | .value.honest_b_relaunch.pools_check // empty]
       | {rows_with_pools_check: length, all_keys_equal: (map(.pool_keys_equal_members) | all), all_958: (map(.pool_runs == 958 and .members == 958) | all),
          max_members_without_units: (map(.members_without_units | length) | max)}' "$R"
