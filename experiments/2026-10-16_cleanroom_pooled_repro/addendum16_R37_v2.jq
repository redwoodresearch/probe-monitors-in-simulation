# Addendum 16 v2 (addendum 17: expected counts as pre-stated --argjson nrows / nmem; otherwise identical to addendum16_R37.jq)
# Addendum 16 (09's ruling, a1 2026-10-20): re-evaluate R37's clean-room compute record under RULE item 37's group gate
# ("nothing is extrapolated"): a group aggregate is EXPECTED withheld at a level where any member is unreadable by the clean
# room's OWN readability. Keys, booleans and counts only.
# Use (with set -o pipefail):
#   jq -n --argjson nrows <N> --argjson nmem <M> --slurpfile cr <R37 compute record> --slurpfile rec <decompressed R37 record> \
#      -f addendum16_R37.jq | tee <summary> | jq -e -s 'length == 1 and .[0].PASS16 == true' > /dev/null
# My readability per label x mode x level: the clean room computed readable_rule37 and compared it with the record. A matching row
# (match_flip_rule true) means mine == the record's boolean; a non-matching one means mine == its negation. So mine is the clean
# room's value, proven by its own comparison, not copied.
def six: ["sol_a_uncut", "FIBS|uncut", "harmfulness|uncut", "refusal|uncut", "high-stakes|uncut", "own_all_rs_uncut_L48"];
def groups: {"headline_six_full": [0, 1, 2, 3, 4, 5], "off_policy_five_full": [0, 1, 2, 3, 4], "auroc06_subset": [0, 1, 2, 5]};
($cr[0]) as $c | ($rec[0]) as $R
| ([$c.comparison[] | select(.family == "readable_rule37")
    | {k: "\(.path[1])|\(.path[3])|\(.path[4])",
       mine: (if .match_flip_rule then $R.pooled[.path[1]].pooled_all_origins[.path[3]][.path[4]].readable_rule37
              else ($R.pooled[.path[1]].pooled_all_origins[.path[3]][.path[4]].readable_rule37 | not) end)}]
   | map({(.k): .mine}) | add // {}) as $myread
| ($c.comparison | map(select(.path[0] == "coheadline_signed" or .path[0] == "panel_groups"))) as $grows
| def gk_of(r): if r.path[0] == "coheadline_signed" then {gk: r.path[4], mode: r.path[2], lvl: r.path[3]}
                else {gk: r.path[5], mode: r.path[3], lvl: r.path[4]} end;
  def expect_withheld(g): (g.gk | split("|")) as [$grp, $ro]
                          | [groups[$grp][] | six[.] + " " + $ro | $myread["\(.)|\(g.mode)|\(g.lvl)"] == false] | any;
  def node_withheld(r): if r.path[0] == "coheadline_signed"
                        then ($R.coheadline_signed.pooled_all_origins[r.path[2]][r.path[3]][r.path[4]] | (type == "object" and has("unavailable") and (has("S") | not)))
                        else ($R.panel_groups.pooled_all_origins.pooled[r.path[3]][r.path[4]][r.path[5]].primary
                              | (type == "object" and has("withheld") and (has("mean_G_minus_mean_Y") | not))) end;
  [$grows[] | select(.verdict) | . as $r | gk_of($r) as $g | expect_withheld($g) as $ew
   | {what, ew: $ew, pass_orig: .match_flip_rule, node_withheld: node_withheld($r),
      pass16: (if $ew then (node_withheld($r) and (.match_flip_rule | not) and .delta_bucket == "ref absent") else .match_flip_rule end)}] as $g16
| [$c.comparison[] | select(.verdict and (.path[0] != "coheadline_signed" and .path[0] != "panel_groups"))] as $nong
| ($c.summary.verdict.group_membership.differs) as $mdiff
| {n_readable_rule37_mine: ($myread | length), n_unreadable_mine: ([$myread[] | select(. == false)] | length),
   n_verdict_rows: ([$c.comparison[] | select(.verdict)] | length),
   nongroup_rows: ($nong | length), nongroup_pass: ([$nong[] | select(.match_flip_rule)] | length),
   group_rows: ($g16 | length), group_expected_withheld: ([$g16[] | select(.ew)] | length),
   group_pass16: ([$g16[] | select(.pass16)] | length),
   group_fail16: [$g16[] | select(.pass16 | not) | .what],
   orig_failing_rows: ([$c.comparison[] | select(.verdict and (.match_flip_rule | not))] | length),
   membership_checked: $c.summary.verdict.group_membership.checked,
   membership_differs: ($mdiff | length),
   membership_differs_all_expected_withheld: ([$mdiff[] | split("|") as [$src, $mode, $lvl, $grp, $ro]
       | expect_withheld({gk: "\($grp)|\($ro)", mode: $mode, lvl: $lvl})] | all),
   other_verdict_parts_pass: ($c.summary.verdict | (.thresholds.passed == .thresholds.expected_from_A and .thresholds.sets_equal
       and .cells.sets_equal and .cells.passed == .cells.expected_from_A and .controls.passed == .controls.registered
       and (.min_cell_weight.rows_below | length) == 0 and (.registered_absent_from_A | length) == 0
       and (.rows.missing_from_mine | length) == 0 and (.rows.not_in_A | length) == 0))}
| .rows_pass16 = (.nongroup_pass + .group_pass16)
| .membership_pass16 = (.membership_checked - .membership_differs + (if .membership_differs_all_expected_withheld then .membership_differs else 0 end))
| .PASS16 = (.nongroup_pass == .nongroup_rows and (.group_fail16 | length) == 0 and .rows_pass16 == .n_verdict_rows
             and .n_verdict_rows == $nrows and .membership_pass16 == .membership_checked and .membership_checked == $nmem
             and .other_verdict_parts_pass)
