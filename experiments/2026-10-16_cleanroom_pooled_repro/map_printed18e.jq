# Addendum 18e: each printed key of 09's mapping (slot table rows 19 / 20; Fig 4 cohort_cells) -> the clean-room VERDICT rows or
# threshold rows that cover it, by key. Keys, booleans and counts only. A printed key is COVERED iff it has >= 1 covering row and
# every covering row passes (verdict rows: match_flip_rule; threshold rows: match and counts_ok and units_above_diff == 0).
# Use: jq --arg kind B|R37 --argjson labels '<labels>' --argjson levels '<levels>' -f map_printed18e.jq <compute record>
#   B:   19.units_above_tau_sim / 19.real_units_above_tau_B (thresholds S / B at 0.001), 19.distinct_runs_above_tau_sim (#distinct_runs
#        rows at 0.001, 4 origin rows), Fig 4 cohort_cells (coverage rows, 4 sets).
#   R37: 20.distinct_runs_above_tau_sim.<label>@<level> (#distinct_runs rows, 4 origin rows).
. as $c
| def vrows(f): [$c.comparison[] | select(.verdict and f)];
  def vk(rs; slot): {slot: slot, rows: (rs | length), passing: ([rs[] | select(.match_flip_rule)] | length),
                    covered: ((rs | length) > 0 and ([rs[] | .match_flip_rule] | all))};
  def tk(side; lab; lvl; slot): ([$c.thresholds[] | select(.label == lab and .level == lvl and .side == side)]) as $t
                    | {slot: slot, rows: ($t | length), passing: ([$t[] | select(.match and .counts_ok and .units_above_diff == 0)] | length),
                       covered: (($t | length) == 1 and ([$t[] | .match and .counts_ok and .units_above_diff == 0] | all))};
  [ if $kind == "B" then
      ($labels[] as $l
       | tk("S"; $l; "0.001"; "19.units_above_tau_sim.\($l)@0.001"),
         tk("B"; $l; "0.001"; "19.real_units_above_tau_B.\($l)@0.001"),
         vk(vrows(.path[0] == "rows" and .path[1] == $l and .path[4] == "0.001" and .path[5] == "S" and .path[6] == "#distinct_runs");
            "19.distinct_runs_above_tau_sim.\($l)@0.001")),
      ($labels[] as $l | vrows(.family == "cohort_cells" and .path[1] == $l)[] | vk([.]; "fig4.cohort_cells.\(.path[1]).\(.path[2] | split(" ")[0])"))
    else
      ($labels[] as $l | $levels[] as $v
       | vk(vrows(.path[0] == "rows" and .path[1] == $l and .path[4] == $v and .path[5] == "S" and .path[6] == "#distinct_runs");
            "20.distinct_runs_above_tau_sim.\($l)@\($v)"))
    end ] as $keys
| {kind: $kind, n_slots: ($keys | length), n_covered: ([$keys[] | select(.covered)] | length),
   rows_per_slot: ([$keys[].rows] | unique), slots: $keys}
