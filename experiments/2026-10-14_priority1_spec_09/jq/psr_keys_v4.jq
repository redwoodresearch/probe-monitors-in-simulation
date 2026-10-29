# 09 S1b-7 PSR, step 4: label / origin KEYS per withheld reason (first 60 chars), and the single tie entry's key fields (no z / T values).
{by_reason: ([.rows | to_entries[] | .key as $l | .value | to_entries[] | select(.value | type == "object" and has("withheld"))
              | {r: (.value.withheld | tostring | .[0:60]), l: $l, o: .key}] | group_by(.r)
              | map({r: .[0].r, labels: (map(.l) | unique), origins: (map(.o) | unique)})),
 tie_entry: ((.meta.without_S4.alarm_indicator_ties.entries // [])[0] | if type == "object" then
              {label, pair, mode, level, pool, order_differed, order_positions_differ, problem,
               tie: (.tie | if type == "array" then map(if type == "object" then del(.z, .T, .zT, .value) else . end) else type end),
               mismatch: (.mismatch | if type == "array" then map(if type == "object" then del(.z, .T, .zT, .value) else . end) else type end),
               n_unit_draws: (.unit_draws | length)} else type end)}
