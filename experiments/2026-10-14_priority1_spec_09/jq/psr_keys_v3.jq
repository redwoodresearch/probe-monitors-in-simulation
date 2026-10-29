# 09 S1b-7 PSR, step 3: withheld REASON strings (counts by origin), nos4 evidence shape and tie / mismatch keys only (no score values).
def bools: if type == "object" then with_entries(select(.value | type == "boolean")) else {} end;
{withheld_by_origin: ([.rows[] | to_entries[] | select(.value | type == "object" and has("withheld")) | .key] | group_by(.) | map({o: .[0], n: length})),
 withheld_reasons: ([.rows[] | .[]? | select(type == "object" and has("withheld")) | (.withheld | tostring | .[0:160])] | group_by(.) | map({r: .[0], n: length})),
 nos4_rule: (.meta.without_S4.rule | tostring | .[0:200]),
 nos4_checked_pairs: (.meta.without_S4.checked_pairs | if type == "array" then length elif type == "object" then (keys | length) else . end),
 nos4_thresholds_compared_equal: (.meta.without_S4.thresholds_compared_equal | if type == "boolean" or type == "number" then . else type end),
 ties: (.meta.without_S4.alarm_indicator_ties | if type == "object" then {keys: keys, bools: bools, n_entries: ((.entries // []) | length),
          entry_keys: ((.entries // [])[0] | if type == "object" then keys else type end),
          n_mismatch: ([(.entries // [])[] | (.mismatch // [])[]] | length),
          problems: ([(.entries // [])[] | .problem // empty] | map(tostring | .[0:120]) | unique)} else type end),
 caveats: ([.rows[] | .[]? | select(type == "object" and has("caveat_without_S4")) | .caveat_without_S4 | tostring | .[0:300]] | unique)}
