# 09 S1b-7 PSR, step 2: shas / paths / booleans / counts / key names only. No score values; no object dumps.
def bools: if type == "object" then with_entries(select(.value | type == "boolean")) else {} end;
def shape: if type == "object" then {keys: keys, bools: bools} elif type == "array" then {array_len: length} else {type: type} end;
{script_sha256: .meta.script_sha256,
 git: .meta.git,
 inputs_full: (.meta.full_length.inputs_full | if type == "object" then {path, sha256, status} else type end),
 row_store: (.meta.row_store | shape),
 without_S4: (.meta.without_S4 | shape),
 pools_check: (.meta.pools_check | shape),
 one_cohort_check: (.meta.one_cohort_check | shape),
 e12_pool_key_check: (.meta.e12_pool_key_check | shape),
 all_origins_total_weight_check: (.all_origins_total_weight_check | shape),
 coverage: (.coverage | shape),
 n_labels: (.rows | length),
 n_rows: ([.rows[] | .[]?] | length),
 n_withheld: ([.rows[] | .[]? | select(type == "object" and has("withheld"))] | length),
 withheld_origins: ([.rows | to_entries[] | .key as $l | .value | to_entries[] | select(.value | type == "object" and has("withheld")) | "\($l) | \(.key)"] | .[:10])}
