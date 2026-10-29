# Addendum 19: the cleanroom_simpool result (laptop file: booleans and counts only) against the pre-stated outcomes. Keys and counts only.
# Use (with set -o pipefail): jq -f simpool_checks.jq <results/cleanroom_simpool_<stamp>.json> | tee <summary> | jq -e '.OK' > /dev/null
{n, n_mapped, n_paths_with_copies, n_unknown, n_stops, pilot, unreadable: (.pilot_header_unreadable | length), equal, verdict_OK}
| .OK = (.n == 868 and .n_mapped == 868 and .n_paths_with_copies == 868 and .n_unknown == 0 and .n_stops == 0
         and .pilot.files == 20 and .pilot.sha_ok == 20 and .pilot.header_read == 20 and .pilot.in_pool == 20 and .unreadable == 0
         and .equal.n and .equal.sol_using and .equal.sol_free and .equal.unknown and .equal.percent_0dp and .verdict_OK)
