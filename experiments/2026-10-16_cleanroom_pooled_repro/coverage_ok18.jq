# Addendum 18c (R2 P2-2): coverage_add18.json against the committed expectation: every one of the coverage_slots slots covered.
# Use (with set -o pipefail): jq --slurpfile e config/add18_expect.json -f coverage_ok18.jq <coverage_add18.json> | jq -e '.COVERAGE_OK' > /dev/null
{n_slots: (.slots | length), n_covered, n_not_covered, want: $e[0].coverage_slots}
| .COVERAGE_OK = (.n_slots == .want and .n_covered == .want and .n_not_covered == 0)
