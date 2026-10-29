# Route C section 3: 09's S1b-item7 sign-off marker, in 09's format (via 2e, 10-18). The marker is
# experiments/2026-10-14_priority1_spec_09/signoff/S1b-item7_<YYYY-MM-DD_HHMMSS>.json on aswin/convergence-audit-09. Required:
#   signoff "S1b-item7"; signed true; by "09"; items exactly {"7": "PASS"}; time and notes present (values not checked);
#   covered exactly 3 entries, SET-equal to the three pooled pins as filled (PWM, PSR, PWA; path@sha256 strings).
# S3LINE checks the marker's commit, path and sha256 first, then runs:
#   jq -c --arg pwm "$PWM" --arg psr "$PSR" --arg pwa "$PWA" -f s1b7_marker.jq <marker>
# Prints {"ok": <bool>, "failing": [<check names>]}: names only.
if type != "object" then {ok: false, failing: ["not_an_object"]}
else
  [ (if .signoff != "S1b-item7" then "signoff" else empty end),
    (if .signed != true then "signed" else empty end),
    (if .by != "09" then "by" else empty end),
    (if .items != {"7": "PASS"} then "items" else empty end),
    (if (has("time") and has("notes")) | not then "time_or_notes" else empty end),
    (if (.covered | type) != "array" or (.covered | length) != 3 or (.covered | sort) != ([$pwm, $psr, $pwa] | sort) then "covered" else empty end) ]
  | {ok: (length == 0), failing: .}
end
