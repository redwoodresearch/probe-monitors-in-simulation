# Route C section 6: 09's EQ4-release marker (GATES Amendment 13 (2), via 2e, 10-18). S6LINE accepts EITHER EQ4's eq_ok.txt for this job
# OR this marker, experiments/2026-10-14_priority1_spec_09/signoff/EQ4-release_<J>_<YYYY-MM-DD_HHMMSS>.json on aswin/convergence-audit-09
# (S6LINE checks the commit on the fetched branch, the path and the file's sha256 first). Required:
#   signoff "EQ4-release:<J>" (J = --arg j: A, B or R37); signed true; by "09"; time and notes present (values unchecked);
#   items exactly {"eq4": "PASS"};
#   covered (strings) contains --arg f4, this job's FINAL section 4 record as OUT@RS (byte-equal to what eq_ok.txt names), and --arg s3,
#   this job's FINAL section 3 record as PKL@PLS (as section 3's ok.txt names it); no entry gives either path another sha, and no entry
#   names another job's record (routec_<K>_ or routec_spec_<K>_ with K != J).
#   jq -c --arg j "$J" --arg f4 "${POUT}@${PRS}" --arg s3 "${PKL}@${PLS}" -f eq4_release_marker.jq <marker>
# Prints {"ok": <bool>, "failing": [<check names>]}: names only.
def path_of: sub("@[^@]*$"; "");
if type != "object" then {ok: false, failing: ["not_an_object"]}
else
  .covered as $c
  | [ (if (["A", "B", "R37"] | index([$j])) == null then "job" else empty end),
      (if .signoff != ("EQ4-release:" + $j) then "signoff" else empty end),
      (if .signed != true then "signed" else empty end),
      (if .by != "09" then "by" else empty end),
      (if .items != {"eq4": "PASS"} then "items" else empty end),
      (if (has("time") and has("notes")) | not then "time_or_notes" else empty end),
      (if ($c | type) != "array" or ($c | length) == 0 or any($c[]; type != "string") then "covered_form"
       elif ([$f4, $s3] - $c) != [] then "covered_missing"
       elif any($c[]; . as $e | any($f4, $s3; path_of == ($e | path_of) and . != $e)) then "covered_other_sha"
       elif any($c[]; test("/routec_(spec_)?(A|B|R37)_") and capture("/routec_(spec_)?(?<k>A|B|R37)_").k != $j) then "covered_other_job"
       else empty end) ]
  | {ok: (length == 0), failing: .}
end
