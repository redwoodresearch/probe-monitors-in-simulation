# Route C section 6 (09's addition to GATES Amendment 13 (2), via 2e, 10-18): bind this job's section 4 pre-flight record to its
# section 3 pickle, so the two entries an EQ4 release names belong together. The record is K by its writer, preflight_routec.py @ 37224810:
# main (L208) reads the pickle at its --spec argument (spec_b, L218) and writes "spec": {"local": <that argument>, "sha256": sha256 of
# those bytes} into the record (L241; the record dict L240, json.dumps L247, the exclusive "xb" write L249-250). PF4S passes --spec "$PKL"
# from section 3's ok.txt. So the record's spec.sha256 must equal PLS and its spec.local must equal PKL.
#   git show "${RH6}:$E/$POUT" | jq -c --arg s3 "${PKL}@${PLS}" -f pf4_record_binds_s3.jq
# Prints {"ok": <bool>, "failing": [<check names>]}: names only.
($s3 | sub("@[^@]*$"; "")) as $pkl | ($s3 | split("@") | last) as $pls
| if type != "object" or (.spec | type) != "object" then {ok: false, failing: ["no_spec"]}
  else
    [ (if ($pls | test("^[0-9a-f]{64}$")) | not then "s3_form" else empty end),
      (if .spec.sha256 != $pls then "spec_sha256" else empty end),
      (if .spec.local != $pkl then "spec_local" else empty end) ]
    | {ok: (length == 0), failing: .}
  end
