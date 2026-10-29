# 09's S1a-consume markers (10-18): the one marker check for S1A3M (kind "S1a-consume", the full marker) and S1A3MC (kind "S1a-consume-CAT",
# CAT only). Called as  jq -e --arg kind <kind> --argjson pickles '[<sha256>, ...]' -f s1a_consume_marker.jq <marker.json>
# $pickles is EXACTLY the set of launched pickle sha256s the marker must cover: S1A3M passes rcPS's and rcCAT's, S1A3MC rcCAT's alone.
# true iff: the kind is one of the two, signoff == kind, signed, by 09, a time string and notes, items exactly {"3": "PASS"}, and the
# covered list's pickle entries (path.pkl@sha256) equal $pickles as a set of sha256s; anything else is false (the gates STOP).
($kind == "S1a-consume" or $kind == "S1a-consume-CAT")
and .signoff == $kind and .signed == true and .by == "09" and (.time | type == "string") and has("notes")
and .items == {"3": "PASS"} and (.covered | type == "array")
and ($pickles | type == "array" and length > 0 and all(.[]; type == "string" and test("^[0-9a-f]{64}$")))
and ([.covered[] | select(type == "string" and test("\\.pkl@[0-9a-f]{64}$")) | split("@") | last] | unique) == ($pickles | unique)
