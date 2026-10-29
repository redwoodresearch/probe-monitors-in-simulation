# Route C section 4.3 (2e, 10-18): checks around build_signoff_covered.py for A/B/R37's S1 covered list (skeleton 5a). Two modes:
#   --arg mode inputs: the filled inputs file (a copy of config/routec_signoff/S1_covered_inputs_TEMPLATE.json). No template placeholder
#     ("<…>") may remain, and .records must list the three FINAL section 4 pre-flight records (--argjson pf, "path@sha" strings; the
#     path part is checked here, the sha by the list mode).
#   --arg mode list: the builder's output. A JSON array of "path@sha256" strings, each path canonical (no "", ".", ".." segment, no
#     leading "/"), sorted and unique, each path once (one path with two shas STOPs: a data mix-up guard, a03f5d93 O-2), containing
#     all three --argjson pf entries exactly.
#   jq -c --arg mode <inputs|list> --argjson pf '["<repo path>@<sha256>", x3]' -f cover_check.jq <file>
# 2e (10-18): any .gz path (a record, a family item, or a list entry) STOPs as gz_path / gz_entry: 09's Amendment 15 hashes .gz DECOMPRESSED,
# the builder hashes the committed bytes, so a .gz entry waits for a reviewed change.
# Prints {"ok": <bool>, "failing": [<check names>]}: names only. The records are K: the pre-flight records by their writer,
# preflight_routec.py @ 37224810 (main L240-250); the list holds paths and shas only.
def canon: test("^[A-Za-z0-9._-]+(/[A-Za-z0-9._-]+)*$") and (split("/") | all(. != "." and . != ".."));
def entry_ok: type == "string" and test("^[^@]+@[0-9a-f]{64}$") and (sub("@[0-9a-f]{64}$"; "") | canon);
def gzname: ascii_downcase | gsub("%2e"; ".") | sub("[\\s.\\\\]+$"; "") | (endswith(".gz") or endswith(".tgz") or endswith(".gzip"));  # a63478 REQUIRED-1
(if $mode == "inputs" then
   if type != "object" then ["not_an_object"]
   else [ (if [.. | strings | select(test("^<.*>$"))] != [] then "template_placeholder_left" else empty end),
          (if (.records | type) != "array" then "records_form"
           elif ($pf | map(sub("@[0-9a-f]{64}$"; ""))) - .records != [] then "preflight_record_not_listed" else empty end),
          ([(.records // [] | if type == "array" then .[] else empty end), (.families // {} | if type == "object" then .[] else empty end | objects
             | ((.harness // [] | if type == "array" then .[] else empty end | objects | .path), (.regression // {} | objects | .path)))]
           | map(select(type == "string"))) as $paths
          | (if any($paths[]; canon | not) then "noncanonical_path" else empty end),  # a63478 A: e.g. .g%7A, which gzname cannot see
            (if any($paths[]; gzname) then "gz_path" else empty end) ]
   end
 elif $mode == "list" then
   if type != "array" then ["not_an_array"]
   else [ (if any(.[]; entry_ok | not) then "entry_form" else empty end),
          (if . != unique then "not_sorted_unique" else empty end),
          (if (unique | map(sub("@[0-9a-f]{64}$"; "")) | length) != (unique | map(sub("@[0-9a-f]{64}$"; "")) | unique | length) then "path_twice" else empty end),
          (if ($pf - .) != [] then "preflight_entry_missing" else empty end),
          (if any(.[]; type == "string" and (sub("@[0-9a-f]{64}$"; "") | gzname)) then "gz_entry" else empty end) ]
   end
 else ["mode"] end)
| {ok: (length == 0), failing: .}
