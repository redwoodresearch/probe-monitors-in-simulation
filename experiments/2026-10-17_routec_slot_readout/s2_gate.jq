# s2_gate.jq: 09's S2 marker gate (GATES Amendment 15) for the slot readout's rc1a / rc1b, the release point (09 and 2e, 10-18:
# "gate release, not computation"). Called by path@sha, never extracted from a NOTE (rule 15); tested by test_s2_gate.py.
#   cover: jq -n -c -f s2_gate.jq --arg mode cover --args <pointer>...
#          -> the run's COVER list: its input pointers as "<path>@<sha256>", sorted, unique (committed, then signed by 09)
#   gate:  <the S2 marker JSON> | jq -e -c -f s2_gate.jq --arg mode gate --argjson cover <the committed COVER list> --args <pointer>...
#          -> true only if the marker is signoff "S2", signed true, by "09", its covered list equals the COVER list as a set with no
#             duplicate entry, and the COVER list is exactly this run's input pointers; else false (exit 1) or an error (exit 5)
# A pointer is git:<40-hex commit>:<path>@<sha256>; its entry drops "git:<commit>:". Sidecars are in every COVER, and a .json.gz entry
# carries its decompressed JSON's sha256 (09: "<path>.json.gz@<sha256 of gzip -dc>"), which is the sha the readout checks.
def entry:
  if type == "string" and test("^git:[0-9a-f]{40}:[A-Za-z0-9._/-]+\\.json(\\.gz)?@[0-9a-f]{64}$")
  then sub("^git:[0-9a-f]{40}:"; "") else error("not a git:<commit>:<path>.json[.gz]@<sha256> pointer: \(.)") end;
def ok_entry: type == "string" and test("^[A-Za-z0-9._/-]+\\.json(\\.gz)?@[0-9a-f]{64}$");
def want:
  $ARGS.positional | map(entry)
  | if length == 0 then error("no input pointers") elif (unique | length) != length then error("two input pointers give one entry")
    else unique end;
$ARGS.named.mode as $mode
| if $mode == "cover" then want
  elif $mode == "gate" then
    want as $want | $ARGS.named.cover as $cover
    | ($cover | type == "array" and length > 0 and all(.[]; ok_entry) and . == unique) as $cover_ok
    | type == "object" and .signoff == "S2" and .signed == true and .by == "09"
      and (.covered | type == "array" and all(.[]; type == "string"))
      and $cover_ok
      and (.covered | unique) == ($cover | unique) and (.covered | length) == (.covered | unique | length)
      and $cover == $want
  else error("--arg mode must be cover or gate") end
