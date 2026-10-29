# (3a') stage 1: for every D1 hunk of kind "c", emit both sides as code points, one per line, tagged "<idx> O <cp>" (X side) and
# "<idx> N <cp>" (v87 side), for a split into per-hunk files that POSIX `diff` aligns character by character. Hunk sides follow D1
# (d1_cores.jq): X stripped lines os..oe; v87 lines ns..ne; joined with "\n". Code points only, never text on the terminal: the caller
# writes this stream to a work file.
#   jq -n -r -f emit_3a.jq --rawfile xs <X stripped> --rawfile vs <v87> --rawfile ht <D1 hunks.txt>
($xs | split("\n")) as $XL | ($vs | split("\n")) as $VL
| [$ht | split("\n")[] | select(startswith("HUNK ")) | split(" ")
   | {idx: (.[1] | tonumber), kind: .[2], os: (.[3] | tonumber), oe: (.[4] | tonumber), ns: (.[5] | tonumber), ne: (.[6] | tonumber)}] as $H
| $H[] | select(.kind == "c") | . as $h
| (($XL[$h.os - 1:$h.oe] | join("\n")) | explode[] | "\($h.idx) O \(.)"),
  (($VL[$h.ns - 1:$h.ne] | join("\n")) | explode[] | "\($h.idx) N \(.)")
