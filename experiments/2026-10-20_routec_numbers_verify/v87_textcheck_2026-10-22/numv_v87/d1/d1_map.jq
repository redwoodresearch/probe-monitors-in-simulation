# D1 row mapping (PRESTATED_D1_FROZEN_2026-10-21_205113.md, Method). Keys only: row ids and counts, never text.
#   jq -n -r -f d1_map.jq --rawfile xs <X stripped> --rawfile vs <other stripped> --slurpfile v19 <provenance v19 jsonl>
#        --slurpfile dfr <deferred v8 jsonl> --slurpfile cons <constants v10 jsonl> --slurpfile hunks <[{idx, kind, os, oe}] json>
# Anchor lines come from string splits (no offsets): occurrence k of anchor a in T starts on line 1 + NL(P[0..k]) + k * NL(a), with P = T | split(a),
# and spans NL(a) more lines. A row maps to every c / d hunk whose old line range [os, oe] overlaps a span. "Vanished" = >= 1 occurrence in X's
# stripped text and 0 in the other's. "not_in_x" = 0 occurrences in X's stripped text (e.g. an anchor holding a hidden comment); listed, not mapped.
# Deferred rows without a row id are named "line<N>" (1-based file line), as verify_post.deferred_row_id does.
def nl: [scan("\n")] | length;
def spans($t): . as $a
  | if ($a | type) != "string" or ($a | length) == 0 then []
    else ($t | split($a)) as $p | ($a | nl) as $na
    | [range(0; ($p | length) - 1) as $k
       | (1 + ([$p[0:$k + 1][] | nl] | add) + $k * $na) as $s | [$s, $s + $na]]
    end;
def rows: [($v19 | to_entries[] | {set: "v19", id: (.value.row | tostring), a: .value.anchor}),
           ($dfr | to_entries[] | {set: "deferred", id: (if .value.row == null then "line\(.key + 1)" else (.value.row | tostring) end), a: .value.anchor}),
           ($cons | to_entries[] | {set: "constants", id: (.value.row | tostring), a: .value.anchor})];
def ids($s): [.[] | select(.set == $s) | .id];
[rows[] | . + {xs: (.a | spans($xs)), vn: (.a | spans($vs) | length)}] as $R
| ($hunks[0] | map(select(.kind == "c" or .kind == "d"))) as $H
| ([$H[] | . as $h | {idx: $h.idx, m: [$R[] | select(any(.xs[]; .[0] <= $h.oe and .[1] >= $h.os))]}] | map(select(.m | length > 0))) as $M
| ([$R[] | select((.xs | length) > 0 and .vn == 0)]) as $V
| ([$R[] | select((.xs | length) == 0)]) as $N
| ($M[] | "D1_ROWS hunk=\(.idx) v19=\(.m | ids("v19") | tojson) deferred=\(.m | ids("deferred") | tojson) constants=\(.m | ids("constants") | tojson)"),
  "D1_VANISHED v19=\($V | ids("v19") | tojson) deferred=\($V | ids("deferred") | tojson) constants=\($V | ids("constants") | tojson)",
  "D1_ANCHOR_NOT_IN_X v19=\($N | ids("v19") | tojson) deferred=\($N | ids("deferred") | tojson) constants=\($N | ids("constants") | tojson)",
  "D1_MAP_COUNTS rows=\($R | length) hunks_with_rows=\($M | length) vanished=\($V | length) not_in_x=\($N | length)"
