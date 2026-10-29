# POST HOC (not pre-stated; disclosed): hunk 8's one added core vs the x100 decimal shift of each v10 literal holding exactly one core.
# Keys and booleans only. jq -n -f posthoc_h8.jq --rawfile vs <v87> --rawfile re <regex> --slurpfile r <real.json> --slurpfile v10 <v10>
def parts($s): ($s | sub("^[−+-]"; "") | sub("%$"; "") | gsub(","; "")) as $u | ($u | split(".")) as $p | {i: ($p[0] | sub("^0+(?=[0-9])"; "")), f: ($p[1] // "")};
def shift2($s): parts($s) as $x | ($x.f + "00") as $f2 | {i: (($x.i + $f2[0:2]) | sub("^0+(?=[0-9])"; "")), f: ($x.f[2:])};
($r[0].hunks[] | select(.idx == 8) | .added_pos) as $ap
| ([$vs | match($re; "g") | select(.offset == $ap[0]) | .string | sub(",+$"; "")][0]) as $c
| parts($c) as $v
| [$v10[] | . as $row | ([$row.text | match($re; "g") | .string]) as $lc | select(($lc | length) == 1)
   | {row: $row.row, shift_equal: (shift2($lc[0]) == $v), plain_equal_without_pct: (parts($lc[0]) == $v and ($lc[0] | endswith("%") | not))}] as $T
| {n_added: ($ap | length), added_has_pct: ($c | endswith("%")),
   row2_one_core: ([$T[] | select(.row == 2)] | length == 1),
   row2_shift_equal: ([$T[] | select(.row == 2) | .shift_equal][0] // null),
   rows_shift_equal: [$T[] | select(.shift_equal) | .row], rows_equal_without_pct: [$T[] | select(.plain_equal_without_pct) | .row]}
