# leafdiff.jq: the exact leaf-level difference between two JSON documents (C21). Leaves are every non-object, non-array value (null and
# false included). Output {removed, added, changed}: lists of paths (arrays), sorted.   jq -n -c -f leafdiff.jq --slurpfile a A --slurpfile b B
def lm: [paths(type != "object" and type != "array") as $p | {key: ($p | tojson), value: getpath($p)}] | from_entries;
($a[0] | lm) as $A | ($b[0] | lm) as $B
| {removed: [$A | keys[] | . as $k | select($B | has($k) | not) | fromjson],
   added: [$B | keys[] | . as $k | select($A | has($k) | not) | fromjson],
   changed: [$A | keys[] | . as $k | select(($B | has($k)) and $A[$k] != $B[$k]) | fromjson]}
