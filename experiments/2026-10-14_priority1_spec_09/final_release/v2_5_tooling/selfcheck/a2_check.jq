# a2_check.jq (09's A2, via 2e): v19 == v18 on every row except v84_sha256, and on rows 119-121 exactly {anchor, printed, v84_sha256} differ.
# Field names only; never prints a value.   jq -n -f a2_check.jq --slurpfile a <v18 jsonl> --slurpfile b <v19 jsonl>   ->   true / false
($a | map({(.row | tostring): .}) | add) as $A | ($b | map({(.row | tostring): .}) | add) as $B
| ([$a[].row] | sort) == ([range(1; 252)]) and ([$b[].row] | sort) == ([range(1; 252)])
  and all(range(1; 252); tostring as $n | ([($A[$n] | keys[]), ($B[$n] | keys[])] | unique | map(select(. as $k | $A[$n][$k] != $B[$n][$k])))
        == (if . >= 119 and . <= 121 then ["anchor", "printed", "v84_sha256"] else ["v84_sha256"] end))
