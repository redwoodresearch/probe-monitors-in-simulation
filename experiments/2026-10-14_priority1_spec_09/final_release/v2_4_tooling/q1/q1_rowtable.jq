# Q1 mechanical per-row table (09, 22:46): for every final17 row NOT in v2.3 (rows 107-251), its family, the half-1 (G7b) entry key(s),
# the half-2 (a1 clean room) entry key(s), and the class under both readings of half 2 for interval components (see q1_rules.json).
# Keys, booleans and counts only. Inputs:
#   --slurpfile PROV final17 rows   --slurpfile V23 the v2.3 record   --slurpfile G7 A's G7b record   --slurpfile A1M a1 A compute
#   --slurpfile A1X a1 A18 compute  --slurpfile RULES q1_rules.json
def pyre: gsub("\\(\\?P<"; "(?<");
def fill($g): reduce ($g | to_entries[]) as $e (.; gsub("\\{" + $e.key + "\\}"; $e.value));
$RULES[0] as $R
| $V23[0] as $v
| ($PROV | map({(.row | tostring): .}) | add) as $P
# half 1: MATCH keys
| (reduce $G7[0].pooled[] as $pb ({}; reduce $pb.ledger[] as $kv (.; if $kv[1] == "MATCH" then .[$pb.row + "\u0001" + $pb.combo + "\u0001" + $kv[0]] = true else . end))) as $GP
| (reduce $G7[0].aggregates.ledger[] as $kv ({}; if $kv[1] == "MATCH" then .[$kv[0]] = true else . end)) as $GA
| ($GP | keys) as $GPK | ($GA | keys) as $GAK
# half 2: a1 verdict rows that pass (verdict, found, match_flip_rule)
| ([($A1M[0].comparison[], $A1X[0].comparison[]) | select(.verdict == true and .found == true and .match_flip_rule == true) | .path | map(tostring) | tojson] | unique | map({(.): true}) | add) as $A1
| def g1_point($fam; $g):
    if $fam.g7b_point.where == "pooled" then (($fam.g7b_point.row | fill($g)) + "\u0001" + $fam.g7b_point.combo + "\u0001" + ($fam.g7b_point.key | fill($g))) as $k | {key: ($k | gsub("\u0001"; " :: ")), ok: ($GP[$k] == true)}
    else ($fam.g7b_point.key | fill($g)) as $k | {key: $k, ok: ($GA[$k] == true)} end;
  def g1_interval($fam; $g):
    if $fam.g7b_interval == null then {key: null, ok: false, n: 0}
    elif $fam.g7b_interval.where == "pooled" then
      (($fam.g7b_interval.row | fill($g)) + "\u0001" + $fam.g7b_interval.combo + "\u0001") as $pre
      | ($fam.g7b_interval.regex | fill($g)) as $re
      | [$GPK[] | select(startswith($pre)) | select(.[($pre | length):] | test($re))] as $hits
      | {key: (($pre | gsub("\u0001"; " :: ")) + $re), ok: (($hits | length) == 21), n: ($hits | length)}
    else ($fam.g7b_interval.regex | fill($g)) as $re | [$GAK[] | select(test($re))] as $hits | {key: $re, ok: (($hits | length) == 21), n: ($hits | length)} end;
  def a1_node($fam; $g): ($fam.a1_path | map(fill($g)) | tojson) as $k | {key: $k, ok: ($A1[$k] == true)};
  def classify($slot; $comp):
    ([$R.families[] | . as $f | select($slot | test($f.regex | pyre)) | {f: $f, g: ($slot | capture($f.regex | pyre))}]) as $m
    | if ($m | length) != 1 then {family: "UNMATCHED", class_R_component: "PRELIMINARY", class_R_node: "PRELIMINARY", note: "no unique rule"}
      else $m[0].f as $f | $m[0].g as $g
      | if $f.id == "halfwidth" then
          [ $f.constituents[$g.which][] as $L | {L: $L} as $gg | ($R.families[] | select(.id == "delta_full")) as $df
            | {node: a1_node($df; $gg), interval: g1_interval($df; $gg)} ] as $cs
          | {family: $f.id, half1_key: [$cs[].interval.key], half1_ok: all($cs[]; .interval.ok), half2_key: [$cs[].node.key], half2_node_ok: all($cs[]; .node.ok),
             half2_component_reproduced: false, component_kind: "interval (arithmetic on interval ends)"}
          | . + {class_R_component: "PRELIMINARY", class_R_node: (if .half1_ok and .half2_node_ok then "FINAL" else "PRELIMINARY" end)}
        elif $f.id == "units_per_run_range" then
          ([$v.exclusions[] | select(.id == $f.exclusion_family)][0]) as $ex
          | ([$ex.entries[] | select(.slot == $slot)]) as $es
          | ($ex.halves | to_entries | map({half: .key, ok: (if .value.covered == null then true else (.value.covered | index([$slot]) != null) end), covered_null_means_all: (.value.covered == null)})) as $hv
          | {family: $f.id, half1_key: ($f.exclusion_family + " :: " + ($hv | map(.half) | join(" + "))), half1_ok: all($hv[]; .ok), half2_key: ($f.exclusion_family + " entry " + $slot),
             half2_node_ok: (($es | length) == 1 and $es[0].status == "FINAL"), half2_component_reproduced: true, component_kind: "range end (family bar A18)"}
          | . + {class_R_component: (if .half1_ok and .half2_node_ok then "FINAL" else "PRELIMINARY" end)} | . + {class_R_node: .class_R_component}
        else
          (if $comp == "interval_low" or $comp == "interval_high" then "interval" else "point" end) as $kind
          | a1_node($f; $g) as $a | (if $kind == "point" then g1_point($f; $g) else g1_interval($f; $g) end) as $h
          | {family: $f.id, half1_key: $h.key, half1_ok: $h.ok, half2_key: $a.key, half2_node_ok: $a.ok, half2_component_reproduced: ($kind == "point"), component_kind: $kind}
          | . + {class_R_component: (if .half1_ok and .half2_node_ok and .half2_component_reproduced then "FINAL" else "PRELIMINARY" end),
                 class_R_node: (if .half1_ok and .half2_node_ok then "FINAL" else "PRELIMINARY" end)}
        end
      end;
[ range($LO; $HI) as $n | $P[$n | tostring] as $p
  | ($p.record | split("/") | last | split("@") | .[0]) as $rec
  | if $rec == "slots_routeC_writing_rc1a.json" then
      if ($p.key_path[0] | type) == "string" then
        {row: $n, record: $rec, slot: $p.key_path[1], key_root: $p.key_path[0], component: ($p.predicate.component // null)} + classify($p.key_path[1]; $p.predicate.component // "point")
      else
        [$p.key_path[] | classify(.[1]; "point")] as $parts
        | {row: $n, record: $rec, slot: [$p.key_path[][1]], key_root: "multi", component: "all parts", family: ([$parts[].family] | unique | join("+")),
           half1_key: [$parts[].half1_key], half1_ok: all($parts[]; .half1_ok), half2_key: [$parts[].half2_key], half2_node_ok: all($parts[]; .half2_node_ok),
           half2_component_reproduced: all($parts[]; .half2_component_reproduced), component_kind: "points (all_within)",
           class_R_component: (if all($parts[]; .class_R_component == "FINAL") then "FINAL" else "PRELIMINARY" end),
           class_R_node: (if all($parts[]; .class_R_node == "FINAL") then "FINAL" else "PRELIMINARY" end)}
      end
    elif $rec == "slots_routeC_writing_rc1b.json" then
      ($p.key_path[1]) as $s
      | {row: $n, record: $rec, slot: $s, component: ($p.predicate.component // null),
         family: (if ($s | startswith("23.")) then "rc1b slot 23" elif ($s | startswith("21.") or startswith("22.")) then "rc1b slots 21/22" elif ($s | startswith("24.")) then "rc1b slot 24" else "rc1b other" end)}
      | . + (if .family == "rc1b slot 23" then {class_R_component: "PRELIMINARY", class_R_node: "PRELIMINARY", basis: $R.ruled_rows["119-122"].basis}
             elif .family == "rc1b slots 21/22" then {class_R_component: "FINAL", class_R_node: "FINAL", basis: $R.ruled_rows["231-240"].basis,
                    half1_key: "W2b S1 ebbfd0ae2 + S2 e4c53f971 (ported value half; per-member margins)", half2_key: "W1 v11 2750f6e18 (independent recompute)"}
             elif .family == "rc1b slot 24" then {class_R_component: "PENDING figs78", class_R_node: "PENDING figs78", basis: $R.ruled_rows["69-74, 241-248"].basis,
                    half1_key: "W2a S1r c2ec8baf5 + S2 6736f5f55 (supporting; not a condition)", half2_key: "a1 addendum 24 anchor package 069b5dfa @ 44f0cae2c (printed key via the figs78 item)"}
             else {class_R_component: "PRELIMINARY", class_R_node: "PRELIMINARY", basis: "no rule"} end)
    elif $rec == "cunits_by_tool_2026-10-20_044328.json" then
      {row: $n, record: $rec, slot: ($p.key_path | map(join(".")) | join(" / ")), family: "App A", class_R_component: "FINAL", class_R_node: "FINAL", basis: $R.ruled_rows["250-251"].basis,
       half1_key: "producer re-run of the original script on route C (cunits_by_tool @ a6d37c024)", half2_key: (if $n == 250 then "a1 / diag: UNRESOLVED (pre-stated), disclosed (ca18e9191)" else "same convention-independent count (the count under every convention: GATES L1362 @ d4466dbe5; ruling GATES 14:28)" end)}
    else {row: $n, record: $rec, family: "UNMATCHED RECORD", class_R_component: "PRELIMINARY", class_R_node: "PRELIMINARY"} end ] as $rows
| {rows: $rows,
   counts: {n: ($rows | length),
            by_family: ($rows | group_by(.family) | map({(.[0].family): {n: length,
                         R_component: (group_by(.class_R_component) | map({(.[0].class_R_component): length}) | add),
                         R_node: (group_by(.class_R_node) | map({(.[0].class_R_node): length}) | add)}}) | add),
            R_component: ($rows | group_by(.class_R_component) | map({(.[0].class_R_component): length}) | add),
            R_node: ($rows | group_by(.class_R_node) | map({(.[0].class_R_node): length}) | add)}}
