# Sensitivity-label mention scan of one output file (port of build_release_record.py L541-579 for a non-slot record): counts only.
# --argjson STEMS ["c_asis_uncut", "own_all_rs10_uncut_L48", "own_all_rs25_uncut_L48"]  --argjson OUTSIDE ["build_profile", "plan", "logs"]
def hits: if type == "object" then ([to_entries[] | (.key as $k | [$STEMS[] | select(. as $s | $k | contains($s))] | length) + (.value | hits)] | add // 0)
          elif type == "array" then (map(hits) | add // 0)
          elif type == "string" then (. as $x | [$STEMS[] | . as $s | ($x | [match($s; "g")] | length)] | add)
          else 0 end;
([to_entries[] | select(.key != "slots" and .key != "full") | .key as $k | {($k): ({($k): .value} | hits)}] | add // {} | with_entries(select(.value > 0))) as $by
| {slots_dict: ((.slots | type) == "object"), mentions_by_top_key_outside_slots: $by,
   mentions_outside_metadata_blocks: ([$by | to_entries[] | select(.key as $k | $OUTSIDE | index([$k]) == null) | .value] | add // 0)}
