# Blobs and git listings that the SC-rest checks parse (mirrors every blob() / git() call site of selfcheck_release_record.py v2.3
# outside L56-86). Emits TSV, keys/paths/commits only:
#   J   commit path      a JSON blob to parse
#   JL  commit path      a JSONL blob (parsed per line)
#   SH  commit -         `git show --name-only --format= <commit>` file list
#   LT  commit dir       `git ls-tree -r --name-only <commit> <dir>` list
def truthy: . != null and . != false and . != "" and . != 0 and . != [] and . != {};
. as $r
| ($r.in_scope_outputs + $r.preliminary_outputs) as $outs
| ([$r.exclusions[] | select(.id == "downstream_pending_decision2")][0]) as $pex
| (
  # L103-104 G7b record
  ["J", $r.final_bar.g7b_value.commit, $r.final_bar.g7b_value.path],
  # L149-178 outputs (printed_items groups are skipped there)
  ($outs[] | select(.kind != "printed_items") |
     (if (.pending_downstream_label | truthy) and .kind != "figure_split" then
        ["J", .s2_marker.commit, .s2_marker.path], ["SH", .commit, "-"]
      elif .kind == "figure_split" then
        ["J", .s2_marker.commit, .s2_marker.path], ["SH", .commit, "-"]
      else
        ["J", .s2_marker.commit, .s2_marker.path], ["J", .cover.commit, .cover.path], ["SH", .commit, "-"]
      end)),
  # L182-191 records with an S2 marker
  (($r.in_scope_records + $r.out_of_scope_records)[] | select(has("s2_marker")) |
     ["J", .s2_marker.marker.commit, .s2_marker.marker.path], ["J", .s2_marker.cover.commit, .s2_marker.cover.path]),
  # L192-196 derivation result
  ((if ($r.derivation_check.result | type) == "object" then $r.derivation_check.result else $r.derivation_check.result_available_but_not_used end) |
     ["J", .commit, .path]),
  # L207-216 (sensitivity parts) and L222-235 (exclusion files): every .json file of every output that has files
  ($outs[] | select(.files | truthy) | .files[] | select(.path | endswith(".json")) | ["J", .commit, .path]),
  # L254-255 element tables
  (($r.element_tables // [])[] | ["J", .commit, .file]),
  # L388-390 family halves; L277-323 B / R37 per-quantity halves
  (($r.exclusions // [])[] | (.halves // {}) | to_entries[] | .value | select(type == "object") |
     (if .item_schema == "family_all_labels" then ["J", .commit, .path] else empty end)),
  (($r.exclusions // [])[] | select(.entry_parsers | truthy) |
     ((.halves.a1_B_R37.union_of | to_entries[] | .value | select((.held | truthy) | not) | ["J", .commit, .path]),
      (.halves.g7b_B_R37.union_of | to_entries[] | .value | .files[] | ["J", .commit, .path]))),
  # L462-464 pending slots regex (rc1a's first file)
  (if ($r.pending_downstream | truthy) and ($pex.slot_regex | truthy) then
     ($r.in_scope_outputs[] | select(.name == "rc1a") | .files[0] | ["J", .commit, .path]) else empty end),
  # L540 counts check result
  (if ($r.counts_check | truthy) then ["J", $r.counts_check.result.commit, $r.counts_check.result.path] else empty end),
  # L137 / L439 / L490 provenance
  (if ($r.printed_items | truthy) then ["JL", $r.printed_items.provenance.commit, $r.printed_items.provenance.path] else empty end),
  # v2.5 (A2): the draft_scope rows' printed token and anchor source (v19)
  (if ($r.row_rulings_v2_5.draft_scope.printed_and_anchor_source | truthy) then ["JL", $r.row_rulings_v2_5.draft_scope.printed_and_anchor_source.commit, $r.row_rulings_v2_5.draft_scope.printed_and_anchor_source.path] else empty end),
  # L545-546 signoff listing
  ["LT", $r.amendment.commit, (($r.amendment.file | split("/") | .[:-1] | join("/")) + "/signoff/")]
  )
| @tsv
