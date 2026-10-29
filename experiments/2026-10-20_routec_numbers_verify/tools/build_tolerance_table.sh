#!/bin/bash
# 09's predicate tolerance table built mechanically from writing's provenance, in the v14 format (2e / 09, 10-20): one row per
# provenance row, in order: {row, op, args, has_report_text, record}. args is the predicate without op / note / report, with the
# expected values masked by the SAME set the container's gate masks (run_numbers_verify._post: MASK below); record is the cited
# record's file name; closed_set is the code's op names (predicates.OPS). Writes a new file (never in place) and prints counts only.
#   bash tools/build_tolerance_table.sh <provenance.jsonl> <what> <out.json>     (from the verifier folder)
set -euo pipefail
PROV=$1
WHAT=$2
OUT=$3
[[ -e $OUT ]] && { echo "STOP: $OUT exists"; exit 1; }
MASK='["value", "values", "target", "min", "max", "value_string", "component_string", "expected"]'
CLOSED=$(sed -n '/^OPS = {/,/^}/p' predicates.py | grep -oE '^    "[a-z_]+"' | tr -d ' "' | jq -R . | jq -s -c .)
[[ $(jq length <<<"$CLOSED") -ge 1 ]] || { echo "STOP: no ops read from predicates.py"; exit 1; }
jq -S -s --argjson mask "$MASK" --argjson closed "$CLOSED" --arg what "$WHAT" '
  {what: $what, closed_set: $closed,
   rows: [.[] | (.predicate // {}) as $p
          | {row, op: $p.op,
             args: ($p | with_entries(select(.key | IN("op", "note", "report") | not))
                       | with_entries(if (.key | IN($mask[])) then .value = "<expected value: checked in the container>" else . end)),
             has_report_text: ($p | has("report")),
             record: ((.record // "") | sub("@.*$"; "") | sub("^.*/"; ""))}]}' "$PROV" > "$OUT"
jq -r '"rows \(.rows | length)  closed_set \(.closed_set | length)"' "$OUT"
