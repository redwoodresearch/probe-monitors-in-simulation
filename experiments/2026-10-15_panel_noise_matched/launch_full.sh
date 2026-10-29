#!/usr/bin/env bash
# Fill the full-length PANEL configs from the full-length Stage A records and launch both PANEL consumers in PARALLEL, detached (2e: speed first).
# git / jq / shasum only (0d's local rule); keys, shas and counts only, never a value. Every check runs BEFORE any repository output is written, committed, pushed
# or launched, and STOPs (exit 1) on a mismatch.
#
#   REC_GIT=<commit>:<path> REC_SHA=<sha256>       the full-length UNPLANTED Stage A record (headline)
#   PLANT_GIT=<commit>:<path> PLANT_SHA=<sha256>   the full-length PLANTED Stage A record (--plant run)
#   INPUTS_FULL_GIT=<commit>:<path> INPUTS_FULL_SHA=<sha256>   the full-length frozen inputs
#   STAMP=<YYYY-MM-DD_HHMM>   LAUNCH=1 to launch after filling (else print the commands)   MODAL=<modal binary> (optional)
#   PART=all (default) | panel (the unplanted record + INPUTS_FULL only: FINAL_full and run_pooled) | plant (the two records only: plant_rows_full and
#   run_plant_rows), so the headline panel never waits for the planted record
#   bash experiments/2026-10-15_panel_noise_matched/launch_full.sh
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
D=experiments/2026-10-15_panel_noise_matched
stop() { echo "STOP: $*" >&2; exit 1; }
PART="${PART:-all}"; case "$PART" in all) DO_PANEL=1; DO_PLANT=1 ;; panel) DO_PANEL=1; DO_PLANT=0 ;; plant) DO_PANEL=0; DO_PLANT=1 ;; *) stop "PART must be all, panel or plant" ;; esac
: "${REC_GIT:?}" "${REC_SHA:?}" "${STAMP:?}"
if [ "$DO_PLANT" = 1 ]; then : "${PLANT_GIT:?}" "${PLANT_SHA:?}"; fi
if [ "$DO_PANEL" = 1 ]; then : "${INPUTS_FULL_GIT:?}" "${INPUTS_FULL_SHA:?}"; fi
[[ "$STAMP" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}_[0-9]{4}$ ]] || stop "STAMP must be YYYY-MM-DD_HHMM"
git diff --quiet && git diff --cached --quiet || stop "the index or tracked files are not clean (commit or stash first; nothing is committed for you)"
MODAL="${MODAL:-/Users/aswinsurya/probe-distribution-shift-from-sims/.venv/bin/modal}"
if [ "${LAUNCH:-0}" = 1 ]; then [ -x "$MODAL" ] || stop "no modal binary at $MODAL"; fi
sha_of() { git show "$1" | shasum -a 256 | cut -c1-64; }
blob_at_sha() {  # the commit:path of the revision of <path> whose blob has <sha256> (any branch, any revision)
  local c; for c in $(git log --all --format=%h -- "$1"); do [ "$(sha_of "$c:$1" 2> /dev/null)" = "$2" ] && { echo "$c:$1"; return 0; }; done; return 1; }
W=$(mktemp -d); trap 'rm -rf "$W"' EXIT
T_FULL="$D/config/FINAL_FULLLENGTH_TEMPLATE.json"; T_PR="$D/config/plant_rows_FULL_TEMPLATE.json"
if [ "${POOLED:-0}" = 1 ]; then T_FULL="$D/config/FINAL_POOLED_FULL_TEMPLATE.json"; T_PR="$D/config/plant_rows_POOLED_FULL_TEMPLATE.json"; fi  # pooled calibration (09 RULE 42-46): the LEAD
NEED_WO=$(jq -r '.cohort_keys.extra_origins | has("S_newmains without S4")' "$T_FULL")  # a pooled build has no without-S4 combo unless the producer ran --pw-without-s4
PFX=""; if [ "${POOLED:-0}" = 1 ]; then PFX="pooled_"; fi  # pooled configs carry their own names (never collide with a per-set fill)
if [ "${PILOT:-0}" = 1 ]; then { [ "${POOLED:-0}" = 1 ] && [ "$PART" = panel ]; } || stop "PILOT=1 needs POOLED=1 and PART=panel"; PFX="pooled_pilot_"; fi
FULL="$D/config/FINAL_${PFX}full_${STAMP}.json"; PR="$D/config/plant_rows_${PFX}full_${STAMP}.json"; WO_F="$D/config/without_S4_nofilter_${PFX}full_${STAMP}.json"
if [ "$DO_PANEL" = 1 ]; then for f in "$FULL" "$WO_F"; do [ ! -e "$f" ] || stop "$f exists"; done; fi
if [ "$DO_PLANT" = 1 ]; then [ ! -e "$PR" ] || stop "$PR exists"; fi
# from here: the panel part runs in "if [ "$DO_PANEL" = 1 ]" blocks, the plant part in "if [ "$DO_PLANT" = 1 ]" blocks

# 1. the three inputs, by sha; both records are full-length
[ "$(sha_of "$REC_GIT")" = "$REC_SHA" ] || stop "record sha256"
git show "$REC_GIT" > "$W/rec.json"; [ "$(jq -r '.meta.window' "$W/rec.json")" = "full" ] || stop "rec record meta.window is not 'full'"
if [ "${POOLED:-0}" = 1 ] && [ "$DO_PANEL" = 1 ]; then  # 2e 10-16: the pooled record's producer revision (meta.git) EQUALS or DESCENDS FROM the reviewed PASS sha
  [ -n "${PRODUCER_SHA:-}" ] || stop "PRODUCER_SHA (the pooled producer reviewed PASS sha) is required for a pooled panel launch"
  GIT_TERMINAL_PROMPT=0 git fetch -q origin 2> /dev/null || true
  MG=$(jq -r '.meta.git // empty' "$W/rec.json")
  [ -n "$MG" ] && git merge-base --is-ancestor "$PRODUCER_SHA" "$MG" 2> /dev/null || stop "record meta.git '$MG' is not $PRODUCER_SHA or a descendant (or a commit is missing locally)"
fi
if [ "$DO_PLANT" = 1 ]; then
  [ "$(sha_of "$PLANT_GIT")" = "$PLANT_SHA" ] || stop "planted record sha256"
  git show "$PLANT_GIT" > "$W/plant.json"; [ "$(jq -r '.meta.window' "$W/plant.json")" = "full" ] || stop "plant record meta.window is not 'full'"
fi
if [ "$DO_PANEL" = 1 ]; then
[ "$(sha_of "$INPUTS_FULL_GIT")" = "$INPUTS_FULL_SHA" ] || stop "INPUTS_FULL sha256"
git show "$INPUTS_FULL_GIT" > "$W/inp.json"
# inputs_full/v1 (2e 10-16): INPUTS_FULL extends a frozen INPUTS_FINAL v2 through .base = {commit, path, sha256}; sets / extensions / label come from that base (pinned by
# sha here and again in run_pooled main()), the full-length exclusions pin from INPUTS_FULL itself. Any other schema is read as one flat file (as before).
if [ "$(jq -r '.schema // empty' "$W/inp.json")" = "inputs_full/v1" ]; then
  jq -e '.base | (.commit | type == "string" and length > 0) and (.path | type == "string" and length > 0) and (.sha256 | type == "string" and test("^[0-9a-f]{64}$"))' "$W/inp.json" > /dev/null \
    || stop "INPUTS_FULL (inputs_full/v1) .base is not a commit:path@sha256"
  IB="$(jq -r '.base.commit' "$W/inp.json"):$(jq -r '.base.path' "$W/inp.json")"
  [ "$(sha_of "$IB")" = "$(jq -r '.base.sha256' "$W/inp.json")" ] || stop "INPUTS_FULL's base $IB sha256 != its pin"
  git show "$IB" > "$W/inp_base.json"
else
  cp "$W/inp.json" "$W/inp_base.json"
fi

# 2. the record's own pointers and rules (keys only)
jq -e '(.npz.path | type == "string" and length > 0) and (.npz.sha256 | type == "string" and test("^[0-9a-f]{64}$"))' "$W/rec.json" > /dev/null || stop "record npz pointer missing or malformed"
NPZ_PATH=$(jq -r '.npz.path' "$W/rec.json"); NPZ_SHA=$(jq -r '.npz.sha256' "$W/rec.json")
jq -e --slurpfile t "$T_FULL" '.meta.cohort_rule == $t[0].final_checks.cohort_rule_exact and .meta.completion_rule == $t[0].final_checks.completion_rule_exact and .meta.waitlist_pending == []' "$W/rec.json" > /dev/null \
  || stop "record cohort rule / completion rule / empty waitlist do not match the template's final_checks"
jq -e '.meta.exclusions | (.commit | type == "string") and (.path | type == "string") and (.sha256 | type == "string")' "$W/rec.json" > /dev/null || stop "record meta.exclusions malformed"
EX_C=$(jq -r '.meta.exclusions.commit' "$W/rec.json"); EX_P=$(jq -r '.meta.exclusions.path' "$W/rec.json"); EX_S=$(jq -r '.meta.exclusions.sha256' "$W/rec.json")
[ "$(sha_of "$EX_C:$EX_P")" = "$EX_S" ] || stop "exclusions file sha256 != the record's meta.exclusions"
NPIN=$(jq --arg s "$EX_S" '[paths(type == "object" and .sha256? == $s)] | length' "$W/inp.json"); [ "$NPIN" = 1 ] || stop "INPUTS_FULL pins the exclusions file $NPIN times (need exactly one)"
PINF=$(jq -c --arg s "$EX_S" '[paths(type == "object" and .sha256? == $s)][0]' "$W/inp.json")
jq -e --slurpfile t "$T_FULL" '.sets.S_newmains.membership710 as $p | ($p.sha256 == $t[0].cohort_keys.membership710.sha256)
   and ($p.path | type == "string" and length > 0) and ($t[0].cohort_keys.membership710.git | split(":")[1] | endswith($p.path))' "$W/inp_base.json" > /dev/null \
  || stop "INPUTS_FULL's (base's) membership710 pin (sha256 and path) != the template's (run_pooled main() checks both)"
for p in '.extensions.S_paper493.cells' '.extensions.S_paper493.P_dup'; do jq -e "$p" "$W/inp_base.json" > /dev/null || stop "INPUTS_FULL (base) lacks $p"; done
REF=$(jq -r '.meta.cohort_list.ref // empty' "$W/rec.json"); [ -n "$REF" ] || stop "record has no meta.cohort_list.ref"
NF=$(blob_at_sha "experiments/2026-10-15_fpr_sweep_507_prep/${REF%@*}" "${REF#*@}") || stop "no revision of the no-filter list $REF has its sha"
git show "$EX_C:$EX_P" > "$W/excl.json"; git show "$NF" > "$W/nf.json"
fi

# 3. the planted record carries what the plant template needs (deltas, labels, the lead level)
if [ "$DO_PLANT" = 1 ]; then
jq -e --slurpfile t "$T_PR" '.meta.plant as $m
   | ($m.deltas | type == "array" and all(type == "number") and length == (unique | length) and ([.[] | select(. == 0)] | length) <= 1)
   and (([$m.deltas[] | select(. != 0)] | sort) == ($t[0].deltas | sort))
   and ($m.points | type == "array" and length > 0 and all(. == 0.01 or . == 0.05) and length == (unique | length) and any(. == 0.01))
   and (.rows as $r | [$t[0].probes[] as $p | ("C", "R") as $ro | "\($p) \($ro)"]
        | all(. as $l | ($r[$l] // {}) | type == "object" and ([.[] | objects | select(.plant | type == "object")] | length) > 0))' "$W/plant.json" > /dev/null \
  || stop "the planted record's meta.plant must state unique numeric deltas (the template's, plus at most one 0) and unique positive numeric levels incl. 0.01, and every probe x readout ROW must carry planted data (by key; meta.plant.labels is descriptive only)"
fi

# 4. without-S4 list (09 ruling (a), as for the 32k @ 56def977a), with the 32k asserts (only when the template pins a without-S4 origin)
if [ "$DO_PANEL" = 1 ] && [ "$NEED_WO" = true ]; then
jq -e '[.origins.S_newmains.headline_cells[] | test("^[ -~]+$")] | all' "$W/nf.json" > /dev/null || stop "a no-filter cell key is not printable ASCII (list sha convention)"
[ "$(jq '.origins.S_newmains as $o | [$o.headline_cells[] | select($o.batch_of_cell[.] == null)] | length' "$W/nf.json")" = 0 ] || stop "a no-filter S_newmains cell has no batch"
jq -c '.origins.S_newmains as $o | [$o.headline_cells[] | select($o.batch_of_cell[.] != "S4" and $o.batch_of_cell[.] != "S4r")] | sort' "$W/nf.json" > "$W/wo.json"
jq -e '[.rows | to_entries[] | .value | to_entries[] | select(.key | test("^S_newmains without S4")) | .value | select(.coverage != null)
        | (.cohort_cell_keys | type == "array" and all(type == "string") and (length == (unique | length)))] | (length > 0) and all' "$W/rec.json" > /dev/null \
  || stop "(i) a without-S4 row with coverage lacks a duplicate-free string-array cohort_cell_keys"
SETS=$(jq -c '[.rows | to_entries[] | .value | to_entries[] | select(.key | test("^S_newmains without S4")) | .value | select(.coverage != null) | (.cohort_cell_keys | sort)] | unique' "$W/rec.json")
[ "$(jq -n --argjson s "$SETS" '$s | length')" = 1 ] || stop "(i) the record's without-S4 rows do not share one key set"
[ "$(jq -n --argjson s "$SETS" --slurpfile w "$W/wo.json" '$s[0] == $w[0]')" = true ] || stop "record without-S4 set != the no-filter derivation"
[ "$(jq -n --slurpfile w "$W/wo.json" --slurpfile e "$W/excl.json" '($w[0] - $e[0].final_cohort_keys.S_newmains.cells) | length')" = 0 ] || stop "(ii) without-S4 not a subset of the S_newmains cohort"
[ "$(jq -n --slurpfile w "$W/wo.json" --slurpfile n "$W/nf.json" '$n[0].origins.S_newmains as $o | (($o.headline_cells - $w[0]) | sort) == ([$o.headline_cells[] | select($o.batch_of_cell[.] == "S4" or $o.batch_of_cell[.] == "S4r")] | sort)')" = true ] \
  || stop "(iv) removed cells are not exactly S4 + S4r"
WO_N=$(jq length "$W/wo.json"); WO_LSHA=$(jq -acj . "$W/wo.json" | shasum -a 256 | cut -c1-64)
jq -n --slurpfile w "$W/wo.json" --arg n "$WO_N" --arg s "$WO_LSHA" --arg src "$NF@${REF#*@}" --arg rec "$REC_GIT@$REC_SHA" \
  '{what: "S_newmains WITHOUT S4 no-filter cohort (full-length build); data only", definition: "without-S4 = headline_cells with batch ∉ {S4, S4r}", source: $src, cross_check: $rec,
    asserts: "PASS (launch_full.sh): (i) every without-S4 row with coverage has one duplicate-free key set, equal to the derivation; (ii) subset of the S_newmains cohort; (iv) the removed cells are exactly S4 + S4r",
    n: ($n | tonumber), cells: $w[0], sha256: $s}' > "$W/wo_file.json"
fi

# 5. render both configs in the temp dir (the without-S4 pin's commit is filled after its commit, below), and refuse leftovers
if [ "$DO_PANEL" = 1 ]; then
jq --arg rg "$REC_GIT" --arg rs "$REC_SHA" --arg np "$NPZ_PATH" --arg ns "$NPZ_SHA" --arg ig "$INPUTS_FULL_GIT" --arg is "$INPUTS_FULL_SHA" \
   --arg eg "$EX_C:$EX_P" --arg es "$EX_S" --argjson pf "$PINF" --arg nw "$NEED_WO" --arg pr "${PRODUCER_SHA:-}" '
  .pooled_label |= sub("TO_FILL"; "Stage A " + $rg)
  | (if has("pooled_world") then .pooled_world.producer_reviewed_sha = $pr else . end)
  | .npz[0].path = $np | .npz[0].sha256 = $ns | .npz[0].record = {git: $rg, sha256: $rs}
  | .inputs_final = {git: $ig, sha256: $is}
  | reduce ("S_old", "S_new", "S_newmains", "S_paper493") as $o (.; .cohort_keys.extra_origins[$o] = {git: $eg, sha256: $es, field: ["final_cohort_keys", $o, "cells"], list_sha256_field: ["final_cohort_keys", $o, "sha256"], pin_field: $pf})
  | (if $nw == "true" then .cohort_keys.extra_origins["S_newmains without S4"] = {git: "WO_GIT", sha256: "WO_SHA", field: ["cells"], list_sha256_field: ["sha256"]} else . end)
  | .cohort_pins_note["no-filter base cohorts"] = ("the record'"'"'s meta.exclusions " + $eg + " (sha256 " + $es + ") final_cohort_keys, pinned by INPUTS_FULL")
' "$T_FULL" > "$W/full.json"
! grep -q 'TO_FILL\|TO_CONFIRM\|"null"' "$W/full.json" || stop "full.json still has TO_FILL / TO_CONFIRM / null"
if [ "${PILOT:-0}" = 1 ]; then  # 2e 10-16: the end-to-end PILOT (a label subset): PANEL's real per-row gates, a gates-only record; scope taken from the record's own meta
  jq -e --slurpfile t "$W/full.json" '.meta.pooled_world as $p | ($p.labels | type == "array" and length > 0 and all(type == "string"))
     and ($p.combos | type == "array" and length > 0) and ($t[0].headline_combo | IN($p.combos[]))' "$W/rec.json" > /dev/null \
    || stop "the pilot record's meta.pooled_world must state labels and combos, the headline combo among them"
  jq --slurpfile r "$W/rec.json" '$r[0].meta.pooled_world as $p
    | .pooled_world.pilot = true | .pooled_world.pilot_delta_full_tol_abs = .coheadline.delta_full_tol_abs | .pooled_world.compare_per_set = false
    | .pooled_label = "PILOT (gates only, no values) " + .pooled_label
    | .npz[0].labels = $p.labels
    | .combos = (.combos | with_entries(select(.key | IN($p.combos[]))))
    | .dedup_combos = [.dedup_combos[] | select(IN($p.combos[]))]
    | .pooled_world.combos = [.pooled_world.combos[] | select(IN($p.combos[]))]
    | .required_rows = [.required_rows[] | .labels = [.labels[] | select(IN($p.labels[]))] | .combos = [.combos[] | select(IN($p.combos[]))]
                        | select((.labels | length) > 0 and (.combos | length) > 0)]
    | del(.coheadline, .panel_groups_gate, .labels_pin)' "$W/full.json" > "$W/full_pilot.json" && mv "$W/full_pilot.json" "$W/full.json"
  [ "$(jq -r '.pooled_world.pilot_delta_full_tol_abs | type' "$W/full.json")" = number ] || stop "the pilot's Delta_full tolerance did not come from the template"
fi
fi
if [ "$DO_PLANT" = 1 ]; then
jq --arg pg "$PLANT_GIT" --arg ps "$PLANT_SHA" --arg rg "$REC_GIT" --arg rs "$REC_SHA" '.record = {git: $pg, sha256: $ps} | .base_record = {git: $rg, sha256: $rs}' "$T_PR" > "$W/pr.json"
! grep -q 'TO_FILL\|TO_CONFIRM\|"null"' "$W/pr.json" || stop "pr.json still has TO_FILL / TO_CONFIRM / null"
fi

# 6. write, commit (explicit paths only) and push
OUT=()
if [ "$DO_PANEL" = 1 ] && [ "$NEED_WO" = true ]; then
  cp "$W/wo_file.json" "$WO_F"; git add -- "$WO_F"; git commit -q -m "Panel: full-length without-S4 list (${WO_N} cells), jq-derived and asserted by launch_full.sh; data only" -- "$WO_F"
  WO_C=$(git rev-parse --short=9 HEAD); WO_S=$(sha_of "$WO_C:$WO_F")
  jq --arg wg "$WO_C:$WO_F" --arg ws "$WO_S" '.cohort_keys.extra_origins["S_newmains without S4"].git = $wg | .cohort_keys.extra_origins["S_newmains without S4"].sha256 = $ws
    | .cohort_pins_note["no-filter base cohorts"] += ("; without-S4 @ " + $wg)' "$W/full.json" > "$FULL"
elif [ "$DO_PANEL" = 1 ]; then
  cp "$W/full.json" "$FULL"
fi
if [ "$DO_PANEL" = 1 ]; then
  ! grep -q 'WO_GIT\|WO_SHA' "$FULL" || stop "the without-S4 pin was not filled"
  OUT+=("$FULL")
fi
if [ "$DO_PLANT" = 1 ]; then cp "$W/pr.json" "$PR"; OUT+=("$PR"); fi
git add -- "${OUT[@]}"; git commit -q -m "Panel: full-length configs ($PART) filled for $STAMP by launch_full.sh (record $REC_GIT${PLANT_GIT:+, planted $PLANT_GIT}${INPUTS_FULL_GIT:+, INPUTS_FULL $INPUTS_FULL_GIT}); data only" -- "${OUT[@]}"
GIT_TERMINAL_PROMPT=0 git push -q origin "HEAD:$(git rev-parse --abbrev-ref HEAD)"
echo "filled: ${OUT[*]} @ $(git rev-parse --short=12 HEAD)"

# 7. the two consumers, in parallel and detached (SPEND rows are written in advance)
C1=("$MODAL" run --detach "$D/run_pooled.py" --cfg "$(basename "$FULL" .json)")
C2=("$MODAL" run --detach "$D/run_plant_rows.py" --cfg "$(basename "$PR" .json)")
if [ "$DO_PANEL" = 1 ]; then printf 'PYTHONUNBUFFERED=1 PYTHONPATH=src'; printf ' %q' "${C1[@]}"; echo; fi
if [ "$DO_PLANT" = 1 ]; then printf 'PYTHONUNBUFFERED=1 PYTHONPATH=src'; printf ' %q' "${C2[@]}"; echo; fi
if [ "${LAUNCH:-0}" = 1 ]; then
  L="$(git rev-parse --show-toplevel)/../full_launch_${STAMP}_${PART}"; mkdir -p "$L"
  set +e
  P1=""; P2=""; S1=0; S2=0
  if [ "$DO_PANEL" = 1 ]; then PYTHONUNBUFFERED=1 PYTHONPATH=src "${C1[@]}" > "$L/pooled.out" 2>&1 & P1=$!; fi
  if [ "$DO_PLANT" = 1 ]; then PYTHONUNBUFFERED=1 PYTHONPATH=src "${C2[@]}" > "$L/plant_rows.out" 2>&1 & P2=$!; fi
  echo "launched ($PART; pids ${P1:-none} ${P2:-none}); logs in $L"
  if [ -n "$P1" ]; then wait "$P1"; S1=$?; echo "EXIT $S1" >> "$L/pooled.out"; echo "run_pooled exit $S1"; fi
  if [ -n "$P2" ]; then wait "$P2"; S2=$?; echo "EXIT $S2" >> "$L/plant_rows.out"; echo "run_plant_rows exit $S2"; fi
  [ "$S1" = 0 ] && [ "$S2" = 0 ] || exit 1
fi
