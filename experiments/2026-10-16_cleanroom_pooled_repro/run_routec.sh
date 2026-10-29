#!/usr/bin/env bash
# Route C clean-room launch (a1, 2026-10-18). Launch steps run only from this committed text (0d).
#   [ROUTEC_PINS=config/routec_{B,R37,AS,BS,R37S,A18}_pins.json] CR_CFG=<pinned input folder> ./run_routec.sh membership   (default: A's config)
#   CR_CFG=<pinned input folder> ./run_routec.sh compute results/cleanroom_membership_<stamp>.json
# Before any Modal call, both stages check:
#   - the HEAD blobs of the reducer, tier 2, the gate and this script equal the reviewed shas, with no local edits;
#   - the config is committed and has no TBD;
#   - no live app of ours exists.
# membership: runs the keys-only stage, then the gate. The gate is gate_membership.jq @ e21ae0959, and the stage passes only if its
#   output is exactly ONE document with PASS true (R1's note). Prints the record path, its sha and the gate rc; exit = gate rc.
# compute / tier2 <record>: each re-runs that gate on the given membership record (rc 0 required, per 0d's pre-approval) and checks
#   the record's config sha, then launches ONE app (compute, or tier 2), so each launch gets its own refresh3 arm. Prints the record
#   path and sha.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
cd "$HERE"
: "${CR_CFG:?set CR_CFG}"
export ROUTEC_PINS="${ROUTEC_PINS:-config/routec_pins.json}"  # addendum 14: one config per record (A, B, R37); the reducer reads the same path
PINS=$ROUTEC_PINS
M=(uv run --no-project --python 3.12 --with modal==1.6.1 modal)
APP=probe-shift_aswin-surya_sub-repro_cleanroom_kill-2026-11-16
LOGS="${CR_LOG:-$CR_CFG/../logs}"
mkdir -p "$LOGS"
EXPECT="cleanroom_reduce_v2.py:991a34be9f290797 tier2_membership_v2.py:f1ec9b75735de55a gate_membership.jq:07f9cc84d6e48db6"  # bash 3.2: no maps
RED=${EXPECT#cleanroom_reduce_v2.py:}; RED=${RED%% *}  # the reviewed reducer blob, bound into the A-regression summary

for f in cleanroom_reduce_v2.py tier2_membership_v2.py gate_membership.jq "$PINS" run_routec.sh ../../src/probe_shift/q1_eval.py; do
  [ -z "$(git status --porcelain -- "$f")" ] || { echo "STOP: $f is not committed as is"; exit 1; }
done
for pair in $EXPECT; do
  f=${pair%%:*} want=${pair#*:}
  got=$(git show "HEAD:experiments/2026-10-16_cleanroom_pooled_repro/$f" | shasum -a 256 | cut -c1-16)
  [ "$got" = "$want" ] || { echo "STOP: $f blob $got != reviewed $want"; exit 1; }
done
jq -e 'del(.what) | tostring | test("TBD") | not' "$PINS" > /dev/null || { echo "STOP: config has TBD"; exit 1; }
live=$("${M[@]}" app list --json | jq -e --arg a "$APP" \
  'if type == "array" and all(.[]; has("state") and has("description"))
   then [.[] | select(.description == $a and .state != "stopped")] | length else error("app list schema") end') \
  || { echo "STOP: modal app list unreadable"; exit 1; }
[ "$live" = 0 ] || { echo "STOP: $live live app(s) named $APP"; exit 1; }
CFG_SHA=$(shasum -a 256 "$PINS" | cut -d' ' -f1)
NL=$(jq -e '((.sensitivity_labels // []) | length + 6) * 2' "$PINS")  # addendum 17: expected membership labels
echo "preflight ok: HEAD $(git rev-parse --short HEAD), config $PINS sha256 $(shasum -a 256 "$PINS" | cut -d' ' -f1)"

gate() {  # membership record, summary suffix -> rc 0 only for exactly one gate document with PASS true
  jq --argjson nl "$NL" -f gate_membership.jq "$1" | tee "${1%.json}.$2.json" | jq -e -s 'length == 1 and .[0].PASS == true' > /dev/null
}

run_one() {  # log name, then the modal run args; prints the "wrote" path
  local log="$LOGS/$1_$(date +%Y%m%d_%H%M%S).log"
  shift
  CR_CFG="$CR_CFG" "${M[@]}" run "$@" > "$log" 2>&1 || { echo "STOP: modal run failed, log $log" >&2; return 1; }
  local out
  out=$(sed -n 's/^wrote //p' "$log" | tail -1)
  [ -n "$out" ] && [ -f "$out" ] || { echo "STOP: no record written, log $log" >&2; return 1; }
  echo "$out"
}

case "${1:-}" in
  membership)
    rec=$(run_one membership cleanroom_reduce_v2.py --stage membership)
    echo "membership record $rec sha256 $(shasum -a 256 "$rec" | cut -d' ' -f1)"
    jq -n --arg h "$(git rev-parse HEAD)" --arg c "$CFG_SHA" '{head: $h, config_sha256: $c}' > "${rec%.json}.prov.json"  # binds the record to its config
    set +e
    gate "$rec" gate
    rc=$?
    set -e
    echo "gate rc $rc (summary ${rec%.json}.gate.json)"
    exit "$rc" ;;
  compute|tier2)  # one app per invocation: refresh3 arms exactly one new app id per launch (2e, 10-20)
    rec=${2:?give the membership record}
    if [ "$1" = compute ] && [ "$(jq -r .tag "$PINS")" != A ]; then  # addendum 15: B / R37 compute only after the A regression is EQUAL under this reducer
      jq -e --arg rb "$RED" '.EQUAL == true and .reducer_blob16 == $rb' "${REGRESS:?set REGRESS to the A-regression summary}" > /dev/null \
        || { echo "STOP: no EQUAL A-regression summary for reducer $RED"; exit 1; }
    fi
    [ "$1" = compute ] || [ "$(jq -r .tag "$PINS")" = A ] || { echo "STOP: tier 2 rebuilds the catalogue A, B and R37 share (addendum 14); run it on A's config"; exit 1; }
    [ "$(jq -r .config_sha256 "${rec%.json}.prov.json")" = "$CFG_SHA" ] || { echo "STOP: membership record was made under another config"; exit 1; }
    set +e
    gate "$rec" "gate_recheck_$(date +%Y%m%d_%H%M%S)"
    rc=$?
    set -e
    [ "$rc" = 0 ] || { echo "STOP: membership gate rc $rc"; exit 1; }
    echo "membership gate rc 0 on $rec (sha256 $(shasum -a 256 "$rec" | cut -d' ' -f1))"
    if [ "$1" = compute ]; then
      c=$(run_one compute cleanroom_reduce_v2.py --stage compute)
      jq -n --arg h "$(git rev-parse HEAD)" --arg rb "$RED" --arg c "$CFG_SHA" --arg p "$PINS" \
        '{head: $h, reducer_blob16: $rb, config_sha256: $c, config: $p}' > "${c%.json}.prov.json"  # binds the record to its code and config (R1)
      echo "compute record $c sha256 $(shasum -a 256 "$c" | cut -d' ' -f1)"
    else
      t=$(run_one tier2 tier2_membership_v2.py)
      echo "tier2 record $t sha256 $(shasum -a 256 "$t" | cut -d' ' -f1)"
    fi ;;
  regress)  # addendum 15 (09's condition): A's rerun under this reducer must equal A's committed compute record
    [ "$(jq -r .tag "$PINS")" = A ] || { echo "STOP: regress runs on A's config"; exit 1; }
    new=${2:?give the A rerun compute record}
    jq -e --arg rb "$RED" --arg cs "$CFG_SHA" '.reducer_blob16 == $rb and .config_sha256 == $cs' "${new%.json}.prov.json" > /dev/null \
      || { echo "STOP: the A rerun record was not made by this reducer under A's current config"; exit 1; }
    out="results/regress_A_$(date +%Y-%m-%d_%H%M%S).json"
    set +e
    jq -n --slurpfile old results/cleanroom_compute_2026-10-20_001845.json --slurpfile new "$new" -f regress_A.jq \
      | jq --arg rb "$RED" --arg ns "$(shasum -a 256 "$new" | cut -d' ' -f1)" --arg nr "$new" '. + {reducer_blob16: $rb, new_record: $nr, new_sha256: $ns}' > "$out"
    jq -e -s 'length == 1 and .[0].EQUAL == true' "$out" > /dev/null
    rc=$?
    set -e
    echo "regress rc $rc ($out)"
    exit "$rc" ;;
  *) echo "usage: run_routec.sh membership | compute <membership record> | tier2 <membership record> | regress <A rerun compute record>"; exit 2 ;;
esac
