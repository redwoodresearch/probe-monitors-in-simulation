#!/usr/bin/env bash
# ilrr_box.sh (RULE.md "Paired reruns"; 09 Append 35 option B): ON an IL host, start ONE registered paired-rerun IL launch:
# per cell one real attack (run3) and one simulated attack (sim2), for exactly the cells whose ORIGINAL sim2 failed purely
# on provider rate limits (sim_ratelimit_failures_2026-10-15.json; every failed attempt's terminal error RateLimitError).
# Same frozen harness, simulator, references (--references-from as the original) and fork point; only the task file
# (those cells' original entries, seeded order) and the caps differ. Each cell is bound by record to its original pair
# before anything starts: the original sim2 run's .eval sha256 must equal rerun_of, and the original run3 and sim2 .eval
# shas are recorded next to the run for the supersession map. Derived from il311_box.sh v9 / il_nm_box.sh @ 2fc7f010
# (Codex-reviewed): helpers and the post-launch block verbatim; socket, guards and launcher command as parameters.
# rlr2 (make_rerun2_files.py from ilrr_box.sh @ 307fc2633f36): the s1/s2 reruns again, and q4's rerun, with caps that admit cells;
# mechanical renames only (run dir _rlr2, session ilrlr2, config pilot_ilrr2_, original-pairs and settings files). Do not edit.
#   bash ~/phase2/il_stage/ilrr2_box.sh <host> <base URLs, comma-separated> <branch commit>
set -euo pipefail
trap '' PIPE  # a closed stdout (ssh gone) must not kill the wrapper: writes fail with EPIPE instead, and are ignored
H=$1 BASES=$2 COMMIT=$3
PY=/home/aswin/sims/simulated-control-evals-sept/.venv/bin/python; TREE=/home/aswin/phase2/sims-sept-strict-off/src
IMAGES=/home/aswin/phase2/attack_check_strict/results/devbox_setup_2026-10-08_163146/images.json; ST=/home/aswin/phase2/il_stage
HF=$H  # file and run base: the original IL launch's host token; a placement override sets it (e.g. q4b1 -> q4)
case $H in
  s1) KIND=nm; MAIN=/home/aswin/phase2/nm/runs/2026-10-14_newmains_s1; PREVS="/home/aswin/phase2/nm/runs/2026-10-14_il_s1 /home/aswin/phase2/nm/runs/2026-10-14_il_s4d"; MANIFEST=/home/aswin/phase2/nm/attack_check_nm/config/harness_manifest_fu.tsv; CAP=269; TSHA=7b03f29dec3883650930e62946df94970f6c65563f7a3841b6afa968b407cef9; CSHA=e76c0d79e1257f42ddd003be26d398f1ce2176f3d5b160c1f383057a6329f35e; MSHA=12f54ee5421502b20584a24fcbe82eb370fcb25238287cbbb9f78ad239ce51a2; ILTASKS=config/tasks_il_s1.json; ILTSHA=ee3d865713f08d806116a48bec0c4c1b853f3e8e5dc54134c702e217d5e0fbf9; RSHA=; ILTASKS_STAGE=- ;;
  s2) KIND=nm; MAIN=/home/aswin/phase2/nm/runs/2026-10-14_newmains_s2; PREVS="/home/aswin/phase2/nm/runs/2026-10-14_il_s2 /home/aswin/phase2/nm/runs/2026-10-15_il_s2_rlr"; MANIFEST=/home/aswin/phase2/nm/attack_check_nm/config/harness_manifest_fu.tsv; CAP=179; TSHA=277356aca98f879516eecb3f6a95489b4541c35b42d6b2725f9a62ac2bb4f857; CSHA=a5b4a28b6374b3ea5108b1a363b9b3ffdddbdc2d943699d35848bb0b450eee12; MSHA=f452a3d0d9fa4356b34cd13b58820f5829e63a4b4bb609baab3db108e9acc224; ILTASKS=config/tasks_il_s2.json; ILTSHA=ab9e06493f6573c7777fa3c9d78d9647945323875da9a912e64e89d4a9c22741; RSHA=; ILTASKS_STAGE=- ;;
  q4b1) HF=q4; KIND=311; MAIN=/home/aswin/phase2/runs/2026-10-14_160246_scaleup311_q4; PREVS="/home/aswin/phase2/runs/2026-10-14_il_q4 /home/aswin/phase2/runs/2026-10-14_il_q1 /home/aswin/phase2/runs/2026-10-15_il_q1_rlr"; MANIFEST=/home/aswin/phase2/attack_check_311/config/harness_manifest_fu.tsv; CAP=536; TSHA=b33739bde3510d2d80c4727196117a80ef1b1e8376fe7f4166f24c05007780ae; CSHA=51665fc36a613117a018948ad0fae871dcac9f7fadff1cadd6d0602ac934fb2e; MSHA=61f54ab6f4f1bd1eb0ed81f6525759630dbdfa9d0d2d508ab8da28d2d2f00cdf; ILTASKS=config/tasks_il_q4_all.json; ILTSHA=f683cf5d28ba8427516cc144bd738f3f61ef634d2479830e12ce215a7251d505; RSHA_UNUSED=; RSHA=f12ed4cd86ed516036bcc47857c7a4d90ba95e21595a8c8f484653b9055a9ad6; ILTASKS_STAGE=staged ;;  # q4 placed on box 1: its main launch.json pinned
  *) echo "host $H: no rerun prepared"; exit 2 ;;
esac
if [[ $KIND == 311 ]]; then
  SRC=/home/aswin/phase2/attack_check_311; SOCK=phase2; GSOCK=guards; CHECKS=/home/aswin/phase2/s311_checks
  RUN=/home/aswin/phase2/runs/2026-10-15_il_${HF}_rlr2; GATEPAUSE=gate_pause_311.py; CAPSTOP=$((CAP + 100)); APPR=/home/aswin/phase2/approaches
  PYP="PYTHONPATH=$TREE "  # as il311_box.sh / fu_launch.sh
else
  SRC=/home/aswin/phase2/nm/attack_check_nm; SOCK=nm; GSOCK=nmguards; CHECKS=/home/aswin/phase2/nm/checks
  RUN=/home/aswin/phase2/nm/runs/2026-10-15_il_${HF}_rlr2; GATEPAUSE=gate_pause_nm.py; CAPSTOP="$((CAP + 100)) $((CAP + 400))"
  APPR=/home/aswin/phase2/nm/approaches; PYP=""  # as il_nm_box.sh / canary.sh
fi
SESSION=ilrlr2$HF; LOG=/home/aswin/phase2/il_${HF}_rlr2.log; TMP=$(mktemp -d)
LAUNCH_CMD="cd $SRC && PATH=$SRC/fixed_digest_lookup:\$PATH IMAGES_JSON=$IMAGES ${PYP}RR_OWNER_ID=aswin-surya rr-env run --env-file config/env.aws.tpl -- $PY $SRC/run_pilot.py --pilot-config config/pilot_ilrr2_$HF.yaml --base-urls $BASES --commit $COMMIT --approaches $APPR --references-from $MAIN --out $RUN > $RUN.launcher.log 2>&1"
log() {  # never fails: a logging error must not change control flow
  local line; line="$(TZ=America/Los_Angeles date '+%F %T PT' 2>/dev/null) [il ${HF}_rlr2 @ $H] $*"
  echo "$line" 2>/dev/null || true; echo "$line" >> "$LOG" 2>/dev/null || true
}
fail() { log "IL GATE FAIL: $*"; exit 1; }
sessions() {  # the session names on a tmux socket; no server = none; any other tmux error returns 1 (callers fail)
  local out rc=0; out=$(tmux -L "$1" ls -F '#S' 2>&1) || rc=$?
  # no server = none: "no server running", or (a socket never created on this box) exactly this one message
  if (( rc == 0 )); then printf '%s\n' "$out"
  elif [[ $out == *"no server running"* || $out == "error connecting to /tmp/tmux-$UID/$1 (No such file or directory)" ]]; then :
  else echo "tmux -L $1: $out" >&2; return 1; fi
}
has_session() {  # 0 = live, 1 = absent; a tmux or grep error exits the script through fail
  local list rc=0; list=$(sessions "$1") || fail "cannot list tmux -L $1 sessions"
  in_list "$list" "$2"
}
no_launcher() {  # fails unless NO run_pilot.py process runs on the box (pgrep's own status checked; it excludes itself)
  local live rc=0; live=$(pgrep -af 'run_pilot\.py') || rc=$?
  (( rc == 0 || rc == 1 )) || fail "pgrep failed ($rc)"
  (( rc == 1 )) && [[ -z $live ]] || fail "another launcher is running: $(grep -o -- '--out [^ ]*' <<< "$live" | sort -u | tr '\n' ' ')"
}
in_list() {  # 0 if line $2 is in the newline-separated list $1 (pure bash: no redirection or process that could fail)
  [[ $'\n'"$1"$'\n' == *$'\n'"$2"$'\n'* ]]
}
absent() {  # fails unless nothing exists at path $1 (a symlink counts, even dangling) and its parent is inspectable
  [[ -d ${1%/*} && -r ${1%/*} && -x ${1%/*} ]] || fail "cannot inspect ${1%/*}"
  [[ ! -e $1 && ! -L $1 ]] || fail "$1 exists"
}
sha_ok() {  # $1 file, $2 expected sha256
  local h; h=$(sha256sum < "$1") || fail "sha256sum $1"; [[ ${h:0:64} == "$2" ]] || fail "$1: sha256 ${h:0:16}... != expected ${2:0:16}..."
}
[[ $BASES =~ ^https://[^,]+(,https://[^,]+){5}$ ]] || fail "expected 6 base URLs"
absent "$RUN"
absent "$SRC/results/PAUSE_SIMS"
! has_session $SOCK "$SESSION" || fail "session $SESSION is live"
# (a) every earlier launch on this host has finished (the original IL launch and any later one in the host's order)
for p in $PREVS; do
  "$PY" -c "import json,sys; sys.exit(0 if json.load(open(sys.argv[1])).get('finished') else 1)" "$p/launch.json" \
    || fail "$p has not finished (or its launch.json can't be read)"
done
log "earlier launches finished: $PREVS"
[[ -z ${RSHA:-} ]] || sha_ok "$MAIN/launch.json" "$RSHA"
# (b) one launcher per settings dir: no run_pilot.py process at all on the box
no_launcher
log "no other run_pilot.py on the box"
# 1. the harness is still frozen (the staged il and ilrr files are excluded by name)
(cd "$SRC" && set -o pipefail && find . -type f ! -path './results/*' ! -path '*/__pycache__/*' ! -path './attempt/results/*' \
  ! -path './config/harness_manifest*.tsv' ! -path './config/pilot_il_*' ! -path './config/tasks_il_*' \
  ! -path './config/pilot_ilrr_*' ! -path './config/pilot_ilrr2_*' ! -path './config/pilot_ilrr3_*' ! -path './config/tasks_ilrr3_*' ! -path './config/ilrr3_*' ! -path './config/pilot_ilc_*' ! -path './config/tasks_ilc_*' ! -path './config/ilc_*' ! -path './config/tasks_ilrr_*' ! -path './config/ilrr_*' -print0 \
  | LC_ALL=C sort -z | xargs -0 sha256sum > "$TMP/now.tsv") || fail "hashing the harness failed"
grep -v '  ./config/harness_manifest' "$MANIFEST" > "$TMP/frozen.tsv" || fail "reading $MANIFEST failed"
[[ -s $TMP/now.tsv && -s $TMP/frozen.tsv ]] || fail "empty hash list"
cmp -s "$TMP/now.tsv" "$TMP/frozen.tsv" || fail "$SRC differs from $MANIFEST"
log "freeze: $SRC matches $MANIFEST ($(wc -l < "$TMP/now.tsv") files)"
# 2. this rerun's three files
cp "$ST/pilot_ilrr2_$HF.yaml" "$ST/tasks_ilrr_$HF.json" "$ST/ilrr_rerun_of_$HF.json" "$SRC/config/"
[[ $ILTASKS_STAGE == - ]] || cp "$ST/${ILTASKS#config/}" "$SRC/$ILTASKS"  # a moved rerun: the original IL task file, staged
sha_ok "$SRC/config/tasks_ilrr_$HF.json" "$TSHA"
sha_ok "$SRC/config/pilot_ilrr2_$HF.yaml" "$CSHA"
sha_ok "$SRC/config/ilrr_rerun_of_$HF.json" "$MSHA"
sha_ok "$SRC/$ILTASKS" "$ILTSHA"
# 3. the rerun cells: verbatim original entries of this host's IL task file, each bound by record to its original pair
(cd "$SRC" && "$PY" - "$HF" "$ILTASKS" "$PREVS" <<'PY') || fail "config, task file or rerun_of binding"
import hashlib, json, sys, yaml
from pathlib import Path
h, iltasks, prevs = sys.argv[1], sys.argv[2], sys.argv[3].split()
cfg = yaml.safe_load(open(f"config/pilot_ilrr2_{h}.yaml"))
assert cfg["real_arms"] == ["attack"] and cfg["sim_arms"] == ["sim_attack"] and cfg["simulators"] == ["september"], cfg
assert cfg["tasks"] == f"config/tasks_ilrr_{h}.json" and cfg["spend"]["cap_usd"] == cfg["spend"]["worst_case_limit_usd"]
cells = json.load(open(cfg["tasks"]))["cells"]
orig = json.load(open(iltasks))["cells"]
canon = lambda c: json.dumps(c, sort_keys=True)
assert cells and set(map(canon, cells)) <= set(map(canon, orig)) and len({c["key"] for c in cells}) == len(cells), "cells are not distinct original IL entries"
rr = json.load(open(f"config/ilrr_rerun_of_{h}.json"))
assert rr["host"] == h and rr["rerun_reason"] == "infra: RateLimitError", rr.get("host")
binds = {b["cell"]: b for b in rr["rerun_of"]}
assert set(binds) == {c["key"] for c in cells}, "rerun_of does not cover exactly the rerun cells"
original = prevs[0]
lj = json.load(open(Path(original) / "launch.json"))
runs = lj["runs"]; runs = list(runs.values()) if isinstance(runs, dict) else runs
by_key = {r["key"]: r for r in runs}
for cell, b in binds.items():
    assert b["launch_run_dir"] == original and b["run_key"] == f"sim_attack@september/{cell}", b
    assert b["pure_rate_limit"] is True, (cell, "not a pure rate-limit failure")
    r = by_key.get(b["run_key"]); assert r is not None and not r["real"], (cell, "no simulated run with that key in the original")
    assert r["state"] in ("done", "no policy call"), (cell, r["state"])
    ev = Path(b["eval_path"]); assert str(ev).startswith(original + "/") and ev.is_file(), (cell, "the failed .eval is missing")
    assert hashlib.sha256(ev.read_bytes()).hexdigest() == b["eval_sha256"], (cell, "the failed .eval's sha256 differs from rerun_of")
def evals(run_key):  # the original run's last attempt's .eval files: path and sha256 (the supersession map's "original")
    r = by_key[run_key]; d = Path(r["attempts"][-1]["log_dir"]) if r.get("attempts") else None
    fs = sorted(d.glob("*.eval")) if d and d.is_dir() else []
    return [{"path": str(f), "sha256": hashlib.sha256(f.read_bytes()).hexdigest()} for f in fs]
orig_pairs = {}
for cell in binds:
    assert f"attack/{cell}" in by_key and by_key[f"attack/{cell}"]["real"], (cell, "no original run3 (real attack) in the original launch")
    orig_pairs[cell] = {"run3": {"run_key": f"attack/{cell}", "state": by_key[f"attack/{cell}"]["state"], "evals": evals(f"attack/{cell}")},
                        "sim2": {"run_key": f"sim_attack@september/{cell}", "state": by_key[f"sim_attack@september/{cell}"]["state"],
                                 "evals": evals(f"sim_attack@september/{cell}")}}
json.dump({"original_launch": original, "pairs": orig_pairs}, open(f"results/ilrr2_original_pairs_{h}.json", "w"), indent=1)
print(f"paired rerun: {len(cells)} cells, all original entries, each bound to its failed original in {original}; cap {cfg['spend']['cap_usd']}")
PY
# 4. settings equality (as fu_launch.sh)
GS="results/gate_settings_ilrr2_${HF}_$(date +%s).json"
(cd "$SRC" && "$PY" gate_settings.py --reference-json config/sim_settings_reference_2026-10-14_125923.json --out "$GS" \
  --harness . > "$GS.log" 2>&1) || fail "settings gate exit"
"$PY" -c "import json,sys; sys.exit(0 if json.load(open(sys.argv[1]))['pass'] else 1)" "$SRC/$GS" || fail "settings gate"
log "settings equality PASS"
printf '{"launch": "il_%s_rlr2", "interleaved_launch": true, "paired_rerun": true, "rerun_reason": "infra: RateLimitError", "rerun_of_file_sha256": "%s", "original_launch": "%s", "worlds": {"attack": "run3", "sim_attack@september": "sim2"}, "references_from": "%s", "settings_dir": "/home/aswin/phase2/settings", "harness": "%s", "sim_attempt_timeout_s": 300, "cap_usd": %s, "commit": "%s"}\n' \
  "$HF" "$MSHA" "${PREVS%% *}" "$MAIN" "$KIND" "$CAP" "$COMMIT" > "$RUN.il_meta.json"
cp "$SRC/config/ilrr_rerun_of_$HF.json" "$RUN.rerun_of.json"
cp "$SRC/results/ilrr2_original_pairs_$HF.json" "$RUN.original_pairs.json"
# 5. the launcher, then the two reviewed guards (pause only); both alive 30 s later, or the launcher is interrupted and
# its exit CONFIRMED. From the launch on, errexit is off and every failure goes through abort().
OURS="--out $RUN( |\$)"  # this launch's processes (pgrep/grep -E): the exact run dir, not a prefix of another
gone() {  # 0 = confirmed gone; 1 = still there; 2 = could not tell (an inspection error is never read as gone)
  local list rc=0; list=$(sessions $SOCK) || return 2
  if in_list "$list" "$SESSION"; then return 1; fi
  pgrep -f -- "$OURS" > /dev/null || rc=$?
  case $rc in 0) return 1 ;; 1) return 0 ;; *) return 2 ;; esac
}
abort() {  # interrupt the launcher and CONFIRM it ended; escalate to kill-session; never report success unconfirmed
  log "ABORT: $1"
  local i j
  for i in 1 2; do
    tmux -L $SOCK send-keys -t "=$SESSION:" C-c > /dev/null 2>&1 || true
    for j in $(seq 30); do gone; if [[ $? == 0 ]]; then log "launcher ended after Ctrl+C"; exit 1; fi; sleep 2; done
  done
  tmux -L $SOCK kill-session -t "=$SESSION" > /dev/null 2>&1 || true
  for j in $(seq 15); do gone; if [[ $? == 0 ]]; then log "launcher session killed"; exit 1; fi; sleep 2; done
  log "LAUNCHER NOT CONFIRMED STOPPED after abort: $SESSION, $RUN (stop it by hand)"; exit 3
}
no_launcher
tmux -L $SOCK new -d -s "$SESSION" "$LAUNCH_CMD" \
  || fail "tmux new (launcher)"
set +e
log "launcher session $SESSION created"
tmux -L $GSOCK new -d -s "sc$SESSION" "$PY $CHECKS/sim_cap_stop.py $RUN $SESSION $CAPSTOP" || abort "tmux new (sim_cap_stop) failed"
tmux -L $GSOCK new -d -s "gp$SESSION" "$PY $CHECKS/$GATEPAUSE $RUN $SESSION" || abort "tmux new ($GATEPAUSE) failed"
log "guards started: sc$SESSION (sim_cap_stop $CAPSTOP), gp$SESSION ($GATEPAUSE) on tmux -L $GSOCK"
sleep 30
list=$(sessions $SOCK) || abort "cannot list tmux -L $SOCK sessions"
in_list "$list" "$SESSION" || abort "the launcher session is gone (see $RUN.launcher.log)"
pgrep -f "sim_cap_stop.py $RUN $SESSION" > /dev/null || abort "sim_cap_stop is not running (or pgrep failed)"
pgrep -f "$GATEPAUSE $RUN $SESSION" > /dev/null || abort "$GATEPAUSE is not running (or pgrep failed)"
rc=0; all=$(pgrep -af 'run_pilot\.py') || rc=$?
[[ $rc == 0 ]] || abort "pgrep finds no launcher process at all (rc $rc)"
others=""; ours=0
while IFS= read -r line; do  # pure bash over pgrep's lines: ours match the exact run dir, anything else is another launcher
  if [[ $line =~ --out\ $RUN(\ |$) ]]; then ours=$((ours + 1)); elif [[ -n $line ]]; then others+="$line"$'\n'; fi
done <<< "$all"
(( ours > 0 )) || abort "no process of this launch found"
[[ -z $others ]] || abort "another launcher is running: ${others:0:300}"
log "STARTED: $SESSION, run $RUN, cap $CAP, guards alive, no other launcher"
