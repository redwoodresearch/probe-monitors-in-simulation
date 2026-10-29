#!/usr/bin/env bash
# simh_rlr_box.sh (sim-honest REPLACEMENT launches; 09 @ 13be06982, 91485ce80, c5420f2f5, c10a14948): ON A HOST BOX, start
# one replacement launch (sim_honest@september only, a few cells, forked at 0 from the SAME pinned references as the runs
# they replace, via --references-from the parent's main). Derived from simhnm_box.sh (Codex r12-r16 AGREE) for both
# families: +311 cells on the 311 harness (as simh311_box.sh launches), nm360 cells on the nm harness; changed where
# marked "rlr:". Fails closed exactly as simhnm_box.sh, with the family's frozen manifests, task files and reference rule.
#   bash ~/phase2/simh_rlr_stage/simh_rlr_box.sh <launch> <6 base URLs> <branch commit> <host's previous run dir> <host label> \
#     <generation manifest sha256> [<env file>]
set -euo pipefail
trap '' PIPE  # a closed stdout (ssh gone) must not kill the wrapper: writes fail with EPIPE instead, and are ignored
G=$1 BASES=$2 COMMIT=$3 PREV=$4 HOST=$5 GSHA=$6 ENVF=${7:-config/env.aws.tpl}
# 0d via d9 ~16:41: queued launches pause at max(4 x expected, $100): $100 for the single-cell q2_rlr2, nm_s4d_rlr2, nm_s4r_rlr,
# q1_rlr2 (expected $20), $156 for q3_rlr2 (t298 + t310, expected $39); hard caps unchanged
case $G in  # rlr: per launch the family, the parent's main, cap, sim_cap_stop pause (2x expected), task/config sha256, the main's own task file
  q1_rlr) FAM=311; MAIN=/home/aswin/phase2/runs/2026-10-14_154401_scaleup311_q1; CAP=628; STOP=196; TSHA=490e6a1ed63d2ee06cfe15caec3dd2000e6395738a7e15ceadbb5ab0f86fa9e8; CSHA=4f3aaca3e4079a997eeeff0ee6a9fedbef68cf9c35ca497293a983c53a2c1671; LTF=tasks_311_q1_launch.json; LSHA=ee413a10eb2c448575fa34a2eafad25428f0947e8ac3450dfbc7e2c7152e4124 ;;
  q2_rlr) FAM=311; MAIN=/home/aswin/phase2/runs/2026-10-14_154401_scaleup311_q2; CAP=628; STOP=196; TSHA=7e9153a45ca28f3ec8f1446f544995249dc1d4f0135a4b0b5f3a93add3a2ba08; CSHA=51ee8a2fc3ae44f12fee48a3db9cd0c643caffa5c6a936b60a3964df62ff9b22; LTF=tasks_311_q2_launch.json; LSHA=eb72f9532e42953fb813d6b226b2eb815c9d161dc3f3e47c5062694de97eaf4d ;;
  q3_rlr) FAM=311; MAIN=/home/aswin/phase2/runs/2026-10-14_155838_scaleup311_q3; CAP=251; STOP=78; TSHA=a8bb0284acce4874ace10f6d6943c8cdd493d00b8abb785672a132574a3b6aa9; CSHA=f7ac5e984506d9167142ae6c614c334b4e4502e82ca38c903424ab44ebd668e2; LTF=tasks_311_q3_launch.json; LSHA=186633516c694d6087ad2454ee5f8f7cb854dd8bf319c1717682330673a916a9 ;;
  q4_rlr) FAM=311; MAIN=/home/aswin/phase2/runs/2026-10-14_160246_scaleup311_q4; CAP=377; STOP=118; TSHA=220b38e8f0c4feee4ca502b96b4ef61a34ee757242609f903fa5bd396cc81f13; CSHA=c66f5870edf96c0f4696f1fdf1f0b12e608ac7371e60d2ed577e52db5bd831fc; LTF=tasks_311_q4_launch.json; LSHA=033d4bebf7bb93c78cada607f698e9ce32e157a6d808adb735d24574d6fc8a2e ;;
  nm_s3_rlr) FAM=nm; MAIN=/home/aswin/phase2/nm/runs/2026-10-14_newmains_s3; CAP=251; STOP=78; TSHA=e1df030aa46852ef40e93e7556516919b5a1cfcb7462b12daa0823ad5ede880d; CSHA=b709a4b06bee5b2c33013afe2b68a957e74ab740b54f8ef1cd28a79452aa7581; LTF=tasks_nm_s3_launch.json; LSHA=83ec89869204591d5390eb95ba4c4f6663f9a9d8fc591c97939cec2a3cbac196 ;;
  nm_s4d_rlr) FAM=nm; MAIN=/home/aswin/phase2/nm/runs/2026-10-14_newmains_s4; CAP=126; STOP=40; TSHA=f107f912923cc7a4a8e7de39db9cd3c344a91f73cf6649d24d2bf3cd15b4afcb; CSHA=cfb7102a3bf6b875cb1bd17877045d6eff47f1a68e3ff40123ec1d628d5f807d; LTF=tasks_nm_s4_launch.json; LSHA=88ddad22c047ab2258010947d01cc40af769ac45106e57e15e5d5ba62cc5e9ab ;;
  nm_s2_rlr) FAM=nm; MAIN=/home/aswin/phase2/nm/runs/2026-10-14_newmains_s2; CAP=126; STOP=40; TSHA=51d50e5512b6f572c849a51e25eb91344afc0f0cf1909ce1e34376677456a061; CSHA=d7188dd87bc3954adef1c8f7524425b4619946e483df5baf18655005ad21b0d2; LTF=tasks_nm_s2_launch.json; LSHA=4e83a59f99c4d03872f904fb7c18e09a69be6a72b6590ad208821ae7cdab4d9f ;;
  q3_rlr2) FAM=311; MAIN=/home/aswin/phase2/runs/2026-10-14_155838_scaleup311_q3; CAP=251; STOP=156; TSHA=a8bb0284acce4874ace10f6d6943c8cdd493d00b8abb785672a132574a3b6aa9; CSHA=3aa6484089d8906b2b83e484b53b131e5b3c6ef1baab74a819b9f8683bce6e47; LTF=tasks_311_q3_launch.json; LSHA=186633516c694d6087ad2454ee5f8f7cb854dd8bf319c1717682330673a916a9 ;;
  q2_rlr2) FAM=311; MAIN=/home/aswin/phase2/runs/2026-10-14_154401_scaleup311_q2; CAP=126; STOP=100; TSHA=a341f290e625e56c688c3357e03d7f118a139d1096d8e6fc443e8208b1040de4; CSHA=198f632ca6c759f0c6f8ff81a9607a64ff5aeab28cd977f1bed856e35218d612; LTF=tasks_311_q2_launch.json; LSHA=eb72f9532e42953fb813d6b226b2eb815c9d161dc3f3e47c5062694de97eaf4d ;;
  nm_s4d_rlr2) FAM=nm; MAIN=/home/aswin/phase2/nm/runs/2026-10-14_newmains_s4; CAP=126; STOP=100; TSHA=830cd608924214e62f86ef9c2884b98cfd832e825a25a947ba1f9dbfcae593bc; CSHA=dd7ff5e31bc5cae0e2e77f236cced3c6527da12f80c00bfe291d7a51dd8e445a; LTF=tasks_nm_s4_launch.json; LSHA=88ddad22c047ab2258010947d01cc40af769ac45106e57e15e5d5ba62cc5e9ab ;;
  nm_s4r_rlr) FAM=nm; MAIN=/home/aswin/phase2/nm/runs/2026-10-14_newmains_s4r; CAP=126; STOP=100; TSHA=0da9a7ea34ba2dfeb2f983b5adbef2c3672a0ab9064665ad7ff4518b6fd660c7; CSHA=28bb7b97cbdfe28583f6c1cfbba15011a6ed6c26521dbeec1b02fa7a0d47a12e; LTF=tasks_nm_s4r_launch.json; LSHA=9b7e3f4d1a5b7735fd0ca1a6aee428d420c41f26f21b82cfd012b3e49127fe7f ;;
  q1_rlr2) FAM=311; MAIN=/home/aswin/phase2/runs/2026-10-14_154401_scaleup311_q1; CAP=126; STOP=100; TSHA=83205f6911b49ec4a25f1a048ec40fa42686aa3b31f98ff7bef67de4411ca628; CSHA=872c21407b5ac7623bba2447ca7ae7f171748278edb31072a6b58160b348ba0f; LTF=tasks_311_q1_launch.json; LSHA=ee413a10eb2c448575fa34a2eafad25428f0947e8ac3450dfbc7e2c7152e4124 ;;
  *) echo "launch $G: not a sim-honest replacement launch"; exit 2 ;;
esac
ST=/home/aswin/phase2/simh_rlr_stage  # rlr: one stage dir for both families
if [[ $FAM == 311 ]]; then SRC=/home/aswin/phase2/attack_check_311; APPR=/home/aswin/phase2/approaches; PYP="PYTHONPATH=/home/aswin/phase2/sims-sept-strict-off/src "
else SRC=/home/aswin/phase2/nm/attack_check_nm; APPR=/home/aswin/phase2/nm/approaches; PYP=""; fi  # rlr: each family's own launch line
CHECKS=/home/aswin/phase2/s311_checks  # the +311 sim-honest spend guard (socket phase2), as simh_q<k>
RLGATE_SHA=624e35e33f393146978a65b63050fbc8de8a8726a9135137c72f67d68a6cc795  # gate_pause_311_rl.py @ e46ba05d0
PY=/home/aswin/sims/simulated-control-evals-sept/.venv/bin/python
IMAGES=/home/aswin/phase2/attack_check_strict/results/devbox_setup_2026-10-08_163146/images.json
RUN=/home/aswin/phase2/runs/2026-10-15_simh_$G; SESSION=simh${G//_/}  # rlr: runs/ so the census glob covers it
LOG=/home/aswin/phase2/simh_$G.log; TMP=$(mktemp -d)
# rlr: the family's committed frozen harness manifests, staged in $ST; box4's s4r-armed one only for nm_s4d_rlr / nm_s4r_rlr on box4; nm_s4d_rlr2 only on box1 (fu_nm)
if [[ $FAM == 311 ]]; then FROZEN="harness_manifest_fu_q123.tsv:71fd27724a045d3df5d8dba9b6e63d5f86514377c025a11a57fd78f6bf1aa287
harness_manifest_fu_q2r.tsv:b8f813c9a1f4ff4371a981b3aea8a9406b7fa0143eda3212a8b053e8123a4992
harness_manifest_fu_q4.tsv:32ed72b3e3aa95b21a41ae0dfcf47cacb9efb6e760e9577536e49b57e898bd3d"
else FROZEN="harness_manifest_fu_nm.tsv:18fbbaa225bbb0f70890f1436c17d254d676495136f8bd2aa616cb626281b7ed
harness_manifest_fu_box4.tsv:363d5b65a03900495372b5160f28bdd96673e37b563a42dd46148a67e4eda45d
harness_manifest_fu_simsastra.tsv:deb5f193780be039ca55b948b2083ac458a0cdaa7231440bbdb8358db6247b83"; fi
[[ $G == nm_s4d_rlr2 && $HOST != box1 ]] && { echo "nm_s4d_rlr2 runs on box1 only (09 via d9 16:3x)"; exit 2; }
[[ $HOST == box4 && ( $G == nm_s4d_rlr || $G == nm_s4r_rlr ) ]] && FROZEN+=$'\nharness_manifest_fu_box4_s4r.tsv:543394b2f6c0d590bea597c83feac7d7f69550dfb5b0c040d83cbd78a4143cc1'
log() {  # never fails: a logging error must not change control flow
  local line; line="$(TZ=America/Los_Angeles date '+%F %T PT' 2>/dev/null) [simh rlr $G] $*"
  echo "$line" 2>/dev/null || true; echo "$line" >> "$LOG" 2>/dev/null || true
}
fail() { log "SIMH GATE FAIL: $*"; exit 1; }
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
# nm: the env file selects the simulator API key by its Secrets Manager label (aws-sm://...); never the key itself
[[ $ENVF == config/env.aws.tpl || $ENVF =~ ^/home/aswin/phase2/simh_nm_stage/env_[a-z0-9_]+\.aws\.tpl$ ]] || fail "env file $ENVF not allowed"
[[ $ENVF == /* ]] && { [[ -f $ENVF ]] || fail "env file $ENVF missing"; } || [[ -f $SRC/$ENVF ]] || fail "env file $SRC/$ENVF missing"
absent "$RUN"
absent "$SRC/results/PAUSE_SIMS"  # rlr: the family harness's own marker
! has_session phase2 "$SESSION" || fail "session $SESSION is live"
# (a) the host's previous launch has FINISHED, and no spend or pause guard of any launch is still running
[[ $PREV == /home/aswin/phase2/* && -r $PREV/launch.json ]] || fail "cannot read the previous launch $PREV/launch.json"
"$PY" -c "import json,sys; sys.exit(0 if json.load(open(sys.argv[1])).get('finished') else 1)" "$PREV/launch.json" \
  || fail "the previous launch $PREV has not finished"
log "host $HOST: previous launch $PREV finished"
rc=0; gl=$(pgrep -af 'sim_cap_stop\.py|gate_pause[a-z0-9_]*\.py') || rc=$?
(( rc == 1 )) && [[ -z $gl ]] || fail "a spend or pause guard is still running (pgrep rc $rc): ${gl:0:300}"
[[ $GSHA =~ ^[0-9a-f]{64}$ ]] || fail "generation manifest sha256 malformed"
[[ -r $MAIN/launch.json ]] || fail "the $G main is not on this host at $MAIN"
# (b) one launcher per settings dir: no run_pilot.py process at all on the box
no_launcher
log "no other run_pilot.py on the box"
# 1. the nm harness is still frozen: it must equal ONE of the three committed frozen nm manifests (each staged copy
# sha-pinned). Other launches' staged config files are excluded by name: pilot_il*, tasks_il*, ilrr*, pilot_simh_* and
# (rlr:) tasks_simh_* (all sim-honest task files).
(cd "$SRC" && set -o pipefail && find . -type f ! -path './results/*' ! -path '*/__pycache__/*' ! -path './attempt/results/*' \
  ! -path './config/harness_manifest*.tsv' ! -path './config/pilot_il*' ! -path './config/tasks_il*' ! -path './config/ilrr*' ! -path './config/pilot_simh_*' \
  ! -path './config/tasks_simh_*' -print0 | LC_ALL=C sort -z | xargs -0 sha256sum > "$TMP/now.tsv") || fail "hashing the harness failed"
[[ -s $TMP/now.tsv ]] || fail "empty hash list"
MATCH=""
while IFS=: read -r mf ms; do
  sha_ok "$ST/$mf" "$ms"
  grep -v '  ./config/harness_manifest' "$ST/$mf" > "$TMP/frozen.tsv" || fail "reading $ST/$mf failed"
  [[ -s $TMP/frozen.tsv ]] || fail "empty $mf"
  if cmp -s "$TMP/now.tsv" "$TMP/frozen.tsv"; then MATCH=$mf; fi
done <<< "$FROZEN"
[[ -n $MATCH ]] || fail "$SRC equals none of the committed frozen manifests allowed for $G on $HOST"
log "freeze: $SRC matches $MATCH ($(wc -l < "$TMP/now.tsv") files)"
sha_ok "$ST/gate_pause_311_rl.py" "$RLGATE_SHA"
# 2. nm: this launch's config and task file (staged, sha-pinned) into the harness config/; the main's own task file
sha_ok "$ST/pilot_simh_$G.yaml" "$CSHA"; sha_ok "$ST/tasks_simh_$G.json" "$TSHA"
cp "$ST/pilot_simh_$G.yaml" "$ST/tasks_simh_$G.json" "$SRC/config/"
sha_ok "$SRC/config/pilot_simh_$G.yaml" "$CSHA"; sha_ok "$SRC/config/tasks_simh_$G.json" "$TSHA"
LTASK=$TMP/$LTF
cp "$ST/$LTF" "$LTASK" || fail "staged $G main task file"
sha_ok "$LTASK" "$LSHA"
# the committed generation manifest (09 Append 25g item 4), staged: its sha256 is the one given; it is this group's; its
# 25g census is clean (every pinned cell's fork source is the unique counted attempt, source == copy sha256) and every
# pinned cell carries the rendered reference hash; and each pinned cell's sha256 equals the main's pinned reference now
GM=$(ls "$ST"/manifest_simh_${G}_*.json 2>/dev/null | head -2) || true
[[ -n $GM && $(wc -l <<< "$GM") == 1 ]] || fail "need exactly one staged generation manifest for $G"
sha_ok "$GM" "$GSHA"
"$PY" - "$GM" "$MAIN/launch.json" "$G" "$SRC/config/tasks_simh_$G.json" "$ST/$MATCH" <<'PY' || fail "generation manifest does not match the main's references or the live harness"
import json, sys
m, launch, g = json.load(open(sys.argv[1])), json.load(open(sys.argv[2])), sys.argv[3]
# 09 @ df46b9be1, on every host: the live harness (it equals $MATCH exactly, checked above) must be the generation
# manifest's PINNED baseline, by name and full sha256 (no fallback between baselines)
import hashlib as _h
_live = open(sys.argv[5], "rb").read()
assert m["harness"]["frozen_manifest"].rsplit("/", 1)[-1] == sys.argv[5].rsplit("/", 1)[-1], (m["harness"]["frozen_manifest"], sys.argv[5])
assert _h.sha256(_live).hexdigest() == m["harness"]["frozen_manifest_sha256"], "live baseline != the manifest's pinned baseline"
fam = "311" if g.startswith("q") else "nm"
assert m["launch"] == g and m["family"] == fam and m["world"] == ("simh_311" if fam == "311" else "simh_nm360") and m["references_from"] == sys.argv[2].rsplit("/", 1)[0], (m["launch"], m["references_from"])
assert m["selected_run_census"]["equal"] and not m["selected_run_census"]["violations"], m["selected_run_census"]
assert {x["task"] for x in m["replacements"]} == {c["task"] for c in m["cells"]}, "replacement rows != cells"
refs = launch["references"]
pinned = {c["task"]: c["eval_sha256"] for c in m["cells"] if c["status"] == "pinned"}
assert pinned and all(c.get("rendered_reference_sha256") for c in m["cells"] if c["status"] == "pinned"), "rendered hash missing"
assert pinned == {t: r["sha256"] for t, r in refs.items() if t in pinned}, "per-cell sha256 mismatch"
assert set(pinned) == set(refs) & {c["task"] for c in m["cells"]}, "pinned set differs from the main's references"
assert {c["task"] for c in m["cells"]} == {c["key"] for c in json.load(open(sys.argv[4]))["cells"]}, "manifest cells != task file cells"
# Codex r12: re-run 25g item 2 on the CURRENT main launch.json right before the launch: its sha256 equals the one the
# manifest froze (a finished main's launch.json never changes), every pinned cell's fork source is the UNIQUE counted
# attempt (not kind rerun, no redrawn_because), and source, copy and pin hash equal
import hashlib
from pathlib import Path
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
assert sha(sys.argv[2]) == m["main_launch_json_sha256"], "the main's launch.json changed since the manifest"
runs = launch["runs"]; runs = list(runs.values()) if isinstance(runs, dict) else runs
ha = {r["key"].split("/", 1)[1]: r for r in runs if r.get("arm") == "honest_a"}
for t, pin in pinned.items():
    ref, atts = refs[t], ha[t]["attempts"]
    counted = [i for i, a in enumerate(atts) if not a.get("redrawn_because") and a.get("kind") != "rerun"]
    src_dir = Path(ref["source"]).resolve().parent
    sel = [i for i, a in enumerate(atts) if a.get("log_dir") and Path(a["log_dir"]).resolve() == src_dir]
    listed = fam == "311" and t in m["selected_run_census"]["listed_25f_in_launch"]  # rlr: 25f-listed +311 cells (t389)
    if listed:
        assert len(sel) == 1 and atts[sel[0]].get("kind") == "rerun" and pin == next(c for c in m["cells"] if c["task"] == t)["ha25f_pin"], f"{t}: not the 25f pin"
    else:
        assert len(counted) == 1 and sel == counted, f"{t}: fork source is not the unique counted attempt"
    assert sha(ref["source"]) == sha(ref["copy"]) == pin, f"{t}: source, copy and pin differ"
print(f"generation manifest: {len(pinned)} pinned cells match the main's references; reference rule clean; re-checked now: launch.json frozen, fork source per the family rule, source = copy = pin")
PY
(cd "$SRC" && "$PY" - "$G" "$LTASK" <<'PY') || fail "config or task file"
import json, sys, yaml
g = sys.argv[1]
cfg = yaml.safe_load(open(f"config/pilot_simh_{g}.yaml"))
assert cfg["real_arms"] == [] and cfg["sim_arms"] == ["sim_honest"] and cfg["simulators"] == ["september"], cfg
assert cfg["tasks"] == f"config/tasks_simh_{g}.json" and cfg["spend"]["cap_usd"] == cfg["spend"]["worst_case_limit_usd"]
assert cfg["spend"]["worst_per_sim_run_usd"] == 106 and cfg["spend"]["prior_per_sim_run_usd"] == 21.13, cfg["spend"]
cells = json.load(open(cfg["tasks"]))["cells"]
main = {c["key"]: c for c in json.load(open(sys.argv[2]))["cells"] if "real_arms" not in c}
canon = lambda c: json.dumps(c, sort_keys=True)
assert cells and all(c["key"] in main and canon(main[c["key"]]) == canon(c) for c in cells), "cells are not the main's r1 entries"
print(f"config: sim_honest@september only; {len(cells)} cells, each the main's r1 entry; cap {cfg['spend']['cap_usd']}")
PY
# 3. settings equality (as the nm fu_launch.sh)
GS="results/gate_settings_simh_${G}_$(date +%s).json"
(cd "$SRC" && "$PY" gate_settings.py --reference-json config/sim_settings_reference_2026-10-14_125923.json --out "$GS" \
  --harness . > "$GS.log" 2>&1) || fail "settings gate exit"
"$PY" -c "import json,sys; sys.exit(0 if json.load(open(sys.argv[1]))['pass'] else 1)" "$SRC/$GS" || fail "settings gate"
log "settings equality PASS"
printf '{"launch": "simh_%s", "world": "simh_311 or simh_nm360 (see the manifest)", "arm": "sim_honest@september", "stratum": "replacement (rlr)", "host": "%s", "previous_launch": "%s", "references_from": "%s", "generation_manifest_sha256": "%s", "frozen_manifest": "%s", "env_file": "%s", "settings_dir": "/home/aswin/phase2/settings", "cap_usd": %s, "commit": "%s", "spec": "09 Append 25g @ e6ded1811"}\n' \
  "$G" "$HOST" "$PREV" "$MAIN" "$GSHA" "$MATCH" "$ENVF" "$CAP" "$COMMIT" > "$RUN.simh_meta.json"
# 4. the launcher (nm: the nm follow-up launch line: nm approaches store, no PYTHONPATH; on socket phase2 for the +311
# sim-honest guards), then the two guards (pause only); both must be alive 30 s later, or the launcher is interrupted and
# its exit CONFIRMED. From the launch on, errexit is off and every failure goes through abort().
OURS="--out $RUN( |\$)"  # this launch's processes (pgrep/grep -E): the exact run dir, not a prefix of another
gone() {  # 0 = confirmed gone; 1 = still there; 2 = could not tell (an inspection error is never read as gone)
  local list rc=0; list=$(sessions phase2) || return 2
  if in_list "$list" "$SESSION"; then return 1; fi
  pgrep -f -- "$OURS" > /dev/null || rc=$?
  case $rc in 0) return 1 ;; 1) return 0 ;; *) return 2 ;; esac
}
abort() {  # interrupt the launcher and CONFIRM it ended; escalate to kill-session; never report success unconfirmed
  log "ABORT: $1"
  local i j
  for i in 1 2; do
    tmux -L phase2 send-keys -t "=$SESSION:" C-c > /dev/null 2>&1 || true
    for j in $(seq 30); do gone; if [[ $? == 0 ]]; then log "launcher ended after Ctrl+C"; exit 1; fi; sleep 2; done
  done
  tmux -L phase2 kill-session -t "=$SESSION" > /dev/null 2>&1 || true
  for j in $(seq 15); do gone; if [[ $? == 0 ]]; then log "launcher session killed"; exit 1; fi; sleep 2; done
  log "LAUNCHER NOT CONFIRMED STOPPED after abort: $SESSION, $RUN (stop it by hand)"; exit 3
}
no_launcher
tmux -L phase2 new -d -s "$SESSION" "cd $SRC && PATH=$SRC/fixed_digest_lookup:\$PATH IMAGES_JSON=$IMAGES ${PYP}RR_OWNER_ID=aswin-surya rr-env run --env-file $ENVF -- $PY $SRC/run_pilot.py --pilot-config config/pilot_simh_$G.yaml --base-urls $BASES --commit $COMMIT --approaches $APPR --references-from $MAIN --out $RUN > $RUN.launcher.log 2>&1" \
  || fail "tmux new (launcher)"
set +e
log "launcher session $SESSION created"
tmux -L guards new -d -s "scsimh${G//_/}" "$PY $CHECKS/sim_cap_stop.py $RUN $SESSION $STOP" || abort "tmux new (sim_cap_stop) failed"
tmux -L guards new -d -s "gpsimh${G//_/}" "$PY $ST/gate_pause_311_rl.py $RUN $SESSION" || abort "tmux new (gate_pause_311_rl) failed"
log "guards started: scsimh${G//_/} (at $STOP), gpsimh${G//_/}"
sleep 30
list=$(sessions phase2) || abort "cannot list tmux -L phase2 sessions"
in_list "$list" "$SESSION" || abort "the launcher session is gone (see $RUN.launcher.log)"
pgrep -f "sim_cap_stop.py $RUN $SESSION" > /dev/null || abort "sim_cap_stop is not running (or pgrep failed)"
pgrep -f "gate_pause_311_rl.py $RUN $SESSION" > /dev/null || abort "gate_pause_311_rl is not running (or pgrep failed)"
rc=0; all=$(pgrep -af 'run_pilot\.py') || rc=$?
[[ $rc == 0 ]] || abort "pgrep finds no launcher process at all (rc $rc)"
others=""; ours=0
while IFS= read -r line; do  # pure bash over pgrep's lines: ours match the exact run dir, anything else is another launcher
  if [[ $line =~ --out\ $RUN(\ |$) ]]; then ours=$((ours + 1)); elif [[ -n $line ]]; then others+="$line"$'\n'; fi
done <<< "$all"
(( ours > 0 )) || abort "no process of this launch found"
[[ -z $others ]] || abort "another launcher is running: ${others:0:300}"
log "STARTED: $SESSION, run $RUN, cap $CAP, guards alive, no other launcher"
