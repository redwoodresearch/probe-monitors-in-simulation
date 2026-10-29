#!/usr/bin/env bash
# ON BOX 4 (q3-scale-canary), RUN BY ASWIN (d9 about 23:25 PT: Aswin "just urn it now" about the S4 61-cell rerun; the
# classifier keeps the frozen-file edit with him). RULE.md Amendment 9. Fails closed; on any failure after the edit, every
# change is restored. Steps:
#  0. preconditions: prelaunch_gates.sh is the frozen file (sha256 8951059e...), both box-4 freezes pass now, no backup
#     yet, no 4r run folder, no s4rwait session;
#  1. back up prelaunch_gates.sh and both manifests OUTSIDE the tree (~/phase2/nm/frozen_backup/; a file inside the tree
#     would fail every freeze);
#  2. the one-token change at line 28, s[1-5] -> s([1-5]|4r), by an exact fixed-string replace; exactly one line differs;
#     the new file's sha256 must be b08e671d...;
#  3. both manifests regenerated as the gates compute them (follow-up form first, then main form); each must differ from
#     the backed-up box-4 manifest in exactly: ./prelaunch_gates.sh (both) and ./config/harness_manifest_fu.tsv (main);
#     then both freezes re-verified;
#  4. box 4's shutdown moved to 14:00 PT 10-15 (later only);
#  5. the 4r follow-up URLs (4 of box 4's +311 servers) and the on-box waiter s4rwait (s4r_wait.sh).
set -euo pipefail  # before the backups: any failure stops with nothing changed
C=/home/aswin/phase2/nm/attack_check_nm; CH=/home/aswin/phase2/nm/checks; B=/home/aswin/phase2/nm/frozen_backup
F=$C/prelaunch_gates.sh; OLD=8951059eadfe2b90f7691ba911224c8fb15871c5ea4ee469ce5ae539b9f0fb36
NEW=b08e671d4fe9f8827f311e9f9a166b6b61ece1ba52b8bdf2c59b79746aa0812e
BM=https://ta-01m4z2cv4x8xkk9gckknb3y3kr-8000-mtxq3h26oq0fs3shwzop71f5e.w.modal.host/v1,https://ta-01m4z2cxtspg3jt4nzkq5wkxer-8000-uqey9n10qrdhwnmrl84osomyw.w.modal.host/v1,https://ta-01m4z2d1eh4061vpkg7sh6y1er-8000-6fzz4rr5jij6n4xp12u852lwo.w.modal.host/v1,https://ta-01m4z2d44yf5bwwc784wxy344r-8000-pqr2k3cp5oo7wfq5jsdubq7p3.w.modal.host/v1,https://ta-01m4z2d6nm7g749ct8739tscsr-8000-hc5e8hiqgpdd0avoyvc21fclg.w.modal.host/v1,https://ta-01m4z2da5wht3y1bhwm8b74fsr-8000-mmtt3x0owx0lbnqfnwn6f5t6x.w.modal.host/v1
log() { echo "$(TZ=America/Los_Angeles date '+%F %T PT') [s4r arm] $*" | tee -a $CH/s4r_arm.log; }
die() { log "NOT ARMED: $*"; exit 1; }
# (Codex r16) generate to a file with its own status, then compare: a failing generator can never pass
fu_freeze() { (cd $C && find . -type f ! -path './results/*' ! -path '*/__pycache__/*' ! -path './attempt/results/*' ! -path './config/harness_manifest.tsv' ! -path './config/harness_manifest_fu.tsv' | LC_ALL=C sort | xargs -d '\n' sha256sum) > /tmp/s4r_ff.tsv || return 2
  cmp -s /tmp/s4r_ff.tsv $C/config/harness_manifest_fu.tsv; }
main_freeze() { (cd $C && find . -type f ! -path './results/*' ! -path '*/__pycache__/*' ! -path './config/harness_manifest.tsv' | LC_ALL=C sort | xargs -d '\n' sha256sum) > /tmp/s4r_fm.tsv || return 2
  cmp -s /tmp/s4r_fm.tsv $C/config/harness_manifest.tsv; }
# 0
[ "$(sha256sum $F | cut -c1-64)" = "$OLD" ] || die "prelaunch_gates.sh is not the frozen file"
fu_freeze && main_freeze || die "box-4 freezes fail before the edit"
[ ! -e $B/prelaunch_gates.sh.frozen_orig ] || die "a backup exists already"
[ ! -e /home/aswin/phase2/nm/runs/2026-10-14_newmains_s4r ] || die "4r run folder exists"
! tmux -L nmguards has-session -t '=s4rwait' 2>/dev/null || die "s4rwait exists"
[ "$(tr ',' '\n' <<< "$BM" | grep -c https)" = 6 ] || die "expected 6 server URLs"
# 1
mkdir -p $B; cp -p $F $B/prelaunch_gates.sh.frozen_orig
cp -p $C/config/harness_manifest.tsv $B/harness_manifest.tsv.box4_before; cp -p $C/config/harness_manifest_fu.tsv $B/harness_manifest_fu.tsv.box4_before
log "backups: $(sha256sum $B/prelaunch_gates.sh.frozen_orig | cut -c1-16) prelaunch_gates.sh.frozen_orig; manifests saved"
# From here on (Codex r13-r15): NO error trap and no set -e. Every command is checked explicitly in THIS shell and any
# failure calls restore(), which puts back the frozen file and both manifests (never through a subshell or a trap).
set +e
restore() {
  rm -f $CH/s4r_ARMED $CH/.s4r_ARMED.tmp; tmux -L nmguards kill-session -t '=s4rwait' 2>/dev/null; sleep 1
  w=ok; tmux -L nmguards has-session -t '=s4rwait' 2>/dev/null && w=STILL-LIVE; [ -e $CH/s4r_ARMED ] && w=FLAG-LEFT
  cp -p $B/prelaunch_gates.sh.frozen_orig $F; cp -p $B/harness_manifest.tsv.box4_before $C/config/harness_manifest.tsv
  cp -p $B/harness_manifest_fu.tsv.box4_before $C/config/harness_manifest_fu.tsv
  if cmp -s $B/prelaunch_gates.sh.frozen_orig $F && cmp -s $B/harness_manifest.tsv.box4_before $C/config/harness_manifest.tsv \
     && cmp -s $B/harness_manifest_fu.tsv.box4_before $C/config/harness_manifest_fu.tsv && [ $w = ok ]; then
    mv $B/prelaunch_gates.sh.frozen_orig $B/prelaunch_gates.sh.frozen_orig.restored
    m="RESTORED the frozen file and both manifests (verified byte-identical to the backups); waiter disarmed: $*"; rc=1
  else
    m="RESTORE FAILED (waiter $w): restore by hand from $B (prelaunch_gates.sh.frozen_orig, harness_manifest*.box4_before): $*"; rc=2
  fi
  echo "[s4r arm] $m"; echo "[s4r arm] $m" >> $CH/s4r_arm.log; exit $rc; }
# 2
python3 - $F <<'PY' || restore "edit failed"
import sys
p = sys.argv[1]; s = open(p).read()
old, new = "^config/tasks_nm_s[1-5]_launch\\.json$", "^config/tasks_nm_s([1-5]|4r)_launch\\.json$"
assert s.count(old) == 1, s.count(old)
open(p, "w").write(s.replace(old, new))
PY
diff $B/prelaunch_gates.sh.frozen_orig $F > /tmp/s4r_gate.diff; rc=$?; [ $rc = 1 ] || restore "diff exit $rc (expected 1: one difference)"
cat /tmp/s4r_gate.diff; cat /tmp/s4r_gate.diff >> $CH/s4r_arm.log || restore "log write failed"
[ "$(grep -c '^[<>]' /tmp/s4r_gate.diff)" = 2 ] && [ "$(head -1 /tmp/s4r_gate.diff)" = 28c28 ] || restore "diff is not exactly line 28"
nh=$(sha256sum $F | cut -c1-64) || restore "sha256sum failed"
[ "$nh" = "$NEW" ] || restore "new sha256 $nh is not $NEW"
log "edited: prelaunch_gates.sh $OLD -> $NEW" || restore "log write failed"
# 3
cd $C || restore "cd failed"
find . -type f ! -path './results/*' ! -path '*/__pycache__/*' ! -path './attempt/results/*' ! -path './config/harness_manifest.tsv' ! -path './config/harness_manifest_fu.tsv' | LC_ALL=C sort | xargs -d '\n' sha256sum > /tmp/s4r_fu.tsv || restore "fu manifest generation failed"
cp /tmp/s4r_fu.tsv config/harness_manifest_fu.tsv || restore "fu manifest install failed"
find . -type f ! -path './results/*' ! -path '*/__pycache__/*' ! -path './config/harness_manifest.tsv' | LC_ALL=C sort | xargs -d '\n' sha256sum > /tmp/s4r_main.tsv || restore "main manifest generation failed"
cp /tmp/s4r_main.tsv config/harness_manifest.tsv || restore "main manifest install failed"
diff $B/harness_manifest_fu.tsv.box4_before config/harness_manifest_fu.tsv > /tmp/s4r_fu.diff; rc=$?; [ $rc = 1 ] || restore "fu manifest diff exit $rc"
diff $B/harness_manifest.tsv.box4_before config/harness_manifest.tsv > /tmp/s4r_main.diff; rc=$?; [ $rc = 1 ] || restore "main manifest diff exit $rc"
grep '^[<>]' /tmp/s4r_fu.diff > /tmp/s4r_fu.lines || restore "fu manifest diff lines"
grep '^[<>]' /tmp/s4r_main.diff > /tmp/s4r_main.lines || restore "main manifest diff lines"
awk '{print $3}' /tmp/s4r_fu.lines | LC_ALL=C sort -u > /tmp/s4r_fu.paths || restore "fu changed paths"
awk '{print $3}' /tmp/s4r_main.lines | LC_ALL=C sort -u > /tmp/s4r_main.paths || restore "main changed paths"
cf=$(tr '\n' ' ' < /tmp/s4r_fu.paths) || restore "fu changed paths read"
cm=$(tr '\n' ' ' < /tmp/s4r_main.paths) || restore "main changed paths read"
log "manifest changes: fu [$cf] main [$cm]" || restore "log write failed"
[ "$cf" = "./prelaunch_gates.sh " ] || restore "fu manifest changed beyond prelaunch_gates.sh: $cf"
[ "$cm" = "./config/harness_manifest_fu.tsv ./prelaunch_gates.sh " ] || restore "main manifest changed beyond the expected two: $cm"
fu_freeze && main_freeze || restore "freezes fail after regeneration"
hm=$(sha256sum config/harness_manifest.tsv | cut -c1-16) && hf=$(sha256sum config/harness_manifest_fu.tsv | cut -c1-16) || restore "sha256sum failed"
log "manifests regenerated; both freezes pass: main $hm, fu $hf" || restore "log write failed"
# 4
f=/run/systemd/shutdown/scheduled; cur=0
if [ -e $f ]; then u=$(sed -n 's/^USEC=//p' $f) && [[ $u =~ ^[0-9]+$ ]] || restore "shutdown schedule unreadable"; cur=$((u / 1000000)); fi
now=$(date +%s) && [[ $now =~ ^[0-9]+$ ]] || restore "date failed"
if (( cur < 1792098000 )); then sudo shutdown -h +$(( (1792098000 - now) / 60 )) >/dev/null 2>&1 || restore "shutdown move failed"; fi
u=$(sed -n 's/^USEC=//p' $f) && [[ $u =~ ^[0-9]+$ ]] && (( u / 1000000 >= 1792098000 - 120 )) || restore "shutdown not at or after 14:00"
log "shutdown $cur -> $((u / 1000000)) (epoch s; want >= 1792098000)" || restore "log write failed"
# 5
cut -d, -f1-4 <<< "$BM" > $CH/fu_bases_s4r.txt || restore "fu_bases_s4r.txt"
[ "$(tr ',' '\n' < $CH/fu_bases_s4r.txt | grep -c https)" = 4 ] || restore "fu_bases_s4r.txt does not hold 4 URLs"
cat > $CH/s4r_wait.sh <<'WAIT' || restore "could not write s4r_wait.sh"
#!/usr/bin/env bash
# s4rwait (RULE.md Amendment 9): every 120 s until 05:00 PT, launch the S4 rerun MAIN (split 4r) through
# launch_nm_onbox_v3.sh ONLY when, read positively in the same pass: 4d has ended (its launch.json finished and no fnms4d
# session); box 4's +311 launch has ended (/home/aswin/phase2/runs/2026-10-14_il_q4/launch.json finished, no il311q4
# session on -L phase2); no run_pilot.py process on the box; every one of the 6 server URLs answers HTTP 200 or 401.
set -uo pipefail
BM=$1; CH=/home/aswin/phase2/nm/checks; L=$CH/s4r_wait.log
log() { echo "$(TZ=America/Los_Angeles date '+%F %T PT') [s4rwait] $*" >> $L; }
fin() { python3 -c 'import json,sys; sys.exit(0 if json.load(open(sys.argv[1])).get("finished") else 1)' "$1" 2>/dev/null; }
absent() { local o; o=$(tmux -L "$1" list-sessions -F '#{session_name}' 2>&1); local rc=$?
  if (( rc == 0 )); then grep -qxF -- "$2" <<< "$o"; [ $? = 1 ]; else grep -qE '^(no server running on |error connecting to .*\(No such file or directory\))' <<< "$o"; fi; }
while (( $(date +%s) < 1792065600 )); do
  why=""
  fin /home/aswin/phase2/nm/runs/2026-10-14_newmains_s4d_fu/launch.json && absent nm fnms4d || why="$why 4d-running"
  fin /home/aswin/phase2/runs/2026-10-14_il_q4/launch.json && absent phase2 il311q4 || why="$why il_q4-not-ended"
  pgrep -f 'run_pilot[.]py' >/dev/null; rc=$?; [[ $rc == 1 ]] || why="$why run_pilot-live-or-pgrep-error($rc)"  # only exit 1 = none
  for t in rpnms4d gsnms4d; do  # 4d's timers: retired only once 4d has ended; launch only on verified absence (Codex r14)
    if [[ $why != *4d-running* ]] && ! absent nmguards $t; then tmux -L nmguards kill-session -t "=$t" 2>/dev/null; log "4d's timer $t: kill requested"; fi
    absent nmguards $t || why="$why timer-$t-alive"; done
  for u in $(tr ',' ' ' <<< "$BM"); do c=$(curl -s -o /dev/null -m 15 -w '%{http_code}' "${u%/}/models"); [[ $c == 200 || $c == 401 ]] || why="$why server-$c"; done
  [ -e $CH/s4r_ARMED ] || why="$why not-armed-yet"  # activation barrier (Codex r16): written last by s4r_arm.sh
  if [ -z "$why" ]; then
    log "all conditions hold: launching 4r"
    K=4r RUN=2026-10-14_newmains_s4r BM=$BM N=6 END_MAIN=1792080000 END_FU=1792080000 SIGNAL=/home/aswin/phase2/nm/runs/2026-10-14_newmains_s4d_fu/launch.json DEADLINE=1792065600 bash $CH/launch_nm_onbox_v3.sh >> $L 2>&1
    log "launch_nm_onbox_v3.sh exit $?"; exit 0
  fi
  log "waiting:$why"; sleep 120
done
log "05:00 deadline: nothing started"; exit 1
WAIT
tmux -L nmguards new -d -s s4rwait "bash $CH/s4r_wait.sh $BM" || restore "s4rwait not created"
sleep 3; tmux -L nmguards has-session -t '=s4rwait' || restore "s4rwait not live"
# activation (Codex r17): the flag is published only by an atomic rename of a complete temp file; nothing after it can restore
date > $CH/.s4r_ARMED.tmp && [ -s $CH/.s4r_ARMED.tmp ] || restore "activation flag temp not written"
mv $CH/.s4r_ARMED.tmp $CH/s4r_ARMED || restore "activation flag rename failed"
log "ARMED: s4rwait (log $CH/s4r_wait.log); 4r main on box 4's 6 +311 servers; follow-up URLs $CH/fu_bases_s4r.txt (4); deadline 05:00 PT" || echo "ARMED (the log write failed)"
exit 0
