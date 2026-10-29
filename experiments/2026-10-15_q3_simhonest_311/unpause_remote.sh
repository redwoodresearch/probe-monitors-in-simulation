#!/bin/bash
# unpause_remote.sh (runs ON the box, sent by unpause_one.sh through stdin): MODE lift | result.
#   bash -s -- lift <run dir> <harness> <session>      (prints LIFT_STARTED <ts> RENAMED, or the reason)
#   bash -s -- result <run dir> <ts>                    (prints the detached job's verdict)
# Codex r19c/r19d: a detached job owns the whole lift: final checks (pause ours alone: RL_PAUSE.txt, no other guard's
# marker, no harness-wide PAUSE_SIMS, both guards alive, ZERO pause evidence in both guards' logs, logs actually read), the
# rename, and rechecks at 3 s and 150 s that restore PAUSE_SIMS on any sign. A lost connection cannot strand a lift.
MODE=$1 RUN=$2
if [[ $MODE == result ]]; then
  f=$RUN/UNPAUSE_CHECK_$3.txt; [[ -r $f ]] && sed 's/^/RESULT /' "$f" || echo "NO result file"
  [[ -e $RUN/PAUSE_SIMS ]] && echo "NOW PAUSE_SIMS present" || echo "NOW PAUSE_SIMS absent"; exit 0
fi
[[ $MODE == lift ]] || { echo "NO such mode"; exit 2; }
HARN=$3 SES=$4; TS=$(date +%s)
[[ -d $RUN ]] || { echo "NO RUN DIR"; exit 1; }
REC=/tmp/unpause_job_${SES}_$TS.sh; OUT=$RUN/UNPAUSE_CHECK_$TS.txt
cat > "$REC" <<REC
#!/bin/bash
RUN=$RUN; HARN=$HARN; SES=$SES; OUT=$OUT; TS=$TS
SC=/home/aswin/phase2/s311_checks/sim_cap_stop_\$SES.log; GP=\$RUN/gate_pause_watch.log
guards() {
  local m
  for m in GATE_PAUSE.txt SIM_CAP_PAUSE.txt SIM_BLIND_PAUSE.txt SIM_HARD_STOP.txt; do [[ ! -e \$RUN/\$m ]] || { echo "\$m present"; return; }; done
  [[ ! -e \$HARN/results/PAUSE_SIMS ]] || { echo "harness-wide PAUSE_SIMS"; return; }
  pgrep -f "sim_cap_stop.py \$RUN \$SES" > /dev/null || { echo "sim_cap_stop dead"; return; }
  pgrep -f "gate_pause_311[a-z_]*.py \$RUN \$SES" > /dev/null || { echo "gate_pause dead"; return; }
  echo clean
}
count() {  # pause evidence in both guards' logs; UNREADABLE unless both are actually read
  local txt; txt=\$(cat "\$SC" "\$GP") || { echo UNREADABLE; return; }
  grep -cE 'SIM CAP PAUSE|GATE PAUSE|BLIND PAUSE|HARD STOP|\(PAUSED\)' <<< "\$txt" || true
}
check() {
  local g c; g=\$(guards); c=\$(count)
  if [[ \$g != clean || \$c != 0 || -e \$RUN/PAUSE_SIMS ]]; then touch "\$RUN/PAUSE_SIMS"; echo "RESTORED (\$1): \$g; pause evidence \$c" >> "\$OUT"; exit 1
  else echo "OK \$1" >> "\$OUT"; fi
}
cd "\$RUN" || exit 1
[[ -e PAUSE_SIMS ]] || { echo "FAILED: not paused" >> "\$OUT"; exit 1; }
[[ -e RL_PAUSE.txt ]] || { echo "FAILED: no RL_PAUSE.txt (not ours)" >> "\$OUT"; exit 1; }
g=\$(guards); c=\$(count)
[[ \$g == clean && \$c == 0 ]] || { echo "FAILED: \$g; pause evidence \$c" >> "\$OUT"; exit 1; }
mv -n PAUSE_SIMS "PAUSE_SIMS.lifted_\$TS" && [[ ! -e PAUSE_SIMS ]] || { echo "FAILED: rename" >> "\$OUT"; exit 1; }
echo "RENAMED \$(TZ=America/Los_Angeles date '+%F %T PT')" >> "\$OUT"
sleep 3; check 3s; sleep 150; check 150s
REC
setsid nohup bash "$REC" > /dev/null 2>&1 < /dev/null &
for i in $(seq 10); do [[ -e $OUT ]] && break; sleep 0.5; done
[[ -e $OUT ]] || { echo "NOT LIFTED: the job wrote nothing"; exit 1; }
head -1 "$OUT" | grep -q '^RENAMED' && echo "LIFT_STARTED $TS $(head -1 "$OUT")" || echo "NOT LIFTED: $(head -1 "$OUT")"
