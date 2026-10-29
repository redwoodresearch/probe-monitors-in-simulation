#!/usr/bin/env bash
# THE laptop step for the follow-up launches (RULE.md Amendment 7), ONE command: fu_servers_311.sh <i|iii>
#   1. for each quarter, on its box: the follow-up harness files go in (tasks_311_q<N>_fu.json, pilot_311_fu_*.yaml,
#      servers_311_fu.yaml, fu_launch.sh, LAUNCH_COMMIT_FU, harness_manifest_fu.tsv) and the follow-up freeze check passes;
#      end_main.py and gate1_311b.py go into ~/phase2/s311_checks;
#   2. all quarters' follow-up servers start at once from the laptop (launch_servers.py --server-config
#      servers_311_fu.yaml: detached, 9 h Modal timeout, a 240-minute wait for the first request, 20-minute idle exit after
#      it): 6 per quarter for variant (i), 3 for (iii); each in its own process group, waited for (URLs) together;
#   3. each box gets its URLs (~/phase2/s311_checks/fu_bases_q<N>.txt), its shutdown moved to the servers' end + 30 min,
#      and end_main.py started (tmux -L guards em311q<N>), which takes over with no laptop.
# Prints the app IDs, the servers' end and every box's shutdown. Variant (i) also needs box_guard per quarter (started
# from the laptop once its Codex verdict is AGREE); end_main waits for it until the last start.
set -uo pipefail
V=${1:?variant i or iii}; [[ $V == i || $V == iii ]] || { echo "variant i or iii" >&2; exit 1; }
SP=/private/tmp/claude-501/-Users-aswinsurya-probe-distribution-shift-from-sims/1faf3281-1269-46cf-883f-277091c9aa65/scratchpad
HERE=$(cd "$(dirname "$0")" && pwd); WD=$SP/scaleup311/watch; N=$([[ $V == i ]] && echo 6 || echo 3)
H123=$SP/scaleup311/harness/experiments/2026-10-07_phase2_attack_check; H4=$SP/scaleup311/harness_q4/experiments/2026-10-07_phase2_attack_check
BOX=([1]=aswin@i-002bbb22c185c0f07 [2]=aswin@i-0c02a38abcde32bfb [3]=aswin@i-0cc7555acb95c694b [4]=aswin@i-07bce857a818b6f69)
MAIN=([1]=$(cut -d' ' -f1 $WD/runs_q1q2.txt) [2]=$(cut -d' ' -f2 $WD/runs_q1q2.txt) [3]=$(cat $WD/run_q3.txt) [4]=$(cat $WD/run_q4.txt))
STAMP=$(TZ=America/Los_Angeles date +%Y-%m-%d_%H%M%S)
log() { echo "$(TZ=America/Los_Angeles date '+%F %T PT') $*"; }
hq() { [[ $1 == 4 ]] && echo "$H4" || echo "$H123"; }
dbx() { DEVBOX=$1 perl -e 'alarm shift; exec @ARGV' "$2" bash "$H123/devbox.sh" "$3"; }
for q in 1 2 3 4; do  # 1. files and the follow-up freeze check
  H=$(hq $q); b=${BOX[$q]}
  for f in config/tasks_311_q{1,2,3,4}_fu.json config/pilot_311_fu_q{1,2,3,4}_{i,iii}.yaml config/servers_311_fu.yaml fu_launch.sh LAUNCH_COMMIT_FU config/harness_manifest_fu.tsv; do  # every quarter's files: the manifest lists them all
    DEVBOX=$b perl -e 'alarm 120; exec @ARGV' bash "$H123/devbox.sh" rsync -a "$H/$f" "box:phase2/attack_check_311/$f" || { log "Q$q: copy of $f failed"; exit 1; }
  done
  DEVBOX=$b perl -e 'alarm 120; exec @ARGV' bash "$H123/devbox.sh" rsync -a "$HERE/end_main.py" "$HERE/gate1_311b.py" "$HERE/sim_cap_stop.py" "$HERE/spend_now.py" box:phase2/s311_checks/ || exit 1
  dbx $b 120 "cd ~/phase2/attack_check_311 && diff <(find . -type f ! -path './results/*' ! -path '*/__pycache__/*' ! -path './attempt/results/*' ! -path './config/harness_manifest.tsv' ! -path './config/harness_manifest_fu.tsv' | LC_ALL=C sort | xargs -d '\n' sha256sum) config/harness_manifest_fu.tsv >/dev/null" ||
    { log "Q$q: follow-up freeze check failed on the box"; exit 1; }
  log "Q$q: follow-up files in place, freeze PASS"
done
S=$(date +%s); END=$((S + 11 * 3600)); echo "$S $END" > $WD/fu_servers_start_end.txt
set -m
for q in 1 2 3 4; do  # 2. every quarter's servers at once, each launcher in its own process group
  (cd "$(hq $q)" && exec uv run --no-project --python 3.12 --with modal==1.2.6 --with pyyaml python launch_servers.py --count "$N" \
    --server-config servers_311_fu.yaml) > "$WD/fu_launch_servers_q$q.log" 2>&1 &
  echo $! > "$WD/fu_launch_servers_q$q.pid"
done
set +m
ok=1
for q in 1 2 3 4; do wait "$(cat $WD/fu_launch_servers_q$q.pid)" || { log "Q$q: launch_servers.py failed (it stops its own apps)"; ok=0; }; done
for q in 1 2 3 4; do  # 3. URLs, shutdown, end_main
  b=${BOX[$q]}
  APPS=$(grep "^app IDs" "$WD/fu_launch_servers_q$q.log" | sed 's/.*: //'); BASES=$(grep "^base URLs" "$WD/fu_launch_servers_q$q.log" | sed 's/.*: //')
  [[ $(tr ',' '\n' <<< "$BASES" | grep -c https) == "$N" ]] || { log "Q$q: expected $N URLs, got '$BASES'"; ok=0; continue; }
  echo "$APPS" > "$WD/fu_app_ids_q$q.txt"
  dbx $b 60 "echo '$BASES' > ~/phase2/s311_checks/fu_bases_q$q.txt && sudo shutdown -h +$(( (END + 1800 - $(date +%s)) / 60 ))" || { log "Q$q: URLs or shutdown failed"; ok=0; continue; }
  FU=/home/aswin/phase2/runs/${STAMP}_scaleup311_q${q}_fu
  dbx $b 60 "if tmux -L guards has-session -t '=em311q$q' 2>/dev/null; then echo 'em311q$q already exists: refusing'; exit 1; fi; tmux -L guards new -d -s em311q$q \"/home/aswin/sims/simulated-control-evals-sept/.venv/bin/python /home/aswin/phase2/s311_checks/end_main.py $q /home/aswin/phase2/runs/${MAIN[$q]} $FU config/pilot_311_fu_q${q}_$V.yaml $END $V\"; sleep 3; tmux -L guards has-session -t '=em311q$q'" ||
    { log "Q$q: end_main not running"; ok=0; continue; }
  echo "$FU" > "$WD/fu_run_q$q.txt"
  log "Q$q: $N servers $APPS; follow-up run $FU; servers end $(date -r $END '+%H:%M'); box shutdown $(date -r $((END + 1800)) '+%H:%M'); end_main em311q$q up"
done
(( ok )) && log "ALL QUARTERS READY (variant $V)" || log "SOME QUARTER FAILED: see above"
