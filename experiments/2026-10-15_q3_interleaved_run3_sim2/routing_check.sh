#!/usr/bin/env bash
# Read-only: does anything still route to a server set's hosts? On every box: processes whose args name a host, and
# unfinished launches (launch.json without "finished") whose launch.json names a host. Hosts file = one hostname per line.
#   routing_check.sh <hosts file>   -> one line per box: "<box> procs=<n> unfinished_launches_using=<n> [paths]" or UNREADABLE
SP=/private/tmp/claude-501/-Users-aswinsurya-probe-distribution-shift-from-sims/1faf3281-1269-46cf-883f-277091c9aa65/scratchpad; H=$SP/simh_launch_X/h
HOSTS=$(tr '\n' ' ' < "$1"); for h in $HOSTS; do [[ $h =~ ^[A-Za-z0-9.-]+$ ]] || { echo "bad host"; exit 1; }; done
CMD="P=\$(ps -eo args= 2>/dev/null | grep -vF unfinished_launches_using); n=0; for h in $HOSTS; do n=\$((n + \$(grep -cF \"\$h\" <<< \"\$P\"))); done; u=''; for lj in \$(ls /home/aswin/phase2/runs/*/launch.json /home/aswin/phase2/nm/runs/*/launch.json 2>/dev/null); do grep -q '\"finished\": null' \$lj || continue; for h in $HOSTS; do grep -qF \"\$h\" \$lj && { u=\"\$u \$lj\"; break; }; done; done; echo \"RESULT procs=\$n unfinished_launches_using=\$(wc -w <<< \"\$u\" | tr -d ' ')\$u\""
for hm in box1:i-002bbb22c185c0f07 box2:i-0c02a38abcde32bfb box3:i-0cc7555acb95c694b box4:i-07bce857a818b6f69 xbox1:i-04331f272d6778df0 xbox2:i-00db08777bb29d430 xbox3:i-0f2299f29dd296259 sims-astra:i-05b1d910fd70016e6; do b=${hm%%:*}; i=${hm##*:}
  ( r=$(cd "$H" && AWS_PROFILE=redwood-sandbox DEVBOX=aswin@$i perl -e 'alarm 90; exec @ARGV' bash experiments/2026-10-07_phase2_attack_check/devbox.sh "$CMD" 2>/dev/null | grep '^RESULT' | tail -1); echo "$b ${r:-UNREADABLE}" ) & done; wait
