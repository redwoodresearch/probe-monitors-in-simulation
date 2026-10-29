#!/usr/bin/env bash
# ON THE DEVBOX. pane_state.sh <%id:pid>: LIVE if that exact pane (id and pid) is among `tmux -L phase2 list-panes -a`,
# GONE if it is not, or if absence is established (tmux says "no server running", or the socket file does not exist),
# UNKNOWN on any other tmux error, e.g. a permission error (callers treat it as live).
# pane_state.sh --ctrlc <%id:pid>: send Ctrl+C only if LIVE.  pane_state.sh --of <session>: "<%id:pid> <start command>"
# of that exact session name, or nothing. (tmux 3.4's display-message exits 0 on a missing target: never used here.)
TMUX_BIN=${TMUX_BIN:-tmux}; SOCK=${TMUX_SOCKET:-/tmp/tmux-$(id -u)/phase2}  # overridable for the tests only
# The socket is provably missing only if its directory is searchable and the socket isn't there, or the directory itself
# is missing under a searchable parent; a stat that fails for lack of permission proves nothing (UNKNOWN).
sock_absent() {
  local d=${SOCK%/*}
  if [[ -d $d && -x $d ]]; then [[ ! -e $SOCK ]]
  else [[ ! -e $d && -d ${d%/*} && -x ${d%/*} ]] && ! ls -d "$d" > /dev/null 2>&1; fi
}
out=$("$TMUX_BIN" -L phase2 list-panes -a -F '#{session_name} #{pane_id}:#{pane_pid} #{pane_start_command}' 2>&1); rc=$?
if [[ $1 == --of ]]; then
  (( rc == 0 )) && printf '%s\n' "$out" | awk -v s="$2" '$1 == s { sub(/^[^ ]+ /, ""); print; exit }'
  exit 0
fi
want=${@: -1}
if (( rc == 0 )); then
  state=$(printf '%s\n' "$out" | awk '{ print $2 }' | grep -qxF -- "$want" && echo LIVE || echo GONE)
elif printf '%s' "$out" | grep -q 'no server running' || sock_absent; then  # absence established: tmux says no
  state=GONE                                                                  # server, or the socket provably missing
else
  state=UNKNOWN
fi
if [[ $1 == --ctrlc ]]; then
  [[ $state == LIVE ]] && "$TMUX_BIN" -L phase2 send-keys -t "${want%%:*}" C-c && echo SENT || echo "NOT SENT ($state)"
else
  echo "$state"
fi
