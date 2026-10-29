# Launch definitions for the PLANT launch line; sourced by the line after a blob check. Definitions only, no side effects; they need M (the
# named modal binary), S (the script), LD (the fresh log folder) and APP (the app-name prefix), which the line sets before using them.
# 2e / 507-prep / refresh2 10-17: D1-D4 as for the header scan and item (c); the PLANT token checks.
# App states in modal 1.6.1 (.../site-packages/modal/cli/app.py L41-49, plus the "unknown" fallback at L138): deployed, "ephemeral (detached)",
# disabled, ephemeral, "initializing...", stopped, "stopping...". `app list --json` rows come from display_table (modal/cli/utils.py
# L188-191): the column headers "App ID", "Description" and "State" become the snake_case keys app_id, description and state.
# Rules: an empty list, a non-array, or a row without string app_id / description / state is a FAILURE, never a pass (D1 / R2).
# Only "stopped" counts as not live. Each `app list` call is capped at 20 s (perl alarm; macOS has no timeout), and each poll has a
# total SECONDS deadline of 60 s (D3); a hung call or a passed deadline STOPs.
applist() {  # one capped call; prints the JSON
  /usr/bin/perl -e 'alarm shift; exec @ARGV' 20 "$M" app list --json < /dev/null
}
poll_stopped() {  # pre-run, $1 = app-name prefix: no row with this prefix is live. 0 = none live; 1 = live at the deadline; 2 = a CLI / jq / schema failure.
  DL=$((SECONDS + 60))
  while [ "$SECONDS" -lt "$DL" ]; do
    AL=$(applist) || return 2
    NL=$( set -o pipefail; printf '%s' "$AL" | jq -e --arg n "$1" 'if type == "array" and length > 0 and all(.[]; type == "object" and (.app_id | type) == "string" and (.description | type) == "string" and (.state | type) == "string") then ([.[] | select((.description | startswith($n)) and .state != "stopped")] | length) else error("app list empty, not an array, or a row without app_id / description / state strings") end' ) || return 2
    [ "$NL" = 0 ] && return 0
    sleep 2.5
  done
  return 1
}
poll_app_stopped() {  # post-run, $1 = this run's app id (D2 / R1): EXACTLY ONE row has .app_id == $1 and its state is "stopped". 0 = so; 1 = not by the deadline; 2 = a CLI / jq / schema failure.
  DL=$((SECONDS + 60))
  SA=unread
  while [ "$SECONDS" -lt "$DL" ]; do
    AL=$(applist) || return 2
    SA=$( set -o pipefail; printf '%s' "$AL" | jq -e -r --arg a "$1" 'if type == "array" and length > 0 and all(.[]; type == "object" and (.app_id | type) == "string" and (.description | type) == "string" and (.state | type) == "string") then ([.[] | select(.app_id == $a)] | if length == 1 then .[0].state else "listed \(length) times" end) else error("app list empty, not an array, or a row without app_id / description / state strings") end' ) || return 2
    [ "$SA" = stopped ] && return 0
    sleep 2.5
  done
  echo "app $1 at the deadline: $SA"
  return 1
}
app_id_of() {  # $1 = log. The app id from modal run's own banner line "Initialized. View run at <url>" (modal/runner.py L444; one line under
  # COLUMNS=1000), which must appear EXACTLY once (D4); every modal.com/apps URL in the log must carry that same id.
  NB=$(grep -c -E 'Initialized\. View run at https://modal\.com/apps/redwood-default/main/ap-[A-Za-z0-9]+' "$1"; true)
  [ "$NB" = 1 ] || return 1
  AID=$( set -o pipefail; grep -o -E 'Initialized\. View run at https://modal\.com/apps/redwood-default/main/ap-[A-Za-z0-9]+' "$1" | sed 's#.*/##' ) || return 1
  NI=$( set -o pipefail; grep -o -E 'https://modal\.com/apps/redwood-default/main/ap-[A-Za-z0-9]+' "$1" | sed 's#.*/##' | sort -u | grep -c . ) || return 1
  [ "$NI" = 1 ] && printf '%s\n' "$AID"
}
after_run() {  # $1 = log, $2 = app-name prefix. Runs WHATEVER the run's exit code (the line captured RC first and checks it after): with the
  # banner id, polls that app to "stopped"; without one, polls the prefix (so a live app is still seen) and returns failure.
  if AID=$(app_id_of "$1"); then
    echo "app_id $AID" && poll_app_stopped "$AID"
  else
    echo "no single banner app id in the log"; poll_stopped "$2"; return 1
  fi
}
tokens_ok() {  # $1 = a log; $2.. = exact lines, each of which must appear in it exactly once (grep -x -F). Prints the first missing or repeated one.
  L="$1"; shift
  for t in "$@"; do
    NT=$(grep -c -x -F -- "$t" "$L"; true)
    [ "$NT" = 1 ] || { echo "token missing or repeated: $t"; return 1; }
  done
}
plant_run3() {  # $1 = the config name (no path); $2.. = its exact token lines. Never opens the results record (0d's rule). The post-run poll runs
  # whatever the exit code (after_run); the exit code is checked after it.
  C="$1"; shift; LOG="$LD/$C.log"
  [ ! -e "$LOG" ] \
  && { PYTHONPATH=src NO_COLOR=1 COLUMNS=1000 caffeinate -i "$M" run --detach "$S" --cfg "$C" < /dev/null > "$LOG" 2>&1; RC=$?; true; } \
  && after_run "$LOG" "$APP" \
  && [ "$RC" = 0 ] \
  && tokens_ok "$LOG" "$@" \
  && RL=$(grep -E -x 'RECORD panel_plant_rows_'"$C"'_[0-9_-]+\.json sha256 [0-9a-f]{64}' "$LOG") \
  && NR=$(printf '%s\n' "$RL" | grep -c .) && [ "$NR" = 1 ] \
  && echo "$C: as stated; $RL"
}
