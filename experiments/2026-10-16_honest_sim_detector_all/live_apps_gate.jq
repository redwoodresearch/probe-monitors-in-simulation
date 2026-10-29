# Live-app gate for the honest-sim detector rerun (route C, 2026-10-18). Input: `modal app list --json` (modal 1.6.1: lower-case keys
# description / state). Usage: jq -e --arg n <app description> -f live_apps_gate.jq. Fails closed: exit 5 (error) on any schema change,
# exit 1 (false) if an app with this description is not stopped, exit 0 (true) only if none is live.
if type != "array" then error("app list schema: not an array")
elif any(.[]; (type != "object") or ((has("description") and has("state")) | not)
           or ((.description | type) != "string") or ((.state | type) != "string")) then error("app list schema: an entry lacks string description / state")
else ([.[] | select(.description == $n and .state != "stopped")] | length == 0)
end
