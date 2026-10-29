# live_apps_gate.jq: true iff `modal app list --json` shows 0 live apps named $app (refit3-l48 launch gate; COMMON_RULES step 4).
# Run: "$M" app list --json | jq -e --arg app <name> -f live_apps_gate.jq      (false or a schema error exits non-zero)
# Fail closed: the input must be a list of objects whose lower-case "description" and "state" are strings (else error, exit 5).
# Any state other than exactly "stopped" counts as live (seen states 10-20: deployed, ephemeral, ephemeral (detached), stopped).
if type != "array" or any(.[]; type != "object" or (.description | type) != "string" or (.state | type) != "string")
then error("schema: not a list of {description, state} strings")
else ([.[] | select(.description == $app and .state != "stopped")] | length) == 0
end
