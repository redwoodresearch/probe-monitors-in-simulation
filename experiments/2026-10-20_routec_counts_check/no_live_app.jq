# COUNTSCHECK launch gate: true iff no app named $n is live. Fails closed on a schema change (modal 1.6.1 `app list --json`:
# an array of objects with lower-case "description" and "state").
if type != "array" then error("app list is not an array")
elif ([.[] | select(type == "object" and has("description") and has("state"))] | length) != length then error("app list schema changed")
else ([.[] | select(.description == $n and .state != "stopped")] | length) == 0 end
