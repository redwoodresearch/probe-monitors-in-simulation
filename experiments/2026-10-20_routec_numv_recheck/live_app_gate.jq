# Live-app gate for `modal app list --json` (modal 1.6.1: lower-case keys app_id, description, state, ...).
# Prints the number of non-stopped apps whose description is $a, or "SCHEMA" when the input is not a non-empty array
# of objects with "state" and "description" (fail closed: an upper-case or missing key must never read as 0 live apps).
if type == "array" and length > 0 and all(.[]; type == "object" and has("state") and has("description"))
then [.[] | select(.description == $a and .state != "stopped")] | length
else "SCHEMA"
end
