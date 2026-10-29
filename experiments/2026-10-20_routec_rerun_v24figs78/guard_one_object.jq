# Run as `jq -e -s -f guard_one_object.jq <file>` before either figs78 gate (Codex r7 REQUIRED 3): true iff the file holds exactly one JSON
# value and it is an object (jq -e otherwise uses the last of several outputs). Prints true or false only.
length == 1 and (.[0] | type) == "object"
