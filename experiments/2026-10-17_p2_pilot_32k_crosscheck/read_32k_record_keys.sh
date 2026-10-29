#!/bin/bash
# 0d case (i), 10-17: keys-only field read of a VK record (writer run_simh_scores.py), values never printed.
# For the cross-check's STOP diagnosis (launch 18:30:58 PT 10-17, ap-d3O3B14fcAuhD6A7qCRIpS). main() reads the 32k pilot record
# simh_manifest_2026-10-14_172305.json @ 41a865a0e ONCE with git show, checks the sha256 of those bytes against the pin, and passes the SAME
# bytes to keys_of, which prints ONLY, by explicit selection: dump.run, dump.n_epochs, the number of staged entries, each staged entry's
# path file name and sha256, and dump.dump_manifest_sha256 (a63478 K1-K4):
# - every printed value is a scalar: a string (an integer for n_epochs), else "non-scalar (<type>)"; nothing non-scalar goes to tojson;
# - staged must be an array (its length and range() index only), else "non-scalar (<type>)"; no to_entries, so no key is ever printed;
# - jq's stdout is captured and its stderr discarded; on any jq error ONLY "STOP: jq rc <n>" is printed, never partial output.
# Run once, from any directory: bash <this file>. Sourcing it defines keys_of and runs nothing (the test feeds it synthetic JSON).
set -u -o pipefail
C=41a865a0efe7f7261d5c801fb00b2889e2cb7c35
P=experiments/2026-10-14_p2_simh_scores/results/simh_manifest_2026-10-14_172305.json
S=663458f13bcfeaff4f06521cede93161c59e1256f139af1ae71512187aba7976
F='def sc_str: if type == "string" then tojson else "non-scalar (\(type))" end;
def sc_int: if type == "number" and . == floor then tojson else "non-scalar (\(type))" end;
.dump as $d
| "dump.run: \($d.run | sc_str)",
  "dump.n_epochs: \($d.n_epochs | sc_int)",
  (if ($d.staged | type) == "array" then
     "dump.staged entries: \($d.staged | length)",
     (range(0; $d.staged | length) as $i | $d.staged[$i] as $e
      | if ($e | type) == "object" then
          "dump.staged[\($i)]: path file name \(if ($e.path | type) == "string" then ($e.path | split("/") | last | tojson) else "non-scalar (\($e.path | type))" end); sha256 \($e.sha | sc_str)"
        else "dump.staged[\($i)]: non-scalar (\($e | type))" end)
   else "dump.staged: non-scalar (\($d.staged | type))" end),
  "dump.dump_manifest_sha256: \($d.dump_manifest_sha256 | sc_str)"'

keys_of() {  # stdin: the record's bytes; stdout: the selected fields, or ONLY "STOP: jq rc <n>"
    local o rc
    o=$(jq -r "$F" 2>/dev/null)
    rc=$?
    if [ "$rc" != 0 ]; then
        echo "STOP: jq rc $rc"
        return 1
    fi
    printf '%s\n' "$o"
}

main() {
    local g r h
    g=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd) || { echo "STOP: no script directory"; return 1; }
    r=$(git -C "$g" show "$C:$P" 2>/dev/null && printf x) || { echo "STOP: git show failed"; return 1; }
    r=${r%x}  # the record's bytes, read ONCE (trailing newlines kept); hashed and parsed below from this one copy
    h=$(printf '%s' "$r" | shasum -a 256 | cut -c1-64)
    [ "$h" = "$S" ] || { echo "STOP: the record's sha256 is not the pin"; return 1; }
    printf '%s' "$r" | keys_of
}

if [ "${BASH_SOURCE[0]}" = "$0" ]; then
    main
    exit $?
fi
