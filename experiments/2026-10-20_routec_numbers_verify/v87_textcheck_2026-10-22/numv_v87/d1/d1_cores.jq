# D1 R3 / R1 (09, GATES @ 3bd22fad4; PRESTATED_D1_R123_ADDENDUM_FROZEN_2026-10-21_220153.md). Counts and booleans only; never a core or text.
#   jq -n -r -f d1_cores.jq --rawfile xs <X stripped> --rawfile vs <v87 stripped> --rawfile re <writing_core_regex.bin> --slurpfile hunks <hunks.json>
# Per hunk: old side = X stripped lines os..oe (none for kind a); new side = v87 lines ns..ne (none for kind d). Cores = every match of Writing's
# regex (verbatim, jq/Oniguruma match "g"), as a multiset; added = new - old, removed = old - new. MOOT counts non-ASCII \p{Nd} characters on
# every compared line (0 => Python's Unicode \d and this engine's agree on these lines).
def mdiff($a; $b): reduce $b[] as $x ($a; (index([$x])) as $i | if $i == null then . else del(.[$i]) end);
def cores: [match($re; "g") | .string];
($xs | split("\n")) as $X | ($vs | split("\n")) as $V
| [$hunks[0][] | . as $h
   | ((if $h.kind == "a" then [] else $X[$h.os - 1:$h.oe] end) | join("\n")) as $o
   | ((if $h.kind == "d" then [] else $V[$h.ns - 1:$h.ne] end) | join("\n")) as $n
   | ($o | cores | sort) as $co | ($n | cores | sort) as $cn
   | {idx: $h.idx, co: ($co | length), cn: ($cn | length), eq: ($co == $cn),
      added: (mdiff($cn; $co) | length), removed: (mdiff($co; $cn) | length),
      moot: ([($o, $n) | match("\\p{Nd}"; "g") | .string | select(test("\\A[0-9]\\z") | not)] | length)}] as $R
| def h($i): ($R[] | select(.idx == $i));
  ([$R[] | select(.added > 0) | .idx]) as $HA
| ([$R[] | select(.idx != 3 and .idx != 4) | .added] | add) as $EX
| ([8, 24, 29, 32] | map(h(.).added)) as $F
| (h(5) | "R3 hunk=5 cores_old=\(.co) cores_new=\(.cn) equal=\(.eq)"),
  (h(21) | "R3 hunk=21 cores_old=\(.co) cores_new=\(.cn) equal=\(.eq)"),
  ($R[] | select(.added > 0 or .removed > 0) | "CORES hunk=\(.idx) removed=\(.removed) added=\(.added)"),
  "R1_COUNTS h8=\($F[0]) h24=\($F[1]) h29=\($F[2]) h32=\($F[3]) sum4=\($F | add) all_hunks_added=\([$R[].added] | add) excl_move_added=\($EX) hunks_with_added=\($HA | tojson)",
  "R1_ADDED_CORE_CONDITION \(($EX == 4) and ([$R[] | select(.idx != 3 and .idx != 4 and .added > 0) | [.idx, .added]] == [[8,1],[24,1],[29,1],[32,1]]))",
  "MOOT nonascii_decimal_digits=\([$R[].moot] | add) hunks=\($R | length)"
