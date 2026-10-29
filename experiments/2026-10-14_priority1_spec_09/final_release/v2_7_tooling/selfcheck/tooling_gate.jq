# tooling_gate.jq: the committed v2_4_tooling/ files must hold keys, shas, counts and class words only (09, via 2e 01:08): no printed or
# measured value and no provenance row's item / anchor / predicate. Output: {n: {...}, bad: [ "file: check (line N)" ]}; never a token.
# Inputs: --slurpfile F  files, one {name, raw, json (parsed, or null for code files)} per file
#         --slurpfile P18 the v18 provenance rows (the item / anchor / predicate / printed strings to look for)
# b8 (fresh review N4-N7, 09): decimals are strict. A decimal token is allowed only if it is
#   (a) a level key: a decimal that occurs in the v18 key paths (LEVELS), used as a key part: followed by "|", by "." + a letter, inside a
#       whole JSON string, or as the level in "(C, joint, x)"; or
#   (b) a release version (VERSIONS) right after "v", "version " or as a whole JSON string; or
#   (c) the trailer's model version right after " Opus ".
# Percents: only the class word "95% interval(s)" / "95% CI"; a number followed by the word percent / per cent / pct is flagged.
# Integers (09): an integer of 2+ digits in a key-ish field (a JSON string under a key containing "key", or "slot" / "a1_path" / "key_path"
#   / "id"; or a jq code string assigned to such a key) that equals a v18 printed integer token is flagged; times (next to ":") are not.
def BANNED_KEYS: ["item", "predicate", "printed", "rounding", "match", "v84_sha256", "row_map", "occurrence_anchors", "occurrences_in_visible_text",
                  "cell_index", "value", "values", "value_string", "expected", "lo", "hi", "tol", "point", "ci95", "estimate"];
def GATES_ANCHOR_KEYS: ["commit", "file", "file_sha256", "line", "line_sha256"];
def VERSIONS: [range(1; 8) | "2." + tostring];   # this record series' release versions (built, so no literal number sits in this file)
def lineno($raw; $off): ($raw[0:$off] | [match("\n"; "g")] | length) + 1;
# identifiers that end in ".N" are keys, not numbers: the release disclosure ids (v2<d>.<item>[.<role>].<n>; v2.6: any release digit) and the short ids in disclosure_rows
# (R<row>.<n>, rows<a>_<b>.<n>). They are masked with "x" of the same length (line numbers unchanged) before the number checks.
def idmask: gsub("(?<id>v2[0-9]\\.[A-Za-z0-9_]+(\\.[A-Za-z0-9_-]+)?\\.[0-9]+|(?<![A-Za-z0-9_])(R[0-9]+|rows[0-9]+_[0-9]+)\\.[0-9]+)"; .id | gsub("."; "x"));
def dec_bad($raw; $LV): [$raw | match("(?<![0-9.])[0-9]+\\.[0-9]+(?![0-9])"; "g") | . as $m | $m.string as $t
    | ($raw[([$m.offset - 10, 0] | max):$m.offset]) as $pre
    | ($raw[($m.offset + $m.length):($m.offset + $m.length + 3)]) as $post3
    | ($post3 | if startswith("\\\"") then "\"" else .[0:1] end) as $post   # an escaped quote counts as a quote
    | (($pre | endswith("\"")) and $post == "\"") as $whole
    | select(
        ( ($LV | index([$t]) != null) and ($post == "|" or ($post3 | test("^\\.[A-Za-z_]")) or $whole or ($pre | endswith("joint, "))) )
        or ( (VERSIONS | index([$t]) != null) and (($pre | test("(^|[^A-Za-z])v$")) or ($pre | endswith("version ")) or $whole) )
        or ( $t == ("5" + ".5") and ($pre | endswith(" Opus ")) )
        | not)
    | lineno($raw; $m.offset)];
def pct_bad($raw): [$raw | match("[0-9]+(\\.[0-9]+)?%"; "g") | . as $m | select(($m.string == ("95" + "%") and ($raw[($m.offset + 3):($m.offset + 13)] | test("^ (interval|CI)"))) | not)
    | lineno($raw; $m.offset)];
def pctword_bad($raw): [$raw | match("(?i)(?<![0-9A-Za-z])[0-9]+(\\.[0-9]+)?[ -]?(percent|per cent|pct)(?![A-Za-z])"; "g") | lineno($raw; .offset)];
# exponent notation; a run of 7+ hex characters is a (short) commit sha, not a number
def exp_bad($raw): [$raw | match("(?<![0-9A-Za-z.])[0-9]+(\\.[0-9]+)?[eE]-?[0-9]+(?![0-9A-Za-z])"; "g") | select(.string | test("^[0-9a-f]{7,}$") | not) | lineno($raw; .offset)];
# thousands separators; a sed address range ("N,Mp") is code, not a number
def thousands_bad($raw): [$raw | match("(?<![0-9])[0-9]{1,3}(,[0-9]{3})+(?![0-9])"; "g") | select(. as $m | $raw[($m.offset + $m.length):($m.offset + $m.length + 1)] != "p") | lineno($raw; .offset)];
# integer tokens (2+ digits, not part of a time, a word or a dotted / slashed / hyphenated id) in a string
def ints($s): [$s | match("(?<![0-9A-Za-z_.:/-])[0-9][0-9,]+(?![0-9A-Za-z_.:/-])"; "g") | .string | gsub(","; "")];
def KEYISH: test("key") or . == "slot" or . == "a1_path" or . == "key_path" or . == "id";
# the text of one value expression: from the start of the input to the first "," or unmatched closing bracket at depth 0, outside
# double-quoted strings (backslash escapes honoured); the whole input if neither occurs
def exprend: explode as $c
  | (reduce range(0; $c | length) as $i ({d: 0, q: false, esc: false, end: null};
      if .end != null then .
      else $c[$i] as $ch
        | if .q then (if .esc then .esc = false elif $ch == 92 then .esc = true elif $ch == 34 then .q = false else . end)
          elif $ch == 34 then .q = true
          elif $ch == 40 or $ch == 91 or $ch == 123 then .d += 1
          elif $ch == 41 or $ch == 93 or $ch == 125 then (if .d == 0 then .end = $i else .d -= 1 end)
          elif $ch == 44 and .d == 0 then .end = $i
          else . end end) | .end) as $e
  | $c[0:($e // ($c | length))] | implode;
($P18 | [.[] | (.item, .anchor) | strings | select(length >= 12)] | unique) as $IA
| ($P18 | [.[] | .key_path] | tojson) as $KP
| ([$KP | match("(?<![0-9.])[0-9]+\\.[0-9]+(?![0-9])"; "g").string] | unique) as $LV
# predicate strings: key vocabulary (labels in key paths) and identifier-like class words (interval_low, slot_component_eq) are allowed
| ($P18 | [.[] | .predicate | .. | strings | select(length >= 12)] | unique
   | map(select(. as $s | ($KP | contains($s) | not) and ($s | test("^[A-Za-z_][A-Za-z0-9_]*$") | not)))) as $PS
| ($P18 | [.[] | .printed | .. | strings | select(test("[0-9]") and test("[.%]"))] | unique) as $PR
| ($P18 | [.[] | .printed | .. | strings | [match("(?<![0-9A-Za-z_.])[0-9][0-9,]*(?![0-9A-Za-z_.])"; "g").string | gsub(","; "")] | .[] | select(length >= 2)] | unique) as $PI
| [ $F[] | . as $f | $f.raw as $raw | ($raw | idmask) as $mraw
    | ( (if $f.json != null then
           ([$f.json | .. | numbers | select(. != floor)] | length) as $nf
           | ([$f.json | .. | objects | keys[] | select(. as $k | BANNED_KEYS | index([$k]) != null)] | length) as $nb
           | ([$f.json | .. | objects | select(has("anchor")) | .anchor | select((type != "object") or ((keys - GATES_ANCHOR_KEYS) != []))] | length) as $na
           | ([$f.json | paths(type == "string") as $p | select(($p | map(select(type == "string")) | last // "") | KEYISH) | ints($f.json | getpath($p))[]
               | select(. as $x | $PI | index([$x]) != null)] | length) as $ni
           | [ (if $nf > 0 then "\($f.name): \($nf) non-integer numbers" else empty end),
               (if $nb > 0 then "\($f.name): \($nb) provenance / value keys" else empty end),
               (if $na > 0 then "\($f.name): \($na) anchor fields that are not GATES line anchors" else empty end),
               (if $ni > 0 then "\($f.name): \($ni) integers in key-ish fields equal a printed token" else empty end) ]
         else
           # b9-tooling (Codex v24 r2 item 12) + b9-2 (Codex v24 r3 item 6): after a key-ish key, unquoted (`key:`) or quoted (`"key":`),
           # the WHOLE value expression across lines (up to the first "," or closing bracket at depth 0, outside strings; 4000 characters
           # at most) is taken, and every string literal in it is scanned
           # v2.5 (#11): also jq update assignments (`.key = ...`, `.key |= ...`, `."key" = ...`), whose value expression is scanned the same way
           ([$raw | (match("(?<![A-Za-z0-9_])\"?([A-Za-z0-9_]*key[A-Za-z0-9_]*|slot|a1_path|key_path|id)\"?\\s*:"; "g"),
                     match("(?<![A-Za-z0-9_])\\.\"?([A-Za-z0-9_]*key[A-Za-z0-9_]*|slot|a1_path|key_path|id)\"?\\s*\\|?=(?!=)"; "g")) | . as $m
             | $raw[($m.offset + $m.length):($m.offset + $m.length + 4000)] | exprend
             | [match("\"(?<s>(?:[^\"\\\\]|\\\\.)*)\""; "g") | .captures[0].string][] | ints(.)[] | select(. as $x | $PI | index([$x]) != null)] | length) as $nc
           | [ (if $nc > 0 then "\($f.name): \($nc) integers in key-ish code strings equal a printed token" else empty end) ]
         end)
        + [ ($IA[] | select(. as $s | $raw | contains($s)) | "\($f.name): holds a provenance item / anchor string") ] [0:1]
        + [ ($PS[] | select(. as $s | $raw | contains($s)) | "\($f.name): holds a provenance predicate string") ] [0:1]
        + [ ($PR[] | . as $s | select($mraw | test("(?<![0-9A-Za-z.])" + ($s | gsub("(?<c>[.%(){}\\[\\]+*?^$|\\\\])"; "\\\(.c)")) + "(?![0-9])")) | "\($f.name): holds a printed token") ] [0:1]
        + [ dec_bad($mraw; $LV)[] | "\($f.name): decimal token outside a key or version (line \(.))" ]
        + [ pct_bad($mraw)[] | "\($f.name): percent token other than the 95% interval class word (line \(.))" ]
        + [ pctword_bad($mraw)[] | "\($f.name): percent as a word (line \(.))" ]
        + [ exp_bad($mraw)[] | "\($f.name): exponent token (line \(.))" ]
        # thousands separators: in code files anywhere; in JSON files only inside string leaves (a compact integer array is not one)
        + (if $f.json == null then [ thousands_bad($mraw)[] | "\($f.name): thousands-separated number (line \(.))" ]
           else ([$f.json | .. | strings] | join("\n")) as $sv | [ thousands_bad($sv)[] | "\($f.name): thousands-separated number (string leaf)" ] end) ) ] as $per
| {n: {files: ($F | length), json_files: ([$F[] | select(.json != null)] | length), item_anchor_strings: ($IA | length), predicate_strings: ($PS | length),
       printed_tokens: ($PR | length), printed_integers: ($PI | length), levels: ($LV | length), files_flagged: ([$per[] | select(length > 0)] | length)},
   bad: [$per[][]]}
