# SC1 node lister (port of selfcheck_release_record.py v2.3, L56-86): one TSV line per object node that the Python walk() checks.
# Emits shas, paths, commits and line numbers only (never a value). Kinds:
#   F  where commit path sha256 decompressed_sha256|-  origin_branches_json|-      ({path, commit, sha256} <= keys)
#   A  where commit file file_sha256 line line_sha256                              ({file, commit, file_sha256, line, line_sha256} <= keys)
#   C  where commit path lo hi lines_sha256                                        ({path, commit, lines, lines_sha256} <= keys)
#   I  where commit input sha256                                                   ({input, commit, sha256} <= keys)
#   N  commit file line                                                            (every {file, commit, line} node: line text for SC-rest)
def where: reduce .[] as $p ("$"; if ($p | type) == "number" then . + "[\($p)]" else . + "." + $p end);
def has_all($ks): . as $o | all($ks[]; . as $k | $o | has($k));
path(.. | objects) as $pp
| getpath($pp) as $d
| ($pp | where) as $w
| (if ($d | has_all(["path", "commit", "sha256"])) then
     ["F", $w, $d.commit, $d.path, $d.sha256, ($d.decompressed_sha256 // "-"), (if $d | has("origin_branches") then ($d.origin_branches | tojson) else "-" end)]
   else empty end),
  (if ($d | has_all(["file", "commit", "file_sha256", "line", "line_sha256"])) then
     ["A", $w, $d.commit, $d.file, $d.file_sha256, ($d.line | tostring), $d.line_sha256]
   else empty end),
  (if ($d | has_all(["path", "commit", "lines", "lines_sha256"])) then
     ["C", $w, $d.commit, $d.path, ($d.lines[0] | tostring), ($d.lines[1] | tostring), $d.lines_sha256]
   else empty end),
  (if ($d | has_all(["input", "commit", "sha256"])) then
     ["I", $w, $d.commit, $d.input, $d.sha256]
   else empty end),
  (if ($d | has_all(["file", "commit", "line"])) then
     ["N", $d.commit, $d.file, ($d.line | tostring)]
   else empty end)
| @tsv
