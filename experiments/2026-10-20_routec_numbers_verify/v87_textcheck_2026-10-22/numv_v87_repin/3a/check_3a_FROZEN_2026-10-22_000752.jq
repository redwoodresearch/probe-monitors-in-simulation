# (3a') stage 2: the text-only numbers check of v87 against the numbers-verified X (09, GATES L3501). Keys, positions (code points),
# line numbers, row ids, counts and booleans only: never a core, a literal or text.
#   jq -n -f check_3a.jq --rawfile xs <X stripped> --rawfile vs <v87 or a planted copy> --rawfile re <writing_core_regex.bin>
#     --rawfile ht <D1 hunks.txt> --slurpfile cmds <per-hunk char-diff commands, JSON lines {idx,k,o1,o2,n1,n2}>
#     --slurpfile v19 <provenance v19> --slurpfile v10 <constants v10> --slurpfile dfr <deferred v8> --slurpfile appr <approved flags>
#     --argjson r241ok <true|false> --argjson d1cores <{"idx": [removed, added]}> --arg mode <label>
# Alignment (09 condition 1): D1's hunks align lines; outside them, lines map one to one by offset and must be byte-equal. Inside a
# kind-c hunk, POSIX diff on code points aligns characters; kind d deletes every X character; kind a adds every v87 character.
# Never a text search in v87. A core is Writing's regex match (09), trailing commas dropped (as verify_post.tokens rstrips them).
def cores($t): [$t | match($re; "g") | (.string | sub(",+$"; "")) as $s
                | {s: .offset, str: $s, e: (.offset + ($s | length)), e2: (.offset + ($s | sub("%$"; "") | length))}];
def starts($L): reduce $L[] as $l ([0]; . + [.[-1] + ($l | length) + 1]);
def lineof($S; $p): [$S[] | select(. <= $p)] | length;
def runmap($C; $no; $nn; $kk):  # diff normal-format commands -> {m: {old index: new index}, bad}; $kk: true = 0-based keys, else 1-based
  (reduce $C[] as $c ({i: 1, j: 1, m: {}, bad: 0};
     (if $c.k == "a" then $c.o1 + 1 else $c.o1 end) as $sx | (if $c.k == "d" then $c.n1 + 1 else $c.n1 end) as $sv
     | .bad += (if ($sx - .i) == ($sv - .j) and ($sx - .i) >= 0 then 0 else 1 end)
     | reduce range(0; $sx - .i) as $q (.; .m[(.i + $q - (if $kk then 1 else 0 end)) | tostring] = (.j + $q - (if $kk then 1 else 0 end)))
     | .i = (if $c.k == "a" then $c.o1 + 1 else $c.o2 + 1 end) | .j = (if $c.k == "d" then $c.n1 + 1 else $c.n2 + 1 end)))
  | ($no + 1 - .i) as $rx | ($nn + 1 - .j) as $rv | .bad += (if $rx == $rv and $rx >= 0 then 0 else 1 end)
  | reduce range(0; $rx) as $q (.; .m[(.i + $q - (if $kk then 1 else 0 end)) | tostring] = (.j + $q - (if $kk then 1 else 0 end)));

# AMEND1 (draft): anchors are located in RAW X (unique there, as verify_post locates them), then mapped to X-stripped positions through
# the hidden-comment spans of D1's pattern ($hp, flag "m" as D1); gate: removing those spans from raw X gives $xs byte for byte.
($xs | split("\n")) as $XL | ($vs | split("\n")) as $VL
| starts($XL) as $XS | starts($VL) as $VS
| ($xs | explode) as $XE | ($vs | explode) as $VE
| ($xr | explode) as $XRE
| [$xr | match($hp; "gm") | {s: .offset, e: (.offset + .length)}] as $HID
| (($xr | gsub($hp; ""; "m")) == $xs) as $strip_ok
| def r2s($p): $p - ([$HID[] | select(.s < $p) | ([.e, $p] | min) - .s] | add // 0);
  def inhid($a; $b): [$HID[] | select(.s < $b and .e > $a)] | length > 0;
  [$ht | split("\n")[] | select(startswith("HUNK ")) | split(" ")
   | {idx: (.[1] | tonumber), kind: .[2], os: (.[3] | tonumber), oe: (.[4] | tonumber), ns: (.[5] | tonumber), ne: (.[6] | tonumber)}] as $H
# ---- lines: D1's hunks as diff commands over lines; every unchanged line pair must be byte-equal
| runmap([$H[] | {k: .kind, o1: .os, o2: .oe, n1: .ns, n2: .ne}]; ($XL | length); ($VL | length); false) as $LM
| ([$LM.m | to_entries[] | select($XL[(.key | tonumber) - 1] != $VL[.value - 1])] | length) as $line_unequal
# ---- characters inside kind-c hunks
| (reduce $H[] as $h ({};
     if $h.kind != "c" then . else
       ([$cmds[] | select(.idx == $h.idx)]) as $C
       | ((($XL[$h.os - 1:$h.oe] | join("\n")) | length)) as $no | ((($VL[$h.ns - 1:$h.ne] | join("\n")) | length)) as $nn
       | .[$h.idx | tostring] = (runmap($C; $no; $nn; true) + {cmds: $C}) end)) as $CM
| (cores($xs) | map(. + {ln: lineof($XS; .s)})) as $XC | (cores($vs) | map(. + {ln: lineof($VS; .s)})) as $VC
| ($VC | map({key: (.s | tostring), value: .}) | from_entries) as $VAT
| def hunk_of($ln): [$H[] | select(.kind != "a" and .os <= $ln and $ln <= .oe)] | .[0];
  def newside_has_core($h; $r0; $n):  # the kind-c commands replacing these X characters: does their v87 side hold a core?
    if $h.kind == "d" then false else
      ($VS[$h.ns - 1]) as $b
      | [$CM[$h.idx | tostring].cmds[] | select(.k == "c" and .o1 <= $r0 + $n and .o2 >= $r0 + 1)
         | ($b + .n1 - 1) as $lo | ($b + .n2) as $hi | $VC[] | select(.s < $hi and .e > $lo)] | length > 0 end;
  def mapcore($c):
    $c.ln as $ln | hunk_of($ln) as $h
    | if $h == null then
        ($LM.m[$ln | tostring]) as $vl
        | if $vl == null then {st: "unmapped_line"} else
            ($VS[$vl - 1] + ($c.s - $XS[$ln - 1])) as $vp | ($VAT[$vp | tostring]) as $vc
            | if $vc != null and $vc.str == $c.str then {st: "kept", vpos: $vp, vline: $vl, hunk: null} else {st: "changed", vline: $vl, hunk: null} end end
      else
        ($c.s - $XS[$h.os - 1]) as $r0 | ($c.str | length) as $n
        | (if $h.kind == "d" then [range(0; $n) | null] else [range(0; $n) as $q | $CM[$h.idx | tostring].m[($r0 + $q) | tostring]] end) as $img
        # kept: every character maps, contiguously, onto a v87 core with the same bytes. not_printed: no mapped character lands inside a
        # v87 core AND the diff commands replacing its unmapped characters put no v87 core in their place (a deletion, or core-free
        # text). Anything else (a partial map onto a core, a different number in its place): changed, a FAIL.
        | ([$img[] | select(. != null) | $VS[$h.ns - 1] + .]) as $vimg
        | ([$vimg[] as $p | $VC[] | select(.s <= $p and $p < .e)] | length > 0) as $hits
        | if ($img | all(. != null)) and ([range(0; $n) as $q | $img[$q] == $img[0] + $q] | all)
             and (($VAT[($VS[$h.ns - 1] + $img[0]) | tostring]) as $vc | $vc != null and $vc.str == $c.str) then
            ($VS[$h.ns - 1] + $img[0]) as $vp | {st: "kept", vpos: $vp, vline: lineof($VS; $vp), hunk: $h.idx}
          elif ($hits | not) and (newside_has_core($h; $r0; $n) | not) then {st: "not_printed", hunk: $h.idx}
          else {st: "changed", hunk: $h.idx} end
      end;
  ($XC | map({key: (.s | tostring), value: (. + mapcore(.))}) | from_entries) as $XM
| ([$XM[] | select(.st == "kept") | .vpos]) as $kept_v
# ---- per hunk: X cores not kept (removed) and v87 cores not hit (added); cross-check with D1's multiset counts (R1 output)
| [$H[] | . as $h
   | (if $h.kind == "a" then [] else [$XC[] | select(.ln >= $h.os and .ln <= $h.oe)] end) as $xo
   | (if $h.kind == "d" then [] else [$VC[] | select(.ln >= $h.ns and .ln <= $h.ne)] end) as $vn
   | ([$xo[] | select($XM[.s | tostring].st != "kept")]) as $rem
   | ([$vn[] | select(.s as $p | $kept_v | index($p) | not)]) as $add
   # AMEND1 (draft): D1's own multiset counts (verbatim regex strings, as d1_cores.jq), reproduced per hunk; the positional counts stay as info
   | def mdiff($a; $b): reduce $b[] as $x ($a; (index([$x])) as $i | if $i == null then . else del(.[$i]) end);
     ((if $h.kind == "a" then "" else ($XL[$h.os - 1:$h.oe] | join("\n")) end) | [match($re; "g") | .string] | sort) as $mo
   | ((if $h.kind == "d" then "" else ($VL[$h.ns - 1:$h.ne] | join("\n")) end) | [match($re; "g") | .string] | sort) as $mn
   | [(mdiff($mo; $mn) | length), (mdiff($mn; $mo) | length)] as $ms
   | {idx: $h.idx, kind: $h.kind, x_cores: ($xo | length), v_cores: ($vn | length), removed: ($rem | length), added: ($add | length),
      added_pos: [$add[].s], added_has_pct: [$add[] | .str | endswith("%")],
      d1: ($d1cores[$h.idx | tostring] // [0, 0]), multiset_removed_added: $ms,
      positional_match: (($d1cores[$h.idx | tostring] // [0, 0]) == [($rem | length), ($add | length)]),
      d1_match: (($d1cores[$h.idx | tostring] // [0, 0]) == $ms),
      charmap_bad: (($CM[$h.idx | tostring].bad) // 0)}] as $HR
# ---- rows: claimed cores located in X by whole-core matching at the anchor's ONE position in X (cell_index as the verifier)
| def resolve($anchor; $printed; $ci):
    ($XRE | indices($anchor | explode)) as $occ
    | if ($occ | length) != 1 then {err: "anchor_not_once_in_raw_X"}
      elif ([$HID[] | select(.s <= $occ[0] and $occ[0] + ($anchor | length) <= .e)] | length) > 0 then {err: "anchor_inside_hidden_comment"}
      else
        $occ[0] as $braw | r2s($braw) as $base | ($anchor | explode) as $AE | ($printed | explode) as $PE | ($PE | length) as $L
        | [cores($printed)[] | .str | sub("%$"; "")] as $pc
        | [($AE | indices($PE))[] as $st | ($braw + $st) as $araw | select(inhid($araw; $araw + $L) | not) | r2s($araw) as $a0
           | [$XC[] | select(.s < $a0 + $L and .e > $a0)] as $ov
           | select(($ov | all(.s >= $a0 and .e2 <= $a0 + $L)) and ($ov | map(.str | sub("%$"; ""))) == $pc)
           | {st: $st, ov: $ov}] as $good
        | (if ($ci | type) == "number" then ($AE | indices(124)) as $bars
             | (if $ci >= 1 and $ci < ($bars | length) then [$good[] | select(.st >= $bars[$ci - 1] + 1 and .st + $L <= $bars[$ci])] else $good end)
           else $good end) as $g2
        | if ($g2 | length) != 1 then {err: "printed_not_one_whole_core_match_in_anchor"}
          else {cores: [$g2[0].ov[].s], base: $base, ps: r2s($braw + $g2[0].st), pl: $L, pstr: $printed,
                overlaps_hidden: inhid($occ[0]; $occ[0] + ($anchor | length))} end
      end;
  # AMEND1 (09, L3515): a no-core span (number words) must be byte-equal at its aligned v87 position, by script
  def mapspan($s; $L; $str):
    lineof($XS; $s) as $ln | hunk_of($ln) as $h
    | if $h == null then
        ($LM.m[$ln | tostring]) as $vl
        | if $vl == null then false else ($VS[$vl - 1] + ($s - $XS[$ln - 1])) as $vp | ($VE[$vp:$vp + $L] | implode) == $str end
      elif $h.kind == "d" then false
      else
        ($s - $XS[$h.os - 1]) as $r0 | [range(0; $L) as $q | $CM[$h.idx | tostring].m[($r0 + $q) | tostring]] as $img
        | if ($img | all(. != null)) and ([range(0; $L) as $q | $img[$q] == $img[0] + $q] | all) then
            ($VS[$h.ns - 1] + $img[0]) as $vp | ($VE[$vp:$vp + $L] | implode) == $str
          else false end
      end;
  def touches($braw; $len):  # the anchor's X lines meet a changed hunk (kind c / d lines; kind a: the anchor spans its insertion point)
    r2s($braw) as $s0 | ([r2s($braw + $len) - 1, $s0] | max) as $s1
    | lineof($XS; $s0) as $l0 | lineof($XS; $s1) as $l1
    | [$H[] | select((.kind != "a" and .os <= $l1 and .oe >= $l0) or (.kind == "a" and $l0 <= .os and $l1 >= .os + 1)) | .idx];
  def status($cs):
    ([$cs[] | $XM[. | tostring].st]) as $s
    | if ($cs | length) == 0 then "no_core" elif ($s | all(. == "kept")) then "kept" elif ($s | all(. == "not_printed")) then "not_printed"
      else "FAIL" end;
  def rowcheck($set; $id; $anchors; $printed; $ci):
    [$anchors[] | resolve(.; $printed; $ci)] as $R
    | ([$R[] | select(.err) | .err]) as $errs
    | ([$R[] | select(.cores) | .cores[]]) as $cs
    | ([$anchors[] as $a | ($XRE | indices($a | explode)) as $o | if ($o | length) == 1 then touches($o[0]; ($a | length)) else [] end] | flatten | unique) as $hk
    | (if ($errs | length) > 0 then "FAIL" else status($cs) end) as $st0
    | (if $st0 == "no_core" then (if ([$R[] | mapspan(.ps; .pl; .pstr)] | all) then "no_core_kept" else "FAIL" end) else $st0 end) as $st
    | {set: $set, row: $id, hunks: $hk, n_cores: ($cs | length), status: $st, errs: $errs,
       overlaps_hidden: ([$R[] | .overlaps_hidden // false] | any),
       core_status: [$cs[] | $XM[. | tostring].st], v87_lines: ([$cs[] | $XM[. | tostring].vline // empty] | unique),
       v87_anchor_counts: [$anchors[] as $a | $VE | indices($a | explode) | length]};
  ([$v19[] | rowcheck("v19"; .row; [.anchor] + (.occurrence_anchors // []); .printed; .cell_index)]
   + [$v10[] | rowcheck("v10"; .row; [.anchor]; .text; null)]) as $ROWS
| ([$dfr | to_entries[] | .value as $d | ($d.row // "line\(.key + 1)") as $id
    | ($XRE | indices($d.anchor | explode)) as $o
    | if ($o | length) != 1 then {set: "deferred", row: $id, status: "FAIL", errs: ["anchor_not_once_in_raw_X"], hunks: []} else
        ([$XC[] | select(.s >= r2s($o[0]) and .e <= r2s($o[0] + ($d.anchor | length))) | .s]) as $cs
        | r2s($o[0]) as $d0 | r2s($o[0] + ($d.anchor | length)) as $d1
        | ([$xs[$d0:$d1] | match("\\b(zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|hundreds|thousand|thousands|million|millions|billion|dozen|dozens|half|halves|third|thirds|quarter|quarters|fifth|fifths|sixth|sixths|seventh|eighth|ninth|tenth|tenths|twice|thrice|single|double|triple|none)\\b"; "gi")
            | mapspan($d0 + .offset; .length; .string)]) as $nw
        | status($cs) as $st0
        | (if $st0 == "no_core" then (if ($nw | length) == 0 then "no_core_no_number_words" elif ($nw | all) then "no_core_kept" else "FAIL" end) else $st0 end) as $st
        | {set: "deferred", row: $id, hunks: touches($o[0]; ($d.anchor | length)), n_cores: ($cs | length), status: $st, n_number_words: ($nw | length),
           core_status: [$cs[] | $XM[. | tostring].st], v87_lines: ([$cs[] | $XM[. | tostring].vline // empty] | unique)} end]) as $DROWS
| ([$ROWS[] | select((.hunks | length) > 0)]) as $IN
| ([$ROWS[] | select((.hunks | length) == 0)]) as $OUT
# ---- check (b): the one added core of hunks 8, 24, 29 equals an approved constants literal (v10 text, approved source bound by sha)
| ($appr[0] | map(select(.approved)) | map(.row)) as $okrows
| [8, 24, 29] as $BH
| [$BH[] as $i | ($HR[] | select(.idx == $i)) as $hr
   | {hunk: $i, n_added: $hr.added,
      approved_rows: [($hr.added_pos[] as $p | $VAT[$p | tostring].str) as $cstr | $v10[] | select(.text == $cstr) | .row | select(. as $r | $okrows | index($r))],
      any_rows: [($hr.added_pos[] as $p | $VAT[$p | tostring].str) as $cstr | $v10[] | select(.text == $cstr) | .row],
      # AMEND1 (09, L3515): NO generic "%" rule. Hunk 8 ONLY: the core minus ONE trailing "%" equals the literal of v10 row 19 or 83
      # (script part only). 09 L3533: final acceptance also needs hunk8_consistency (h8_consistency.jq, on X's own definition) and the
      # reader's (ii); both are joined in the driver's OUTCOME line, not here
      h8_rows_19_83: (if $i == 8 then [($hr.added_pos[] as $p | $VAT[$p | tostring].str) as $cstr | select($cstr | endswith("%"))
                       | ($cstr | .[0:-1]) as $c0 | $v10[] | select((.row == 19 or .row == 83) and .text == $c0) | .row
                       | select(. as $r | $okrows | index($r))] else [] end)}
   | . + {ok_exact: (.n_added == 1 and (.approved_rows | length) > 0)}
   | . + {ok_script: (.ok_exact or ($i == 8 and .n_added == 1 and (.h8_rows_19_83 | length) > 0))}
   | . + {ok: .ok_script,
          matched_by: (if .ok_exact then "exact" elif .ok_script then "h8_rows_19_83_pct (needs hunk8_consistency and the reader)" else "none" end)}] as $B
# ---- check (c): hunk 32's one added core = row 241's verified printed core x 100 (decimal shift, string arithmetic), rounding pre-stated
| def parts($s): ($s | sub("^[−+-]"; "") | sub("%$"; "") | gsub(","; "")) as $u | ($u | split(".")) as $p | {i: ($p[0] | sub("^0+(?=[0-9])"; "")), f: ($p[1] // "")};
  def sgn($s): ($s[0:1]) as $c | if $c == "+" then "+" elif $c == "-" or $c == "−" then "-" else null end;
  ([$v19[] | select(.row == 241)][0]) as $r241
| (cores($r241.printed)) as $c241s
| ([$HR[] | select(.idx == 32)][0] // {added: 0, added_pos: []}) as $h32
| (if ($h32.added == 1) then $VAT[$h32.added_pos[0] | tostring].str else null end) as $c32
| (if ($c241s | length) == 1 and $c32 != null then
     parts($c241s[0].str) as $x | ($x.f + "00") as $f2 | {i: (($x.i + $f2[0:2]) | sub("^0+(?=[0-9])"; "")), f: ($x.f[2:])} as $sh
     | parts($c32) as $v | ($sh.f | length) as $dpx | ($v.f | length) as $dpv
     | (if $dpv > $dpx then {verdict: "STOP_extra_precision"}
        elif $dpv == $dpx then {verdict: (if $v.i == $sh.i and $v.f == $sh.f then "EQUAL" else "UNEQUAL" end), rounding: "none (exact shift)"}
        else ($sh.f[$dpv:]) as $drop
          | if ($drop | test("^50*$")) then {verdict: "STOP_tie_not_determined"} else
              (($sh.i + $sh.f[0:$dpv]) | tonumber) as $k | (if ($drop[0:1] | tonumber) >= 5 then $k + 1 else $k end) as $rk
              | ($rk | tostring) as $rs | (if $dpv == 0 then {i: $rs, f: ""} else
                  (([range(0; [$dpv + 1 - ($rs | length), 0] | max)] | map("0") | join("")) + $rs) as $pad | {i: ($pad[0:-$dpv] | sub("^0+(?=[0-9])"; "")), f: $pad[-$dpv:]} end) as $rr
              | {verdict: (if $v.i == $rr.i and $v.f == $rr.f then "EQUAL" else "UNEQUAL" end), rounding: "half-up on a non-tie (= half-even)"} end end)
     + {dp_x_shift: $dpx, dp_v87: $dpv, has_pct: ($c32 | endswith("%")),
        sign_ok: (sgn($c32) == null or sgn($c32) == sgn($c241s[0].str)), x_sign_explicit: (sgn($c241s[0].str) != null)}
   else {verdict: "STOP_shape", n_241_cores: ($c241s | length), n_added_h32: $h32.added} end) as $C0
| (if ($H | length) == 0 then {verdict: "NA_identity", na: true, ok: null}  # AMEND1: (c) has nothing to apply to with no hunks (identity)
   else ($C0 + {na: false, r241_verified_in_X: $r241ok, ok: ($r241ok and $C0.verdict == "EQUAL" and ($C0.has_pct // false) and ($C0.sign_ok // false))}) end) as $C
| {mode: $mode,
   lines: {x: ($XL | length), v87: ($VL | length), hunks: ($H | length), runs_bad: $LM.bad, unchanged_pairs: ($LM.m | length), unchanged_unequal: $line_unequal,
           raw_x_strips_to_x_stripped: $strip_ok, hidden_spans: ($HID | length)},
   cores: {x: ($XC | length), v87: ($VC | length), x_kept: ([$XM[] | select(.st == "kept")] | length)},
   hunks: $HR,
   a: {rows_in_changed_hunks: ($IN | length), by_set: ($IN | group_by(.set) | map({key: .[0].set, value: length}) | from_entries),
       kept: ([$IN[] | select(.status == "kept")] | length), no_core: [$IN[] | select(.status == "no_core_kept") | "\(.set):\(.row)"],
       not_printed: [$IN[] | select(.status == "not_printed") | "\(.set):\(.row)"],
       fail: [$IN[] | select(.status == "FAIL") | {set, row, errs, core_status}],
       outside_rows: ($OUT | length), outside_fail: [$OUT[] | select(.status != "kept" and .status != "no_core_kept") | {set, row, status, errs}],
       anchors_overlapping_hidden_comments: [$ROWS[] | select(.overlaps_hidden) | "\(.set):\(.row)"],
       deferred: [$DROWS[] | select((.hunks | length) > 0) | {row, status, n_cores, n_number_words}],
       deferred_outside_not_kept: [$DROWS[] | select((.hunks | length) == 0 and (.status | IN("kept", "no_core_kept", "no_core_no_number_words") | not)) | .row],
       multi_occurrence_in_v87: [$ROWS[] | select(.v87_anchor_counts | any(. > 1)) | {set, row, v87_anchor_counts, status, v87_lines}],
       ok: (([$IN[] | select(.status == "FAIL")] | length) == 0 and $strip_ok
            and ([$OUT[] | select(.status != "kept" and .status != "no_core_kept")] | length) == 0 and $line_unequal == 0 and $LM.bad == 0
            and ([$DROWS[] | select(.status == "FAIL")] | length) == 0
            and ([$HR[] | select(.charmap_bad > 0)] | length) == 0)},
   b: {per_hunk: $B, ok: ($B | all(.ok)), ok_script: ($B | all(.ok_script))},
   c: $C,
   d1_cross_check: {all_match: ($HR | all(.d1_match)), mismatched_hunks: [$HR[] | select(.d1_match | not) | .idx]},
   reader_rows: ([$IN[] | {set, row, hunks, v87_lines, status}] + [$DROWS[] | select((.hunks | length) > 0) | {set, row, hunks, v87_lines, status}]),
   neg_targets: {h8_added_vpos: ([$HR[] | select(.idx == 8) | .added_pos[]] | .[0]),
                 h24_added_vpos: ([$HR[] | select(.idx == 24) | .added_pos[]] | .[0]),
                 h32_added_vpos: ([$HR[] | select(.idx == 32) | .added_pos[]] | .[0]),
                 v19_105_core_vpos: ([$v19[] | select(.row == 105)][0] | resolve(.anchor; .printed; .cell_index) | (.cores // [])
                                     | map($XM[. | tostring].vpos))}}
