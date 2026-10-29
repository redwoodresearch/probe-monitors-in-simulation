# AMEND1 hunk 8 (09, GATES L3533 @ eb7247196): a CONSISTENCY check of v87's hunk-8 "N%" against X's own verified definition. NOT a
# source check. Booleans, forms, X-stripped line ids and code-point positions only; never a core or text.
#   jq -n -f h8_consistency.jq --rawfile xr <raw X or a copy> --rawfile hp <D1 hidden pattern> --rawfile re <Writing's core regex>
#      --rawfile vs <v87 or a copy> --argjson h8pos <code point of hunk 8's one added core in vs, or null> --slurpfile v10 <constants v10>
# (1) N := v87's hunk-8 core minus ONE trailing "%" (the core must end in "%"). For EACH of v10 rows 19 and 83: its anchor occurs once in
# raw X and not inside a hidden comment; its literal sits at whole-core boundaries in the anchor (as check_3a resolve); literal == N and
# X's core there (minus a trailing "%") == N; then, in the VISIBLE (stripped) sentence holding that core (bounds: the nearest "\n" or
# [.?!] + whitespace before it, the nearest "\n" or [.?!] + whitespace / end after it), the core is immediately followed by "%" (form
# "%"), or by "per 100" / "in 100" with <= 3 whitespace-separated tokens between N and that phrase, "100" not followed by a digit or ","
# (forms "per 100" / "in 100"). Checked in that order. hunk8_consistency = all of the above for both rows.
def cores($t): [$t | match($re; "g") | (.string | sub(",+$"; "")) as $s | {s: .offset, str: $s, e: (.offset + ($s | length)), e2: (.offset + ($s | sub("%$"; "") | length))}];
def starts($L): reduce $L[] as $l ([0]; . + [.[-1] + ($l | length) + 1]);
def lineof($S; $p): [$S[] | select(. <= $p)] | length;
($xr | explode) as $XRE
| [$xr | match($hp; "gm") | {s: .offset, e: (.offset + .length)}] as $HID
| ($xr | gsub($hp; ""; "m")) as $xs
| starts($xs | split("\n")) as $XS
| cores($xs) as $XC
| def r2s($p): $p - ([$HID[] | select(.s < $p) | ([.e, $p] | min) - .s] | add // 0);
  def s2r($q): reduce $HID[] as $h ($q; if $h.s <= . then . + ($h.e - $h.s) else . end);  # spans in raw order: shift past each comment before
  ([$vs | match($re; "g") | select(.offset == $h8pos) | .string | sub(",+$"; "")][0]) as $c8
| ($c8 != null and ($c8 | endswith("%"))) as $c8pct
| (if $c8pct then ($c8 | .[0:-1]) else null end) as $N
| [$v10[] | select(.row == 19 or .row == 83) | . as $r
   | ($XRE | indices($r.anchor | explode)) as $occ
   | {row: $r.row, anchor_once_raw: (($occ | length) == 1),
      anchor_inside_hidden: (($occ | length) == 1 and ([$HID[] | select(.s <= $occ[0] and $occ[0] + ($r.anchor | length) <= .e)] | length) > 0),
      literal_eq_N: ($N != null and $r.text == $N)}
   | if (.anchor_once_raw | not) or .anchor_inside_hidden then . + {resolved: false, form: "none"} else
       $occ[0] as $braw | ($r.anchor | explode) as $AE | ($r.text | explode) as $PE | ($PE | length) as $L
       | [cores($r.text)[] | .str | sub("%$"; "")] as $pc
       | [($AE | indices($PE))[] as $st | ($braw + $st) as $araw | r2s($araw) as $a0
          | [$XC[] | select(.s < $a0 + $L and .e > $a0)] as $ov
          | select(($ov | length) == 1 and ($ov | all(.s >= $a0 and .e2 <= $a0 + $L)) and ($ov | map(.str | sub("%$"; ""))) == $pc)
          | {araw: $araw, core: $ov[0]}] as $g
       | if ($g | length) != 1 then . + {resolved: false, form: "none"} else
           $g[0].core as $k | ($k.str | sub("%$"; "")) as $kn
           | ([$xs[0:$k.s] | match("(\\n|[.?!]\\s)"; "g") | .offset + .length] | max // 0) as $b0
           | ([$xs[$k.e:] | match("(\\n|[.?!](\\s|$))"; "g") | .offset] | min // ($xs[$k.e:] | length)) as $b1r
           | ($xs[$k.e2:$k.e + $b1r]) as $after
           | (if ($xs[$k.e2:$k.e2 + 1] == "%") then "%"
              elif ($after | test("^(\\s+\\S+){0,3}?\\s+per\\s+100(?![0-9,])")) then "per 100"
              elif ($after | test("^(\\s+\\S+){0,3}?\\s+in\\s+100(?![0-9,])")) then "in 100" else "none" end) as $form
           | (if $form == "%" then $k.e2 elif $form == "none" then null
              else $k.e2 + ([$after | match("\\s+(per|in)\\s+100"; "g") | .captures[0].offset][0]) end) as $mpos
           | . + {resolved: true, x_core_eq_N: ($N != null and $kn == $N), form: $form,
                  sentence_x_lines: [lineof($XS; $b0), lineof($XS; $k.e + $b1r)],
                  core_raw_pos: $g[0].araw, marker_raw_pos: (if $mpos == null then null else s2r($mpos) end)}
         end
     end] as $rows
| {h8_core_found: ($c8 != null), h8_core_has_pct: $c8pct, raw_rows_found: ($rows | length),
   rows: $rows,
   hunk8_consistency: ($c8pct and ($rows | length) == 2
                       and ($rows | all(.resolved and .literal_eq_N and .x_core_eq_N and .form != "none" and (.anchor_inside_hidden | not)))),
   forms: [$rows[] | .form]}
