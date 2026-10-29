# POST HOC after the AMEND1 run (disclosed; not part of its class). Keys, counts and booleans only.
def starts($L): reduce $L[] as $l ([0]; . + [.[-1] + ($l | length) + 1]);
def lineof($S; $p): [$S[] | select(. <= $p)] | length;
($xr | explode) as $XRE | [$xr | match($hp; "gm") | {s: .offset, e: (.offset + .length)}] as $HID
| def r2s($p): $p - ([$HID[] | select(.s < $p) | ([.e, $p] | min) - .s] | add // 0);
  starts($xs | split("\n")) as $XS | starts($vs | split("\n")) as $VS | ($vs | explode) as $VE
| ($v19 | map(select(.row == 74 or .row == 176))) as $R
| [$R[] | . as $p | ($XRE | indices($p.anchor | explode)) as $o | ($p.anchor | explode | indices($p.printed | explode)) as $st
   | (r2s($o[0] + $st[0])) as $s | lineof($XS; $s) as $ln
   | {row: $p.row, anchor_once_raw: (($o | length) == 1), printed_in_anchor: ($st | length), x_line: $ln,
      in_hunk: ([$H[] | select(.kind != "a" and .os <= $ln and $ln <= .oe) | .idx]),
      printed_len: ($p.printed | length),
      printed_count_in_v87: ($VE | indices($p.printed | explode) | length)}
   | . as $q | ($H[] | select(.idx == ($q.in_hunk[0] // -1))) as $h
   | . + (if $h == null then {} else
       ([$cmds[] | select(.idx == $h.idx)]) as $C
       | ($s - $XS[$h.os - 1]) as $r0
       | {hunk_v87_line: $h.ns, printed_count_in_aligned_v87_line: ($VE[$VS[$h.ns - 1]:$VS[$h.ne]] | indices($p.printed | explode) | length),
          chars_in_changed_cmds: ([range(0; $q.printed_len) as $k | ($r0 + $k + 1) as $i | [$C[] | select(.k != "a" and .o1 <= $i and $i <= .o2)] | length > 0] | map(select(.)) | length)} end)]
