# v2.4 VARIANT of sc_rest.jq: the v2.3-only invariants (Python L146, L542, L550, L437-447 "slots" only, printed-items groups only in preliminary_outputs) are replaced per 09's v2.4 rulings (GATES 22:46 Q1, Q9). Every other line is sc_rest.jq unchanged.
# SC-rest: port of selfcheck_release_record.py v2.3 (sha 2b0eaec4…), L89-573 (everything except the L56-86 walk, which sc_core.zsh does).
# Reads blobs that sc_core.zsh extracted from git (by commit:path), never git itself. Prints nothing: it returns one JSON object
#   {lines: [the Python's PERQ / FAMILY / SENS / PENDING_DOWNSTREAM / V2COND lines, in order], n: {counts}, bad: [mismatch texts]}
# Mismatch texts are the Python's want() texts: JSON paths, ids, slot names and key paths only, never a record value.
# Inputs: --slurpfile REC <record>  --slurpfile ST <store.jsonl {k, sha, v}>  --slurpfile LN <lines.jsonl {k, v}>
#         --slurpfile SH <shows.jsonl {k, v}>  --slurpfile LT <lstree.jsonl {k, v}>  --argjson HASMSG <bool>  --rawfile MSG <file>
#         --arg RECSHA <sha256 of the record file>
def truthy: . != null and . != false and . != "" and . != 0 and . != [] and . != {};
def W($c; $w): if $c then empty else $w end;
def has_all($ks): . as $o | (type == "object") and all($ks[]; . as $k | $o | has($k));
def pyre: gsub("\\(\\?P<"; "(?<");
def rsearch($re): test($re | pyre);
def rmatch($re): [capture("^(?:" + ($re | pyre) + ")")] | .[0];
def fmt($g): reduce ($g | to_entries[]) as $e (.; gsub("\\{" + $e.key + "\\}"; ($e.value // "None" | tostring)));
def pyfloat: try (tonumber | . + 0) catch null;
# v2.5 branch (i), 09's ruling on finding 2 (b1), pinned (via 2e): a figs78_item table passes iff its drawn elements (status other than
# "disclosure only (not drawn)") are all FINAL and number 3934 (fig 7) / 42 (fig 8); its not-drawn elements are exactly 2, one of each
# F78_ND_CLASSES string (copied from c2's tables by script; equal to 09's literals); elements_all_final is true. $J = the source blob.
def F78_ND_STATUS: "disclosure only (not drawn)";
def F78_DRAWN_N: {"7": 3934, "8": 42};
def F78_ND_CLASSES: ["metadata: bootstrap set descriptor (not drawn; disclosure only)","metadata: gate boolean (not drawn; disclosure only)"];
def b1ok($J): . as $et | ($J.fig_number | tostring) as $fn
  | [$et.elements[] | select(.status != F78_ND_STATUS)] as $dr | [$et.elements[] | select(.status == F78_ND_STATUS)] as $nd
  | (F78_DRAWN_N | has($fn)) and all($dr[]; .status == "FINAL") and ($dr | length) == F78_DRAWN_N[$fn]
    and ($nd | length) == 2 and ([$nd[].class] | sort) == F78_ND_CLASSES and $et.elements_all_final == true;
def pyrepr_list: "[" + (map("'" + . + "'") | join(", ")) + "]";
def pydump: "{" + (to_entries | sort_by(.key) | map("\(.key | tojson): \(.value | tojson)") | join(", ")) + "}";
def pybool: if . then "True" else "False" end;
def ce: "\(.path)@\(if has("decompressed_sha256") then .decompressed_sha256 else .sha256 end)";
def set_of: map({(.): true}) | add // {};
def basename: split("/") | last;
def oget($k): if type == "object" then .[$k] else null end;
# ---- v2.7 (09, GATES L3266-3284 @ 1bed0178a): constants for the ported-check rules (A)-(D) (PREDICTIONS_v27_addendum2). Each is checked against
# its source by run_screst_neg_v27.zsh (C0): K1 L3270, K2 L3272, K3 L3280, K4 L3274 + git rev-parse, K5 sc_v27.jq T6 literals, K6 sc_v27.zsh L16-17,
# K7 the v2.6 record (count + two checksums of the sorted entry key tuples; the tuples themselves are not embedded). ----
def V27_ALLOW: ["FINAL (INTERVALS REPRODUCED (EXACT); GATES L2869 @ 4d43ba0a4)","FINAL (INTERVALS REPRODUCED (EXACT); GATES L3160 @ f80f3cfe6)"];
def V27_PIO: "FINAL (printed items only)";
def V27_D: ["PRELIMINARY","caption-only, not plotted"];
def V27_CITE: {"line": 2982, "commit": "97b6e634da7f42e351ff0bbfbedc1f72e33efca0"};
def V27_DEP_SLOT_RE: "^(23|25)\\.";
def V27_DEP_STR_RE: "detector_vs_drop|panel_plant_rows|slots_routeC_writing_rc1b";
def V27_P18_SHA: "13d29fb90cac3e3e54145b5d9f41319614a0ff63ebf33b5b4de3d251419741b0";
def V27_P19_SHA: "a8bf8fced33aa6dc91be4a26f5cc862a0115af1c47b8acf1c80ca8f912757fda";
def V27_CK($b; $m): reduce (tojson | explode[]) as $c (0; ((. * $b) + $c) as $x | $x - ($m * (($x / $m) | floor)));
def V26_C3: {"C3a_g7b_and_cleanroom_gap":{"n":36,"h1":953965265,"h2":845360375},"C3b_not_cleanroom_verdict":{"n":47,"h1":490440513,"h2":544958330}};
# K8 (09's correction, GATES L3291 @ eca0ca69c): (B)(i-c)'s gate-input key (the token after "key == " on L3291)
def V27_GATE_KEY: "LABELS_171";

$REC[0] as $r
| ($ST | map({(.k): .v}) | add // {}) as $S
| ($ST | map({(.k): .sha}) | add // {}) as $SHA
| ($LN | map({(.k): .v}) | add // {}) as $L
| ($SH | map({(.k): .v}) | add // {}) as $G
| ($LT | map({(.k): .v}) | add // {}) as $T
| def blob($c; $p): ($c + ":" + $p) as $k | if $S | has($k) then $S[$k] else error("NOBLOB " + $k) end;
  def bsha($c; $p): $SHA[$c + ":" + $p];
  def line($an): $L[$an.commit + ":" + $an.file + ":" + ($an.line | tostring)];
  def shown($c): ($G[$c] // error("NOSHOW " + $c)) | sort;

# ---- L93-98 in-scope set ----
([$r.in_scope_records[] | ce, (select(.sidecar | truthy) | .sidecar | ce)] | unique) as $in_list
| ($in_list | set_of) as $INS
| ([($r.gate_inputs_no_values // [])[] | ce] | unique) as $gate_list
| ($gate_list | set_of) as $GS
| ([$r.in_scope_records[] | select(.name == "A")][0]) as $A
| $r.final_bar.g7b_value as $g
| blob($g.commit; $g.path) as $GJ
| $r.release_wording as $rw
| ([line($rw.source) // "" | match("Final wording: \"(.+?)\"\\."; "g") | .captures[0].string]) as $tmpl
| (($tmpl[0] // "") | gsub("N bootstrap draws"; "\($GJ.boot.n) bootstrap draws") | gsub("\\(seed s\\)"; "(seed \($GJ.boot.seed))")) as $filled
| ([line($rw.addendum_cite) // "" | match("Release text = L1917 verbatim \\+ \"([^\"]+)\""; "g") | .captures[0].string]) as $add
| $g.nofilter_source as $nf
| $g.aggregates_labels.ack_09 as $ack
| ($r.in_scope_outputs + $r.preliminary_outputs) as $outs
| (reduce ($r.figure_labels // [])[] as $fl ({}; .[$fl.file] = $fl.label)) as $labs

# ---- L89-133 ----
| [
    W($in_list == $r.in_scope_set; "in_scope_set != records + sidecars"),
    (("compute", "note", "prov") as $k | W(($r.final_bar.cleanroom[$k] | has_all(["path", "commit", "sha256"])); "clean-room \($k) citation missing")),
    W($GJ.record_sha256 == $A.decompressed_sha256 and ($GJ.record | endswith($A.commit)); "G7b record is not A"),
    W($GJ.boot.n == $rw.N.value and $GJ.boot.seed == $rw.s.value; "N / s"),
    W(($tmpl | length) == 1 and $tmpl[0] == $rw.template_verbatim; "L1917 template"),
    W(($add | length) == 1 and $add[0] == $rw.addendum; "addendum vs READ_LOG L2892"),
    W(line($rw.addendum_cite) == line($rw.addendum_cite_added_at); "L2892 changed since it was added"),
    W($rw.release_text == $filled + (if ($add | length) > 0 then $add[0] else "" end); "release_text"),
    W([line($nf.note_cite) // "" | match("\\(\"([^\"]+)\"\\)"; "g") | .captures[0].string] == [$nf.note]; "nofilter note vs READ_LOG L2893"),
    W(line($nf.note_cite) == line($nf.note_cite_added_at); "L2893 changed since it was added"),
    W(([$gate_list[] | select($INS[.])] | length) == 0; "a gate input is in the in-scope set"),
    W(($ack | type) == "object" and ((line($ack) // "") | contains("as_recorded_labels: CLOSED")); "ack_09 cite"),
    W(($GJ.aggregates.stated_gap | if type == "string" then contains($filled) else (index([$filled]) != null) end); "filled wording not in G7b stated_gap"),
    W($GJ.aggregates.stated_gap == $g.aggregates_labels.stated_gap; "stated_gap verbatim"),
    W(($GJ.aggregates.as_recorded_labels | length) == $g.aggregates_labels.as_recorded_labels_n; "as_recorded_labels_n"),
    W(($GJ.aggregates.reproduced_labels | length) == $g.aggregates_labels.reproduced_labels_n; "reproduced_labels_n")
  ] as $bad3
| ([$r | .. | objects | .[] | strings | select(startswith("PENDING"))] | length) as $n_pending

# ---- L135-148 printed items ----
| $r.printed_items as $pi
| (if ($pi | truthy) then (blob($pi.provenance.commit; $pi.provenance.path) | reduce .[] as $x ({}; .[$x.row | tostring] = $x)) else {} end) as $prow
# v2.5 (A2): rows 119-121's printed token and anchor come from v19 (the draft_scope source); v18 stays the pin for every other row
| ($r.row_rulings_v2_5.draft_scope // null) as $ds25
| (if $ds25 != null then (blob($ds25.printed_and_anchor_source.commit; $ds25.printed_and_anchor_source.path) | reduce .[] as $x ({}; .[$x.row | tostring] = $x)) else {} end) as $prow19
| (if ($pi | truthy) then [($r.in_scope_outputs + $r.preliminary_outputs)[] | select(.kind == "printed_items") | .items[]] | length else null end) as $n_printed
| (if ($pi | truthy) then
     [ ((["in_scope_outputs", "FINAL"], ["preliminary_outputs", "PRELIMINARY"]) as [$grp, $gst] | $r[$grp][] | select(.kind == "printed_items") as $o |
          ($o.items[] | . as $it | (if $ds25 != null and ($ds25.rows | index([$it.row])) != null then $prow19 else $prow end)[$it.row | tostring] as $pr |
             W($pr != null and all("item", "anchor", "key_path", "record", "record_sha256"; . as $k | $pr[$k] == $it[$k]); "printed item row \($it.row)")),
          W($o.status == $gst and $o.rows == [$o.items[].row]; "\($o.name) rows")),
       W(([$pi.all_rows[].row] | sort) == ($prow | keys | map(tonumber) | sort); "all_rows covers every provenance row"),
       W(([$pi.all_rows_status_counts[]] | add) == ($prow | length); "row status counts") ]
   else [] end) as $bad4

# ---- v2.7 (B) (09, GATES L3272-3277 @ 1bed0178a): the printed-items-only predicate, computed once; used by the status rule (A)/(B),
# $pre_split (B) and $fo (C). Only a non-printed-items output whose status is exactly V27_PIO is considered. (i) as corrected by 09 (GATES
# L3289-3291 @ eca0ca69c): (i-a) / (i-b) / (i-c); path@sha is mapped to the keys of the output's own .json inputs by sha256 + path
# (PREDICTIONS_v27_addendum2); (iii) is SC7 T6's (a)-(c) re-derived from the store's v18 / v19 blobs. ----
| ([$r.disclosures[] | select((.status_superseded_by | truthy) | not) | .id] | set_of) as $CUR
| ($r.printed_items.provenance // {}) as $pv18
| ((($r.row_rulings_v2_5 // {}).draft_scope // {}).printed_and_anchor_source // {}) as $pv19
| ((($pv18.commit | type) == "string") and (($pv19.commit | type) == "string")
   and bsha($pv18.commit; $pv18.path) == V27_P18_SHA and bsha($pv19.commit; $pv19.path) == V27_P19_SHA) as $pvok
| (if $pvok then blob($pv18.commit; $pv18.path) + blob($pv19.commit; $pv19.path) else [] end) as $PROWS
| [ $outs[] | select(.kind != "printed_items" and .status == V27_PIO) as $o
    | blob($o.s2_marker.commit; $o.s2_marker.path) as $m
    | ([$m.covered[] | select(($INS[.] or $GS[.]) | not)] | sort) as $outside
    | ([$m.covered[] | select($GS[.] and ($INS[.] | not))] | sort) as $gcov
    | [($o.files // [])[] | select(.path | endswith(".json"))] as $jf
    | (if ($jf | length) == 1 then blob($jf[0].commit; $jf[0].path) else {} end) as $OJ
    | [(($OJ | oget("inputs")) // {}) | to_entries[] | select((.value | type) == "object")] as $IE
    | def keys_of($ces): [$ces[] | split("@") as $pp | [$IE[] | select(($pp | length) == 2 and .value.sha256 == $pp[1] and .value.path == $pp[0]) | .key]];
      keys_of($outside + $gcov) as $mk
    | keys_of($gcov) as $mg
    | keys_of([($o.out_of_scope_inputs // [])[].input]) as $mo
    | (($o.out_of_scope_keys // []) | if type == "array" then . else [] end) as $ok5
    # (i-a) the covered inputs not in in_scope_records (gate inputs included), mapped to keys, == set(out_of_scope_keys)
    | ((($mk | length) > 0) and all($mk[]; length == 1) and (([$mk[][]] | sort) == ($ok5 | sort)) and (($ok5 | unique | length) == ($ok5 | length))) as $bia
    # (i-b) $outside == set(out_of_scope_inputs[].input) (the v2.6 check, restated here)
    | ($outside == ([($o.out_of_scope_inputs // [])[].input] | sort)) as $bib
    # (i-c) the one gate input's key == V27_GATE_KEY: exactly one covered gate input, it maps to V27_GATE_KEY, and V27_GATE_KEY is the one
    # out_of_scope_keys entry that is not an out-of-scope input's key
    | ((($mg | length) == 1) and ($mg[0] == [V27_GATE_KEY]) and all($mo[]; length == 1)
       and (([$ok5[] | select(. as $k | [$mo[][]] | index([$k]) == null)] | sort) == [V27_GATE_KEY])) as $bic
    | ((($o.ruling_cite | oget("line")) == V27_CITE.line) and (($o.ruling_cite | oget("commit")) == V27_CITE.commit)) as $bii
    | (if ($jf | length) == 1 then $jf[0].path + "@" else null end) as $pref
    | (($pref != null) and $pvok
       and (([$PROWS[] | select(((.record | type) == "string") and (.record | startswith($pref)) and ((.key_path | type) == "array")
                                and ((.key_path[1] // "") | tostring | test(V27_DEP_SLOT_RE)))] | length) == 0)
       and (([$PROWS[] | select(((.record | type) == "string") and (.record | startswith($pref)))] | length) >= 1)
       and (([$r.element_tables[] | .. | strings | select(test(V27_DEP_STR_RE))] | length) == 0)) as $biii
    | {name: $o.name, ia: $bia, ib: $bib, ic: $bic, ii: $bii, iii: $biii} ] as $PIOR
| ([$PIOR[] | select(.ia and .ib and .ic and .ii and .iii) | .name] | set_of) as $PIO
| [ $PIOR[] | W(.ia; "\(.name) \(V27_PIO): (B)(i-a) the covered inputs outside in_scope_records (gate inputs included), mapped to keys, differ from out_of_scope_keys"),
              W(.ib; "\(.name) \(V27_PIO): (B)(i-b) the outside inputs differ from out_of_scope_inputs"),
              W(.ic; "\(.name) \(V27_PIO): (B)(i-c) the gate input's key is not \(V27_GATE_KEY)"),
              W(.ii; "\(.name) \(V27_PIO): (B)(ii) ruling_cite is not GATES L2982 @ 97b6e634d"),
              W(.iii; "\(.name) \(V27_PIO): (B)(iii) T6_zero_printed_dependence is false") ] as $bad27b
| [ $PIOR[] | "V27B \(.name): i_a \(.ia) i_b \(.ib) i_c \(.ic) ii_cite \(.ii) iii_T6_zero_printed_dependence \(.iii)" ] as $lines27b

# ---- L149-178 outputs ----
| [ (["in_scope_outputs", "FINAL"], ["preliminary_outputs", "PRELIMINARY"]) as [$grp, $status]
    | $r[$grp][] | select(.kind != "printed_items") as $o
    | if ($o.pending_downstream_label | truthy) and $o.kind != "figure_split" then
        blob($o.s2_marker.commit; $o.s2_marker.path) as $m
        | {kind: "pending", bad: [
            W(([$m.covered[] | select(($INS[.] or $GS[.]) | not)] | length) == 0; "\($o.name) pending: COVER closure"),
            W($status == "PRELIMINARY" and $o.status == "PRELIMINARY" and $o.reason == $o.pending_downstream_label; "\($o.name) pending-downstream status / reason"),
            W(shown($o.commit) == ([$o.files[].path] | sort); "\($o.name) files") ]}
      elif $o.kind == "figure_split" then
        blob($o.s2_marker.commit; $o.s2_marker.path) as $m
        | ((if any($o.files[]; $labs[.path] == "PRELIMINARY") then "PRELIMINARY" else "FINAL" end)) as $want
        | {kind: "split", bad: [
            W(([$m.covered[] | select(($INS[.] or $GS[.]) | not)] | length) == 0; "\($o.name) split: COVER closure"),
            W($o.status == $want and $want == $status; "\($o.name) split status"),
            # v2.7 (A) extension (09, GATES L3295 @ eca0ca69c): a FINAL split output's interval_rule is allow-listed with current disclosures
            W((($o.interval_rule | truthy) | not) or (($o.status | startswith("FINAL")) | not)
              or ((($o.interval_rule | type) == "object") and (V27_ALLOW | index([$o.interval_rule.rule]) != null)
                  and all(($o.interval_rule.disclosures // [])[]; $CUR[.] == true));
              "\($o.name) split: a FINAL interval_rule is not allow-listed or names a disclosure that is not current (v2.7 (A))") ]}
      else
        blob($o.s2_marker.commit; $o.s2_marker.path) as $m
        | blob($o.cover.commit; $o.cover.path) as $cov
        | ([$m.covered[] | select(($INS[.] or $GS[.]) | not)] | sort) as $outside
        | {kind: "normal", bad: [
            W($m.signed == true and $m.by == "09" and ($cov | sort) == ($m.covered | sort); "\($o.name) marker / COVER"),
            W(($m.for | if type == "string" then contains($o.cover.sha256[:8]) else (index([$o.cover.sha256[:8]]) != null) end); "\($o.name) COVER sha not named in the marker"),
            W(([$m.covered[] | select($GS[.])] | sort) == ($o.covered_gate_inputs_no_values | sort); "\($o.name) gate inputs"),
            # v2.7 (A) + (B) (09, GATES L3270 / L3272 @ 1bed0178a): an interval_rule gives FINAL only with an allow-listed rule string and
            # only current disclosures; outside inputs give FINAL (printed items only) only for a $PIO output; the group follows the prefix.
            W((if ($outside | length) == 0 then
                 (if ($o.interval_rule | truthy) then
                    (if (($o.interval_rule | type) == "object") and (V27_ALLOW | index([$o.interval_rule.rule]) != null)
                        and all(($o.interval_rule.disclosures // [])[]; $CUR[.] == true) then "FINAL" else "PRELIMINARY" end)
                  else "FINAL" end)
               elif $PIO[$o.name] == true then V27_PIO
               else "PRELIMINARY" end) as $want
              | ($want == $o.status) and ((if ($want | startswith("FINAL")) then "FINAL" else "PRELIMINARY" end) == $status);
              "\($o.name) status (v2.7: (A) interval-rule allow-list; (B) printed items only)"),
            W($outside == ([($o.out_of_scope_inputs // [])[].input] | sort); "\($o.name) out-of-scope inputs"),
            W(($m.covered | length) == $o.covered_n; "\($o.name) covered_n"),
            W(shown($o.commit) == ([$o.files[].path] | sort); "\($o.name) files") ]}
      end ] as $outres
| ([$outres[] | .kind] | length) as $n_reclass
| ([$outres[] | select(.kind == "normal")] | length) as $n_cov_out

# ---- L180-191 records ----
| ([$r.out_of_scope_records[] | ce, (select(.sidecar | truthy) | .sidecar | ce)]) as $oos_entries
| [ ($r.in_scope_records | map(. + {_ins: true})) + ($r.out_of_scope_records | map(. + {_ins: false})) | .[] | select(has("s2_marker")) as $rr
    | blob($rr.s2_marker.marker.commit; $rr.s2_marker.marker.path) as $m
    | ($rr | del(._ins) | ce) as $own
    | ([$m.covered[] | select((($INS[.] or $GS[.]) | not) and . != $own)] | sort) as $outside
    | {bad: [
        W((blob($rr.s2_marker.cover.commit; $rr.s2_marker.cover.path) | sort) == ($m.covered | sort); "\($rr.name) COVER"),
        (if $rr._ins then W(($outside | length) == 0; "in-scope record \($rr.name) has COVER inputs outside scope")
         else W($outside == ([$rr.s2_marker.cover_inputs_outside_in_scope[].input] | sort); "\($rr.name) outside list") end) ]} ] as $recres
| [ W(([$oos_entries[] | select($INS[.] or $GS[.])] | length) == 0; "an out-of-scope record is in the in-scope or gate-input set") ] as $bad6a

# ---- L192-220 derivation; sensitivity rows ----
| $r.derivation_check.result as $dres
| (if ($dres | type) == "object" then $dres else $r.derivation_check.result_available_but_not_used end) as $dsrc
| blob($dsrc.commit; $dsrc.path) as $D
| $D.other_labels as $labs7
# v2.7 (B) (GATES L3276 @ 1bed0178a): a $PIO output leaves $pre_split
| ([($r.in_scope_outputs[] | select(.kind != "printed_items" and ($PIO[.name] != true))), ($r.preliminary_outputs[] | select(.kind == "figure_split" or (.pending_downstream_label | truthy) or (.interval_rule | truthy))) | .split_from // .name] | unique) as $pre_split
| ((($r.sensitivity_halves // {}) | .flipped // []) | set_of) as $FL7
| [ $r.sensitivity_rows.per_output[] as $so
    | ([$outs[] | select((.files | truthy) and (.name == $so.output or .split_from == $so.output))]) as $parts
    | (reduce ($parts[] | .files[]) as $f ({}; .[$f.path] = $f)) as $files
    | [ $files[] | select(.path | endswith(".json")) | blob(.commit; .path) | select((oget("slots") | type) == "object") | .slots | keys_unsorted[]
        | . as $k | select(any($labs7[]; . as $lb | $k | contains($lb))) ] as $got
    | {n: ($got | length), bad: [
        W(($got | sort) == ([$so.entries[].slot] | sort); "\($so.output) sensitivity slots"),
        W(all($so.entries[]; if $FL7[.slot] then (.status == "FINAL" and .entry == "sensitivity row; reproduced by both halves")
                                   else (.status == "PRELIMINARY" and .entry == "sensitivity row; not reproduced") end); "\($so.output) entry labels") ]} ] as $sensres
| [ (if $r.variant == "pass" then W(($dres | type) == "object"; "pass variant without a derivation result") else empty end),
    (if ($dres | type) == "object" then
       W($D.pass == true and $D.n_violations == 0 and $D.record_sha256 == $A.decompressed_sha256; "derivation result gates"),
       W(($D.compared_labels | sort) == ([$GJ.pooled[].row] | unique); "derivation compared labels != G7b pooled rows")
     else empty end),
    W($r.sensitivity_rows.labels == $labs7; "sensitivity labels"),
    W(([($r.not_reproduced[] | select(.label | truthy) | .label), ((($r.reproduced_since_v1 // {}) | .entries // [])[] | select(.label | truthy) | .label)] | sort) == ($labs7 | sort);
      "each sensitivity label is listed once, in not_reproduced or in reproduced_since_v1"),
    W(([$r.sensitivity_rows.per_output[].output] | sort) == $pre_split; "sensitivity rows cover the FINAL outputs (before figure splits)")
  ] as $bad7

# ---- L221-253 exclusions ----
# v2.7 (C) (GATES L3278 @ 1bed0178a): a $PIO output is not scanned for exclusion entries (as in v2.6, where rc1b was PRELIMINARY)
| (reduce (($r.in_scope_outputs[] | select(.kind != "printed_items" and ($PIO[.name] != true))), ($r.preliminary_outputs[] | select(.kind == "figure_split" or (.interval_rule | truthy)))) as $o ({};
     reduce ($o.files[] | select(.path | endswith(".json"))) as $f (.; .[[($o.split_from // $o.name), $f.path] | tojson] = {on: ($o.split_from // $o.name), f: $f}))) as $fo
| [ ($r.exclusions // [])[] as $ex
    | [ $fo[] as $x
        | select((($ex.outputs | truthy) | not) or ([$ex.outputs[] | select(. == $x.on)] | length) > 0)
        | blob($x.f.commit; $x.f.path) as $J
        | ( if (($J | oget("slots")) | type) == "object" and ($ex.slot_regex | truthy) then
              (($J | oget("slots")) | keys_unsorted[] | select(rsearch($ex.slot_regex)) | [$x.on, $x.f.path, "slot", .])
            elif (($J | oget("slots")) | type) == "object" then
              ( if ($ex.all_slots | truthy) then
                  (($J | oget("slots")) | keys_unsorted[] | . as $k | select(any(($ex.exclude_slot_prefixes // [])[]; . as $x | $k == $x or ($k | startswith($x + "."))) | not) | [$x.on, $x.f.path, "slot", $k])
                elif ($ex.slot_prefixes | truthy) then
                  (($J | oget("slots")) | keys_unsorted[] | . as $k | select(any($ex.slot_prefixes[]; . as $x | $k | startswith($x + "."))) | [$x.on, $x.f.path, "slot", $k])
                else empty end )
            else empty end ),
          ( if (($J | oget("values")) | type) == "array" and (($ex.all_values | truthy) or ($ex.value_key_regex | truthy)) then
              (($J | oget("values")) | to_entries[] | select(($ex.all_values | truthy) or ((.value.key | map(tostring) | join("/")) | rsearch($ex.value_key_regex))) | [$x.on, $x.f.path, "value", .key])
            else empty end ) ] as $got0
    | (if ($ex.gate_slot_regex | truthy) then [$got0[] | select(.[2] == "slot" and (.[3] | rsearch($ex.gate_slot_regex)))] else [] end) as $gate_got
    | ([$gate_got[] | tojson] | set_of) as $GG
    | (if ($ex.gate_slot_regex | truthy) then [$got0[] | select($GG[tojson] | not)] else $got0 end) as $got
    | [ $ex.entries[] | [.output, .file, (if has("slot") then "slot" else "value" end), (if has("slot") then .slot else .value_index end)] ] as $mine
    | ((($ex.halves // {}) | length) > 0 and all(($ex.halves // {})[]; type == "object")) as $ready
    | ([$ex.entries[] | (if (.slot | truthy) then .slot else .key_path end)] | unique) as $ids
    | (if $ready then
         reduce ([$ex.halves[] | if .covered == null then $ids else (.covered | unique) end])[] as $h (null; if . == null then $h else [.[] as $i | select($h | index([$i]) != null) | $i] end)
       else [] end | set_of) as $COV
    | {n: ($got | length), bad: [
        (if ($ex.gate_slot_regex | truthy) then W(([($ex.gate_entries // [])[] | [.output, .slot] | tojson] | sort) == ([$gate_got[] | [.[0], .[3]] | tojson] | sort); "exclusion \($ex.id) gates") else empty end),
        W(([$got[] | tojson] | sort) == ([$mine[] | tojson] | sort); "exclusion \($ex.id) entries"),
        # v2.7 (C) outcome check (GATES L3278 @ 1bed0178a): C3a / C3b scan results equal v2.6's (count + key set, via V26_C3's two checksums)
        (if (V26_C3 | has($ex.id)) then
           ([$got[] | tojson] | sort) as $gk
           | W((($gk | length) == V26_C3[$ex.id].n) and (($gk | V27_CK(131; 2147483647)) == V26_C3[$ex.id].h1) and (($gk | V27_CK(257; 1000000007)) == V26_C3[$ex.id].h2);
               "exclusion \($ex.id) entries differ from v2.6's (count + key set; v2.7 (C))")
         else empty end),
        W(all($ex.entries[]; (.status == "FINAL") == (($COV[(if (.slot | truthy) then .slot else .key_path end)]) == true)); "exclusion \($ex.id) flip by intersection") ]} ] as $exres

# ---- L254-261 element tables ----
| [ ($r.element_tables // [])[] | select(.kind != "figs78_item") as $et
    | blob($et.commit; $et.file) as $J
    | W(($J.values | length) == ($et.elements | length); "element table \($et.file) size"),
      ($J.values | to_entries[] | .key as $i | .value as $v | ("\($v.input):\($v.key | map(tostring) | join("/"))") as $s
       | [$et.classes[] | select(. as $c | $s | rsearch($c.regex))] as $hit
       | W(($hit | length) == 1 and $hit[0].class == $et.elements[$i].class; "element \($i) class")),
      W($et.figure_status == (if any($et.elements[]; .status == "PRELIMINARY") then "PRELIMINARY" else "FINAL" end); "figure status") ] as $bad9
# ---- v2.5 branch (i): the figs78_item tables (854f901b6's schema; v2.3 had none). Their equality to the 854f901b6 record is SC5 S9i-c. ----
| [ ($r.element_tables // [])[] | select(.kind == "figs78_item") as $et
    | W(bsha($et.commit; $et.file) == $et.sha256; "figs78 element table source sha"),
      W(($et | b1ok(blob($et.commit; $et.file))); "figs78 element table (b1) pin"),
      W(($et.elements | map(.id) | unique | length) == ($et.elements | length); "figs78 element table ids"),
      W(all($et.classes | to_entries[]; .key as $c | .value as $cv | [$et.elements[] | select(.class == $c)] as $ec
            | $cv.n == ($ec | length) and all($cv | keys[] | select(. != "n"); . as $s | $cv[$s] == ([$ec[] | select(.status == $s)] | length)))
        and ([$et.elements[].class] | unique) == ($et.classes | keys)
        and all($et.elements[]; . as $e | $et.classes[$e.class] | has($e.status)); "figs78 element table class counts"),
      W($et.figure_status == (if any($et.elements[]; .status == "PRELIMINARY") then "PRELIMINARY" else "FINAL" end); "figure status"),
      W($et.elements_all_final == all($et.elements[] | select(.status != "disclosure only (not drawn)"); .status == "FINAL"); "figs78 element table elements_all_final") ] as $bad9b

# ---- L262-380 per-quantity B / R37 ----
| [ ($r.exclusions // [])[] | select(.entry_parsers | truthy) as $ex
    | (reduce $ex.entries[] as $e ({p: {}, bad: []};
         ((if ($e.slot | truthy) then $e.slot else $e.key_path end)) as $ident
         | [ $ex.entry_parsers[] as $ps | select($ps.output == null or $ps.output == $e.output)
             | ($ident | rmatch($ps.regex)) as $m | select($m != null)
             | [$ps.record, $ps.kind, $m.L, $m.mode, (if ($m.lv | truthy) then ($m.lv | tonumber | . + 0) else null end), ($ps.quantity | fmt($m)), $m.origin] ] as $h
         | .bad += [W(($h | length) <= 1; "\($ident) parses twice")]
         | .p[$ident] = (if ($h | length) > 0 then $h[0] else null end))) as $PR
    | $PR.p as $parsed
    # a1 half
    | (reduce ($ex.halves.a1_B_R37.union_of[] | select((.held | truthy) | not)) as $af ({ok: {}, bad: []};
         if $af.item_schema == "printed_map" then
           blob($af.commit; $af.path) as $MJ
           | .bad += [W(bsha($af.commit; $af.path) == $af.sha256; "a1 printed map sha")]
           | (reduce $ex.entries[] as $e ({names: {}, bad: []};
                ((if ($e.slot | truthy) then $e.slot else $e.key_path end)) as $ident
                | (reduce ($af.value_name_rules // [])[] as $rl ((if ($e.slot | truthy) then $ident else null end);
                     ($ident | rmatch($rl.regex)) as $mm | if $mm != null and $rl.output == $e.output then ($rl.name | fmt($mm)) else . end)) as $nm
                | if $nm != null then (.bad += [W((.names | has($nm)) | not; "printed map name \($nm) maps twice")] | .names[$nm] = $ident) else . end)) as $NM
           | .bad += $NM.bad
           | reduce $MJ[$af.covered_key][] as $it (.;
               .bad += [W($NM.names | has($it[$af.name_field]); "printed map item \($it[$af.name_field]) matches no entry")]
               | if $it[$af.covered_field] == true and ($NM.names | has($it[$af.name_field])) then .ok[$NM.names[$it[$af.name_field]]] = true else . end)
         else
           blob($af.commit; $af.path) as $AJ
           | .bad += [W(bsha($af.commit; $af.path) == $af.sha256; "a1 list sha")]
           | (reduce $AJ.slots[] as $it ({items: {}, bad: []};
                ([$it.kind, $it.label, $it.quantity] | tojson) as $k
                | .bad += [W((.items | has($k)) | not; "a1 duplicate \($k)")] | .items[$k] = $it)) as $IT
           | .bad += $IT.bad
           | reduce ($parsed | to_entries[] | select(.value != null and .value[0] == $af.record)) as $pp (.;
               $IT.items[[$pp.value[1], $pp.value[2], $pp.value[5]] | tojson] as $it
               | if ($it | truthy) and $it[$af.covered_field] == true
                    and ([$it.modes_levels[] | split("|") | [.[0], (.[1] | tonumber | . + 0)] | tojson] | index([[$pp.value[3], $pp.value[4]] | tojson]) != null)
                 then .ok[$pp.key] = true else . end)
         end)) as $A1
    # G7b half
    | (reduce $ex.halves.g7b_B_R37.union_of[] as $gp ({ok: {}, bad: []};
         (reduce $gp.files[] as $f ({d0: {}, bt: {}, rl: {}, rx: {}, bad: []};
            blob($f.commit; $f.path) as $J
            | .bad += [W(bsha($f.commit; $f.path) == $f.sha256; "G7b ledger file sha")]
            | reduce ($J.rows // [])[] as $rb (.;
                if (($rb.origin | if type == "string" then . elif . == null then "None" else tostring end) | contains("[pooled: \($gp.combo)]")) | not then .
                else reduce $rb.ledger[] as $kv (.;
                  .rx[[$rb.row, $rb.origin, $kv[0]] | tojson] += [$kv[1]]
                  | ($kv[0] | index("|")) as $i
                  | if $i == null then . else (($kv[0][:$i]) | pyfloat) as $fl
                      | if $fl == null then . else .rl[[$rb.row, $fl, $kv[0][$i + 1:]] | tojson] += [$kv[1]] end end)
                end)
            | reduce $J.pooled[] as $pb (.;
                if $pb.combo != $gp.combo then . else
                reduce $pb.ledger[] as $kv (.; ($kv[0] | split("|")) as $ks
                  | if ($ks | length) == 4 and ($ks[1] == "joint" or $ks[1] == "fixed") then
                      (($ks[0] | pyfloat) as $fl | if $fl == null then . else .d0[["pooled", $pb.row, $ks[1], $fl, $ks[2] + "." + $ks[3]] | tojson] += [$kv[1]] end)
                    elif $ks[0] == "boot" and ($ks | length) >= 7 and $ks[4] == "pooled" then
                      (($ks[2] | pyfloat) as $fl | if $fl == null then . else .bt[["pooled", $pb.row, $ks[1], $fl, $ks[-2] + "." + $ks[-1]] | tojson] += [$kv[1]] end)
                    else . end)
                end)
            | reduce $J.aggregates.ledger[] as $kv (.;
                ([$kv[0] | capture("^(?<a>[^|]+)\\|(?<b>joint|fixed)\\|(?<c>[0-9.e-]+)\\|coh\\|(?<d>.+)\\|(?<e>[^|]+)$")] | .[0]) as $mm
                | if $mm != null and $mm.a == $gp.combo then .d0[["coheadline_signed", $mm.d, $mm.b, ($mm.c | tonumber | . + 0), $mm.e] | tojson] += [$kv[1]] else . end))) as $LG
         | .bad += $LG.bad
         | reduce ($parsed | to_entries[] | select(.value != null and .value[0] == $gp.record)) as $pp (.;
             $pp.value as $pe
             | if $pe[1] == "member_rows" or $pe[1] == "member_row_exact" then
                 [ $pe[5] | split("+")[] as $q
                   | if $pe[1] == "member_rows" then ($LG.rl[[$pe[2], $pe[4], $q] | tojson] // []) else ($LG.rx[[$pe[2], $pe[6], $q] | tojson] // []) end ] as $sts
                 | if all($sts[]; (length > 0) and all(.[]; . == "MATCH")) then .ok[$pp.key] = true else . end
               else
                 ([$pe[1], $pe[2], $pe[3], $pe[4], $pe[5]] | tojson) as $key
                 | if (($LG.d0[$key] // []) | length) > 0 and all($LG.d0[$key][]; . == "MATCH") and all(($LG.bt[$key] // [])[]; . == "MATCH") then .ok[$pp.key] = true else . end
               end))) as $G7
    | (reduce $ex.entries[] as $e ({per: {}, bad: []};
         ((if ($e.slot | truthy) then $e.slot else $e.key_path end)) as $ident
         | ($e.record // "?") as $rk
         | ($A1.ok[$ident] == true) as $ina | ($G7.ok[$ident] == true) as $ing
         | .per[$rk] |= ((. // {printed: 0, a1: 0, g7b: 0, both: 0, FINAL: 0, PRELIM: {}})
              | .printed += 1 | .a1 += (if $ina then 1 else 0 end) | .g7b += (if $ing then 1 else 0 end) | .both += (if $ina and $ing then 1 else 0 end))
         | if $e.status == "FINAL" then
             (.per[$rk].FINAL += 1 | .bad += [W($ina and $ing; "FINAL key \($ident) fails a condition (a1 \($ina | pybool), G7b \($ing | pybool))")])
           else
             ((if ($ina or $ing) | not then "neither" elif ($ina | not) then "not in a1" else "not G7b-compared" end) as $why
              | .per[$rk].PRELIM[$why] += 1
              | .bad += [W((($ina and $ing) | not) or $ex.halves == null; "\($ident) meets both conditions but is not FINAL")])
           end)) as $C
    | {bad: ($PR.bad + $A1.bad + $G7.bad + $C.bad),
       lines: ([$C.per | to_entries | sort_by(.key)[] | .key as $rk | .value as $c
                | "PERQ \($ex.id) \($rk): printed \($c.printed) a1_scope \($c.a1) g7b_compared \($c.g7b) both \($c.both) FINAL \($c.FINAL) PRELIMINARY \([$c.PRELIM[]] | add // 0) by_reason \($c.PRELIM | pydump)"]
               + ["PERQ \($ex.id) gates (disclosure only): \(($ex.gate_entries // []) | length)"])} ] as $perq

# ---- L382-408 families ----
| [ ($r.exclusions // [])[] as $ex
    | $ex.halves as $hv
    # v2.7 (D) (GATES L3280-3282 @ 1bed0178a): with no halves, every entry status starts with V27_D[0] or is exactly V27_D[1]; message wording fixed
    | if ($hv | truthy) | not then {lines: [], bad: [W(all($ex.entries[]; ((.status | type) == "string") and ((.status | startswith(V27_D[0])) or .status == V27_D[1]));
        "\($ex.id): no halves, and an entry status neither starts with PRELIMINARY nor is exactly caption-only, not plotted (v2.7 (D))")]}
      else
        { lines: ["FAMILY \($ex.id): entries \($ex.entries | length) FINAL \([$ex.entries[] | select(.status == "FINAL")] | length) PRELIMINARY \([$ex.entries[] | select(.status != "FINAL")] | length) halves \($hv | keys | pyrepr_list)"],
          bad: [
            ($hv | to_entries[] | .key as $hn | .value as $h | select(($h | type) == "object" and $h.item_schema == "family_all_labels")
             | blob($h.commit; $h.path) as $FJ
             | (reduce $FJ.slots[] as $it ({}; .[$it.family] += [$it]) | map_values(all(.[]; .covered == true))) as $okf
             | W($okf == $h.families_covered_for_all_labels; "\($ex.id) \($hn) family coverage"),
               ($ex.entries[] | (if (.slot | truthy) then .slot else .key_path end) as $ident
                | [$h.family_map[] | select(. as $fm | $ident | rsearch($fm.regex))] as $mf
                | ((($mf | length) > 0) and all($mf[0].families[]; $okf[.] == true)) as $cv
                | W($cv == ($h.covered | index([$ident]) != null); "\($ex.id) \($hn) \($ident) family re-derivation"))),
            ($ex.entries[] | (if (.slot | truthy) then .slot else .key_path end) as $ident
             | all($hv[]; (type == "object") and (.covered == null or (.covered | index([$ident]) != null))) as $allc
             | if .status == "FINAL" then W($allc; "FINAL key \($ident) (\($ex.id)) is not covered by every half")
               elif ((.reason // "") | startswith("G7b derivations disagree")) | not then W($allc | not; "\($ident) (\($ex.id)) is covered by every half but not FINAL")
               else empty end) ] }
      end ] as $fam

# ---- L410-416 SENS ----
| $r.sensitivity_rows as $sr
| (if ($sr | truthy) then
     ([$sr.per_output[].entries[]]) as $sens
     | ($r.sensitivity_halves // {}) as $sh
     | (($sh.halves // {}) | map_values(if type == "object" then "covered \((.covered_slots // []) | length)" else .[:40] end)) as $states
     | {lines: ["SENS entries \($sens | length) FINAL \([$sens[] | select(.status == "FINAL")] | length) PRELIMINARY \([$sens[] | select(.status != "FINAL")] | length) halves \($states | pydump)"],
        bad: [W(all($sens[] | select(.slot as $s | ((($sh.flipped // []) | index([$s])) != null) | not); .status == "PRELIMINARY"); "a sensitivity slot is FINAL without both halves")]}
   else {lines: [], bad: []} end) as $sens12

# ---- L418-454 R-1, R-2, v2.1 ----
| (reduce ($outs[] | select(.kind == "figure_split")) as $o ({}; .[[$o.split_from, $o.commit] | tojson] += [$o.files[].path])) as $splits
| [ ($r.in_scope_outputs[] as $o | W((any(($o.files // [])[]; $labs[.path] == "PRELIMINARY")) | not; "FINAL output \($o.name) names a PRELIMINARY figure")),
    ($splits | to_entries[] | (.key | fromjson) as [$nm, $c] | W((.value | unique) == ($G[$c] // [] | unique); "split \($nm): parts' files == the commit's files")),
    (if ($r.sensitivity_rows | truthy) and $r.sensitivity_halves != null then
       ([$r.sensitivity_rows.per_output[].entries[]] | length) as $all
       | ($r.sensitivity_halves.flipped | length) as $fl
       | $r.sensitivity_rows.status as $st
       | W((if $fl == $all and $all > 0 then ($st | startswith("FINAL")) elif $fl > 0 then ($st | startswith("MIXED")) else $st == "PRELIMINARY" end); "sensitivity_rows.status contradicts sensitivity_halves"),
         W(($r.sensitivity_halves.still_preliminary | sort) == ([$r.sensitivity_rows.per_output[].entries[] | select(.status != "FINAL") | .slot] | sort); "sensitivity still_preliminary list")
     else empty end),
    (if ($r.printed_items | truthy) and $r.sensitivity_halves != null then
       ($r.sensitivity_halves.flipped | set_of) as $flip
       | ([$r.sensitivity_rows.per_output[].entries[].slot] | unique) as $sslots
       | ($sslots | set_of) as $SS
       | ($r.printed_items.all_rows[] | . as $row | $prow[$row.row | tostring].key_path as $kp
          | (if ($kp | type) == "array" and ($kp | length) > 1 and ($kp[0] == "slots" or $kp[0] == "full") then $kp[1] else null end) as $sl
          | ((($r.row_rulings_v2_4 // {}) | .r_node_switch // {}) as $sw | ($sw.state == "OFF" and ($sw.superseded_by_ruling == null) and (($sw.rows_v2_3_corrections // []) | index([$row.row]) != null))) as $corr   # v2.6 (09 Q4): held only until superseded_by_ruling
          | if $sl != null and $SS[$sl] then W(($row.status | startswith("FINAL")) == (($flip[$sl] == true) and ($corr | not)); "printed row \($row.row) on sensitivity slot \($sl): status vs sensitivity_halves") else empty end),
         ([$r.sensitivity_rows.labels[] as $lb | select(all($sslots[] | select(contains($lb)); $flip[.] == true) and any($sslots[]; contains($lb))) | $lb] | set_of) as $flabs
         | W(([($r.not_reproduced // [])[] | select($flabs[.label // ""] == true)] | length) == 0; "not_reproduced names a label whose printed slots are all FINAL")
     else empty end),
    (if (($r.sensitivity_rows // {}) | .status // "" | startswith("FINAL")) then W(($r.sensitivity_rows.entry | contains("not reproduced")) | not; "sensitivity_rows.entry says not reproduced while the status is FINAL") else empty end),
    W(all(($r.figure_labels // []) | group_by(.output)[]; ([.[].label] | unique | length) == 1); "a figure_labels output name groups FINAL with PRELIMINARY figures")
  ] as $bad13

# ---- L456-509 pending downstream ----
| $r.pending_downstream as $pdn
| (if ($pdn | truthy) then
     $pdn.label as $PL
     | ([$r.exclusions[] | select(.id == "downstream_pending_decision2")][0]) as $pex
     | (($pdn.printed_point_only_final_rows // {}) | .sim_slot_label) as $simL
     | {lines: ["PENDING_DOWNSTREAM slots \([$pex.entries[] | select(.status == "PRELIMINARY")] | length)/\($pex.entries_n) outputs \($pdn.outputs | length) figures \($pdn.figure_files | length) printed_rows \($pdn.printed_rows | length) recomputed \(($pdn.recomputed // {}) | length) not_downstream \(((($pdn.not_downstream // {}) | .slots) // []) | length)"],
        bad: [
          W(([$pex.entries[].slot] | sort) == ($pdn.rc1a_slots | sort) and ($pdn.rc1a_slots | length) == $pex.entries_n; "the pending slots == the exclusion's entries"),
          (if ($pex.slot_regex | truthy) then
             ($r.in_scope_outputs[] | select(.name == "rc1a") | .files[0]) as $f1
             | W(([blob($f1.commit; $f1.path).slots | keys_unsorted[] | select(rsearch($pex.slot_regex))] | sort) == ($pdn.rc1a_slots | sort); "pending slots re-derived by the exclusion's regex")
           else empty end),
          W(all($pex.entries[]; .status == "PRELIMINARY" and .entry == .reason and (.entry == $PL or (($simL | truthy) and (.slot | endswith(".sim")) and .entry == $simL)));
            "each pending slot is PRELIMINARY with the exact label (sim slots: the point-part label)"),
          ($pdn.outputs[] as $nm
           | W(([$r.in_scope_outputs[] | select(.name == $nm)] | length) == 0; "\($nm) must not be a FINAL output"),
             ([$r.preliminary_outputs[] | select(.name == $nm)] | .[0]) as $o
             | W($o != null and $o.status == "PRELIMINARY" and $o.reason == $PL; "\($nm) PRELIMINARY with the exact label")),
          (($r.figure_labels // [])[] | select(.file as $ff | $pdn.figure_files | index([$ff]) != null)
           | W(.label == "PRELIMINARY" and .reason == $PL; "figure \(.file | basename): PRELIMINARY with the exact label")),
          W(($pdn.figure_files | sort) == ([$r.preliminary_outputs[] | select(.name as $n | $pdn.outputs | index([$n]) != null) | .files[].path | select(endswith("_source.json"))] | sort);
            "pending figure files == the pending outputs' figure sources"),
          ($r.in_scope_outputs[] | select(.downstream_tag | truthy) as $o | $o.downstream_tag as $dt
           | ([$pex.entries[] | select(.status == "PRELIMINARY") | .slot]) as $held
           | ((($dt.recomputed_slots // {}) | keys)) as $rec
           | if $dt.scope == "output" then
               ([$o.files[].path | select(endswith("_source.json"))]) as $srcs
               | (reduce ($r.element_tables // [])[] as $t ({}; .[$t.file] = $t)) as $ets
               # v2.5 (09, finding 2 (b1)): a figs78_item source table passes by the pinned b1ok form; every other table keeps the v2.3 all-elements rule
               | W(($dt.recomputed_output | truthy) and all($srcs[]; . as $x | ($ets | has($x))
                     and (if $ets[$x].kind == "figs78_item" then ($ets[$x] | b1ok(blob($ets[$x].commit; $ets[$x].file))) else all($ets[$x].elements[]; .status == "FINAL") end));
                   "FINAL output \($o.name) reads a downstream record without a recorded recompute and full FINAL element tables")
             else
               W($dt.scope == "slots" and all(($dt.slots // [])[]; . as $s | ($held + $rec) | index([$s]) != null); "FINAL output \($o.name) reads a pending downstream record")
             end),
          ($r.printed_items.all_rows[] | . as $row | $prow[$row.row | tostring].key_path as $kp
           | if ($kp | type) == "array" and ($kp | length) > 1 and $kp[0] == "slots" and ($pdn.rc1a_slots | index([$kp[1]]) != null) then
               (if (((($pdn.printed_point_only_final_rows // {}) | .rows) // []) | index([$row.row])) != null then
                  W(($row.status | startswith("FINAL")) and ($kp[1] | endswith(".sim")); "printed row \($row.row): point-only FINAL only for a sim slot")
                else W($row.status | startswith("PRELIMINARY"); "printed row \($row.row) on pending slot \($kp[1]) is not PRELIMINARY") end)
             else empty end),
          (($r.element_tables // [])[] | .elements[] | select(.class | startswith("bar: four-probe (group_excess)"))
           | if ($pdn.fig5_elements | truthy) then W(.status == "PRELIMINARY" and .label == $PL; "Fig 5 four-probe bar element pending")
             else W(.status == "FINAL" and (($pdn.recomputed // {}) | has("Fig 5 four-probe bar")); "Fig 5 four-probe bar FINAL only via a recorded recompute") end),
          ($r.printed_items.all_rows[] | . as $row | $prow[$row.row | tostring].key_path as $kp
           | if ($kp | type) == "array" and ($kp | length) > 1 and $kp[0] == "slots"
                and (((((($pdn.not_downstream // {}) | .slots) // []) + (($pdn.recomputed // {}) | keys)) | index([$kp[1]])) != null) then
               W($row.status | startswith("FINAL"); "printed row \($row.row) on \($kp[1]) should be FINAL")
             else empty end) ]}
   else {lines: [], bad: []} end) as $pd14

# ---- L511-522 committed_in_git ----
| ([$r | .. | objects | select((.committed | type) == "object" and (.shas_file | type) == "object") | .shas_file.name] | unique) as $cnames
| ([$r | .. | objects | select(has("committed_in_git"))]) as $cig
| [ $cig[] | if .committed_in_git == false then W(($cnames | length) == 0; "committed_in_git false for a recheck that is committed")
             else W((.committed | type) == "object" and (.committed | has_all(["path", "commit", "sha256"])); "committed_in_git true without a committed file ref") end ] as $bad15

# ---- L524-536 v2 conditions ----
| $r.v2_conditions as $vc
| (if ($vc | truthy) then
     ([$r.figure_labels[] | select(.file | endswith($vc.fig4_elements.file))][0]) as $fl
     | [$r.exclusions[] | .entries[] | select(.file == $fl.file)] as $fe
     | ([$r.exclusions[] | select(.id == "C3_BR37_per_quantity")][0]) as $br
     | ([$r.extra_final_bars[] | select(.record == "R37")][0].cleanroom.verdict_wording_permanent) as $w37
     | [$br.entries[] | select(.record == "R37" and .status == "FINAL")] as $r37f
     | [$br.entries[] | select(.status != "FINAL")] as $pre
     | {lines: ["V2COND fig4 \($fl.label) FINAL \([$fe[] | select(.status == "FINAL")] | length) PRELIMINARY \([$fe[] | select(.status != "FINAL")] | length); R37 FINAL with wording \([$r37f[] | select(.cleanroom_wording == $w37)] | length)/\($r37f | length); B/R37 PRELIMINARY listed \($pre | length) reason_ok \(all($pre[]; .reason == $vc.br37_preliminary_reason) | pybool)"],
        bad: [
          W($fl.label == $vc.fig4_elements.label and ([$fe[] | select(.status == "FINAL")] | length) == $vc.fig4_elements.final and ([$fe[] | select(.status != "FINAL")] | length) == $vc.fig4_elements.preliminary; "condition 1: Fig 4 label / element counts"),
          W(($r37f | length) == $vc.r37_final_wording_n and all($r37f[]; .cleanroom_wording == $w37); "condition 2: R37 wording on every R37 FINAL slot"),
          W(($pre | length) == $vc.br37_preliminary_n and all($pre[]; .reason == $vc.br37_preliminary_reason); "condition 3: the \($vc.br37_preliminary_n | if type == "number" then tostring else "n/a" end) PRELIMINARY keys and their reason") ]}
   else {lines: [], bad: []} end) as $vc16

# ---- L538-543 counts check ----
| $r.counts_check as $cc
| (if ($cc | truthy) then
     blob($cc.result.commit; $cc.result.path) as $CJ
     | {n: ($cc.items | length), bad: [
         W([$CJ.items[] | [.id, .label, .record, .record_key, .equal]] == [$cc.items[] | [.id, .label, .record, .record_key, .equal]]; "counts items by key"),
         W(all($cc.items[]; .status | startswith("PRELIMINARY")) and all($cc.printed_numbers[]; . as $pn | ([$r.printed_items.all_rows[] | select(.row == $pn.row)][0].status) as $rs | $rs != null and ($pn.status | startswith($rs))); "a counts item is not PRELIMINARY, or a printed number disagrees with its row status (v2.4)") ]}
   else null end) as $cc17

# ---- L545-550 S2 markers; out-of-scope status ----
| (($r.amendment.file | split("/") | .[:-1] | join("/")) + "/signoff/") as $sdir
| ([($T[$r.amendment.commit + ":" + $sdir] // [])[] | select(basename | startswith("S2_"))] | sort) as $all_s2
| ([($outs[] | select(has("s2_marker")) | .s2_marker.path), (($r.in_scope_records + $r.out_of_scope_records)[] | select(has("s2_marker")) | .s2_marker.marker.path)] | unique) as $used
| [ W($used == $all_s2; "not every signed S2 marker at the spec commit is accounted for"),
    W(all($r.out_of_scope_records[]; if .name == "detector_vs_drop" then .status == "PRELIMINARY pending Aswin" else .status == "PRELIMINARY (record not reproduced as a whole); rows per row_rulings_v2_4" end) and ([$r.out_of_scope_records[].name] | sort) == ["detector_vs_drop", "honest_all_routec", "routec_simpool_numbers"]; "out-of-scope status (v2.4: Q9)") ] as $bad18

# ---- L552-573 commit message ----
| (if $HASMSG then
     $MSG as $msg
     | ([$r.sensitivity_rows.per_output[]?.entries[]]) as $sa
     | ([$sa[] | select(.status == "FINAL")] | length) as $nfin
     | ([$r | .. | objects | .[] | strings | select(contains("AWAITING"))] | length > 0) as $has_await
     | [ ($msg | contains($RECSHA)),
         (if ($sa | length) > 0 and $nfin == ($sa | length) then (($msg | contains("the 16 rc1a sensitivity slots are FINAL")) and (($msg | contains("stay PRELIMINARY until")) | not))
          else ($msg | contains("\(($sa | length) - $nfin) of \($sa | length) rc1a sensitivity slots stay PRELIMINARY")) end),
         ((($msg | contains("AWAITING")) == $has_await) or (($msg | contains("AWAITING")) | not)),
         ($msg | contains("FINAL outputs: " + ([$r.in_scope_outputs[].name] | join(", ")) + ".")),
         (($r.figure_labels // [])[] | . as $fl | $msg | contains("\($fl.file | basename | gsub("_source\\.json"; "")) \($fl.label)")),
         (($r.exclusions // [])[] | . as $x | $msg | contains("\($x.id) \($x.entries_n - $x.flipped_n) PRELIMINARY of \($x.entries_n)")) ] as $mc
     | {n: ($mc | length), bad: [ $mc | to_entries[] | select(.value | not) | "message: check \(.key)" ]}
   else null end) as $msg19

# ---- assemble, in the Python's order ----
| { lines: ([$perq[].lines[]] + [$fam[].lines[]] + $sens12.lines + $pd14.lines + $vc16.lines + $lines27b),
    n: ({outputs_reclassified: $n_reclass,
         covers_reread: ($n_cov_out + ($recres | length)),
         pending_strings: $n_pending}
        + (if ($pi | truthy) then {printed_items: $n_printed} else {} end)
        + {sensitivity_entries: ([$sensres[].n] | add // 0),
           exclusion_entries: ([$exres[].n] | add // 0),
           committed_in_git_nodes: ($cig | length)}
        + (if $cc17 != null then {counts_items: $cc17.n} else {} end)
        + (if $msg19 != null then {message_checks: $msg19.n} else {} end)),
    in_scope_set: ($in_list | length), store_n: ($ST | length), store_expected: $NSTORE,
    bad: ($bad3 + $bad4 + $bad27b + [$outres[].bad[]] + $bad6a + [$recres[].bad[]] + $bad7 + [$sensres[].bad[]] + [$exres[].bad[]] + $bad9 + $bad9b
          + [$perq[].bad[]] + [$fam[].bad[]] + $sens12.bad + $bad13 + $pd14.bad + $bad15 + $vc16.bad + (if $cc17 != null then $cc17.bad else [] end)
          + $bad18 + (if $msg19 != null then $msg19.bad else [] end)) }
