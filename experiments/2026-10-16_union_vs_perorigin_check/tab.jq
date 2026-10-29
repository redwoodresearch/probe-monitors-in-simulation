def hw: ((.ci95[1]-.ci95[0])/2);
def z: (if .ci95[0] > 0 or .ci95[1] < 0 then "excl0" else "incl0" end);
def row($name; $po; $un):
  [$name, ($po.point), ($un.point), (($un.point-$po.point)), ($po.ci95[0]), ($po.ci95[1]), ($po|hw), ($po|z), ($un.ci95[0]), ($un.ci95[1]), ($un|hw), ($un|z), (($un|hw)/($po|hw)), $po.finite_draws, $un.finite_draws];
def P: ["Sol (a)","FIBS","harmfulness","refusal","high-stakes","own_all_rs_L48"];
. as $d |
(["pair","item","po_point","un_point","un_minus_po","po_lo","po_hi","po_hw","po_0","un_lo","un_hi","un_hw","un_0","ratio","po_nd","un_nd"]|@tsv),
( [["A","pooled_all_origins_per_origin_streams","pooled_all_origins"],["B","pooled_all_registered_cohort","pooled_all_origins_registered_cohort"],["C","paper_cells_per_origin_streams","paper_cells"]][] as [$pair,$PO,$UN] |
  ( ["C","R"][] as $r | ["0.01","0.05"][] as $f |
    ( P[] as $p | ([$pair+"|"+$r+"|"+$f] + row($p; $d.pooled[$p+" "+$r][$PO].joint[$f].gap.delta_full; $d.pooled[$p+" "+$r][$UN].joint[$f].gap.delta_full)) | @tsv ),
    ( ["S","Y","abs_Y"][] as $s | ([$pair+"|"+$r+"|"+$f] + row("six_"+$s; $d.coheadline_signed[$PO].joint[$f]["headline_six|"+$r][$s]; $d.coheadline_signed[$UN].joint[$f]["headline_six|"+$r][$s])) | @tsv ),
    ( ([$pair+"|"+$r+"|"+$f] + row("excess_six"; $d.panel_groups[$PO].pooled.joint[$f]["headline_own_all_rs|"+$r].primary.mean_G_minus_mean_Y; $d.panel_groups[$UN].pooled.joint[$f]["headline_own_all_rs|"+$r].primary.mean_G_minus_mean_Y)) | @tsv )
  )
)
