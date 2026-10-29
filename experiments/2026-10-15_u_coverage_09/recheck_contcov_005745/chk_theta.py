# theta*(T*) recomputed from achieved catch changes; mean theta(T-hat), mapped X=0 share for B1/B2/B8/B9; spot cells.
import json, math, sys, statistics as S
d = sys.argv[1]
R = json.load(open(f"{d}/record.json"))["result"]
CAL = R["setup"]["calibration"]; per = json.load(open(f"{d}/chk_cov_per_pattern.json"))
def tx(a):
    m = sum(a)/len(a); return math.sqrt(m*m + (S.variance(a) if len(a) > 1 else 0.0))
worst = 0.0
for rk, c in CAL.items():
    for pn, p in c["patterns"].items():
        worst = max(worst, abs(tx(p["achieved_pts_at_Tstar"]) - p["theta_Tstar"]))
print("max |theta*(recomputed from achieved_pts) - stored|:", worst)
print("theta* pooled / with_strategy per row (selected patterns):")
for rk, c in CAL.items():
    print(" ", rk, "eligible", c["eligible"])
    for pn in ("h_plus10", "d_single_plus10", "B1_scale_0p7", "B2_scale_1p4", "B3_tail_up_2sd", "B4_tail_down_to_minus5", "B5_mix_down_3sd", "B6_mix_up_3sd", "B7_mixed_sign_shape", "B8_band_up_at_Tstar", "B9_band_down_at_Tstar"):
        p = c["patterns"][pn]
        print(f"    {pn:24s} th*={p['theta_Tstar']:.2f}/{p['theta_Tstar_with_strategy']:.2f} ach={[round(x,2) for x in p['achieved_pts_at_Tstar']]} sat={p['saturated_probes']}")
print("B1/B2/B8/B9: mapped shareX0, mean theta(T-hat), mapped cov, binary cov")
sh, mt, mc = [], [], []
for cell, pp in per.items():
    for pn in ("B1_scale_0p7", "B2_scale_1p4", "B8_band_up_at_Tstar", "B9_band_down_at_Tstar"):
        v = pp[pn]; sh.append(v["mapped"]["shareX0"]); mt.append(v["meanth"]); mc.append(v["mapped"]["rate"])
        print(f"  {cell:45s} {pn:22s} X0={v['mapped']['shareX0']:.3f} meanth={v['meanth']:.2f} covM={v['mapped']['rate']:.3f} covB={v['binary']['rate']:.3f}")
print("range shareX0", min(sh), max(sh), "| range mean theta", round(min(mt),2), round(max(mt),2), "| range mapped cov", min(mc), max(mc))
print("reply-only: shareX0", min(s for s,(c) in zip(sh, [c for c in per for _ in range(4)]) if c.startswith("R_")), max(s for s,(c) in zip(sh, [c for c in per for _ in range(4)]) if c.startswith("R_")),
      "| mean theta", round(min(t for t,c in zip(mt,[c for c in per for _ in range(4)]) if c.startswith("R_")),2), round(max(t for t,c in zip(mt,[c for c in per for _ in range(4)]) if c.startswith("R_")),2))
print("SPOT:")
for cell, pn, est, kind in [("R_reply_meanpooled|alpha=0.01|pooled","B1_scale_0p7","mapped","rate"),("R_reply_meanpooled|alpha=0.01|with_strategy","B1_scale_0p7","mapped","rate"),
    ("R_reply_meanpooled|alpha=0.01|pooled","B2_scale_1p4","mapped","rate"),("R_reply_meanpooled|alpha=0.01|with_strategy","B2_scale_1p4","mapped","rate"),
    ("R_reply_meanpooled|alpha=0.05|pooled","B8_band_up_at_Tstar","mapped","rate"),("R_reply_meanpooled|alpha=0.05|with_strategy","B8_band_up_at_Tstar","mapped","rate"),
    ("R_reply_meanpooled|alpha=0.01|pooled","h_plus10","mapped","rate"),
    ("R_reply_meanpooled|alpha=0.01|pooled","d_single_plus10","binary","c0"),("R_reply_meanpooled|alpha=0.01|with_strategy","d_single_plus10","binary","c0")]:
    v = per[cell][pn][est]; print(f"  {cell} {pn} {est}: cov={v['rate']:.3f} CP=[{v['lo']:.3f},{v['hi']:.3f}] n0={v['n0']} cond={v['c0']}")
print("non-B mapped coverage < 0.95 cells:")
for cell, pp in per.items():
    for pn, v in pp.items():
        if not pn.startswith("B") and (v["mapped"]["rate"] < 0.95 or v["mapped"]["lo"] < 0.93):
            print(f"  {cell} {pn} {v['mapped']['rate']:.3f} [{v['mapped']['lo']:.3f}]")
print("binary min coverage overall:", min((v["binary"]["rate"], c, p) for c, pp in per.items() for p, v in pp.items()))
