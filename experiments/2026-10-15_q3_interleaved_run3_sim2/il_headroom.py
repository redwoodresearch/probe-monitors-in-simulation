"""IL headroom under Aswin's ceiling ("A: raise ceiling to ~$24.5k"), for 0d (d9 02:50): IL scope = the IL arms for 507
(il_q1..q4, il_old), the new mains' IL arms (il_s1..s5, il_s4d), their reruns, and S4's not-run cells (4r).
  spent      = simulator spend per IL launch (launch.json last spend actual_usd, read on the hosts; --sim)
             + server sets billed by Modal (hourly report, summed per set by the app-name hex in each il_servers log)
             + unbilled server time since the last full billed hour (--unbilled-usd, stated estimate) + EC2 (--ec2-usd)
  committed  = per remaining item an expected value (sims at the done-IL MEAN per sim run + 30%; the mean, not the median,
               because these are sums of a right-skewed cost) and a hard bound (caps / hard server timeouts)
  headroom   = ceiling - spent - committed; usable = headroom - the 10% reserve of the ceiling ($2,450) (d9 03:22);
             the completion cutoff is the usable headroom on the expected basis.
IL scope only: il_q1..q4, il_old, il_s1, il_s2, il_s3, il_s5, il_s4d (each confirmed by its <run>.il_meta.json:
"interleaved_launch": true, worlds run3/sim2), their reruns and completions, and 4r (main and follow-up, servers). Not
in scope: the new-mains mains and follow-ups, and 5r (new-mains purchase, its own $11,000 cap).
  python3 il_headroom.py --billing <modal billing json> --server-logs <dir> --items <items json> --out <json>
"""

import argparse
import glob
import json
import re
from collections import defaultdict

CEILING = 24500.0
RESERVE = 0.10 * CEILING


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--billing", required=True)
    ap.add_argument("--server-logs", required=True)
    ap.add_argument("--items", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    items = json.load(open(a.items))
    sets = {}
    for f in glob.glob(f"{a.server_logs}/servers_*.log"):
        for x in set(re.findall(r"phase2_srv([0-9a-f]{6})", open(f, errors="replace").read())):
            assert x not in sets, x
            sets[x] = f.split("/")[-1]
    billed, hours = defaultdict(float), set()
    for r in json.load(open(a.billing)):
        m = re.search(r"phase2_srv([0-9a-f]{6})", r.get("description", ""))
        if m and m.group(1) in sets:
            billed[sets[m.group(1)]] += float(r["cost"])
            hours.add(r["interval_start"])
    spent = {"simulator": round(sum(v["usd"] for v in items["sim_spent"].values()), 2),
             "servers_billed": round(sum(billed.values()), 2), "servers_unbilled_est": items["unbilled_usd"],
             "ec2_est": items["ec2_usd"]}
    exp = round(sum(v["expected_usd"] for v in items["committed"].values()), 2)
    hard = round(sum(v["hard_usd"] for v in items["committed"].values()), 2)
    s = round(sum(spent.values()), 2)
    out = {"ceiling_usd": CEILING, "spent_usd": s, "spent": spent, "servers_billed_by_set": {k: round(v, 2) for k, v in sorted(billed.items())},
           "billing_last_full_hour_utc": max(hours) if hours else None, "committed_expected_usd": exp, "committed_hard_usd": hard,
           "headroom_expected_usd": round(CEILING - s - exp, 2), "headroom_hard_usd": round(CEILING - s - hard, 2),
           "reserve_usd": RESERVE, "usable_expected_usd": round(CEILING - s - exp - RESERVE, 2),
           "usable_hard_usd": round(CEILING - s - hard - RESERVE, 2), "completion_cutoff_usd": round(CEILING - s - exp - RESERVE, 2),
           "items": items}
    json.dump(out, open(a.out, "w"), indent=1)
    print(f"IL headroom: ceiling ${CEILING:,.0f} - spent ${s:,.0f} (sims ${spent['simulator']:,.0f}, servers billed "
          f"${spent['servers_billed']:,.0f} to {out['billing_last_full_hour_utc']} UTC + ~${items['unbilled_usd']:,.0f} unbilled, EC2 ~${items['ec2_usd']:,.0f})"
          f" - committed ${exp:,.0f} expected (${hard:,.0f} hard) = ${out['headroom_expected_usd']:,.0f} expected"
          f" (${out['headroom_hard_usd']:,.0f} hard); minus the 10% reserve ${RESERVE:,.0f} = usable ${out['usable_expected_usd']:,.0f}"
          f" expected (${out['usable_hard_usd']:,.0f} hard) = the completion cutoff")


if __name__ == "__main__":
    main()
