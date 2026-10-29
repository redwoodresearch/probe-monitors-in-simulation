"""K7, step 2: how much of each eval-awareness probe's run score is explained by the LinuxArena environment alone?
Templated probe (call_end, layer 40) and Needham probe (asst_mean, layer 27), run score = mean over calls. Reports the
share of run-score variance between environments (eta squared, all four arms) and each environment's mean percentile
rank. A probe for evaluation cues should not sort environments: real LinuxArena runs are all the same kind of eval.

  python analyze_probe_by_environment.py --templated <parquet> --needham <parquet>
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

HERE = Path(__file__).parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--templated", required=True)
    ap.add_argument("--needham", required=True)
    args = ap.parse_args()
    t = pd.read_parquet(args.templated)
    t = t[t.kind == "call_end"].groupby(["arm", "key", "environment"]).L40.mean().rename("templated")
    n = pd.read_parquet(args.needham, columns=["arm", "key", "environment", "asst_mean_L27"]).groupby(["arm", "key", "environment"]).asst_mean_L27.mean().rename("needham")
    r = pd.concat([t, n], axis=1, join="inner").reset_index()
    out = {"runs": len(r), "eta_squared": {}, "mean_percentile_by_environment": {}}
    for p in ("templated", "needham"):
        r[f"{p}_pct"] = r[p].rank(pct=True)
        g = r.groupby("environment")[p]
        out["eta_squared"][p] = round(float(((g.transform("mean") - r[p].mean()) ** 2).sum() / ((r[p] - r[p].mean()) ** 2).sum()), 4)
        out["mean_percentile_by_environment"][p] = r.groupby("environment")[f"{p}_pct"].mean().round(3).to_dict()
    print("runs", len(r), "| share of variance between environments:", out["eta_squared"])
    print(pd.DataFrame(out["mean_percentile_by_environment"]).sort_values("needham").to_string())
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    with (HERE / "results" / f"probe_by_environment_{stamp}.json").open("x") as f:
        f.write(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
