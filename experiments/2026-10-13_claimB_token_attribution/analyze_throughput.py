"""Token attribution: reader throughput by item class and GPU-idle share, from the job's own log lines and the staged
token counts (no scores read). Writes results/throughput_<stamp>.json with the log lines it used.

  python3 experiments/2026-10-13_claimB_token_attribution/analyze_throughput.py <launch log> <app start epoch s> <app stop epoch s>
"""

import json
import re
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent


def main(log, t_start, t_stop):
    lines = [x for x in Path(log).read_text().splitlines() if x.startswith("[token-attr]")]
    at = {int(m.group(2)): int(m.group(1)) for x in lines if (m := re.match(r"\[token-attr\]\s+(\d+)s item (\d+) of", x))}
    sec = lambda key: next(int(re.match(r"\[token-attr\]\s+(\d+)s", x).group(1)) for x in lines if key in x)  # noqa: E731
    at[423] = sec("done:")
    items = [json.loads(x) for x in (HERE / "data/staged.jsonl").read_text().splitlines()]
    tok = lambda a, b: sum(min(x["n_tokens"], 32768) for x in items[a:b])  # noqa: E731  (items a..b-1 done between the log lines)
    classes = {"sites (items 0-259)": (0, 260), "planted and twins (items 280-379)": (280, 380), "whole runs (items 400-422)": (400, 423)}
    out = {"tokens_per_second": {k: round(tok(a + 1, b + 1) / (at[b] - at[a]), 1) if b != 423 else round(tok(a + 1, b) / (at[b] - at[a]), 1)
                                 for k, (a, b) in classes.items()},
           "rows_assembled_s": sec("rows assembled"), "reader_loaded_s": sec("reader loaded"), "done_s": at[423],
           "app_seconds": int(t_stop) - int(t_start)}
    out["gpu_idle_share_of_app_time"] = round(out["reader_loaded_s"] / out["app_seconds"], 3)
    out["log_lines"] = lines
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    (HERE / "results" / f"throughput_{stamp}.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k != "log_lines"}, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:4])
