"""Where the shift analysis reads its inputs and writes its results.

DATA (gitignored, archived on S3; see README.md) holds:
- tables/: per-call headline scores, built by assemble*.py;
- levels/: each text level's score files for layers 9/16/17;
- vol/, vol_qwen35/: Volume files the assemble scripts read;
- context_features.parquet;
- the attempted/declined labels.
Results go to the experiment's results/ as shift_<name>_<stamp>.json.
"""
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).parent
EXP = HERE.parent
ROOT = EXP.parents[1]
DATA = ROOT / ".run_cache" / "q2_shift"
RESULTS = EXP / "results"


def stamp() -> str:
    return datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
