"""September attack approaches, kept out of git. A pilot task file records each cell's approach as its sha256 and S3
URI; the text lives in the git-ignored data/phase2_attack_check/approaches/<sha256>.txt (on the devbox, the folder
run_pilot.py --approaches names) and at APPROACHES_S3. Some approaches quote a sandbox environment's test credentials,
which the repo's secret scanner rightly won't let into git, and approach texts are task data, not code.

store() writes one text and returns the fields a task file keeps; materialize() writes a run-local copy of a task file
with every text restored and checked against its sha256, for tasks.py (PHASE2_CELLS), which reads `approach`.
"""

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
APPROACHES = HERE.parents[1] / "data" / "phase2_attack_check" / "approaches"
APPROACHES_S3 = "s3://individual-aswin-surya-probe-shift-kill-2027-04-06/phase2/approaches/"


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def store(text: str, directory: Path = APPROACHES) -> dict:
    """Write the text (once) under its hash; return what the task file keeps instead of it."""
    digest = sha256(text)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{digest}.txt"
    if not path.exists():
        path.write_text(text)
    assert sha256(path.read_text()) == digest, f"{path} doesn't hash to its name"
    return {"approach_sha256": digest, "approach_chars": len(text), "approach_s3": f"{APPROACHES_S3}{digest}.txt"}


def materialize(task_file: Path, out: Path, directory: Path = APPROACHES) -> Path:
    """A copy of the task file with each cell's approach text restored from `directory`, refusing any text whose
    sha256 isn't the one recorded. A file that still carries its texts (step 1's cells.json) is copied as is."""
    data = json.loads(task_file.read_text())
    for cell in data["cells"]:
        if "approach" in cell:
            continue
        text = (directory / f"{cell['approach_sha256']}.txt").read_text()
        assert sha256(text) == cell["approach_sha256"], f"{cell['cell']}: the approach text doesn't match its sha256"
        cell["approach"] = text
    out.write_text(json.dumps(data, indent=1))
    return out
