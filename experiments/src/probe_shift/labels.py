"""Find the Q2 label table to analyze (run_label_calls.py writes them to .run_cache/q2_call_labels/)."""

import json
from pathlib import Path

from probe_shift.calls import fingerprint


def resolve_labels(root: Path, labels_file: str) -> tuple[Path, str]:
    """The label table and its labeler hash. `latest` is the newest table made by the current labeler; anything else is
    a path relative to root. Refuses a table made by other rules."""
    if labels_file == "latest":
        tables = [p for p in sorted((root / ".run_cache" / "q2_call_labels").glob("*.jsonl"))
                  if p.with_suffix(".meta.json").exists()  # tables from before fingerprints have none
                  and json.loads(p.with_suffix(".meta.json").read_text())["labeler_sha"] == fingerprint()]
        if not tables:
            raise ValueError(f"no label table from labeler {fingerprint()}: run run_label_calls.py first (about 30 s)")
        path = tables[-1]
    else:
        path = root / labels_file
    made_by = json.loads(path.with_suffix(".meta.json").read_text())["labeler_sha"]
    if made_by != fingerprint():
        raise ValueError(f"{path.name} was made by labeler {made_by}, not the current {fingerprint()}: rerun run_label_calls.py")
    return path, made_by
