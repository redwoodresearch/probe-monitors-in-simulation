"""Review check for #526 (thread LWts): the meaning control picks its site line with str.splitlines() but plants with
str.split("\\n"). For each of its 150 runs (Modal CPU, tokenizer only), recompute the site (run_meaning_control_modal's
site_of), check it against the stored run file's (site_msg, after_line), and flag outputs where the two splitters
disagree before the chosen boundary (a bare \\r or another splitlines-only separator), i.e. where the line went to a
different boundary than intended. Numbers and cell names only.

  PYTHONPATH=src uv run --no-project --python 3.12 --with 'modal>=1.4.3' --with numpy modal run \\
      experiments/2026-10-10_claimB_meaning_control/recheck_cr_sites_modal.py
"""

import hashlib
import json
import re
import sys
from pathlib import Path

import modal

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE) if modal.is_local() else "/root")
import run_meaning_plant_modal as MP  # noqa: E402  (its reals() mirrors the meaning control's run selection)

M, P = MP.M, MP.P
app = modal.App("probe-shift_aswin-surya_sub-claimb_review-checks_kill-2026-11-10")
image = MP.image.add_local_file(HERE / "run_meaning_plant_modal.py", "/root/run_meaning_plant_modal.py")


@app.function(image=image, cpu=4, memory=16384, volumes={"/acts": P.vol, "/hf": P.hf}, timeout=1200, retries=0)
def recheck(model: str, revision: str) -> list:
    import numpy as np
    from transformers import AutoTokenizer

    from probe_shift.reader import segment_end_positions

    tok = AutoTokenizer.from_pretrained(model, revision=revision)
    out = []
    for cell, r in sorted(MP.reals().items()):
        h = hashlib.sha256(cell.encode()).hexdigest()[:16]
        rec = {"cell": cell, "file": h}
        site = M.site_of(tok, r["messages"], cell, segment_end_positions)
        stored = Path("/acts") / M.OUT / "runs" / f"{h}.npz"
        meta = json.loads(str(np.load(stored)["meta"])) if stored.exists() else None
        rec["stored"] = meta is not None
        if site is None:
            out.append(rec | {"site": None})
            continue
        i, j = site
        text = r["messages"][i]["content"]
        sl, nl = text.splitlines(), text.split("\n")
        rec |= {"matches_stored": meta is not None and (meta["site_msg"], meta["after_line"]) == (i, j),
                "bare_cr": bool(re.search(r"\r(?!\n)", text)),
                "same_boundary": j <= len(nl) and sl[:j] == [x.rstrip("\r") if x.endswith("\r") else x for x in nl[:j]]}
        out.append(rec)
    return out


@app.local_entrypoint()
def main():
    from probe_shift.models import QWEN3_32B

    out = recheck.remote(QWEN3_32B["name"], QWEN3_32B["revision"])
    sites = [r for r in out if r.get("site", 1) is not None]
    print(json.dumps({"runs": len(out), "with_site": len(sites), "stored": sum(r["stored"] for r in out),
                      "matches_stored": sum(r.get("matches_stored", False) for r in sites),
                      "bare_cr": sum(r.get("bare_cr", False) for r in sites),
                      "different_boundary": [r["cell"] for r in sites if not r["same_boundary"]]}, indent=1))
    print("PER_RUN " + json.dumps(out))  # cells and flags only, for the local recompute over unaffected runs
