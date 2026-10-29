"""POST HOC (after the result, for #1365's review thread): for every chosen run, is the position that sets its run score
(its highest epoch's argmax) on chat-template markup that the readout counted as eligible (Qwen3.5's plain-text
`<function=…>`, `<parameter=…>`, `</parameter>`, `</function>`), on whitespace, or on content? If no run's maximum is on
markup, the content-only maximum equals the reported one for every run. Modal CPU, about $0.05 (Aswin's "<$5 just run");
only classes, counts, scores and run ids leave.

  PYTHONPATH=src uv run --no-project --python 3.12 --with modal modal run experiments/2026-10-13_simcue_onpolicy/check_POSTHOC_argmax_markup.py
"""

import hashlib
import json
import re
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).parent
ROOT = HERE.parent.parent
ROWS, ROWS_SHA = ROOT / ".run_cache/simcue_onpolicy/rows_2026-10-13_121721.json", "1abd1dc0e7437db5708ffc306a831b1a3ba6a918ec1390d9287593618decfab1"
ARM = {"real1": "attack", "real2": "attack", "sim_sept": "sim_attack@september", "honest_a": "honest_a", "honest_b": "honest_b"}
MARKUP = re.compile(r"<function=[^>\n]*>|<parameter=[^>\n]*>|</parameter>|</function>")
app = modal.App("probe-shift_aswin-surya_sub-simcue-markup_kill-2026-11-13")
q3 = modal.Volume.from_name("probe-shift_aswin-surya_sub-q3onpol_acts_kill-2026-11-08", version=2).read_only()
hf = modal.Volume.from_name("probe-shift_aswin-surya_hf-cache").read_only()
image = (modal.Image.debian_slim(python_version="3.12").uv_pip_install("transformers==5.19.0", "jinja2", "huggingface_hub")
         .env({"HF_HOME": "/hf", "HF_HUB_OFFLINE": "1"}).add_local_python_source("probe_shift"))


@app.function(image=image, volumes={"/q3": q3, "/hf": hf}, cpu=8, memory=32768, timeout=3600, retries=0)
def classify(items: list) -> list:
    import os
    import sys
    from collections import defaultdict

    from transformers import AutoTokenizer

    from probe_shift.models import QWEN35_27B
    sys.stdout.flush(), sys.stderr.flush()
    null = os.open(os.devnull, os.O_WRONLY)
    os.dup2(null, 1), os.dup2(null, 2)  # nothing printed leaves
    tok = AutoTokenizer.from_pretrained(QWEN35_27B["name"], revision=QWEN35_27B["revision"])
    want = defaultdict(dict)  # (stage, arm) -> {(run, epoch): item}
    for it in items:
        want[(it["stage"], ARM[it["world"]])][(it["run"], it["epoch"])] = it
    out = []
    for (stage, arm), need in want.items():
        for line in (Path("/q3") / stage / "inputs" / f"{arm}.jsonl").read_text().splitlines():
            ep = json.loads(line)
            it = need.get((ep["run"], ep["epoch"]))
            if it is None:
                continue
            text = tok.decode(ep["ids"])
            enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
            if enc["input_ids"] != ep["ids"]:
                out.append(it | {"class": "round trip failed"})
                continue
            a, b = enc["offset_mapping"][it["argmax"]]
            piece = text[a:b]
            on_markup = any(m.start() < b and m.end() > a for m in MARKUP.finditer(text, max(0, a - 64), b + 64))
            out.append(it | {"class": "markup" if on_markup else ("whitespace" if not piece.strip() else "content")})
    return out


@app.local_entrypoint()
def main():
    git = lambda *c: subprocess.run(["git", *c], capture_output=True, text=True, cwd=ROOT)
    st = git("status", "--porcelain", "--untracked-files=no")
    if st.returncode or st.stdout.strip() or git("ls-files", "--error-unmatch", __file__).returncode:
        raise SystemExit("commit first")
    if hashlib.sha256(ROWS.read_bytes()).hexdigest() != ROWS_SHA:
        raise SystemExit("the rows file is not the collected one")
    chosen = json.loads((HERE / "results/check_2026-10-13_121629_check.json").read_text())["result"]["chosen"]
    best = {}
    for r in json.loads(ROWS.read_text()):
        k = (r["world"], r["run"])
        if r["run"] in set(chosen[r["world"]]) and (k not in best or r["max"] > best[k]["max"]):
            best[k] = r
    items = [{"world": r["world"], "stage": r["stage"], "run": r["run"], "epoch": r["epoch"], "argmax": r["argmax"], "score": r["max"]} for r in best.values()]
    res = classify.remote(items)
    if len(res) != len(items):
        raise ValueError(f"{len(items) - len(res)} runs were not classified")
    t1 = 6.018515315380688  # results/analysis_2026-10-13_124152.json
    from collections import Counter
    out = {"git": git("rev-parse", "HEAD").stdout.strip(), "runs": len(res),
           "class of each run's maximizing token, per world": {w: dict(Counter(x["class"] for x in res if x["world"] == w)) for w in ARM},
           "runs above the 1% threshold 6.0185": sorted([[x["world"], x["class"], round(x["score"], 4)] for x in res if x["score"] > t1]),
           "the two highest first-real runs (they set the threshold)": [[x["class"], round(x["score"], 4)] for x in sorted((x for x in res if x["world"] == "real1"), key=lambda x: -x["score"])[:2]]}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    path = HERE / "results" / f"argmax_markup_POSTHOC_{stamp}.json"
    path.parent.mkdir(exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1), f"\n-> {path}")
