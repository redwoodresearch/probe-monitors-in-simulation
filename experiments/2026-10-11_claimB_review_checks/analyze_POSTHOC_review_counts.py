"""POST HOC count for review thread #529 PRRT_kwDOU9RNXc6rN14q (4f, 2026-10-11), from D's stored span metadata and
projections only (no activations, no text). Local and free.

In D's all-tokens scope, "agent text or template" includes the prompts. Counted: tokens before the run's first tool
call (the initial system and user prompt, plus any assistant text before that call): their share of each run's top-20
tokens, their token fraction, and their signed part of the sim-minus-real gap in mean token projection (D's formula).
Later user messages also fall in "agent text or template" and are not separated here, and signed parts can cancel, so
this is the initial-prompt part only, not a bound on all prompt tokens' share.

  uv run --no-project --python 3.12 --with 'modal>=1.4.3' --with numpy \\
      python experiments/2026-10-11_claimB_review_checks/analyze_POSTHOC_review_counts.py
"""

import hashlib
import io
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal
import numpy as np

HERE = Path(__file__).parent
VOL, D_DIR = "probe-shift_aswin-surya_sub-la-activations", "claimB/attribution_tokens_v1/d"
REAL, SIM = "real-attack-e1", "sim-attack-fixed"


def main():
    vol = modal.Volume.from_name(VOL, version=2)
    read = lambda p: b"".join(vol.read_file(p))  # noqa: E731
    out = {"label": "POST HOC count for #529 N14q (4f, 2026-10-11): the initial-prompt part of D's agent-text-or-template category; metadata only"}

    names = [e.path for e in vol.listdir(D_DIR) if e.path.endswith(".npz") and not e.path.endswith(".tmp.npz")]
    with ThreadPoolExecutor(16) as ex:
        blobs = list(ex.map(read, names))
    top_pre, contrib, n_runs, pre_frac = {SIM: 0, REAL: 0}, {SIM: [], REAL: []}, {SIM: 0, REAL: 0}, {SIM: [], REAL: []}
    total = {SIM: [], REAL: []}
    for b in blobs:
        z = np.load(io.BytesIO(b))
        m, p = json.loads(str(z["meta"])), z["proj_L63"].astype(np.float64)
        first = min(s["span"][0] for s in m["spans"])
        use = np.arange(len(p)) < max(s["span"][1] for s in m["spans"])  # D's all-tokens scope
        owner = np.full(len(p), -1)
        for k, s in enumerate(m["spans"]):
            owner[s["span"][0]:s["span"][1]] = k
        pre = use & (np.arange(len(p)) < first) & (owner < 0)
        top = np.argsort(np.where(use, p, -np.inf))[::-1][:20]
        a = m["arm"]
        n_runs[a] += 1
        top_pre[a] += int(pre[top].sum())
        contrib[a].append(p[pre].sum() / use.sum())
        total[a].append(p[use].sum() / use.sum())
        pre_frac[a].append(pre.sum() / use.sum())
    gap = np.mean(total[SIM]) - np.mean(total[REAL])
    g_pre = np.mean(contrib[SIM]) - np.mean(contrib[REAL])
    out["attribution_prompt"] = {"runs": n_runs, "gap_all_tokens": float(gap), "initial_prompt_gap": float(g_pre), "initial_prompt_share": float(g_pre / gap),
                                 "top20_tokens_initial_prompt": {a: {"tokens": top_pre[a], "of": 20 * n_runs[a]} for a in (SIM, REAL)},
                                 "token_fraction_initial_prompt": {a: float(np.mean(pre_frac[a])) for a in (SIM, REAL)},
                                 "files_sha256": hashlib.sha256(b"".join(sorted(hashlib.sha256(b).digest() for b in blobs))).hexdigest()}

    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip()
    out |= {"git": git, "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    (HERE / "results").mkdir(exist_ok=True)
    dst = HERE / "results" / f"review_counts_POSTHOC_{stamp}.json"
    with open(dst, "x") as fh:
        fh.write(json.dumps(out, indent=1))
    a = out["attribution_prompt"]
    print(f"#529: gap {a['gap_all_tokens']:.6f}; initial-prompt part {a['initial_prompt_gap']:+.6f} = {100 * a['initial_prompt_share']:+.2f}% of it; "
          f"top-20 tokens in the initial prompt: {a['top20_tokens_initial_prompt']}; token fraction before it {a['token_fraction_initial_prompt']}")
    print("->", dst)


if __name__ == "__main__":
    main()
