"""Score histograms, real vs sim, negatives and positives: pre_call (the `<tool_call>` template token), next to its
content-token control prev_content, and call_end; attack view, probes trained on real."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, str(Path(__file__).parent))
from paths import DATA, RESULTS, ROOT, stamp as now  # noqa: E402

HERE = Path(__file__).parent
READOUTS = ["pre_call", "prev_content", "call_end"]  # pre_call is reported only next to its content-token control
NEED = {(rd, ro, "attack_only", "real") for rd in ("Qwen3-32B", "Qwen3.5") for ro in READOUTS}
reports = [{(r["reader"], r["readout"], r["view"], r["train"]): r for r in json.load(open(f))["rows"]}
           for f in sorted(RESULTS.glob("shift_stage1_*.json"), reverse=True)]
st = next(r for r in reports if NEED <= set(r))  # the newest Stage 1 report with every panel (a filtered rerun may lack some)
fig, axes = plt.subplots(3, 2, figsize=(10, 9.4), sharey="row")
col = {"real": "#1f77b4", "sim": "#d62728"}
for i, readout in enumerate(READOUTS):
    for j, (reader, f) in enumerate([("Qwen3-32B", "qwen3-32b"), ("Qwen3.5", "qwen3.5")]):
        ax = axes[i, j]
        bins = np.linspace(-20, 15, 141) if readout == "call_end" else np.linspace(-12, 5, 103)  # outside: piled in the end bins
        d = pd.read_json(DATA / "tables" / f"{f}__{readout}__attack_only.jsonl.gz", lines=True).query("train_set == 'real'")
        d["margin"] = d.score - d.threshold
        for dom in ("real", "sim"):
            for y, ls, lab in ((0, "--", "read-only"), (1, "-", "impactful")):
                x = d[(d.domain == dom) & (d.y == y)].margin.clip(bins[0], bins[-1])
                ax.hist(x, bins=bins, density=True, histtype="step", ls=ls, color=col[dom], lw=1.3, label=f"{dom} {lab}")
        ax.axvline(0, color="k", lw=0.8)
        ax.set_yscale("log"); ax.set_ylim(1e-4, 1)
        r = st[(reader, readout, "attack_only", "real")]
        ax.set_title(f"{reader}, {readout}: AUROC real {r['real']['auroc']['value']:.3f} / sim {r['sim']['auroc']['value']:.3f}\n"
                     f"recall at frozen 1% real {r['real']['recall']['value']:.3f} / sim {r['sim']['recall']['value']:.3f} "
                     f"(FPR {r['real']['fpr']['value']:.3f} / {r['sim']['fpr']['value']:.3f})", fontsize=9)
        if j == 0: ax.set_ylabel("density (log)")
axes[0, 0].legend(fontsize=8, loc="upper left")
fig.suptitle("Probes trained on real attack runs, tested on real vs simulated attack runs (attack view, headline layer per fold)", fontsize=10)
fig.supxlabel("score minus the fold's frozen 1% threshold, in logits (flagged if > 0; right = more suspicious)", fontsize=9)
fig.tight_layout()
out = RESULTS / f"shift_stage1_histograms_{now()}.png"; fig.savefig(out, dpi=130); print(out)
