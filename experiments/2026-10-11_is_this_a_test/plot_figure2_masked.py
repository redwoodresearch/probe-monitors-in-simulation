"""Figure 2 (appendix), masked: render run_figure2_masked.py's skeletons as a markdown file beside this script and,
optionally, a PNG. The input holds masked skeletons only, so nothing here reads run text.

  uv run --no-project --with matplotlib python experiments/2026-10-11_is_this_a_test/plot_figure2_masked.py \
      .run_cache/is_this_a_test/figure2_<git8>.json [out.png]
"""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parent.parent
PANELS = [("real str_replace", "simulated str_replace"), ("real insert", "simulated insert")]
WIDTH = 100  # PNG only: longer lines are cut at this many characters, with "…"
git = lambda *c: subprocess.run(["git", *c], capture_output=True, text=True, cwd=ROOT)
st = git("status", "--porcelain", "--untracked-files=no")
if st.returncode or st.stdout.strip() or git("ls-files", "--error-unmatch", __file__).returncode:
    raise SystemExit("commit first: the figure must record the commit of the code that made it")
src = ROOT / sys.argv[1]
res = json.loads(src.read_text())
fig = res["figure"]
post = res.get("selection")  # set by run_figure2_masked_POSTHOC.py
md = [
    "# Figure 2, masked: real and simulated editor outputs for one task (real first, then simulated)"
    + (f", {post}" if post else ""),
    "",
    *([f"**{post[0].upper() + post[1:]}**, made by `run_figure2_masked_POSTHOC.py` after seeing the as-run figure "
       "(`figure2_masked.md`, unchanged). Insert outputs are the first whose first line is an editor success header, "
       "and the diff's fixed file-name lines are kept on exact equality. It illustrates the editor's format only, not "
       "typical simulator fidelity.", ""] if post else []),
    f"Made by `plot_figure2_masked.py` at {git('rev-parse', 'HEAD').stdout.strip()[:8]} from `{sys.argv[1]}` (sha256 "
    f"{hashlib.sha256(src.read_bytes()).hexdigest()[:12]}), which `{'run_figure2_masked_POSTHOC.py' if post else 'run_figure2_masked.py'}` wrote at {res['git'][:8]}. From raw "
    "data. Selection and masking are in that script's docstring: one task (sha256 of its name "
    f"`{res['task sha256']}`), its first real honest run by key and its simulated honest run, and in each the first "
    f"str_replace and {'success-header' if post else 'first'} insert output. Letters and digits are masked to `x` inside the Modal container. Template lines "
    "and line-number gutters are kept only where they match the editor's exact format. Skeletons longer than 15 lines "
    "keep their first and last 7. Run IDs are the first 12 hex characters of sha256(run key); steps count agent turns "
    "from 0.",
    "",
    f"Honest runs read: {res['honest runs']['real honest']} real, {res['honest runs']['simulated honest']} simulated.",
]
for row in PANELS:
    for k in row:
        v = fig[k]
        md += ["", f"## {k[0].upper() + k[1:]}", ""] + ([f"Run `{v['run key sha256']}`: none ({v['none']})."] if "none" in v else
               [f"Run `{v['run key sha256']}`, step {v['step']}.", "", "```text", v["skeleton"], "```"])
out = HERE / ("figure2_masked_posthoc.md" if post else "figure2_masked.md")
if post and out.exists():
    raise FileExistsError(out)
out.write_text("\n".join(md) + "\n")
print(f"-> {out}")
if len(sys.argv) > 2:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    clip = lambda s: "\n".join(x if len(x) <= WIDTH else x[: WIDTH - 1] + "…" for x in s.expandtabs().split("\n"))  # tabs as a terminal shows them
    f, axes = plt.subplots(2, 2, figsize=(22, 11))
    for r, row in enumerate(PANELS):
        for c, k in enumerate(row):
            ax = axes[r][c]
            ax.axis("off")
            ax.set_title(f"{k[0].upper() + k[1:]} " + ("(none)" if "none" in fig[k] else f"(step {fig[k]['step']})"), loc="left", fontsize=14,
                         color="#1f4e79" if c == 0 else "#a04000", fontweight="bold")
            ax.text(0, 1, clip(fig[k].get("skeleton", fig[k].get("none", ""))), va="top", ha="left", family=["Menlo", "DejaVu Sans Mono"],
                    fontsize=8.5, transform=ax.transAxes)
    f.suptitle("The same task's editor outputs, real (left) and simulated (right), with content letters and digits masked"
               + ("\n(insert outputs: the first with an editor success header; chosen after seeing the first version)" if post else ""),
               fontsize=16, x=0.01, ha="left")
    f.text(0.01, 0.01, "Kept: template lines and line-number gutters that match the editor's exact format, punctuation, "
           "whitespace and the end-of-output newline. Lines over 100 characters cut here; full lines in "
           + (out.name + ". Illustrates the editor's format only, not typical simulator fidelity." if post else out.name + "."),
           fontsize=10, color="#555")
    f.tight_layout(rect=(0, 0.03, 1, 0.95))
    f.savefig(sys.argv[2], dpi=130)
    print(f"-> {sys.argv[2]}")
