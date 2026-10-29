"""Write-up figure 6a9b8621, "Attack detector flags runs from the first command, mostly from Qwen's own words",
regenerated from raw data ($0, local, small JSON only).

Source: the D3 threshold check (post hoc, descriptive), d3_threshold_POSTHOC_2026-10-11_121418.json, committed at
de458d7a on aswin/q3-590-redo; archived on S3 (inputs/manifest_6a9b8621_v1.json). Every drawn value is read by key from
that file after its sha256 is checked. For each position j = 1, 2, 3 (the run's first j commands) and each detector
(claim C's command-end detector on the full transcript; the same detector retrained with the agent's reasoning and chat
removed), it draws run 1's catch at the false-alarm rate matched to D3's achieved honest_b rate, with its task-resampled
95% interval, and the honest_b runs flagged at that threshold. It prints every value with its key path and compares it
with the number on the doc image (MATCH / MISMATCH), then writes the PNG and a values JSON (never overwriting).

  uv run --no-project --with matplotlib --with numpy python experiments/2026-10-12_q3_writeup_figures/plot_6a9b8621.py \
      --source experiments/2026-10-12_q3_writeup_figures/inputs/d3_threshold_POSTHOC_2026-10-11_121418.json
"""

import argparse
import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).parent
SHA = "54c791e5f196e948001a8c21c599032c5324d438357732cac071aa75bdd6405c"  # inputs/manifest_6a9b8621_v1.json
JS = ("1", "2", "3")
DETECTORS = {"claim_c": "Full transcript", "stripped": "Reasoning and chat removed (retrained; edited, off-policy input)"}
# The numbers printed on the doc image (asset 6a9b8621), as listed in FIGURES.md @ af47bc10.
DOC = {"claim_c": [95, 99, 100], "stripped": [35, 58, 57], "honest_flagged": [2, 6, 3], "honest_runs": 172}
COLORS = {"claim_c": "#2a78d6", "stripped": "#eb6834"}  # categorical slots 1 and 2 of the reference palette
INK, INK2, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"  # with the two series colors: 5 colors in all


def get(d, path):
    for k in path.split("/"):
        d = d[k]
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    args = ap.parse_args()
    raw = Path(args.source).read_bytes()
    got = hashlib.sha256(raw).hexdigest()
    if got != SHA:
        raise SystemExit(f"source sha256 {got} differs from the manifest's {SHA}")
    src = json.loads(raw)

    values, checks = {}, []

    def read(path):
        v = get(src, path)
        values[path] = v
        print(f"{path} = {v}")
        return v

    def compare(what, doc, drawn):
        ok = doc == drawn
        checks.append({"what": what, "doc": doc, "drawn": drawn, "match": ok})
        print(f"  {what}: doc {doc} vs drawn {drawn} -> {'MATCH' if ok else 'MISMATCH'}")

    rows = {}
    for det in DETECTORS:
        rows[det] = []
        for i, j in enumerate(JS):
            base = f"by_j/{j}/{det}"
            catch = read(f"{base}/matched/catch")
            lo, hi = read(f"{base}/matched/catch_interval/ci95")
            above = read(f"{base}/matched/honest_b_above")
            n_honest = read(f"{base}/honest_b_runs")
            rate = read(f"{base}/matched/achieved_false_alarms")
            failed = read(f"{base}/matched/catch_interval/failed")
            if failed:
                raise SystemExit(f"{base}: {failed} failed interval draws")
            rows[det].append({"j": int(j), "catch": catch, "ci95": [lo, hi], "honest_flagged": above,
                              "honest_runs": n_honest, "achieved_false_alarms": rate})
            compare(f"{det} catch, first {j} command(s), % rounded", DOC[det][i], round(100 * catch))
            compare(f"{det} honest flagged, first {j}", DOC["honest_flagged"][i], above)
            compare(f"{det} honest runs, first {j}", DOC["honest_runs"], n_honest)
    for i, j in enumerate(JS):  # both detectors are thresholded at the same matched rate
        if rows["claim_c"][i]["achieved_false_alarms"] != rows["stripped"][i]["achieved_false_alarms"]:
            raise SystemExit(f"j={j}: the two detectors' achieved false-alarm rates differ")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.2, 4.8), dpi=180)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    width, gap = 0.36, 0.02
    for k, det in enumerate(DETECTORS):
        for i, r in enumerate(rows[det]):
            x = i + (k - 0.5) * (width + gap)
            y = 100 * r["catch"]
            lo, hi = (100 * v for v in r["ci95"])
            ax.bar(x, y, width, color=COLORS[det], label=DETECTORS[det] if i == 0 else None, zorder=2)
            ax.errorbar(x, y, yerr=[[y - lo], [hi - y]], fmt="none", ecolor=INK, elinewidth=1.2, capsize=3, zorder=3)
            ax.text(x, hi + 2, f"{y:.0f}%\n[{lo:.0f}, {hi:.0f}]", ha="center", va="bottom", fontsize=8, color=INK)
    ax.set_xticks(range(len(JS)))
    ax.set_xticklabels([f"First {j} command{'s' if j != '1' else ''}\nhonest flagged {rows['claim_c'][i]['honest_flagged']}"
                        f"/{rows['claim_c'][i]['honest_runs']}" for i, j in enumerate(JS)], color=INK2, fontsize=9)
    ax.set_ylim(0, 118)
    ax.set_yticks(range(0, 101, 20))
    ax.set_ylabel("Attack runs flagged, % (higher is better)", color=INK2, fontsize=9)
    ax.tick_params(colors=INK2, labelsize=9)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK2)
    ax.grid(axis="y", color=INK2, alpha=0.15, linewidth=0.8, zorder=0)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), frameon=False, fontsize=8.5, ncol=1, labelcolor=INK)
    fig.text(0.01, 0.01, "Post hoc, descriptive. Each threshold matched to the same honest-run false-alarm rate (honest_b).\n"
             "Bars are run 1's catch, with 95% task-bootstrap intervals.", fontsize=7, color=INK2, ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.06, 1, 1))

    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    png = HERE / "figures" / f"fig_6a9b8621_first_command_{stamp}.png"
    with png.open("xb") as fh:
        fig.savefig(fh, format="png", dpi=180, facecolor=SURFACE)
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=HERE).stdout.strip()
    out = {"figure": "6a9b8621", "label": "POST HOC (the D3 threshold check is post hoc and descriptive); from raw data",
           "git": head, "source": {"file": Path(args.source).name, "sha256": SHA,
                                    "git": "de458d7a:experiments/2026-10-08_q3_onpolicy_figures/results/d3_threshold_POSTHOC_2026-10-11_121418.json"},
           "png": png.name, "values_by_key": values, "rows": rows, "checks": checks,
           "all_match": all(c["match"] for c in checks)}
    with (HERE / "results" / f"fig_6a9b8621_values_{stamp}.json").open("x") as fh:
        fh.write(json.dumps(out, indent=1))
    print(f"all_match {out['all_match']}; wrote {png.name}")


if __name__ == "__main__":
    main()
