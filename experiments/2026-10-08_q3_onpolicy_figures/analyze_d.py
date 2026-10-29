"""D = G_on − G_off (amendment 15) from run_score_onpolicy.py's D-mode draws. $0, local.

G = AUROC on simulated attack calls minus AUROC on real attack calls, scored by the real-trained probe, in each phase:
G_on (Phase 2: Qwen3.5-27B acting) and G_off (Phase 1: Qwen3.5-27B reading Sol's runs), on the matched subset (b).
Each draw recomputes both on the same resampled tasks (paired). Intervals as analyze_onpolicy_2x2.py: failed draws at
the scale's floor and ceiling, more than 2.5% failed is "unreliable at this size". The standardized pAUC version is
descriptive. Descriptive at pilot size; no verdict (amendment 15).

  uv run --no-project --with numpy python analyze_d.py results/onpolicy_2x2_D_<variant>_draws0-2000_<stamp>.json
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from analyze_onpolicy_2x2 import interval

HERE = Path(__file__).parent


def main(path):
    R = json.loads(Path(path).read_text())
    raw_ids = [d["draw"] for d in R["draws"]]
    if len(raw_ids) != len(set(raw_ids)):
        raise ValueError(f"{len(raw_ids) - len(set(raw_ids))} duplicate draw ids")
    draws = {d["draw"]: d for d in R["draws"]}
    ids = sorted(draws)
    complete = ids == list(range(max(ids) + 1)) and max(ids) >= 2000
    obs, boots = draws[0], [draws[b] for b in ids if b > 0]
    v = lambda d, c, m: d["cells"][c]["values"][m]

    CELLS = {"G_on": ("on_real->on_sim", "on_real->on_real"), "G_off": ("off_real->off_sim", "off_real->off_real")}

    def gaps(d, m):
        """G_on and G_off each from its own two cells; D only when both exist."""
        g = {}
        for name, (a, b) in CELLS.items():
            x, y = v(d, a, m), v(d, b, m)
            g[name] = None if x is None or y is None else x - y
        return g["G_on"], g["G_off"], None if None in g.values() else g["G_on"] - g["G_off"]

    out = {"input": Path(path).name, "label": R["label"], "variant": R["config"]["variant"], "amendments": R["amendments"],
           "git_of_scoring": R["git"], "draws": len(boots), "bootstrap_complete": complete, "summary": R["summary"],
           "verdict": "none: descriptive at pilot size (amendment 15)", "measures": {}}
    for m in ("auroc", "pauc"):
        o = gaps(obs, m)
        res = {}
        for i, name in enumerate(("G_on", "G_off", "D = G_on − G_off")):
            cells_used = CELLS[name] if name in CELLS else CELLS["G_on"] + CELLS["G_off"]
            causes = [sum((d["cells"][c]["failed"].get(m, []) for c in cells_used), []) for d in boots]
            res[name] = {"value": None if o[i] is None else round(o[i], 4),
                         **interval([gaps(d, m)[i] for d in boots], causes, -1.0 if i < 2 else -2.0, 1.0 if i < 2 else 2.0)}
        out["measures"][m + ("" if m == "auroc" else " (descriptive)")] = res
    out["cells"] = {c: {m: obs["cells"][c]["values"][m] for m in ("auroc", "pauc")} for c in obs["cells"]}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    dst = HERE / "results" / f"d_analysis_{R['label']}_{stamp}.json"
    with dst.open("x") as fh:
        fh.write(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("label", "variant", "draws", "bootstrap_complete", "verdict")}))
    print("tasks", R["summary"]["analyzable_tasks"], "excluded", R["summary"]["excluded_tasks"], "by_world", R["summary"]["by_world"])
    for m, res in out["measures"].items():
        for name, r in res.items():
            print(m, "|", name, r["value"], r.get("ci95"), "failed", r.get("failed"), r.get("status", ""))
    for c, x in out["cells"].items():
        print("cell", c, {k: round(val, 4) if val is not None else None for k, val in x.items()})
    print("->", dst)


if __name__ == "__main__":
    main(sys.argv[1])
