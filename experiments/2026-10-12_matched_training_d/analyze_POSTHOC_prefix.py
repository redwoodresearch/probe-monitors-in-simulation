"""Item D, POST HOC (POSTHOC_RULE_prefix.md, frozen at caf3dac7): item C's exemption, the reproduction gate, then the frozen
screen() of analyze_matched_d.py on the masked token-max scores ($0, laptop; numbers only). No masked number is computed
unless every new unmasked and downstream score equals the stored one to within 1e-9.

  uv run --no-project --with numpy python experiments/2026-10-12_matched_training_d/analyze_POSTHOC_prefix.py \
      results/prefix_POSTHOC_<stamp>_check.json results/prefix_POSTHOC_<stamp>_rescore.json
"""

import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))
from analyze_matched_d import VERSIONS, screen  # noqa: E402  (frozen; imported unchanged)

TOL, MASK, N_TEST = 1e-9, 29, 58
STORED, STORED_SHA = ROOT / ".run_cache/matched_d/rows_2026-10-12_172812.json", "276a95dfa7d2c49b3017ba5216f646a4d31c589647cb28d0cae9664ca1ea48e5"
C_DIR = ROOT / "experiments/2026-10-12_cue_pilot"
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()


def c_margin():
    """Item C: each threshold (as analyze_cue_pilot.py: the calibration tasks' untouched and placement-matched neutral
    versions, sorted from the highest, 0-based index floor(0.03 n)) minus that column's minimum over all 280 versions."""
    full = json.loads((C_DIR / "results/full_2026-10-12_151458.json").read_text())
    f = ROOT / ".run_cache/cue_pilot" / full["rows"]["file"]
    if sha(f) != full["rows"]["sha256"]:
        raise ValueError("item C's rows file is not the collected one")
    rows = json.loads(f.read_text())
    cols = [("L32", lambda r: r["l32"])] + [(f"aimed L{l} column {k}", (lambda l, k: lambda r: r["aimed"][l][k])(l, k)) for l in ("2", "4") for k in range(21)]
    margins = {}
    for name, score in cols:
        lo = min(score(r) for r in rows)
        for place in ("far", "near"):
            cal = sorted((score(r) for r in rows if r["split"] == "cal" and r["version"] in ("untouched", f"neutral_{place}")), reverse=True)
            margins[f"{name} | {place}"] = cal[len(cal) * 3 // 100] - lo
    return {"thresholds checked": len(margins), "smallest margin": min(margins.values()), "where": min(margins, key=margins.get)}


if __name__ == "__main__":
    git = lambda *c: subprocess.run(["git", *c], capture_output=True, text=True, cwd=ROOT)
    st = git("status", "--porcelain", "--untracked-files=no")
    if st.returncode or st.stdout.strip() or git("ls-files", "--error-unmatch", __file__).returncode:
        raise SystemExit("commit first: the result must record the commit of the code that made it")
    chk, res = (json.loads((HERE / a).read_text()) for a in sys.argv[1:3])
    dry = json.loads((HERE / "results/dry_2026-10-12_171629.json").read_text())["result"]
    if chk["provenance"]["code_sha256"] != res["provenance"]["code_sha256"] or {chk["provenance"]["roster_sha256"], res["provenance"]["roster_sha256"]} != {dry["roster_sha256"]}:
        raise ValueError("the check and the rescore are not the same code and roster")
    c = chk["result"]
    if c["D shared prefix"] < MASK:
        raise SystemExit("item D's versions share fewer than 29 tokens: stop for an amendment (the rule)")
    out = {"git": git("rev-parse", "HEAD").stdout.strip(), "inputs": sys.argv[1:3], "check": {k: v for k, v in c.items() if k != "provenance"}}
    cm = c_margin()
    out["item C"] = cm | {"exempt": bool(c["C shared prefix"] >= MASK and c["C opening equals D's (first 29 ids)"] and cm["smallest margin"] > 0)}
    f = ROOT / ".run_cache/matched_d_prefix" / res["rows"]["file"]
    if sha(f) != res["rows"]["sha256"] or sha(STORED) != STORED_SHA:
        raise ValueError("a rows file is not the collected one")
    new_l, old_l = json.loads(f.read_text()), json.loads(STORED.read_text())
    new = {(r["split"], r["run"], r["eval"], r["sim"]): r for r in new_l}
    old = {(r["split"], r["run"], r["eval"], r["sim"]): r for r in old_l}
    want = {(s, h, e, sm) for s in ("sel", "test") for h in dry["roster"][s] for e, sm in VERSIONS}
    if len(new_l) != len(want) or len(old_l) != len(want) or set(new) != want or set(old) != want or any(new[k]["task"] != old[k]["task"] or new[k]["environment"] != old[k]["environment"] for k in want):
        raise ValueError("the rows are not exactly the dry run's roster")
    worst = max(abs(a - b) for k in want for key, per_c in old[k]["scores"].items() for ca, cb in zip(per_c, new[k]["scores"][key]) for a, b in zip(ca, cb))
    shapes = all(len(new[k]["scores"][key][i]) == len(old[k]["scores"][key][i]) for k in want for key in old[k]["scores"] for i in range(4))
    masked_n = {new[k]["masked eligible"] for k in want}
    out["reproduction gate"] = {"largest |new - stored|": worst, "tolerance": TOL, "same shapes": shapes, "masked eligible per version": sorted(masked_n),
                                "passes": bool(worst <= TOL and shapes and masked_n == {26})}  # 26 eligible positions in 0-28 (floor check)
    if out["reproduction gate"]["passes"]:
        R = {s: [] for s in ("sel", "test")}
        for k in sorted(want):
            r = dict(old[k])
            r["scores"] = {key: (new[k]["scores masked"][key] if key.endswith("|tok") else vals) for key, vals in old[k]["scores"].items()}
            R[k[0]].append(r)
        frozen = json.loads((HERE / "results/screen_2026-10-12_193321.json").read_text())["targets"]
        out["targets"] = screen(R, N_TEST)
        for name, x in out["targets"].items():
            was = "passed" if all(frozen[name]["checks"].values()) else "failed"
            failed = [g for g, ok in x["checks"].items() if not ok]
            x["reading"] = (f"with the shared opening masked, matched training gives a{'n' if name == 'evaluation' else ''} {name} cue monitor that transfers "
                            f"to new wording and these two environments (post hoc; the frozen screen {was})" if not failed else
                            f"the masked {name} readout fails the screen (post hoc; the frozen screen {was}); failed: {', '.join(failed)}")
            x["frozen screen beside"] = {k2: frozen[name][k2] for k2 in ("chosen", "test", "checks")}
    out["status"] = "post hoc; the frozen screen (results/screen_2026-10-12_193321.json) stays the primary record"
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    path = HERE / "results" / f"prefix_POSTHOC_screen_{stamp}.json"
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k != "targets"}, indent=1))
    for n, x in (out.get("targets") or {}).items():
        print(n, x["chosen"], x["test"], "null max", max(x["null E (20 refits, selection redone)"]), x["checks"], "|", x["reading"])
    print("->", path)
