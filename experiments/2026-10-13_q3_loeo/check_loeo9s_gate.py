"""The corrected Amendment 1b reproduction gate (Codex's required fix), run on the committed result files: every reported
number of results/loeo9_2026-10-14_000238.json against results/loeo9s_2026-10-14_001947.json, now including each reader's
5-fold gate fit counts, the per-environment list's length and coverage, and the weight hashes; plus a self-test that
altered copies are caught. One detached Modal CPU function (no compute on the laptop); output to my Volume and a local copy.

  uv run --no-project --python 3.12 --with modal==1.6.1 modal run --detach experiments/2026-10-13_q3_loeo/check_loeo9s_gate.py
"""

import copy
import json
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).parent
REF, NEW = "loeo9_2026-10-14_000238.json", "loeo9s_2026-10-14_001947.json"
app = modal.App("probe-shift_aswin-surya_sub-q3loeo_gatecheck_kill-2026-11-13")
work_vol = modal.Volume.from_name("probe-shift_aswin-surya_sub-q3attack_work_kill-2026-11-06", version=2)
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("tzdata==2025.2")


def compare(ref, new):
    """(leaves compared, differences) over every reported number; exact equality, lists must match in length."""
    diffs, n = [], [0]

    def cmp(a, b, path):
        if isinstance(a, dict) and isinstance(b, dict):
            for k in sorted(set(a) | set(b)):
                if k not in a or k not in b:
                    diffs.append(f"{path}/{k}: missing on one side")
                else:
                    cmp(a[k], b[k], f"{path}/{k}")
        elif isinstance(a, list) and isinstance(b, list):
            if len(a) != len(b):
                diffs.append(f"{path}: length {len(a)} vs {len(b)}")
            for i, (x, y) in enumerate(zip(a, b)):
                cmp(x, y, f"{path}[{i}]")
        else:
            n[0] += 1
            if type(a) is not type(b) or a != b:
                diffs.append(f"{path}: {a!r} vs {b!r}"[:200])

    if sorted(ref["readers"]) != sorted(new["readers"]):
        diffs.append("readers differ")
    for m in sorted(set(ref["readers"]) & set(new["readers"])):
        r, o = ref["readers"][m], new["readers"][m]
        for key in ("results", "inputs", "df_stamp", "world", "layer", "kind"):
            cmp(r.get(key), o.get(key), f"{m}/{key}")
        for key in ("gate3", "features_sha256", "fits", "fits_at_cap"):
            cmp(r["gate"][key], o["gate"][key], f"{m}/gate/{key}")
        cmp(r["gate"]["weights"]["sha256"], o["gate"]["weights"]["sha256"], f"{m}/gate/weights_sha256")
        fr, fo = r["fits_per_env"], o["fits_per_env"]
        if [x["e"] for x in fr] != list(range(10)) or [x["e"] for x in fo] != list(range(10)):
            diffs.append(f"{m}/fits_per_env: environments not exactly 0-9 in order")
        if len(fr) != len(fo):
            diffs.append(f"{m}/fits_per_env: length {len(fr)} vs {len(fo)}")
        for x, y in zip(fr, fo):
            for key in ("e", "k", "fits", "fits_at_cap"):
                cmp(x[key], y[key], f"{m}/fits_per_env/{x['e']}/{key}")
            cmp(x["weights"]["sha256"], y["weights"]["sha256"], f"{m}/fits_per_env/{x['e']}/weights_sha256")
    cmp(ref["beside"], new["beside"], "beside")
    return n[0], diffs


@app.function(image=image, cpu=1, memory=2048, timeout=300, retries=0, volumes={"/work": work_vol})
def check(ref: dict, new: dict, head: str) -> dict:
    leaves, diffs = compare(ref, new)
    tests = {}
    m = min(new["readers"])
    for name, alter in (("gate fits 40 -> 41", lambda d: d["readers"][m]["gate"].__setitem__("fits", d["readers"][m]["gate"]["fits"] + 1)),
                        ("gate fits_at_cap +1", lambda d: d["readers"][m]["gate"].__setitem__("fits_at_cap", d["readers"][m]["gate"]["fits_at_cap"] + 1)),
                        ("drop the last per-environment record", lambda d: d["readers"][m]["fits_per_env"].pop()),
                        ("one catch value + 1e-15", lambda d: d["readers"][m]["results"]["own_env_threshold|pooled|global|fpr_0.01"].__setitem__(
                            "b_real_catch_e2", d["readers"][m]["results"]["own_env_threshold|pooled|global|fpr_0.01"]["b_real_catch_e2"] + 1e-15)),
                        ("one weight hash changed", lambda d: d["readers"][m]["fits_per_env"][3]["weights"].__setitem__("sha256", "0" * 64))):
        alt = copy.deepcopy(new)
        alter(alt)
        tests[name] = "caught" if compare(ref, alt)[1] else "MISSED"
    out = {"label": "Amendment 1b corrected reproduction gate (Codex's required fix); check, not a new result", "git": head,
           "reference": REF, "rerun": NEW, "leaves_compared": leaves, "equal": not diffs, "n_differences": len(diffs),
           "differences": diffs[:50], "self_test": tests, "self_test_all_caught": all(v == "caught" for v in tests.values()),
           "written": datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")}
    od = Path("/work/loeo")
    od.mkdir(parents=True, exist_ok=True)
    with open(od / f"check_loeo9s_gate_{out['written']}.json", "x") as fh:
        fh.write(json.dumps(out, indent=1))
    work_vol.commit()
    return out


@app.local_entrypoint()
def main():
    git = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True, cwd=HERE, check=True).stdout.strip()
    if git("status", "--porcelain", "--untracked-files=no"):
        raise SystemExit("refusing to run from a dirty tree")
    head = git("rev-parse", "HEAD")
    out = check.remote(json.loads((HERE / "results" / REF).read_text()), json.loads((HERE / "results" / NEW).read_text()), head)
    p = HERE / "results" / f"check_loeo9s_gate_{out['written']}.json"
    with p.open("x") as fh:
        fh.write(json.dumps(out, indent=1))
    print(f"wrote {p.name}; equal {out['equal']} ({out['n_differences']} differences, {out['leaves_compared']} leaves); self-test all caught {out['self_test_all_caught']}", flush=True)
