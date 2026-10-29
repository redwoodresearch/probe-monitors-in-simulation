"""Cue exclusion, step 0 (RULE.md here), POST HOC, $0-scale Modal CPU: the cue-bearing run list for the on-policy staging.

The cue (09, format_smell_2026-10-14_102229 @ 49f48e38): a GENUINE user turn (not the first, not a tool output) before the last
tool output. Two lists, which must be equal:
  (1) 09's per-run rows (Volume probe-shift_aswin-surya_sub-convaudit_work_kill-2026-11-14:/format_smell/rows_2026-10-14_102229.jsonl,
      sha256 pinned to the result JSON's rows_sha256): staging "d9-onpol", render "qwen3", user_turn_before_last_tool true;
  (2) this script's STRUCTURAL check on the staged messages (2026-10-14_q3onpol_q35own_inputs_v1): message roles and positions
      only (system / user / tool / assistant); a user message other than the first that sits before the last tool message.
Also counts the cue in the repaired simulated copies (fullF9, full) and checks their keys' flags equal the as-is simulated runs'.
Never reads or prints message content; returns keys, arms and counts.

  PYTHONPATH=src uv run --no-project --python 3.12 --with modal==1.6.1 modal run --detach \\
      experiments/2026-10-14_q3_onpol_cue_exclusion/run_cue_step0.py
"""
import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).resolve().parent
APP = "probe-shift_aswin-surya_sub-q3onpol_cuestep0_kill-2026-11-13"
OUTV = "probe-shift_aswin-surya_sub-q3onpol_work_kill-2026-11-13"
ROWS, ROWS_SHA = "/conv/format_smell/rows_2026-10-14_102229.jsonl", "9960167498564c7ac7bbea17bfb5bd3fc467a10cf536db0f28fa7cd133bb81e3"
STAGED = {"main": "2026-10-14_q3onpol_q35own_inputs_v1", "fullF9": "2026-10-14_q3onpol_q35own_repair_fullF9_inputs_v1",
          "full": "2026-10-14_q3onpol_q35own_repair_full_inputs_v1"}

app = modal.App(APP)
vols = {"/acts": modal.Volume.from_name("probe-shift_aswin-surya_sub-la-activations", version=2).read_only(),
        "/conv": modal.Volume.from_name("probe-shift_aswin-surya_sub-convaudit_work_kill-2026-11-14", version=2).read_only()}
out_vol = modal.Volume.from_name(OUTV, version=2)
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("tzdata==2025.2")


def cue_of(roles: list) -> bool:
    """Roles only: a user message other than the first that comes before the last tool message."""
    tools = [i for i, r in enumerate(roles) if r == "tool"]
    users = [i for i, r in enumerate(roles) if r == "user"]
    if not tools:
        return False
    return any(i < tools[-1] for i in users[1:])


@app.function(image=image, cpu=4, memory=16384, timeout=1800, retries=0, volumes={**vols, "/out": out_vol})
def step0(git: str, stamp: str) -> dict:
    import collections
    roles_ok = {"system", "user", "tool", "assistant"}
    mine, per_set = {}, {}
    for name, d in STAGED.items():
        files = sorted((Path("/acts") / d / "inputs").glob("shard_*.jsonl"))
        if not files:
            raise RuntimeError(f"{name}: no staged shards")
        cnt = collections.Counter()
        flags = {}
        for p in files:
            for line in p.read_text().splitlines():
                rec = json.loads(line)
                roles = [m["role"] for m in rec["messages"]]
                if not set(roles) <= roles_ok:
                    raise RuntimeError(f"{name}: an unexpected role")
                f = cue_of(roles)
                k = (rec["arm"], rec["key"])
                if k in flags:
                    raise RuntimeError(f"{name}: duplicate arm/key")
                flags[k] = f
                cnt[(rec["arm"], f)] += 1
        per_set[name] = {"files": len(files), "files_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
                         "by_arm": {a: {"runs": cnt[(a, True)] + cnt[(a, False)], "cue": cnt[(a, True)]}
                                    for a in sorted({a for a, _ in cnt})}}
        mine[name] = flags
    # 09's rows
    if hashlib.sha256(Path(ROWS).read_bytes()).hexdigest() != ROWS_SHA:
        raise RuntimeError("09's rows file differs from its pinned sha256")
    nine = {}
    for line in Path(ROWS).read_text().splitlines():
        r = json.loads(line)
        if r["staging"] == "d9-onpol" and r["render"] == "qwen3":
            nine[(r["arm"], r["key"])] = bool(r["user_turn_before_last_tool"])
    mine_main = mine["main"]
    only_mine = sorted(k for k in mine_main if k not in nine)
    only_nine = sorted(k for k in nine if k not in mine_main)
    disagree = sorted(k for k in mine_main if k in nine and mine_main[k] != nine[k])
    cue_keys = sorted(k for k, f in mine_main.items() if f)
    # repaired copies: same key as the as-is simulated run; the cue must be unchanged (repairs edit tool outputs only)
    sim_cue = {k[1]: f for k, f in mine_main.items() if k[0] == "q35-sim-attack"}
    simh_cue = {k[1]: f for k, f in mine_main.items() if k[0] == "q35-sim-honest"}
    rep_vs_asis = {}
    for name in ("fullF9", "full"):
        diff = 0
        for (a, k), f in mine[name].items():
            ref = sim_cue.get(k) if "attack" in a else simh_cue.get(k)
            if ref is None or ref != f:
                diff += 1
        rep_vs_asis[name] = {"copies": len(mine[name]), "flag differs from the as-is run (or no as-is run)": diff}
    out = {"label": "POST HOC (RULE.md in experiments/2026-10-14_q3_onpol_cue_exclusion); step 0, structural, keys and counts only",
           "git": git, "stamp": stamp, "rows_09": {"path": ROWS, "sha256": ROWS_SHA, "d9_onpol_qwen3_rows": len(nine),
                                                     "cue_true": sum(nine.values())},
           "per_set": per_set, "lists_equal": not (only_mine or only_nine or disagree),
           "only_in_mine": only_mine, "only_in_09": only_nine, "flag_disagreements": disagree,
           "repaired_vs_asis": rep_vs_asis, "cue_runs": [{"arm": a, "key": k} for a, k in cue_keys],
           "cue_by_arm": dict(collections.Counter(a for a, _ in cue_keys))}
    d = Path("/out/cue_excl")
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"cue_step0_{stamp}.json"
    p.write_text(json.dumps(out, indent=1))
    out_vol.commit()
    out["file"] = str(p)
    out["file_sha256"] = hashlib.sha256(p.read_bytes()).hexdigest()
    return out


@app.local_entrypoint()
def main():
    if subprocess.run(["git", "diff", "--quiet", "HEAD", "--", str(HERE)], cwd=HERE).returncode:
        raise SystemExit("commit this folder before running it")
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=HERE).stdout.strip()
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    out = step0.remote(git, stamp)
    (HERE / "results" / f"cue_step0_{stamp}.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("lists_equal", "cue_by_arm", "repaired_vs_asis", "file_sha256")}
                     | {"per_set_by_arm": {n: v["by_arm"] for n, v in out["per_set"].items()},
                        "rows_09": out["rows_09"], "n_only_mine": len(out["only_in_mine"]), "n_only_09": len(out["only_in_09"]),
                        "n_disagree": len(out["flag_disagreements"])}, indent=1))
