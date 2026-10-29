"""d9's CUT fork (Append 41): the staged-input prefix check of a cutfull staging against its registered 32k staging, EVERY run of
every arm (no activations; Modal CPU, never the laptop). Endorsed by 2e for the S_old (b) sets restaged by stage_chain.py (its
stage_q4b.py copy differs from the q4 copy), and run on the chain-target sets too. Token ids, positions and keys only.

Per pair (C = the registered cut staged set on its Volume, U = the cutfull staged set on the q3cutfull Volume), per arm:
1. keys: C's staged keys == U's staged keys (unique on both sides).
2. per run, L = len(C ids): U ids[:L] == C ids except inside the last 3 pre-boundary positions [L-3, L) when U is longer (listed).
3. per run, units (a call's (irid, kind, src_irid, call_index) with its readout, and for output_end its span): every C unit with a
   readout before L-3 has the same readout and span in U; a C readout inside [L-3, L) is listed and excluded but its span must start
   where U's does; a unit with no C readout may have a U readout only at or past L-3 (inside the zone: listed). U's calls are C's
   calls (same identities, same order). Nothing past L-3 excuses a difference before it.
4. len(U) == the run's full render length as the REGISTERED staging's own stage record recorded it (per_run tokens_before),
   independent of U's stager; C ids at most 32,768, equal to U's length where C was not cut.
5. the spec's arms == every arm file of both stagings (repro-c aside), except arms listed in dropped_arms, which exist in the
   registered staging only; no arm empty.
Any failure: the pair FAILS (listed by key); the command exits nonzero after writing the record.

  CUTPREFIX_SPEC=<path of a json file {"pairs": [{"cut_mount": "/chain"|"/q4own"|"/rh"|"/onpol", "cut_set", "uncut_set", "arms": [...]}]}> \
  uv run --no-project --python 3.12 --with modal==1.6.1 modal run experiments/2026-10-14_q3_overnight_chain/cutfull/prefix_check.py
"""

import hashlib
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).resolve().parent
CHAIN = HERE.parent
REPO = CHAIN.parents[1] if len(CHAIN.parents) > 1 else CHAIN  # import-safe inside the Modal container (/root)
APP = "probe-shift_aswin-surya_sub-q3cutfull_prefix_kill-2026-11-14"
ZONE = 3
STAGE_REC = {}
MOUNTS = {"/chain": "probe-shift_aswin-surya_sub-q3su311_acts_kill-2026-11-14", "/q4own": "probe-shift_aswin-surya_sub-q4own_evals_kill-2026-11-13",
          "/rh": "probe-shift_aswin-surya_sub-q3realhonest_data_kill-2026-11-14", "/onpol": "probe-shift_aswin-surya_sub-q3onpol_work_kill-2026-11-13",
          "/unc": "probe-shift_aswin-surya_sub-q3cutfull_acts_kill-2026-11-14"}
app = modal.App(APP)
image = modal.Image.debian_slim(python_version="3.12")
VOLS = {m: modal.Volume.from_name(n).read_only() for m, n in MOUNTS.items()}


def read(path: Path) -> tuple[dict, str]:
    raw = path.read_bytes()
    eps = [json.loads(line) for line in raw.decode().splitlines()]
    by = {e["key"]: e for e in eps}
    if len(by) != len(eps):
        raise ValueError(f"{path}: duplicate keys")
    return by, hashlib.sha256(raw).hexdigest()


def units(calls: list) -> dict:
    out = {}
    for c in calls:
        u = (c["irid"], c["kind"], c["src_irid"], c["call_index"])
        if u in out:
            raise ValueError(f"duplicate unit {u}")
        out[u] = (c["readout"], tuple(c["span"]) if c.get("span") else None)
    return out


def touches(r, span, lo, hi) -> bool:
    return (r is not None and lo <= r < hi) or (span is not None and span[0] < hi and span[1] > lo)


def full_lengths(root: Path) -> dict:
    """run key -> the registered staging's own pre-cut render length (per_run tokens_before of its one stage record)."""
    recs = sorted([*root.glob("stage_2*.json"), *root.glob("inputs/stage_2*.json")])
    if len(recs) != 1:
        raise ValueError(f"{root}: {len(recs)} stage records")
    raw = recs[0].read_bytes()
    rec = json.loads(raw)
    STAGE_REC[str(root)] = {"name": recs[0].name, "sha256": hashlib.sha256(raw).hexdigest(), "staged_inputs_sha256": rec.get("staged_inputs_sha256"), "epochs": rec.get("epochs")}
    per_run = rec["per_run"]
    rows = [(f"{r['run']}#e0", r["tokens_before"]) for rs in per_run.values() for r in rs if "excluded" not in r]
    if len({k for k, _ in rows}) != len(rows):
        raise ValueError(f"{recs[0]}: duplicate run records")
    return dict(rows)


@app.function(image=image, cpu=4, memory=32768, timeout=3600, retries=0, volumes=VOLS)
def check_pair(pair: dict) -> dict:
    res = {**pair, "arms_checked": {}, "fail": []}
    c_root, u_root = Path(pair["cut_mount"]) / pair["cut_set"], Path("/unc") / pair["uncut_set"]
    arms_on_disk = {side: sorted(f.stem for f in (r / "inputs").glob("*.jsonl") if f.stem != "repro-c") for side, r in (("cut", c_root), ("uncut", u_root))}
    dropped = pair.get("dropped_arms", [])  # arms of the registered staging deliberately NOT restaged (listed, e.g. q4f_main's pilot)
    if not pair["arms"] or set(dropped) & set(pair["arms"]) or sorted(pair["arms"] + dropped) != arms_on_disk["cut"] or sorted(pair["arms"]) != arms_on_disk["uncut"]:
        res["fail"].append(f"arms: spec {sorted(pair['arms'])}, on disk {arms_on_disk}")  # every arm of both stagings, no more, no fewer
    full = full_lengths(c_root)
    res["cut_stage_record"] = STAGE_REC[str(c_root)]  # the registered staging's own stage record: name, sha256, its staged hashes
    for arm in pair["arms"]:
        C, c_sha = read(c_root / "inputs" / f"{arm}.jsonl")
        U, u_sha = read(u_root / "inputs" / f"{arm}.jsonl")
        a = {"cut_sha": c_sha, "uncut_sha": u_sha, "runs": len(C), "keys_equal": set(C) == set(U), "bad": [], "zone_ids": {}, "excluded_units": {}}
        if not C:
            a["bad"].append(["", "empty arm"])
        if not a["keys_equal"]:
            a["bad"] += [[k, "missing in uncut"] for k in sorted(set(C) - set(U))] + [[k, "extra in uncut"] for k in sorted(set(U) - set(C))]
        for k in sorted(set(C) & set(U)):
            c, u = C[k], U[k]
            L, n = len(c["ids"]), len(u["ids"])
            longer, lo = n > L, L - ZONE
            if L > 32768 or k not in full or n != full[k] or (L < 32768 and n != L) or (L == 32768 and n < L):
                a["bad"].append([k, f"length: cut {L}, uncut {n}, registered full {full.get(k)}"])  # U is the WHOLE render, by the C record
                continue
            diff = [i for i in range(L) if c["ids"][i] != u["ids"][i]]
            if [i for i in diff if not (longer and i >= lo)]:
                a["bad"].append([k, "ids"])
                continue
            if diff:
                a["zone_ids"][k] = diff
            cu, uu = units(c["calls"]), units(u["calls"])
            if list(cu) != list(uu):
                a["bad"].append([k, "call identities"])
                continue
            ex, bad = [], None
            for x, (rc, sc) in cu.items():
                ru, su = uu[x]
                if not longer:  # nothing past L: identical units
                    ok = (rc, sc) == (ru, su)
                elif rc is not None and rc < lo:  # C's readout before the zone: identical readout and span (the span ends at rc + 1)
                    ok = (rc, sc) == (ru, su)
                elif rc is not None:  # C's readout inside the zone: listed and excluded, but a span must still START where C's does
                    ex.append(str(x))
                    ok = sc is None or sc[0] >= lo or (su is not None and su[0] == sc[0])
                else:  # no readout in C: U's readout, if any, lies in the zone or past L
                    ok = ru is None or ru >= lo
                    if ok and ru is not None and ru < L:
                        ex.append(str(x))
                if not ok:
                    bad = x
                    break
            if bad is not None:
                a["bad"].append([k, f"unit {bad}"])
            if ex:
                a["excluded_units"][k] = ex
        res["arms_checked"][arm] = a
        if a["bad"]:
            res["fail"].append(arm)
    res["passes"] = not res["fail"]
    return res


@app.local_entrypoint()
def main():
    spec_path = Path(os.environ["CUTPREFIX_SPEC"]).resolve()
    spec = json.loads(spec_path.read_text())
    if any(p["cut_mount"] not in MOUNTS or p["cut_mount"] == "/unc" for p in spec["pairs"]):
        raise SystemExit("cut_mount must be one of the registered stagings' Volumes")
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=REPO).stdout.strip()
    rows = list(check_pair.map(spec["pairs"], return_exceptions=True))
    res = [r if isinstance(r, dict) else {**p, "error": repr(r)[:300], "passes": False} for r, p in zip(rows, spec["pairs"])]
    ok = bool(res) and all(r.get("passes") for r in res)
    out = {"what": "cutfull staged-input prefix check vs the registered 32k staging (Append 41)", "git": head, "spec": str(spec_path.relative_to(REPO)),
           "spec_sha256": hashlib.sha256(spec_path.read_bytes()).hexdigest(), "zone": ZONE, "pairs": res, "n_pairs": len(res),
           "n_pass": sum(bool(r.get("passes")) for r in res), "passes": ok}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    dest = CHAIN / "results" / f"cutfull_prefix_check_{stamp}.json"
    dest.open("x").write(json.dumps(out, indent=1))
    print(dest.name, "PASS" if ok else "FAIL", f"{out['n_pass']}/{out['n_pairs']}")
    if not ok:
        raise SystemExit(f"prefix check FAILED: {[(r['uncut_set'], r.get('fail') or r.get('error')) for r in res if not r.get('passes')]}")
