"""Build the --decisive override (Append 41 decisive ΔG): for every member probe, every S_new arm folder and every H_sim root of the FROZEN
configuration, the scorer's uncut per-row file (path + sha256 from the scorer's own manifest) and its runs without rows, bound by key to
inventory v3 (cutfull folder -> registered folder). Reads manifests and listings only; the runs-without-rows lists are pulled by a Modal CPU
function that returns ONLY run keys from each scorer's per-run JSON (no score value leaves the container). Fail closed: anything missing
is listed and the override is written as INCOMPLETE (the driver then refuses it).

  PYTHONPATH=../../src uv run --no-project --python 3.12 --with modal==1.6.1 modal run build_cutfull_override.py \\
      --inventory <path>@<sha256> [--scorers sol,p2,famexp] [--dry]
"""

import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).resolve().parent
CONV = "probe-shift_aswin-surya_sub-convaudit_work_kill-2026-11-14"
APP = "probe-shift_aswin-surya_sub-cutfulldg_override_kill-2026-11-14"
SCORERS = {  # per scorer: manifest name; where runs without rows are listed: (per-run JSON name pattern, key path) or the manifest field
    "sol": {"manifest": "manifest_sol.json", "norows": ("{stem}.json", ["runs_without_reply_rows"])},         # Sol launcher (Amendment 46)
    "p2": {"manifest": "manifest_p2.json", "norows": "p2"},  # 4f's p2 @ fb9c5a37d: counts.<probe>.reply_mean.dropped; the list in a per-run JSON
    "famexp": {"manifest": "manifest_famexp.json", "norows": None}}  # famexp: the manifest's runs_without_rows field, if any
MEMBER_FILES = {  # member -> (scorer folder, per-row file name) = the basename of common455.json's p["new"] for the reply readout
    "Sol (a) R": ("sol", "a_tight__avg_top2.npz"), "FIBS R": ("p2", "fibs__reply_mean.npz"), "harmfulness R": ("p2", "harmfulness__reply_mean.npz"),
    "refusal R": ("p2", "refusal__reply_mean.npz"), "high-stakes R": ("p2", "highstakes__reply_mean.npz"),
    "own_frac10_L48 R": ("famexp", "own_frac10_L48__R.npz"), "own_frac25_L48 R": ("famexp", "own_frac25_L48__R.npz"),
    "own_frac100_L48 R": ("famexp", "own_frac100_L48__R.npz")}
MEMBER_FILES_C = {  # the command readout (uncut sensitivity, 09 @ 669ae8e09): common455.json's p["new"] basenames; Sol (a) C has no uncut file
    "FIBS C": ("p2", "fibs__command.npz"), "harmfulness C": ("p2", "harmfulness__command.npz"), "refusal C": ("p2", "refusal__command.npz"),
    "high-stakes C": ("p2", "highstakes__command.npz"), "own_frac10_L48 C": ("famexp", "own_frac10_L48__C.npz"),
    "own_frac25_L48 C": ("famexp", "own_frac25_L48__C.npz"), "own_frac100_L48 C": ("famexp", "own_frac100_L48__C.npz")}
app = modal.App(APP)
vol = modal.Volume.from_name(CONV, version=2)
image = modal.Image.debian_slim(python_version="3.12")


@app.function(image=image, cpu=2, memory=4096, timeout=900, retries=0, volumes={"/conv": modal.Volume.from_name(CONV, version=2).read_only()})
def no_rows(paths: dict) -> dict:
    """{json path: [keys without rows]} for {json path: [expected sha256, key path]}: verifies each per-run JSON's sha256 against its manifest,
    then returns ONLY the list at the key path (no score value leaves the container). A sha mismatch or absence returns an error string."""
    import hashlib as hl

    out = {}
    for p, (h, kp) in paths.items():
        f = Path("/conv") / p
        if not f.is_file():  # a manifest-listed JSON absent from the Volume: listed, never an exception (Codex r2)
            out[p] = "missing from the Volume"
            continue
        b = f.read_bytes()
        if hl.sha256(b).hexdigest() != h:
            out[p] = "sha256 mismatch"
            continue
        d = json.loads(b)
        hit = None
        for path in (kp if kp and isinstance(kp[0], list) else [kp]):  # one key path, or alternatives tried in order
            x = d
            for k in path:
                x = x.get(k) if isinstance(x, dict) else None
            if isinstance(x, list):
                hit = x
                break
        out[p] = sorted(hit) if hit is not None else "no runs-without-rows list at the expected key paths"
    return out


def sha(b):
    return hashlib.sha256(b).hexdigest()


def staged_shas(x, under=False):
    """Every 64-hex value found under a key whose name contains 'staged' (any depth), sorted."""
    import re

    out = set()
    if isinstance(x, dict):
        for k, v in x.items():
            u = under or ("staged" in str(k).lower())
            if u and isinstance(v, str) and re.fullmatch(r"[0-9a-f]{64}", v):
                out.add(v)
            out |= set(staged_shas(v, u))
    elif isinstance(x, list):
        for v in x:
            if under and isinstance(v, str) and re.fullmatch(r"[0-9a-f]{64}", v):  # Codex r8 #5: string leaves inside lists
                out.add(v)
            out |= set(staged_shas(v, under))
    return sorted(out)


@app.local_entrypoint()
def main(inventory: str, scorers: str = "sol,p2,famexp", dry: bool = False, readout: str = "R"):
    ip, ih = inventory.rsplit("@", 1)
    ib = (HERE / ip).read_bytes()
    if sha(ib) != ih:
        raise SystemExit("--inventory sha256")
    inv = json.loads(ib)
    want = {s for s in scorers.split(",") if s}
    cut2orig, wt2run = {}, {}
    for r in inv["runs"]:
        cut2orig[r["uncut_planned"]["folder"]] = r["original_folder"]
        wt2run[(r["original_folder"], r["task"])] = r["key"]  # by task: one world per folder; scorers' world tokens differ (simh_311|, sim_honest|)
    folders = sorted(set(cut2orig))
    # every scorer manifest under /cutfull/<batch>/<scorer>/<cutfull folder>/manifest_<scorer>.json
    found, manifests, dups = {}, {}, []
    for e in vol.listdir("/cutfull", recursive=True):
        parts = e.path.strip("/").split("/")
        if len(parts) == 5 and parts[0] == "cutfull" and parts[2] in want and parts[4] == SCORERS[parts[2]]["manifest"] and parts[3] in cut2orig:
            if (parts[2], parts[3]) in found:
                dups.append(f"{parts[2]}|{parts[3]}: {found[(parts[2], parts[3])]} and {e.path.strip('/')}")
            found[(parts[2], parts[3])] = e.path.strip("/")
    for k, p in found.items():
        b = b"".join(vol.read_file(p))
        manifests[k] = {"path": p, "sha256": sha(b), "json": json.loads(b)}
    ov, missing, jsons, no_staged = {"arms": {}, "simh": {}}, [], {}, set()
    if readout not in ("R", "C"):
        raise SystemExit("--readout R or C")
    p2ro = "reply_mean" if readout == "R" else "command"  # p2's counts / per-run JSON key for this readout
    for m, (sc, fn) in (MEMBER_FILES if readout == "R" else MEMBER_FILES_C).items():
        if sc not in want:
            continue
        for cf in folders:
            orig = cut2orig[cf]
            mk = (sc, cf)
            if mk not in manifests:
                missing.append(f"{m}|{cf}: no manifest_{sc}.json")
                continue
            mj = manifests[mk]["json"]
            if ih not in json.dumps(mj):
                missing.append(f"{m}|{cf}: manifest_{sc}.json does not cite inventory sha {ih[:12]}... (epoch binding unproven)")
                continue
            files = {Path(k).name: (v if isinstance(v, str) else v.get("sha256")) for k, v in (mj.get("files") or {}).items()}
            if fn not in files:
                missing.append(f"{m}|{cf}: {fn} not in the manifest")
                continue
            base = str(Path(manifests[mk]["path"]).parent)
            ent = {"path": f"{base}/{fn}", "sha256": files[fn], "manifest": {"path": manifests[mk]["path"], "sha256": manifests[mk]["sha256"]}}
            blk = (mj.get("gates") or {}).get("cut_side_registered_score_files", mj.get("cut_side_registered_score_files"))
            if blk is not None:  # Sol's Q4 manifests (under gates): the cut-side registered files, checked by the driver against the FINAL spec
                ent["cut_side_registered_score_files"] = blk
            ent["staged_shas"] = staged_shas(mj)  # every sha under a key naming 'staged' (for the re-attestation condition)
            if not ent["staged_shas"]:
                no_staged.add(f"{sc}|{cf}")
            nr_spec = SCORERS[sc]["norows"]
            if nr_spec == "p2":  # no drop: none to list; drops: the list from the per-run JSON (real arms run_means, sim-honest reply_mean)
                pr_ = fn.split("__", 1)[0]
                dropped = (((mj.get("counts") or {}).get(pr_) or {}).get(p2ro) or {}).get("dropped")
                if dropped is None:
                    missing.append(f"{m}|{cf}: manifest_p2.json lacks counts.{pr_}.{p2ro}.dropped")
                    continue
                if dropped == 0:
                    ent["runs_without_rows"] = []
                else:
                    cand_ = (f"{pr_}__run_means.json", f"{pr_}__reply_mean.json") + ((f"{pr_}__command.json",) if readout == "C" else ())
                    js = next((j for j in cand_ if j in files), None)  # Codex r12 #1: sim-honest packs list drops in <probe>__reply_mean.json
                    if js is None:
                        missing.append(f"{m}|{cf}: {dropped} dropped runs but no per-run JSON in the manifest")
                        continue
                    kp_ = [["dropped", p2ro]] if readout == "C" else [["dropped", "reply_mean"], ["dropped"], ["runs_without_rows"]]
                    jsons[f"{base}/{js}"] = (m, orig, ent, files[js], kp_, dropped)
            elif nr_spec:
                js = nr_spec[0].format(stem=fn.removesuffix(".npz"), probe=fn.split("__", 1)[0])
                if js not in files:
                    missing.append(f"{m}|{cf}: {js} (runs without rows) not in the manifest")
                    continue
                jsons[f"{base}/{js}"] = (m, orig, ent, files[js], nr_spec[1])
            else:  # listed in the manifest itself: famexp keys runs_without_rows by file stem (<probe>__<readout>) -> [<world>|<task>]
                nr_ = mj.get("runs_without_rows")
                if not isinstance(nr_, dict) or fn.removesuffix(".npz") not in nr_:  # Codex r8 #4: a bare list is not per-file evidence
                    missing.append(f"{m}|{cf}: manifest_{sc}.json runs_without_rows has no per-file entry for {fn}")
                    continue
                nr_ = nr_[fn.removesuffix(".npz")]
                if not isinstance(nr_, list):
                    missing.append(f"{m}|{cf}: manifest_{sc}.json runs_without_rows is not a list for {fn}")
                    continue
                ent["runs_without_rows"] = sorted({wt2run.get((orig, k.split("|", 1)[-1]), k) for k in nr_})
                unk_ = [k for k in ent["runs_without_rows"] if k not in {v for (fo, _), v in wt2run.items() if fo == orig}]
                if unk_:  # same rule as the per-run JSON path: only this folder's inventory runs
                    missing.append(f"{m}|{orig}: runs without rows not in this folder's inventory: {unk_[:3]}")
            key_new = sc  # the scorer folder is the config's file prefix (sol/, p2/, famexp/)
            pnew = f"{key_new}/{fn}"
            if orig.endswith("__sim_honest"):
                ent.update({"batch": orig.split("_sim")[0], "root": orig})  # the same dict object, so runs_without_rows lands in it (Codex r1 #1)
                ov["simh"].setdefault(pnew, []).append(ent)
            else:
                ov["arms"].setdefault(orig, {})[pnew] = ent
    if dry:  # coverage only: no container, no file written (duplicates counted separately, Codex r7/r8 optional)
        missing += [f"duplicate manifests: {d}" for d in dups]
        print(json.dumps({"dry": True, "manifests": len(manifests), "missing": len(missing), "missing_examples": missing[:8],
                          "duplicates": len(dups), "duplicate_examples": dups[:4],
                          "by_scorer": {sc: sum(1 for k in manifests if k[0] == sc) for sc in sorted(want)}, "folders": len(folders),
                          "manifests_without_staged_sha": len(no_staged), "without_staged_examples": sorted(no_staged)[:6]}, indent=1))
        return
    # runs without rows: run keys only, from inside a container
    if jsons:
        nr = no_rows.remote({p: [v[3], v[4]] for p, v in sorted(jsons.items())})
        inv_runs = {}
        for (fo, _), rk in wt2run.items():
            inv_runs.setdefault(fo, set()).add(rk)
        for p, keys in nr.items():
            m, orig, ent = jsons[p][:3]
            if isinstance(keys, str):
                missing.append(f"{m}|{orig}: per-run JSON {keys}")
                continue
            if len(jsons[p]) > 5 and len(keys) != jsons[p][5]:
                missing.append(f"{m}|{orig}: {len(keys)} listed runs without rows, the manifest counts {jsons[p][5]}")
            conv = [wt2run.get((orig, k.split("|", 1)[-1]), k) for k in keys]  # <world>|<task> -> the inventory's run_key (by task)
            unknown = [k for k in conv if k not in inv_runs.get(orig, set())]  # Codex r1 #5: only this folder's inventory runs
            if unknown:
                missing.append(f"{m}|{orig}: runs without rows not in this folder's inventory: {unknown[:3]}")
            ent["runs_without_rows"] = sorted(conv)
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    missing += [f"duplicate manifests: {d}" for d in dups]
    out = {"label": "Append 41 decisive ΔG override (uncut window inputs); keys, paths and sha256s only", "status": "COMPLETE" if not missing else "INCOMPLETE",
           "inventory": {"path": ip, "sha256": ih}, "readout": readout, "scorers": sorted(want), "missing": missing, "n_manifests": len(manifests),
           "manifests_without_staged_sha": sorted(no_staged),
           "arms": ov["arms"], "simh": ov["simh"]}
    p = HERE / "config" / "cutfull" / (f"override_{stamp}.json" if readout == "R" else f"override_C_{stamp}.json")
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("x") as fh:
        fh.write(json.dumps(out, indent=1))
    print(json.dumps({"override": str(p.relative_to(HERE)), "sha256": sha(p.read_bytes()), "status": out["status"], "missing": len(missing),
                      "manifests": len(manifests), "arms": len(ov["arms"]), "simh_probe_files": {k: len(v) for k, v in ov["simh"].items()}}, indent=1))
