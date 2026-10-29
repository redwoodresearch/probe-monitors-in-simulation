"""Seed call_end (PLAN.md here): the extraction check, as a script of its own so the as-run extractor stays unchanged
(Codex's post-launch review of run_extract_seedcall.py asked for these fixes to its check mode). Modal CPU, counts only:
  - provenance: the run's stored config.json must name the extractor bytes, run ids, render record and inputs this check
    is given (so an older dump, or other inputs with the same ids, can't pass);
  - every staged run extracted exactly once (meta files by arm = the manifest's), every chunk complete with its done
    marker, chunk index and meta agree, each layer file (rows, 5120), non-finite values (every layer from the index; first
    and last recounted), every meta holds all four readout kinds (call_end, call_end_mean, output_end, output_end_mean);
  - kept rows by arm and kind (call_end raw / primary / Seed content-only, the means, output_end);
  - reproduction (PLAN.md gate 2) BY GPU TYPE: for every GPU type that ran a shard, up to 8 seeded runs per arm with a
    reference (G3's run; for as-simulated honest runs fix-full's when the messages are byte-identical), output_end rows
    against the reference's rows at the same (step, action), all 7 layers; the gate needs every GPU type covered and min
    cosine >= 0.999;
  - per shard: GPU, load, function and forward seconds, tokens per forward second.

  PYTHONPATH=src uv run --no-project --python 3.12 --with "modal==1.6.1" modal run \\
      experiments/2026-10-13_q2_seed_callend/check_seedcall.py --render <seedcall_render_all record> --manifest <seedcall_manifest record>
"""
import hashlib
import json
import random
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).parent
APP = "probe-shift_aswin-surya_sub-q2-seedcall_check_kill-2026-11-13"
OUT = "probe-shift_aswin-surya_sub-q2-g3_acts_kill-2026-11-12"
RUN = "2026-10-13_seedcall_seed-oss-36b_v1"
EXTRACTOR = "run_extract_seedcall.py"
LAYERS, H, SEED, PER = [16, 24, 32, 40, 48, 56, 63], 5120, 20261013, 8
KINDS = {"call_end", "call_end_mean", "output_end", "output_end_mean"}

app = modal.App(APP)
out_ro = modal.Volume.from_name(OUT, version=2).read_only()
work = modal.Volume.from_name("probe-shift_aswin-surya_sub-q2-seedcall_work_kill-2026-11-13", version=2, create_if_missing=True)  # the result, so a detached run keeps it
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy")


@app.function(image=image, cpu=8, memory=32768, volumes={"/out": out_ro, "/work": work}, timeout=3600, retries=0)
def check(want: dict, expected: dict, left_out: dict, refs: dict, ids: list, stamp: str = "") -> dict:
    res = _check(want, expected, left_out, refs, ids)
    Path("/work/checks").mkdir(parents=True, exist_ok=True)
    Path(f"/work/checks/extract_check_{RUN}_{stamp}.json").write_text(json.dumps(res, indent=1) + "\n")
    work.commit()
    return res


def _check(want: dict, expected: dict, left_out: dict, refs: dict, ids: list) -> dict:
    import numpy as np
    from collections import Counter, defaultdict
    d = Path("/out") / RUN
    cfg = json.loads((d / "config.json").read_text())
    prov = {k: [cfg.get(k) if k != "render record sha256" else (cfg.get("render gate") or {}).get(k), v] for k, v in want.items()}
    prov_ok = all(a == b for a, b in prov.values())
    metas = Counter(); kept = 0; problems = Counter(); rows_total = 0; nonfinite_index = 0; nonfinite_recount = 0; kinds = Counter()
    index = {}
    for idx in sorted((d / "chunks").glob("shard*_chunk*.json")):
        c = json.loads(idx.read_text())
        problems["chunk without its done marker"] += not (d / "chunks" / f"{idx.stem}.done").exists()
        problems["chunk layers != the plan's"] += c["layers"] != LAYERS
        problems["chunk index rows != its transcripts' rows"] += c["rows"] != sum(t["n_rows"] for t in c["transcripts"])
        problems["chunk index offsets not cumulative from 0"] += [t["row_start"] for t in c["transcripts"]] != [sum(u["n_rows"] for u in c["transcripts"][:j]) for j in range(len(c["transcripts"]))]
        for t in c["transcripts"]:
            problems["duplicate index entry"] += (t["arm"], t["filename"], idx.stem) in index
            index[(t["arm"], t["filename"], idx.stem)] = (t["row_start"], t["n_rows"], t["key"])
        nonfinite_index += c["nonfinite"]
        for l in LAYERS:
            x = np.load(d / "by_layer" / f"L{l}" / f"{idx.stem}.npy", mmap_mode="r")
            problems["layer file with the wrong shape"] += x.shape != (c["rows"], H)
            if l in (LAYERS[0], LAYERS[-1]):
                nonfinite_recount += int((~np.isfinite(np.asarray(x))).sum())
        rows_total += c["rows"]
    where = {}  # (arm, stem) -> chunk
    seen_ids = Counter()
    stem_of = lambda k: k if len(k.encode()) <= 240 else f"{k[:150]}__{hashlib.sha1(k.encode()).hexdigest()[:12]}"  # noqa: E731
    for arm_dir in sorted((d / "meta").iterdir()):
        for f in arm_dir.glob("*.json"):
            m = json.loads(f.read_text()); metas[arm_dir.name] += 1; where[(arm_dir.name, f.stem)] = m["chunk"]
            seen_ids[(arm_dir.name, m["key"])] += 1
            problems["meta arm, filename or key disagree"] += m["arm"] != arm_dir.name or m["filename"] != f.stem or stem_of(m["key"]) != f.stem
            problems["meta left_out differs from the gated render's"] += m.get("left_out") != left_out.get(arm_dir.name, {}).get(m["key"])
            problems["meta without all four readout kinds"] += not KINDS <= {p["kind"] for p in m["positions"]}
            kr = [p["row"] for p in m["positions"] if p["kept"]]; kept += len(kr)
            for p in m["positions"]:
                if p["kept"]:
                    kinds[(arm_dir.name, p["kind"])] += 1
                    if p["kind"] == "call_end":
                        kinds[(arm_dir.name, "call_end primary")] += bool(p.get("primary"))
                        kinds[(arm_dir.name, "call_end Seed content-only")] += bool(p.get("content_only"))
            problems["meta rows != kept readouts"] += m["n_rows"] != len(kr)
            problems["kept rows not 0..n-1 in order"] += kr != list(range(len(kr)))
            problems["unkept position with a row"] += any(p.get("row") is not None for p in m["positions"] if not p["kept"])
            problems["meta not in its chunk's index, or row_start/n_rows/key differ"] += index.pop((arm_dir.name, m["filename"], m["chunk"]), None) != (m["row_start"], m["n_rows"], m["key"])
    problems["index entries with no live meta"] += len(index)
    problems["kept readouts (meta) != rows (chunks)"] += kept != rows_total
    problems["transcripts by arm != expected"] += dict(metas) != expected
    problems["extracted (arm, key) multiset != the manifest's"] += seen_ids != Counter((a, k) for a, k in ids)
    stats = {int(f.stem.split("_")[1]): json.loads(f.read_text()) for f in sorted((d / "stats").glob("shard_*.json"))}
    problems["shards without stats"] += len(stats) != cfg["shards"]
    before = set((cfg.get("resume") or {}).get("chunks with a done marker before", []))  # attempt 1's chunks: GPU unrecorded
    gpu_of = lambda chunk: "attempt 1 (H200 or B200, unrecorded)" if chunk in before else stats.get(int(chunk[5:8]), {}).get("gpu")  # noqa: E731
    util = [{"shard": i, "gpu": s.get("gpu"), "transcripts": s.get("transcripts"), "load s": s.get("load seconds (function start to weights on the GPU)"),
             "function s": s.get("function seconds"), "forward s": round(sum(t[1] for t in s["forward [tokens, seconds]"]), 1),
             "forward tokens": sum(t[0] for t in s["forward [tokens, seconds]"]),
             "tokens per forward second": round(sum(t[0] for t in s["forward [tokens, seconds]"]) / max(sum(t[1] for t in s["forward [tokens, seconds]"]), 1e-9))}
            for i, s in sorted(stats.items())]

    def rows_of(rd, m):
        ks = [(p["step"], p["action"]) for p in m["positions"] if p["kind"] == "output_end" and p["kept"]]
        if len(ks) != len(set(ks)):
            raise ValueError("duplicate output_end positions in a meta")
        want_ = {(p["step"], p["action"]): m["row_start"] + p["row"] for p in m["positions"] if p["kind"] == "output_end" and p["kept"]}
        got = defaultdict(dict)
        for l in LAYERS:
            x = np.load(Path("/out") / rd / "by_layer" / f"L{l}" / f"{m['chunk']}.npy", mmap_mode="r")
            for k_, row in want_.items():
                got[k_][l] = np.asarray(x[row], dtype=np.float32)
        return got
    by_gpu = defaultdict(lambda: defaultdict(list))  # gpu -> arm -> [(stem, ref run, ref arm)]
    for arm, lst in refs.items():
        for stem, ref_run, ref_arm in lst:
            if (arm, stem) in where:
                by_gpu[gpu_of(where[(arm, stem)])][arm].append((stem, ref_run, ref_arm))
    rng = random.Random(SEED)
    rep = {}
    for gpu in sorted(by_gpu, key=str):
        r_ = rep.setdefault(str(gpu), {"runs": 0, "arms": 0, "rows": 0, "rows missing in the reference": 0, "runs whose kept output_end sets differ": 0,
                                       "non-finite rows": 0, "bit-exact rows": 0, "min cosine": 1.0, "max abs diff": 0.0, "sampled": []})
        for arm in sorted(by_gpu[gpu]):
            pick = rng.sample(sorted(by_gpu[gpu][arm]), min(PER, len(by_gpu[gpu][arm])))
            r_["arms"] += bool(pick)
            for stem, ref_run, ref_arm in pick:
                m = json.loads((d / "meta" / arm / f"{stem}.json").read_text())
                g = json.loads((Path("/out") / ref_run / "meta" / ref_arm / f"{stem}.json").read_text())
                a_, b_ = rows_of(RUN, m), rows_of(ref_run, g)
                r_["runs"] += 1; r_["sampled"].append([arm, stem, ref_run, ref_arm])
                r_["runs whose kept output_end sets differ"] += set(a_) != set(b_)
                for k_, by_l in a_.items():
                    if k_ not in b_:
                        r_["rows missing in the reference"] += 1
                        continue
                    exact = True
                    for l in LAYERS:
                        u, v = by_l[l], b_[k_][l]
                        if not (np.isfinite(u).all() and np.isfinite(v).all()):
                            r_["non-finite rows"] += 1; exact = False
                            continue
                        r_["min cosine"] = min(r_["min cosine"], float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-12)))
                        r_["max abs diff"] = max(r_["max abs diff"], float(np.abs(u - v).max()))
                        exact &= bool(np.array_equal(u, v))
                    r_["rows"] += 1; r_["bit-exact rows"] += exact
    gpus_used = sorted({str(s.get("gpu")) for s in stats.values()} | ({"attempt 1 (H200 or B200, unrecorded)"} if before else set()))
    repro_ok = bool(rep) and set(gpus_used) <= set(rep) and all(v["rows"] > 0 and v["min cosine"] >= 0.999 and v["rows missing in the reference"] == 0
                                                                and v["runs whose kept output_end sets differ"] == 0 and v["non-finite rows"] == 0 for v in rep.values())
    timing = [t for s in stats.values() for t in s["forward [tokens, seconds]"]]
    return {"run": RUN, "provenance (stored config vs given)": prov, "provenance": "pass" if prov_ok else "FAIL",
            "meta files by arm": dict(metas), "expected transcripts by arm": expected, "kept readouts (meta)": kept, "rows (chunks)": rows_total,
            "non-finite values: every layer (chunk indexes)": nonfinite_index, "non-finite values: first and last layer recounted": nonfinite_recount,
            "problems": dict(problems), "kept rows by arm and kind": {f"{a} | {k}": v for (a, k), v in sorted(kinds.items())},
            "forward passes": len(timing), "forward tokens": sum(t[0] for t in timing), "forward seconds": round(sum(t[1] for t in timing), 1),
            "longest forward pass [tokens, seconds]": max(timing) if timing else None, "GPU types used": gpus_used,
            "per shard": util, "reproduction (output_end vs reference rows), by GPU type": rep, "reproduction gate": "pass" if repro_ok else "FAIL"}


def git(*args):
    return subprocess.run(["git", *args], cwd=HERE, capture_output=True, text=True, check=True).stdout.strip()


@app.local_entrypoint()
def main(render: str, manifest: str):
    for f in (Path(__file__).resolve(), HERE / EXTRACTOR, Path(render).resolve(), Path(manifest).resolve()):
        if subprocess.run(["git", "ls-files", "--error-unmatch", str(f)], cwd=HERE, capture_output=True).returncode or \
                subprocess.run(["git", "diff", "--quiet", "HEAD", "--", str(f)], cwd=HERE).returncode:
            raise SystemExit(f"{f.name} differs from HEAD: commit it before running")
    rr, man = json.loads(Path(render).read_text()), json.loads(Path(manifest).read_text())
    if not rr["gate"]["pass"] or rr["ids sha256"] != hashlib.sha256(json.dumps(sorted([[x["arm"], x["key"]] for x in man["runs"]])).encode()).hexdigest():
        raise SystemExit("the render record didn't pass, or its ids differ from the manifest's")
    want = {"source sha256": hashlib.sha256((HERE / EXTRACTOR).read_bytes()).hexdigest(), "ids sha256": rr["ids sha256"],
            "render record sha256": hashlib.sha256(Path(render).read_bytes()).hexdigest(), "inputs sha256": rr["inputs sha256"]}
    lo, refs = {}, {}
    stem = lambda k: k if len(k.encode()) <= 240 else f"{k[:150]}__{hashlib.sha1(k.encode()).hexdigest()[:12]}"  # noqa: E731
    for x in rr["runs"]:
        lo.setdefault(x["arm"], {})[x["key"]] = x["left out"]
        if x.get("reference"):
            p = Path(x["reference"]).parts  # /out/<run>/meta/<arm>/<stem>.json
            refs.setdefault(x["arm"], []).append([stem(x["key"]), p[2], p[4]])
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    print(f"the result is also written to the work Volume as checks/extract_check_{RUN}_{stamp}.json", flush=True)
    out = {"git": git("rev-parse", "HEAD"), "render record": Path(render).name, "manifest": Path(manifest).name} | check.remote(want, man["by arm"], lo, refs, [[x["arm"], x["key"]] for x in man["runs"]], stamp)
    with open(dest := HERE / "results" / f"extract_check_{RUN}_{stamp}.json", "x") as fh:
        fh.write(json.dumps(out, indent=1) + "\n")
    print(dest)
    print(json.dumps({k: out[k] for k in ("provenance", "reproduction gate", "GPU types used", "problems", "meta files by arm",
                                           "non-finite values: every layer (chunk indexes)", "non-finite values: first and last layer recounted")}, indent=1))
    if (out["provenance"] != "pass" or out["reproduction gate"] != "pass" or any(out["problems"].values())
            or out["non-finite values: every layer (chunk indexes)"] or out["non-finite values: first and last layer recounted"]):
        raise SystemExit("check failed: see the record")
