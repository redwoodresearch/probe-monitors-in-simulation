"""Seed call_end (PLAN.md here): check_seedcall.py's checks, fanned out (the single-container check ran past 15 minutes
with no progress output; 0d via q2, 10-13 about 22:15: "relaunch fanned out ... Merge the parts in a final step and write
to the work Volume. 16+ CPU cores each"). Same checks, same pass rules; only the work is split:
  - chunk parts (16): per group of chunks, the done marker, the chunk index (rows, cumulative offsets), each layer file's
    shape, and non-finite values (from the index; the first and last layer recounted), files read by 32 threads;
  - meta parts (one per arm): identities (arm, filename, key), the gated render's left-out rows, all four readout kinds,
    kept rows 0..n-1, kept rows by kind (call_end raw / primary / Seed content-only, the means, output_end);
  - one reproduction part: output_end rows of up to 8 seeded runs per arm per GPU type against the reference rows (G3's
    run; for as-simulated honest runs fix-full's when byte-identical), every layer; equal kept sets, finite, min cosine
    >= 0.999, every GPU type used covered (attempt 1's chunks are their own group: their GPU type is unrecorded);
  - merge (the remote orchestrator): provenance (the stored config vs the given extractor sha256, ids, render record,
    inputs), every meta in its chunk's index with equal row_start / n_rows / key, no index entry without a meta, kept rows
    = chunk rows, transcripts by arm and the (arm, key) multiset = the manifest's, one stats file per shard, per-shard
    timing; the result goes to the work Volume and back to the client.

  PYTHONPATH=src uv run --no-project --python 3.12 --with "modal==1.6.1" modal run --detach \\
      experiments/2026-10-13_q2_seed_callend/check_seedcall_fan.py --render <seedcall_render_all record> --manifest <seedcall_manifest record>
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
APP = "probe-shift_aswin-surya_sub-q2-seedcall_checkfan_kill-2026-11-13"
OUT = "probe-shift_aswin-surya_sub-q2-g3_acts_kill-2026-11-12"
RUN = "2026-10-13_seedcall_seed-oss-36b_v1"
EXTRACTOR = "run_extract_seedcall.py"
LAYERS, H, SEED, PER, GROUPS, THREADS = [16, 24, 32, 40, 48, 56, 63], 5120, 20261013, 8, 16, 32
KINDS = {"call_end", "call_end_mean", "output_end", "output_end_mean"}
ATT1 = "attempt 1 (H200 or B200, unrecorded)"

app = modal.App(APP)
out_ro = modal.Volume.from_name(OUT, version=2).read_only()
work = modal.Volume.from_name("probe-shift_aswin-surya_sub-q2-seedcall_work_kill-2026-11-13", version=2, create_if_missing=True)
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy")


def pmap(f, xs):
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(THREADS) as ex:
        return list(ex.map(f, xs))


@app.function(image=image, cpu=16, memory=65536, volumes={"/out": out_ro}, timeout=3600, retries=0)
def chunk_part(stems: list[str]) -> dict:
    import numpy as np
    from collections import Counter
    d = Path("/out") / RUN

    def one(stem):
        p = Counter()
        c = json.loads((d / "chunks" / f"{stem}.json").read_text())
        p["chunk without its done marker"] += not (d / "chunks" / f"{stem}.done").exists()
        p["chunk layers != the plan's"] += c["layers"] != LAYERS
        p["chunk index rows != its transcripts' rows"] += c["rows"] != sum(t["n_rows"] for t in c["transcripts"])
        p["chunk index offsets not cumulative from 0"] += [t["row_start"] for t in c["transcripts"]] != [sum(u["n_rows"] for u in c["transcripts"][:j]) for j in range(len(c["transcripts"]))]
        rec = 0
        for l in LAYERS:
            x = np.load(d / "by_layer" / f"L{l}" / f"{stem}.npy", mmap_mode="r")
            p["layer file with the wrong shape"] += x.shape != (c["rows"], H)
            if l in (LAYERS[0], LAYERS[-1]):
                rec += int((~np.isfinite(np.asarray(x))).sum())
        idx = [[t["arm"], t["filename"], stem, t["row_start"], t["n_rows"], t["key"]] for t in c["transcripts"]]
        return p, idx, c["rows"], c["nonfinite"], rec
    problems, index, rows, nf_i, nf_r = Counter(), [], 0, 0, 0
    for p, idx, r, a, b in pmap(one, stems):
        problems.update(p); index += idx; rows += r; nf_i += a; nf_r += b
    return {"problems": dict(problems), "index": index, "rows": rows, "nonfinite index": nf_i, "nonfinite recount": nf_r}


@app.function(image=image, cpu=16, memory=65536, volumes={"/out": out_ro}, timeout=3600, retries=0)
def meta_part(arm: str, left_out: dict) -> dict:
    from collections import Counter
    d = Path("/out") / RUN / "meta" / arm
    stem_of = lambda k: k if len(k.encode()) <= 240 else f"{k[:150]}__{hashlib.sha1(k.encode()).hexdigest()[:12]}"  # noqa: E731

    def one(f):
        p, kinds = Counter(), Counter()
        m = json.loads(f.read_text())
        p["meta arm, filename or key disagree"] += m["arm"] != arm or m["filename"] != f.stem or stem_of(m["key"]) != f.stem
        p["meta left_out differs from the gated render's"] += m.get("left_out") != left_out.get(m["key"])
        p["meta without all four readout kinds"] += not KINDS <= {q["kind"] for q in m["positions"]}
        kr = [q["row"] for q in m["positions"] if q["kept"]]
        for q in m["positions"]:
            if q["kept"]:
                kinds[q["kind"]] += 1
                if q["kind"] == "call_end":
                    kinds["call_end primary"] += bool(q.get("primary"))
                    kinds["call_end Seed content-only"] += bool(q.get("content_only"))
        p["meta rows != kept readouts"] += m["n_rows"] != len(kr)
        p["kept rows not 0..n-1 in order"] += kr != list(range(len(kr)))
        p["unkept position with a row"] += any(q.get("row") is not None for q in m["positions"] if not q["kept"])
        return p, kinds, [arm, m["filename"], m["chunk"], m["row_start"], m["n_rows"], m["key"]], len(kr)
    problems, kinds, entries, kept = Counter(), Counter(), [], 0
    for p, k, e, n in pmap(one, sorted(d.glob("*.json"))):
        problems.update(p); kinds.update(k); entries.append(e); kept += n
    return {"arm": arm, "problems": dict(problems), "kinds": dict(kinds), "entries": entries, "kept": kept}


@app.function(image=image, cpu=16, memory=65536, volumes={"/out": out_ro}, timeout=3600, retries=0)
def repro_part(refs: dict) -> dict:
    import numpy as np
    from collections import defaultdict
    d = Path("/out") / RUN
    cfg = json.loads((d / "config.json").read_text())
    before = set((cfg.get("resume") or {}).get("chunks with a done marker before", []))
    stats = {int(f.stem.split("_")[1]): json.loads(f.read_text()) for f in sorted((d / "stats").glob("shard_*.json"))}
    where = {}
    for c, stem in zip(pmap(lambda f: json.loads(f.read_text()), sorted((d / "chunks").glob("shard*_chunk*.json"))),
                       [f.stem for f in sorted((d / "chunks").glob("shard*_chunk*.json"))]):
        for t in c["transcripts"]:
            where[(t["arm"], t["filename"])] = stem
    gpu_of = lambda chunk: ATT1 if chunk in before else stats.get(int(chunk[5:8]), {}).get("gpu")  # noqa: E731

    def rows_of(rd, m):
        ks = [(q["step"], q["action"]) for q in m["positions"] if q["kind"] == "output_end" and q["kept"]]
        if len(ks) != len(set(ks)):
            raise ValueError("duplicate output_end positions in a meta")
        want = {(q["step"], q["action"]): m["row_start"] + q["row"] for q in m["positions"] if q["kind"] == "output_end" and q["kept"]}
        got = defaultdict(dict)
        for l in LAYERS:
            x = np.load(Path("/out") / rd / "by_layer" / f"L{l}" / f"{m['chunk']}.npy", mmap_mode="r")
            for k_, row in want.items():
                got[k_][l] = np.asarray(x[row], dtype=np.float32)
        return got
    by_gpu = defaultdict(lambda: defaultdict(list))
    for arm, lst in refs.items():
        for stem, ref_run, ref_arm in lst:
            if (arm, stem) in where:
                by_gpu[str(gpu_of(where[(arm, stem)]))][arm].append((stem, ref_run, ref_arm))
    rng = random.Random(SEED)
    jobs = []
    for gpu in sorted(by_gpu):
        for arm in sorted(by_gpu[gpu]):
            for stem, ref_run, ref_arm in rng.sample(sorted(by_gpu[gpu][arm]), min(PER, len(by_gpu[gpu][arm]))):
                jobs.append((gpu, arm, stem, ref_run, ref_arm))

    def one(j):
        gpu, arm, stem, ref_run, ref_arm = j
        m = json.loads((d / "meta" / arm / f"{stem}.json").read_text())
        g = json.loads((Path("/out") / ref_run / "meta" / ref_arm / f"{stem}.json").read_text())
        a_, b_ = rows_of(RUN, m), rows_of(ref_run, g)
        r = {"rows": 0, "missing": 0, "differ": set(a_) != set(b_), "nonfinite": 0, "exact": 0, "cos": 1.0, "diff": 0.0}
        for k_, by_l in a_.items():
            if k_ not in b_:
                r["missing"] += 1
                continue
            exact = True
            for l in LAYERS:
                u, v = by_l[l], b_[k_][l]
                if not (np.isfinite(u).all() and np.isfinite(v).all()):
                    r["nonfinite"] += 1; exact = False
                    continue
                r["cos"] = min(r["cos"], float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-12)))
                r["diff"] = max(r["diff"], float(np.abs(u - v).max()))
                exact &= bool(np.array_equal(u, v))
            r["rows"] += 1; r["exact"] += exact
        return j, r
    rep = {}
    for (gpu, arm, stem, ref_run, ref_arm), r in pmap(one, jobs):
        x = rep.setdefault(gpu, {"runs": 0, "arms": set(), "rows": 0, "rows missing in the reference": 0, "runs whose kept output_end sets differ": 0,
                                 "non-finite rows": 0, "bit-exact rows": 0, "min cosine": 1.0, "max abs diff": 0.0, "sampled": []})
        x["runs"] += 1; x["arms"].add(arm); x["sampled"].append([arm, stem, ref_run, ref_arm])
        x["rows"] += r["rows"]; x["rows missing in the reference"] += r["missing"]; x["runs whose kept output_end sets differ"] += r["differ"]
        x["non-finite rows"] += r["nonfinite"]; x["bit-exact rows"] += r["exact"]
        x["min cosine"] = min(x["min cosine"], r["cos"]); x["max abs diff"] = max(x["max abs diff"], r["diff"])
    for x in rep.values():
        x["arms"] = len(x["arms"])
    gpus_used = sorted({str(s.get("gpu")) for s in stats.values()} | ({ATT1} if before else set()))
    ok = bool(rep) and set(gpus_used) <= set(rep) and all(v["rows"] > 0 and v["min cosine"] >= 0.999 and v["rows missing in the reference"] == 0
                                                          and v["runs whose kept output_end sets differ"] == 0 and v["non-finite rows"] == 0 for v in rep.values())
    return {"rep": rep, "gpus used": gpus_used, "pass": ok}


@app.function(image=image, cpu=4, memory=16384, volumes={"/out": out_ro, "/work": work}, timeout=3600, retries=0)
def orchestrate(want: dict, expected: dict, left_out: dict, refs: dict, ids: list, stamp: str) -> dict:
    from collections import Counter
    d = Path("/out") / RUN
    cfg = json.loads((d / "config.json").read_text())
    prov = {k: [cfg.get(k) if k != "render record sha256" else (cfg.get("render gate") or {}).get(k), v] for k, v in want.items()}
    stems = sorted(f.stem for f in (d / "chunks").glob("shard*_chunk*.json"))
    groups = [stems[i::GROUPS] for i in range(GROUPS)]
    arms = sorted(expected)
    rcall = repro_part.spawn(refs)
    mcalls = [meta_part.spawn(a, left_out.get(a, {})) for a in arms]
    cparts = list(chunk_part.map(groups))
    mparts = [c.get() for c in mcalls]
    rp = rcall.get()
    problems = Counter()
    index = {}
    for c in cparts:
        problems.update(c["problems"])
        for a, f, ch, rs, n, k in c["index"]:
            problems["duplicate index entry"] += (a, f, ch) in index
            index[(a, f, ch)] = (rs, n, k)
    rows_total = sum(c["rows"] for c in cparts)
    metas, kinds, kept, seen = Counter(), {}, 0, Counter()
    for m in mparts:
        problems.update(m["problems"]); kept += m["kept"]; metas[m["arm"]] = len(m["entries"])
        for k, v in m["kinds"].items():
            kinds[f"{m['arm']} | {k}"] = v
        for a, f, ch, rs, n, k in m["entries"]:
            seen[(a, k)] += 1
            problems["meta not in its chunk's index, or row_start/n_rows/key differ"] += index.pop((a, f, ch), None) != (rs, n, k)
    problems["index entries with no live meta"] += len(index)
    problems["kept readouts (meta) != rows (chunks)"] += kept != rows_total
    problems["transcripts by arm != expected"] += dict(metas) != expected
    problems["extracted (arm, key) multiset != the manifest's"] += seen != Counter((a, k) for a, k in ids)
    stats = {int(f.stem.split("_")[1]): json.loads(f.read_text()) for f in sorted((d / "stats").glob("shard_*.json"))}
    problems["shards without stats"] += len(stats) != cfg["shards"]
    util = [{"shard": i, "gpu": s.get("gpu"), "transcripts": s.get("transcripts"), "load s": s.get("load seconds (function start to weights on the GPU)"),
             "function s": s.get("function seconds"), "forward s": round(sum(t[1] for t in s["forward [tokens, seconds]"]), 1),
             "forward tokens": sum(t[0] for t in s["forward [tokens, seconds]"]),
             "tokens per forward second": round(sum(t[0] for t in s["forward [tokens, seconds]"]) / max(sum(t[1] for t in s["forward [tokens, seconds]"]), 1e-9))}
            for i, s in sorted(stats.items())]
    nf_i, nf_r = sum(c["nonfinite index"] for c in cparts), sum(c["nonfinite recount"] for c in cparts)
    res = {"run": RUN, "provenance (stored config vs given)": prov, "provenance": "pass" if all(a == b for a, b in prov.values()) else "FAIL",
           "meta files by arm": dict(metas), "expected transcripts by arm": expected, "kept readouts (meta)": kept, "rows (chunks)": rows_total,
           "chunks": len(stems), "non-finite values: every layer (chunk indexes)": nf_i, "non-finite values: first and last layer recounted": nf_r,
           "problems": dict(problems), "kept rows by arm and kind": dict(sorted(kinds.items())), "GPU types used": rp["gpus used"], "per shard": util,
           "reproduction (output_end vs reference rows), by GPU type": rp["rep"], "reproduction gate": "pass" if rp["pass"] else "FAIL"}
    Path("/work/checks").mkdir(parents=True, exist_ok=True)
    Path(f"/work/checks/extract_check_fan_{RUN}_{stamp}.json").write_text(json.dumps(res, indent=1) + "\n")
    work.commit()
    return res


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
    print(f"the result is also written to the work Volume as checks/extract_check_fan_{RUN}_{stamp}.json", flush=True)
    out = {"git": git("rev-parse", "HEAD"), "render record": Path(render).name, "manifest": Path(manifest).name} | \
        orchestrate.remote(want, man["by arm"], lo, refs, [[x["arm"], x["key"]] for x in man["runs"]], stamp)
    with open(dest := HERE / "results" / f"extract_check_{RUN}_{stamp}.json", "x") as fh:
        fh.write(json.dumps(out, indent=1) + "\n")
    print(dest)
    print(json.dumps({k: out[k] for k in ("provenance", "reproduction gate", "GPU types used", "problems", "meta files by arm",
                                           "non-finite values: every layer (chunk indexes)", "non-finite values: first and last layer recounted")}, indent=1))
    if (out["provenance"] != "pass" or out["reproduction gate"] != "pass" or any(out["problems"].values())
            or out["non-finite values: every layer (chunk indexes)"] or out["non-finite values: first and last layer recounted"]):
        raise SystemExit("check failed: see the record")
