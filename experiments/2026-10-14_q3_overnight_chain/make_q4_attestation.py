"""Q4 H200 writer attestation (09 + 0d's joint rule via 4f, 10-14 ~21:30): built from the committed records by script (keys,
counts, hashes only). If v2 equals v1 bit for bit, v1 stands; otherwise v2 replaces it.

  python3 experiments/2026-10-14_q3_overnight_chain/make_q4_attestation.py
"""
import glob
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
R = HERE / "results"


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def rel(p: Path) -> str:
    return str(p.relative_to(HERE.parents[1]))


def one(pattern: str) -> Path:
    (p,) = [Path(x) for x in glob.glob(str(R / pattern))]
    return p


out = {"rule": "09 + 0d via 4f (10-14 ~21:30): full H200 re-extraction (v2) of the same staged inputs; v1 stands if v2 is bit-identical on every scored epoch",
       "a1_code": {"commit": "a637274925e12d238e70582ec76984cfdea3f204", "file": "experiments/2026-10-14_q3_overnight_chain/extract_chain.py",
                   "gpu_pin": "line 104: @app.function(image=image, gpu=\"H200\", ...) (a single type; no fallback list, no B200)",
                   "in_container_device_assert": "absent (line 249 only logs torch.cuda.get_device_name(0) into each shard's returned record)"},
       "roots": {}}
for t in ("main", "run2"):
    v1, v2 = f"su311_q4_{t}_extract_v1", f"su311_q4_{t}_extract_v2"
    a1 = one(f"extract_chain_{v1}_a1_*.json")
    a7 = one(f"extract_chain_{v1}_a7_*.json")
    r2 = one(f"extract_chain_{v2}_a1_*.json")
    cmp_ = one(f"compare_v1v2_{v2}_*.json")
    rep = one(f"compare_repro_{v2}_repro-c_*.json")
    A1, A7, V2, C, RP = (json.loads(p.read_text()) for p in (a1, a7, r2, cmp_, rep))
    errored = {i for i, s in enumerate(A1["shards"]) if "error" in s}  # a shard's position in the record is its index (starmap order)
    unlogged = {}  # a1 epochs written by a shard that returned no record: no gpu_name by key
    for arm, eps in C["per_epoch"].items():
        for k, e in eps.items():
            ch = e.get("chunk_v1", "") if isinstance(e, dict) else ""
            if ch.startswith("shard") and "_a1_" in ch and int(ch[5:8]) in errored:
                unlogged.setdefault(arm, []).append({"key": k, "chunk_v1": ch, "bit_exact_vs_v2": e["bit_exact"], "layers": e["layers"]})
    v2_shards = [{"shard": s.get("shard"), "gpu_name": s.get("gpu_name"), "gpu_count": s.get("gpu_count"), "epochs": s.get("epochs"), "error": "error" in s} for s in V2["shards"]]
    out["roots"][t] = {
        "v1": {"root": v1, "a1_record": {"path": rel(a1), "sha256": sha(a1), "git": A1["git"],
                                         "shards_with_record": sum("error" not in s for s in A1["shards"]), "shards_errored": sum("error" in s for s in A1["shards"]),
                                         "logged_gpu": sorted({s.get("gpu_name") for s in A1["shards"] if "error" not in s})},
               "a7_record": {"path": rel(a7), "sha256": sha(a7), "git": A7["git"], "done_before_sum": sum(s.get("done_before", 0) for s in A7["shards"] if "error" not in s),
                             "logged_gpu": sorted({s.get("gpu_name") for s in A7["shards"] if "error" not in s}), "shards_errored": sum("error" in s for s in A7["shards"])}},
        "v2": {"root": v2, "record": {"path": rel(r2), "sha256": sha(r2), "git": V2["git"]}, "shards": v2_shards,
               "every_shard_h200": all((not s["error"]) and s["gpu_name"] == "NVIDIA H200" and s["gpu_count"] == 1 for s in v2_shards),
               "epochs": sum(s["epochs"] or 0 for s in v2_shards),
               "repro_c": {"path": rel(rep), "sha256": sha(rep), "epochs": RP["epochs"], "bit_exact": RP["bit_exact"], "failing_shards": RP["failing_shards"]}},
        "v1_vs_v2": {"path": rel(cmp_), "sha256": sha(cmp_), "layers": C["layers"], "summary": C["summary"], "all_identical": C["all_identical"],
                     "inputs_sha256": C["inputs_sha256"], "per_epoch_in": "the compare file's per_epoch (keyed by epoch: layers, bit_exact, max_abs_diff, chunks)"},
        "a1_without_gpu_name": {"errored_a1_shards": sorted(errored), "epochs_by_arm": {a_: len(v) for a_, v in unlogged.items()},
                                "total": sum(len(v) for v in unlogged.values()), "total_excluding_repro_c": sum(len(v) for a_, v in unlogged.items() if a_ != "repro-c"),
                                "all_bit_exact_vs_v2": all(x["bit_exact_vs_v2"] and x["layers"] == 64 for v in unlogged.values() for x in v), "epochs": unlogged}}
bill = [R / "q4_attest" / "billing_q4_a1_2026-10-15.txt", R / "q4_attest" / "billing_q4a1_2026-10-15.json"]
out["billing_a1"] = [{"path": rel(p), "sha256": sha(p)} for p in bill]
out["pass"] = all(r["v2"]["every_shard_h200"] and r["v1_vs_v2"]["all_identical"] and not r["v2"]["repro_c"]["failing_shards"] for r in out["roots"].values())
out["verdict"] = ("v1 stands: full H200 re-extraction (v2), bit-identical on every staged epoch at all 64 layers; attestation supersedes the seeded spot check"
                  if out["pass"] else "v2 replaces v1 where they differ, or the failing epochs stay unscored")
dest = R / "q4_attest" / "q4_h200_attestation_2026-10-14.json"
dest.write_text(json.dumps(out, indent=1) + "\n")
print(json.dumps({"pass": out["pass"], "verdict": out["verdict"], **{t: {"v2_every_shard_h200": r["v2"]["every_shard_h200"], "v2_shards": len(r["v2"]["shards"]),
      "a1_without_gpu_name": {k: r["a1_without_gpu_name"][k] for k in ("errored_a1_shards", "epochs_by_arm", "total", "total_excluding_repro_c", "all_bit_exact_vs_v2")},
      "compared_identical": {a_: (x["compared"], x["identical"]) for a_, x in r["v1_vs_v2"]["summary"].items()}, "a7_done_before_sum": r["v1"]["a7_record"]["done_before_sum"],
      "v2_gpu": sorted({x["gpu_name"] for x in r["v2"]["shards"]})} for t, r in out["roots"].items()}}, indent=1))
