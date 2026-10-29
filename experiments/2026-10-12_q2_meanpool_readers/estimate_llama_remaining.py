"""The Llama resume's projection input: the share of the extraction's work (by tokens) that the 1,800 finished transcripts
hold, from #36's runs.json token counts, rebuilding the extractor's longest-first round-robin deal over 24 shards (repaired
runs take their simulated original's count) and taking the first 75 of each shard's list as done (1,800 / 24). Local, $0.
  python3 experiments/2026-10-12_q2_meanpool_readers/estimate_llama_remaining.py <#36 runs.json> <fix939 per-run jsonl>
"""
import json
import sys
runs = json.load(open(sys.argv[1]))
fix = sorted({json.loads(l)["key"] for l in open(sys.argv[2]) if json.loads(l)["label"] == "FIX"})
tok = {r["key"]: r["tokens"] for r in runs if r["arm"] == "sim-attack"}
items = [(r["tokens"] or 0, r["arm"], r["key"]) for r in runs] + [(tok[k] or 0, "sim-attack-fix939", k) for k in fix]
items.sort(key=lambda it: (-it[0], it[1], it[2]))
total = sum(min(t, 131072) for t, _, _ in items)
out = {"transcripts": len(items), "tokens (Qwen counts, capped at Llama's context)": total}
for k in (70, 75, 80):
    done = sum(min(t, 131072) for s in range(24) for t, _, _ in items[s::24][:k])
    out[f"done share if the first {k} of each shard ({24 * k} transcripts) finished"] = round(done / total, 4)
print(json.dumps(out, indent=1))
