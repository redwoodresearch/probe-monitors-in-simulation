"""Attachment check (Aswin's priority via 0d, about 16:55 PT 10-11), $0, counts only. Inspect stores long strings in
.eval samples as `attachment://<hash>` with the text in the sample's `attachments` map; `probe_shift.transcripts`
reads samples raw and never resolves them. Counts unresolved `attachment://` strings, before and after resolving
each sample from its own map, in:
  A. the k = 1 inputs: each site's real history (prefix outputs and call arguments), step t's call, its real output, and
     the simulator's inputs (the cell's stored September metadata and stored system prompt);
  B. the Oct 7 matched-pairs inputs, rebuilt from the same samples and manifest (results/pairs_2026-10-06_175844.json):
     prefix, step, real output, reader messages, and each reference's metadata and stored prompt;
  C. the staged reader inputs on the Volume (streamed, never saved): level 0 (2026-10-06_la_segment-end_qwen3-32b_v1)
     and step B's level 1 (2026-10-10_la_fixed-sim-v2_probe-inputs-b-l1_v1), every message content and tool-call
     argument of the test-environment runs, and specifically each k = 1 site's slot and each planted site's slot.
Samples are cached outside the repo (--cache). Prints and saves counts only.

  PYTHONPATH=src:$HOME/simulated-control-evals/src uv run --no-project --python 3.14 --with boto3 --with pyyaml \\
      --with 'modal>=1.4.3' python experiments/2026-10-11_claimB_k_replies/count_attachments.py --sites results/k1_sites_<stamp>.json --cache <dir>
"""

import argparse
import hashlib
import json
import subprocess
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "2026-10-06_matched_step_pairs"))
import build_k1_records as BR  # noqa: E402  (cached sample loader, call_at)
import run_select_pairs as RSP  # noqa: E402  (actions_of, reader_turns)
from probe_shift.transcripts import reader_messages, system_prompt  # noqa: E402

REF = "attachment://"
TEST = {"clinical_trial", "media_processing", "shared_dotfiles", "tiktok", "user_tracking"}
LEVEL0, LEVEL1 = "2026-10-06_la_segment-end_qwen3-32b_v1", "2026-10-10_la_fixed-sim-v2_probe-inputs-b-l1_v1"


def refs(x) -> int:
    if isinstance(x, str):
        return x.count(REF)
    if isinstance(x, dict):
        return sum(refs(v) for v in x.values())
    if isinstance(x, list):
        return sum(refs(v) for v in x)
    return 0


def resolve(x, att: dict):
    """Each string that is exactly `attachment://<hash>` replaced by the sample's attachment text (Inspect's scheme)."""
    if isinstance(x, str):
        return att.get(x[len(REF):], x) if x.startswith(REF) else x
    if isinstance(x, dict):
        return {k: resolve(v, att) for k, v in x.items()}
    if isinstance(x, list):
        return [resolve(v, att) for v in x]
    return x


def resolved(sample: dict) -> dict:
    att = sample.get("attachments") or {}
    return {k: (v if k == "attachments" else resolve(v, att)) for k, v in sample.items()}


def inputs_counts(sample: dict, k: int, ref_sample: dict, prompt: str | None) -> Counter:
    c = Counter()
    acts = RSP.actions_of(sample)
    for x in acts[:k]:
        c["prefix_outputs"] += refs(x["output"])
        c["prefix_arguments"] += refs(x["arguments"])
    c["step_arguments"] += refs(acts[k]["arguments"])
    c["real_output"] += refs(acts[k]["output"])
    c["sim_metadata"] += refs(ref_sample["metadata"])
    c["stored_system_prompt"] += refs(ref_sample["store"]["SimulatorState:sim_messages"][0]["content"])
    if prompt is not None:
        c["reader_messages"] += refs(reader_messages(sample, prompt))
    return c


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sites", type=Path, required=True)
    p.add_argument("--cache", type=Path, required=True)
    a = p.parse_args()
    out = {"label": "unresolved attachment:// counts (before and after resolving from each sample's own map)"}
    shared = json.loads((ROOT / "splits" / "exclusions_v3.json").read_text())
    e1_rows = {r["key"]: r for r in shared["samples"] if r["arm"] == "real-attack-e1"}
    hon_rows = {r["key"]: r for r in shared["samples"] if r["arm"] == "real-honest"}
    sim_rows = {r["cell"]: r for r in shared["samples"] if r["arm"] == "sim-attack" and not r["reason"]}

    def load(alias, files):
        with ThreadPoolExecutor(16) as pool:
            return {(f, x["id"], x["epoch"]): x for f, xs in zip(files, pool.map(lambda f: BR.cached(a.cache, alias, f), files)) for x in xs}

    # A. k = 1 inputs
    sites = json.loads((HERE / a.sites).read_text())["sites"]
    e1 = load(BR.COHORTS["attack"], sorted({e1_rows[s["key_e1"]]["file"] for s in sites}))
    sim = load(BR.COHORTS["sim"], sorted({sim_rows[s["cell"]]["file"] for s in sites}))
    before, after, att_present, missing_call = Counter(), Counter(), 0, 0
    for s in sites:
        r, sr = e1_rows[s["key_e1"]], sim_rows[s["cell"]]
        smp, ref = e1[(r["file"], r["sample_id"], r["epoch"])], sim[(sr["file"], sr["sample_id"], sr["epoch"])]
        att_present += bool(smp.get("attachments"))
        for tag, x, y in (("before", smp, ref), ("after", resolved(smp), resolved(ref))):
            cid = BR.call_at(x, s["step"], s["action"])
            ids = [z["id"] for z in RSP.actions_of(x)]
            if cid not in ids:
                missing_call += tag == "before"
                continue
            (before if tag == "before" else after).update(inputs_counts(x, ids.index(cid), y, None))
    out["A_k1_inputs"] = {"sites": len(sites), "samples_with_attachments_map": att_present, "call_not_found": missing_call,
                          "before": dict(before), "after": dict(after), "sum_before": sum(before.values()), "sum_after": sum(after.values())}
    print("A", out["A_k1_inputs"], flush=True)
    # B. Oct 7 matched pairs, rebuilt from the same samples
    man = json.loads((HERE.parent / "2026-10-06_matched_step_pairs" / "results" / "pairs_2026-10-06_175844.json").read_text())
    refs_json = json.loads((ROOT / "experiments" / "2026-10-06_astra_sim_honest" / "results" / "references_20261006-142754.json").read_text())
    att_files = sorted({e1_rows[p["key"]]["file"] for p in man["pairs"] if p["arm"] == "attack"})
    hon_files = sorted({hon_rows[p["key"]]["file"] for p in man["pairs"] if p["arm"] == "honest"})
    run_s = load("la_public10_real_attack_n1000", att_files) | load("la_public10_real_honest_n1000", hon_files)
    att_ref_files = sorted({p["reference"]["file"] for p in man["pairs"] if p["arm"] == "attack"})
    ref_s = load(BR.COHORTS["sim"], att_ref_files)
    hon_uris = {}
    for mt, rr in refs_json["references"].items():
        hon_uris.setdefault(rr["sim_uri"].rsplit("/", 1)[0].removesuffix("/logs") + "/", set()).add(rr["sim_uri"].rsplit("/", 1)[-1])
    for uri, files in hon_uris.items():
        ref_s |= load(uri, sorted(files))
    prompts = {}
    for x in load("la_public10_real_honest_n1000", hon_files).values():
        prompts.setdefault((x.get("metadata") or {}).get("main_task", {}).get("id") if isinstance((x.get("metadata") or {}).get("main_task"), dict) else (x.get("metadata") or {}).get("main_task"), system_prompt(x))
    bb, aa, nf = Counter(), Counter(), 0
    for pr in man["pairs"]:
        row = (e1_rows if pr["arm"] == "attack" else hon_rows)[pr["key"]]
        smp = run_s[(row["file"], row["sample_id"], row["epoch"])]
        rf = pr["reference"]
        ref = ref_s.get((rf["file"], rf["sample_id"], rf["epoch"]))
        if ref is None:
            nf += 1
            continue
        prompt = prompts.get(pr["main_task"])
        bb.update(inputs_counts(smp, pr["k"], ref, prompt))
        aa.update(inputs_counts(resolved(smp), pr["k"], resolved(ref), prompt))
    out["B_oct7_pairs"] = {"pairs": len(man["pairs"]), "reference_not_found": nf, "before": dict(bb), "after": dict(aa),
                           "sum_before": sum(bb.values()), "sum_after": sum(aa.values())}
    print("B", out["B_oct7_pairs"], flush=True)
    # C. staged reader inputs on the Volume (streamed)
    import modal

    vol = modal.Volume.from_name("probe-shift_aswin-surya_sub-la-activations", version=2)
    site_slots = {(s["cell"], "real-attack-e1"): (s["step"], s["action"]) for s in sites}
    tplan = json.loads(b"".join(vol.read_file("claimB/h1_test_swap_v1/cells.json")))
    planted = {(p["cell"], arm): p["i"] for p in tplan["planted"] for arm in ("sim-attack-fixed", "sim-attack-fixed-plant2")}

    def step_action(msgs):
        out_, step, action = {}, -1, 0
        for i, m in enumerate(msgs):
            if m["role"] == "assistant":
                step, action = step + 1, 0
            elif m["role"] == "tool":
                out_[(step, action)] = i
                action += 1
        return out_
    for run in (LEVEL0, LEVEL1):
        c, slots = Counter(), Counter()
        for e in vol.listdir(f"{run}/inputs"):
            if not e.path.rsplit("/", 1)[-1].startswith("shard_"):
                continue
            for line in b"".join(vol.read_file(e.path)).decode().splitlines():
                r = json.loads(line)
                if r.get("environment") not in TEST:
                    continue
                c[f"{r['arm']}|contents"] += sum(refs(m.get("content")) for m in r["messages"])
                c[f"{r['arm']}|tool_call_arguments"] += sum(refs(m.get("tool_calls")) for m in r["messages"])
                c[f"{r['arm']}|runs"] += 1
                key = (r["cell"], r["arm"])
                if key in site_slots:
                    m = step_action(r["messages"]).get(site_slots[key])
                    slots["k1_site_slots"] += 1
                    slots["k1_site_slot_refs"] += refs(r["messages"][m]["content"]) if m is not None else 0
                    slots["k1_site_prefix_refs"] += sum(refs(x.get("content")) + refs(x.get("tool_calls")) for x in r["messages"][: (m or 0) + 1])
                if run == LEVEL1 and key in planted:
                    slots["planted_slots"] += 1
                    slots["planted_slot_refs"] += refs(r["messages"][planted[key]]["content"])
        out[f"C_staged_{run}"] = {"test_env_counts": dict(c), "slots": dict(slots)}
        print("C", run, dict(slots), {k: v for k, v in c.items() if not k.endswith("|runs") and v}, flush=True)
    out |= {"git": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip(),
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    dst = HERE / "results" / f"attachment_counts_{stamp}.json"
    with open(dst, "x") as fh:
        fh.write(json.dumps(out, indent=1))
    print("->", dst)


if __name__ == "__main__":
    main()
