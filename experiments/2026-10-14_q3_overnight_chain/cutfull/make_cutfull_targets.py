"""d9's CUT fork (Append 41): write a PRIVATE chain-targets file for an uncut ("cutfull") batch, never the chain's shared
config/chain_targets.json. Per original chain target T (cutfull_naming_v1.json): target "T_cutfull" (out T_cutfull_v1) or, for the
step-0 gate, "T_cutfull_gate" (out T_cutfull_gate_v1, only_keys = the gate's run keys of T, one staging part per key). Every field
is T's own except: out, "cut": "full", and the .eval cache read from the chain's Volume mounted read only (/ev/... -> /chainro/...).
Names, keys and hashes only.

  python3 experiments/2026-10-14_q3_overnight_chain/cutfull/make_cutfull_targets.py --full T1,T2,...
  python3 .../make_cutfull_targets.py --step0 <commit>:<path of 2e's step0_spec_v2 json>   (also writes the step-0 compare spec)
  python3 .../make_cutfull_targets.py --from-spec <commit>:<path> --scope "(a'')"   (every chain-target staged set of that scope)
  python3 .../make_cutfull_targets.py --extract <targets file> [--shards-per-target T=n,...]   (extraction configs from the stage records)
"""
import argparse
import hashlib
import json
import re
import subprocess
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent
CHAIN = HERE.parent
REPO = CHAIN.parents[1]
OUT_DIR = CHAIN / "config" / "cutfull"
AUTH_FILES: set = set()
TOKENS_PER_SHARD = 1_000_000  # ~9 min of forward pass per H200 shard at ~1,830 tokens/s (quote r3's rate): max parallel H200s (Aswin 21:4x)


def convert(t: dict, out: str) -> dict:
    if t.get("cut") is not None:
        raise ValueError("the original target already has a cut")
    if not str(t["cache"]).startswith("/ev/"):
        raise ValueError(f"cache {t['cache']} is not on the chain's Volume")
    return t | {"out": out, "cut": "full", "cache": "/chainro/" + str(t["cache"])[len("/ev/"):]}


def original_target(chain: dict, staged_set: str) -> str:
    t = re.sub(r"_v1(_try[0-9]+)?$", "", staged_set)  # a chain retry stages the target's out plus "_try<n>"
    if t == staged_set or chain.get(t, {}).get("out") != re.sub(r"_try[0-9]+$", "", staged_set):
        raise ValueError(f"{staged_set}: not the output of a chain target")
    return t


def write(kind: str, obj: dict, stamp: str, ext: str = "json") -> Path:
    OUT_DIR.mkdir(exist_ok=True)
    dest = OUT_DIR / f"{kind}_{stamp}.{ext}"
    dest.open("x").write(obj if isinstance(obj, str) else json.dumps(obj, indent=1))
    print(dest.relative_to(REPO))
    return dest


def step0(chain: dict, ref: str, stamp: str) -> None:
    raw = subprocess.run(["git", "show", ref], capture_output=True, check=True, cwd=REPO).stdout
    spec = json.loads(raw)
    s0 = spec["step0"]
    if sorted(r["key"] for r in s0["runs"]) != sorted(s0["selection"]["keys"]) or len(s0["runs"]) != s0["selection"]["n"]:
        raise ValueError("step-0 runs and key list differ")
    keys, rows = defaultdict(list), []
    for r in s0["runs"]:
        t = original_target(chain, r["cut"]["staged_set"])
        if r["uncut_planned"]["gate_folder"] != f"{t}_cutfull_gate_v1__{r['arm']}" or r["cut"]["volume"] != "probe-shift_aswin-surya_sub-q3su311_acts_kill-2026-11-14":
            raise ValueError(f"{r['key']}: gate folder or cut Volume is not the naming rule's")
        keys[t].append(r["key"])
        rows.append({"key": r["key"], "arm": r["arm"], "original_target": t, "tokens_before": r["tokens_before"],
                     "cut_staged_set": r["cut"]["staged_set"], "cut_extraction_root": r["cut"]["extraction_root"],
                     "gate_staged_set": f"{t}_cutfull_gate_v1", "gate_extraction_root": f"{t}_cutfull_gate_extract_v1"})
    targets = {f"{t}_cutfull_gate": convert(chain[t], f"{t}_cutfull_gate_v1") | {"only_keys": sorted(set(ks)), "parts": len(set(ks))}
               for t, ks in sorted(keys.items())}
    tfile = write("cutfull_targets_gate", targets, stamp)
    name, rev = (x.strip() for x in s0["pins"]["reader"].split("@"))
    write("step0_compare_spec", {"source": {"ref": ref, "sha256": hashlib.sha256(raw).hexdigest()}, "targets_file": str(tfile.relative_to(REPO)),
                                 "reader": {"name": name, "revision": rev},
                                 "weights_pin": {"files": s0["pins"]["snapshot_files"], "sha256": s0["pins"]["snapshot_sha256"]}, "runs": rows}, stamp)


def from_spec(chain: dict, ref: str, scope: str, stamp: str) -> None:
    """Every staged set of one scope of 2e's arm list (step0_spec_v2 arm_list): its cutfull target, with the arm list's epochs per arm
    (checked against the staging later) and staging parts scaled to its size. Sets that are not chain targets are listed, not written."""
    raw = subprocess.run(["git", "show", ref], capture_output=True, check=True, cwd=REPO).stdout
    arms = [a for a in json.loads(raw)["arm_list"]["arms"] if a["scope"] == scope]
    by_set, other = defaultdict(dict), set()
    if not arms:
        raise ValueError(f"no arms in scope {scope!r}")
    for a in arms:
        if a["arm"] in by_set[a["staged_set"]]:
            raise ValueError(f"{a['staged_set']} {a['arm']}: listed twice")
        by_set[a["staged_set"]][a["arm"]] = a["epochs_counted"]
    out = {}
    for ss, ep in sorted(by_set.items()):
        try:
            t = original_target(chain, ss)
        except ValueError:
            other.add(ss)
            continue
        if f"{t}_cutfull" in out:
            raise ValueError(f"{t}: two staged sets in one scope")
        out[f"{t}_cutfull"] = convert(chain[t], f"{t}_cutfull_v1") | {"parts": min(48, max(2, sum(ep.values()) // 3)),
                                                                     "arm_list": {"ref": ref, "sha256": hashlib.sha256(raw).hexdigest(),
                                                                                  "scope": scope, "original_staged_set": ss, "epochs": ep}}
    write(f"cutfull_targets_{re.sub('[^a-z]', '', scope)}", out, stamp)
    print(len(out), "targets;", sum(sum(t["arm_list"]["epochs"].values()) for t in out.values()), "epochs; not chain targets:", sorted(other))


def from_routing(chain: dict, routing_dir: Path, staged_sets: list, label: str, stamp: str, named: dict = None) -> None:
    """A cutfull target per ROUTED original staged set (the chain's routing directory; one record per set, superseded files aside),
    with that record's epochs per arm as the arm list (one_to_one with the original). `named` {staged set: chain target}: a retry
    TARGET of its own (09 via d9 10-16 ~03:38: ilnm_s4rgap_sim2_try2, the nm145 donor-fix retry, whose staged set naming v1 maps onto
    ilnm_s4rgap_sim2): the routing record's own target, asserted, and the chain target's out == the staged set minus its _try<n>."""
    out = {}
    if set(named or {}) - set(staged_sets):
        raise ValueError(f"--named sets not in --from-routing: {sorted(set(named) - set(staged_sets))}")
    for ss in staged_sets:
        recs = []
        for f in sorted(routing_dir.glob("*__*.json")):
            d = json.loads(f.read_bytes())
            if isinstance(d, dict) and d.get("staged_set") == ss and "superseded" not in f.name:
                recs.append((f, d))
        if len(recs) > 1 and AUTH_FILES:  # several records of one staged set (a re-extraction revision): the routing authority's one
            recs = [(f, d) for f, d in recs if f.name in AUTH_FILES]
        if len(recs) != 1:
            raise ValueError(f"{ss}: {len(recs)} routing records")
        f, d = recs[0]
        if ss in (named or {}):
            t = named[ss]
            if d.get("target") != t or chain.get(t, {}).get("out") != re.sub(r"_try[0-9]+$", "", ss) or not re.search(r"_try[0-9]+$", ss):
                raise ValueError(f"{ss}: the routing record's target is not {t}, or {t} does not stage it")
        else:
            t = original_target(chain, ss)
        if f"{t}_cutfull" in out:
            raise ValueError(f"{t}: two staged sets")
        ep = {a["meta_arm"]: a["epochs"] for a in d["arms"]}
        out[f"{t}_cutfull"] = convert(chain[t], f"{t}_cutfull_v1") | {"parts": min(48, max(2, sum(ep.values()) // 3)),
                                                                     "arm_list": {"ref": f"routing record {f.name} (sha256 {hashlib.sha256(f.read_bytes()).hexdigest()})",
                                                                                  "scope": label, "original_staged_set": ss, "epochs": ep}}
    write(f"cutfull_targets_{label}", out, stamp)
    print(len(out), "targets;", sum(sum(t["arm_list"]["epochs"].values()) for t in out.values()), "epochs")


def extract_configs(tfile: Path, shards: dict, stamp: str) -> None:
    targets = json.loads(tfile.read_text())
    for name, t in sorted(targets.items()):
        recs = sorted((CHAIN / "results").glob(f"chain_stage_{name}_2*.json"))
        if len(recs) != 1:
            raise ValueError(f"{name}: {len(recs)} stage records")
        rec = json.loads(recs[0].read_text())
        a = rec[t.get("set", "c")]["assembled"]
        if a.get("STOP") or rec[t.get("set", "c")]["out"] != t["out"] or not a.get("staged_inputs_sha256"):
            raise ValueError(f"{name}: staging stopped or is not {t['out']}")
        al = t.get("arm_list", {}).get("epochs")  # the arm list counts the cohort's runs; a staging holds every run of the original (one_to_one)
        if al is not None and (set(a["epochs"]) != set(al) or any(a["epochs"][k] < al[k] for k in al)):
            raise ValueError(f"{name}: staged epochs {a['epochs']} do not cover the arm list's {al}")
        root = re.sub(r"_v1(_try[0-9]+)?$", r"_extract_v1\1", t["out"]) if re.search(r"_v1(_try[0-9]+)?$", t["out"]) else None  # the TRAILING _v1[_try<n>]
        if root is None or not re.fullmatch(r"[a-z0-9_]+_cutfull(_gate)?_extract_v1(_try[0-9]+)?", root):
            raise ValueError(f"{name}: extraction root {root}")
        epochs = sum(a["epochs"].values())
        tokens = sum(v["totals"]["tokens_before"] for v in a["coverage"].values())  # the whole renders (no cut): the forward-pass work
        n = shards[name] if name in shards else (epochs if "only_keys" in t else min(epochs, -(-tokens // TOKENS_PER_SHARD)))
        if not isinstance(n, int) or isinstance(n, bool) or not 1 <= n <= epochs:
            raise ValueError(f"{name}: shard count {n!r} for {epochs} epochs")
        lines = [f"# d9's CUT fork (Append 41): {name} staged by stage_chain.py (record {recs[0].name}); every layer, last-token rows and",
                 "# output_end span means, as the chain's extractions (chunk 8, pad 4096), H200; no cut, no repro-c arm.",
                 f"run: {root}", f"inputs: {t['out']}", "inputs_sha256:",
                 *[f"  {k}: {v}" for k, v in sorted(a["staged_inputs_sha256"].items())],
                 "model: Qwen/Qwen3.5-27B", f"shards: {n}", "chunk_size: 8", "pad_to: 4096", "cut: full", ""]
        write(f"extract_{root}", "\n".join(lines), stamp, "yaml")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", default="")
    ap.add_argument("--step0", default="")
    ap.add_argument("--extract", default="")
    ap.add_argument("--from-spec", default="", help="<commit>:<path of step0_spec_v2>; with --scope")
    ap.add_argument("--scope", default="")
    ap.add_argument("--from-routing", default="", help="comma-separated ROUTED original staged sets; with --routing-dir and --label")
    ap.add_argument("--routing-dir", default="")
    ap.add_argument("--label", default="")
    ap.add_argument("--named", default="", help="with --from-routing: staged_set=chain target,... (a retry target of its own; 09's ruling)")
    ap.add_argument("--authority", default="", help="routing authority json: picks the authoritative record when a staged set has several")
    ap.add_argument("--chain-targets", default="", help="chain_targets.json to read (default: config/chain_targets.json)")
    ap.add_argument("--shards-per-target", default="")
    a = ap.parse_args()
    chain = json.loads((CHAIN / "config" / "chain_targets.json").read_text())
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    if a.step0:
        return step0(chain, a.step0, stamp)
    if a.chain_targets:
        chain = json.loads(Path(a.chain_targets).read_text())
    global AUTH_FILES
    AUTH_FILES = {v["routing_file"] for v in json.loads(Path(a.authority).read_text())["batches"].values()} if a.authority else set()
    if a.from_routing:
        return from_routing(chain, Path(a.routing_dir), [x for x in a.from_routing.split(",") if x], a.label, stamp,
                            dict(x.split("=") for x in a.named.split(",") if x))
    if a.from_spec:
        return from_spec(chain, a.from_spec, a.scope, stamp)
    if a.extract:
        return extract_configs(Path(a.extract), {k: int(v) for k, v in (x.split("=") for x in a.shards_per_target.split(",") if x)}, stamp)
    out = {f"{t}_cutfull": convert(chain[t], f"{t}_cutfull_v1") for t in [x for x in a.full.split(",") if x]}
    if not out:
        raise SystemExit("nothing to write")
    write("cutfull_targets_full", out, stamp)


if __name__ == "__main__":
    main()
