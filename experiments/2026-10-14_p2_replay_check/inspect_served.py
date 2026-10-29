"""Append 28 Codex items 2 and 4 (RULE.md): a scripted inspection of the staged served-context layouts BEFORE any score. Token ids,
positions, counts and hashes only: nothing is decoded, printed or written as text (template marker ids are looked up by name).

Over EVERY staged epoch, per world and layout:
- A0 vs A (attack worlds): the ids after the system turn are identical, and every unit's readout and request end sit at the same
  offsets from the system turn's end (the only edit between them is inside the system turn);
- B vs A: the same number of think blocks, every B interior equal to the template's empty form, the same units;
- units per kind; output_end spans that end at their readout.
For runs drawn with random.Random(20261014) (8 per world), paired with the reader layout's staged epoch (corrected B / batch C):
turn and role-marker counts, system-turn lengths, `<tools>` marker presence, think-block counts and non-empty interiors, and per unit
whether the readout token id, the 8 ids before it, and (output_end) the content span's ids are equal to the reader's.

  PYTHONPATH=src uv run --no-project --python 3.12 --with "modal==1.6.1" modal run \\
      experiments/2026-10-14_p2_replay_check/inspect_served.py --manifest-b <correctedb_manifest_v1.json> --stage-record results/stage_served_<stamp>.json
"""

import hashlib
import json
import random
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).resolve().parent
APP = "probe-shift_aswin-surya_sub-p2served_inspect_kill-2026-11-14"
READER_VOL = "probe-shift_aswin-surya_sub-q4own_evals_kill-2026-11-13"
OUT_VOL = "probe-shift_aswin-surya_sub-p2served_acts_kill-2026-11-14"
HF_CACHE = "probe-shift_aswin-surya_hf-cache"
MODEL, REVISION = "Qwen/Qwen3.5-27B", "fc05daec18b0a78c049392ed2e771dde82bdf654"
PIN_B = "bf847d425b80d95ebb75a9ed1d524fc68c9efcd014710ddc18e445ed9d373ebc"
C_STAGE, C_PIN = "q4f_c_v1/inputs/honest_b.jsonl", "513f621a9e71e91f8188cccc76cb222e9c9f4acb5a91fc90c45a283964729b31"
ATTACK_WORLDS, HONEST_WORLDS = ("run1", "run2", "sim"), ("honest_a", "honest_b", "honest_c")
SEED, N_PAIRED, K_BEFORE = 20261014, 8, 8

app = modal.App(APP)
image = (modal.Image.debian_slim(python_version="3.12").uv_pip_install("transformers==5.19.0", "jinja2", "numpy==2.3.3")
         .env({"HF_HOME": "/hf", "HF_HUB_OFFLINE": "1"}))


@app.function(image=image, cpu=4, memory=32768, timeout=1800, retries=0,
              volumes={"/acts": modal.Volume.from_name(OUT_VOL).read_only(), "/reader": modal.Volume.from_name(READER_VOL).read_only(),
                       "/hf": modal.Volume.from_name(HF_CACHE).read_only()})
def inspect_world(world: str, files: dict, reader_src: list) -> dict:
    from collections import Counter

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    tid = lambda s: tok.convert_tokens_to_ids(s)
    im_start, im_end, th_open, th_close = tid("<|im_start|>"), tid("<|im_end|>"), tid("<think>"), tid("</think>")
    tools_marker = tok.encode("<tools>", add_special_tokens=False)  # a template marker, matched as an id sequence
    probe = tok.encode(tok.apply_chat_template([{"role": "user", "content": "x"}, {"role": "assistant", "content": "y"}, {"role": "user", "content": "z"}],
                                               tokenize=False, add_generation_prompt=True, enable_thinking=False), add_special_tokens=False)
    gen = probe[max(i for i, x in enumerate(probe) if x == im_start):]
    EMPTY = gen[gen.index(th_open) + 1: gen.index(th_close)]
    h = lambda ids: hashlib.sha256(json.dumps(ids).encode()).hexdigest()

    def sys_end(ids):
        return ids.index(im_end) + 1

    role_asst = tok.encode("assistant\n", add_special_tokens=False)

    def thinks(ids):
        """the interiors of the reasoning blocks that open assistant turns (stage_served.py's rule), and the count of think
        markers anywhere else"""
        out, i, starts, cuts = [], 0, [], []
        while i < len(ids):
            if ids[i] == im_start:
                j = next((q for q in range(i + 1, len(ids)) if ids[q] == im_end), len(ids))
                h = i + 1 + len(role_asst)
                if ids[i + 1: h] == role_asst and h < j and ids[h] == th_open:
                    k = next((q for q in range(h + 1, j) if ids[q] == th_close), None)
                    if k is not None:
                        out.append(ids[h + 1: k])
                        starts.append(h)
                        cuts.append((h + 1, k))
                i = j
            i += 1
        kept, prev = [], 0
        for lo, hi in cuts:
            kept += ids[prev:lo]
            prev = hi
        thinks.kept = kept + ids[prev:]  # the ids with every assistant reasoning interior removed (read right after the call)
        return out, sum(x == th_open for x in ids) - len(starts)

    def has_seq(ids, seq):
        return any(ids[i: i + len(seq)] == seq for i in range(len(ids) - len(seq) + 1))

    ukey = lambda c: (c["irid"], c["kind"], c.get("src_irid"), c.get("call_index"))
    lay = {}
    for L, f in files.items():
        raw = (Path("/acts") / f["path"]).read_bytes()
        if hashlib.sha256(raw).hexdigest() != f["sha256"]:
            raise ValueError(f"{world} {L}: staged file differs from the stage record")
        lay[L] = [json.loads(x) for x in raw.decode().splitlines()]
    c = Counter()
    by = {L: {e["key"]: e for e in eps} for L, eps in lay.items()}
    for L, eps in lay.items():
        for e in eps:
            for cc in e["calls"]:
                c[f"{L}: units {cc['kind']}"] += 1
                if cc["kind"] == "output_end":
                    c[f"{L}: output_end spans ending at the readout"] += int(cc["span"][1] - 1 == cc["readout"])
    if "A0" in lay:  # A0 vs A: identical after the system turn, units at the same offsets from its end
        for k, a in by["A"].items():
            z = by["A0"].get(k)
            if z is None:
                c["A0 vs A: epochs missing in A0"] += 1
                continue
            ea, ez = sys_end(a["ids"]), sys_end(z["ids"])
            c["A0 vs A: epochs compared"] += 1
            c["A0 vs A: ids after the system turn identical"] += int(a["ids"][ea:] == z["ids"][ez:])
            ua, uz = {ukey(x): x for x in a["calls"]}, {ukey(x): x for x in z["calls"]}
            c["A0 vs A: same units"] += int(set(ua) == set(uz))
            c["A0 vs A: units at the same offsets"] += sum(ua[q]["readout"] - ea == uz[q]["readout"] - ez and ua[q]["n"] - ea == uz[q]["n"] - ez for q in ua if q in uz)
    b_alias_fail = []
    for k, a in by["A"].items():  # B vs A
        b = by["B"].get(k)
        if b is None:
            c["B vs A: epochs missing in B"] += 1
            continue
        ta, oa = thinks(a["ids"])
        ka = thinks.kept
        tb, ob = thinks(b["ids"])
        kb = thinks.kept
        c["A: think markers outside assistant reasoning blocks"] += oa
        c["B vs A: same think markers outside assistant reasoning blocks"] += int(oa == ob)
        c["B vs A: ids identical outside assistant reasoning interiors"] += int(ka == kb)
        c["B vs A: epochs compared"] += 1
        c["B vs A: same think-block count"] += int(len(ta) == len(tb))
        c["B: think interiors equal to the empty form"] += sum(t == EMPTY for t in tb)
        c["B: think interiors"] += len(tb)
        c["A: non-empty think interiors"] += sum(t != EMPTY for t in ta)
        c["B vs A: same units"] += int({ukey(x) for x in a["calls"]} == {ukey(x) for x in b["calls"]})
        same_sys = a["ids"][: sys_end(a["ids"])] == b["ids"][: sys_end(b["ids"])]
        c["B vs A: system turns identical"] += int(same_sys)
        if not same_sys:  # Codex r2 item 7: B's prompt and submit are A's, verified per epoch (failures listed by key)
            b_alias_fail.append(k)
    # paired renders: reader vs each served layout, for the drawn runs
    rd = {}
    for src in reader_src:
        raw = (Path("/reader") / src["file"]).read_bytes()
        if hashlib.sha256(raw).hexdigest() != src["sha256"]:
            raise ValueError(f"{src['file']}: reader staged file differs from its pin")
        for x in raw.decode().splitlines():
            e = json.loads(x)
            if e["key"] in src["keys"]:
                rd[e["key"]] = e
    paired = Counter()
    for rk in sorted(rd):
        r = rd[rk]
        rcalls = {ukey(x): x for x in r["calls"] if x.get("readout") is not None}
        paired["runs"] += 1
        paired["reader: turns"] += r["ids"].count(im_start)
        paired["reader: system-turn tokens"] += sys_end(r["ids"])
        paired["reader: tools marker in the system turn"] += int(has_seq(r["ids"][: sys_end(r["ids"])], tools_marker))
        paired["reader: think blocks"] += len(thinks(r["ids"])[0])
        paired["reader: non-empty think interiors"] += sum(t != EMPTY for t in thinks(r["ids"])[0])
        for L, eps in lay.items():
            mine = [e for e in eps if any(cc["reader_key"] == rk for cc in e["calls"])]
            paired[f"{L}: served epochs"] += len(mine)
            for e in mine:
                ids = e["ids"]
                paired[f"{L}: turns"] += ids.count(im_start)
                paired[f"{L}: system-turn tokens"] += sys_end(ids)
                paired[f"{L}: tools marker in the system turn"] += int(has_seq(ids[: sys_end(ids)], tools_marker))
                paired[f"{L}: think blocks"] += len(thinks(ids)[0])
                paired[f"{L}: non-empty think interiors"] += sum(t != EMPTY for t in thinks(ids)[0])
                for cc in e["calls"]:
                    rc = rcalls.get(ukey(cc))
                    paired[f"{L}: units"] += 1
                    if rc is None:
                        paired[f"{L}: units without a reader readout"] += 1
                        continue
                    paired[f"{L}: readout token id equal to the reader's ({cc['kind']})"] += int(ids[cc["readout"]] == r["ids"][rc["readout"]])
                    paired[f"{L}: {K_BEFORE} ids before the readout equal ({cc['kind']})"] += int(
                        ids[max(0, cc["readout"] - K_BEFORE): cc["readout"]] == r["ids"][max(0, rc["readout"] - K_BEFORE): rc["readout"]])
                    paired[f"{L}: units compared ({cc['kind']})"] += 1
                    if cc["kind"] == "output_end" and rc.get("span"):
                        s1, s2 = ids[cc["span"][0]: cc["span"][1]], r["ids"][rc["span"][0]: rc["span"][1]]
                        paired[f"{L}: output content spans with equal ids"] += int(h(s1) == h(s2))
                        paired[f"{L}: output content spans compared"] += 1
                        paired[f"{L}: output span tokens (served)"] += len(s1)
                        paired[f"{L}: output span tokens (reader)"] += len(s2)
    return {"world": world, "all_epochs": dict(sorted(c.items())),
            "B_alias_of_A": {"status": "B's system turn (donor prompt; submit as in A) equals A's, ids compared per epoch",
                             "epochs_verified": c["B vs A: system turns identical"], "failures": b_alias_fail}, "paired": dict(sorted(paired.items())), "paired_runs": sorted(rd),
            "empty_form_sha256": h(EMPTY), "tools_marker_ids": tools_marker}


@app.local_entrypoint()
def main(manifest_b: str, stage_record: str):
    git = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip()
    if git("status", "--porcelain", "--untracked-files=no", "--", str(HERE)):
        raise SystemExit("commit this folder before launching")
    mb = Path(manifest_b).read_bytes()
    if hashlib.sha256(mb).hexdigest() != PIN_B:
        raise SystemExit("corrected-B manifest differs from its pin")
    sp = Path(stage_record).resolve()
    st = json.loads(sp.read_bytes())
    man = json.loads(mb)
    staged_b = {e["key"]: (e["staged_inputs_file"], e["staged_inputs_sha256"]) for e in man["epochs"]}
    args = []
    for w in st["worlds"]:
        keys = [e["key"] for e in st["cohort"]["worlds"][w["world"]]]
        rng = random.Random(f"{SEED}|{w['world']}")
        pick = sorted(rng.sample(sorted(keys), min(N_PAIRED, len(keys))))
        src = {}
        for k in pick:
            f, s = staged_b[k] if w["world"] != "honest_c" else (C_STAGE, C_PIN)
            src.setdefault((f, s), []).append(k)
        args.append((w["world"], w["files"], [{"file": f, "sha256": s, "keys": ks} for (f, s), ks in sorted(src.items())]))
    outs = list(inspect_world.starmap(args))
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    rec = {"label": "PRELIMINARY: inspection before any score (ids, counts, hashes only)", "stamp": stamp, "git": git("rev-parse", "HEAD"),
           "stage_record": {"path": str(sp.relative_to(HERE.parents[1])), "sha256": hashlib.sha256(sp.read_bytes()).hexdigest(), "stamp": st["stamp"]},
           "seed": SEED, "n_paired_per_world": N_PAIRED, "worlds": outs}
    path = HERE / "results" / f"inspect_served_{stamp}.json"
    with path.open("x") as fh:
        fh.write(json.dumps(rec, indent=1))
    for o in outs:
        print(o["world"], json.dumps(o["all_epochs"]))
        print("  paired:", json.dumps(o["paired"]))
    print("->", path)
