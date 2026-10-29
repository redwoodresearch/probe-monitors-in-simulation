"""Row 4's staging: splice tool-definition lines in the system turn of staged replay epochs, at the token level. No text
is decoded. Modal CPU, cents.

For every epoch of the given staged arm files, each non-submit tool line whose ids differ between the source and the
target definitions is found exactly once inside the system turn (before the first <|im_end|>) and replaced. Every call's
request length n and readout move by the total length change, and its sha256 is recomputed over the edited ids[:n]
(int32 bytes), so run_extract_replay.py's gate checks the edited inputs. The step asserts that undoing the splice
restores the staged ids exactly, so no token outside the spliced spans changed. Sets:

- `sim`: the simulated attack arm (sim_attack@september) of the four 196-task stagings, simulated lines -> real lines;
- `mirror`: real attack run 1 of the same stagings, real lines -> simulated lines;
- `noop`: 20 seeded tasks' simulated epochs and 20 seeded tasks' run-2 epochs, each line -> itself (ids must not change).

Tool lines: build_tool_lines.py's ids, from the structured tools field. A task's real lines are run 1's, else run 2's;
the two are identical on every non-submit tool in all 192 tasks that have both.

  uv run --no-project --python 3.12 --with 'modal>=1.4.3' modal run stage_toolswap.py --lines <tool_lines.json> --sets sim,mirror,noop
"""

import hashlib
import json
from pathlib import Path

import modal

APP = "probe-shift_aswin-surya_sub-q3toolswap_stage_kill-2026-11-08"
ACTS = "probe-shift_aswin-surya_sub-q3onpol_acts_kill-2026-11-08"
STAGES = ["rerun_strict_segends_v2", "scaleup_canary_segends_v1", "scaleup_l2_segends_v1", "scaleup_topup_segends_v1"]
RUN2 = "second_attack_segends_v1"
INPUTS_SHA = {  # the extraction configs' recorded staged-input hashes (config/extract_*.yaml)
    ("rerun_strict_segends_v2", "sim_attack@september.jsonl"): "6111e4e8c4469876f937524536b7c319dade99ea1860853a85ee841c00b044b1",
    ("rerun_strict_segends_v2", "attack.jsonl"): "b5ff12088b626520d96152e8f2b12a93a60bb3c04f81e64919b52fddb717e07b",
    ("scaleup_canary_segends_v1", "sim_attack@september.jsonl"): "831d79c8474a52ec99da5468cfdf84ffc990a9d5258238491e6b0f9e33da274a",
    ("scaleup_canary_segends_v1", "attack.jsonl"): "58a5018bfec97475f91421479c6006035f2af1c91ab520690f07386abcda7b50",
    ("scaleup_l2_segends_v1", "sim_attack@september.jsonl"): "95cc51603c80d7f1d9ae8020b27f093b30c4800fee0ae3006690b90c2388538c",
    ("scaleup_l2_segends_v1", "attack.jsonl"): "44cb9cd757e15a640d3e0081db246f12623c5f558c5a907975b03eca417bf0ab",
    ("scaleup_topup_segends_v1", "sim_attack@september.jsonl"): "0e15a2fed0ea2225f70c2f6ad0f6026fd7593f4126d8448befff198b51cdc114",
    ("scaleup_topup_segends_v1", "attack.jsonl"): "515b9a1ae03fcaec2895df485336779c0968083fc282d7680895229b6f09f1e3",
    (RUN2, "attack.jsonl"): "672edb0dded2e739acb999f3a7a084905d75ab79b3b2b023dcaaefa55be69a1a",
}
IM_END, SEED, N_NOOP = 248046, 20261011, 20  # <|im_end|> in the Qwen3.5-27B tokenizer (checked locally before launch)
OUT = {"sim": "toolswap_sim_v2", "mirror": "toolswap_mirror_v2", "noop": "toolswap_noop_v2"}  # v1: a first attempt stopped at its first epoch (unresolved attachment references in the lines)

app = modal.App(APP)
vol = modal.Volume.from_name(ACTS, version=2)
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy")


def find_once(ids, sub, end):
    hits = [i for i in range(end - len(sub) + 1) if ids[i] == sub[0] and ids[i:i + len(sub)] == sub]
    if len(hits) != 1:
        raise ValueError(f"a tool line found {len(hits)} times in the system turn")
    return hits[0]


@app.function(image=image, cpu=4, memory=32768, timeout=3600, retries=0, volumes={"/acts": vol})
def stage(lines: dict, sets: list) -> dict:
    import numpy as np

    sha = lambda ids: hashlib.sha256(np.asarray(ids, np.int32).tobytes()).hexdigest()  # noqa: E731

    def splice(ep, src, dst):
        ids = ep["ids"]
        for c in ep["calls"]:
            if sha(ids[: c["n"]]) != c["sha256"]:
                raise ValueError(f"{ep['key']}: the staged ids differ from the staged hash")
        end = ids.index(IM_END)
        spans = sorted((find_once(ids, src[k], end), src[k], dst[k]) for k in src if k != "submit" and src[k] != dst[k])  # submit is never touched
        new, prev = [], 0
        for i, a, b in spans:
            if i < prev:
                raise ValueError(f"{ep['key']}: overlapping tool lines")
            new += ids[prev:i] + b
            prev = i + len(a)
        new += ids[prev:]
        delta = len(new) - len(ids)
        back, prev, j = [], 0, 0  # undo: every token outside the spans must be unchanged
        for i, a, b in spans:
            k = i + j
            back += new[prev:k] + a
            prev, j = k + len(b), j + len(b) - len(a)
        back += new[prev:]
        if back != ids:
            raise ValueError(f"{ep['key']}: undoing the splice does not restore the staged ids")
        sys_end = new.index(IM_END)
        calls = []
        for c in ep["calls"]:
            n = c["n"] + delta
            r = None if c["readout"] is None else c["readout"] + delta
            if r is not None and not sys_end < r < n:
                raise ValueError(f"{ep['key']}: a readout falls inside the system turn or past its request")
            if not n <= len(new):
                raise ValueError(f"{ep['key']}: a request is longer than its edited epoch")
            calls.append(c | {"n": n, "readout": r, "sha256": sha(new[:n])})
        return ep | {"ids": new, "calls": calls, "toolswap": {"delta": delta, "lines_replaced": len(spans)}}, delta, len(spans)

    def read(stage_, fname):
        p = Path("/acts") / stage_ / "inputs" / fname
        b = p.read_bytes()
        if hashlib.sha256(b).hexdigest() != INPUTS_SHA[(stage_, fname)]:
            raise ValueError(f"{stage_}/{fname}: staged file differs from its extraction config's hash")
        return [json.loads(x) for x in b.decode().splitlines()]

    def real_of(task):
        return lines[task]["real"]

    rng = np.random.default_rng(SEED)
    out = {}
    for s in sets:
        d = Path("/acts") / OUT[s] / "inputs"
        if d.exists():
            raise FileExistsError(f"{d} exists; stage under a new name")
        files, info = {}, {"epochs": 0, "tokens": 0, "delta": {}, "lines_replaced": 0, "tasks": set()}
        if s in ("sim", "mirror"):
            fname = "sim_attack@september.jsonl" if s == "sim" else "attack.jsonl"
            eps = [ep for st in STAGES for ep in read(st, fname)]
            keep = [ep for ep in eps if "sim" in lines.get(ep["task"], {}) and "real" in lines.get(ep["task"], {})]
            info["epochs_skipped_no_tool_pair"] = len(eps) - len(keep)  # tasks with one arm only: never scored (no pair)
            eps = keep
            todo = [(ep, lines[ep["task"]]["sim"], real_of(ep["task"])) if s == "sim" else (ep, real_of(ep["task"]), lines[ep["task"]]["sim"]) for ep in eps]
            files[fname] = todo
        else:
            sim_eps = [ep for st in STAGES for ep in read(st, "sim_attack@september.jsonl")]
            r2_eps = read(RUN2, "attack.jsonl")
            pick_s = set(rng.choice(sorted({e["task"] for e in sim_eps}), N_NOOP, replace=False).tolist())
            pick_r = set(rng.choice(sorted({e["task"] for e in r2_eps}), N_NOOP, replace=False).tolist())
            files["sim_attack@september.jsonl"] = [(ep, lines[ep["task"]]["sim"], lines[ep["task"]]["sim"]) for ep in sim_eps if ep["task"] in pick_s]
            files["attack.jsonl"] = [(ep, real_of(ep["task"]), real_of(ep["task"])) for ep in r2_eps if ep["task"] in pick_r]
            info["noop_tasks"] = {"sim": sorted(pick_s), "run2": sorted(pick_r)}
        d.mkdir(parents=True)
        shas = {}
        for fname, todo in files.items():
            res = []
            for ep, src, dst in todo:
                e2, delta, nrep = splice(ep, src, dst)
                if s == "noop" and (e2["ids"] != ep["ids"] or nrep != 0):
                    raise ValueError(f"{ep['key']}: the no-op splice changed the ids")
                res.append(e2)
                info["epochs"] += 1
                info["tokens"] += len(e2["ids"])
                info["delta"][str(delta)] = info["delta"].get(str(delta), 0) + 1
                info["lines_replaced"] += nrep
                info["tasks"].add(ep["task"])
            b = "".join(json.dumps(e) + "\n" for e in res).encode()
            (d / fname).write_bytes(b)
            shas[fname] = hashlib.sha256(b).hexdigest()
        vol.commit()
        out[s] = {k: (sorted(v) if isinstance(v, set) else v) for k, v in info.items()} | {"stage": OUT[s], "staged_inputs_sha256": shas}
        out[s]["tasks"] = len(out[s]["tasks"])
    return out


@app.function(image=image, cpu=2, memory=32768, timeout=1800, retries=0, volumes={"/acts": vol.read_only()})
def summarize(stage_name: str, fname: str) -> dict:
    """Read-only summary of a staged file already written (sha256, epochs, tokens, length changes, lines replaced)."""
    b = (Path("/acts") / stage_name / "inputs" / fname).read_bytes()
    eps = [json.loads(x) for x in b.decode().splitlines()]
    delta = {}
    for e in eps:
        delta[str(e["toolswap"]["delta"])] = delta.get(str(e["toolswap"]["delta"]), 0) + 1
    return {"stage": stage_name, "file": fname, "sha256": hashlib.sha256(b).hexdigest(), "epochs": len(eps),
            "tokens": sum(len(e["ids"]) for e in eps), "tasks": len({e["task"] for e in eps}), "delta": delta,
            "lines_replaced": sum(e["toolswap"]["lines_replaced"] for e in eps)}


@app.local_entrypoint()
def main(lines: str, sets: str = "sim,mirror,noop", summarize_only: str = ""):
    if summarize_only:
        st, fname = summarize_only.split(":")
        print(json.dumps(summarize.remote(st, fname), indent=1))
        return
    raw = json.loads(Path(lines).read_text())["sets"]
    merged = {}
    for stage_, tasks in raw.items():
        for task, v in tasks.items():
            m = merged.setdefault(task, {})
            for side in ("real", "sim"):
                if side not in v:
                    continue
                key = "run2" if stage_ == RUN2 else side
                cur = {x["name"]: x["ids"] for x in v[side]}
                if key in m and m[key] != cur:
                    raise ValueError(f"{task}: {key} lines differ between stagings")
                m[key] = cur
    for task, m in merged.items():
        if "real" not in m and "run2" in m:
            m["real"] = m["run2"]  # identical to run 1 on every non-submit tool wherever both exist (checked locally)
        if "real" in m and "run2" in m and {k: v for k, v in m["real"].items() if k != "submit"} != {k: v for k, v in m["run2"].items() if k != "submit"}:
            raise ValueError(f"{task}: run-1 and run-2 tool lines differ")
    print(json.dumps(stage.remote(merged, sets.split(",")), indent=1))
