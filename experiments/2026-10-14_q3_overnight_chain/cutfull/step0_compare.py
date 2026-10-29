"""RULE.md Append 41 step 0 (09 @ ca431944c, amendment 1 @ 6a37d4193, §9/§10 @ 0660c0fa8; key list and spec v2 @ f3bfe9208): the
reproduction gate of the uncut ("cutfull") extraction against the registered 32k extraction, per selected run, on Modal CPU (never
the laptop). Token ids, positions, unit keys, activations and hashes only: no transcript text is read, printed or written.

Per run (key, arm), with C = the registered cut staging / extraction and U = the cutfull gate staging / extraction:
0. binding: each side's staged file is the one its extraction read (config.json inputs == the staged set, inputs_sha256 of the
   arm's file == the file's sha256); each meta names the run's key and arm and its chunk index lists the key; no duplicate unit.
1. ids: U's staged ids begin with C's staged ids (L = len(C ids), at most 32,768). A difference before the last 3 pre-boundary
   positions [L-3, L) FAILS; differences inside them are listed (amendment 1 §5, corrected-render differences).
2. units: a unit is (irid, kind, src_irid, call_index) at its readout position (a last-token row), or the same with "mean" and its
   span [start, end) (an output_end span-mean row). Inside the first L tokens: U's units (readout < L; span end <= L) must EQUAL C's
   units (09 §10 addition 1: no missing, no extra; identical positions and spans). Units whose readout or span touches [L-3, L) are
   EXCLUDED from both sides and listed by key, when the run is longer than L (amendment 1 §5).
3. activations: every compared unit at EVERY layer, all 64 (spec v2: both extractions must store exactly layers 0..63), float32:
   max|U - C| / ||C||_2 <= 1e-2 AND cosine(U, C) >= 0.9999 (09 §10). The worst unit per run and overall is reported for both
   metrics, by key, unit, position and layer (09 §10 addition 2).
4. past L (amendment 1 §5): U's meta n_tokens == len(U staged ids) == the run's tokens_before (required, from the step-0 file) ==
   its tokens_total; one forward pass over n_tokens padded only after the last id (padded_tokens == n + (-n % pad_to); positions
   0..n-1 by construction); U meta says cut "full" and truncated false; every U readout < n_tokens; U's stage record says cut "full".
5. pins: C's and U's extraction config.json name the same (non-null) model, revision, pad_to, chunk_size, and the revision is the
   spec's reader pin; their stage records' scrub level, labeler fingerprint and model are equal and non-null; the HF-cache snapshot
   files' sha256 equal the spec's (weights_pin). The gate record also pins this checkout's stager, render, extractor and compare
   code (sha256 per file, git HEAD, clean tree) and the extractor's execution settings, read from its source.
Any failure, a run missing its C or U units, or a pin mismatch: the gate FAILS (STOP; no main extraction) and the command exits
nonzero after the record is written. The fan-out runs in gate_driver on Modal (run with --detach): its started marker (call id) and
its record go to the q3cutfull Volume under step0_gate/, so a laptop sleep cannot cancel the gate or lose its result.

  CUTCMP_SPEC=<compare spec json from make_cutfull_targets.py --step0> uv run --no-project --python 3.12 --with modal==1.6.1 \
      modal run --detach experiments/2026-10-14_q3_overnight_chain/cutfull/step0_compare.py
"""

import hashlib
import json
import os
import re
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).resolve().parent
CHAIN = HERE.parent
REPO = CHAIN.parents[1] if len(CHAIN.parents) > 1 else CHAIN  # inside the Modal container the script sits at /root: import-safe
APP = "probe-shift_aswin-surya_sub-q3cutfull_cmp_kill-2026-11-14"
CUT_VOL = "probe-shift_aswin-surya_sub-q3su311_acts_kill-2026-11-14"  # the registered 32k stagings and extractions (read only)
UNC_VOL = "probe-shift_aswin-surya_sub-q3cutfull_acts_kill-2026-11-14"  # the cutfull stagings and extractions (read only here)
HF_CACHE = "probe-shift_aswin-surya_hf-cache"
TOL_MAXABS_REL, TOL_COS, BOUNDARY_ZONE, N_LAYERS = 1e-2, 0.9999, 3, 64  # 09 §10, amendment 1 §5, spec v2; fixed BEFORE the run
PINNED_FILES = ["experiments/2026-10-14_q3_overnight_chain/stage_q4b.py", "experiments/2026-10-14_q3_overnight_chain/stage_chain.py",
                "experiments/2026-10-14_q3_overnight_chain/fmt_gate.py", "experiments/2026-10-14_q3_overnight_chain/evalzip.py",
                "experiments/2026-10-14_q3_overnight_chain/extract_chain.py", "experiments/2026-10-14_q3_overnight_chain/cutfull/step0_compare.py",
                "src/probe_shift/reader.py", "src/probe_shift/transcripts.py", "src/probe_shift/models.py"]

app = modal.App(APP)
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy==2.3.3")
VOLS = {"/cut": modal.Volume.from_name(CUT_VOL, version=2).read_only(), "/unc": modal.Volume.from_name(UNC_VOL, version=2).read_only()}


def filename(key: str) -> str:  # extract_chain.filename
    return hashlib.sha1(key.encode()).hexdigest()[:20]


def ids_sha(ids) -> str:
    import numpy as np

    return hashlib.sha256(np.asarray(ids, np.int32).tobytes()).hexdigest()


def staged_epoch(root: Path, arm: str, key: str) -> tuple[dict, str]:
    """The run's staged epoch and the sha256 of the arm's whole staged file."""
    raw = (root / "inputs" / f"{arm}.jsonl").read_bytes()
    hits = [e for e in (json.loads(line) for line in raw.decode().splitlines() if f'"{key}"' in line) if e["key"] == key]
    if len(hits) != 1:
        raise KeyError(f"{key}: staged {len(hits)} times in {root}/inputs/{arm}.jsonl")
    return hits[0], hashlib.sha256(raw).hexdigest()


def stage_record(root: Path) -> dict:
    recs = sorted(root.glob("stage_2*.json"))
    if len(recs) != 1:
        raise ValueError(f"{root}: {len(recs)} stage records")
    return json.loads(recs[0].read_text())


def units(meta: dict, limit: int | None) -> dict:
    """(unit, form, position) -> row: last-token rows at their readout and output_end span means at their span; with `limit`,
    only those inside the first `limit` tokens. A repeated unit raises."""
    out = {}

    def add(u, r):
        if u in out:
            raise ValueError(f"duplicate unit {u}")
        out[u] = r

    for c in meta["calls"]:
        uid = (c.get("irid"), c.get("kind"), c.get("src_irid"), c.get("call_index"))
        if c.get("row") is not None and (limit is None or c["readout"] < limit):
            add((uid, "last", c["readout"]), meta["row_start"] + c["row"])
        if c.get("row_mean") is not None and (limit is None or c["span"][1] <= limit):
            add((uid, "mean", tuple(c["span"])), meta["row_start"] + c["row_mean"])
    return out


def bind(s: dict, key: str, arm: str):
    """True, or the first broken link: the staged file is the one the extraction read (config inputs + inputs_sha256, meta
    inputs_sha256); the meta is this run's; exactly one chunk entry lists it at the meta's row_start and n_rows; its rows are
    distinct, inside the entry and fill it; the chunk's arrays hold every row; the meta's calls are the staged calls."""
    cfg, m, idx = s["cfg"], s["meta"], s["idx"]
    if cfg.get("inputs") != s["staged"] or cfg.get("inputs_sha256", {}).get(f"{arm}.jsonl") != s["file_sha"] or m.get("inputs_sha256") != s["file_sha"]:
        return "inputs"
    if m.get("key") != key or m.get("arm") != arm:
        return "meta key/arm"
    ent = [e for e in idx["epochs"] if e["key"] == key and e["arm"] == arm]
    if len(ent) != 1 or ent[0]["row_start"] != m["row_start"] or ent[0]["n_rows"] != m["n_rows"]:
        return "chunk entry"
    if not all(isinstance(x, int) and not isinstance(x, bool) for x in (m["row_start"], m["n_rows"], idx["rows"])) or not (
            0 <= m["row_start"] <= m["row_start"] + m["n_rows"] <= idx["rows"]):
        return "row bounds"
    strip = lambda c: {k: v for k, v in c.items() if k not in ("sha256", "row", "row_mean")}  # noqa: E731
    st = s["ep"]["calls"]
    if [strip(c) for c in m["calls"]] != [strip(c) for c in st]:
        return "calls"
    # the extractor's allocation (extract_chain.extract_shard), from the STAGED calls: a last-token row per non-null readout in call
    # order, then a span-mean row per output_end with a span, in call order
    kept = [i for i, c in enumerate(st) if c["readout"] is not None]
    means = [i for i in kept if st[i]["kind"] == "output_end" and st[i].get("span")]
    row, mean = {i: j for j, i in enumerate(kept)}, {i: len(kept) + j for j, i in enumerate(means)}
    if m["n_rows"] != len(kept) + len(means) or any(c.get("row") != row.get(i) or c.get("row_mean") != mean.get(i) for i, c in enumerate(m["calls"])):
        return "rows"
    return True


def touches_zone(u, lo: int, hi: int) -> bool:
    _, form, pos = u
    if form == "last":
        return lo <= pos < hi
    a, b = pos
    return a < hi and b > lo  # span [a, b) overlaps [lo, hi)


@app.function(image=image, cpu=8, memory=65536, timeout=3 * 3600, retries=0, volumes=VOLS)
def compare_run(row: dict, reader_revision: str) -> dict:
    import numpy as np

    key, arm = row["key"], row["arm"]
    out = {"key": key, "arm": arm, "checks": {}}
    fail = []
    C = {"vol": Path("/cut"), "staged": row["cut_staged_set"], "root": row["cut_extraction_root"]}
    U = {"vol": Path("/unc"), "staged": row["gate_staged_set"], "root": row["gate_extraction_root"]}
    for s in (C, U):  # 0. binding
        s["ep"], s["file_sha"] = staged_epoch(s["vol"] / s["staged"], arm, key)
        s["cfg"] = json.loads((s["vol"] / s["root"] / "config.json").read_text())
        s["meta"] = json.loads((s["vol"] / s["root"] / "meta" / arm / f"{filename(key)}.json").read_text())
        s["idx"] = json.loads((s["vol"] / s["root"] / "chunks" / f"{s['meta']['chunk']}.json").read_text())
        s["stage"] = stage_record(s["vol"] / s["staged"])
        s["bound"] = bind(s, key, arm)
        s["attempts"] = {p.name: json.loads(p.read_text()).get("git") for p in sorted((s["vol"] / s["root"]).glob("attempt_*.json"))}
    out["checks"]["binding"] = {"cut": C["bound"], "uncut": U["bound"], "cut_file_sha": C["file_sha"], "uncut_file_sha": U["file_sha"]}
    out["extraction_attempts_git"] = {"cut": C["attempts"], "uncut": U["attempts"]}
    if C["bound"] is not True or U["bound"] is not True:
        fail.append("binding")
    cids, uids = C["ep"]["ids"], U["ep"]["ids"]
    L = len(cids)  # 1. ids; the reference is the registered 32k cut
    out["checks"]["reference_cut"] = {"stage_cut": C["stage"].get("cut"), "meta_n_tokens": C["meta"].get("n_tokens"), "L": L,
                                      "expected_L": min(32768, row["tokens_before"])}
    if not (C["stage"].get("cut") == 32768 and C["meta"].get("n_tokens") == L == min(32768, row["tokens_before"])):
        fail.append("reference cut")
    diff = [i for i in range(min(L, len(uids))) if cids[i] != uids[i]]
    early = [i for i in diff if i < L - BOUNDARY_ZONE]
    out["checks"]["ids"] = {"cut_len": L, "uncut_len": len(uids), "cut_sha": ids_sha(cids), "uncut_prefix_sha": ids_sha(uids[:L]),
                            "diff_positions_in_boundary_zone": [i for i in diff if i >= L - BOUNDARY_ZONE], "n_diff_before_zone": len(early)}
    if early or len(uids) < L:
        fail.append("ids prefix")
    cu, uu = units(C["meta"], None), units(U["meta"], L)  # 2. units
    excluded = []
    if len(uids) > L:  # amendment 1 §5: units touching the last BOUNDARY_ZONE pre-boundary tokens, both sides, listed by key
        zone = (L - BOUNDARY_ZONE, L)
        excluded = sorted({u for u in [*cu, *uu] if touches_zone(u, *zone)}, key=str)
        cu = {u: r for u, r in cu.items() if u not in excluded}
        uu = {u: r for u, r in uu.items() if u not in excluded}
    missing, extra = sorted(set(cu) - set(uu), key=str), sorted(set(uu) - set(cu), key=str)
    out["checks"]["units"] = {"cut_units": len(cu), "uncut_units_inside": len(uu), "excluded_boundary": [list(map(str, u)) for u in excluded],
                              "uncut_units_past_L": len(units(U["meta"], None)) - len(units(U["meta"], L)),
                              "missing_in_uncut": len(missing), "extra_in_uncut": len(extra),
                              "missing_examples": [list(map(str, u)) for u in missing[:5]], "extra_examples": [list(map(str, u)) for u in extra[:5]]}
    if not cu:
        fail.append("no cut units")
    if missing or extra:
        fail.append("key-set equality")
    layers = list(range(N_LAYERS))  # 3. activations, every layer
    stored = {"cut": C["idx"]["layers"], "uncut": U["idx"]["layers"]}
    if any(sorted(v) != layers for v in stored.values()):
        fail.append("not all 64 layers stored")
        layers = sorted(set(stored["cut"]) & set(stored["uncut"]))
    common = sorted(set(cu) & set(uu), key=str)
    worst = {"maxabs_rel": (-1.0, None), "cosine": (2.0, None)}
    n_bad, n_pairs = 0, 0
    for l in layers:
        ca = np.load(C["vol"] / C["root"] / "by_layer" / f"L{l}" / f"{C['meta']['chunk']}.npy", mmap_mode="r")
        ua = np.load(U["vol"] / U["root"] / "by_layer" / f"L{l}" / f"{U['meta']['chunk']}.npy", mmap_mode="r")
        if ca.shape[0] != C["idx"]["rows"] or ua.shape[0] != U["idx"]["rows"]:
            fail.append(f"L{l} array rows")
            continue
        for u in common:
            c = np.asarray(ca[cu[u]], np.float32)
            v = np.asarray(ua[uu[u]], np.float32)
            nc, nv = float(np.linalg.norm(c)), float(np.linalg.norm(v))
            rel = float(np.abs(v - c).max()) / nc if nc > 0 else float("inf")
            cos = float(c @ v / (nc * nv)) if nc > 0 and nv > 0 else -1.0
            bad = not (rel <= TOL_MAXABS_REL and cos >= TOL_COS)
            n_bad += bad
            n_pairs += 1
            where = {"key": key, "arm": arm, "unit": list(map(str, u[0])), "form": u[1], "position": u[2] if u[1] == "last" else list(u[2]), "layer": l}
            if rel > worst["maxabs_rel"][0]:
                worst["maxabs_rel"] = (rel, where)
            if cos < worst["cosine"][0]:
                worst["cosine"] = (cos, where)
    out["checks"]["activations"] = {"layers_compared": len(layers), "units_compared": len(common), "unit_layer_pairs": n_pairs,
                                    "unit_layer_failures": n_bad,
                                    "worst_maxabs_rel": {"value": worst["maxabs_rel"][0], **(worst["maxabs_rel"][1] or {})},
                                    "worst_cosine": {"value": worst["cosine"][0], **(worst["cosine"][1] or {})},
                                    "tolerance": {"maxabs_rel_le": TOL_MAXABS_REL, "cosine_ge": TOL_COS}}
    if n_bad or not n_pairs:
        fail.append("activation tolerance")
    um, n = U["meta"], U["meta"]["n_tokens"]  # 4. past L: full consumption, continuity, no truncation
    pad = U["cfg"].get("pad_to")
    past = {"n_tokens": n, "uncut_staged_len": len(uids), "tokens_before_spec": row["tokens_before"], "padded_tokens": um.get("padded_tokens"),
            "padded_expected": n + (-n % pad) if pad else None, "meta_cut": um.get("cut"), "meta_truncated": um.get("truncated"),
            "meta_tokens_total": um.get("tokens_total"), "stage_cut": U["stage"].get("cut"),
            "max_readout": max((c["readout"] for c in um["calls"] if c.get("readout") is not None), default=None)}
    out["checks"]["past_boundary"] = past
    if not (n == len(uids) == row["tokens_before"] == past["meta_tokens_total"] and pad and past["padded_tokens"] == past["padded_expected"]
            and past["meta_cut"] == "full" and past["meta_truncated"] is False and past["stage_cut"] == "full"
            and (past["max_readout"] is None or past["max_readout"] < n)):
        fail.append("past-boundary consumption")
    pins = {k: [C["cfg"].get(k), U["cfg"].get(k)] for k in ("model", "revision", "pad_to", "chunk_size")}  # 5. pins
    for k in ("scrub_level", "labeler_fingerprint", "model"):
        pins[f"stage_{k}"] = [C["stage"].get(k), U["stage"].get(k)]
    out["checks"]["pins"] = pins
    if any(a is None or a != b for a, b in pins.values()) or C["cfg"].get("revision") != reader_revision:
        fail.append("pins")
    out["gate_extraction_attempts"] = sorted(p.name for p in (U["vol"] / U["root"]).glob("attempt_*.json"))
    out["fail"], out["passes"] = fail, not fail
    return out


@app.function(image=image, cpu=2, memory=4096, timeout=1800, retries=0, volumes={"/hf": modal.Volume.from_name(HF_CACHE).read_only()})
def weights_pin(model: str, revision: str, files: dict) -> dict:
    """sha256 of the reader's pinned snapshot files on the HF cache Volume (spec v2 pins)."""
    snap = Path("/hf/hub") / f"models--{model.replace('/', '--')}" / "snapshots" / revision
    got = {k: hashlib.sha256((snap / f).read_bytes()).hexdigest() if (snap / f).exists() else None for k, f in files.items()}
    return {"snapshot": str(snap), "sha256": got}


def settings_from_source() -> dict:
    """The extractor's execution settings, read from its source at this checkout (not retyped)."""
    src = (CHAIN / "extract_chain.py").read_text()
    return {"gpu": re.findall(r'gpu="([^"]+)"', src), "dtype": re.findall(r"dtype=torch\.(\w+)", src),
            "attn_implementation": re.findall(r'attn_implementation="(\w+)"', src), "pip": json.loads(re.search(r"^PIP = (\[.*\])$", src, re.M).group(1)),
            "python": re.findall(r'python_version="([\d.]+)"', src)}


def expected_runs(spec: dict) -> None:
    """Re-read the registered step-0 file (git ref), check its sha256, and require the spec's runs to be exactly its selection: the
    same (key, arm) set, unique, each with its tokens_before, cut staged set and root, and the naming rule's gate set and root."""
    raw = subprocess.run(["git", "show", spec["source"]["ref"]], capture_output=True, check=True, cwd=REPO).stdout
    if hashlib.sha256(raw).hexdigest() != spec["source"]["sha256"]:
        raise SystemExit("the step-0 source file's sha256 differs from the spec's")
    s0 = json.loads(raw)["step0"]
    want = []
    for r in s0["runs"]:
        t = re.sub(r"_v1(_try[0-9]+)?$", "", r["cut"]["staged_set"])
        want.append({"key": r["key"], "arm": r["arm"], "original_target": t, "tokens_before": r["tokens_before"],
                     "cut_staged_set": r["cut"]["staged_set"], "cut_extraction_root": r["cut"]["extraction_root"],
                     "gate_staged_set": f"{t}_cutfull_gate_v1", "gate_extraction_root": f"{t}_cutfull_gate_extract_v1"})
    ka = [(r["key"], r["arm"]) for r in spec["runs"]]
    canon = lambda rs: sorted(json.dumps(r, sort_keys=True) for r in rs)  # noqa: E731
    if (len(ka) != len(set(ka)) or canon(spec["runs"]) != canon(want)
            or sorted(r["key"] for r in want) != sorted(s0["selection"]["keys"]) or len(want) != s0["selection"]["n"]):
        raise SystemExit("the compare spec's runs are not exactly the registered step-0 selection")


@app.function(image=image, cpu=1, memory=2048, timeout=4 * 3600, retries=0)
def gate_driver(spec: dict, spec_meta: dict, pins: dict, stamp: str) -> dict:
    """The fan-out on Modal, not in the laptop client (0d: a laptop sleep must not cancel the gate): a started marker with this
    call's id, then the weights pin and the per-run compares, then the record, all on the q3cutfull Volume under step0_gate/."""
    import io

    vol = modal.Volume.from_name(UNC_VOL, version=2)
    name = f"cutfull_step0_compare_{stamp}"
    with vol.batch_upload(force=True) as b:
        b.put_file(io.BytesIO(json.dumps({"gate_driver_call_id": modal.current_function_call_id(), "stamp": stamp}).encode()),
                   f"step0_gate/{name}.started.json")
    wp = spec["weights_pin"]
    pins["weights"] = weights_pin.remote(spec["reader"]["name"], spec["reader"]["revision"], wp["files"]) | {"want": wp["sha256"]}
    pins["weights"]["passes"] = pins["weights"]["sha256"] == wp["sha256"]
    rows = list(compare_run.starmap([(r, spec["reader"]["revision"]) for r in spec["runs"]], return_exceptions=True))
    res = [r if isinstance(r, dict) else {"error": repr(r)[:300]} for r in rows]
    for r, s in zip(res, spec["runs"]):
        if "error" in r:
            r |= {"key": s["key"], "arm": s["arm"], "passes": False, "fail": ["error"]}
    allw = [r["checks"]["activations"] for r in res if "activations" in r.get("checks", {})]
    overall = {"maxabs_rel": max((a["worst_maxabs_rel"] for a in allw), key=lambda w: w["value"], default=None),
               "cosine": min((a["worst_cosine"] for a in allw), key=lambda w: w["value"], default=None)}
    ok = (bool(res) and len(res) == len(spec["runs"]) and all(r.get("passes", False) for r in res) and pins["weights"]["passes"]
          and not pins["git_dirty"])
    out = {"what": "Append 41 step-0 reproduction gate (cutfull vs registered 32k)", "spec_file": str(spec_meta["spec_file"]),
           "spec_sha256": spec_meta["spec_sha256"], "step0_source": spec["source"],
           "tolerance": {"maxabs_rel_le": TOL_MAXABS_REL, "cosine_ge": TOL_COS, "boundary_zone": BOUNDARY_ZONE, "layers": N_LAYERS},
           "pins": pins, "runs": res, "worst_overall": overall, "n_runs": len(res), "n_pass": sum(r.get("passes", False) for r in res),
           "excluded_boundary_units": {r["key"]: r["checks"]["units"]["excluded_boundary"] for r in res
                                       if r.get("checks", {}).get("units", {}).get("excluded_boundary")},
           "passes": ok}
    with vol.batch_upload(force=True) as b:
        b.put_file(io.BytesIO(json.dumps(out, indent=1).encode()), f"step0_gate/{name}.json")
    return out


@app.local_entrypoint()
def main():
    spec_path = Path(os.environ["CUTCMP_SPEC"]).resolve()
    raw = spec_path.read_bytes()
    spec = json.loads(raw)
    git = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True, check=True, cwd=REPO).stdout.strip()  # noqa: E731
    expected_runs(spec)  # the run list is exactly the registered selection (raises otherwise, before any Modal call)
    pins = {"git_head": git("rev-parse", "HEAD"), "git_dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
            "files_sha256": {f: hashlib.sha256((REPO / f).read_bytes()).hexdigest() for f in PINNED_FILES},
            "execution_settings": settings_from_source(), "reader": spec["reader"]}
    if pins["git_dirty"]:
        raise SystemExit("refusing to run the gate from a dirty tree")
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    call = gate_driver.spawn(spec, {"spec_file": str(spec_path.relative_to(REPO)), "spec_sha256": hashlib.sha256(raw).hexdigest()}, pins, stamp)
    print(json.dumps({"gate_driver_call_id": call.object_id, "stamp": stamp, "volume_record": f"{UNC_VOL}:step0_gate/cutfull_step0_compare_{stamp}.json"}), flush=True)
    out = call.get()  # run with --detach: a disconnect cannot cancel the driver; its record is on the Volume either way
    ok = out["passes"]
    dest = CHAIN / "results" / f"cutfull_step0_compare_{stamp}.json"
    dest.open("x").write(json.dumps(out, indent=1))
    print(dest.name, "PASS" if ok else "FAIL", f"{out['n_pass']}/{out['n_runs']}", "weights", out["pins"]["weights"]["passes"])
    if not ok:
        raise SystemExit(f"step-0 gate FAILED: {[(r['key'], r.get('fail')) for r in out['runs'] if not r.get('passes')]}")
