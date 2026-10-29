"""POST HOC (PLAN_identical_POSTHOC.md here; 0d's go about 11:40 PT 10-13): which of the registered Llama-3.3-70B rows are
byte-identical to an output of the other arm's run of the same cell. One Modal CPU container, about $0.02, no model and no
activations. It reads the staged transcripts the registered job (run_rows.py) read for Llama and hashes each tool output's
raw content (sha256, utf-8) inside the container. Only per-output flags and counts leave it: no text, no hash (the staged
files are verified inside against the registered job's digests, which the local side passes in). A failed
check crosses with its count-only message; any other error with its type, file and line only.

Units are the registered rows record's Llama units (real e2, and repaired-plus from its three sources), paired by cell.
An output of a unit is identical if its content's sha256 equals that of at least one tool output, kept or not, of the
paired unit; such outputs also get a group id, an integer local to the cell shared by every output (in both runs) with the
same content (PLAN Amendment 1). Step and action are numbered as run_rows.py's categories() numbers them, so the local side joins the flags to
the registered rows by (unit, step, action).

  PYTHONPATH=src uv run --no-project --python 3.12 --with "modal==1.6.1" --with numpy modal run \\
      experiments/2026-10-13_q2_llama_gap_where/run_identical_POSTHOC.py --rows <results/rows_2026-10-13_110759.json> \\
      --manifest <results/rows_2026-10-13_110759_manifest.json> --llama-runs <Llama run_scores_2026-10-12_143129> \\
      --build <G2's build_plus_2026-10-13_014244.json>
"""
import hashlib
import io
import json
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).parent
APP = "probe-shift_aswin-surya_sub-q2-gapwhere-ident_rows_kill-2026-11-13"
LA, L_ACTS = "probe-shift_aswin-surya_sub-la-activations", "probe-shift_aswin-surya_sub-q2-llama_acts_kill-2026-11-12"
DUMP, FIX_INPUTS = "2026-10-06_la_segment-end_qwen3-32b_v1", "2026-10-11_q3_fix939_inputs_v1"
FIX_INPUTS_SHA = "57cfbfbf1e4571690769605d2093d73edc93b958e06fed5e8efe8fa88587530d"
PLUS = "2026-10-12_la_output-end-r2j_llama33-70b_alllayers_v1_plus/inputs/plus.jsonl"

app = modal.App(APP)
vols = {"/la": modal.Volume.from_name(LA, version=2).read_only(), "/lacts": modal.Volume.from_name(L_ACTS, version=2).read_only()}
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy==2.3.3")


class Gate(Exception):
    """Our own checks only: the message holds counts and names, never staged text."""


def check(ok, msg):
    if not ok:
        raise Gate(msg)


def hashes(msgs):
    """Per tool output, numbered as run_rows.categories() numbers them: (step, action, sha256 of its raw content)."""
    out, step, action = [], -1, 0
    for m in msgs:
        if m["role"] == "assistant":
            step, action = step + 1, 0
        elif m["role"] == "tool":
            c = m.get("content")
            text = "".join(p.get("text", "") for p in c if p.get("type") == "text") if isinstance(c, list) else (c or "")
            out.append((step, action, hashlib.sha256(text.encode("utf-8", "surrogatepass")).hexdigest())); action += 1
    return out


@app.function(image=image, volumes=vols, cpu=8, memory=32768, timeout=3600, retries=0)
def flags(units: list[dict], plus_sha: str, dump_sha: dict) -> dict:
    try:
        return _flags(units, plus_sha, dump_sha)
    except Gate as e:
        raise RuntimeError(f"check failed: {e}") from None
    except Exception as e:  # any other error crosses as its type, file and line only (its message may quote staged text)
        import traceback
        f = traceback.extract_tb(e.__traceback__)[-1]
        raise RuntimeError(f"identical failed: {type(e).__name__} at {Path(f.filename).name}:{f.lineno}") from None


def _flags(units, plus_sha, dump_sha):
    import numpy as np
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()  # noqa: E731
    need, H, rec = {(u["src"], u["key"]) for u in units}, {}, {}

    def take(src, key, m):
        if (src, key) in need:
            check((src, key) not in H, f"a staged {src} transcript appears twice")
            H[(src, key)] = hashes(m)
    shards = sorted((Path("/la") / DUMP / "inputs").glob("shard_*.jsonl"))
    check(sorted(f"{DUMP}/inputs/{p.name}" for p in shards) == sorted(dump_sha), "the staged shards aren't the registered job's")
    for p in shards:
        data = p.read_bytes()
        check(hashlib.sha256(data).hexdigest() == dump_sha[f"{DUMP}/inputs/{p.name}"], "a staged shard differs from the registered job's")
        for line in data.splitlines():
            it = json.loads(line)
            if it["arm"] in ("real-attack-e2", "sim-attack"):
                take("e2" if it["arm"] == "real-attack-e2" else "sim", it["key"], it["messages"])
        del data
    fx = sorted((Path("/la") / FIX_INPUTS / "inputs").glob("shard_*.jsonl"))
    check([p.name for p in fx] == ["shard_0.jsonl"] and sha(fx[0]) == FIX_INPUTS_SHA, "fix939's inputs are not the pinned ones")
    for it in map(json.loads, fx[0].read_text().splitlines()):
        take("fix", it["key"], it["messages"])
    pp = Path("/lacts") / PLUS
    check(sha(pp) == plus_sha, "plus.jsonl is not the build's")
    for it in map(json.loads, pp.read_text().splitlines()):
        take("plus", it["key"], it["messages"])
    check(set(H) == need, f"{len(need - set(H))} needed staged transcripts not found")
    rec["staged inputs verified against the registered job's (shards, fix939, plus)"] = [len(shards), 1, 1]
    by_cell = {}
    for j, u in enumerate(units):
        check(u["arm"] not in by_cell.get(u["cell"], {}), "a cell with two units of one arm")
        by_cell.setdefault(u["cell"], {})[u["arm"]] = j
    check(all(set(v) == {"e2", "rep"} for v in by_cell.values()), "a cell without exactly one real e2 and one repaired unit")
    rows = []
    for v in by_cell.values():  # per cell: a group id per content hash both runs share (an integer local to the cell), -1 otherwise
        h_e, h_r = H[(units[v["e2"]]["src"], units[v["e2"]]["key"])], H[(units[v["rep"]]["src"], units[v["rep"]]["key"])]
        shared = sorted({h for _, _, h in h_e} & {h for _, _, h in h_r})
        gid = {h: g for g, h in enumerate(shared)}
        for j, hs in ((v["e2"], h_e), (v["rep"], h_r)):
            rows += [(j, s, a, int(h in gid), gid.get(h, -1)) for s, a, h in hs]
    A = np.array(rows, dtype=np.int64)
    rec["tool outputs, identical (all outputs, both arms)"] = [int(len(A)), int(A[:, 3].sum())]
    buf = io.BytesIO(); np.savez(buf, unit=A[:, 0], step=A[:, 1], action=A[:, 2], identical=A[:, 3], group=A[:, 4])
    return {"npz": buf.getvalue(), "record": rec}


@app.local_entrypoint()
def main(rows: str, manifest: str, llama_runs: str, build: str):
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()  # noqa: E731
    for f in (Path(__file__).resolve(), HERE / "PLAN_identical_POSTHOC.md", Path(manifest).resolve()):
        if subprocess.run(["git", "ls-files", "--error-unmatch", str(f)], cwd=HERE, capture_output=True).returncode \
                or subprocess.run(["git", "diff", "--quiet", "HEAD", "--", str(f)], cwd=HERE).returncode:
            raise SystemExit(f"{f.name} differs from HEAD: commit it before running")
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=HERE).stdout.strip()
    man = json.loads(Path(manifest).read_text())
    if sha(rows) != man["rows record sha256"] or sha(build) != "1ab8c14c2465d8f4575e9212f6f09c8a45fe719365e5e4b63496a4e4fd20e8c0":
        raise SystemExit("not the registered rows record or G2's build record")
    rec_ = json.loads(Path(rows).read_text())
    runs = json.loads((Path(llama_runs) / "runs.json").read_text())
    if sha(Path(llama_runs) / "runs.json") != rec_["inputs sha256"]["llama runs.json"]:
        raise SystemExit("not the runs.json the registered rows were built from")
    U = rec_["units (per reader: runs.json index, key sha256, source, fold; arrays' unit indexes this)"]["llama"]
    units = [{"key": runs[i]["key"], "src": s, "cell": runs[i]["cell"], "arm": "e2" if s == "e2" else "rep"} for i, s in zip(U["runs.json index"], U["source"])]
    if [sum(u["arm"] == a for u in units) for a in ("e2", "rep")] != [939, 939]:
        raise SystemExit("not 939 real e2 and 939 repaired units")
    dump_sha = {k: v for k, v in rec_["container"]["inputs sha256"].items() if k.startswith(f"{DUMP}/inputs/")}
    if len(dump_sha) < 1 or len({(u["cell"], u["arm"]) for u in units}) != 1878:
        raise SystemExit("no registered shard digests, or a cell without one unit per arm")
    (HERE / "results").mkdir(exist_ok=True)
    res = flags.remote(units, json.loads(Path(build).read_text())["plus sha256"], dump_sha)
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    npz = HERE / "results" / f"identical_POSTHOC_{stamp}.npz"
    with open(npz, "xb") as fh:
        fh.write(res["npz"])
    rec = {"POST HOC": True, "plan": "PLAN_identical_POSTHOC.md (this folder)", "git": head, "app": APP,
           "definition": "an output is identical if its raw content's sha256 equals that of at least one tool output, kept or not, of the same cell's other-arm run",
           "units": "the registered rows record's Llama units, in its order (arrays' unit indexes them)",
           "inputs sha256": {Path(rows).name: sha(rows), Path(manifest).name: sha(manifest), "llama runs.json": sha(Path(llama_runs) / "runs.json"), Path(build).name: sha(build)},
           "staged inputs": "verified in the container against the registered rows record's digests (its container 'inputs sha256'), fix939's pin and the build's plus sha256",
           "container": res["record"], "flags file": npz.name, "flags sha256": sha(npz)}
    with open(dest := HERE / "results" / f"identical_POSTHOC_{stamp}.json", "x") as fh:
        fh.write(json.dumps(rec, indent=1) + "\n")
    print(json.dumps({k: v for k, v in rec.items() if k != "inputs sha256"}, indent=1))
    print(dest)
