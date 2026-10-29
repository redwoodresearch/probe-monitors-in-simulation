"""d9's CUT fork (Append 41, cutfull_routing_v2.json): the extraction side of a cutfull root's attestation, EVERY epoch, on Modal CPU
(never the laptop). Keys, counts and positions only. The staging side (prefix identity and full length against the REGISTERED
staging's tokens_before) is cutfull/prefix_check.py; the routing records cite both records.

Per root (extraction root R of cutfull staged set S, arms A = every arm file of S, repro-c aside):
1. R/config.json: inputs == S, cut "full", inputs_sha256 == the sha256 of each staged arm file, and no other arm.
2. R/complete.json present, its epochs == the staged epochs and its inputs_sha256 == config.json's.
3. per staged epoch (arm, key): its meta exists (meta/<arm>/<sha1(key)[:20]>.json), names the key and arm and the staged file's sha256;
   its calls ARE the staged calls (minus sha256/row/row_mean) with rows allocated as extract_shard does (a row per non-null readout,
   a mean row per output_end with a span, span end - 1 == readout); its chunk index lists it once at its row_start / n_rows;
   n_tokens == tokens_total == len(staged ids); cut "full"; truncated false; padded_tokens == n + (-n % pad_to); every unit (a call
   with a non-null row or row_mean) has an integer readout 0 <= readout < n; max_position is null iff there is no unit, else the
   max unit readout.
Failures go to `failing` {reason: [[arm token, key], ...]} ({} when none); the command exits nonzero after writing the record.
The fan-out runs in attest_driver on Modal (run with --detach); its record also goes to the q3cutfull Volume under attest/.

  CUTATTEST_SPEC=<path of a json file {"roots": [{"root", "uncut_set"}]}> uv run --no-project --python 3.12 --with modal==1.6.1 \
      modal run --detach experiments/2026-10-14_q3_overnight_chain/cutfull/attest_metas.py
"""

import hashlib
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).resolve().parent
CHAIN = HERE.parent
REPO = CHAIN.parents[1] if len(CHAIN.parents) > 1 else CHAIN  # import-safe inside the Modal container (/root)
APP = "probe-shift_aswin-surya_sub-q3cutfull_attest_kill-2026-11-14"
UNC_VOL = "probe-shift_aswin-surya_sub-q3cutfull_acts_kill-2026-11-14"
app = modal.App(APP)
image = modal.Image.debian_slim(python_version="3.12")


def token(arm: str) -> str:
    return arm.split("@")[0]


@app.function(image=image, cpu=4, memory=16384, timeout=3600, retries=0, volumes={"/unc": modal.Volume.from_name(UNC_VOL, version=2).read_only()})
def check_root(row: dict) -> dict:
    S, R = Path("/unc") / row["uncut_set"], Path("/unc") / row["root"]
    fail, counts = {}, {}

    def bad(reason, arm, key):
        fail.setdefault(reason, []).append([token(arm), key])

    files = {f.stem: f for f in sorted((S / "inputs").glob("*.jsonl")) if f.stem != "repro-c"}
    cfg = json.loads((R / "config.json").read_text())
    shas = {arm: hashlib.sha256(f.read_bytes()).hexdigest() for arm, f in files.items()}
    if not files or cfg.get("inputs") != row["uncut_set"] or cfg.get("cut") != "full" or cfg.get("inputs_sha256") != {f"{a}.jsonl": s for a, s in shas.items()}:
        bad("config", "", row["root"])
    pad = cfg.get("pad_to")
    total, chunks = 0, {}
    if len({token(a) for a in files}) != len(files):
        bad("arm tokens not unique", "", row["root"])
    for arm, f in files.items():
        n_units = n_past = 0
        eps = [json.loads(line) for line in f.read_text().splitlines()]
        total += len(eps)
        for e in eps:
            key = e["key"]
            mp = R / "meta" / arm / f"{hashlib.sha1(key.encode()).hexdigest()[:20]}.json"
            if not mp.exists():
                bad("meta missing", arm, key)
                continue
            m = json.loads(mp.read_text())
            n = len(e["ids"])
            if m.get("key") != key or m.get("arm") != arm or m.get("inputs_sha256") != shas[arm]:
                bad("meta binding", arm, key)
                continue
            if not (m.get("n_tokens") == m.get("tokens_total") == n and m.get("cut") == "full" and m.get("truncated") is False
                    and isinstance(pad, int) and pad > 0 and m.get("padded_tokens") == n + (-n % pad)):
                bad("full consumption", arm, key)
            st = e["calls"]  # the extracted calls are the staged calls, rows allocated as extract_shard does (step0_compare.bind)
            strip = lambda c: {k: v for k, v in c.items() if k not in ("sha256", "row", "row_mean")}  # noqa: E731
            kept = [i for i, c in enumerate(st) if c["readout"] is not None]
            means = [i for i in kept if st[i]["kind"] == "output_end" and st[i].get("span")]
            alloc, mean = {i: j for j, i in enumerate(kept)}, {i: len(kept) + j for j, i in enumerate(means)}  # never `row`: the input
            if ([strip(c) for c in m["calls"]] != [strip(c) for c in st] or m.get("n_rows") != len(kept) + len(means)
                    or any(c.get("row") != alloc.get(i) or c.get("row_mean") != mean.get(i) for i, c in enumerate(m["calls"]))):
                bad("calls/rows", arm, key)
                continue
            if any(st[i]["span"][1] - 1 != st[i]["readout"] or not 0 <= st[i]["span"][0] < st[i]["span"][1] for i in means):
                bad("span", arm, key)
            cp = R / "chunks" / f"{m.get('chunk')}.json"
            if m.get("chunk") not in chunks:
                chunks[m.get("chunk")] = json.loads(cp.read_text()) if cp.exists() else None
            ch = chunks[m.get("chunk")]
            if ch is None or [(x["row_start"], x["n_rows"]) for x in ch["epochs"] if x["key"] == key and x["arm"] == arm] != [(m.get("row_start"), m.get("n_rows"))]:
                bad("chunk index", arm, key)
            ro = [st[i]["readout"] for i in kept]
            if any(not isinstance(r, int) or isinstance(r, bool) or not 0 <= r < n for r in ro):
                bad("unit position", arm, key)
            elif m.get("max_position") != (max(ro) if ro else None):
                bad("max_position", arm, key)
            n_units += len(ro)
            n_past += sum(r >= 32768 for r in ro if isinstance(r, int))
        counts[token(arm)] = {"meta_arm": arm, "epochs": len(eps), "units": n_units, "units_past_32768": n_past, "staged_sha256": shas[arm],
                              "keys": sorted(e["key"] for e in eps)}
    comp = R / "complete.json"
    if not comp.exists() or json.loads(comp.read_text()).get("epochs") != total or json.loads(comp.read_text()).get("inputs_sha256") != cfg.get("inputs_sha256"):
        bad("complete.json", "", row["root"])
    recs = sorted(S.glob("stage_2*.json"))  # the cutfull stage record: its template pins (fixed_staged, original), bound to these files
    sr = json.loads(recs[0].read_text()) if len(recs) == 1 else {}
    tmpl = sr.get("chat_template_sha256")
    if not (isinstance(tmpl, dict) and tmpl.get("fixed_staged") and tmpl.get("original") and sr.get("cut") == "full"
            and sr.get("staged_inputs_sha256") == {f"{a}.jsonl": x for a, x in shas.items()}
            and sr.get("epochs") == {a: v["epochs"] for a, v in ((c["meta_arm"], c) for c in counts.values())}):
        bad("stage record template/cut/binding", "", row["root"])
    return {**row, "arms": counts, "epochs": total, "chat_template_sha256": tmpl, "stage_record": recs[0].name if len(recs) == 1 else None, "failing": fail}


@app.function(image=image, cpu=1, memory=2048, timeout=2 * 3600, retries=0)
def attest_driver(roots: list, meta: dict, stamp: str) -> dict:
    import io

    vol = modal.Volume.from_name(UNC_VOL, version=2)
    name = f"cutfull_attest_metas_{stamp}"
    with vol.batch_upload(force=True) as b:
        b.put_file(io.BytesIO(json.dumps({"attest_driver_call_id": modal.current_function_call_id(), "stamp": stamp}).encode()), f"attest/{name}.started.json")
    rows = list(check_root.map(roots, return_exceptions=True))
    res = [r if isinstance(r, dict) else {**x, "failing": {"error": [["", repr(r)[:300]]]}} for r, x in zip(rows, roots)]
    out = {"what": "cutfull extraction-side attestation (cutfull_routing_v2.json), every epoch", **meta, "roots": res, "n_roots": len(res),
           "n_ok": sum(not r["failing"] for r in res), "passes": bool(res) and all(not r["failing"] for r in res)}
    with vol.batch_upload(force=True) as b:
        b.put_file(io.BytesIO(json.dumps(out, indent=1).encode()), f"attest/{name}.json")
    return out


@app.local_entrypoint()
def main():
    spec_path = Path(os.environ["CUTATTEST_SPEC"]).resolve()
    raw = spec_path.read_bytes()
    roots = json.loads(raw)["roots"]
    if not roots or len({r["root"] for r in roots}) != len(roots):
        raise SystemExit("empty or duplicate roots")
    git = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True, check=True, cwd=REPO).stdout.strip()  # noqa: E731
    if git("status", "--porcelain", "--untracked-files=no"):
        raise SystemExit("refusing to run from a dirty tree")
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    meta = {"git": git("rev-parse", "HEAD"), "spec": str(spec_path.relative_to(REPO)), "spec_sha256": hashlib.sha256(raw).hexdigest()}
    call = attest_driver.spawn(roots, meta, stamp)
    print(json.dumps({"attest_driver_call_id": call.object_id, "stamp": stamp}), flush=True)
    out = call.get()
    dest = CHAIN / "results" / f"cutfull_attest_metas_{stamp}.json"
    dest.open("x").write(json.dumps(out, indent=1))
    print(dest.name, "PASS" if out["passes"] else "FAIL", f"{out['n_ok']}/{out['n_roots']}")
    if not out["passes"]:
        raise SystemExit(f"attestation FAILED: {[(r['root'], list(r['failing'])) for r in out['roots'] if r['failing']]}")
