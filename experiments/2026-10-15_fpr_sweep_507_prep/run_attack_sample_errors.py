"""09 (via 2e, 10-16 ~00:5x): every attack-arm run's sample error BY KEY, metadata only, on Modal CPU, for the no-filter exclusion list.
Per .eval (one sample each): log status, the sample error class (the donor scan's frozen rule @ 4299f6350: the first ...Error/Exception/
Timeout/Exceeded name in the error message or traceback, else "unclassified"; None when there is no error), the sample limit type, and
the count of tool messages other than the runner's submit (the toolerr rule, 22bc87114 L51-68). No message content leaves the container.
Explicit allowlisted paths only (sha256 asserted), from pinned records:
  S_old: manifest_q4b_eval_upload_v1.json @ 0abe8f1af (run1, run2, sim);
  S_new: toolerr_su311_* @ 806b7a63a (run1, run2, sim; + honest_a/honest_b of t404, whose sim never launched);
  S_newmains: toolerr_nm360_* @ f57da7029 (run1, run2, sim).

  modal run --detach run_attack_sample_errors.py     # writes results/attack_sample_errors_<stamp>.json (exclusive create)
"""

import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).resolve().parent
APP = "probe-shift_aswin-surya_sub-fpr507prep_attackerr_kill-2026-11-14"
VOLS = {"chain": "probe-shift_aswin-surya_sub-q3su311_acts_kill-2026-11-14", "q4own": "probe-shift_aswin-surya_sub-q4own_evals_kill-2026-11-13"}
HYG = "experiments/2026-10-14_q3_overnight_chain/results/hygiene"
Q4B = ("0abe8f1af", "experiments/2026-10-14_q3_overnight_chain/config/manifest_q4b_eval_upload_v1.json")
TE = {"S_new": ("806b7a63a", "toolerr_su311_"), "S_newmains": ("f57da7029", "toolerr_nm360_")}
W3 = ("run1", "run2", "sim")
image = modal.Image.debian_slim(python_version="3.12").pip_install("inspect_ai==0.3.251", "zstandard==0.25.0")
app = modal.App(APP, image=image)
vols = {k: modal.Volume.from_name(v, version=2) for k, v in VOLS.items()}


def git(*a):
    return subprocess.run(["git", *a], capture_output=True, check=True, cwd=HERE).stdout


def allowlist():
    rows, pins = [], {}
    b = git("show", f"{Q4B[0]}:{Q4B[1]}")
    pins["S_old"] = {"commit": Q4B[0], "path": Q4B[1], "sha256": hashlib.sha256(b).hexdigest()}
    q = json.loads(b)
    sha = {f"{f['launch']}/{f['rel']}": f["sha256"] for f in q["files"]}
    for w in W3:
        for x in q["worlds"][w]:
            rows.append({"origin": "S_old", "key": f"{w}|{x['task']}", "source": Q4B[1].rsplit("/", 1)[1], "path": x["path"], "sha256": sha[x["path"]],
                         "volume_order": ["q4own", "chain"]})
    for o, (c, pre) in TE.items():
        names = sorted(n.rsplit("/", 1)[-1] for n in git("-C", str(HERE.parents[1]), "ls-tree", "--name-only", c, f"{HYG}/").decode().split()
                       if n.rsplit("/", 1)[-1].startswith(pre))
        keep = [n for n in names if n.removesuffix(".json").rsplit("_", 1)[1] in W3 or (o == "S_new" and n.endswith(("_honest_a.json", "_honest_b.json")))]
        pins[o] = {"commit": c, "files": {}}
        for n in keep:
            fb = git("show", f"{c}:{HYG}/{n}")
            pins[o]["files"][n] = hashlib.sha256(fb).hexdigest()
            d = json.loads(fb)
            if len(d["rows"]) != len(d["source_evals"]):
                raise SystemExit(f"{n}: rows and source_evals differ in length")
            for r, s in zip(d["rows"], d["source_evals"]):
                cell = r["key"].split("|", 1)[1]
                if not any(g == cell or g.startswith((f"{cell}_r2", f"{cell}_redraw")) for g in s["path"].split("/")):
                    raise SystemExit(f"{n}: {r['key']} paired with {s['path']}")
                if r["key"].startswith("honest") and not cell.startswith("t404_"):
                    continue
                rows.append({"origin": o, "key": r["key"], "source": n, "path": s["path"], "sha256": s["sha256"], "volume_order": ["chain"],
                             "toolerr_n_tool_calls": r["n_tool_calls"]})
    return rows, pins


@app.function(volumes={f"/{k}": v.read_only() for k, v in vols.items()}, cpu=2, memory=4096, timeout=1800, retries=0)
def scan(chunk: list) -> list:
    import os
    import re

    from inspect_ai.log import read_eval_log

    cls = re.compile(r"\b([A-Z][A-Za-z]*(?:Error|Exception|Timeout|Exceeded))\b")

    def klass(err):  # the donor scan's frozen rule (@ 4299f6350)
        if err is None:
            return None
        m = cls.search(err.message or "") or cls.search(err.traceback or "")
        return m.group(1) if m else "unclassified"

    out = []
    for r in chunk:
        rec = {k: r[k] for k in ("origin", "key", "source", "path")}
        hit = next((v for v in r["volume_order"] if os.path.isfile(f"/{v}/evals_v1/{r['path']}")), None)
        if hit is None:
            out.append(rec | {"missing": True})
            continue
        p = f"/{hit}/evals_v1/{r['path']}"
        if hashlib.sha256(open(p, "rb").read()).hexdigest() != r["sha256"]:
            out.append(rec | {"volume": hit, "sha_mismatch": True})
            continue
        log = read_eval_log(p)
        if len(log.samples or []) != 1:
            raise ValueError(f"{r['path']}: {len(log.samples or [])} samples")
        s = log.samples[0]
        sub = {tc.id for m in s.messages if m.role == "assistant" for tc in (m.tool_calls or []) if tc.function == "submit"}
        ntool = sum(1 for m in s.messages if m.role == "tool" and m.tool_call_id not in sub and getattr(m, "function", None) != "submit")
        out.append(rec | {"volume": hit, "sha_ok": True, "log_status": log.status, "sample_error": klass(s.error), "has_error": s.error is not None,
                          "limit": (s.limit.type if s.limit else None), "n_tool_nonsubmit": ntool, "n_messages": len(s.messages),
                          "toolerr_n_tool_calls": r.get("toolerr_n_tool_calls")})
    return out


@app.local_entrypoint()
def main():
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    if git("status", "--porcelain", "--untracked-files=no").strip():
        raise SystemExit("commit first: tracked files differ from HEAD")
    rows, pins = allowlist()
    print(json.dumps({"allowlist_rows": len(rows), "by_origin": {o: sum(r["origin"] == o for r in rows) for o in ("S_old", "S_new", "S_newmains")}}), flush=True)
    chunks = [rows[i:i + 40] for i in range(0, len(rows), 40)]
    res = [x for part in scan.map(chunks) for x in part]
    if len(res) != len(rows) or [x["path"] for x in res] != [r["path"] for r in rows]:
        raise SystemExit("reader output does not match the allowlist")
    bad = [x["key"] for x in res if x.get("sha_ok") and x["toolerr_n_tool_calls"] is not None and x["toolerr_n_tool_calls"] != x["n_tool_nonsubmit"]]
    meta = {"git": git("rev-parse", "HEAD").decode().strip(), "stamp_pt": stamp, "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "app": APP, "volumes": VOLS, "pins": pins, "rule": __doc__.split("\n\n")[0],
            "counts": {"rows": len(res), "missing": sum(bool(x.get("missing")) for x in res), "sha_mismatch": sum(bool(x.get("sha_mismatch")) for x in res),
                       "with_sample_error": sum(bool(x.get("has_error")) for x in res), "zero_tool": sum(x.get("n_tool_nonsubmit") == 0 for x in res),
                       "tool_count_differs_from_toolerr": bad}}
    blob = json.dumps({"meta": meta, "rows": res}, indent=1).encode()
    dest = HERE / "results" / f"attack_sample_errors_{stamp}.json"
    dest.open("xb").write(blob)
    print(json.dumps({"record": dest.name, "sha256": hashlib.sha256(blob).hexdigest(), "counts": meta["counts"]}), flush=True)
