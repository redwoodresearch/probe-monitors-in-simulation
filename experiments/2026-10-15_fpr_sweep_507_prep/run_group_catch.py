"""Group catch averages for the left panel of Figures 1 and 5 (2e via Aswin, 10-16): per group (the right panel's six and four-probe blocks) and
readout, the EQUAL-WEIGHT mean over the group's probes of the pooled_all_origins joint catch at 1 alarm per 100 honest actions, real (the mean of
the two runs) and simulated. The Stage A record stores no group catch, so this rebuilds each probe's pooled per-draw catch from the record's
per-draw npz exactly as run_peraction311 pools it (per origin catch x D, summed over the members in order, over the summed D), GATES each probe's
rebuilt summary against the record's own catch node (point, interval, draw counts) and its pooled Delta_full array, then averages the probes draw
by draw (np.mean over probes, as the record's co-headline S) and summarises with the record's summ (point = draw 0; 95% percentile interval over
the finite resampled draws, withheld below 95% complete). Modal CPU; nothing printed but keys, counts and check results.

  cd ~/probe-distribution-shift-from-sims && uv run modal run --detach <this file> --record-src <sol path or git:<commit>:<path>> --record-sha <sha256> [--npz-path <Volume path>] \
      --record-source <path@commit> --groups six=<p>,<p>,... auroc_subset=<p>,... --readouts C R
"""

import datetime
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).resolve().parent
app = modal.App("probe-shift_aswin-surya_sub-routec-grp_catch_kill-2026-11-14")  # route C (10-17): its own description, not the 10-16 renders
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy==2.3.3")
SOL = modal.Volume.from_name("probe-shift_aswin-surya_sub-q35solavg_kill-2026-11-13", version=2)  # Stage A's record and npz (read only)
OUT = modal.Volume.from_name("probe-shift_aswin-surya_sub-render_final-figs_kill-2026-11-13", create_if_missing=True)
LEAD, MODE, F = "pooled_all_origins", "joint", "0.01"
ARMS = {"real_mean": "real_mean@B", "sim": "sim@S|primary"}  # the figure's catch keys (estimator A) -> the per-origin draw keys
RESULTS = "experiments/2026-10-15_fpr_sweep_507_prep/results"  # the committed copies (the label sidecar's "record")


def rel_ok(p, prefix):  # route C (a37cc): a Volume path is relative, under prefix, with no ".." (Path("/sol", "/abs") escapes /sol)
    return isinstance(p, str) and not p.startswith("/") and p.startswith(prefix) and ".." not in Path(p).parts


def npzkey(*parts):  # run_peraction311.npzkey @ 20c6fbd20, verbatim rule
    out = []
    for p_ in parts:
        q = str(p_).replace("@", "_at_").replace("|", "_").replace(" ", "_").replace("(", "").replace(")", "")
        if "__" in q or not q:
            raise SystemExit(f"npz key part {p_!r} would be ambiguous")
        out.append(q)
    return "__".join(out)


def a_label(record_sha, side_bytes, side_sha, step):
    """09 (10-17): the record's label comes from A's pinned, committed LABEL SIDECAR (pooled A writes no meta.label), never typed:
    {"record": <A path>, "record_sha256": <A sha>, "label": <str>}. STOP unless (1) the sidecar's bytes hash to their pin (before they
    are parsed), (2) it has exactly those keys and names the A record this step consumes (record_sha256 == A's sha), and (3) its label
    is a non-empty string containing "PRELIMINARY" (no FINAL-release path exists yet). -> "<label> | <step> on route C pooled A @ <A sha>"."""
    if not isinstance(side_bytes, bytes) or hashlib.sha256(side_bytes).hexdigest() != side_sha:
        raise SystemExit("STOP: the label sidecar's sha256 differs from its pin")
    sc = json.loads(side_bytes)
    if not isinstance(sc, dict) or set(sc) != {"record", "record_sha256", "label"} or not isinstance(sc["record"], str) or not sc["record"]:
        raise SystemExit("STOP: the label sidecar is not {record, record_sha256, label}")
    if sc["record_sha256"] != record_sha:
        raise SystemExit("STOP: the label sidecar names another A record than the one this step consumes")
    lab = sc["label"]
    if not isinstance(lab, str) or not lab.strip():
        raise SystemExit("STOP: the label sidecar's label is missing or empty")
    if "PRELIMINARY" not in lab:
        raise SystemExit("STOP: the label is not PRELIMINARY and no FINAL-release record is pinned (09)")
    return f"{lab} | {step} on route C pooled A @ {record_sha}"


def label_sidecar(record_bytes, label, record_path):
    """09 via 2e (10-18; a3fd1612 O8): the record's own label sidecar {record, record_sha256, label}: the sha256 of the record's bytes and
    the record's composed label, byte for byte. -> bytes, written next to the record (exclusive create); the client commits a copy."""
    return json.dumps({"record": record_path, "record_sha256": hashlib.sha256(record_bytes).hexdigest(), "label": label}, indent=1,
                      ensure_ascii=False).encode()


def sidecar_bytes(pointer):
    """Laptop side: git:<commit>:<path>@<sha256> -> (bytes, sha256), the blob read unparsed (the container checks and parses it)."""
    spec, _, h = pointer.rpartition("@")
    src, _, rest = spec.partition(":")
    c_, _, p_ = rest.partition(":")
    if src != "git" or len(c_) != 40 or not p_.endswith(".json") or len(h) != 64:
        sys.exit("STOP: --label-sidecar must be git:<40-hex commit>:<path>.json@<sha256>")
    return subprocess.run(["git", "show", f"{c_}:{p_}"], capture_output=True, check=True, cwd=HERE).stdout, h


def load_record(record_src, record_sha, record_bytes, npz_path, read):
    """The record (a Volume path, or a git-held record's bytes) and its npz, each sha-checked BEFORE use (the record before it is parsed);
    read(relative path) -> bytes. -> (record, npz bytes, npz sha256). Fail closed."""
    if not record_bytes and not rel_ok(record_src, "xfer/fpr507prep/records/"):
        raise SystemExit("STOP: --record-src is not a relative path under xfer/fpr507prep/records/")
    rb = record_bytes or read(record_src)  # a git-held record (e.g. the pooled stub) arrives as bytes
    if hashlib.sha256(rb).hexdigest() != record_sha:
        raise SystemExit("STOP: the Stage A record's sha256 differs")
    rec = json.loads(rb)
    npz_sha = rec["npz"]["sha256"]  # the record names its own npz (per-set or pooled world); the file must hash to it
    if not rel_ok(npz_path or rec["npz"]["path"], "xfer/fpr507prep/"):
        raise SystemExit("STOP: the npz path is not a relative path under xfer/fpr507prep/")
    nb = read(npz_path or rec["npz"]["path"])  # --npz-path: the file elsewhere on the Volume; it must still hash to npz.sha256
    if hashlib.sha256(nb).hexdigest() != npz_sha:
        raise SystemExit("STOP: the npz's sha256 differs from the record's npz.sha256")
    return rec, nb, npz_sha


@app.function(image=image, cpu=2, memory=8192, timeout=1200, retries=0, volumes={"/sol": SOL, "/out": OUT})
def build(record_src: str, record_sha: str, groups: dict, readouts: list, stamp: str, record_bytes: bytes = b"", npz_path: str = "",
          launch: dict | None = None, label_bytes: bytes = b"", label_sha: str = "") -> dict:
    import numpy as np

    def summ(v, D):  # run_fprsweep_507.summ @ 20c6fbd20 (D = the record's draw count)
        f = v[1:][np.isfinite(v[1:])]
        o = {"point": float(v[0]), "finite_draws": int(len(f)), "draws": D}
        if len(f) < 0.95 * D:
            return o | {"ci95": None, "withheld": "fewer than 95% complete draws"}
        return o | {"ci95": [float(np.percentile(f, 2.5)), float(np.percentile(f, 97.5))], "degenerate": bool(np.ptp(f) == 0)}

    rec, nb, NPZ_SHA = load_record(record_src, record_sha, record_bytes, npz_path, lambda p_: Path("/sol", p_).read_bytes())
    lab_ = a_label(record_sha, label_bytes, label_sha, "run_group_catch")  # 09: the sidecar checks, before any compute
    import io

    Z = np.load(io.BytesIO(nb))
    gate, rows, out = {}, {}, {}
    for ro in readouts:
        for p in sorted({p for ps in groups.values() for p in ps}):
            lab = f"{p} {ro}"
            node = rec["pooled"][lab][LEAD]
            mem = node["members"]
            cat = node[MODE][F]["catch"]
            D = cat["sim"]["draws"]
            per = {}
            for o in mem:
                on = o.split(" [pooled: ", 1)[0]  # a pooled-world member's npz part drops its ' [pooled: <combo>]' suffix (run_peraction311 @ 8a15fb10f)
                ks = {a: npzkey(lab, LEAD, MODE, F, on, "catch_" + k) for a, k in ARMS.items()} | {"D": npzkey(lab, LEAD, MODE, F, on, "D")}
                miss = [k for k in ks.values() if k not in Z.files]
                if miss:
                    pre = npzkey(lab, LEAD, MODE, F, on)
                    raise SystemExit(f"STOP: {lab} {o}: npz keys absent {miss}; present with that prefix: {sorted(k for k in Z.files if k.startswith(pre))[:20]}")
                per[o] = {a: np.asarray(Z[k], dtype=np.float64) for a, k in ks.items()}
            Dd = sum(per[o]["D"] for o in mem)  # run_peraction311's all-origins combination, same order and operations
            arr = {a: sum(np.where(per[o]["D"] > 0, per[o][a] * per[o]["D"], 0.0) for o in mem) / Dd for a in ARMS}
            g = {}
            for a in ARMS:  # GATE: the rebuilt summary equals the record's catch node exactly
                s_, r_ = summ(arr[a], D), cat[a]
                g[a] = all(s_.get(k) == r_.get(k) for k in ("point", "ci95", "draws", "finite_draws", "withheld"))
            dfa = np.asarray(Z[npzkey(lab, LEAD, MODE, F, "pooled", "gap_delta_full")], dtype=np.float64)
            dd_ = arr["sim"] - arr["real_mean"]
            g["delta_full_array"] = bool(np.array_equal(dd_, dfa, equal_nan=True))
            g["delta_full_max_abs_diff"] = float(np.nanmax(np.abs(dd_ - dfa))) if np.isfinite(dd_ - dfa).any() else 0.0
            gate[lab] = g
            rows[lab] = arr
        for gname, ps in groups.items():
            labs = [f"{p} {ro}" for p in ps]
            node = {"members": labs, "n": len(labs), "level": F, "mode": MODE, "combo": LEAD, "catch": {}}
            for a in ARMS:
                m_ = np.mean(np.stack([rows[lb][a] for lb in labs]), 0)  # equal weight, draw by draw (NaN in any probe -> NaN, dropped and counted)
                s_ = summ(m_, rec["pooled"][labs[0]][LEAD][MODE][F]["catch"][a]["draws"])
                pts = [rec["pooled"][lb][LEAD][MODE][F]["catch"][a]["point"] for lb in labs]
                s_["point_equals_mean_of_recorded_points"] = bool(abs(s_["point"] - float(np.mean(pts))) <= 1e-12)
                node["catch"][a] = s_
            out[f"{gname}|{ro}"] = node
    ok = all(v_ is True for g in gate.values() for k, v_ in g.items() if k != "delta_full_max_abs_diff") and all(
        n_["catch"][a]["point_equals_mean_of_recorded_points"] for n_ in out.values() for a in ARMS)
    doc = {"label": lab_,  # 09: "<sidecar label> | <step> on route C pooled A @ <A sha>"
           "what": "group catch averages (equal-weight mean over the group's probes; left panel of Figures 1 and 5)", "groups": out,
           "gate": {"pass": ok, "per_label": gate, "rule": "each probe's rebuilt pooled catch summary == the record's catch node (point, ci95, draws, "
                    "finite_draws, withheld) and its rebuilt sim - real_mean == the npz's pooled gap_delta_full array"},
           "meta": {"record": {"src": record_src, "sha256": record_sha}, "npz": {"path": npz_path or rec["npz"]["path"], "record_npz_path": rec["npz"]["path"], "sha256": NPZ_SHA},
                    "arms": ARMS, "level": F, "mode": MODE, "combo": LEAD, "readouts": readouts, "groups_requested": groups,
                    "method": "per draw: each probe's pooled catch rebuilt from the per-origin npz arrays (sum over members of where(D>0, catch*D, 0) "
                              "/ sum D, run_peraction311 @ 20c6fbd20), np.mean over the group's probes, then summ (point = draw 0; 2.5/97.5 "
                              "percentiles of the finite resampled draws; withheld below 95% complete)",
                    "created_pt": datetime.datetime.now(ZoneInfo("America/Los_Angeles")).isoformat(timespec="seconds"), "stamp": stamp}
                   | (launch or {})}  # route C (10-17): the launch fields go in here, so the laptop never parses the record
    b = json.dumps(doc, indent=1).encode()
    p_ = Path("/out/group_catch") / f"group_catch{stamp}.json"
    sb_ = label_sidecar(b, doc["label"], f"{RESULTS}/group_catch{stamp}.json") if ok else None  # 09 (2e 10-18, a3fd1612 O8); a37cc O1: none if failed
    sp_ = p_.with_name(f"group_catch{stamp}_LABEL.json")
    p_.parent.mkdir(parents=True, exist_ok=True)
    with open(p_, "xb") as f:
        f.write(b)
    if sb_ is not None:  # a37cc O1 (10-18): a failed record gets no sidecar, as the anchor, so it is never picked up labelled
        with open(sp_, "xb") as f:
            f.write(sb_)
    OUT.commit()
    return {"bytes": b, "sha256": hashlib.sha256(b).hexdigest(), "volume_path": str(p_.relative_to("/out")), "pass": ok,
            "label_sidecar": {"bytes": sb_, "sha256": hashlib.sha256(sb_).hexdigest(), "volume_path": str(sp_.relative_to("/out"))} if sb_ is not None else None,
            "fails": {lb: [k for k, v_ in g.items() if v_ is False] for lb, g in gate.items() if any(v_ is False for v_ in g.values())},
            "groups": {k: v["members"] for k, v in out.items()}}  # keys only


def git(*a):
    return subprocess.run(["git", *a], capture_output=True, check=True, cwd=HERE).stdout.decode().strip()


@app.local_entrypoint()
def main(record_src: str, record_sha: str, record_source: str, label_sidecar: str, groups: str, readouts: str = "C R", npz_path: str = ""):
    if git("status", "--porcelain", "--", Path(__file__).name):
        sys.exit("commit this script before running it")
    if not record_src.startswith("git:") and not rel_ok(record_src, "xfer/fpr507prep/records/"):
        sys.exit("STOP: --record-src must be git:<commit>:<path> or a relative path under xfer/fpr507prep/records/ (no leading slash, no ..)")
    if npz_path and not rel_ok(npz_path, "xfer/fpr507prep/"):
        sys.exit("STOP: --npz-path must be a relative path under xfer/fpr507prep/")
    gd = {g.split("=", 1)[0]: [p for p in g.split("=", 1)[1].split(",") if p] for g in groups.split()}
    if not gd or not readouts.split() or any(not ps for ps in gd.values()):  # never a vacuous gate (Codex r92)
        sys.exit("STOP: give at least one readout and one non-empty group")
    stamp = "_" + datetime.datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    rb = b""
    if record_src.startswith("git:"):  # git:<commit>:<path>: read here, sent as bytes (the container re-checks the sha256)
        _, c_, p_ = record_src.split(":", 2)
        rb = subprocess.run(["git", "show", f"{c_}:{p_}"], capture_output=True, check=True, cwd=HERE).stdout
    lb_, lh_ = sidecar_bytes(label_sidecar)  # 09: the pinned label sidecar, as bytes (checked and parsed in the container)
    launch = {"record_source": record_source, "label_sidecar": label_sidecar, "git": git("rev-parse", "HEAD"),
              "script_sha256": hashlib.sha256((HERE / Path(__file__).name).read_bytes()).hexdigest()}
    res = build.remote(record_src, record_sha, gd, readouts.split(), stamp, rb, npz_path, launch, lb_, lh_)
    b = res["bytes"]  # written as returned: byte-identical to the Volume copy, never parsed here
    if hashlib.sha256(b).hexdigest() != res["sha256"]:
        sys.exit("STOP: the returned bytes differ from the container's sha256")
    out = HERE / "results" / f"group_catch{stamp}.json"
    with open(out, "xb") as f:  # exclusive create; results/ is append-only
        f.write(b)
    sout = None
    if res["label_sidecar"] is not None:  # the record's own label sidecar, written as returned (never parsed here); none for a failed record
        sb = res["label_sidecar"]["bytes"]
        if hashlib.sha256(sb).hexdigest() != res["label_sidecar"]["sha256"]:
            sys.exit("STOP: the returned label sidecar differs from the container's sha256")
        sout = HERE / "results" / f"group_catch{stamp}_LABEL.json"
        with open(sout, "xb") as f:
            f.write(sb)
    print(json.dumps({"out": str(out), "sha256": res["sha256"], "volume_path": res["volume_path"], "gate_pass": res["pass"],
                      "gate_fails": res["fails"], "groups": res["groups"],
                      "label_sidecar": {"out": str(sout), "volume_path": res["label_sidecar"]["volume_path"], "sha256": res["label_sidecar"]["sha256"]}
                      if res["label_sidecar"] is not None else None}))
    if res["pass"] and res["label_sidecar"] is None:
        sys.exit("STOP: the record passed but came back without its label sidecar")
    if not res["pass"]:
        sys.exit("STOP: the gate failed; the file is written for diagnosis only, never for a figure")
