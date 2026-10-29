"""Four-probe EXCESS interval on the pooled-world Stage A record A (Aswin's GO via writing / 0d / 2e, 10-16 ~23:5x: "compute interval with
poooling"). The record stores the six-probe excess (panel_groups[...]['headline_six_full|C'].primary.mean_G_minus_mean_Y) but no four-probe
(AUROC >= 0.6 subset) group for it. This rebuilds it exactly as run_peraction311 @ 704122edf builds a pooled PANEL group (l.3176-3177, 3273-3275):
per probe and draw, G = 0.5|sim - run1| + 0.5|sim - run2| and Y = |run2@B - run1@B| (pooled: y1 is run1@B, Y1_DEF_PW), read from the record's
per-draw npz pooled gap arrays (gap_sim_minus_run1, gap_sim_minus_run2, gap_run2_minus_run1_at_tauB); Gm, Ym = np.mean over the group's probes
draw by draw; excess = summ(Gm - Ym) with the record's summ (run_fprsweep_507.summ: point = draw 0; 2.5 / 97.5 np.percentile of the finite
resampled draws; withheld below 95% complete). Same draws as the record (the npz holds them), pooled_all_origins, joint, 0.01, readout C.
Gates (any failure STOPs): (1) record and npz sha256; (2) the four = meta.auroc06_subset.members (the record's own membership), each readable;
(3) per probe, the rebuilt summ(G), summ(Y), summ(G - Y) equal the record's panel.pooled node exactly; (4) POSITIVE CONTROL: the same code on the
record's six-probe group (its own rows, in order) reproduces mean_G, mean_Y and mean_G_minus_mean_Y (point and ci95 within 1e-12, counts equal).
Modal CPU. 2e's ruling (10-16 23:58): the four-probe point / ci95 (also at %.17g) go ONLY into the exclusive-create results file (the
Volume copy under xfer/fpr507prep/group_excess/ and its byte-identical committed copy in results/, path@sha; 0d reads it); stdout / exceptions carry keys, gate verdicts, the control's MATCH / MISMATCH and max abs diff (%.3e), path@sha only.
On ANY gate failure the file carries counts and "withheld: gate failed", never a four-probe value (results/ is append-only), and no rebuilt
six-probe summaries are stored at all (the control keeps verdicts and diffs only). The ONE remote function writes the record to the Volume
(exclusive create, commit()) before returning, so a --detach launch whose client disconnects still leaves it (2e: --detach, not foreground);
the client writes the same bytes to results/ when it is connected, else `modal volume get` fetches them.

  cd <worktree root> && uv run modal run --detach experiments/2026-10-15_fpr_sweep_507_prep/run_group_excess.py \
      --record-src <the UNCOMPRESSED .json path on the solavg Volume> --record-sha <its sha256> --npz-sha <sha256> \
      --record-source <path@commit> --pool-label "<label, e.g. 171-pool; superseded when the fix lands>"
"""

import datetime
import hashlib
import json
import subprocess
import sys
import uuid
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).resolve().parent
app = modal.App("probe-shift_aswin-surya_sub-routec-grp_excess_kill-2026-11-14")  # route C (10-17): <= 63 characters
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy==2.3.3")
SOL = modal.Volume.from_name("probe-shift_aswin-surya_sub-q35solavg_kill-2026-11-13", version=2)  # Stage A's record and npz (read); this job's
# output under xfer/fpr507prep/group_excess/ (exclusive create; the only write)
LEAD, MODE, F, RO, ST = "pooled_all_origins", "joint", "0.01", "C", "pooled"
SIX = "headline_six_full|C"
FOUR_EXPECTED = {"own_all_rs_uncut_L48", "sol_a_uncut", "FIBS|uncut", "harmfulness|uncut"}  # 2e's list; must equal meta.auroc06_subset.members
TOL = 1e-12
VOL_OUT = "xfer/fpr507prep/group_excess"  # on SOL; exclusive create
RESULTS = "experiments/2026-10-15_fpr_sweep_507_prep/results"  # the committed copies (the label sidecar's "record")


def rel_ok(p, prefix):  # route C (a37cc): a Volume path is relative, under prefix, with no ".." (Path("/sol", "/abs") escapes /sol)
    return isinstance(p, str) and not p.startswith("/") and p.startswith(prefix) and ".." not in Path(p).parts


def npzkey(*parts):  # run_peraction311.npzkey @ 20c6fbd20, verbatim rule (as run_group_catch.py)
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


def load_record(record_src, record_sha, npz_sha, read):
    """The record and its npz, each sha-checked BEFORE use (the record before it is parsed); read(relative path) -> bytes. Fail closed."""
    if not rel_ok(record_src, "xfer/fpr507prep/records/"):
        raise SystemExit("STOP: --record-src is not a relative path under xfer/fpr507prep/records/")
    rb = read(record_src)
    if hashlib.sha256(rb).hexdigest() != record_sha:
        raise SystemExit("STOP: the Stage A record's sha256 differs")
    rec = json.loads(rb)
    if rec["npz"]["sha256"] != npz_sha:
        raise SystemExit("STOP: the record's npz.sha256 is not the expected npz")
    if not rel_ok(rec["npz"]["path"], "xfer/fpr507prep/"):
        raise SystemExit("STOP: the record's npz.path is not a relative path under xfer/fpr507prep/")
    nb = read(rec["npz"]["path"])
    if hashlib.sha256(nb).hexdigest() != npz_sha:
        raise SystemExit("STOP: the npz's sha256 differs from the record's npz.sha256")
    return rec, nb


@app.function(image=image, cpu=2, memory=8192, timeout=1200, retries=0, volumes={"/sol": SOL})
def build(record_src: str, record_sha: str, npz_sha: str, stamp: str, launch: dict, label_bytes: bytes = b"", label_sha: str = "") -> dict:
    import io

    import numpy as np

    def summ(v, D):  # run_fprsweep_507.summ @ 704122edf (D = the record's draw count)
        f = v[1:][np.isfinite(v[1:])]
        o = {"point": float(v[0]), "finite_draws": int(len(f)), "draws": D}
        if len(f) < 0.95 * D:
            return o | {"ci95": None, "withheld": "fewer than 95% complete draws"}
        return o | {"ci95": [float(np.percentile(f, 2.5)), float(np.percentile(f, 97.5))], "degenerate": bool(np.ptp(f) == 0)}

    rec, nb = load_record(record_src, record_sha, npz_sha, lambda p_: Path("/sol", p_).read_bytes())
    lab_ = a_label(record_sha, label_bytes, label_sha, "run_group_excess")  # 09: the sidecar checks, before any compute
    Z = np.load(io.BytesIO(nb))
    esc_ = rec["npz"].get("label_escapes") or {}  # 507-prep N2: npzkey here is the plain rule; an escaped label would need the escaped key

    four_meta = rec["meta"]["auroc06_subset"]["members"]
    if set(four_meta) != FOUR_EXPECTED or len(four_meta) != 4:
        raise SystemExit(f"STOP: meta.auroc06_subset.members {four_meta} is not the expected four {sorted(FOUR_EXPECTED)}")
    six = rec["panel_groups"][LEAD][ST][MODE][F][SIX]
    six_p = six["primary"]
    rows6 = six_p["rows"]
    if len(rows6) != 6 or six.get("unreadable_rows") or any(not lb.endswith(" " + RO) for lb in rows6):
        raise SystemExit(f"STOP: the record's {SIX} primary group is not six readable {RO} rows")
    rows4 = [lb for lb in rows6 if lb.rsplit(" ", 1)[0] in set(four_meta)]  # the producer's member order (panel_rows insertion), filtered
    if {lb.rsplit(" ", 1)[0] for lb in rows4} != set(four_meta) or len(rows4) != 4:
        raise SystemExit(f"STOP: the four {sorted(four_meta)} are not all among the six rows {rows6}")
    if any(lb in esc_ for lb in rows6):
        raise SystemExit("STOP: a six-probe label has an npz label escape; the plain npzkey rule does not apply")

    KEYS = ("point", "ci95", "draws", "finite_draws", "withheld")
    G, Y, gate = {}, {}, {}
    for lb in rows6:
        node = rec["pooled"][lb][LEAD][MODE][F]
        D = node["gap"]["delta_full"]["draws"]
        ks = {k: npzkey(lb, LEAD, MODE, F, "pooled", "gap_" + k) for k in ("sim_minus_run1", "sim_minus_run2", "run2_minus_run1_at_tauB")}
        miss = [k for k in ks.values() if k not in Z.files]
        if miss:
            raise SystemExit(f"STOP: {lb}: npz keys absent {miss}")
        a = {k: np.asarray(Z[v], dtype=np.float64) for k, v in ks.items()}
        G[lb] = 0.5 * np.abs(a["sim_minus_run1"]) + 0.5 * np.abs(a["sim_minus_run2"])  # run_peraction311 l.3176 (sim - r1, sim - r2)
        Y[lb] = np.abs(a["run2_minus_run1_at_tauB"])  # l.3177 pooled: |run2@B - y1|, y1 = run1@B (Y1_DEF_PW)
        pn = node["panel"]["pooled"]
        g = {f"{nm}_exact": all(summ(v, D).get(k) == pn[nm].get(k) for k in KEYS) for nm, v in (("G", G[lb]), ("Y", Y[lb]), ("G_minus_Y", G[lb] - Y[lb]))}
        g["readable_all"] = node["readable"]["all"] is True
        gate[lb] = g

    def group(labs):
        D = rec["pooled"][labs[0]][LEAD][MODE][F]["gap"]["delta_full"]["draws"]
        Gm = np.mean([G[lb] for lb in labs], 0)  # the producer's np.mean over the use list (l.3273-3274)
        Ym = np.mean([Y[lb] for lb in labs], 0)
        return {"rows": labs, "n": len(labs), "mean_G": summ(Gm, D), "mean_Y": summ(Ym, D), "mean_G_minus_mean_Y": summ(Gm - Ym, D)}

    def maxdiff(mine, ref):  # max abs diff over point and both ci95 ends; inf if counts or withheld differ or either ci95 is absent (never a value)
        if any(mine.get(k) != ref.get(k) for k in ("draws", "finite_draws", "withheld")) or mine.get("ci95") is None or ref.get("ci95") is None:
            return float("inf")  # 507-prep N1: the control requires an interval on both sides
        if not all(isinstance(c_, list) and len(c_) == 2 for c_ in (mine["ci95"], ref["ci95"])):  # Astra r1/r2: a two-endpoint list each
            return float("inf")
        xs_ = [(mine["point"], ref["point"])] + list(zip(mine["ci95"], ref["ci95"]))
        if not all(np.isfinite(x) and np.isfinite(y) for x, y in xs_):  # a NaN point or end never passes (max() would skip a later NaN)
            return float("inf")
        return max(abs(x - y) for x, y in xs_)

    ctl = group(rows6)
    control = {}
    for nm in ("mean_G", "mean_Y", "mean_G_minus_mean_Y"):
        md_ = maxdiff(ctl[nm], six_p[nm])
        control[nm] = {"verdict": "MATCH" if md_ <= TOL else "MISMATCH", "max_abs_diff": "%.3e" % md_,
                       "exact": all(ctl[nm].get(k) == six_p[nm].get(k) for k in KEYS)}
    control_ok = all(c["verdict"] == "MATCH" for c in control.values())  # route C (10-17): only this boolean leaves the container
    four = group(rows4)
    ok = (all(v_ is True for g in gate.values() for v_ in g.values()) and control_ok
          and four["mean_G_minus_mean_Y"].get("ci95") is not None)
    ex = four["mean_G_minus_mean_Y"]
    if ok:
        four_out = four
        ex17 = {"point": "%.17g" % ex["point"], "ci95": ["%.17g" % x for x in ex["ci95"]]}
    else:  # 507-prep R1 (2e): a failed-gate file never carries the value
        four_out, ex17 = {"rows": rows4, "n": len(rows4), "withheld": "gate failed"}, None
    doc = {"label": lab_,  # 09: "<sidecar label> | <step> on route C pooled A @ <A sha>"
           "what": "four-probe (AUROC >= 0.6 subset) excess = mean_G_minus_mean_Y on the pooled-world Stage A record A",
           "four_probe": four_out, "four_probe_excess_17g": ex17,
           "positive_control_six": {"rows": rows6, "matches_record": control, "tolerance": TOL, "control_ok": control_ok},  # the record STORES max_abs_diff
           "gate": {"pass": ok, "per_label": gate},
           "meta": {"record": {"src": record_src, "sha256": record_sha}, "npz": {"path": rec["npz"]["path"], "sha256": npz_sha},
                    "combo": LEAD, "mode": MODE, "level": F, "readout": RO, "stream": ST, "tag": "primary",
                    "four_membership": {"from": "meta.auroc06_subset.members", "members": four_meta, "rows_in_order": rows4},
                    "method": "per probe and draw G = 0.5|gap_sim_minus_run1| + 0.5|gap_sim_minus_run2|, Y = |gap_run2_minus_run1_at_tauB| (npz, pooled); "
                              "Gm, Ym = np.mean over the group draw by draw; summ (point = draw 0; 2.5/97.5 np.percentile of finite resampled draws; "
                              "withheld below 95% complete); run_peraction311 @ 704122edf l.3176-3177, 3273-3275; run_fprsweep_507.summ",
                    "created_pt": datetime.datetime.now(ZoneInfo("America/Los_Angeles")).isoformat(timespec="seconds"), "stamp": stamp} | launch}
    b = json.dumps(doc, indent=1).encode()
    vp_ = Path("/sol", VOL_OUT, f"group_excess{stamp}.json")
    sb_ = label_sidecar(b, doc["label"], f"{RESULTS}/group_excess{stamp}.json") if ok else None  # 09 (2e 10-18, a3fd1612 O8); a37cc O1: none if failed
    sp_ = vp_.with_name(f"group_excess{stamp}_LABEL.json")
    vp_.parent.mkdir(parents=True, exist_ok=True)
    with open(vp_, "xb") as f:  # exclusive create, then commit, BEFORE returning (a detached client may be gone)
        f.write(b)
    if sb_ is not None:  # a37cc O1 (10-18): a failed record (gate or control) gets no sidecar, as the anchor, so it is never picked up labelled
        with open(sp_, "xb") as f:
            f.write(sb_)
    SOL.commit()
    if hashlib.sha256(vp_.read_bytes()).hexdigest() != hashlib.sha256(b).hexdigest():  # 507-prep R2: verify the file actually written
        raise SystemExit("STOP: the Volume copy's sha256 differs")
    if sb_ is not None and hashlib.sha256(sp_.read_bytes()).hexdigest() != hashlib.sha256(sb_).hexdigest():
        raise SystemExit("STOP: the Volume copy of the label sidecar differs")
    sc_ = {"volume_path": str(sp_.relative_to("/sol")), "sha256": hashlib.sha256(sb_).hexdigest()} if sb_ is not None else None
    print(json.dumps({"volume_path": str(vp_.relative_to("/sol")), "sha256": hashlib.sha256(b).hexdigest(), "label_sidecar": sc_}))  # path@sha in the app logs (detached)
    return {"bytes": b, "sha256": hashlib.sha256(b).hexdigest(), "volume_path": str(vp_.relative_to("/sol")), "pass": ok,
            "label_sidecar": sc_ | {"bytes": sb_} if sb_ is not None else None,
            "four_ci95_present": ex.get("ci95") is not None,  # keys / verdicts only
            "fails": {lb: [k for k, v_ in g.items() if v_ is not True] for lb, g in gate.items() if any(v_ is not True for v_ in g.values())},
            "control_ok": control_ok, "tolerance": TOL, "inputs": {"record_sha256": record_sha, "npz_sha256": npz_sha}}  # no diff value


def summary(res, out, sout):
    """The laptop's one printed line: keys, shas and booleans only, never a measured value (route C: no float on laptop stdout)."""
    return {"out": str(out), "volume_path": res["volume_path"], "sha256": res["sha256"], "gate_pass": res["pass"], "gate_fails": res["fails"],
            "control_ok": res["control_ok"] is True, "four_ci95_present": res["four_ci95_present"] is True, "inputs": res["inputs"],
            "label_sidecar": {"out": str(sout), "volume_path": res["label_sidecar"]["volume_path"], "sha256": res["label_sidecar"]["sha256"]}
            if res["label_sidecar"] is not None else None}


def git(*a):
    return subprocess.run(["git", *a], capture_output=True, check=True, cwd=HERE).stdout.decode().strip()


@app.local_entrypoint()
def main(record_src: str, record_sha: str, npz_sha: str, record_source: str, label_sidecar: str):
    if git("status", "--porcelain", "--", Path(__file__).name):
        sys.exit("commit this script before running it")
    if not rel_ok(record_src, "xfer/fpr507prep/records/"):
        sys.exit("STOP: --record-src must be a relative path under xfer/fpr507prep/records/ (no leading slash, no ..)")
    lb_, lh_ = sidecar_bytes(label_sidecar)  # 09: the pinned label sidecar, as bytes (checked and parsed in the container)
    launch = {"record_source": record_source, "label_sidecar": label_sidecar, "git": git("rev-parse", "HEAD"),  # 507-prep N4: captured at launch
              "script_sha256": hashlib.sha256((HERE / Path(__file__).name).read_bytes()).hexdigest()}
    stamp = "_" + datetime.datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S") + "_" + uuid.uuid4().hex  # Astra r3/r4: unique per launch (full uuid)
    print(json.dumps({"launch": launch, "stamp": stamp, "volume_out": f"{VOL_OUT}/group_excess{stamp}.json"}))
    res = build.remote(record_src, record_sha, npz_sha, stamp, launch, lb_, lh_)
    b = res["bytes"]
    if hashlib.sha256(b).hexdigest() != res["sha256"]:
        sys.exit("STOP: the returned bytes differ from the Volume copy's sha256")
    out = HERE / "results" / f"group_excess{stamp}.json"
    with open(out, "xb") as f:  # exclusive create; results/ is append-only; byte-identical to the Volume copy
        f.write(b)
    sout = None
    if res["label_sidecar"] is not None:  # the record's own label sidecar, written as returned (never parsed here); none for a failed record
        sb = res["label_sidecar"]["bytes"]
        if hashlib.sha256(sb).hexdigest() != res["label_sidecar"]["sha256"]:
            sys.exit("STOP: the returned label sidecar differs from the Volume copy's sha256")
        sout = HERE / "results" / f"group_excess{stamp}_LABEL.json"
        with open(sout, "xb") as f:
            f.write(sb)
    print(json.dumps(summary(res, out, sout)))  # no value: max_abs_diff lives in the file only (2e; route C)
    if res["pass"] and res["label_sidecar"] is None:
        sys.exit("STOP: the record passed but came back without its label sidecar")
    if res["control_ok"] is not True:
        sys.exit("STOP: the six-probe control did not match the record (control_ok false)")
    if not res["pass"]:
        sys.exit("STOP: a gate failed; the file is written for diagnosis only, never for the write-up")
