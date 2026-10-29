"""Five-probe kappa sensitivity + kappa degeneracy evidence (09 @ 3fbc0e451, via 2e 10-16 ~15:0x): a REDUCE over a finished kappa job's
stored rows; no row is re-scored. One Modal CPU container reads the job's spec (its reference record bytes, launch-share table, groups) and its
row store from the solavg Volume, applies run_rows' full_as relabel, and re-runs run_peraction311.kappa_reduce with the job's groups plus
five_probe_sensitivity (headline_six minus own_all_rs_uncut_L48).
GATE (exact): the re-run's per-row / per-combo kappa and every original group equal the committed kappa record's; a mismatch STOPs.
Degeneracy (counts only), for each of the six lead probes x readout x union combo x mode, on the combo's per-draw 2x2 margins (combined as
kappa_reduce combines them): draws (1..D) where kappa(sim, run1) / kappa(run2, run1) is UNDEFINED (1 - p_e not > 0, or a non-finite margin) and
draws where each world's catch (run1, run2, sim) is at 0 or at 1; draw 0 reported separately; its draw-0 kappa equals the record's point (gated).
Per combo and mode, every probe at least as degenerate as the on-policy probe is listed, and an on-policy probe that is NOT degenerate says so.

  uv run --no-project --with modal==1.6.1 --with numpy==2.3.3 modal run --detach experiments/2026-10-15_fpr_sweep_507_prep/run_kappa_extra.py \
      --job kappa_CRD_2026-10-16_135837 --record-sha <the committed record JSON sha256>
"""

import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_peraction311 as R  # noqa: E402

app = modal.App("probe-shift_aswin-surya_sub-pertask_kappaextra_kill-2026-11-14")
image = R.image.add_local_file(HERE / "run_peraction311.py", "/root/run_peraction311.py")
ON_POLICY = "own_all_rs_uncut_L48"
FIVE_LABEL = ("POST HOC (chosen after the per-set values were seen): excludes the on-policy probe (own_all_rs_uncut_L48), whose kappa is degenerate "
              "at the catch ceiling. The six-probe group is primary.")  # 09 @ 3fbc0e451, verbatim
POOLED_SUFFIX = ("POST HOC; five-probe without on-policy; the five-probe sensitivity (09 @ 3fbc0e451) and kappa degeneracy counts; a reduce over "
                 "the job's stored rows; chosen after the per-set values were seen (0d, 10-16); the six-probe group stays PRIMARY")  # 09 10-17: a pooled
# kappa job's record label = 09's kappa label (the job's own) + this, verbatim


def run_label(spec_label, per_set_label: str, pooled: bool = False, d1=None) -> str:
    """The record's meta.label. A job whose spec carries a meta.label (the pooled kappa: KAPPA_D1_LABEL, 09's exact string) must carry exactly
    run_peraction311.KAPPA_D1_LABEL, else STOP; its label is that string + '; ' + POOLED_SUFFIX. A pooled job (spec meta.pooled_world) with no
    label STOPs (2e 10-17). A per-set job (no spec label) keeps its label."""
    if d1 is not None:  # route C's A (09's label ruling, 10-17): the spec ships no label and carries kappa._d1 (A's label sidecar bytes):
        if spec_label is not None or not pooled:  # compose it exactly as the driver did (R.routec_d1_label checks and composes) + the suffix
            raise SystemExit("a route C kappa job must be pooled and carry no spec meta.label (09's label ruling; STOP)")
        return f"{R.routec_d1_label(d1)}; {POOLED_SUFFIX}"
    if spec_label is None:
        if pooled:
            raise SystemExit("a pooled-world kappa job's spec carries no meta.label (09's kappa label; STOP)")
        return per_set_label
    if spec_label != R.KAPPA_D1_LABEL:
        raise SystemExit("the kappa job's meta.label is not run_peraction311.KAPPA_D1_LABEL (09's kappa label; STOP)")
    return f"{spec_label}; {POOLED_SUFFIX}"


def _counts(Ks: list) -> dict:
    """Degeneracy counts on the given rows' per-draw 2x2 margins, combined as kappa_reduce combines a union combo (one row: its own margins);
    counts and booleans only. 09 10-17: also reported per probe and per set / origin row (not only per pooled combo)."""
    import numpy as np

    with np.errstate(divide="ignore", invalid="ignore"):
        Dd = sum(K["D"] for K in Ks)
        comb = lambda k_: sum(np.where(K["D"] > 0, K[k_] * K["D"], 0.0) for K in Ks) / Dd  # noqa: E731
        e = {"undefined_draws": {}, "undefined_draw0": {}, "catch_at_ceiling_draws": {}, "catch_at_ceiling_draw0": {}}
        for pair in ("run2", "sim"):
            po, p1, q1 = comb(f"{pair}|po"), comb(f"{pair}|p1"), comb(f"{pair}|q1")
            pe = p1 * q1 + (1 - p1) * (1 - q1)
            undef = ~(np.isfinite(po) & np.isfinite(pe) & (1 - pe > 0))
            e["undefined_draws"][f"kappa_{pair}_vs_run1"] = int(undef[1:].sum())
            e["undefined_draw0"][f"kappa_{pair}_vs_run1"] = bool(undef[0])
        for w, p in {"run1": comb("run2|p1"), "run2": comb("run2|q1"), "sim": comb("sim|q1")}.items():
            e["catch_at_ceiling_draws"][w] = {"at_0": int((p[1:] == 0).sum()), "at_1": int((p[1:] == 1).sum())}
            e["catch_at_ceiling_draw0"][w] = "0" if p[0] == 0 else "1" if p[0] == 1 else None
    e["draws"] = int(len(Dd) - 1)
    return e


def degeneracy(outs, ref, labels):
    import numpy as np

    by = {}
    for o in outs:
        if "_kappa" in o:
            by.setdefault(o["label"], {})[o["origin"]] = o
    res, gate = {}, {"compared": 0, "mismatch": []}
    for lb in labels:
        for cb in sorted(R.COMBOS_ALLO):
            rc = ref["pooled"].get(lb, {}).get(cb)
            if rc is None:
                continue  # the combo is not in this build for this label (its absence is checked per combo in the summary)
            if rc.get("not_readable"):  # 09 item 45(2) (pooled record): withheld in kappa_reduce, so no degeneracy counts either
                for mode in ("joint", "fixed"):
                    res.setdefault(cb, {}).setdefault(mode, {})[lb] = {"unavailable": R.NOT_READABLE}
                continue
            if "unavailable" in rc or any(by.get(lb, {}).get(m_) is None for m_ in rc["members"]):
                for mode in ("joint", "fixed"):  # explicit, never a silent skip (Codex r1)
                    res.setdefault(cb, {}).setdefault(mode, {})[lb] = {"unavailable": rc.get("unavailable") or "members without stored kappa rows: "
                                                                       + str([m_ for m_ in rc["members"] if by.get(lb, {}).get(m_) is None])}
                continue
            for mode in ("joint", "fixed"):
                Ks = [{k_: np.asarray(v_, float) for k_, v_ in by[lb][m_]["_kappa"][mode].items()} for m_ in rc["members"]]
                Dd = sum(K["D"] for K in Ks)
                comb = lambda k_: sum(np.where(K["D"] > 0, K[k_] * K["D"], 0.0) for K in Ks) / Dd  # noqa: E731  (kappa_reduce's union-combo arithmetic)
                e = {"undefined_draws": {}, "undefined_draw0": {}, "catch_at_ceiling_draws": {}, "catch_at_ceiling_draw0": {}}
                cat = {"run1": comb("run2|p1"), "run2": comb("run2|q1"), "sim": comb("sim|q1")}
                for pair in ("run2", "sim"):
                    po, p1, q1 = comb(f"{pair}|po"), comb(f"{pair}|p1"), comb(f"{pair}|q1")
                    pe = p1 * q1 + (1 - p1) * (1 - q1)
                    undef = ~(np.isfinite(po) & np.isfinite(pe) & (1 - pe > 0))
                    e["undefined_draws"][f"kappa_{pair}_vs_run1"] = int(undef[1:].sum())
                    e["undefined_draw0"][f"kappa_{pair}_vs_run1"] = bool(undef[0])
                    with np.errstate(divide="ignore", invalid="ignore"):
                        k0 = float(np.where(1 - pe > 0, (po - pe) / (1 - pe), np.nan)[0])
                    want = ref["_kappa_points"].get((lb, cb, mode, pair), "absent")  # the record's point (None if undefined)
                    if want == "absent":
                        gate["mismatch"].append(f"{lb}|{cb}|{mode}|{pair}: no point in the record")
                        continue
                    gate["compared"] += 1
                    if not ((want is None and not np.isfinite(k0)) or (want is not None and k0 == want)):
                        gate["mismatch"].append(f"{lb}|{cb}|{mode}|{pair}: draw-0 kappa differs from the record's point")
                for w, p in cat.items():
                    e["catch_at_ceiling_draws"][w] = {"at_0": int((p[1:] == 0).sum()), "at_1": int((p[1:] == 1).sum())}
                    e["catch_at_ceiling_draw0"][w] = "0" if p[0] == 0 else "1" if p[0] == 1 else None
                e["draws"] = int(len(Dd) - 1)
                res.setdefault(cb, {}).setdefault(mode, {})[lb] = e
    def score(e):  # (undefined bootstrap draws, max over the two pairs; draw 0 undefined in either pair)
        return max(e["undefined_draws"].values()), any(e["undefined_draw0"].values())

    summary = {}
    for cb, md in res.items():
        for mode, d in md.items():
            for ro in ("C", "R"):
                want_ = [lb for lb in labels if lb.endswith(f" {ro}")]
                miss_ = [lb for lb in want_ if lb not in d or "unavailable" in d[lb]]
                if miss_:  # comparative claims need every probe (Codex r1)
                    summary[f"{cb}|{mode}|{ro}"] = {"withheld": f"incomplete coverage: {miss_}"}
                    continue
                sc = {lb: score(d[lb]) for lb in want_}
                op_lb = f"{ON_POLICY} {ro}"
                opn, op0 = sc[op_lb]
                degen = sorted(lb for lb, (n_, z_) in sc.items() if n_ > 0 or z_)
                same = sorted(lb for lb, (n_, z_) in sc.items() if lb != op_lb and (n_ > 0 or z_) and n_ >= opn and (z_ or not op0))
                summary[f"{cb}|{mode}|{ro}"] = {
                    "undefined_draws_and_draw0_by_probe": {lb: {"undefined_draws_max_over_pairs": n_, "draw0_undefined": z_} for lb, (n_, z_) in sc.items()},
                    "on_policy_degenerate": opn > 0 or op0, "probes_with_any_undefined_kappa": degen,
                    "other_probes_at_least_as_degenerate_as_on_policy": same,
                    "statement": (("the on-policy probe is NOT degenerate here (no undefined kappa draw)"
                                   + (f"; probes with undefined kappa draws: {', '.join(degen)}" if degen else "")) if not (opn > 0 or op0)
                                  else ("other probes are at least as degenerate as the on-policy probe: " + ", ".join(same)) if same
                                  else "no other probe is as degenerate as the on-policy probe")}
    by_origin = {}  # 09 10-17: per probe x readout x set / origin row x mode (each stored row's own margins)
    for lb in labels:
        ogs_ = sorted(set((ref.get("rows") or {}).get(lb, {})) | set(by.get(lb, {})))  # every origin the reference or the row store names
        if not ogs_:
            by_origin[lb] = {"unavailable": "no stored kappa rows for this probe x readout"}
            continue
        for og in ogs_:
            rp_ = ((ref.get("rows") or {}).get(lb) or {}).get(og) or {}
            if rp_.get("not_readable"):  # 09 item 45(2): kappa_reduce withholds this row, so no counts either (2e / 507-prep's R1)
                by_origin.setdefault(lb, {})[og] = {"unavailable": R.NOT_READABLE}
            elif "withheld" in rp_:
                by_origin.setdefault(lb, {})[og] = {"unavailable": rp_["withheld"]}
            elif og not in by.get(lb, {}):
                by_origin.setdefault(lb, {})[og] = {"unavailable": "no stored kappa row"}
            else:
                by_origin.setdefault(lb, {})[og] = {mode: _counts([{k_: np.asarray(v_, float) for k_, v_ in by[lb][og]["_kappa"][mode].items()}])
                                                    for mode in ("joint", "fixed")}
    return res, summary, gate, by_origin


@app.function(image=image, cpu=(4, 8), memory=(32768, 65536), timeout=3600, retries=0, volumes={"/sol": R.vol_out})
def extra(job: str, meta: dict) -> dict:
    import pickle

    spec = pickle.loads((Path("/sol/xfer/fpr507prep/drivers") / f"{job}.pkl").read_bytes())
    if spec["mode"] != "kappa":
        raise SystemExit("not a kappa job spec")
    km = spec["meta"]["kappa"]
    label_ = run_label(spec["meta"].get("label"), meta["label"], bool(spec["meta"].get("pooled_world")), km.get("_d1"))  # before any work: a pooled job
    # must carry 09's exact kappa label
    ref = R.kappa_ref(R.ref_bytes(km["_ref_blob"], km["_ref_sha256"], "--kappa"))
    sd_ = f"xfer/fpr507prep/rowstore/{Path(spec['dest']).stem}"
    outs = []
    for i, (lb, cf, _) in enumerate(spec["jobs"]):  # every stored row (a missing one STOPs); run_rows' full_as relabel, as the job applied it
        o = R.stored_row(f"{sd_}/{i:04d}")
        if o is None:
            raise SystemExit(f"row {i:04d} ({lb}) is not in the row store")
        if cf.get("full_as"):
            if o.get("label") != lb:
                raise SystemExit(f"full_as: row label {o.get('label')} != job label {lb}")
            o["label"], o["full_variant_of"] = cf["full_as"], lb
        o.pop("_calib", None)
        outs.append(o)
    groups = dict(km.get("groups") or {})
    six = list(groups["headline_six"])
    groups["five_probe_sensitivity"] = [f_ for f_ in six if f_ != ON_POLICY]
    if len(groups["five_probe_sensitivity"]) != 5:
        raise SystemExit("headline_six does not contain the on-policy probe exactly once")
    red = R.kappa_reduce.local(outs, ref, km["share"], groups)
    rb_ = (Path("/sol") / spec["dest"]).read_bytes()
    rec_sha = hashlib.sha256(rb_).hexdigest()
    if rec_sha != meta["record_sha256"]:  # the comparison record must be the committed one (its JSON's sha, pinned on the command line; Codex r1)
        raise SystemExit(f"{spec['dest']}: sha256 {rec_sha} is not the pinned committed record's {meta['record_sha256']}")
    rec_ = json.loads(rb_)
    gate = {"record": spec["dest"], "record_sha256": rec_sha, "mismatch": []}
    norm = lambda x: json.dumps(x, sort_keys=True, default=float)  # noqa: E731
    if norm(red["kappa"]) != norm(rec_["kappa"]):
        gate["mismatch"].append("per-row / per-combo kappa differ from the record's")
    for k_, v_ in rec_["kappa_groups"].items():
        if norm(red["groups"].get(k_)) != norm(v_):
            gate["mismatch"].append(f"group {k_} differs from the record's")
    if not red["gate"]["pass"]:
        gate["mismatch"].append("kappa_reduce's own gate failed on the re-run")
    pts = {}
    for lb, d_ in rec_["kappa"].items():
        for cb, e_ in (d_.get("combos") or {}).items():
            for mode in ("joint", "fixed"):
                for pair in ("run2", "sim"):
                    if isinstance(e_.get(mode), dict):
                        pts[(lb, cb, mode, pair)] = e_[mode][f"kappa_{pair}_vs_run1"]["point"]
    ref["_kappa_points"] = pts
    deg, summ_, dgate, by_og_ = degeneracy(outs, ref, [f"{f_} {ro}" for f_ in six for ro in ("C", "R")])
    gate["mismatch"] += dgate["mismatch"]
    gate |= {"draw0_points_compared": dgate["compared"], "pass": not gate["mismatch"]}
    out = {"meta": meta | {"job": job, "spec_dest": spec["dest"], "spec_git": spec["meta"].get("git"), "label": label_}, "gate": gate,
           "five_probe_sensitivity": {"label": FIVE_LABEL, "members": groups["five_probe_sensitivity"],
                                      "groups": {k_: v_ for k_, v_ in red["groups"].items() if k_.startswith("five_probe_sensitivity|")}},
           "degeneracy": {"rule": ("undefined: 1 - p_e not > 0 (p_e = p1 q1 + (1 - p1)(1 - q1)) or a non-finite margin; ceiling: the world's catch "
                                   "exactly 0 or 1; per union combo x mode, the combo's per-draw 2x2 margins as kappa_reduce combines them; counts over "
                                   "draws 1..D, draw 0 separately"), "by_combo": deg, "by_probe_origin": by_og_, "summary": summ_}}
    b = json.dumps(out, indent=1, default=float).encode()
    nm = f"{meta['name']}{'' if gate['pass'] else '_GATEFAIL'}.json"
    p = Path("/sol/xfer/pertask") / nm
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("xb") as fh:
        fh.write(b)
    R.vol_out.commit()
    if not gate["pass"]:  # the diagnostic record is kept on the Volume (_GATEFAIL); the run fails (as the per-task driver does; Codex r1)
        raise SystemExit(json.dumps({"GATE_FAIL": gate["mismatch"][:20], "record": str(p).removeprefix("/sol/")}))
    return {"name": nm, "volume_path": str(p).removeprefix("/sol/"), "sha256": hashlib.sha256(b).hexdigest(), "blob": b, "gate_pass": gate["pass"],
            "label": label_}


@app.local_entrypoint()
def main(job: str, record_sha: str):  # record_sha: the committed kappa record's JSON sha256 (the gate's comparison record)
    git = lambda *a: subprocess.run(["git", *a], capture_output=True, check=True, cwd=HERE).stdout  # noqa: E731
    if git("status", "--porcelain", "--untracked-files=no").strip():
        raise SystemExit("uncommitted tracked changes")
    for f_ in (Path(__file__), HERE / "run_peraction311.py"):  # the recorded git must contain the code that runs
        if git("show", f"HEAD:./{f_.name}") != f_.read_bytes():
            raise SystemExit(f"{f_.name} is not committed at HEAD as it is on disk")
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    meta = {"git": git("rev-parse", "HEAD").decode().strip(), "stamp": stamp, "app_id": app.app_id, "name": f"kappa_extra_{job}_{stamp}_{app.app_id}",
            "label": "POST HOC; the five-probe sensitivity (09 @ 3fbc0e451) and kappa degeneracy counts; a reduce over the job's stored rows",
            "record_sha256": record_sha}
    r = extra.remote(job, meta)
    if hashlib.sha256(r["blob"]).hexdigest() != r["sha256"]:
        raise SystemExit("result sha256")
    (HERE / "results" / r["name"]).open("xb").write(r["blob"])
    print(json.dumps({"record": r["name"], "volume_path": r["volume_path"], "sha256": r["sha256"], "gate_pass": r["gate_pass"], "label": r["label"]}),
          flush=True)  # 2e 10-17: the launch line checks the label
