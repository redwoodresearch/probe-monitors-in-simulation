"""APPEND-ONLY INPUTS_FULL companion for own_all_rs{,25,10}_uncut_L48 (4f / 2e, 10-16): top-level pass; the exact R and C score files (+ sha256) per
variant, read from the score manifests (each sha-checked against the committed score record); the refit-match gate (both gate files' sha256, the 30
comparisons, the fit record sha256); and a by-key assertion that every scored row's model / training-manifest sha256 is one of fit.fits (cut none),
from the committed verify record (row level), and that none can be a descriptive cut-32768 refit (those were fit in memory only: no weights, no sha)."""

import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def _git_bytes(rel):
    b = (REPO / rel).read_bytes()
    import subprocess

    if subprocess.run(["git", "-C", str(REPO), "show", f"HEAD:{rel}"], capture_output=True).stdout != b:
        raise SystemExit(f"{rel}: not committed as read")
    return b


def main(fit_rel, score_rel, verify_rel, man_dir, out_rel):
    fb, sb, vb = _git_bytes(fit_rel), _git_bytes(score_rel), _git_bytes(verify_rel)
    fr, sr, vr = json.loads(fb), json.loads(sb), json.loads(vb)
    fit = fr["fit"]
    fsha = fit["fit_record_sha256"]
    if sr["fit_record_sha256"] != fsha or not vr["fit_record"].startswith(f"Volume fit record {fsha[:8]}") or vr["score_records"] != [{"path": score_rel, "sha256": hashlib.sha256(sb).hexdigest()}]:
        raise SystemExit("the score / verify records are not of this fit, or the verify did not check this score record")
    fits = {(f["variant"], f["readout"], f["fold"]): f for f in fit["fits"]}
    wsha, tsha = {f["weights_sha256"] for f in fits.values()}, {f["training_manifest_sha256"] for f in fits.values()}
    variants = sorted({f["variant"] for f in fits.values()})
    files, bad = {v: {"R": [], "C": []} for v in variants}, []
    for r in sr["results"].values():
        for fo, x in r["folders"].items():
            mb = (Path(man_dir) / f"{fo}.json").read_bytes()
            if hashlib.sha256(mb).hexdigest() != x["manifest_sha256"]:
                raise SystemExit(f"{fo}: manifest differs from the score record's sha256")
            man = json.loads(mb)
            for k, mm in man["models"].items():  # every model the manifest names is a fit.fits model, with the fit record's shas
                v, rd, f_ = k.split("|")
                ff = fits.get((v, rd, int(f_[1:])))
                if not ff or (mm["weights_sha256"], mm["training_manifest_sha256"], mm["preproc_sha256"]) != (ff["weights_sha256"], ff["training_manifest_sha256"], ff["preproc_sha256"]):
                    bad.append(f"{fo} {k}")
            for name, h in man["files"].items():
                v, rd = name.removesuffix(".npz").split("__")
                files[v][rd].append({"folder": fo, "batch": x["batch"], "path": f"{x['path']}/{name}", "sha256": h})
    desc = fit["descriptive_cut32768"]
    desc_with_weights = [k for k, d in desc.items() if any("sha256" in kk for kk in d)]
    vc = vr["rows"]["counts"]
    rows_ok = all(c["mismatch"] == 0 and c["match"] == c["rows"] and c["rows_fold_minus1"] == 0 for c in vc.values()) and set(vc) == {f"{v}|{rd}" for v in variants for rd in "RC"}
    gate = fit["gate"]
    ok = (gate["refit_match_pass"] and gate["all_certified"] and len(gate["refit_match"]) == 30 and all(g["pass"] for g in gate["refit_match"].values()) and not bad
          and rows_ok and not desc_with_weights and vr["pass"] and fit["cut"] == "none" and all(len(files[v][rd]) == len(files[variants[0]]["R"]) for v in variants for rd in "RC"))
    note = {"label": "own_all_rs{,25,10}_uncut_L48 INPUTS_FULL companion (APPEND-ONLY; 4f / 2e, 10-16): new probes trained on full transcripts (OOD class (i)); POST HOC; PRELIMINARY",
            "pass": ok, "probes": variants, "readouts": ["R", "C"],
            "fit_record": {"git": {"path": fit_rel, "sha256": hashlib.sha256(fb).hexdigest()}, "fit_record_sha256": fsha, "cut": fit["cut"],
                           "training_units_past_32768": fit["training_units_past_32768"], "fits": len(fits), "all_certified": gate["all_certified"]},
            "refit_match": {"pass": gate["refit_match_pass"], "on": gate["refit_match_on"], "tol_refit": gate["tol_refit"], "comparisons": len(gate["refit_match"]),
                            "max_abs_diff": max(g["weights_max_abs_diff"] for g in gate["refit_match"].values()),
                            "gate_records": {rd: {"path": f"/own_all_rs_uncut/{fr['stamp (PT)']}/gate/{rd}.json on probe-shift_aswin-surya_sub-p2probes_scores_kill-2026-11-13", "sha256": h}
                                             for rd, h in fit["gate_files_sha256"].items()},
                            "fit_record_sha256": fsha},
            "score_record": {"path": score_rel, "sha256": hashlib.sha256(sb).hexdigest()},
            "verify_record": {"path": verify_rel, "sha256": hashlib.sha256(vb).hexdigest(), "pass": vr["pass"], "negative_e2e_failed_closed": vr["negative_e2e"]["failed_closed"]},
            "by_key_assertion": {"rows_checked": {k: {"rows": c["rows"], "match_fit_fits": c["match"], "mismatch": c["mismatch"]} for k, c in sorted(vc.items())},
                                 "manifest_models_not_in_fit_fits": len(bad), "fit_fits_weights_sha256": len(wsha), "fit_fits_training_manifest_sha256": len(tsha),
                                 "descriptive_cut32768_entries": len(desc), "descriptive_cut32768_with_saved_weights": len(desc_with_weights),
                                 "rows_from_descriptive_cut32768": 0,
                                 "how": "verify_rows read every emitted row's model_weights_sha256 / preproc_sha256 / training_manifest_sha256 back from the npz files and matched "
                                        "them to fit.fits' entry for the row's designated fold model; the descriptive cut-32768 refits were fit in memory only (no weights file, "
                                        "no sha), so no row can carry one"},
            "score_files": files, "score_files_per_variant_readout": len(files[variants[0]]["R"])}
    (REPO / out_rel).open("x").write(json.dumps(note, indent=1))
    print(json.dumps({"pass": ok, "folders_per_variant_readout": note["score_files_per_variant_readout"], "manifest_models_not_in_fit": len(bad),
                      "desc_with_weights": len(desc_with_weights), "rows_ok": rows_ok}))
    if not ok:
        raise SystemExit("STOP: the companion does not pass")


if __name__ == "__main__":
    main(*sys.argv[1:6])
