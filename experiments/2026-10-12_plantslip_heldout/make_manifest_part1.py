"""Manifest part 1 for the held-out planting test: pins code, configs, cohort/folds source, readouts, existing detector
artifacts (path + sha256 from the fit run records) and the Qwen3-32B fitting specification. Reads git only; prints no
metric value."""
import json, subprocess, sys
R = sys.argv[1]
def git(*a): return subprocess.run(["git", "-C", R, *a], capture_output=True, text=True, check=True).stdout
def full(h): return git("rev-parse", h + "^{commit}").strip()
def show(ref, path): return json.loads(git("show", f"{ref}:{path}"))
runs = {"olmo": ("09e0e57f", "experiments/2026-10-12_q2_olmo_reader/results/run_2026-10-12_110549.json", [16, 24, 48, 63]),
        "llama": ("1d6c881d", "experiments/2026-10-12_q2_llama_reader/results/run_2026-10-12_143129.json", [20, 30, 40, 50, 79])}
arts = {}
for rd, (ref, path, layers) in runs.items():
    objs = show(ref, path)["archive"]["objects"]
    arts[rd] = {"run_record": f"{path}@{full(ref)}", "layers": layers,
                "objects": {k: {kk: v[kk] for kk in v if kk in ("sha256", "bytes", "key", "s3", "url")}
                            for k, v in objs.items() if any(k in (f"probes_L{L}.npz", f"scores_L{L}.npz", f"fits_L{L}.json") for L in layers)}}
    missing = [f"{n}_L{L}" for L in layers for n in ("probes", "scores") if f"{n}_L{L}.npz" not in arts[rd]["objects"]]
    if missing:
        raise SystemExit(f"{rd}: missing artifacts {missing}")
m = {
 "rule": "experiments/2026-10-12_plantslip_heldout/RULE.md (this commit)",
 "configurations": {"qwen3-32b": [20, 28, 36, 44, 52, 60], "olmo": [16, 24, 48, 63], "llama": [20, 30, 40, 50, 79]},
 "cohort": "#36's 939 cells (the forest B rows' cohort); folds and splits as in each reader's config `splits` block",
 "readout": "output_end, mean over a run's outputs (#36 recipe); Olmo R2 (C2), Llama R2J, Qwen3-32B segment-end last content token",
 "rate": "1% false alarms, nested frozen per-fold thresholds",
 "code": {"fit_runner": f"experiments/2026-10-06_q3_attack_detector/run_q3_attack.py@{full('f0ff845f')}",
          "selection_and_reading": f"experiments/2026-10-06_q3_attack_detector/analyze_q3_attack.py@{full('f0ff845f')}",
          "plant": "experiments/2026-10-12_plantslip_transfer/plant_both.py@" + full("6bafb0c3")},
 "configs": {"qwen3-32b": f"experiments/2026-10-06_q3_attack_detector/config/q3_attack.yaml@{full('dea0a4e7')}",
             "olmo": f"experiments/2026-10-12_q2_olmo_reader/config/q3_attack_olmo.yaml@{full('a0411fba')}",
             "llama": f"experiments/2026-10-12_q2_llama_reader/config/q3_attack_llama.yaml@{full('5a0aa1dc')}"},
 "existing_detector_artifacts": arts,
 "qwen3_fitting_spec": {"runner": "run_q3_attack.py as at f0ff845f, copied into this folder with the load_arms filter "
                        "[real-honest, real-attack-e1, real-attack-e2] after select_runs, load_arms in the data identity, "
                        "and no-sim-attack assertions in pack_layer and probe_layer", "config": "q3_attack.yaml@dea0a4e7 with layers [20, 28, 36, 44, 52, 60]",
                        "dump": "2026-10-06_la_segment-end_qwen3-32b_alllayers_v1"},
 "split_half": "task halves by parity of the first byte of sha256(task id); identical for every configuration",
 "baseline_sources": {"B1": "experiments/2026-10-06_q1_text_baselines/results/tfidf_raw_summary_2026-10-06_134708.json@" + full("607c93ce"),
                      "B3": "b3_forecasts.json (this commit), made by make_b3.py from metrics@c24106ca, reply_break_olmo@668b902a, reply_break_llama@ab5caa5d",
                      "calibration_offset": "1.432 = median(y - x) over P1-P8, scatter_values_2026-10-12_155238.json behind results_manifest_v3.json@" + full("79a02578")},
}
json.dump(m, sys.stdout, indent=1)
