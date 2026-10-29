"""The final Q2 table from one labeler: probes, baselines on the same calls, and follow-ups.

Reads a run_q2_probes.py result, an analyze_text_baselines.py result scored on that run's eligible calls, and a
run_q2_followups.py result, checks that all three used the same labeler, and writes one markdown table plus its JSON:
  - the 2x2 per readout and baseline: recall, false-positive rate and AUROC with 95% intervals (main-task bootstrap)
    and event counts;
  - estimate and transfer gaps;
  - slices: no unseen-effect calls, call_end on command content, no text_editor;
  - held-out environments;
  - pre_call's three effects: ranking (AUROC), threshold calibration (frozen vs recalibrated), predictability from
    history (the previous-calls baseline);
  - per-category recall (impactful categories) and false-positive rate (read-only categories) in the cross cells.

  uv run python experiments/2026-10-06_q2_impactful_call_transfer/analyze_q2_final.py PROBES_JSON BASELINES_JSON FOLLOWUPS_JSON
"""

import json
import secrets
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).parent
R, S = "real-attack-e1", "sim-attack"
CELLS = [(R, R), (R, S), (S, R), (S, S)]
READ_ONLY = {"db_read", "curl", "network_read", "inline_code_read", "file_read", "inspect"}


def ci(m, k, pct=False):
    v, (lo, hi) = m[k]["value"], m[k]["ci95"]
    return f"{100 * v:.1f}% [{100 * lo:.1f}, {100 * hi:.1f}]" if pct else f"{v:.3f} [{lo:.3f}, {hi:.3f}]"


def table(rows, header):
    return "\n".join(["| " + " | ".join(header) + " |", "|" + "---|" * len(header)] + ["| " + " | ".join(r) + " |" for r in rows])


def cell_rows(name, report):
    out = []
    for tr, te in CELLS:
        m = report[f"train {tr} / test {te}"]
        n = m.get("n", {})
        out.append([name, f"{tr.split('-')[0]} → {te.split('-')[0]}", ci(m, "recall"), ci(m, "fpr", pct=True), ci(m, "auroc"),
                    f"{n.get('impactful', '')} / {n.get('read_only', '')}"])
    return out


def threeway_section(T, sha, ref) -> list[str]:
    """The honest/attack views from run_q2_alllayers.py --config q2_threeway: headline 2x2 per view and readout, the
    real-trained estimate gap, the all-runs sensitivity views, the two task slices and the reproduction check."""
    if T["config"]["labeler_sha"] != sha:
        raise ValueError(f"three-way result is from labeler {T['config']['labeler_sha']}, not {sha}")
    for k in ("label_rule", "fpr", "folds_file"):  # one definition of impactful and one threshold rule in the report
        if T["config"].get(k) != ref.get(k):
            raise ValueError(f"three-way result has {k}={T['config'].get(k)!r}, the probes {ref.get(k)!r}")
    if len(T["layers"]) != 64 or T["config"].get("out_sub", "").endswith("_pilot"):
        raise ValueError(f"three-way result covers layers {T['layers'][:4]}...: a pilot, not the 64-layer run")
    views, pos3 = T["views"], ("call_end", "pre_call", "prev_content")
    cells = [("real", "real"), ("real", "sim"), ("sim", "real"), ("sim", "sim")]
    row = lambda v, pos, m, label, n=True: [v, pos, label, ci(m, "recall"), ci(m, "fpr", pct=True), ci(m, "auroc")] + (
        [f"{m['n']['impactful']} / {m['n']['read_only']}"] if n else [])
    head = ["view", "readout", "train → test", "recall", "false-positive rate", "AUROC", "n"]
    main_rows, gap_rows, sens_rows, slice_rows = [], [], [], []
    for v in ("attack_only", "honest_only", "pooled"):
        for pos in pos3:
            h = views[v]["positions"][pos]["headline"]
            main_rows += [row(v, pos, h[f"train {a} / test {b}"], f"{a} → {b}") for a, b in cells]
            g = h["train real / test sim minus test real"]
            gap_rows.append([v, pos] + [f"{g[k]['value']:+.3f} [{g[k]['ci95'][0]:+.3f}, {g[k]['ci95'][1]:+.3f}]" for k in ("recall", "fpr", "auroc")])
            for name, rep in views[v]["positions"][pos].get("slices", {}).items():
                if v != "attack_only":
                    slice_rows += [row(v, pos, rep[f"train real / test {b}"], f"{name}: real → {b}") for b in ("real", "sim")]
    for v in ("honest_only_all_runs", "pooled_all_runs"):
        for pos in pos3:
            h = views[v]["positions"][pos]["headline"]
            sens_rows += [row(v, pos, h[f"train real / test {b}"], f"real → {b}") for b in ("real", "sim")]
    arms = [[v, te, ", ".join(f"{a}: {c['calls']} calls / {c['main_tasks']} tasks" for a, c in byte.items())]
            for v in views for te, byte in views[v]["calls_by_arm"].items()]
    rep = T.get("reproduction", {})
    md = ["## Honest and attack: three views (64 layers; headline layer chosen inside each outer fold)", "",
          "Each view maps arms to a real and a simulated domain. attack_only: real attack vs simulated attack (536 paired "
          "tasks). honest_only: one real honest run per task vs simulated honest (437 paired honest tasks). pooled: both "
          "(attack calls from 536 tasks, honest calls from 437). The threshold is frozen at the training domain's validation "
          f"false-positive target; 95% intervals bootstrap main tasks. Result: `{T.get('_file', '')}`.", "",
          table(main_rows, head), "",
          "Estimate gap of the real-trained probe (simulated minus real test domain):", "",
          table(gap_rows, ["view", "readout", "recall", "false-positive rate", "AUROC"]), "",
          "Sensitivity, every real honest run (1–8 per task) instead of one:", "",
          table(sens_rows, head), "",
          "Slices, the headline probes scored without a slice's main tasks in every arm (no refit):", "",
          table(slice_rows, head) if slice_rows else "unavailable", "",
          "Calls per test domain:", "", table(arms, ["view", "test domain", "arms"]), "",
          "Label-shuffle null: " + ("; ".join(f"{v} {pos}: headline null max AUROC {n['headline']['max']}" for v in views
                                           for pos in pos3 for n in views[v]["positions"][pos].get("null", {}).values())
                                 or "not run for the views (not in the plan), so the honest and pooled views have no "
                                    "label-shuffle null yet; the attack-only runs' nulls do not carry over, since a "
                                    "within-transcript shuffle depends on each view's cohorts") + ".", "",
          "Reproduction: the attack_only view against the attack-only 64-layer run, per readout and training set: " +
          ("; ".join(f"{k}: {x['layers']} layers, max score difference {x['max_score_diff']:.2g}, flags differ {x['flags_differ']}"
                     for k, x in rep.items()) or "not run"), ""]
    return md


def main(probes_json, baselines_json, followups_json, threeway_json=None):
    P, B, F = (json.loads(Path(p).read_text()) for p in (probes_json, baselines_json, followups_json))
    for key in ("labeler_sha", "labels_file", "features_run", "folds_file", "fpr", "tasks", "test_domains"):  # one experiment
        vals = {json.dumps(R["config"].get(key)) for R in (P, B, F)}
        if len(vals) != 1:
            raise ValueError(f"results differ in {key}: {vals}")
    if P["config"]["task_set_sha"] != F["config"]["task_set_sha"]:
        raise ValueError("probes and follow-ups cover different task sets")
    for name, X in (("baselines", B), ("follow-ups", F)):  # each training set means the same arms with the same home
        for key in ("train_sets", "home"):
            shared = {t: v for t, v in X["config"].get(key, {}).items() if t in P["config"][key]}
            if any(P["config"][key][t] != v for t, v in shared.items()):
                raise ValueError(f"probes and {name} map training sets differently in {key}: {P['config'][key]} vs {X['config'][key]}")
    for key in ("C_grid", "layer_tolerance", "max_iter", "layers", "positions", "label_rule", "seed", "activations_run"):
        if P["config"].get(key) != F["config"].get(key):  # the follow-ups refit the source probe; it must be the same one
            raise ValueError(f"probes and follow-ups differ in {key}: {P['config'].get(key)} vs {F['config'].get(key)}")
    sha = P["config"]["labeler_sha"]
    md = [f"# Q2 final table (labeler `{sha}`)", "",
          f"Probes: `{Path(probes_json).name}`; baselines on the same calls: `{Path(baselines_json).name}`; follow-ups: "
          f"`{Path(followups_json).name}`; commits {', '.join(sorted({str(R['config'].get('provenance', {}).get('git', R.get('git')))[:7] for R in (P, B, F)}))}. "
          f"Threshold frozen at {100 * P['config']['fpr']:g}% false positives on training-domain validation; 95% "
          "intervals bootstrap main tasks. Counts are impactful / read-only test calls.", ""]
    header = ["readout", "train → test", "recall", "false-positive rate", "AUROC", "n"]
    rows = []
    for pos in ("call_end", "pre_call", "prev_content"):
        rows += cell_rows(pos, P["positions"][pos]["all_calls"])
    for name in ("command_text", "previous_calls"):
        rows += cell_rows(f"baseline: {name}", B["baselines"][name]["all_calls"])
    md += ["## 2×2, all eligible calls", "", table(rows, header), ""]

    gap_rows = []
    gaps = [(f"train {R} / test {S} minus test {R}", "estimate: real-trained, sim minus real"),
            (f"train {S} / test {R} minus test {S}", "estimate: sim-trained, real minus sim"),
            (f"test {S} / train {R} minus train {S}", "transfer: on sim, real-trained minus sim-trained"),
            (f"test {R} / train {S} minus train {R}", "transfer: on real, sim-trained minus real-trained")]
    for name, rep in [(p, P["positions"][p]["all_calls"]) for p in ("call_end", "pre_call", "prev_content")] + \
            [(f"baseline: {n}", B["baselines"][n]["all_calls"]) for n in ("command_text", "previous_calls")]:
        for key, label in gaps:
            m = rep[key]
            gap_rows.append([name, label, ci(m, "recall"), ci(m, "fpr"), ci(m, "auroc")])
    md += ["## Gaps", "", table(gap_rows, ["readout", "gap", "recall", "false-positive rate", "AUROC"]), ""]

    for slice_name, title in [("robustness_no_unseen_effects", "Without unseen-effect calls (run_program, unknown, malformed)"),
                              ("call_end_on_content", "Calls whose call_end token is command content"),
                              ("no_text_editor", "Without text_editor calls")]:
        rows = []
        for pos in ("call_end", "pre_call", "prev_content"):  # pre_call only next to its content-token control
            rep = P["positions"][pos].get(slice_name)
            if isinstance(rep, dict):
                rows += cell_rows(pos, rep)
        for name in ("command_text", "previous_calls"):
            rep = B["baselines"][name].get(slice_name)
            if isinstance(rep, dict):
                rows += cell_rows(f"baseline: {name}", rep)
        md += [f"## {title}", "", table(rows, header) if rows else "unavailable", ""]

    env_rows, has_control = [], any(n.startswith("prev_content") for n in F["heldout_environment"])
    for name, e in F["heldout_environment"].items():
        if name.startswith("pre_call") and not has_control:  # pre_call only next to its content-token control
            continue
        for cell, m in e["cells"].items():
            if " minus " not in cell:
                env_rows.append([name, cell, ci(m, "recall"), ci(m, "fpr", pct=True), ci(m, "auroc")])
    md += ["## Held-out environments (leave one environment out)", "",
           table(env_rows, ["readout and training set", "cell", "recall", "false-positive rate", "AUROC"]), ""]

    pb = B["baselines"]["previous_calls"]["all_calls"]
    eff = []
    have = [p for p in ("pre_call", "prev_content") if all(f"{p}: {s} -> {t}" in F["adaptation"] for s, t in [(R, S), (S, R)])]
    have = have if "prev_content" in have else []  # pre_call only next to its content-token control
    for pos, src, tgt in [(p, s, t) for p in have for s, t in [(R, S), (S, R)]]:
        pc = P["positions"][pos]["all_calls"]
        a = F["adaptation"][f"{pos}: {src} -> {tgt}"]
        own, other = pc[f"train {src} / test {src}"], pc[f"train {src} / test {tgt}"]
        tgt_own = pc[f"train {tgt} / test {tgt}"]
        eff.append([pos, f"{src.split('-')[0]} → {tgt.split('-')[0]}",
                    f"{other['auroc']['value']:.3f} vs {own['auroc']['value']:.3f} on its own domain",
                    f"{a['frozen']['recall']['value']:.3f} / {100 * a['frozen']['fpr']['value']:.1f}% frozen; "
                    f"{a['recalibrated']['recall']['value']:.3f} / {100 * a['recalibrated']['fpr']['value']:.1f}% recalibrated; "
                    f"{tgt_own['recall']['value']:.3f} target-trained",
                    "; ".join(f"{k.replace('Cx', 'C×')}: {u['recall']['mean']:.3f} / {100 * u['fpr']['mean']:.1f}% / {u['auroc']['mean']:.3f}"
                              for k, u in a.get("updated_over_draws", {}).items()) or "not run",
                    f"{pb[f'train {src} / test {tgt}']['auroc']['value']:.3f} vs {pb[f'train {src} / test {src}']['auroc']['value']:.3f}"])
    md += ["## pre_call and its content-token control: ranking, calibration, predictability", "",
           table(eff, ["readout", "direction", "ranking: AUROC on target vs own", "calibration: recall / FPR on target",
                       f"update with {F['config'].get('n_shots', 32)} target calls: recall / FPR / AUROC, mean over draws, by prior strength",
                       "predictability: previous-calls AUROC on target vs own"]), ""]

    cat_rows = []
    for pos in ("call_end", "pre_call", "prev_content"):
        for cat, rep in sorted(P["positions"][pos]["by_category"].items(), key=lambda kv: (kv[0] in READ_ONLY, kv[0])):
            ms = [(k, cls) for k, cls in (("recall", "impactful"), ("fpr", "read_only"))  # by the calls' labels, so a
                  if any(rep[f"train {tr} / test {te}"]["n"][cls] for tr, te in CELLS)]  # mixed category gets both
            for k, cls in ms:
                vals = [f"{rep[f'train {tr} / test {te}'][k]['value']:.3f} (n={rep[f'train {tr} / test {te}']['n'][cls]})"
                        for tr, te in CELLS]
                cat_rows.append([pos, cat, k] + vals)
    md += ["## Per category (recall over its impactful calls, false-positive rate over its read-only calls)", "",
           table(cat_rows, ["readout", "category", "metric"] + [f"{tr.split('-')[0]} → {te.split('-')[0]}" for tr, te in CELLS]), ""]

    nulls = [[pos, cell, f"{m['label_shuffle_null']['max']:.3f}", f"{m['auroc']['value']:.3f}"]
             for pos in ("call_end", "pre_call", "prev_content") for cell, m in P["positions"][pos]["all_calls"].items()
             if "label_shuffle_null" in m]
    md += ["## Label-shuffle null (labels permuted within transcripts; selection repeated)", "",
           table(nulls, ["readout", "cell", "null max AUROC", "observed AUROC"]), ""]

    if threeway_json:
        T = json.loads(Path(threeway_json).read_text()) | {"_file": Path(threeway_json).name}
        md += threeway_section(T, sha, P["config"])
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    path = HERE / "results" / f"q2_final_table_{stamp}_{secrets.token_hex(3)}.md"
    with open(path, "x") as f:  # results/ is append-only: never overwrite a table
        f.write("\n".join(md))
    print(f"labeler {sha} -> {path}")


if __name__ == "__main__":
    main(*sys.argv[1:5])
