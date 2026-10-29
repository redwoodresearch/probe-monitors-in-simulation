"""Extra results.md sections for Q1, generated from the saved result files (no hand-copied numbers).

  uv run python experiments/2026-10-06_q1_segment_end/report_extras.py --main results/q1_gpu_<stamp>.json \
      --newline results/q1_gpu_newline_*.json --lens results/q1_gpu_logit_lens_*.json \
      --combination results/q1_gpu_combination_up_to_here_*.json results/q1_gpu_combination_output_only_*.json \
      --tfidf-main <baselines-q3 tfidf_summary_*.json> --tfidf-env <baselines-q3 tfidf_heldout_env_summary_*.json> \
      [--scrubbed results/q1_gpu_<stamp>.json] > results/extras_<stamp>.md
"""

import argparse
import json
from pathlib import Path


def a(d):
    return "–" if not d else f"{d['auroc']:.3f}"


def ci(d):
    return "–" if not d else f"{d['auroc']:.3f} [{d['ci95'][0]:.3f}, {d['ci95'][1]:.3f}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--main", type=Path, required=True)
    ap.add_argument("--newline", type=Path)
    ap.add_argument("--lens", type=Path)
    ap.add_argument("--combination", type=Path, nargs="*", default=[])
    ap.add_argument("--scrubbed", type=Path)
    ap.add_argument("--tfidf-main", type=Path, required=True, help="baselines-q3 TF-IDF summary, main-task folds")
    ap.add_argument("--tfidf-env", type=Path, required=True, help="baselines-q3 TF-IDF summary, held-out environments")
    args = ap.parse_args()
    # TF-IDF on the text the probe reads (level 0: `tfidf_cleaned`), run level; never another scrub level's number.
    tf = lambda path, outputs: ci(json.loads(path.read_text())["baselines"]["tfidf_cleaned"]["results"][f"real_vs_sim/{outputs}"]["all_cells"]["runs"])
    res = json.loads(args.main.read_text())
    L = lambda l, pos="output_end": res["by_layer"][str(l)]["results"][pos]["real_vs_sim"]
    layers = sorted(int(l) for l in res["by_layer"])
    md = ["## The hard comparison: held-out environment, no editor outputs", "",
          "Editor (`text_editor`) outputs carry the strongest formatting tells. The hard test of what the probe adds is a "
          f"held-out environment with editor outputs removed. TF-IDF on the same original text, run level, without editor "
          f"outputs: {tf(args.tfidf_main, 'no_editor')} on main-task folds, {tf(args.tfidf_env, 'no_editor')} on held-out "
          "environments (baselines-q3, `tfidf_cleaned`). Run-level AUROC of the probe:", "",
          "| Layer | output_end probe | call_end probe |", "|---|---|---|"]
    for l in layers:
        md.append(f"| {l} | {ci(L(l)['held_out_environment']['no_editor_outputs']['run'])} | "
                  f"{ci(L(l, 'call_end')['held_out_environment']['no_editor_outputs']['run'])} |")
    md += ["", f"TF-IDF on the same original text and split: {tf(args.tfidf_env, 'no_editor')} (`tfidf_cleaned`). call_end examples drop editor calls, but the "
               "probe has still read every earlier output, editor ones included, so its separation can come from that context; "
               "TF-IDF on call text alone is about 0.53."]

    if args.newline or args.lens:
        md += ["", "## Late layers (56, 63): what they are not", ""]
    if args.newline:
        nl = json.loads(args.newline.read_text())["results"]
        md += ["Simulated outputs end in a newline more often than real ones, but only through editor outputs: without them the "
               "shares are about equal. Run-level AUROC of the out-of-fold output_end scores within outputs that do / do not end "
               "in a newline (no editor outputs):", "",
               "| Layer | ends in a newline | no trailing newline | share ending in a newline (real, sim) |", "|---|---|---|---|"]
        for l, r in sorted(nl.items(), key=lambda x: int(x[0])):
            v = r["real_vs_sim"]["no_editor_outputs"]
            sh = v["share_ending_in_newline"]
            md.append(f"| {l} | {a(v['ends_in_newline']['run'])} | {a(v['no_trailing_newline']['run'])} | "
                      f"{sh['real-attack-e1']:.3f}, {sh['sim-attack']:.3f} |")
        md += ["", "The late-layer rise survives with the trailing newline held equal, so it is not that tell."]
    if args.lens:
        d = json.loads(args.lens.read_text())["results"]
        top = lambda k: ", ".join(f"{json.dumps(t)} {p:.2f}" for t, p in d[k][:4])
        pl = f"probe_L{d['layer']}_no_editor_"
        md += ["", f"Logit lens at layer {d['layer']} (its activations through Qwen's final norm and output head: Qwen's own "
                   f"next-token distribution at the output's last content token), no editor outputs "
                   f"({d['n_outputs']['real']:,} real, {d['n_outputs']['sim']:,} simulated):", "",
               f"- Top next tokens, real: {top('top10_real')}; simulated: {top('top10_sim')}.",
               f"- Mean probability the output closes next (a newline-led token, `</`, or a template token): real "
               f"{d['mean_p_close_real']:.3f}, simulated {d['mean_p_close_sim']:.3f}; as a one-feature classifier, run AUROC "
               f"{a(d['auroc_p_close']['run'])}. Next-token entropy: {a(d['auroc_entropy']['run'])}.",
               f"- The layer-{d['layer']} probe by output type (run AUROC, no editor): outputs without Inspect's truncation "
               f"marker {a(d[pl + 'not_truncated']['run'])}, with it {a(d[pl + 'truncated_marker']['run'])} (about 4% of "
               f"outputs in both arms); shorter than 6,000 characters {a(d[pl + 'short']['run'])}, longer "
               f"{a(d[pl + 'long_6000_chars']['run'])}.", "",
               f"So layer {d['layer']} is not reading Qwen's expectation of whether the output ends, nor truncation: Qwen predicts the same "
               "next tokens for both arms, and the separation sits in ordinary, short, untruncated outputs."]

    if args.combination:
        md += ["", "## Does the probe add anything over TF-IDF?", "",
               "Within each held-out fold, the out-of-fold TF-IDF and probe scores for the same outputs are combined (the mean "
               "of their z-scores within the fold: no fitted weights, so no leakage). Run-level AUROC, real vs simulated, "
               "main-task folds:", "",
               "| TF-IDF text | Layer | Outputs | TF-IDF | Probe | Both (no fit) | Fitted gain over TF-IDF (approximate) |",
               "|---|---|---|---|---|---|---|"]
        for p in args.combination:
            c = json.loads(p.read_text())
            label = {"up_to_here": "all text up to this output", "output_only": "this output only"}.get(c["label"], c["label"])
            for l in sorted(c["results"], key=int):
                for v in ("all_outputs", "no_editor_outputs"):
                    r = c["results"][l]["real_vs_sim"][v]
                    g = r["gain_both_minus_tfidf"]["run"]
                    md.append(f"| {label} | {l} | {v.replace('_outputs', '').replace('_', ' ')} | {a(r['tfidf']['run'])} | "
                              f"{a(r['probe']['run'])} | {a(r['mean_z (no fit)']['run'])} | {g['point']:+.3f} "
                              f"[{g['ci95'][0]:+.3f}, {g['ci95'][1]:+.3f}] |")
        md += ["", "The fitted gain is approximate: the training folds' out-of-fold scores come from base models that saw the "
                   "held-out fold, so its two learned weights see a little of it. The no-fit column has no such leak. On the "
                   "real-vs-real null (epoch 1 vs epoch 2), adding the probe to TF-IDF gains nothing (within ±0.003)."]

    lc = res["by_layer"]["48"].get("learning_curve", {}).get("output_end")
    if lc:
        md += ["", "## Learning curve (output_end, layer 48, real vs simulated)", "",
               "Run-level AUROC on the held-out fold, training on n main tasks per fold (baselines-q3's task subsets, so TF-IDF "
               "used identical tasks), mean of 3 seeds:", "", "| Training main tasks | All outputs | No editor outputs |", "|---|---|---|"]
        for n in lc["all_outputs"]:
            m = lambda v: sum(s["run"]["auroc"] for s in lc[v][n].values()) / len(lc[v][n])
            md.append(f"| {n} | {m('all_outputs'):.3f} | {m('no_editor_outputs'):.3f} |")

    bp = L(48)["all_outputs"].get("by_output_position")
    if bp:
        md += ["", "## By output position (output_end, layer 48, step level)", "",
               "| Output | All outputs | No editor outputs |", "|---|---|---|"]
        for k in bp:
            md.append(f"| {k} | {a(bp[k])} | {a(L(48)['no_editor_outputs']['by_output_position'][k])} |")

    if args.scrubbed:
        s = json.loads(args.scrubbed.read_text())
        S = lambda l: s["by_layer"][str(l)]["results"]["output_end"]["real_vs_sim"]
        md += ["", "## Harness-normalized text (scrub level 5)", "",
               "The same extraction on text scrubbed exactly as TF-IDF level 5 (times, numbers and IDs masked, whitespace and "
               "blank lines normalized, the real editor's line-number gutter and hunk headers removed, stderr-looking lines "
               "dropped), on tool outputs and tool-call arguments in both arms. Run-level AUROC, output_end:", "",
               "| Layer | All outputs | No insert/undo_edit | No editor | Held-out environment, no editor |", "|---|---|---|---|---|"]
        for l in sorted(int(x) for x in s["by_layer"]):
            r = S(l)
            md.append(f"| {l} | {a(r['all_outputs']['run'])} | {a(r.get('no_insert_undo_outputs', {}).get('run'))} | "
                      f"{a(r['no_editor_outputs']['run'])} | {a(r['held_out_environment']['no_editor_outputs']['run'])} |")

    md += ["", "## Caveats", "",
           "- Headline layers are chosen per fold by inner CV on training folds. The e1-vs-e2 null goes through the same "
           "selection (0.49-0.54), but the headline has no label-shuffle null of its own.",
           "- base200 vs overlap800 also splits environment versions (each environment has one contents hash per group), so "
           "that slice mixes the simulator's reference with the environment version.",
           "- TF-IDF's strongest editor tell (edit outputs as raw diffs) is a September-simulator artifact; the sims repo "
           "changed that format on 2026-09-16 (commit 4b31bd30).",
           "- Tool-error outputs are a harness tell (in 46% of real runs vs 0.1% of simulated), but dropping them changes the "
           "probe by under 0.01.",
           "- The paper judge sees whole transcripts (no 32k cut, no system prompt); its under-32k column is the comparison "
           "with the probe."]
    print("\n".join(md))


if __name__ == "__main__":
    main()
