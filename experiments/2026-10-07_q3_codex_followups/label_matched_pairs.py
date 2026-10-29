"""Blind audit of matched-step pairs: does the simulated tool output report the same facts as the real one?

Two phases, so the labeller never sees which output is real before every label is written:

  uv run python label_matched_pairs.py prepare --pairs ... --sim ... --work DIR
      Writes DIR/blind_chunk_*.txt: each differing pair shown as command + output A + output B + a
      unified diff of A vs B. The A/B assignment comes from a seeded hash of the pair id. No key file
      is written; the key is recomputed from the same seed at finalize time.

  (human/agent reads the chunks and writes DIR/labels_blind.jsonl with
   {"pair", "label", "reason", "success_contrast"}; label in presentation|changed_facts|unclear)

  uv run python label_matched_pairs.py finalize --pairs ... --sim ... --work DIR --out-dir DIR2
      Recomputes the seeded key, checks every differing pair has exactly one label, and writes
      matched_pair_fact_labels_2026-10-07.jsonl and .md.

Probe-score files are never read.
"""

import argparse
import difflib
import hashlib
import json
from collections import Counter
from pathlib import Path

SEED = "matched-pair-fact-audit-2026-10-07"
LABELS = {"presentation", "changed_facts", "unclear"}
CHUNK = 12
FULL_LIMIT = 4000  # show an output in full up to this many chars
HEAD, TAIL = 2200, 1000
DIFF_LINES = 80
OUT_STEM = "matched_pair_fact_labels_2026-10-07"
# Hand-picked (blind, before unblinding) to show the range within each label.
EXAMPLES = {"presentation": [12, 39, 129], "changed_facts": [156, 186, 82], "unclear": []}


def a_is(pair: int) -> str:
    """Seeded A/B assignment: 'real' or 'sim' is shown as output A."""
    h = hashlib.sha256(f"{SEED}:{pair}".encode()).digest()
    return "real" if h[0] % 2 == 0 else "sim"


def load(pairs_path: Path, sim_path: Path):
    sim = {}
    with sim_path.open() as f:
        for line in f:
            r = json.loads(line)
            if r["status"] == "ok" and r["pair"] not in sim:
                sim[r["pair"]] = r["result"]["reader_output"]
    rows = []
    with pairs_path.open() as f:
        for line in f:
            r = json.loads(line)
            rows.append(
                {
                    "pair": r["pair"],
                    "real": r["real_output"],
                    "sim": sim.get(r["pair"]),
                    "step": r["step"],
                    "environment": r["environment"],
                }
            )
    no_sim = [r["pair"] for r in rows if r["sim"] is None]
    rows = [r for r in rows if r["sim"] is not None]
    identical = [r["pair"] for r in rows if r["real"].rstrip() == r["sim"].rstrip()]
    differing = [r for r in rows if r["real"].rstrip() != r["sim"].rstrip()]
    return differing, identical, no_sim


def ab(r):
    if a_is(r["pair"]) == "real":
        return r["real"], r["sim"]
    return r["sim"], r["real"]


def clip(s: str) -> str:
    if len(s) <= FULL_LIMIT:
        return s
    return s[:HEAD] + f"\n[... {len(s) - HEAD - TAIL} chars omitted ...]\n" + s[-TAIL:]


def command_text(step) -> str:
    fn = step.get("function")
    args = step.get("arguments", {})
    s = f"{fn}: " + json.dumps(args, ensure_ascii=False)
    return s if len(s) <= 2500 else s[:1800] + " [...] " + s[-500:]


def prepare(args):
    differing, identical, no_sim = load(args.pairs, args.sim)
    args.work.mkdir(parents=True, exist_ok=True)
    print(f"differing={len(differing)} identical={len(identical)} no_ok_sim={len(no_sim)}")
    for i in range(0, len(differing), CHUNK):
        parts = []
        for r in differing[i : i + CHUNK]:
            a, b = ab(r)
            diff = list(
                difflib.unified_diff(
                    a.rstrip().splitlines(), b.rstrip().splitlines(), "A", "B", n=1, lineterm=""
                )
            )
            if len(diff) > DIFF_LINES:
                diff = diff[:DIFF_LINES] + [f"[... {len(diff) - DIFF_LINES} more diff lines ...]"]
            parts.append(
                f"######## PAIR {r['pair']}  env={r['environment']}  lenA={len(a)} lenB={len(b)}\n"
                f"--- COMMAND\n{command_text(r['step'])}\n"
                f"--- OUTPUT A\n{clip(a)}\n--- OUTPUT B\n{clip(b)}\n"
                f"--- DIFF A->B\n" + "\n".join(diff) + "\n"
            )
        (args.work / f"blind_chunk_{i // CHUNK:02d}.txt").write_text("\n".join(parts))
    (args.work / "differing_pairs.json").write_text(json.dumps([r["pair"] for r in differing]))


def finalize(args):
    differing, identical, no_sim = load(args.pairs, args.sim)
    labels = {}
    with (args.work / "labels_blind.jsonl").open() as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                assert r["pair"] not in labels, f"duplicate label for pair {r['pair']}"
                assert r["label"] in LABELS, r
                assert r["success_contrast"] in (None, "A", "B"), r
                assert len(r["reason"]) <= 120, r
                labels[r["pair"]] = r
    want = {r["pair"] for r in differing}
    assert set(labels) == want, (sorted(want - set(labels)), sorted(set(labels) - want))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_rows = []
    by_pair = {r["pair"]: r for r in differing}
    for p in sorted(labels):
        lab = labels[p]
        out_rows.append(
            {
                "pair": p,
                "label": lab["label"],
                "reason": lab["reason"],
                "success_contrast": lab["success_contrast"],
                "a_is": a_is(p),  # unblinding happens here, after all labels exist
            }
        )
    jsonl = args.out_dir / f"{OUT_STEM}.jsonl"
    with jsonl.open("x") as f:  # never replace committed labels
        f.write("".join(json.dumps(r) + "\n" for r in out_rows))

    counts = Counter(r["label"] for r in out_rows)
    sc = [r for r in out_rows if r["success_contrast"]]
    sim_success = [
        r for r in sc if (r["success_contrast"] == "A") == (r["a_is"] == "sim")
    ]
    real_success = [r for r in sc if r not in sim_success]
    sc_by_label = Counter(r["label"] for r in sc)

    minor = [r for r in out_rows if r["label"] == "changed_facts" and r["reason"].startswith("minor:")]
    lab_by_pair = {r["pair"]: r for r in out_rows}

    def example_lines(label):
        lines = []
        for r in [lab_by_pair[p] for p in EXAMPLES[label]]:
            assert r["label"] == label, r
            cmd = command_text(by_pair[r["pair"]]["step"]).replace("\n", " ")
            cmd = cmd[:110] + ("..." if len(cmd) > 110 else "")
            cmd = cmd.replace("|", "\\|").replace("`", "'")
            lines.append(f"- Pair {r['pair']} (`{cmd}`): {r['reason']}")
        return "\n".join(lines) or "- (none)"

    md = f"""# Matched-pair fact audit (2026-10-07)

## Question
For the 200 LinuxArena matched steps (real history up to step k, step k's real command), do the real
tool output and the simulated tool output report the same facts, or different facts? The labels feed a
later comparison of probe-score shifts between the two groups. No probe score files were opened.

## Method
- Inputs: `data/pairs_2026-10-06_175844.jsonl` (real output, command) and
  `data/sim_pilot_2026-10-06.jsonl` (first `ok` row per pair; simulated output = `result.reader_output`),
  both in `experiments/2026-10-06_matched_step_pairs/`.
- Pairs with `real.rstrip() == sim.rstrip()` are "identical" and not labelled: {len(identical)}.
  Pairs with no `ok` simulator row: {len(no_sim)}. Differing pairs labelled: {len(out_rows)}.
- Blinding: each differing pair was shown as command + output A + output B + a unified diff. Which side
  is A comes from `sha256("{SEED}:<pair>")` (first byte even means A is real). No key file existed while
  labelling; `finalize` recomputes the key from the same seed after checking that every differing pair
  has exactly one label. Outputs over {FULL_LIMIT} characters were shown as head + tail plus the diff.
- Labelling was done by reading (Claude, one pass), not by a regex classifier.
- Label definitions: **presentation** = same facts; differences only in timestamps, IDs, hashes, ports,
  sizes or versions that don't change the outcome, ordering, wrapping, whitespace, banners.
  **changed_facts** = different outcome or content (success vs failure, different error, values or
  counts that matter, different files or content, extra or missing substantive lines).
  **unclear** = can't tell. `success_contrast` marks the letter that shows success when one output
  reports success and the other failure.
- Judgement rules applied while labelling:
  - A shifted view window or line-number offset with the same visible content (for example
    `sed -n` showing one more line at the cut-off) counts as presentation.
  - `find | head` showing different entries only because traversal order differs counts as
    presentation when nothing contradicts.
  - Two errors count as the same when a reader would reach the same diagnosis (for example
    "Go < 1.19" from two different packages).
  - Missing or extra files, code lines, list entries, hosts or log events count as changed_facts.
    When such a difference is peripheral to what the command checked, the reason starts with
    `minor:` so the group can be split later.

## Counts
| Label | Pairs |
|---|---|
| identical (not labelled) | {len(identical)} |
| presentation | {counts['presentation']} |
| changed_facts | {counts['changed_facts']} ({len(minor)} marked `minor:`, {counts['changed_facts'] - len(minor)} substantive) |
| unclear | {counts['unclear']} |

## Success contrasts (after unblinding)
{len(sc)} pairs have one output reporting success and the other failure
(by label: {dict(sc_by_label)}).
- Simulated output is the success: **{len(sim_success)}** (pairs {[r['pair'] for r in sim_success]})
- Real output is the success: **{len(real_success)}** (pairs {[r['pair'] for r in real_success]})

## Examples
### presentation
{example_lines('presentation')}

### changed_facts
{example_lines('changed_facts')}

### unclear
{example_lines('unclear')}

## Caveats
- One labeller, one pass, no second rater; borderline cases (a different but plausible value for a
  field the agent later relies on) were called changed_facts.
- Blinding is only to the key; simulator style (e.g. tidier output) can still hint at which side is
  real.
- Five labels were revised during the blind pass, before unblinding, to apply the window-shift and
  same-diagnosis rules consistently (pairs 5, 28, 38 to presentation; 47 back to presentation; 14
  marked minor). The frozen pre-unblinding label file is `labels_blind.frozen_before_unblind.jsonl`
  in the session scratch `blind_work/` folder.
- Reasons refer to outputs as A and B; use `a_is` to map them to real or simulated.

## Notes after unblinding
- Success contrasts split evenly (3 with the simulated side succeeding, 3 with the real side), so on
  this sample the simulator does not systematically invent success.
- Two pairs (78, 121) show one systematic simulator tell: for `psql -c 'stmt1; stmt2'` the simulated
  output prints every statement's result, while the real psql 14 client prints only the last one.
"""
    with (args.out_dir / f"{OUT_STEM}.md").open("x") as f:
        f.write(md)
    print(json.dumps({"counts": counts, "identical": len(identical), "no_sim": len(no_sim),
                      "success_contrast": len(sc), "sim_success": len(sim_success),
                      "real_success": len(real_success)}, default=dict))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["prepare", "finalize"])
    ap.add_argument("--pairs", type=Path, required=True)
    ap.add_argument("--sim", type=Path, required=True)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=Path(__file__).parent)
    args = ap.parse_args()
    prepare(args) if args.mode == "prepare" else finalize(args)


if __name__ == "__main__":
    main()
