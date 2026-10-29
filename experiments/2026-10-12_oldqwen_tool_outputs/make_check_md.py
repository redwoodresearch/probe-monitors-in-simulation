"""Write CHECK.md and results_manifest_v1.json from the check's result JSON (numbers read by key; no transcript text).
  uv run --no-project --python 3.12 python experiments/2026-10-12_oldqwen_tool_outputs/make_check_md.py \\
      data/oldqwen_tool_outputs_<stamp>.json data/excerpt_<stamp>.txt
"""
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
BUCKET = "individual-aswin-surya-probe-shift-kill-2027-04-06"
res_p, exc_p = (HERE / a for a in sys.argv[1:3])
d = json.loads(res_p.read_text())
c, run, x, agree = d["counts"], d["run"], d["excerpt_output"], d["rerender_matches_meta"]
pct = lambda n: f"{100 * n / c['tool_outputs']:.1f}%"
assert c["inside_cut"] + c["beyond_cut"] + c["empty_no_content_token"] == c["tool_outputs"]
if hashlib.sha256(exc_p.read_bytes()).hexdigest() != d["excerpt_sha256"]:
    raise SystemExit(f"{exc_p.name} does not match the result's excerpt_sha256")
md = f"""# Did tool outputs reach Qwen3-32B's input? (James's old-Qwen setup check, 10-12)

**Yes.** Source: the v4 Qwen3-32B extraction's own records on the Volume: its staged inputs (`{d['staged_inputs']}`) and its
per-run meta (`{d['dump']}/meta/{d['arm']}`). Script `run_check_modal.py` (one Modal CPU container), result
`data/{res_p.name}` on S3 (`results_manifest_v1.json`; it is kept out of git because it holds one token of transcript text).
Honest runs only.

**Counts over all {c['runs']:,} real-honest runs** (from the meta: what the extraction read, cut at {d['max_tokens']:,} tokens):
- Tool outputs: {c['tool_outputs']:,}.
- Inside the input (last content token before the cut): **{c['inside_cut']:,} ({pct(c['inside_cut'])})**.
- Beyond the cut: {c['beyond_cut']:,} ({pct(c['beyond_cut'])}), all in the {c['runs_truncated']} runs longer than {d['max_tokens']:,} tokens.
- Empty (no content token): {c['empty_no_content_token']:,} ({pct(c['empty_no_content_token'])}).
- Runs with at least one tool output inside the input: {c['runs_with_any_tool_output_inside_cut']:,} of {c['runs']:,}.

**One example** (honest runs in the order `random.Random({d['seed']}).shuffle` gives the sorted keys; the first with a kept,
non-empty bash `ls` output; {len(d['runs_tried'])} runs tried). Run `{run['key']}` ({run['environment']}, main task
`{run['main_task']}`), {run['n_tokens']:,} tokens, {'cut' if run['truncated'] else 'not cut'}.
- Step {x['step']}, action {x['action']}, {x['selected_as']}. The call's command starts with `ls`; at {x['content_lines']} lines it may
  also list subdirectories or do more than `ls`. The excerpt's first 8 content lines are a plain directory listing.
- `<tool_response>` at token {x['tool_response_token']:,}; content tokens {x['content_tokens'][0]:,}–{x['content_tokens'][1] - 1:,}
  ({x['n_content_tokens']:,} tokens, {x['content_lines']} lines); `</tool_response>` at token {x['close_token']:,}.
- The probe's output_end readout is token {x['readout_output_end_token']:,}, the output's last content token, inside the input.
- Re-rendering the staged messages with the extraction's own code (`probe_shift.reader.segment_end_positions`, the
  Qwen3-32B tokenizer at its pinned revision) {'matched the meta exactly: token count, truncation and every output_end position' if all(agree.values()) else 'did NOT match the meta: ' + json.dumps(agree)}.
- The excerpt (first 8 content lines, as rendered into the model input) is on S3 only: `data/{exc_p.name}`, sha256
  {hashlib.sha256(exc_p.read_bytes()).hexdigest()[:16]}…. The call's command text was not returned.

**Rerun note.** The result came from `run_check_modal.py` as committed at fe8ed9b6 (its sha256 is in the result JSON), so
the script stays as it ran. It writes the result JSON to `results/`, which it does not create (make the folder first),
and names its files with a UTC stamp (`...Z`), not Pacific Time. After the run, the result JSON was moved to the
git-ignored `data/` and uploaded to S3, because it holds one token of transcript text.
"""
files = {}
for p in (res_p, exc_p):
    key = f"experiments/2026-10-12_oldqwen_tool_outputs/data/{p.name}"
    files[f"data/{p.name}"] = {"s3": f"s3://{BUCKET}/{key}", "sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "bytes": p.stat().st_size}
manifest = json.dumps({"bucket": BUCKET, "note": "The old-Qwen check's result JSON and its 8-line ls excerpt (transcript text: S3 only, never git). Uploaded from the laptop with --if-none-match; sha256 checked after download.", "files": files}, indent=1) + "\n"
mp = HERE / "results_manifest_v1.json"
if mp.exists() and mp.read_bytes() != manifest.encode():  # manifests are append-only: a new run gets a new version, never an overwrite
    raise SystemExit(f"{mp.name} exists with different contents; write results_manifest_v<N+1>.json for a new run")
(HERE / "CHECK.md").write_text(md)
mp.write_text(manifest)
print(md)
