"""Q3: off-policy linear probes (trained outside LinuxArena) read on the segment-end activations.

- `call_table`: one row per tool call, indexed as `reader_messages` and `reader.segment_end_positions` index them,
  with what the probe's readout depends on (the call's last argument) and what the slices need (its output's error).
- `load_probe` / `probe_scores`: Apollo-style probes saved as npz (coef, scaler_mean, scaler_scale); the score of an
  activation x is ((x - scaler_mean) / scaler_scale) @ coef, no intercept (Apollo's LogisticRegressionDetector).
- `fit_token_probe`: that recipe's fit, per-token logistic regression on standardized activations.
"""

import io


def call_table(sample: dict) -> list[dict]:
    """Per tool call the reader sees: step (kept agent turn, 0-based), action (k-th non-submit call of the turn,
    answered by the k-th tool output after it), tool, last_arg (name of the call's last argument: its value's last
    token is the call_end readout), editor_command and path (text_editor calls only), and error (the Inspect error type of its output, None if it ran cleanly,
    "no_output" if no output answers it). Raises if the k-th output's tool does not match the k-th call's."""
    from probe_shift.transcripts import RUNNER_SUBMIT, visible_text  # needs boto3; GPU/CPU images import the rest

    calls, outputs, step, action, submit_ids = [], {}, -1, 0, set()
    for m in sample["messages"]:
        if m["role"] == "assistant":
            tcs = m.get("tool_calls") or []
            submit_ids |= {tc.get("id") for tc in tcs if tc["function"] == RUNNER_SUBMIT}
            kept = [tc for tc in tcs if tc["function"] != RUNNER_SUBMIT]
            if kept or visible_text(m["content"]).strip():  # reader_messages keeps exactly these turns
                step, action = step + 1, 0
                for k, tc in enumerate(kept):
                    args = tc.get("arguments") or {}
                    calls.append({"step": step, "action": k, "tool": tc["function"], "last_arg": list(args)[-1] if args else None,
                                  "editor_command": args.get("command") if tc["function"] == "text_editor" else None,
                                  "path": args.get("path") if tc["function"] == "text_editor" else None})
        elif m["role"] == "tool":
            if m.get("tool_call_id") in submit_ids or m.get("function") == RUNNER_SUBMIT:
                continue
            error = m.get("error")
            outputs[(step, action)] = (m.get("function"), (error.get("type") or "unknown") if error else None)
            action += 1
    rows = []
    for c in calls:
        tool, error = outputs.get((c["step"], c["action"]), (c["tool"], "no_output"))
        if tool != c["tool"]:
            raise ValueError(f"step {c['step']} action {c['action']}: call {c['tool']} answered by a {tool} output")
        rows.append(c | {"error": error})
    return rows


def provenance(*paths) -> dict:
    """Git commit, whether the tree has uncommitted changes, and the sha256 of each given source file: written into
    results so a run can be tied to the exact code (cache fingerprints cover inputs and settings, not code)."""
    import hashlib
    import subprocess
    from pathlib import Path

    here = Path(paths[0]).parent if paths else Path.cwd()
    git = lambda *a: subprocess.run(["git", "-C", str(here), *a], capture_output=True, text=True).stdout.strip()
    return {"git_commit": git("rev-parse", "HEAD"), "git_dirty": bool(git("status", "--porcelain")),
            "source_sha256": {Path(p).name: hashlib.sha256(Path(p).read_bytes()).hexdigest()[:16] for p in paths}}


def load_probe(raw: bytes) -> dict:
    """An npz probe (coef, scaler_mean, scaler_scale) as float32 arrays."""
    import numpy as np

    z = np.load(io.BytesIO(raw))
    return {k: z[k].astype(np.float32).reshape(-1) for k in ("coef", "scaler_mean", "scaler_scale")}


def probe_scores(x, probe: dict):
    """Score rows of x [n, H] (any float dtype) with an Apollo-style probe."""
    import numpy as np

    return ((x.astype(np.float32) - probe["scaler_mean"]) / probe["scaler_scale"]) @ probe["coef"]


def fit_token_probe(x, y, reg_coeff: float, max_iter: int, seed: int = 42) -> tuple[dict, bool]:
    """Apollo's recipe: standardize per-token activations x [n, H], then L2 logistic regression with no intercept,
    C = 1 / reg_coeff. Returns the probe (as `load_probe` gives it) and whether the solver converged."""
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(x)
    lr = LogisticRegression(C=1 / reg_coeff, fit_intercept=False, random_state=seed, max_iter=max_iter)
    lr.fit(scaler.transform(x), y)
    probe = {"coef": lr.coef_[0].astype(np.float32), "scaler_mean": scaler.mean_.astype(np.float32),
             "scaler_scale": scaler.scale_.astype(np.float32)}
    return probe, bool(lr.n_iter_[0] < max_iter)
