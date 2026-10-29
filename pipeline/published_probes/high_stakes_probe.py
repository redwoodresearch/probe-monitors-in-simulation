# /// script
# requires-python = "==3.12.*"
# dependencies = ["numpy==2.3.3", "safetensors==0.8.0", "scikit-learn==1.7.2", "joblib", "torch==2.8.0", "transformers==5.19.0",
#                 "accelerate>=1.1.0", "flash-linear-attention==0.5.2", "jinja2"]
# ///
"""The high-stakes probe: how we trained it for Qwen3.5-27B, and how it scores a unit.

Source. McKenzie et al. (2025), "Detecting High-Stakes Interactions with Activation Probes", arXiv 2506.10805. The paper does
not publish a probe for Qwen3.5-27B, so we trained one with its mean-pooled linear method on its public data: Hugging Face
dataset Arrrlex/models-under-pressure at revision 940d830f639617752f54fdf2bca0da7e0d856a9e (MIT license). We use
training/train.jsonl (8,000 conversations written and labelled by GPT-4o) to train, training/test.jsonl (2,000) to test, and
anthropic_hh_balanced/test.jsonl (2,984 relabelled Anthropic helpful-and-harmless conversations) as a second, descriptive test.
Only the labels "high-stakes" (1) and "low-stakes" (0) occur. We do not redistribute the data: this script downloads it and checks
each file's sha256. The probe is ours, and we ship the exact file the post used in weights/published/high_stakes_probe.safetensors.

Reading. A training or test conversation is one user turn; a helpful-and-harmless one is its list of messages. Each is rendered
with Qwen3.5-27B's chat template and no generation prompt, and read once (bfloat16, attention "sdpa"). Its content tokens are the
tokens that lie inside a message's (trimmed) text and are not special tokens; everything else is chat-format text. An item is
left out if our segment-by-segment rendering differs from the template's (no item was). At every layer we keep the mean of the
activations over the content tokens (averaged in float32, then rounded to bfloat16, stored as float16). Layer L is the output of
decoder block L (Hugging Face's hidden_states[L + 1]). We read on two H200 GPUs, each taking every other item; both also read the
same 16 test items, which agreed to cosine 1.000000 at every layer.

Fit. Logistic regression (scikit-learn 1.7.2: L2 penalty, C = 1, lbfgs, at most 2,000 iterations, tolerance 1e-6) on rows
standardized by the training rows' mean and standard deviation (plus 1e-6). To choose the layer, random.Random(20261014) holds out
20% of each label of the training set (1,600 items); we fit on the other 6,400 at each of the 64 layers and take the layer with the
highest held-out AUROC: layer 43 (0.999). We then refit at layer 43 on all 8,000 training items. Test AUROC 0.999; on the
helpful-and-harmless test 0.886. Our transcripts never enter the choice. The refit has converged: the largest component of the
gradient of scikit-learn's objective at the saved weights is 8.7e-7, below our bar of 1e-6.

Scoring a unit. The file holds the fit (W, intercept, mean, std) and its folded form: direction r = W / std and bias
c = intercept - sum(mean * W / std), so that score = x_43 . r + c, the logistic regression's log-odds that the situation is high
stakes. The product runs in float32 and the result is kept as float64. Higher means higher stakes, so more suspicious. The probe
was trained on means over whole messages; we apply it at our units: for a command, the activation at the last token inside the
tool call's main argument (a shell call's command; a file edit's new text, file_text for create and new_str for str_replace and
insert; a view's path); for a tool output, the mean over its content tokens, the closer match (Appendix A of the post; positions
defined in our internal src/probe_shift/reader.py, command_param and command_end_positions).

Original code (our internal repository, unchanged since it ran): experiments/2026-10-14_p2_highstakes/run_read_fit.py (run
2026-10-14 13:12 PT): the data at lines 36-53, the rendering at lines 56-73, the read at lines 85-159 (content tokens at lines
115-133, the mean at lines 137-148), the fit at lines 163-252 (held-out split at lines 213-219, layer choice at lines 231-240,
refit and folding at lines 241-247); the convergence check is run_convergence.py in the same folder; the per-unit score is
experiments/2026-10-14_p2_refusal_harm/run_score_linear.py lines 116-147, called by
experiments/2026-10-14_p2_simh_scores/run_simh_scores.py, the scorer behind every score in the post.

Run from the repository root on one GPU with at least 80 GB of memory (one pass over 12,984 conversations, about half an hour on
an H200; the model weights, about 54 GB, download on first use):
    uv run pipeline/published_probes/high_stakes_probe.py --out output/high_stakes_probe
reads the data, chooses the layer, refits, writes the probe and compares it with the copy we used.
"""

import argparse
import hashlib
import json
import random
import urllib.request
from pathlib import Path

import numpy as np
import safetensors.numpy

MODEL, REVISION = "Qwen/Qwen3.5-27B", "fc05daec18b0a78c049392ed2e771dde82bdf654"
SOURCE = "https://huggingface.co/datasets/Arrrlex/models-under-pressure/resolve/940d830f639617752f54fdf2bca0da7e0d856a9e/"
FILES = {"train": ("training/train.jsonl", "04216df3a8b3ebf3ffb617f1a6701b0429964d28a6c6df677896359891fe2f34"),
         "test": ("training/test.jsonl", "6af73f76248f46547cfbf346ba15cf0fafb5eac3986faae43de61c5e1660621f"),
         "hh": ("anthropic_hh_balanced/test.jsonl", "9ea4beff6de77023044373f44c67de1dbea09708e2af63d9dfcaa964998daf1d")}
LABEL = {"high-stakes": 1, "low-stakes": 0}
SEED, N_LAYERS, USED_LAYER = 20261014, 64, 43
SHIPPED = "weights/published/high_stakes_probe.safetensors"


def load_items() -> tuple:
    """Every labelled conversation of the three files in file order, as (split, label, messages); and counts of other labels."""
    items, other = [], {}
    for split, (path, want) in FILES.items():
        raw = urllib.request.urlopen(SOURCE + path, timeout=120).read()
        if hashlib.sha256(raw).hexdigest() != want:
            raise ValueError(f"{path}: sha256 differs from the pinned file")
        for line in raw.decode().splitlines():  # parsed as JSON data only
            row = json.loads(line)
            if row["labels"] not in LABEL:
                other[f"{split}|{row['labels']}"] = other.get(f"{split}|{row['labels']}", 0) + 1
                continue
            messages = json.loads(row["inputs"]) if split == "hh" else [{"role": "user", "content": row["inputs"]}]
            items.append((split, LABEL[row["labels"]], messages))
    return items, other


def segments(messages: list) -> list | None:
    """The chat template's rendering of a conversation as (text, is_content) pieces, written out piece by piece: role headers and
    end-of-turn markers are chat-format text, each message's trimmed text is content. None when the template would change a
    message (a think block inside an assistant message, non-text content, a system message after the first)."""
    if any(not isinstance(m.get("content"), str) or m.get("role") not in ("system", "user", "assistant") for m in messages):
        return None
    if any(m["role"] == "system" for m in messages[1:]) or not any(m["role"] == "user" for m in messages):
        return None
    last_user = max(j for j, m in enumerate(messages) if m["role"] == "user")
    seg = []
    for j, m in enumerate(messages):
        c = m["content"].strip()
        if m["role"] == "assistant" and "</think>" in c:
            return None
        head = f"<|im_start|>{m['role']}\n" + ("<think>\n\n</think>\n\n" if m["role"] == "assistant" and j > last_user else "")
        seg += [(head, False), (c, True), ("<|im_end|>\n", False)]
    return seg


def content_tokens(tok, messages: list) -> tuple:
    """(token ids, positions of the content tokens), or (None, reason) when the item is left out."""
    import jinja2

    seg = segments(messages)
    text = "".join(t for t, _ in seg) if seg else None
    try:
        official = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=False) if seg else None
    except jinja2.TemplateError:  # e.g. a conversation the template rejects
        seg, official = None, None
    if seg is None or text != official:
        return None, "template transform or error" if seg is None else "rendering differs from the template"
    spans, at = [], 0
    for t, is_content in seg:
        if is_content and t:
            spans.append((at, at + len(t)))
        at += len(t)
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    special = set(tok.all_special_ids)
    idx = [j for j, ((a, b), tid) in enumerate(zip(enc["offset_mapping"], enc["input_ids"], strict=True))
           if b > a and tid not in special and any(s <= a and b <= e for s, e in spans)]
    return (enc["input_ids"], idx) if idx else (None, "no content token")


def load_model() -> tuple:
    """(tokenizer, model): Qwen3.5-27B at the pinned revision, in bfloat16, on one GPU."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    lm = AutoModelForCausalLM.from_pretrained(MODEL, revision=REVISION, dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa").eval()
    return tok, lm


def read(items: list, tok, lm) -> tuple:
    """Each item's mean activation over its content tokens at all 64 layers: float16, shape (items, 64, 5120); and the items left
    out, {index: reason} (their rows stay zero)."""
    import torch

    rows, left_out = np.zeros((len(items), N_LAYERS, 5120), np.float16), {}
    for i, (_, _, messages) in enumerate(items):
        ids, idx = content_tokens(tok, messages)
        if ids is None:
            left_out[i] = idx
            continue
        got, ix = {}, torch.tensor(idx, device="cuda")

        def grab(layer, out, got=got, ix=ix):  # mean over content tokens in float32, rounded to the model's bfloat16
            h = (out[0] if isinstance(out, tuple) else out)[0]
            got[layer] = h.index_select(0, ix).float().mean(0).to(h.dtype)

        hooks = [lm.model.layers[layer].register_forward_hook(lambda m, a, out, layer=layer: grab(layer, out)) for layer in range(N_LAYERS)]
        try:
            with torch.inference_mode():
                lm.model(input_ids=torch.tensor([ids], device="cuda"), use_cache=False)
        finally:
            for h in hooks:
                h.remove()
        rows[i] = torch.stack([got[layer] for layer in range(N_LAYERS)]).to(torch.float16).cpu().numpy()
    if not np.isfinite(rows).all():
        raise ValueError("non-finite activations")
    return rows, left_out


# ---------------------------------------------------------------- fit


def heldout_split(y_train: np.ndarray) -> tuple:
    """Indices into the training set: (the 80% to fit on, the 20% held out), stratified by label, drawn with random.Random(SEED)."""
    rng, held = random.Random(SEED), set()
    train = list(range(len(y_train)))
    for c in (0, 1):
        of_label = [j for j in train if y_train[j] == c]
        held |= set(rng.sample(of_label, round(0.2 * len(of_label))))
    return np.array([j for j in train if j not in held]), np.array(sorted(held))


def fit_logistic(x: np.ndarray, y: np.ndarray) -> tuple:
    """(classifier, mean, std): logistic regression on rows standardized by their own mean and standard deviation (+ 1e-6)."""
    from sklearn.linear_model import LogisticRegression

    mean, std = x.mean(0), x.std(0) + 1e-6
    return LogisticRegression(C=1.0, max_iter=2000, tol=1e-6).fit((x - mean) / std, y), mean, std


def choose_layer(train_rows_at, y_train: np.ndarray, n_jobs: int = 16) -> tuple:
    """(chosen layer, {layer: held-out AUROC}). train_rows_at(layer) returns the training items' rows at that layer as float32
    (items x 5120). Ties go to the lowest layer."""
    from joblib import Parallel, delayed
    from sklearn.metrics import roc_auc_score

    fit_i, held_i = heldout_split(y_train)

    def one(layer):
        x = train_rows_at(layer)
        clf, mean, std = fit_logistic(x[fit_i], y_train[fit_i])
        return layer, float(roc_auc_score(y_train[held_i], clf.decision_function((x[held_i] - mean) / std)))

    heldout = dict(Parallel(n_jobs=n_jobs)(delayed(one)(layer) for layer in range(N_LAYERS)))
    return max(heldout, key=lambda layer: heldout[layer]), heldout


def fit_probe(x_train: np.ndarray, y_train: np.ndarray) -> dict:
    """The probe at one layer from all training rows (float32), in the form we saved: the fit and its folded direction and bias."""
    clf, mean, std = fit_logistic(x_train, y_train)
    w, b = clf.coef_[0].astype(np.float32), float(clf.intercept_[0])
    r, c = (w / std).astype(np.float32), np.float32(b - float((mean * w / std).sum()))  # score = x . r + c
    return {"direction": r, "bias": np.array([c], np.float32), "W": w, "intercept": np.array([b], np.float32),
            "mean": mean.astype(np.float32), "std": std.astype(np.float32)}


# ---------------------------------------------------------------- scoring


def score(x: np.ndarray, probe: dict) -> np.ndarray:
    """Each unit's score: x . r + c for its activations at layer 43 (units x 5120)."""
    x = np.asarray(x).astype(np.float32)
    if not np.isfinite(x).all() or (np.abs(x).max(axis=1) == 0).any():
        raise ValueError("a non-finite or all-zero activation row")
    return (x @ probe["direction"].astype(np.float32) + np.float32(probe["bias"].reshape(-1)[0])).astype(np.float64)


def main():
    from sklearn.metrics import roc_auc_score

    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--out", required=True, help="folder for the probe file")
    args = p.parse_args()
    root, out = Path(__file__).resolve().parents[2], Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    items, other = load_items()
    rows, left_out = read(items, *load_model())
    keep = {s: [i for i, it in enumerate(items) if it[0] == s and i not in left_out] for s in FILES}
    y = {s: np.array([items[i][1] for i in keep[s]]) for s in FILES}
    layer, heldout = choose_layer(lambda layer: rows[keep["train"], layer].astype(np.float32), y["train"])
    probe = fit_probe(rows[keep["train"], layer].astype(np.float32), y["train"])
    safetensors.numpy.save_file(probe, str(out / "high_stakes_probe.safetensors"), metadata={"layer": str(layer)})
    test = {s: roc_auc_score(y[s], score(rows[keep[s], layer], probe)) for s in ("test", "hh")}
    used = safetensors.numpy.load_file(str(root / SHIPPED))
    cos = float(probe["direction"].astype(np.float64) @ used["direction"].astype(np.float64)
                / (np.linalg.norm(probe["direction"].astype(np.float64)) * np.linalg.norm(used["direction"].astype(np.float64))))
    print(f"other labels {other}; left out {len(left_out)}; layer {layer} (the post's: {USED_LAYER}), held-out AUROC {heldout[layer]:.3f}, "
          f"test AUROC {test['test']:.3f}, helpful-and-harmless AUROC {test['hh']:.3f}; cosine with the probe the post used {cos:.6f}")


if __name__ == "__main__":
    main()
