# /// script
# requires-python = "==3.12.*"
# dependencies = ["numpy==2.3.3", "safetensors==0.8.0", "torch==2.8.0", "transformers==5.19.0", "accelerate>=1.1.0",
#                 "flash-linear-attention==0.5.2"]
# ///
"""The harmfulness and refusal directions: how we built them for Qwen3.5-27B, and how each scores a unit.

Sources. The refusal direction follows Arditi et al. (2024), "Refusal in Language Models Is Mediated by a Single Direction",
arXiv 2406.11717; the harmfulness direction follows Zhao et al. (2025), "LLMs Encode Harmfulness and Refusal Separately", arXiv
2507.11878. Neither paper publishes weights for Qwen3.5-27B, so we built both with their method. The prompts are the harmful and
harmless splits of Arditi et al.'s code, GitHub andyrdt/refusal_direction at 9d852fae1a9121c78b29142de733cb1340770cc3,
dataset/splits/ (Apache-2.0 repository; the harmful prompts come from AdvBench, MaliciousInstruct, TDC2023 and HarmBench, the
harmless ones from Alpaca). We do not redistribute the prompts: this script downloads them, checks each file's git hash, and draws
the same sample. The two directions are ours, and we ship the exact files the post used in weights/published/.

Sample. One random.Random(20261014) draws, in this order: 128 prompts from the harmful training file, 128 from the harmless
training file, then 32 harmful and 32 harmless validation prompts, then 64 harmful and 64 harmless test prompts; 448 in all.
Arditi et al. keep only training prompts the model actually refuses; we skipped that filter, because it needs generation.

Reading. Each prompt is rendered as one user turn with Qwen3.5-27B's own chat template and the generation prompt, and read once
(bfloat16, attention "sdpa"). At every layer we keep two tokens: the last token of the request (the token before the <|im_end|>
that closes the user turn; never a special token), and the last token of the rendered prompt, a template token just before the
model's reply. Layer L is the output of decoder block L (Hugging Face's hidden_states[L + 1]); rows are stored as float16.

Fit. At each layer, r = mean(harmful training rows) - mean(harmless training rows), in float32, scaled to unit length. The
harmfulness direction uses the request's last token; the refusal direction uses the prompt's last token. Each direction's layer
is the one among layers 0-50 (Arditi et al. leave out the last 20% of layers) with the highest validation AUROC, ties broken by
the larger validation Cohen's d; our transcripts never enter the choice. Harmfulness: layer 34. Refusal: layer 47. We first read
the prompts on an H100 GPU and chose the layers there; the transcripts were read on H200 GPUs, so we re-read the prompts on an
H200 with the same code and recomputed both directions at the same layers. The H200 read chooses the same layers by itself, each
H200 direction has cosine above 0.99999 with its H100 version, and the H200 directions are the ones the post used. Their
validation AUROC is 1.000 for both; their test AUROC is 0.9998 (harmfulness) and 1.000 (refusal). We did not check that the refusal direction controls refusals in Qwen3.5, so it
is a candidate refusal direction.

Scoring a unit. score = x_L . r, the projection of the unit's activation at the direction's layer on the unit direction; the
product runs in float32 and the result is kept as float64. No bias, no centring, no scaling. Higher means more harmful (or closer
to a refusal), so more suspicious. Both directions were fit at prompt tokens but are applied only at our transcripts' content
tokens: for a command, the last token inside the tool call's main argument (a shell call's command; a file edit's new text, file_text
for create and new_str for str_replace and insert; a view's path); for a tool output, the mean over its content tokens (Appendix A
of the post; positions defined in our internal src/probe_shift/reader.py, command_param and command_end_positions). The scores
are in the direction's own units.

Original code (our internal repository, unchanged since it ran): the read and fit are
experiments/2026-10-14_p2_refusal_harm/run_read_fit.py lines 38-139 (H100 run 2026-10-14 12:55 PT; the sample at lines 63-71,
the two token positions at lines 79-89, the direction and layer choice at lines 120-134); the H200 directions are
run_h200_check.py lines 36-69 in the same folder (run 14:52 PT); the per-unit score is
experiments/2026-10-14_p2_refusal_harm/run_score_linear.py lines 116-147, called by
experiments/2026-10-14_p2_simh_scores/run_simh_scores.py, the scorer behind every score in the post.

Run from the repository root on one GPU with at least 80 GB of memory (the model weights, about 54 GB, download on first use):
    uv run pipeline/published_probes/harmfulness_and_refusal_directions.py --out output/directions
reads the 448 prompts, writes both directions at the post's layers (34 and 47), reports the layers this read alone would choose,
and compares the directions with the copies we used.
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
SOURCE = "https://raw.githubusercontent.com/andyrdt/refusal_direction/9d852fae1a9121c78b29142de733cb1340770cc3/dataset/splits/"
GIT_BLOBS = {"harmful_train": "5ca6b46750e06bc401ebd93171a6b0dc0590cdd8", "harmful_val": "c3128d37e8b255e765f4f51f1a6715572595e3e7",
             "harmful_test": "2dc705dc1a50e7773efca46fedab71229169b3bb", "harmless_train": "700a497bd1d20ab074fcc576e9bd79ac604543c5",
             "harmless_val": "6b9ee6e9c789799354b618a046758276da445bd8", "harmless_test": "6033b711a3c0bf3d49fb88a4824fdac8be792f25"}
SAMPLE = {"train": 128, "val": 32, "test": 64}  # per split, for each of harmful and harmless
SEED = 20261014
N_LAYERS, CANDIDATE_LAYERS = 64, range(51)
POSITIONS = ("request_end", "prompt_end")  # the two tokens read per prompt
DIRECTIONS = {"harmfulness": "request_end", "refusal": "prompt_end"}
USED_LAYER = {"harmfulness": 34, "refusal": 47}  # the post's directions
SHIPPED = {"harmfulness": "weights/published/harmfulness_direction.safetensors", "refusal": "weights/published/refusal_direction.safetensors"}


def load_prompts() -> list:
    """The 448 sampled prompts as (split, label, text), label 1 harmful and 0 harmless, in reading order."""
    data = {}
    for name, blob in GIT_BLOBS.items():  # parsed as JSON data only
        raw = urllib.request.urlopen(SOURCE + f"{name}.json", timeout=60).read()
        if hashlib.sha1(b"blob %d\0" % len(raw) + raw).hexdigest() != blob:
            raise ValueError(f"{name}: the file's git hash differs from the pinned one")
        data[name] = [x["instruction"] for x in json.loads(raw)]
    rng, prompts = random.Random(SEED), []
    for split, n in SAMPLE.items():
        for label, kind in ((1, "harmful"), (0, "harmless")):
            prompts += [(split, label, text) for text in rng.sample(data[f"{kind}_{split}"], n)]
    return prompts


def token_positions(tok, text: str) -> tuple:
    """The prompt rendered as one user turn with the generation prompt: its token ids, and the positions of the request's last
    token and of the prompt's last token."""
    ids = tok(tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True, tokenize=False),
              add_special_tokens=False)["input_ids"]
    start_id, end_id = tok.convert_tokens_to_ids("<|im_start|>"), tok.convert_tokens_to_ids("<|im_end|>")
    user = [j for j in range(len(ids) - 1) if ids[j] == start_id and tok.decode([ids[j + 1]]).strip() == "user"]
    ends = [j for j in range(user[-1] + 2, len(ids)) if ids[j] == end_id] if user else []
    if len(user) != 1 or not ends or ends[0] - 1 <= user[0] + 2 or ids[ends[0] - 1] in set(tok.all_special_ids):
        raise ValueError("no single user turn with a content token before its <|im_end|>")
    return ids, (ends[0] - 1, len(ids) - 1)


def load_model() -> tuple:
    """(tokenizer, model): Qwen3.5-27B at the pinned revision, in bfloat16, on one GPU."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    lm = AutoModelForCausalLM.from_pretrained(MODEL, revision=REVISION, dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa").eval()
    return tok, lm


def read(prompts: list, tok, lm) -> np.ndarray:
    """Activations of every prompt at its two positions and all 64 layers: float16, shape (prompts, 2, 64, 5120)."""
    import torch

    rows = np.zeros((len(prompts), len(POSITIONS), N_LAYERS, 5120), np.float16)
    for i, (_, _, text) in enumerate(prompts):
        ids, at = token_positions(tok, text)
        got, idx = {}, torch.tensor(at, device="cuda")

        def grab(layer, out, got=got, idx=idx):  # the output of decoder block `layer`, at the two positions
            got[layer] = (out[0] if isinstance(out, tuple) else out)[0].index_select(0, idx).to(torch.float16).cpu().numpy()

        hooks = [lm.model.layers[layer].register_forward_hook(lambda m, a, out, layer=layer: grab(layer, out)) for layer in range(N_LAYERS)]
        try:
            with torch.inference_mode():
                lm.model(input_ids=torch.tensor([ids], device="cuda"), use_cache=False)
        finally:
            for h in hooks:
                h.remove()
        for layer in range(N_LAYERS):
            rows[i, :, layer] = got[layer]
    if not np.isfinite(rows).all() or (np.abs(rows).max(axis=-1) == 0).any():
        raise ValueError("a non-finite or all-zero activation row")
    return rows


# ---------------------------------------------------------------- fit


def auroc(pos: np.ndarray, neg: np.ndarray) -> float:
    """Probability that a positive scores above a negative; ties count one half."""
    s = np.concatenate([pos, neg])
    o = np.argsort(s, kind="mergesort")
    first, cnt = np.unique(s[o], return_index=True, return_counts=True)[1:]
    rank = np.empty(len(s))
    rank[o] = np.repeat(first + (cnt - 1) / 2 + 1, cnt)
    return float((rank[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def difference_of_means(train_rows: np.ndarray, train_labels: np.ndarray) -> np.ndarray:
    """Unit vector along mean(harmful) - mean(harmless) of one layer's training rows (float32)."""
    x = np.asarray(train_rows).astype(np.float32)
    r = x[train_labels == 1].mean(0) - x[train_labels == 0].mean(0)
    return (r / np.linalg.norm(r)).astype(np.float32)


def layer_metrics(rows: np.ndarray, split: np.ndarray, label: np.ndarray, position: str) -> dict:
    """For every layer: the direction fit on the training prompts, and its validation and test AUROC and Cohen's d."""
    p, out = POSITIONS.index(position), {}
    for layer in range(N_LAYERS):
        u = difference_of_means(rows[split == "train", p, layer], label[split == "train"])
        m = {"direction": u}
        for s in ("val", "test"):
            sc, y = rows[split == s, p, layer].astype(np.float32) @ u, label[split == s]
            a, b = sc[y == 1], sc[y == 0]
            m[f"{s}_auroc"] = auroc(a, b)
            m[f"{s}_d"] = float((a.mean() - b.mean()) / np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2))
        out[layer] = m
    return out


def choose_layer(metrics: dict) -> int:
    """Among layers 0-50, the highest validation AUROC, then the larger validation Cohen's d."""
    return max(CANDIDATE_LAYERS, key=lambda layer: (metrics[layer]["val_auroc"], metrics[layer]["val_d"]))


def fit(rows: np.ndarray, split: np.ndarray, label: np.ndarray, layers: dict | None = None) -> dict:
    """Both directions: {name: (layer, unit direction, that layer's metrics, the layer this read would choose)}. With `layers`
    ({name: layer}) each direction is taken at the given layer, as for the post's directions (layers chosen on the first read);
    without, at the layer this read chooses."""
    out = {}
    for name, position in DIRECTIONS.items():
        metrics = layer_metrics(rows, split, label, position)
        chosen = choose_layer(metrics)
        layer = chosen if layers is None else layers[name]
        out[name] = (layer, metrics[layer]["direction"], {k: v for k, v in metrics[layer].items() if k != "direction"}, chosen)
    return out


# ---------------------------------------------------------------- scoring


def score(x: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """Each unit's score: its activation at the direction's layer (units x 5120) projected on the unit direction."""
    x = np.asarray(x).astype(np.float32)
    if not np.isfinite(x).all() or (np.abs(x).max(axis=1) == 0).any():
        raise ValueError("a non-finite or all-zero activation row")
    return (x @ direction.astype(np.float32) + np.float32(0)).astype(np.float64)


def load_direction(path) -> tuple:
    """(layer, unit direction) from a direction file; the layer is in the file's metadata."""
    with open(path, "rb") as f:
        data = f.read()
    header = json.loads(data[8: 8 + int.from_bytes(data[:8], "little")])
    return int(header["__metadata__"]["layer"]), safetensors.numpy.load(data)["direction"]


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--out", required=True, help="folder for the two direction files")
    args = p.parse_args()
    root, out = Path(__file__).resolve().parents[2], Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    prompts = load_prompts()
    split, label = np.array([s for s, _, _ in prompts]), np.array([y for _, y, _ in prompts])
    rows = read(prompts, *load_model())
    for name, (layer, u, m, chosen) in fit(rows, split, label, layers=USED_LAYER).items():
        safetensors.numpy.save_file({"direction": u}, str(out / f"{name}_direction.safetensors"), metadata={"layer": str(layer)})
        used_layer, used = load_direction(root / SHIPPED[name])
        cos = float(u.astype(np.float64) @ used.astype(np.float64) / (np.linalg.norm(u.astype(np.float64)) * np.linalg.norm(used.astype(np.float64))))
        print(f"{name}: layer {layer} (the post's: {used_layer}; this read alone would choose {chosen}), validation AUROC {m['val_auroc']:.3f}, "
              f"test AUROC {m['test_auroc']:.3f}; cosine with the direction the post used {cos:.6f}")


if __name__ == "__main__":
    main()
