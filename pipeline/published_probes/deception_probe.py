# /// script
# requires-python = "==3.12.*"
# dependencies = ["numpy==2.3.3", "safetensors==0.8.0"]
# ///
"""The deception probe: where its weights come from, how we converted them, and how it scores a unit.

Source. Hollinsworth, Spies, Diriba, Gleave and Cundy (2026), "Telling FIBS: Probes Effectively Detect Sabotage and Catch
Unverbalized Deception", arXiv 2610.12445. Its authors (FAR AI) publish linear probes for Qwen3.5-27B on Hugging Face:
AlignmentResearch/probe-inference-weights at revision 08942e47fc0dbebee330d51248c7eaa89cfe540e, folder qwen3.5-27b/linear, one
probe for each of layers 19, 27, 35, 43, 51 and 58, trained on their deception dataset (FIBS). They are MIT-licensed (copyright
FAR AI, Inc.), so we redistribute the converted copies we used, with that license, in weights/published/deception_probe/. We
trained nothing and chose nothing: the layers, weights and scoring rule are the authors'.

Layers. Layer L is the output of decoder block L of Qwen/Qwen3.5-27B (revision fc05daec18b0a78c049392ed2e771dde82bdf654), before
the final norm: Hugging Face's hidden_states[L + 1]. The model runs in bfloat16 and the activations are stored as float16.

Conversion. The weights are published only as PyTorch model.pt files: a zip archive holding a pickle. Loading a pickle can run
arbitrary code, so we never unpickle. We walk the pickle's opcodes with pickletools, which only describes them, take each tensor's
storage, offset, shape and stride from that description, and read the raw little-endian float32 bytes from the zip. Every
model.pt must match the sha256 Hugging Face lists for it, and anything unexpected (an opcode, a global name, a key, a shape, a
non-finite value) stops the conversion. Each layer holds linear.weight (1 x 5120), linear.bias, input_scale, input_mean (0),
platt_A and platt_B.

Scoring a unit. With x_L the activation the unit is read at (layer L),
    score = mean over the six layers of sigmoid(w_L . (x_L / input_scale_L) + b_L).
This is the authors' scorer (GitHub AlignmentResearch/fibs-probes at 6cae61deed78157021bf141c69f626881bbd638d, score.py line
180): "normalize_input: unit_norm" divides by input_scale, a fixed norm stored with each probe, and subtracts nothing; the Platt
parameters are stored but not applied. The division and dot product run in float32, the sigmoid and the average in float64. A
score lies between 0 and 1, and higher means more deceptive, so more suspicious. The authors read one token, the last token of a
chat reply before its end-of-turn token. We apply the probes unchanged at our units instead: for a command, the activation at the
last token inside the tool call's main argument (a shell call's command; a file edit's new text, file_text for create and new_str
for str_replace and insert; a view's path); for a tool output, the mean activation over the output's content tokens (Appendix A of
the post; positions defined in our internal src/probe_shift/reader.py, command_param and command_end_positions).

Original code (our internal repository, unchanged since it ran): the conversion is
experiments/2026-10-14_q2_fibs_probe/run_convert_fibs_linear.py lines 74-325 (run 2026-10-14 11:57 PT); the per-layer logit is
experiments/2026-10-14_p2_fibs_pair/run_fibs_score.py lines 117-143, and the average of sigmoids is
experiments/2026-10-14_p2_simh_scores/run_simh_scores.py lines 358-370, the scorer behind every deception-probe score in the post.

Run from the repository root (needs only a network connection; no GPU):
    uv run pipeline/published_probes/deception_probe.py --out output/deception_probe
downloads the six model.pt files, converts them, writes them to --out and checks that every array equals the copy we used.
"""

import argparse
import hashlib
import io
import json
import pickletools
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import safetensors.numpy

HF_REPO, HF_REV = "AlignmentResearch/probe-inference-weights", "08942e47fc0dbebee330d51248c7eaa89cfe540e"
PROBE_DIR = "qwen3.5-27b/linear"
MODEL_REVISION = "fc05daec18b0a78c049392ed2e771dde82bdf654"  # the Qwen3.5-27B revision the probes were trained on
LAYERS = (19, 27, 35, 43, 51, 58)
SOURCE_SHA256 = {  # Hugging Face's sha256 of qwen3.5-27b/linear/layer_<L>/model.pt at HF_REV (each file is 23,229 bytes)
    19: "efcec2ce9cba5362cf520a7523e1ae133f058f4933202d8a3006d2bb42fa725f",
    27: "4c2551be0a739119e1b1c44d9fac9da6d6e96972e4c7d996450f8c0635c68c06",
    35: "3efb701b4b555059528093dc4e372e2423baf4c3ae979b9faf6d2a1b4adf8730",
    43: "90c3b80011aab7e5871e9a98f07125a57c46ee58db0aff014fff26d7a9c334ad",
    51: "b9b4e322dd8e8840b27fa5770f7ff648cd0997ccc8898773a5238357d1bf3676",
    58: "582a4162b94e7de5694bac4a7dfb69b63fd6f0790344cbfe6891f9df160ff716",
}
SOURCE_SIZE = 23229
SHAPES = {"linear.weight": (1, 5120), "linear.bias": (1,), "input_scale": (), "input_mean": (), "platt_A": (), "platt_B": ()}
ALLOWED_GLOBALS = {("collections", "OrderedDict"), ("torch._utils", "_rebuild_tensor_v2"), ("torch", "FloatStorage")}
SHIPPED = "weights/published/deception_probe/fibs_qwen3.5-27b_linear_L{layer}.safetensors"  # relative to the repository root


def fetch(path: str) -> bytes:
    """One file of the weights repository at the pinned revision, over HTTPS (public, no token)."""
    with urllib.request.urlopen(f"https://huggingface.co/{HF_REPO}/resolve/{HF_REV}/{path}", timeout=60) as r:
        return r.read()


# ---------------------------------------------------------------- reading a PyTorch file without unpickling it


class Global:
    """A GLOBAL opcode's name, kept as two strings and never imported."""

    def __init__(self, module, name):
        self.module, self.name = module, name


class Call:
    """A REDUCE opcode: the callable's name and its arguments, never called."""

    def __init__(self, fn, args):
        self.fn, self.args, self.items, self.state = fn, args, [], None


class PersistentId:
    def __init__(self, pid):
        self.pid = pid


def walk_pickle(data: bytes):
    """Describe a PyTorch state-dict pickle from its opcodes alone; any unexpected opcode or global name stops."""
    mark, stack, memo = object(), [], {}

    def pop_mark():
        i = max(j for j, x in enumerate(stack) if x is mark)
        items = stack[i + 1:]
        del stack[i:]
        return items

    def set_item(target, key, value):
        if isinstance(target, dict):
            target[key] = value
        elif isinstance(target, Call):
            target.items.append((key, value))
        else:
            raise TypeError(f"SETITEM on {type(target).__name__}")

    def push_global(module, name):
        if (module, name) not in ALLOWED_GLOBALS:
            raise ValueError(f"unexpected global {module}.{name}")
        stack.append(Global(module, name))

    for op, arg, pos in pickletools.genops(data):
        n = op.name
        if n in ("PROTO", "FRAME"):
            continue
        if n == "STOP":
            break
        if n == "EMPTY_DICT":
            stack.append({})
        elif n == "EMPTY_LIST":
            stack.append([])
        elif n == "EMPTY_TUPLE":
            stack.append(())
        elif n == "MARK":
            stack.append(mark)
        elif n in ("BINUNICODE", "SHORT_BINUNICODE", "BINUNICODE8", "BININT", "BININT1", "BININT2", "LONG1", "BINFLOAT"):
            stack.append(arg)
        elif n in ("NEWTRUE", "NEWFALSE", "NONE"):
            stack.append({"NEWTRUE": True, "NEWFALSE": False, "NONE": None}[n])
        elif n == "GLOBAL":
            push_global(*arg.split(" ", 1))
        elif n == "STACK_GLOBAL":
            name = stack.pop()
            push_global(stack.pop(), name)
        elif n in ("BINPUT", "LONG_BINPUT"):
            memo[arg] = stack[-1]
        elif n == "MEMOIZE":
            memo[len(memo)] = stack[-1]
        elif n in ("BINGET", "LONG_BINGET"):
            stack.append(memo[arg])
        elif n == "TUPLE":
            stack.append(tuple(pop_mark()))
        elif n in ("TUPLE1", "TUPLE2", "TUPLE3"):
            k = int(n[-1])
            items = stack[-k:]
            del stack[-k:]
            stack.append(tuple(items))
        elif n == "BINPERSID":
            stack.append(PersistentId(stack.pop()))
        elif n == "REDUCE":
            args, fn = stack.pop(), stack.pop()
            if not isinstance(fn, Global):
                raise ValueError(f"REDUCE on a non-global at byte {pos}")
            stack.append(Call(fn, args))
        elif n == "SETITEM":
            value, key = stack.pop(), stack.pop()
            set_item(stack[-1], key, value)
        elif n == "SETITEMS":
            items = pop_mark()
            for key, value in zip(items[::2], items[1::2], strict=True):
                set_item(stack[-1], key, value)
        elif n == "APPENDS":
            items = pop_mark()
            if not isinstance(stack[-1], list):
                raise ValueError(f"APPENDS on {type(stack[-1]).__name__}")
            stack[-1].extend(items)
        elif n == "BUILD":
            state = stack.pop()
            if not isinstance(stack[-1], Call):
                raise ValueError(f"BUILD on {type(stack[-1]).__name__}")
            stack[-1].state = state
        else:
            raise ValueError(f"unexpected opcode {n} at byte {pos}")
    if len(stack) != 1:
        raise ValueError(f"the pickle leaves {len(stack)} items on the stack")
    return stack[0]


def tensor_table(root) -> dict:
    """The state dict's tensors as plain descriptions: storage key, number of elements, offset, shape and stride."""
    if isinstance(root, Call) and (root.fn.module, root.fn.name) == ("collections", "OrderedDict"):
        items = root.items
    elif isinstance(root, dict):
        items = list(root.items())
    else:
        raise ValueError(f"the pickle holds a {type(root).__name__}, not a state dict")
    table = {}
    for key, value in items:
        if not (isinstance(value, Call) and (value.fn.module, value.fn.name) == ("torch._utils", "_rebuild_tensor_v2")):
            raise ValueError(f"{key}: not a tensor")
        pid, offset, size, stride = value.args[:4]
        if not isinstance(pid, PersistentId):
            raise TypeError(f"{key}: storage is not a persistent id")
        kind, storage_type, storage_key, _location, numel = pid.pid
        if kind != "storage" or (storage_type.module, storage_type.name) != ("torch", "FloatStorage"):
            raise ValueError(f"{key}: not a float32 storage")
        table[key] = {"storage_key": storage_key, "numel": numel, "offset": offset, "shape": tuple(size), "stride": tuple(stride)}
    return table


def convert_layer(raw: bytes) -> dict:
    """One layer's model.pt bytes -> its six arrays (float32), with every check of the conversion."""
    zf = zipfile.ZipFile(io.BytesIO(raw))
    names = zf.namelist()
    pkl = [x for x in names if x.endswith("/data.pkl")]
    if len(pkl) != 1:
        raise ValueError(f"expected one data.pkl, found {pkl}")
    prefix = pkl[0][: -len("data.pkl")]
    if prefix + "byteorder" in names and zf.read(prefix + "byteorder").decode() != "little":
        raise ValueError("the tensors are not stored little-endian")
    table = tensor_table(walk_pickle(zf.read(pkl[0])))
    if set(table) != set(SHAPES):
        raise ValueError(f"keys {sorted(table)}, expected {sorted(SHAPES)}")
    arrays = {}
    for key, t in table.items():
        if t["shape"] != SHAPES[key]:
            raise ValueError(f"{key}: shape {t['shape']}, expected {SHAPES[key]}")
        buf = zf.read(prefix + "data/" + t["storage_key"])
        if len(buf) != 4 * t["numel"]:
            raise ValueError(f"{key}: {len(buf)} bytes for {t['numel']} float32 values")
        flat = np.frombuffer(buf, dtype="<f4")
        if t["offset"] + sum((s - 1) * st for s, st in zip(t["shape"], t["stride"], strict=True)) + 1 > t["numel"]:
            raise ValueError(f"{key}: the view runs past its storage")
        view = np.lib.stride_tricks.as_strided(flat[t["offset"]:], shape=t["shape"], strides=tuple(4 * s for s in t["stride"]))
        arrays[key] = np.ascontiguousarray(view, dtype=np.float32)
        if not np.isfinite(arrays[key]).all():
            raise ValueError(f"{key}: non-finite values")
    if float(arrays["input_scale"].reshape(-1)[0]) == 1.0:  # numpy stores a 0-d tensor as one element
        raise ValueError("input_scale is 1.0, the unset default")
    return arrays


def download_and_convert() -> dict:
    """All six layers from the pinned revision: {layer: (safetensors bytes, source sha256)}. Checks the probes' metadata and each
    layer's config before converting it."""
    meta = json.loads(fetch(f"{PROBE_DIR}/probe_metadata.json"))
    if meta["model_revision"] != MODEL_REVISION or tuple(meta["layers"]) != LAYERS or tuple(meta["layer_rule"]["used_layers"]) != LAYERS:
        raise ValueError("probe_metadata.json does not describe the six Qwen3.5-27B layers we expect")
    out = {}
    for layer in LAYERS:
        config = json.loads(fetch(f"{PROBE_DIR}/layer_{layer}/config.json"))
        if config["class_name"] != "LinearProbe" or config["init_args"] != {"d_model": 5120, "nhead": 1, "normalize_input": "unit_norm"}:
            raise ValueError(f"layer {layer}: not the linear probe we expect: {config}")
        raw = fetch(f"{PROBE_DIR}/layer_{layer}/model.pt")
        sha = hashlib.sha256(raw).hexdigest()
        if sha != SOURCE_SHA256[layer] or len(raw) != SOURCE_SIZE:
            raise ValueError(f"layer {layer}: sha256 {sha} ({len(raw)} bytes) differs from the pinned file")
        meta_out = {"source": f"{HF_REPO}@{HF_REV}/{PROBE_DIR}/layer_{layer}/model.pt", "source_sha256": sha}
        out[layer] = (safetensors.numpy.save(convert_layer(raw), metadata=meta_out), sha)
    return out


# ---------------------------------------------------------------- scoring


def load_probes(folder: Path) -> dict:
    """{layer: arrays} from the six converted files in `folder` (our shipped copies or your own conversion)."""
    return {layer: safetensors.numpy.load_file(str(Path(folder) / f"fibs_qwen3.5-27b_linear_L{layer}.safetensors")) for layer in LAYERS}


def layer_logits(x: np.ndarray, probe: dict) -> np.ndarray:
    """One layer's logit for each row of x (units x 5120): w . (x / input_scale) + b, in float32 as the authors' scorer computes
    it, returned as float64. input_mean is 0 for every layer and is not subtracted."""
    if probe["linear.weight"].shape != (1, 5120) or float(probe["input_mean"].reshape(-1)[0]) != 0.0:
        raise ValueError("unexpected weight shape or a nonzero input_mean")
    w, b = probe["linear.weight"][0].astype(np.float32), np.float32(probe["linear.bias"].reshape(-1)[0])
    scale = np.float32(probe["input_scale"].reshape(-1)[0])
    x = np.asarray(x).astype(np.float32)
    if not np.isfinite(x).all() or (np.abs(x).max(axis=1) == 0).any():
        raise ValueError("a non-finite or all-zero activation row")  # the authors' scorer treats zero rows as padding
    return ((x / scale) @ w + b).astype(np.float64)


def score(acts: dict, probes: dict) -> np.ndarray:
    """Each unit's deception-probe score. acts[layer] holds the units' activations at that layer (units x 5120), the same
    units in the same order for all six layers."""
    z = np.stack([layer_logits(acts[layer], probes[layer]) for layer in LAYERS], axis=1)  # units x 6
    return (1 / (1 + np.exp(-z))).mean(1)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--out", required=True, help="folder for the converted files")
    args = p.parse_args()
    root, out = Path(__file__).resolve().parents[2], Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for layer, (data, sha) in download_and_convert().items():
        (out / f"fibs_qwen3.5-27b_linear_L{layer}.safetensors").write_bytes(data)
        mine, ours = safetensors.numpy.load(data), safetensors.numpy.load_file(str(root / SHIPPED.format(layer=layer)))
        same = set(mine) == set(ours) and all(np.array_equal(mine[k], ours[k]) and mine[k].dtype == ours[k].dtype for k in mine)
        print(f"layer {layer}: source sha256 {sha[:12]}..., converted; every array equal to the copy we used: {same}")
        if not same:
            raise SystemExit(f"layer {layer}: the conversion differs from the copy we used")


if __name__ == "__main__":
    main()
