"""Convert FAR.AI's FIBS Qwen3.5-27B linear probes (model.pt) to safetensors without unpickling them (RULE.md).

Run from the repo root: uv run modal run experiments/2026-10-14_q2_fibs_probe/run_convert_fibs_linear.py

One isolated Modal CPU container (no Secrets, no Volumes). For each layer it downloads the pinned files over HTTPS,
checks the model.pt sha256 against the Hugging Face LFS hash, walks data.pkl with pickletools.genops (never
pickle.load, torch.load or numpy.load(allow_pickle=True)), reads the raw float32 storages from the zip, gates keys,
shapes and values, and returns safetensors bytes plus a record. The local entrypoint only writes them to results/.
"""

import datetime
import hashlib
import json
import pathlib
import subprocess
from zoneinfo import ZoneInfo

import modal

APP = "probe-shift_aswin-surya_sub-fibs_convert_kill-2026-11-13"
HF_REPO = "AlignmentResearch/probe-inference-weights"
HF_REV = "08942e47fc0dbebee330d51248c7eaa89cfe540e"
PROBE_DIR = "qwen3.5-27b/linear"
LAYERS = (19, 27, 35, 43, 51, 58)
# Hugging Face LFS sha256 of qwen3.5-27b/linear/layer_<L>/model.pt at HF_REV (HF API, read 2026-10-14 11:50 PT).
EXPECTED_SHA256 = {
    19: "efcec2ce9cba5362cf520a7523e1ae133f058f4933202d8a3006d2bb42fa725f",
    27: "4c2551be0a739119e1b1c44d9fac9da6d6e96972e4c7d996450f8c0635c68c06",
    35: "3efb701b4b555059528093dc4e372e2423baf4c3ae979b9faf6d2a1b4adf8730",
    43: "90c3b80011aab7e5871e9a98f07125a57c46ee58db0aff014fff26d7a9c334ad",
    51: "b9b4e322dd8e8840b27fa5770f7ff648cd0997ccc8898773a5238357d1bf3676",
    58: "582a4162b94e7de5694bac4a7dfb69b63fd6f0790344cbfe6891f9df160ff716",
}
EXPECTED_SIZE = 23229
D_MODEL = 5120
EXPECTED_SHAPES = {
    "linear.weight": (1, D_MODEL),
    "linear.bias": (1,),
    "input_scale": (),
    "input_mean": (),
    "platt_A": (),
    "platt_B": (),
}
ALLOWED_GLOBALS = {
    ("collections", "OrderedDict"),
    ("torch._utils", "_rebuild_tensor_v2"),
    ("torch", "FloatStorage"),
}
HERE = pathlib.Path(__file__).resolve().parent

app = modal.App(APP)
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("numpy==2.3.3", "safetensors==0.8.0")


class Global:
    """A GLOBAL opcode's name. It is recorded as a string pair and never imported."""

    def __init__(self, module, name):
        self.module, self.name = module, name


class Call:
    """A REDUCE node: the callable's name and its arguments, never called."""

    def __init__(self, fn, args):
        self.fn, self.args, self.items, self.state = fn, args, [], None


class PersistentId:
    def __init__(self, pid):
        self.pid = pid


def walk_pickle(data):
    """Describe a torch state-dict pickle from its opcodes alone. Any unexpected opcode or global fails."""
    import collections
    import pickletools

    mark = object()
    stack, memo, ops, globals_seen = [], {}, collections.Counter(), []

    def pop_mark():
        i = max(j for j, x in enumerate(stack) if x is mark)
        items = stack[i + 1 :]
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
        globals_seen.append(f"{module}.{name}")
        if (module, name) not in ALLOWED_GLOBALS:
            raise ValueError(f"unexpected global {module}.{name}")
        stack.append(Global(module, name))

    for op, arg, pos in pickletools.genops(data):
        n = op.name
        ops[n] += 1
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
        elif n in (
            "BINUNICODE",
            "SHORT_BINUNICODE",
            "BINUNICODE8",
            "BININT",
            "BININT1",
            "BININT2",
            "LONG1",
            "BINFLOAT",
        ):
            stack.append(arg)
        elif n == "NEWTRUE":
            stack.append(True)
        elif n == "NEWFALSE":
            stack.append(False)
        elif n == "NONE":
            stack.append(None)
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
        elif n == "TUPLE1":
            stack.append((stack.pop(),))
        elif n == "TUPLE2":
            b = stack.pop()
            stack.append((stack.pop(), b))
        elif n == "TUPLE3":
            c, b = stack.pop(), stack.pop()
            stack.append((stack.pop(), b, c))
        elif n == "BINPERSID":
            stack.append(PersistentId(stack.pop()))
        elif n == "REDUCE":
            args = stack.pop()
            fn = stack.pop()
            if not isinstance(fn, Global):
                raise ValueError(f"REDUCE on a non-global at {pos}")
            stack.append(Call(fn, args))
        elif n == "SETITEM":
            value = stack.pop()
            key = stack.pop()
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
        raise ValueError(f"stack holds {len(stack)} items at STOP")
    return stack[0], dict(ops), globals_seen


def tensor_table(root):
    """The state dict's tensors as plain descriptions: storage key, type, numel, offset, shape, stride."""
    if isinstance(root, Call) and (root.fn.module, root.fn.name) == (
        "collections",
        "OrderedDict",
    ):
        items = root.items
    elif isinstance(root, dict):
        items = list(root.items())
    else:
        raise ValueError(f"root is {type(root).__name__}, not a state dict")
    table = {}
    for key, value in items:
        if not (isinstance(value, Call) and (value.fn.module, value.fn.name) == ("torch._utils", "_rebuild_tensor_v2")):
            raise ValueError(f"{key}: not a _rebuild_tensor_v2 call")
        pid, offset, size, stride = value.args[:4]
        if not isinstance(pid, PersistentId):
            raise TypeError(f"{key}: storage is not a persistent id")
        kind, storage_type, storage_key, location, numel = pid.pid
        if kind != "storage" or (storage_type.module, storage_type.name) != (
            "torch",
            "FloatStorage",
        ):
            raise ValueError(f"{key}: storage {kind} {storage_type.module}.{storage_type.name}, not FloatStorage")
        table[key] = {
            "storage_key": storage_key,
            "storage_type": f"{storage_type.module}.{storage_type.name}",
            "location": location,
            "numel": numel,
            "offset": offset,
            "shape": list(size),
            "stride": list(stride),
            "requires_grad": value.args[4],
        }
    return table


def fetch(path):
    import urllib.request

    url = f"https://huggingface.co/{HF_REPO}/resolve/{HF_REV}/{path}"
    with urllib.request.urlopen(url, timeout=60) as r:
        return r.read()


@app.function(image=image, cpu=1.0, memory=1024, timeout=600, retries=0)
def convert():
    import io
    import zipfile

    import numpy as np
    import safetensors.numpy

    metadata_bytes = fetch(f"{PROBE_DIR}/probe_metadata.json")
    probe_metadata = json.loads(metadata_bytes)
    assert probe_metadata["model_revision"] == "fc05daec18b0a78c049392ed2e771dde82bdf654", probe_metadata
    assert tuple(probe_metadata["layers"]) == LAYERS, probe_metadata
    assert tuple(probe_metadata["layer_rule"]["used_layers"]) == LAYERS, probe_metadata
    record = {
        "hf_repo": HF_REPO,
        "hf_revision": HF_REV,
        "probe_metadata": probe_metadata,
        "probe_metadata_sha256": hashlib.sha256(metadata_bytes).hexdigest(),
        "layers": {},
    }
    files = {}
    for layer in LAYERS:
        config_bytes = fetch(f"{PROBE_DIR}/layer_{layer}/config.json")
        config = json.loads(config_bytes)
        assert config["class_name"] == "LinearProbe", config
        assert config["init_args"] == {
            "d_model": D_MODEL,
            "nhead": 1,
            "normalize_input": "unit_norm",
        }, config
        raw = fetch(f"{PROBE_DIR}/layer_{layer}/model.pt")
        sha = hashlib.sha256(raw).hexdigest()
        if sha != EXPECTED_SHA256[layer] or len(raw) != EXPECTED_SIZE:
            raise ValueError(f"layer {layer}: sha256 {sha} / {len(raw)} bytes, expected {EXPECTED_SHA256[layer]}")
        zf = zipfile.ZipFile(io.BytesIO(raw))
        names = zf.namelist()
        pkl = [x for x in names if x.endswith("/data.pkl")]
        assert len(pkl) == 1, names
        prefix = pkl[0][: -len("data.pkl")]
        byteorder = zf.read(prefix + "byteorder").decode() if prefix + "byteorder" in names else "absent"
        if byteorder not in ("little", "absent"):
            raise ValueError(f"layer {layer}: byteorder {byteorder}")
        root, ops, globals_seen = walk_pickle(zf.read(pkl[0]))
        table = tensor_table(root)
        if set(table) != set(EXPECTED_SHAPES):
            raise ValueError(f"layer {layer}: keys {sorted(table)}, expected {sorted(EXPECTED_SHAPES)}")
        arrays = {}
        for key, t in table.items():
            if tuple(t["shape"]) != EXPECTED_SHAPES[key]:
                raise ValueError(f"layer {layer} {key}: shape {t['shape']}, expected {EXPECTED_SHAPES[key]}")
            buf = zf.read(prefix + "data/" + t["storage_key"])
            if len(buf) != 4 * t["numel"]:
                raise ValueError(f"layer {layer} {key}: {len(buf)} bytes for numel {t['numel']}")
            flat = np.frombuffer(buf, dtype="<f4")
            extent = t["offset"] + sum((s - 1) * st for s, st in zip(t["shape"], t["stride"])) + 1
            if extent > t["numel"]:
                raise ValueError(f"layer {layer} {key}: view runs past its storage")
            view = np.lib.stride_tricks.as_strided(
                flat[t["offset"] :],
                shape=tuple(t["shape"]),
                strides=tuple(4 * s for s in t["stride"]),
            )
            arrays[key] = np.ascontiguousarray(view, dtype=np.float32)
            if not np.isfinite(arrays[key]).all():
                raise ValueError(f"layer {layer} {key}: non-finite values")
        if float(arrays["input_scale"]) == 1.0:
            raise ValueError(f"layer {layer}: input_scale is 1.0, the unset default")
        out = safetensors.numpy.save(
            arrays,
            metadata={
                "source": f"{HF_REPO}@{HF_REV}/{PROBE_DIR}/layer_{layer}/model.pt",
                "source_sha256": sha,
                "converted_by": "pickletools.genops walk, no unpickling (experiments/2026-10-14_q2_fibs_probe)",
            },
        )
        back = safetensors.numpy.load(out)
        if set(back) != set(arrays) or any(not np.array_equal(back[k], arrays[k]) for k in arrays):
            raise ValueError(f"layer {layer}: safetensors round trip differs")
        name = f"fibs_qwen3.5-27b_linear_L{layer}.safetensors"
        files[name] = out
        record["layers"][str(layer)] = {
            "source_path": f"{PROBE_DIR}/layer_{layer}/model.pt",
            "source_sha256_expected": EXPECTED_SHA256[layer],
            "source_sha256": sha,
            "source_bytes": len(raw),
            "config": config,
            "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
            "zip_names": names,
            "byteorder": byteorder,
            "opcodes": ops,
            "globals": sorted(set(globals_seen)),
            "tensors": table,
            "output": name,
            "output_sha256": hashlib.sha256(out).hexdigest(),
            "output_bytes": len(out),
            "values (descriptive)": {
                "input_scale": float(arrays["input_scale"]),
                "input_mean": float(arrays["input_mean"]),
                "platt_A": float(arrays["platt_A"]),
                "platt_B": float(arrays["platt_B"]),
                "bias": float(arrays["linear.bias"][0]),
                "weight_l2": float(np.linalg.norm(arrays["linear.weight"])),
            },
        }
    record["gates pass"] = True
    return {"record": record, "files": files}


@app.local_entrypoint()
def main():
    rel = pathlib.Path(__file__).resolve().relative_to(pathlib.Path.cwd())
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--", str(rel)],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    if dirty.strip():
        raise SystemExit(f"{rel} has uncommitted changes; commit it before running")
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    result = convert.remote()
    stamp = datetime.datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    out_dir = HERE / "results"
    out_dir.mkdir(exist_ok=True)
    for name, data in result["files"].items():
        (out_dir / name).write_bytes(data)
    record = {
        "POST HOC": True,
        "status": "preliminary",
        "git": head,
        "stamp (PT)": stamp,
        "app": APP,
    } | result["record"]
    path = out_dir / f"convert_fibs_linear_{stamp}.json"
    path.write_text(json.dumps(record, indent=1) + "\n")
    print(f"wrote {path} and {len(result['files'])} safetensors files; gates pass: {record['gates pass']}")
