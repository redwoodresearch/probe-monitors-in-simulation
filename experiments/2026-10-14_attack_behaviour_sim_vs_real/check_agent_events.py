"""Smell check on run_behaviour.py's token metrics: do the x-irid model events it sums include calls other than the agent's
(e.g. the simulator's own model calls in sim runs)? Modal CPU, counts and model NAMES only (identifier allowlist); no text.

  PYTHONPATH=src uv run --no-project --python 3.12 --with modal==1.6.1 modal run \\
      experiments/2026-10-14_attack_behaviour_sim_vs_real/check_agent_events.py
"""

import json
import re
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).parent
app = modal.App("probe-shift_aswin-surya_sub-attack_behaviour_kill-2026-11-14")
EV_ROOT = "/ev/evals_v1"
SAMPLE_RE = re.compile(r"samples/.+_epoch_(\d+)\.json")
PATH_RE = re.compile(r"[A-Za-z0-9_@.:/-]+\.eval")
NAME_RE = re.compile(r"[A-Za-z0-9_.:-]{1,64}")
MODEL_RE = re.compile(r"[A-Za-z0-9_.:/-]{1,128}")  # model names may carry provider/org prefixes
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("zstandard==0.23.0")
ev = modal.Volume.from_name("probe-shift_aswin-surya_sub-q4own_evals_kill-2026-11-13", version=2).read_only()


def world_of_path(rel: str):  # 09's run_tool_behaviour.world_of_path @ 19421fa7f, attack worlds only
    arm = Path(rel).parts[1]
    if arm == "attack":
        return "run2" if rel.startswith("2026-10-10_001314_second_attack/") else "run1"
    return "sim" if arm == "sim_attack@september" else None


def zstd_zip():  # 09's run_tool_behaviour.zstd_zip @ 19421fa7f, unchanged
    import zipfile

    import zstandard

    if hasattr(zipfile, "ZIP_ZSTANDARD") or getattr(zipfile, "_zstd_patched", False):
        return zipfile
    check, get = zipfile._check_compression, zipfile._get_decompressor
    zipfile._check_compression = lambda ct: None if ct == 93 else check(ct)
    zipfile._get_decompressor = lambda ct: zstandard.ZstdDecompressor().decompressobj() if ct == 93 else get(ct)
    zipfile._zstd_patched = True
    return zipfile


def ident(x):
    if not isinstance(x, str) or not NAME_RE.fullmatch(x):
        raise ValueError("a name or id field is not an identifier")
    return x

def model_name(x):
    if not isinstance(x, str) or not MODEL_RE.fullmatch(x):
        raise ValueError("a model name is not an identifier")
    return x


@app.function(image=image, cpu=1, memory=8192, timeout=1800, retries=0, volumes={"/ev": ev})
def scan(rels: list) -> list:
    zipfile = zstd_zip()
    out = []
    for rel in rels:
        with zipfile.ZipFile(Path(EV_ROOT) / rel) as z:
            for n in [n for n in z.namelist() if SAMPLE_RE.fullmatch(n)]:
                s = json.loads(z.read(n))
                by = {}  # (has x-irid, model name) -> [events, events with tool calls, output tokens, max input tokens]
                for e in s.get("events") or []:
                    if not isinstance(e, dict) or e.get("event") != "model":
                        continue
                    hdr = ((e.get("call") or {}).get("request") or {}).get("extra_headers") or {}
                    key = f"{'irid' if isinstance(hdr, dict) and hdr.get('x-irid') else 'no_irid'}|{model_name(e.get('model') or 'none')}"
                    o = e.get("output") or {}
                    u = o.get("usage") or {}
                    tc = ((o.get("choices") or [{}])[0].get("message") or {}).get("tool_calls") or []
                    v = by.setdefault(key, [0, 0, 0, 0])
                    v[0] += 1
                    v[1] += bool(tc)
                    v[2] += int(u.get("output_tokens") or 0)
                    v[3] = max(v[3], int(u.get("input_tokens") or 0))
                steps = sum(m.get("role") == "assistant" for m in s.get("messages") or [])
                out.append({"world": world_of_path(rel), "steps": steps, "by": by})
    return out

@app.local_entrypoint()
def main():
    rels = scan_list.remote()
    rows = [r for part in scan.map([rels[i:i + 20] for i in range(0, len(rels), 20)]) for r in part]
    summ = {}
    for r in rows:
        w = summ.setdefault(r["world"], {"samples": 0, "steps": 0, "by": {}})
        w["samples"] += 1
        w["steps"] += r["steps"]
        for k, v in r["by"].items():
            a = w["by"].setdefault(k, {"samples_with": 0, "events": 0, "events_with_tool_calls": 0, "output_tokens": 0, "max_input_tokens_sum": 0})
            a["samples_with"] += 1
            a["events"] += v[0]
            a["events_with_tool_calls"] += v[1]
            a["output_tokens"] += v[2]
            a["max_input_tokens_sum"] += v[3]
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=HERE, check=True).stdout.strip()
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    p = HERE / "results" / f"agent_events_check_{stamp}.json"
    with p.open("x") as fh:
        json.dump({"git": git, "summary": summ}, fh, indent=1)
    print(f"wrote {p.name}")
    for w, d in sorted(summ.items()):
        print(f"[{w}] samples {d['samples']} steps/sample {d['steps'] / d['samples']:.1f}")
        for k, a in sorted(d["by"].items()):
            print(f"  {k}: in {a['samples_with']} samples; events/sample {a['events'] / d['samples']:.1f}, with tool calls "
                  f"{a['events_with_tool_calls'] / d['samples']:.1f}; output tokens/sample {a['output_tokens'] / d['samples']:.0f}; "
                  f"mean per-sample max input tokens {a['max_input_tokens_sum'] / a['samples_with']:.0f}")

@app.function(image=image, cpu=1, memory=2048, timeout=600, retries=0, volumes={"/ev": ev})
def scan_list() -> list:
    rels = sorted(str(p.relative_to(EV_ROOT)) for p in Path(EV_ROOT).glob("*/*/*/*.eval") if world_of_path(str(p.relative_to(EV_ROOT))))
    if not rels or not all(PATH_RE.fullmatch(r) for r in rels):
        raise ValueError("unexpected eval path")
    return rels
