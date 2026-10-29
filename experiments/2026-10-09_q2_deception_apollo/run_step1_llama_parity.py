"""Step 1 of the deception tests, the positive control for the new test sets: does our port read the same tokens and
activations as Apollo on them? Llama-3.3-70B with Apollo's shipped layer-22 probe (example_results/instructed_pairs/
detector.pt), compared per dialogue with Apollo's shipped scores (scores.json, control_scores.json). Weights and the
tokenizer (Meta-Llama-3.1-70B-Instruct, as Apollo's code) load from our Volume with local_files_only; no credentials.

Apollo's shipped token lists include its mask padding (deception-detection tokenized_data.py: 0 tokens before and the
end-of-turn token after on the trimmed sets, 4 header tokens before and the end-of-turn token after on Alpaca, none on
Roleplaying), so the port's spans get the same `pad` here.
- `counts` (Modal CPU, cents): detected-token counts with the port's spans (src/probe_shift/apollo.py render_detect,
  Llama 3.1 template dated 26 Jul 2024, as the 10-06 Llama check) for every dialogue of the three labelled sets and
  2,000 Alpaca controls (seed 0), against the shipped per-token score counts. Run first: a mismatch pattern here means
  the span rule needs fixing before any GPU time.
- `parity` (2 H100s): on a stratified subset (seed 0: 25 honest + 25 deceptive per labelled set, the double-down set's
  24 honest + 26 deceptive, 100 Alpaca, 20 Roleplaying as the anchor that matched on 10-06), each dialogue's mean
  activation at Apollo's layer 22, scored with the shipped probe. As Apollo, the model is cut at 22 blocks, so its last
  hidden state is the final norm of block 21's output and the other 58 blocks are not run.
  Pass (written before running): detected-token counts equal on at least 99% of dialogues, Pearson r of mean scores at
  least 0.999, largest absolute difference at most 0.05 (10-06 Roleplaying: 100%, 0.9999997, 0.008).

  uv run modal run experiments/2026-10-09_q2_deception_apollo/run_step1_llama_parity.py::counts
  uv run modal run experiments/2026-10-09_q2_deception_apollo/run_step1_llama_parity.py::parity
"""
import io
import json
import pickle
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

HERE = Path(__file__).parent
APOLLO_REPO = Path.home() / "deception-detection"
APP = "probe-shift_aswin-surya_sub-q2-deception_llama-parity_kill-2026-11-08"
MODEL, TOKENIZER, CACHE = "meta-llama/Llama-3.3-70B-Instruct", "meta-llama/Meta-Llama-3.1-70B-Instruct", "/llama-cache"
TEMPLATE = {"date_string": "26 Jul 2024"}
app = modal.App(APP)
llama = modal.Volume.from_name("probe-shift_aswin-surya_sub-apollo_llama-cache")
image = (modal.Image.debian_slim(python_version="3.12")
         .uv_pip_install("torch==2.8.0", "transformers==4.56.2", "accelerate==1.10.1", "numpy", "scikit-learn==1.7.2")
         .env({"HF_HUB_OFFLINE": "1"}).add_local_python_source("probe_shift"))


def build() -> dict:
    from probe_shift import apollo

    return {"insider_trading__upscale": apollo.insider_trading(APOLLO_REPO),
            "insider_trading_doubledown__upscale": apollo.insider_trading(APOLLO_REPO, double_down=True),
            "sandbagging_v2__wmdp_mmlu": apollo.from_rollouts(APOLLO_REPO, "sandbagging_v2__wmdp_mmlu__llama-70b-3.3.json", "sandbagging", "<step4>"),
            "alpaca__plain": apollo.from_rollouts(APOLLO_REPO, "alpaca__plain__llama-70b-3.3.json", "alpaca", None)}


def pad(name: str, d: dict) -> tuple[int, int]:
    return (0, 0) if name == "roleplaying__plain" else (0, 1) if d.get("trimmed") else (4, 1)


def shipped() -> dict:
    d = APOLLO_REPO / "example_results" / "instructed_pairs"
    s = json.loads((d / "scores.json").read_text())
    return s | {"alpaca__plain": json.loads((d / "control_scores.json").read_text())["alpaca__plain"]}


@app.function(image=image, volumes={CACHE: llama.read_only()}, cpu=8, memory=16384, timeout=1800, retries=0)
def count_remote(dialogues: dict) -> dict:
    from transformers import AutoTokenizer

    from probe_shift.apollo import render_detect

    tok = AutoTokenizer.from_pretrained(TOKENIZER, cache_dir=CACHE, local_files_only=True)
    return {name: [len(render_detect(tok, d, TEMPLATE, pad(name, d))[1]) for d in ds] for name, ds in dialogues.items()}


@app.function(image=image, gpu="H100:2", volumes={CACHE: llama.read_only()}, cpu=16, memory=131072, timeout=3600, retries=0)
def parity_remote(dialogues: list[tuple[str, dict]], detector: bytes) -> dict:
    import time

    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from probe_shift.apollo import render_detect

    t0 = time.time()
    det = pickle.load(io.BytesIO(detector))
    (layer,) = det["layers"]
    w, mu, sd = (np.asarray(det[k].float().numpy(), np.float64).reshape(-1) for k in ("directions", "scaler_mean", "scaler_scale"))
    tok = AutoTokenizer.from_pretrained(TOKENIZER, cache_dir=CACHE, local_files_only=True)
    base = AutoModelForCausalLM.from_pretrained(MODEL, cache_dir=CACHE, local_files_only=True, dtype=torch.float16, device_map="auto",
                                                attn_implementation="sdpa").model.eval()
    base.layers = base.layers[:layer]  # Apollo's cut model: last_hidden_state = the final norm of block (layer - 1)'s output
    load_s = time.time() - t0
    print(f"loaded in {load_s:.0f}s", flush=True)
    device = base.embed_tokens.weight.device
    scores, counts = [], []
    for k, (name, d) in enumerate(dialogues):
        ids, det = render_detect(tok, d, TEMPLATE, pad(name, d))
        with torch.inference_mode():
            h = base(input_ids=torch.tensor([ids], device=device), use_cache=False).last_hidden_state[0, det]
        a = h.float().cpu().numpy().astype(np.float64)
        scores.append(float((((a - mu) / sd) @ w).mean()) if len(det) else float("nan"))  # Apollo: mean of per-token scores
        counts.append(len(det))
        if k % 25 == 0:
            print(f"{k + 1}/{len(dialogues)} at {time.time() - t0:.0f}s", flush=True)
    return {"scores": scores, "n_tokens": counts, "layer": int(layer),
            "load_seconds": round(load_s, 1), "seconds": round(time.time() - t0, 1), "device": torch.cuda.get_device_name()}


def _save(name: str, obj: dict) -> Path:
    stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S")
    path = HERE / "results" / f"{name}_{stamp}.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(obj, indent=1) + "\n")
    return path


@app.local_entrypoint()
def counts():
    import numpy as np

    sets, ref = build(), shipped()
    pick = {n: (sorted(np.random.default_rng(0).choice(len(ds), 2000, replace=False).tolist()) if n == "alpaca__plain" else list(range(len(ds))))
            for n, ds in sets.items()}
    res = count_remote.remote({n: [sets[n][i] for i in pick[n]] for n in sets})
    out = {}
    for n, c in res.items():
        theirs = np.array([len(ref[n]["scores"][i]) for i in pick[n]])
        diff = np.array(c) - theirs
        vals, freq = np.unique(diff, return_counts=True)
        out[n] = {"dialogues": len(c), "token counts equal": f"{int((diff == 0).sum())} of {len(c)}",
                  "ours minus shipped (value: dialogues)": {int(v): int(f) for v, f in zip(vals, freq)}}
    print(json.dumps(out, indent=1))
    print(_save("step1_token_counts", {"per set": out, "template": TEMPLATE, "tokenizer": TOKENIZER}))


@app.local_entrypoint()
def parity():
    import numpy as np

    from probe_shift import apollo

    sets, ref = build(), shipped()
    sets["roleplaying__plain"] = apollo.roleplaying(APOLLO_REPO)  # the port's order is the file's; Apollo's is shuffled
    np.random.seed(42)  # Apollo's Roleplaying order (the 10-06 check's mapping): shipped j is the file's perm[j]
    rp_perm = np.random.permutation(len(sets["roleplaying__plain"]))
    rng = np.random.default_rng(0)
    chosen = []  # (set, index in Apollo's order)
    for n in ("insider_trading__upscale", "insider_trading_doubledown__upscale", "sandbagging_v2__wmdp_mmlu"):
        labs = [d["apollo_label"] for d in sets[n]]
        honest = [i for i, x in enumerate(labs) if x == "HONEST"]
        deceptive = [i for i, x in enumerate(labs) if x == "DECEPTIVE"]
        n_h = min(25, len(honest))  # the double-down set has 24 honest dialogues
        chosen += [(n, int(i)) for i in rng.choice(honest, n_h, replace=False)] + [(n, int(i)) for i in rng.choice(deceptive, 50 - n_h, replace=False)]
    chosen += [("alpaca__plain", int(i)) for i in rng.choice(len(sets["alpaca__plain"]), 100, replace=False)]
    chosen += [("roleplaying__plain", int(j)) for j in rng.choice(len(rp_perm), 20, replace=False)]
    dialogue = lambda n, j: sets[n][int(rp_perm[j])] if n == "roleplaying__plain" else sets[n][j]
    res = parity_remote.remote([(n, dialogue(n, j)) for n, j in chosen], (APOLLO_REPO / "example_results" / "instructed_pairs" / "detector.pt").read_bytes())
    theirs = np.array([float(np.mean(ref[n]["scores"][j])) for n, j in chosen])
    their_n = np.array([len(ref[n]["scores"][j]) for n, j in chosen])
    ours, our_n = np.array(res["scores"]), np.array(res["n_tokens"])
    per = {}
    for n in dict.fromkeys(x for x, _ in chosen):
        m = np.array([x == n for x, _ in chosen])
        per[n] = {"dialogues": int(m.sum()), "token counts equal": f"{int((our_n[m] == their_n[m]).sum())} of {int(m.sum())}",
                  "pearson r": round(float(np.corrcoef(ours[m], theirs[m])[0, 1]), 7), "largest absolute difference": round(float(np.abs(ours[m] - theirs[m]).max()), 4)}
    allr, share, worst = float(np.corrcoef(ours, theirs)[0, 1]), float((our_n == their_n).mean()), float(np.abs(ours - theirs).max())
    summary = {"token counts equal (share)": round(share, 4), "pearson r": round(allr, 7), "largest absolute difference": round(worst, 4),
               "passed": bool(share >= 0.99 and allr >= 0.999 and worst <= 0.05)}  # on the unrounded values
    out = {"summary": summary, "per set": per, "load_seconds": res["load_seconds"], "seconds": res["seconds"], "device": res["device"],
           "chosen (set, index in Apollo's order)": chosen, "our mean scores": res["scores"], "our token counts": res["n_tokens"]}
    print(json.dumps({k: out[k] for k in ("summary", "per set", "load_seconds", "seconds")}, indent=1))
    print(_save("step1_llama_parity", out))
