"""1(b) agent behavior after a text_editor output: next-call kind, the agent's visible text, and the gap matched on kind."""
import json, re, sys
from collections import Counter
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).parent
from boot import report, fmt
import sys
sys.path.insert(0, str(Path(__file__).parent))
from paths import DATA, RESULTS, ROOT, stamp as now  # noqa: E402
ctx = pd.read_parquet(DATA / "context_features.parquet")
K = ["arm", "key", "step", "action"]
EDITOR = {"view", "edit", "create", "other editor"}
out = {}
for reader, pos in [(r, p) for r in ("qwen3-32b", "qwen3.5") for p in ("pre_call", "prev_content")]:  # prev_content: the content-token control
    t = pd.read_json(DATA / "tables" / f"{reader}__{pos}__attack_only.jsonl.gz", lines=True).query("train_set == 'real'")
    df = t.merge(ctx, on=K, how="left", validate="one_to_one").reset_index(drop=True)
    after = df.prev_kind.isin(EDITOR).to_numpy()
    real, sim = (df.domain == "real").to_numpy(), (df.domain == "sim").to_numpy()
    if reader == "qwen3-32b" and pos == "pre_call":
        mix = pd.crosstab(df[after].next_kind, df[after].domain, normalize="columns").round(3)
        mix["sim minus real"] = (mix.sim - mix.real).round(3)
        print("next call after a text_editor output (share of calls):\n", mix.to_string())
        imp = df[after & (df.y == 1)]
        print("\nimpactful calls only:\n", pd.crosstab(imp.next_kind, imp.domain, normalize="columns").round(3).to_string())
        a = df[after]
        txt = a.groupby("domain").agent_chars.describe(percentiles=[0.5]).round(1)
        txt["empty %"] = a.groupby("domain").agent_chars.apply(lambda c: round(100 * (c == 0).mean(), 1))
        print("\nagent text before the call (chars):\n", txt.to_string())
        # distinguishing words: log-odds with an informative Dirichlet prior (Monroe et al. 2008)
        tok = lambda s: re.findall(r"[a-z][a-z_']+", s.lower())
        cr, cs = Counter(), Counter()
        for d, s in zip(a.domain, a.agent_text):
            (cr if d == "real" else cs).update(tok(s))
        prior = cr + cs; a0 = sum(prior.values()); nr, ns = sum(cr.values()), sum(cs.values())
        z = {}
        for w, pw in prior.items():
            if pw < 30: continue
            aw = pw / a0 * 1000
            lr = np.log((cr[w] + aw) / (nr + 1000 - cr[w] - aw)); ls = np.log((cs[w] + aw) / (ns + 1000 - cs[w] - aw))
            z[w] = (lr - ls) / np.sqrt(1 / (cr[w] + aw) + 1 / (cs[w] + aw))
        top = sorted(z.items(), key=lambda x: x[1])
        print("\nmore in sim:", ", ".join(f"{w} ({cs[w]}/{cr[w]})" for w, _ in top[:15]))
        print("more in real:", ", ".join(f"{w} ({cr[w]}/{cs[w]})" for w, _ in top[::-1][:15]))
        out["mix"] = mix.to_dict(); out["text"] = txt.to_dict(); out["words_sim"] = top[:30]; out["words_real"] = top[::-1][:30]
    subs = {"after a text_editor output": (real & after, sim & after)}
    for k in ("edit", "create", "view", "bash-like"):
        m = (df.next_kind == k).to_numpy()
        subs[f"after a text_editor output, next call {k}"] = (real & after & m, sim & after & m)
    rep = report(df, subs); out[f"{reader} {pos}"] = rep
    # the gap with sim reweighted to real's next-call mix (impactful calls; recall is over impactful calls)
    g = {k: rep[f"after a text_editor output, next call {k}"]["real minus sim"]["recall"]["value"] for k in ("edit", "create", "bash-like")}
    w = df[real & after & (df.y == 1)].next_kind.value_counts(normalize=True)
    print(f"\n### {reader} {pos}\n" + fmt(rep))
    print("recall gap reweighted to real's impactful next-call mix:",
          round(sum(w.get(k, 0) * g[k] for k in g if g[k] == g[k]) / sum(w.get(k, 0) for k in g if g[k] == g[k]), 4))
stamp = pd.Timestamp.now(tz="America/Los_Angeles").strftime("%Y-%m-%d_%H%M%S")
(RESULTS / f"shift_behavior_{stamp}.json").write_text(json.dumps(out, indent=1, default=str))
