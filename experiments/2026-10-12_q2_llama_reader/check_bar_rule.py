"""Review check for #1094 (Codex rrzy_): analyze_edit_effect_llama.py (and the Olmo reading it copies) returns "explains it"
when the fixed point is within ±5 and the as-simulated gap lies outside the fixed range, without fix939's further condition
that the gap shrank (|fixed - e2| < |as simulated - e2|). This re-applies fix939's rule exactly
(../2026-10-08_q3_swap_cue/analyze_fix939.py at f0ff845f, lines 255-261) to the committed readings' own bar numbers and
asserts the verdicts are unchanged; exits nonzero if any differs.
  uv run --no-project python experiments/2026-10-12_q2_llama_reader/check_bar_rule.py
"""
import json, subprocess, sys
BAR = 5.0
FILES = {"Llama part 2": ("dddc8324", "experiments/2026-10-12_q2_llama_reader/results/edit_effect_llama_2026-10-12_153720.json"),
         "Olmo part 2": ("03af438b", "experiments/2026-10-12_q2_olmo_reader/results/edit_effect_olmo_2026-10-12_115023.json")}
bad = 0
for name, (c, p) in FILES.items():
    r = json.loads(subprocess.run(["git", "show", f"{c}:{p}"], capture_output=True, text=True, check=True).stdout)
    for a, v in r["rates"].items():
        b = v["fix939's bar (on fixed - e2)"]; fx, (lo, hi), asis = b["fixed - e2"], b["95%"], b["as simulated - e2"]
        outside = not (lo <= asis <= hi); shrinks = outside and abs(fx) < abs(asis); within = abs(fx) <= BAR
        rule = "explains it" if (within and shrinks) else ("partial" if shrinks else ("doesn't explain it (the gap grew)" if outside else "doesn't explain it"))
        committed = v["reading"].split(" (")[0] if v["reading"].startswith("partial") else v["reading"]
        same = rule.split(" (")[0] == committed.split(" (")[0]
        bad += not same
        print(f"{name} at {a}: fixed - e2 {fx:+.2f} [{lo:+.2f}, {hi:+.2f}], as simulated - e2 {asis:+.2f} | shrinks {shrinks}, within {within} | "
              f"committed '{v['reading']}', fix939's rule '{rule}' | {'same' if same else 'DIFFERENT'}")
sys.exit(1 if bad else 0)
