"""By key: inventory_il_v1's sim2 rows (scored by own_all_rs_uncut_cutfull_il_score_2026-10-16_035921) vs inventory_il_v2's same keys. Names / counts only."""
import json, subprocess, sys, collections
g = lambda c, p: subprocess.run(["git", "show", f"{c}:{p}"], capture_output=True, check=True).stdout  # noqa: E731
v1 = json.loads(g("154382a0e", "experiments/2026-10-15_cut_sensitivity/results/inventory_il_v1_2026-10-16_030502.json"))["runs"]
v2 = {r["key"]: r for r in json.loads(g("52532a296", "experiments/2026-10-15_cut_sensitivity/results/inventory_il_v2_2026-10-16_040129.json"))["runs"]}
s1 = [r for r in v1 if r["world"] == "sim2"]
diff = collections.Counter(); per_field = collections.defaultdict(collections.Counter); missing = []
for r in s1:
    r2 = v2.get(r["key"])
    if r2 is None:
        missing.append(r["key"]); continue
    fields = sorted(k for k in set(r) | set(r2) if r.get(k, "<absent>") != r2.get(k, "<absent>"))
    diff[tuple(fields)] += 1
    for k in fields:
        per_field[k][f"{r.get(k, '<absent>')} -> {r2.get(k, '<absent>')}" if k != "cell" else ("cell added, equals v1 task" if r2.get("cell") == r.get("task") else "cell added, differs from v1 task")] += 1
print(json.dumps({"v1_sim2_rows": len(s1), "missing_in_v2": len(missing), "rows_by_changed_fields": {"|".join(k) or "(none)": v for k, v in diff.items()},
                  "changes": {k: dict(v) for k, v in per_field.items()}}, indent=1))
