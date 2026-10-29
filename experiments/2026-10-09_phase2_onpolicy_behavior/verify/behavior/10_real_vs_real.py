"""Real-vs-real baseline for two on-policy behavior claims, from verify/behavior/per_run.csv (same counting script).
honest_a and honest_b are two real runs of each of the same 20 tasks; attack has one real run per task, so no
real-vs-real attack pair exists. "valid_n" counts the runs with per-call shares (a run with a sample error has
none); the medians use those, the counts use all runs."""
import sys
import pandas as pd
df = pd.read_csv(sys.argv[1])
df = df[df.state == "done"]
def grp(name, m):
    d = df[m]
    return {"group": name, "runs": len(d), "valid_n": int(d.sh_edit.notna().sum()),  # sample-error rows have no shares
            "errfield_total": int(d.n_errfield.sum()), "runs_with_errfield": int((d.n_errfield > 0).sum()),
            "median_sh_errfield": round(d.sh_errfield.median(), 3), "median_sh_failtext": round(d.sh_failtext.median(), 3),
            "median_sh_edit": round(d.sh_edit.median(), 3), "runs_pw_hits": int((d.pw_hits > 0).sum()), "runs_mod_hits": int((d.mod_hits > 0).sum())}
rows = [grp("real attack + honest_b (as reported)", (df.world == "real") & df.arm.isin(["attack", "honest_b"])),
        grp("sim, both simulators (as reported)", df.world == "sim"),
        grp("real honest_a", df.arm == "honest_a"), grp("real honest_b", df.arm == "honest_b"),
        grp("sim honest@september", (df.arm == "sim_honest") & (df.simulator == "september")),
        grp("sim honest@today", (df.arm == "sim_honest") & (df.simulator == "today")),
        grp("real attack", df.arm == "attack"), grp("sim attack (both)", df.arm == "sim_attack")]
print(pd.DataFrame(rows).to_string(index=False))
