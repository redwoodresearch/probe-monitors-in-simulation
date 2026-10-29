# find.py <run dir>: per cell with a sample error on its last attempt: task, state, attempts, error class (first token only),
# and the .eval paths under sim_honest@september/<task>/. No message text.
import glob, json, os, re, sys
run = sys.argv[1]
d = json.load(open(run + "/launch.json"))
for r in d["runs"]:
    a = r["attempts"][-1] if r["attempts"] else {}
    errs = (a.get("outcome") or {}).get("sample_errors", [])
    if not errs:
        continue
    cls = [re.match(r"^([A-Za-z_]\w*)", str(e)).group(1) if re.match(r"^([A-Za-z_]\w*)", str(e)) else "?" for e in errs]
    print("ERRCELL", r["task"], "state", r["state"], "attempts", len(r["attempts"]), "classes", cls)
    for f in sorted(glob.glob(f"{run}/sim_honest@september/{r['task']}*/*.eval")):
        print("  eval", os.path.relpath(f, run), os.path.getsize(f))
