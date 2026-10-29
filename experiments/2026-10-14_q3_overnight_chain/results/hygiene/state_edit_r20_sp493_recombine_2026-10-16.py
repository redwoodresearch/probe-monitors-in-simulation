# Restart-20 state edit (10-16 ~01:20): pool sp493's combined finals (attempt_labels_sp493.json @ fd3ab9d458, 00:26:01) were built while sp493-c
# was STOPPED; the restart-19 re-open did not clear final_combined["sp493"], so sp493-c's labels (attested 01:14:44) are not in them. Clear the flag:
# the chain recombines the pool once every sp493 batch is terminal again (a new commit of the same file names, as for any new family).
import json, os, sys, time
p = os.path.join(sys.argv[1], "state.json")
d = json.load(open(p))
c = d["batches"]["sp493-c"]
assert c["status"] != "STOPPED" and "attest" in c["done"], (c["status"], c["done"])
assert d["final_combined"].get("sp493") is True, d["final_combined"].get("sp493")
d["final_combined"].pop("sp493")
open(p + ".tmp", "w").write(json.dumps(d, indent=1)); os.replace(p + ".tmp", p)
print("final_combined[sp493] cleared (recombine with sp493-c)", time.strftime("%H:%M:%S"))
