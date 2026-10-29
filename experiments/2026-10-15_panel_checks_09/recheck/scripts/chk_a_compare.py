import json, sys
mine = json.load(open(sys.argv[1])); fz = json.load(open(sys.argv[2]))
maxd = 0.0; mism = []
for rs, bins in fz["bins"].items():
    for b, v in bins.items():
        m = mine["bins"][f"{rs}|{b}"]
        if (m["n"], m["k"]) != (v["n"], v["k_zero"]): mism.append((rs, b, m["n"], m["k"], v["n"], v["k_zero"]))
        if v["n"]:
            maxd = max(maxd, abs(m["share"]-v["share_zero"]), abs(m["cp"][0]-v["cp95"][0]), abs(m["cp"][1]-v["cp95"][1]))
print("bins compared:", sum(len(b) for b in fz["bins"].values()), "n/k mismatches:", mism, "max |share/CP diff|:", f"{maxd:.2e}")
# pattern composition of the (4,6] and =0 bins, with-strategy
for rs in [k for k in fz["bins"] if k.endswith("with_strategy")]:
    for b in ["(4,6]", "=0"]:
        comp = mine["patterns_in_bin"][f"{rs}|{b}"]
        tot = sum(comp.values())
        print(rs, b, "n", tot, "patterns:", ", ".join(f"{p}:{c}" for p, c in comp.items()))
