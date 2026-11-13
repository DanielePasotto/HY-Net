import json, os
os.chdir(os.path.expanduser("~/projects/HY-Net"))
def row(m, r):
    p = f"outputs/{r}/organoid/eval/{m}/fold0/summary.json"
    if not os.path.exists(p): return None
    j = json.load(open(p)); d = j["per_well"]["dice"]
    bs = {b["bin"]: (round(100*b["recall"]), round(100*b["precision"])) for b in j.get("by_size", [])}
    return 100*d["mean"], bs
S = ["", "_s43", "_s44"]
groups = [("ynet_raw", "scale (ref)", [f"lodo_scale{s}" for s in S]),
          ("hynet_res", "res (ref)", [f"lodo_res{s}" for s in S]),
          ("hynet_res", "B cons+target", [f"lodo_cB{s}" for s in S]),
          ("hynet_res", "A cons src", [f"lodo_cA{s}" for s in S]),
          ("ynet_raw", "C cons+target noharm", [f"lodo_cC{s}" for s in S]),
          ("ynet_raw", "E aug+scale", [f"lodo_augsc{s}" for s in S])]
for m, name, roots in groups:
    v = []
    for r in roots:
        x = row(m, r)
        if x is None: print(f"{name:22s} {r:16s} -"); continue
        v.append(x[0]); print(f"{name:22s} {r:16s} Dice {x[0]:.2f}  rec/prec {x[1]}")
    if v: print(f"  -> {name} mean {sum(v)/len(v):.2f} (n={len(v)})")
