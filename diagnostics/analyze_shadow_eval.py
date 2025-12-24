python - <<'PY'
import json
from collections import Counter, defaultdict

path="/tmp/shadow.log"
rows=[]
for line in open(path, "r", encoding="utf-8", errors="ignore"):
    # line: "... strategy_shadow_eval {json}"
    i=line.find("strategy_shadow_eval ")
    if i<0: 
        continue
    js=line[i+len("strategy_shadow_eval "):].strip()
    try:
        rows.append(json.loads(js))
    except Exception:
        pass

print("rows:", len(rows))

# 1) Decision distribution
c=Counter((r.get("strategy"), r.get("decision"), r.get("fail_reason")) for r in rows)
print("\nTop decisions:")
for k,v in c.most_common(20):
    print(v, k)

# 2) STR counterfactual: RSI threshold sweep + rip pass
thr_list=[48.0,49.0,49.5,50.0,55.0]
str_rows=[r for r in rows if r.get("strategy")=="adaptive_str"]
print("\nSTR rows:", len(str_rows))

def fnum(x):
    try: return float(x)
    except: return None

for thr in thr_list:
    n_rsi=0
    n_rsi_and_rip=0
    n_rip=0
    for r in str_rows:
        rsi=fnum(r.get("rsi"))
        rip=fnum(r.get("rip_pass_shadow"))
        # rip_pass_shadow is boolean in your logs; float(False)=0.0, float(True)=1.0
        rip_pass = bool(r.get("rip_pass_shadow")) if "rip_pass_shadow" in r else None
        if rip_pass: n_rip += 1
        if rsi is not None and rsi >= thr:
            n_rsi += 1
            if rip_pass:
                n_rsi_and_rip += 1
    print(f"\nCounterfactual STR: rsi_threshold={thr}")
    print("  RSI would pass:", n_rsi)
    print("  RIP passes (regardless of RSI):", n_rip)
    print("  RSI+RIP would pass:", n_rsi_and_rip)

# 3) Rip delta range (how far from rip)
rip_deltas=[fnum(r.get("rip_delta")) for r in str_rows if r.get("rip_delta") is not None]
rip_deltas=[x for x in rip_deltas if x is not None]
if rip_deltas:
    rip_deltas.sort()
    print("\nRIP delta (close - rip_threshold) stats:")
    print("  min:", rip_deltas[0], "max:", rip_deltas[-1])
    print("  p50:", rip_deltas[len(rip_deltas)//2])
PY
