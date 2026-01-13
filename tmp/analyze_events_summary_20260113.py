import json
from collections import Counter

log_path = r"C:\Users\sefaa\bearish-alpha-bot\logs\live_trading_20260113_122647_441809.log"

def extract_json(line: str):
    i = line.find('{')
    j = line.rfind('}')
    if i < 0 or j <= i:
        return None
    try:
        return json.loads(line[i:j+1])
    except Exception:
        return None

events = []
with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
    for line in f:
        obj = extract_json(line)
        if isinstance(obj, dict) and ("event" in obj or "event_type" in obj):
            events.append(obj)

def name(e):
    return e.get("event") or e.get("event_type")

sro = [e for e in events if name(e) == "soft_deferral_recheck_outcome"]
print("soft_deferral_recheck_outcome:", Counter(e.get("outcome") for e in sro))

sds = [e for e in events if name(e) == "soft_deferral_salvaged"]
print("soft_deferral_salvaged:", Counter(e.get("final_status") for e in sds))

wra = [e for e in events if name(e) == "waiting_room_add" and str(e.get("reason_code", "")).startswith("strategy.")]
wrd = [e for e in events if name(e) == "waiting_room_drop" and str(e.get("reason_code", "")).startswith("strategy.")]
print("waiting_room_add(strategy.*):", len(wra))
print("waiting_room_drop(strategy.*):", len(wrd))
