import sys, json
from collections import Counter


def extract_json(line: str):
    i = line.find('{')
    j = line.rfind('}')
    if i < 0 or j <= i:
        return None
    blob = line[i:j+1]
    try:
        return json.loads(blob)
    except Exception:
        return None


wr_outcome = Counter()
wr_success_kind = Counter()
recheck_outcome = Counter()
salvaged_status = Counter()

recheck_parent_ids_signal_emitted = set()
salvaged_parent_ids = set()

total_lines = 0
parsed = 0

for line in sys.stdin:
    total_lines += 1
    obj = extract_json(line)
    if not obj:
        continue
    parsed += 1
    ev = obj.get("event")

    if ev == "waiting_room_outcome":
        outcome = obj.get("outcome", "unknown")
        sk = obj.get("success_kind", "none")
        wr_outcome[outcome] += 1
        wr_success_kind[sk] += 1

    elif ev == "soft_deferral_recheck_outcome":
        outcome = obj.get("outcome", "unknown")
        recheck_outcome[outcome] += 1
        if outcome == "signal_emitted":
            pid = obj.get("parent_pending_id")
            if pid:
                recheck_parent_ids_signal_emitted.add(pid)

    elif ev == "soft_deferral_salvaged":
        st = obj.get("final_status", "unknown")
        salvaged_status[st] += 1
        pid = obj.get("parent_pending_id")
        if pid:
            salvaged_parent_ids.add(pid)

print("Parsed JSON lines:", parsed, "/", total_lines)
print("\n1) waiting_room_outcome")
print("  by outcome:", dict(wr_outcome))
print("  by success_kind:", dict(wr_success_kind))

print("\n2) soft_deferral_recheck_outcome")
print("  by outcome:", dict(recheck_outcome))
print("  unique parent_pending_id (signal_emitted):", len(recheck_parent_ids_signal_emitted))

print("\n3) soft_deferral_salvaged")
print("  by final_status:", dict(salvaged_status))
print("  unique parent_pending_id:", len(salvaged_parent_ids))

print("\n4) Salvage numerator (unique parent_pending_id):", len(salvaged_parent_ids))
