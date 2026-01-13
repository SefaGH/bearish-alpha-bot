import sys, json, collections

log_path = sys.argv[1]

events = collections.Counter()
wr_add_reason = collections.Counter()
wr_drop_reason = collections.Counter()
wr_drop_kind = collections.Counter()
wr_outcome = collections.Counter()
wr_success_kind = collections.Counter()

recheck_outcome = collections.Counter()
recheck_dropped_reason = collections.Counter()
recheck_signal_emitted_parents = set()

salvaged_final = collections.Counter()
salvaged_parents = set()

parsed = 0
json_candidates = 0
action_signal_lines = 0


def try_extract_json(line: str):
    i = line.find('{')
    j = line.rfind('}')
    if i >= 0 and j > i:
        return line[i:j+1]
    return None


with open(log_path, 'r', encoding='utf-8', errors='replace') as f:
    for line in f:
        if "Action: SIGNAL" in line:
            action_signal_lines += 1

        s = try_extract_json(line)
        if not s:
            continue
        json_candidates += 1
        try:
            obj = json.loads(s)
        except Exception:
            continue
        if not isinstance(obj, dict):
            continue
        ev = obj.get("event")
        if not ev:
            continue
        parsed += 1
        events[ev] += 1

        if ev == "waiting_room_add":
            rc = obj.get("reason_code") or obj.get("reason") or "unknown"
            wr_add_reason[rc] += 1

        elif ev == "waiting_room_drop":
            rc = obj.get("reason_code") or obj.get("reason") or "unknown"
            dk = obj.get("drop_kind") or "unknown"
            wr_drop_reason[rc] += 1
            wr_drop_kind[dk] += 1

        elif ev == "waiting_room_outcome":
            oc = obj.get("outcome") or "unknown"
            sk = obj.get("success_kind") or "unknown"
            wr_outcome[oc] += 1
            wr_success_kind[sk] += 1

        elif ev == "soft_deferral_recheck_outcome":
            oc = obj.get("outcome") or "unknown"
            recheck_outcome[oc] += 1
            dr = obj.get("dropped_reason")
            if dr:
                recheck_dropped_reason[dr] += 1
            if oc == "signal_emitted":
                pid = obj.get("parent_pending_id")
                if pid:
                    recheck_signal_emitted_parents.add(pid)

        elif ev == "soft_deferral_salvaged":
            fs = obj.get("final_status") or "unknown"
            salvaged_final[fs] += 1
            pid = obj.get("parent_pending_id")
            if pid:
                salvaged_parents.add(pid)

print(f"Parsed event JSON objects: {parsed} (from {json_candidates} JSON-like lines)")
print("\nTop events:")
for k, v in events.most_common(15):
    print(f"  {k}={v}")

print("\nwaiting_room_outcome:")
print("  by outcome:", dict(wr_outcome))
print("  by success_kind:", dict(wr_success_kind))

print("\nsoft_deferral_recheck_outcome:")
print("  by outcome:", dict(recheck_outcome))
if recheck_dropped_reason:
    print("  dropped_reason:", dict(recheck_dropped_reason))
print("  unique parent_pending_id (signal_emitted):", len(recheck_signal_emitted_parents))

print("\nsoft_deferral_salvaged:")
print("  by final_status:", dict(salvaged_final))
print("  unique parent_pending_id (salvaged):", len(salvaged_parents))

near_miss_adds = sum(v for k, v in wr_add_reason.items() if str(k).startswith("strategy."))
print("\nNear-miss proxy:")
print("  waiting_room_add(strategy.*):", near_miss_adds)

print("\nApprox normal trade proxy:")
print("  lines with 'Action: SIGNAL':", action_signal_lines)

if near_miss_adds:
    print("\nConversion:")
    print(
        "  salvaged / strategy.* adds =",
        f"{len(salvaged_parents)}/{near_miss_adds} = {len(salvaged_parents)/near_miss_adds:.4%}",
    )
