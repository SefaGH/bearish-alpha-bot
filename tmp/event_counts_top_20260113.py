import re, json, collections, sys

log_path = sys.argv[1] if len(sys.argv) > 1 else None
if not log_path:
    print("usage: python script.py C:/path/to/log")
    raise SystemExit(2)

pat = re.compile(r"(\{.*\})\s*$")

c_event = collections.Counter()
recheck_outcome = collections.Counter()
salvaged_status = collections.Counter()

uniq_recheck_emitted = set()
uniq_salvaged = set()

with open(log_path, "r", errors="ignore", encoding="utf-8") as f:
    for line in f:
        m = pat.search(line)
        if not m:
            continue
        try:
            ev = json.loads(m.group(1))
        except Exception:
            continue

        et = ev.get("event")
        if not et:
            continue
        c_event[et] += 1

        if et == "soft_deferral_recheck_outcome":
            out = ev.get("outcome", "unknown")
            recheck_outcome[out] += 1
            if out == "signal_emitted":
                pid = ev.get("parent_pending_id")
                if pid:
                    uniq_recheck_emitted.add(pid)

        if et == "soft_deferral_salvaged":
            st = ev.get("final_status", "unknown")
            salvaged_status[st] += 1
            pid = ev.get("parent_pending_id")
            if pid:
                uniq_salvaged.add(pid)

print("Event counts (top 15):", c_event.most_common(15))
print("soft_deferral_recheck_outcome by outcome:", dict(recheck_outcome))
print("unique parent_pending_id (signal_emitted):", len(uniq_recheck_emitted))
print("soft_deferral_salvaged by final_status:", dict(salvaged_status))
print("unique parent_pending_id (salvaged):", len(uniq_salvaged))
