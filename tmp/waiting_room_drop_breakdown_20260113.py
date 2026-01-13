import sys, json, collections

log = sys.argv[1]
rc = collections.Counter()
dk = collections.Counter()


def extract(line: str):
    i = line.find('{')
    j = line.rfind('}')
    if i >= 0 and j > i:
        return line[i:j+1]
    return None


with open(log, 'r', encoding='utf-8', errors='replace') as f:
    for line in f:
        s = extract(line)
        if not s:
            continue
        try:
            obj = json.loads(s)
        except Exception:
            continue
        if obj.get("event") != "waiting_room_drop":
            continue
        rc[obj.get("reason_code", "unknown")] += 1
        dk[obj.get("drop_kind", "unknown")] += 1

print("Drop reason_code top 20:")
for k, v in rc.most_common(20):
    print(f"  {k}: {v}")
print("\nDrop kind:")
for k, v in dk.most_common(20):
    print(f"  {k}: {v}")
