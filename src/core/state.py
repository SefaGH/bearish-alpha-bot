import os, json, datetime
from datetime import timezone

STATE_PATH = 'data/state.json'
TODAY_PATH = 'data/day_stats.json'

def _now():
    return datetime.datetime.now(timezone.utc).isoformat()

def load_state():
    if not os.path.exists(STATE_PATH):
        return {'open': {}, 'closed': []}
    with open(STATE_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_state(state):
    os.makedirs('data', exist_ok=True)
    with open(STATE_PATH, 'w', encoding='utf-8') as f:
        json.dump(state, f, ensure_ascii=False, indent=2)

def load_day_stats():
    if not os.path.exists(TODAY_PATH):
        return {'date': _today(), 'signals': 0, 'tp': 0, 'sl': 0, 'pnl': 0.0}
    with open(TODAY_PATH, 'r', encoding='utf-8') as f:
        d = json.load(f)
    if d.get('date') != _today():
        d = {'date': _today(), 'signals': 0, 'tp': 0, 'sl': 0, 'pnl': 0.0}
    return d

def save_day_stats(stats):
    os.makedirs('data', exist_ok=True)
    with open(TODAY_PATH, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

def _today():
    return datetime.date.today().isoformat()
