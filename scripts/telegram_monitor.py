#!/usr/bin/env python3
"""Telegram-based monitoring for GitHub Actions."""

import json
import os
from datetime import datetime
from pathlib import Path

def get_trading_stats():
    """Get current trading statistics."""
    stats = {
        'timestamp': datetime.now().isoformat(),
        'status': 'running',
        'total_trades': 0,
        'open_positions': 0,
        'total_pnl': 0.0,
        'win_rate': 0.0
    }
    
    # Read from state files
    state_file = Path('data/state.json')
    if state_file.exists():
        with open(state_file) as f:
            state = json.load(f)
            closed = state.get('closed', [])
            stats['total_trades'] = len(closed)
            stats['open_positions'] = len(state.get('open', {}))
            
            if closed:
                wins = sum(1 for t in closed if t.get('pnl', 0) > 0)
                stats['win_rate'] = wins / len(closed)
                stats['total_pnl'] = sum(t.get('pnl', 0) for t in closed)
    
    day_stats_file = Path('data/day_stats.json')
    if day_stats_file.exists():
        with open(day_stats_file) as f:
            day = json.load(f)
            stats['daily_pnl'] = day.get('pnl', 0)
            stats['daily_signals'] = day.get('signals', 0)
    
    return stats

def format_telegram_report(stats):
    """Format stats as Telegram message."""
    msg = f"""📊 <b>LIVE TRADING REPORT</b>
    
🔸 Status: {stats['status'].upper()}
🔸 Total Trades: {stats['total_trades']}
🔸 Open Positions: {stats['open_positions']}
🔸 Total P&L: ${stats['total_pnl']:.2f}
🔸 Win Rate: {stats['win_rate']*100:.1f}%

📅 Daily Stats:
• Signals: {stats.get('daily_signals', 0)}
• P&L: ${stats.get('daily_pnl', 0):.2f}

⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}"""
    
    return msg

if __name__ == '__main__':
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
    from core.notify import Telegram
    
    token = os.getenv('TELEGRAM_BOT_TOKEN')
    chat_id = os.getenv('TELEGRAM_CHAT_ID')
    
    if token and chat_id:
        tg = Telegram(token, chat_id)
        stats = get_trading_stats()
        msg = format_telegram_report(stats)
        tg.send(msg)
        print("✅ Monitoring report sent to Telegram")
    else:
        print("❌ Telegram credentials not set")
